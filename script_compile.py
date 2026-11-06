"""Plain-text event script -> Tencent's structured Hy-Nomas world JSON.

Why this exists: Graydient's chat frontend mangles '<...>' and treats '[...]' as the
negative prompt, but Hy-Nomas wants '<Action1><|Subject1|> ...</Action1>.' tags. Users
write a plain script; this adds the tags, perspective sentence and timestamps.

Script format (blank lines ignored):
    style: CG                      (CG | real | anime | game)
    view: observer                 (observer | follow | first-person)
    subject: small orange fox = A small fox with orange fur ...   (name = look)
    scene: A flagstone path crosses a luminous forest ...         (first segment's scene)
    ---
    5s | walks slowly to the left along the stone path
    5s | glowing blue flowers open beside the path | stops and turns to face the camera ; sniffs the flowers

Segment line: DURATIONs | [scene change |] action ; action ...
Durations snap to 0.5 s and clamp to 1..6 s (2-5 s works best). Actions in one segment
split it evenly. A line starting with '#' is a comment. If the text already starts with
'{' it is treated as finished world JSON and passed through untouched.
"""
import json
import re

FPS = 24
STYLES = {"cg": "CG", "real": "real-world footage", "real-world footage": "real-world footage",
          "footage": "real-world footage", "anime": "animation/anime", "animation": "animation/anime",
          "animation/anime": "animation/anime", "game": "video game", "video game": "video game"}
VIEWS = {"observer": "third-person observer", "third-person observer": "third-person observer",
         "follow": "third-person follow", "third-person follow": "third-person follow",
         "first-person": "first-person", "first": "first-person", "fp": "first-person"}
UNCHANGED = ("Compared to the previous segment, the visible scene, background, and other-object layout "
             "remain unchanged.")
MIN_SEG, MAX_SEG = 1.0, 6.0


def _perspective_sentence(view, name):
    if view == "third-person follow":
        return (f"The camera is placed behind the {name}, and follows the {name} throughout the whole video. "
                f"The camera is tracking the {name} and the {name} should be placed in the center of the video.")
    if view == "third-person observer":
        return (f"The camera films the {name} from an external position, and the {name} is visible in the video. "
                f"The framing of the video from the camera is independent of the {name}, and does not follow "
                f"the movement of the {name}.")
    return ("The first-person operator is embodied by the camera, and has the same first-person viewpoint as the "
            "camera throughout the whole video. Most of the first-person operator is not visible, and optionally "
            "only a small part of the first-person operator can be seen in the video.")


def _ts(sec):
    ms = int(round(sec * 1000))
    return f"{ms // 60000:02d}:{(ms // 1000) % 60:02d}.{ms % 1000:03d}"


def _snap(sec):
    return max(MIN_SEG, min(MAX_SEG, round(sec * 2) / 2))


def _sentence(t):
    t = t.strip().rstrip(".")
    return t[0].lower() + t[1:] if t else t


def compile_script(text, length=0):
    """length: target total frames at 24 fps (0 = whatever the script says)."""
    text = (text or "").strip()
    if text.startswith("{"):
        json.loads(text)  # validate only
        return text
    head, segs = {}, []
    in_body = False
    for raw in text.replace("\r", "").split("\n"):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("---"):
            in_body = True
            continue
        m = re.match(r"^(\d+(?:\.\d+)?)\s*s\s*\|(.*)$", line, re.I)
        if m:
            in_body = True
        if not in_body and not m:
            k, _, v = line.partition(":")
            head[k.strip().lower()] = v.strip()
            continue
        if not m:
            raise ValueError(f"segment line must look like '5s | actions': {line!r}")
        parts = [p.strip() for p in m.group(2).split("|")]
        if len(parts) > 2:
            raise ValueError(f"too many '|' in segment line: {line!r}")
        change, acts = (parts[0], parts[1]) if len(parts) == 2 else ("", parts[0])
        segs.append((_snap(float(m.group(1))), change, [a.strip() for a in acts.split(";") if a.strip()]))
    if not segs:
        raise ValueError("no segment lines (e.g. '5s | walks slowly to the left')")
    for need in ("subject", "scene"):
        if not head.get(need):
            raise ValueError(f"missing header line '{need}: ...'")

    style = STYLES.get(head.get("style", "cg").lower())
    view = VIEWS.get(head.get("view", "observer").lower())
    if not style or not view:
        raise ValueError(f"unknown style/view: {head.get('style')!r} / {head.get('view')!r}")
    name, _, look = head["subject"].partition("=")
    name, look = name.strip(), look.strip()
    if view == "first-person":
        name = name or "first-person operator"
        attr = (look or "An adult holding the camera at eye height.").rstrip(".") + ". " + _perspective_sentence(view, name)
    else:
        if not look:
            raise ValueError("subject needs 'name = description'")
        attr = look.rstrip(".") + ". " + _perspective_sentence(view, name)

    # fit to the requested length (24 fps): trim, or extend with 'unchanged' continuation segments
    if length and length > 0:
        total = max(MIN_SEG, round(length / FPS * 2) / 2)
        out, used = [], 0.0
        for d, ch, acts in segs:
            if used >= total:
                break
            left = total - used
            out.append((min(d, left) if left >= MIN_SEG else left, ch, acts))
            used += out[-1][0]
        while used < total - 1e-6:
            d = min(5.0, total - used)
            if d < MIN_SEG:
                out[-1] = (out[-1][0] + d, out[-1][1], out[-1][2])
                used += d
                break
            out.append((d, "", out[-1][2][-1:] or ["continues"]))
            used += d
        segs = out

    segments, t = [], 0.0
    for i, (d, change, acts) in enumerate(segs):
        end = t + d
        acts = acts or ["continues"]
        n = len(acts)
        actions = []
        for j, a in enumerate(acts):
            a0 = t + round(j * d / n * 2) / 2
            a1 = end if j == n - 1 else t + round((j + 1) * d / n * 2) / 2
            actions.append({"start_time": _ts(a0), "end_time": _ts(a1),
                            "action_description": f"<Action{j + 1}><|Subject1|> {_sentence(a)}</Action{j + 1}>."})
        seg = {"start_time": _ts(t), "end_time": _ts(end), "contained_actions": actions}
        if i == 0:
            seg["visual_description_long"] = head["scene"]
            seg["description_change"] = None
        else:
            seg["visual_description_long"] = None
            c = change.strip().rstrip(".")
            seg["description_change"] = (("Compared to the previous segment, " + _sentence(c) +
                                          ", while the rest of the scene remains unchanged.") if c else UNCHANGED)
        segments.append(seg)
        t = end
    world = {"perspective_view": view, "Style": style,
             "subject_list": [{"subject_id": "<|Subject1|>", "display_name": name, "subject_attribute": attr}],
             "segments": segments}
    return json.dumps(world, indent=2)


DEFAULT_SCRIPT = """style: CG
view: observer
subject: small orange fox = A small fox with orange fur, a white chest and a fluffy white-tipped tail, standing on a stone path in side view
scene: A flagstone path crosses a luminous forest of tall dark trunks under a misty canopy, with drifting fog and tiny fireflies glowing between the trees.
---
5s | walks slowly to the left along the stone path
5s | clusters of glowing blue flowers open in the grass beside the path | stops beside the flowers and turns around to face the camera ; sniffs the glowing flowers and looks around
"""
