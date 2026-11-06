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
         "first-person": "first-person", "first": "first-person", "fp": "first-person", "fpv": "first-person",
         "second-person": "second-person", "second": "second-person", "selfie": "second-person"}
UNCHANGED = ("Compared to the previous segment, the visible scene, background, and other-object layout "
             "remain unchanged.")
MIN_SEG, MAX_SEG = 1.0, 6.0
MAX_FRAMES = 480  # 20 s at 24 fps: 110 s on a 4090. 30 s completed but at 183 s (too close to the limit)


def _perspective_sentence(view, name, machine=False):
    if view == "third-person follow":
        return (f"The camera is placed behind the {name}, and follows the {name} throughout the whole video. "
                f"The camera is tracking the {name} and the {name} should be placed in the center of the video.")
    if view == "third-person observer":
        return (f"The camera films the {name} from an external position, and the {name} is visible in the video. "
                f"The framing of the video from the camera is independent of the {name}, and does not follow "
                f"the movement of the {name}.")
    if view == "second-person":
        return (f"The {name} is facing the camera and looking directly at the camera throughout the whole video. The "
                f"{name} should be the dominant object in the video. If the camera lose the {name} or the {name} "
                f"becomes too small in the video frame, the camera should quickly refocus on the {name}, and place "
                f"the face or the front view of the {name} in the center of the video frame.")
    text = (f"The {name} is embodied by the camera, and has the same first-person viewpoint as the camera "
            f"throughout the whole video. Most of the {name} is not visible, and optionally only a small part of "
            f"the {name} can be seen in the video.")
    if machine:
        text += (f" The camera is the eye of the {name}. Most of the {name} should be placed behind the camera. "
                 f"The video can not show the full body of {name}.")
    return text


def _ts(sec):
    ms = int(round(sec * 1000))
    return f"{ms // 60000:02d}:{(ms // 1000) % 60:02d}.{ms % 1000:03d}"


def _snap(sec):
    return max(MIN_SEG, min(MAX_SEG, round(sec * 2) / 2))


def _sentence(t):
    t = t.strip().rstrip(".")
    return t[0].lower() + t[1:] if t else t


def compile_full(text, length=0):
    """-> (world_json, cameras_json). length: target total frames at 24 fps (0 = whatever the script says).
    cameras_json is a JSON list with one camera key timeline (or null) per segment."""
    text = (text or "").strip()
    if text.startswith("{"):
        json.loads(text)  # validate only
        return text, "[]"
    original = text
    # Graydient may deliver the prompt with its newlines collapsed (or dashes swapped), so re-insert line
    # breaks before the header keys, the divider and each 'Ns |' segment start.
    text = text.replace("\r", "").replace("—", "---").replace("–", "-")
    text = re.sub(r"[ \t]*\b(?=(?:style|view|subject|operator|scene)[ \t]*:)", "\n", text, flags=re.I)
    text = re.sub(r"[ \t]*(?=-{3,})", "\n", text)
    text = re.sub(r"(-{3,})[ \t]*", r"\1\n", text)
    text = re.sub(r"[ \t]+(?=\d+(?:\.\d+)?[ \t]*s[ \t]*\|)", "\n", text, flags=re.I)
    head, segs = {}, []
    in_body = False
    for raw in text.split("\n"):
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
        camera = move = None
        for p in list(parts):
            cm = re.match(r"^cam(?:era)?\s*:\s*(.*)$", p, re.I)
            mm = re.match(r"^(?:move|shot)\s*:\s*(.*)$", p, re.I)
            if cm:
                camera = cm.group(1).strip() or None
                parts.remove(p)
            elif mm:
                move = mm.group(1).strip() or None
                parts.remove(p)
        if not parts or len(parts) > 2:
            raise ValueError(f"segment line needs 'Ns | actions' or 'Ns | change | actions' (+ optional "
                             f"'| camera: W 2, D+Left 1' keys and/or '| move: slowly pushes in'): {line!r}")
        change, acts = (parts[0], parts[1]) if len(parts) == 2 else ("", parts[0])
        segs.append((_snap(float(m.group(1))), change, [a.strip() for a in acts.split(";") if a.strip()],
                     (camera, move)))
    if not segs:
        raise ValueError("no segment lines (e.g. '5s | walks slowly to the left'); received: " + repr(original[:300]))
    for need in ("subject", "scene"):
        if not head.get(need):
            raise ValueError(f"missing header line '{need}: ...'")

    view = VIEWS.get(head.get("view", "observer").lower())
    # Tencent's rewriter defaults: third-person follow -> video game, everything else -> real-world footage
    style = (STYLES.get(head["style"].lower()) if head.get("style")
             else ("video game" if view == "third-person follow" else "real-world footage"))
    if not style or not view:
        raise ValueError(f"unknown style/view: {head.get('style')!r} / {head.get('view')!r}")
    name, _, look = head["subject"].partition("=")
    name, look = name.strip(), look.strip()
    machine = head.get("operator", "").lower() in ("machine", "drone", "vehicle", "robot", "nonhuman", "non-human")
    if view == "first-person":
        name = name or "first-person operator"
        if machine and not look:
            raise ValueError("a machine operator needs 'subject: name = what the visible part looks like'")
        attr = ((look or "An adult holding the camera at eye height").rstrip(".") + ". "
                + _perspective_sentence(view, name, machine))
    else:
        if not look:
            raise ValueError("subject needs 'name = description'")
        attr = look.rstrip(".") + ". " + _perspective_sentence(view, name)

    # Hard cap: only 10 s is proven to fit Graydient's ~180 s run budget (15 s / 20 s timed out in the decoder).
    if (not length or length <= 0) and sum(sg[0] for sg in segs) * FPS > MAX_FRAMES:
        length = MAX_FRAMES
    elif length > MAX_FRAMES:
        length = MAX_FRAMES

    # fit to the requested length (24 fps): trim, or extend with 'unchanged' continuation segments
    if length and length > 0:
        total = max(MIN_SEG, round(length / FPS * 2) / 2)
        out, used = [], 0.0
        for d, ch, acts, cam in segs:
            if used >= total:
                break
            left = total - used
            out.append((min(d, left) if left >= MIN_SEG else left, ch, acts, cam))
            used += out[-1][0]
        while used < total - 1e-6:
            d = min(5.0, total - used)
            if d < MIN_SEG:
                out[-1] = (out[-1][0] + d, out[-1][1], out[-1][2], out[-1][3])
                used += d
                break
            out.append((d, "", out[-1][2][-1:] or ["continues"], (None, None)))
            used += d
        segs = out

    segments, cameras, t = [], [], 0.0
    for i, (d, change, acts, cam) in enumerate(segs):
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
        keys, move = cam if cam else (None, None)
        # A text camera sentence is only written when no key path steers the segment (Tencent: camera-controlled
        # segments describe the scene from the new viewpoint, never a camera sentence).
        camera_sentence = ""
        if move and not keys:
            text_move = move.strip().rstrip(".")
            if not re.match(r"^(the |camera)", text_move, re.I):
                text_move = "The camera " + text_move
            camera_sentence = f" <Camera1>{text_move[0].upper() + text_move[1:]}</Camera1>."
        if i == 0:
            seg["visual_description_long"] = head["scene"].rstrip() + camera_sentence
            seg["description_change"] = None
        else:
            seg["visual_description_long"] = None
            c = change.strip().rstrip(".")
            seg["description_change"] = ((("Compared to the previous segment, " + _sentence(c) +
                                           ", while the rest of the scene remains unchanged.") if c else UNCHANGED)
                                         + camera_sentence)
        segments.append(seg)
        cameras.append(keys)
        t = end
    world = {"perspective_view": view, "Style": style,
             "subject_list": [{"subject_id": "<|Subject1|>", "display_name": name, "subject_attribute": attr}],
             "segments": segments}
    return json.dumps(world, indent=2), json.dumps(cameras)


def compile_script(text, length=0):
    return compile_full(text, length)[0]


DEFAULT_SCRIPT = """style: CG
view: observer
subject: small orange fox = A small fox with orange fur, a white chest and a fluffy white-tipped tail, standing on a stone path in side view
scene: A flagstone path crosses a luminous forest of tall dark trunks under a misty canopy, with drifting fog and tiny fireflies glowing between the trees.
---
5s | walks slowly to the left along the stone path
5s | clusters of glowing blue flowers open in the grass beside the path | stops beside the flowers and turns around to face the camera ; sniffs the glowing flowers and looks around
"""
