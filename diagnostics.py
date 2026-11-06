"""Stage timing + decoder-file prefetch for ComfyUI-Hy-Nomas on Graydient.

Observed 2026-11-06: the same 10 s script took 85 s on a machine that already had the weights and 198 s on one whose
weights were "mirror-fetched"; sampling speed was identical, the extra time was in the decoder stage. The node's
HyNomasLoadDecoder has no upstream dependency, so ComfyUI runs it right before decode, and pixelumm.load_state_dict
then reads the whole 9 GB FP8 file into RAM inside the run clock. On a cold disk that read is the suspect.

1. timing: wrap the node's heavy stages and print wall-clock durations ("[HyNomasGlue] t=..s stage took ..s").
2. prefetch: read the decoder file sequentially in a background thread at job start, so the page cache is warm
   (and the disk read overlaps with GPU sampling) by the time load_state_dict runs.
"""
import sys
import threading
import time

_T0 = None
_DONE = False
_CHUNK = 64 * 1024 * 1024


def _now():
    return time.perf_counter() - _T0


def _wrap(mod, name, label):
    fn = getattr(mod, name, None)
    if fn is None or getattr(fn, "_hn_wrapped", False):
        return False

    def wrapper(*a, **k):
        t = time.perf_counter()
        print(f"[HyNomasGlue] t={_now():6.1f}s {label} start", flush=True)
        try:
            return fn(*a, **k)
        finally:
            print(f"[HyNomasGlue] t={_now():6.1f}s {label} took {time.perf_counter() - t:.1f}s", flush=True)

    wrapper._hn_wrapped = True
    setattr(mod, name, wrapper)
    return True


def _wrap_slow(mod, name, label, threshold=1.0):
    """Like _wrap but only prints calls slower than `threshold` seconds (for hot functions)."""
    fn = getattr(mod, name, None)
    if fn is None or getattr(fn, "_hn_wrapped", False):
        return False
    count = [0]

    def wrapper(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            count[0] += 1
            dt = time.perf_counter() - t
            if dt > threshold:
                print(f"[HyNomasGlue] t={_now():6.1f}s {label} call #{count[0]} took {dt:.1f}s", flush=True)

    wrapper._hn_wrapped = True
    setattr(mod, name, wrapper)
    return True


def _persist_caches(vae_path):
    """Point the Inductor and Triton caches at the persistent models folder so a machine that already compiled
    the decoder's FlexAttention kernels can reuse them on the next job."""
    import os
    root = os.path.dirname(os.path.dirname(vae_path))
    cache = os.path.join(root, "_hynomas_cache")
    try:
        os.makedirs(cache, exist_ok=True)
        probe = os.path.join(cache, ".probe")
        with open(probe, "w") as f:
            f.write("x")
        os.remove(probe)
    except Exception as e:
        print(f"[HyNomasGlue] cache dir {cache} not writable ({e}); caches stay ephemeral", flush=True)
        return
    n = sum(len(fs) for _, _, fs in os.walk(cache))
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(cache, "inductor")
    os.environ["TRITON_CACHE_DIR"] = os.path.join(cache, "triton")
    print(f"[HyNomasGlue] persistent compile caches at {cache} ({n} files already there)", flush=True)


def _prefetch(path):
    t = time.perf_counter()
    n = 0
    try:
        buf = bytearray(_CHUNK)
        with open(path, "rb", buffering=0) as f:
            while True:
                got = f.readinto(buf)
                if not got:
                    break
                n += got
    except Exception as e:
        print(f"[HyNomasGlue] prefetch failed: {e}", flush=True)
        return
    dt = time.perf_counter() - t
    print(f"[HyNomasGlue] t={_now():6.1f}s prefetch of decoder file: {n / 1e9:.1f} GB in {dt:.1f}s "
          f"({n / 1e6 / max(dt, 1e-6):.0f} MB/s)", flush=True)


def start():
    """Idempotent; call from the first node that executes."""
    global _T0, _DONE
    if _DONE:
        return
    _DONE = True
    _T0 = time.perf_counter()
    wrapped = 0
    for name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if name.endswith("hy_nomas.pixelumm"):
            for fn in ("load_state_dict", "decode", "decode_clip", "local_attention_layout"):
                wrapped += _wrap(mod, fn, "pixelumm." + fn)
            wrapped += _wrap_slow(mod, "_flex_attention", "pixelumm.flex_attention", 1.0)
            blender = getattr(mod, "TemporalBlender", None)
            if blender is not None:
                wrapped += _wrap(blender, "add", "pixelumm.window_done")
        elif name.endswith("hy_nomas.loader"):
            wrapped += _wrap(mod, "load_dit", "loader.load_dit")
        elif name.endswith("hy_nomas.session"):
            for fn in ("segment_conditioning", "extend_world", "start_world"):
                wrapped += _wrap(mod, fn, "session." + fn)
        elif name.endswith("hy_nomas.dcae"):
            for fn in ("load_encoder", "encode"):
                wrapped += _wrap(mod, fn, "dcae." + fn)
    print(f"[HyNomasGlue] timing wrappers installed on {wrapped} functions", flush=True)
    try:
        import folder_paths
        names = [n for n in folder_paths.get_filename_list("vae") if "pixelumm" in n.lower()]
        if names:
            path = folder_paths.get_full_path("vae", names[0])
            _persist_caches(path)
            threading.Thread(target=_prefetch, args=(path,), daemon=True).start()
            print(f"[HyNomasGlue] prefetching {names[0]} in the background", flush=True)
    except Exception as e:
        print(f"[HyNomasGlue] prefetch not started: {e}", flush=True)
