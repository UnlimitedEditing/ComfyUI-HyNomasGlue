"""Pin ComfyUI-Hy-Nomas's Triton GEMM kernels to ONE config so they never autotune.

Why: hy_nomas/kernels.py wraps its block-scaled GEMM in triton.autotune with 7 configs per distinct
(M_BUCKET, N, K, ...) shape. Graydient containers are ephemeral, so the autotune cache is always cold and every
shape is benchmarked from scratch: ~26-34 s on the DiT's first step, and the FP8 decoder (many more shapes)
never finished even a 5 s clip inside the time budget (measured 2026-11-06, RTX 4090 / 5090).
Triton skips benchmarking when an Autotuner has a single config, so replace the list with the first one
((128, 128) tiles, 8 warps, 3 stages: ~96 KB shared memory, fits RTX 30/40/50).

Applied lazily (first node execution), when every custom node package is already imported.
"""
import sys


def apply():
    try:
        import triton
    except Exception as e:  # pragma: no cover
        print(f"[HyNomasGlue] triton not importable, autotune patch skipped: {e}")
        return 0
    pinned = 0
    for name, mod in list(sys.modules.items()):
        if mod is None or not name.endswith("kernels"):
            continue
        for attr, bn in (("_block_gemm", None), ("_block_gemm_n128", 128)):
            tuner = getattr(mod, attr, None)
            configs = getattr(tuner, "configs", None)
            if not configs or len(configs) == 1:
                continue
            # prefer the (128, 128, 8 warps, 3 stages) config the node lists first; fall back to the first one
            pick = next((c for c in configs
                         if c.kwargs.get("BLOCK_M") == 128 and c.kwargs.get("BLOCK_N") == 128
                         and c.num_warps == 8 and c.num_stages == 3), configs[0])
            tuner.configs = [pick]
            if hasattr(tuner, "cache"):
                tuner.cache = {}
            pinned += 1
    print(f"[HyNomasGlue] pinned {pinned} Triton GEMM autotuners to a single config (no autotune)")
    return pinned
