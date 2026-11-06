"""Replace the PixelUMM decoder's compiled FlexAttention with plain SDPA + a dense mask.

Why (measured 2026-11-06 on Graydient, RTX 5090 / 4090): the decoder's local (sliding-tile) attention uses
create_block_mask(..., _compile=True) and torch.compile(flex_attention). On an ephemeral container the Inductor cache
is cold, and `local_attention_layout` alone took 159 s before the first decode window could start (the cause of the
85 s vs 198 s vs timeout variance: a machine with a warm cache is fast, a cold one is not).

Equivalent without any compile: build the same tile mask as a dense bool [N, N] tensor straight from the mask_mod
logic (N = frames * h * w tokens, ~243 MB at 832x480, 10 latent frames) and call F.scaled_dot_product_attention
(memory-efficient kernel). K/V heads are expanded by hand because GQA + mask can push SDPA onto the math backend.
"""
import sys

import torch
import torch.nn.functional as F


def local_mask(grid, tile, device):
    """(perm, inverse perm, dense mask [1, 1, N, N]) with the exact semantics of the node's flex mask_mod."""
    frames, height, width = grid
    tiles_h, tiles_w = -(-height // tile), -(-width // tile)
    t, h, w = torch.meshgrid(*(torch.arange(n, device=device) for n in grid), indexing="ij")
    tile_h, tile_w = (h // tile).flatten(), (w // tile).flatten()
    order = (((tile_h * tiles_w + tile_w) * frames + t.flatten()) * tile + (h % tile).flatten()) * tile
    perm = torch.argsort(order + (w % tile).flatten())
    tile_h, tile_w = tile_h[perm], tile_w[perm]
    center_h = tile_h.clamp(1, max(tiles_h - 2, 1))
    center_w = tile_w.clamp(1, max(tiles_w - 2, 1))
    mask = ((center_h[:, None] - tile_h[None, :]).abs() <= 1) & ((center_w[:, None] - tile_w[None, :]).abs() <= 1)
    return perm, torch.argsort(perm), mask[None, None]


def sdpa_attention(q, k, v, mask):
    """q [B, H, N, D], k/v [B, KV, N, D]; `mask` is the dense bool mask from local_mask."""
    rep = q.shape[1] // k.shape[1]
    if rep > 1:
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def install():
    """Patch hy_nomas.pixelumm in place. Returns the number of modules patched."""
    patched = 0
    for name, mod in list(sys.modules.items()):
        if mod is None or not name.endswith("hy_nomas.pixelumm"):
            continue
        if not hasattr(mod, "local_attention_layout") or not hasattr(mod, "_flex_attention"):
            continue
        layouts = getattr(mod, "_LAYOUTS", {})

        def layout(grid, tile, device, _layouts=layouts):
            frames, height, width = grid
            tiles_h, tiles_w = -(-height // tile), -(-width // tile)
            if tiles_h <= 3 and tiles_w <= 3:
                return None
            key = (grid, tile, str(device))
            if key not in _layouts:
                _layouts[key] = local_mask(grid, tile, device)
            return _layouts[key]

        mod.local_attention_layout = layout
        mod._flex_attention = sdpa_attention
        patched += 1
    print(f"[HyNomasGlue] decoder FlexAttention replaced by SDPA + dense mask in {patched} module(s)", flush=True)
    return patched
