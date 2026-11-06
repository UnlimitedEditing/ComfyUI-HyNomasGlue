import json
import sys

from .script_compile import compile_full, DEFAULT_SCRIPT


def _hn_module(suffix):
    """A loaded ComfyUI-Hy-Nomas submodule (registered under whatever name ComfyUI gave the pack folder)."""
    for name, mod in list(sys.modules.items()):
        if mod is not None and name.endswith(suffix):
            return mod
    raise RuntimeError(f"ComfyUI-Hy-Nomas module '{suffix}' is not loaded (is the node pack installed?)")


class HyNomasScriptCompile:
    """Plain-text event script -> Hy-Nomas world JSON string (feeds HyNomasWorldPrompt.world_json) and a JSON
    list of per-segment camera key timelines (feeds HyNomasGenerateWorldPlus.cameras)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "script": ("STRING", {"multiline": True, "default": DEFAULT_SCRIPT}),
            "length": ("INT", {"default": 0, "min": 0, "max": 960,
                               "tooltip": "Total frames at 24 fps (0 = what the script says). Trims or extends the script."}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("world_json", "cameras")
    FUNCTION = "run"
    CATEGORY = "Hy-Nomas"

    def run(self, script, length):
        from .autotune_patch import apply
        from . import diagnostics
        from .fast_attention import install as install_fast_attention
        apply()
        install_fast_attention()
        diagnostics.start()
        return compile_full(script, length)


class HyNomasGenerateWorldPlus:
    """Generate World with (a) a camera key timeline per segment and (b) an optional start picture.

    Same loop as ComfyUI-Hy-Nomas's HyNomasGenerateWorld (segment conditioning, KSampler-equivalent sampling,
    Extend World), but each segment gets its own camera (from the `cameras` JSON list), and if `image_name` names a
    file in the ComfyUI input folder the world starts from that picture (Start From Image): the picture becomes
    segment 0 and generation continues from segment 1 (so the script needs at least 2 segments)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",), "clip": ("CLIP",), "script": ("HYNOMAS_SCRIPT",),
                "width": ("INT", {"default": 832, "min": 32, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 480, "min": 32, "max": 4096, "step": 32}),
                "world_seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "cameras": ("STRING", {"default": "[]", "forceInput": True}),
                "image_name": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("HYNOMAS_WORLD",)
    RETURN_NAMES = ("world",)
    FUNCTION = "run"
    CATEGORY = "Hy-Nomas"

    def run(self, model, clip, script, width, height, world_seed, cameras="[]", image_name=""):
        import comfy.sample
        import folder_paths
        session = _hn_module("hy_nomas.session")
        cam = _hn_module("hy_nomas.camera")

        try:
            timelines = json.loads(cameras) if cameras else []
        except Exception:
            timelines = []
        paths = []
        for i in range(len(script)):
            tl = timelines[i] if i < len(timelines) else None
            paths.append(cam.CameraPath(cam.simulate_keys(cam.parse_timeline(tl)), 1.0, 2.5, tl) if tl else None)

        world = None
        name = (image_name or "").strip()
        if name:
            world = self._start_from_image(model, clip, script, width, height, world_seed, name, folder_paths)

        first = len(world.records) if world is not None else 0
        for index in range(first, len(script)):
            positive, latent = session.segment_conditioning(clip, script, index, width, height, world=world,
                                                            camera=paths[index])
            negative = session.negative_conditioning(clip, positive)
            segment_seed = world_seed + index
            noise = comfy.sample.prepare_noise(latent["samples"], segment_seed)
            samples = comfy.sample.sample(model, noise, 8, 1.0, "euler", "simple", positive, negative,
                                          latent["samples"], seed=segment_seed, callback=None, disable_pbar=False)
            world = session.extend_world(model, positive, {"samples": samples}, seed=segment_seed,
                                         past_frames=18, kv_int8=True)
        return (world,)

    @staticmethod
    def _start_from_image(model, clip, script, width, height, seed, name, folder_paths):
        import numpy as np
        import torch
        from PIL import Image, ImageOps
        session = _hn_module("hy_nomas.session")
        dcae = _hn_module("hy_nomas.dcae")
        if len(script) < 2:
            raise ValueError("a start picture needs at least 2 script segments: segment 0 describes the picture, "
                             "the rest are generated")
        try:
            path = folder_paths.get_annotated_filepath(name)
            img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        except Exception as e:
            raise ValueError(f"could not load the start picture {name!r}: {e}")
        image = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None]
        files = [n for n in folder_paths.get_filename_list("vae") if "pixelumm" in n.lower()]
        if not files:
            raise ValueError("no PixelUMM file in models/vae (it carries the image encoder)")
        encoder = dcae.load_encoder(folder_paths.get_full_path_or_raise("vae", files[0]))
        return session.start_world(model, clip, encoder, script.world, image, width, height, still_frames=3,
                                   seed=seed, past_frames=18, kv_int8=True)


NODE_CLASS_MAPPINGS = {"HyNomasScriptCompile": HyNomasScriptCompile,
                       "HyNomasGenerateWorldPlus": HyNomasGenerateWorldPlus}
NODE_DISPLAY_NAME_MAPPINGS = {"HyNomasScriptCompile": "Hy-Nomas Script Compile",
                              "HyNomasGenerateWorldPlus": "Hy-Nomas Generate World (camera + image)"}
