from .script_compile import compile_script, DEFAULT_SCRIPT


class HyNomasScriptCompile:
    """Plain-text event script -> Hy-Nomas world JSON string (feeds HyNomasWorldPrompt.world_json)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "script": ("STRING", {"multiline": True, "default": DEFAULT_SCRIPT}),
            "length": ("INT", {"default": 0, "min": 0, "max": 960,
                               "tooltip": "Total frames at 24 fps (0 = what the script says). Trims or extends the script."}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("world_json",)
    FUNCTION = "run"
    CATEGORY = "Hy-Nomas"

    def run(self, script, length):
        from .autotune_patch import apply
        from . import diagnostics
        apply()
        diagnostics.start()
        return (compile_script(script, length),)


NODE_CLASS_MAPPINGS = {"HyNomasScriptCompile": HyNomasScriptCompile}
NODE_DISPLAY_NAME_MAPPINGS = {"HyNomasScriptCompile": "Hy-Nomas Script Compile"}
