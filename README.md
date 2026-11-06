# ComfyUI-HyNomasGlue

Glue for [PedroMarinhoDev/ComfyUI-Hy-Nomas](https://github.com/PedroMarinhoDev/ComfyUI-Hy-Nomas): one node,
**Hy-Nomas Script Compile**, that turns a plain-text event script into Tencent's structured world JSON
(adds the `<Action1><|Subject1|>` tags, perspective sentence and timestamps). Needed on Graydient because the
chat frontend mangles `<...>` and `[...]`. See `script_compile.py` for the script format.
