from .nodes import LoRAPlotNode, LoRAPlotImageSaver

NODE_CLASS_MAPPINGS = {
    "LoRAPlotNode": LoRAPlotNode,
    "LoRAPlotImageSaver": LoRAPlotImageSaver
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoRAPlotNode": "LoRA Plot Node",
    "LoRAPlotImageSaver": "LoRA Plot Image Saver"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

