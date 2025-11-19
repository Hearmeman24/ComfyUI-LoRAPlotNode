# ComfyUI LoRA Plot Node

A custom ComfyUI node for testing multiple LoRA models across different strength values. Automatically generates model/clip pairs for each LoRA-strength combination and includes an image saver node with text overlay for easy identification.

## Features

- Test multiple LoRAs with different strength values in a single workflow
- Automatic generation of model/clip pairs for each combination
- Image saver node with text overlay showing LoRA name and strength
- Optimized performance with LoRA dictionary caching
- Font caching for faster image processing
- Color dropdowns for easy customization

## Installation

### Manual Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Hearmeman24/ComfyUI-LoRAPlotNode.git
```

Then restart ComfyUI.

## Usage

### LoRA Plot Node

1. Add the "LoRA Plot Node" to your workflow
2. Connect your base model and CLIP
3. Select up to 10 LoRAs from the dropdown menus
4. Enter comma-separated strength values (e.g., "0.8,0.9,1.0")
5. Connect the outputs to your KSampler nodes

The node will output:
- Multiple model outputs (one per LoRA-strength combination)
- Multiple CLIP outputs (one per LoRA-strength combination)
- Metadata strings for each combination

### LoRA Plot Image Saver

1. Add the "LoRA Plot Image Saver" node
2. Connect your decoded images
3. Connect the metadata output from LoRA Plot Node
4. Customize text color, background color, font size, padding, and opacity
5. The node will overlay text on each image showing the LoRA name and strength

## Example Workflow

1. Load Model -> LoRA Plot Node -> KSampler -> VAE Decode -> LoRA Plot Image Saver -> Save Image

The LoRA Plot Node will automatically create multiple model/clip pairs, and ComfyUI will execute the workflow once for each combination.

## Performance Optimizations

- LoRA files are cached in memory (up to 10 LoRAs) to avoid redundant disk I/O
- Fonts are cached by size for faster image processing
- LoRA files are loaded once per LoRA, not per strength value

## Requirements

- ComfyUI
- PIL/Pillow (usually included with ComfyUI)
- NumPy (usually included with ComfyUI)
- PyTorch (usually included with ComfyUI)

## License

MIT

