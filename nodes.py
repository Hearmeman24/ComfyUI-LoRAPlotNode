import comfy.sd
import comfy.utils
import os
import re
import traceback
import folder_paths
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch

class LoRAPlotNode:
    """
    A ComfyUI node that takes multiple LoRAs and strength values,
    applies each combination to the base model/clip, and outputs
    multiple model/clip pairs for downstream processing.
    """
    # Class-level cache for LoRA dictionaries to avoid reloading
    _lora_cache = {}
    _cache_max_size = 10  # Limit cache size to prevent memory issues
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get list of available LoRAs
        loras = folder_paths.get_filename_list("loras")
        # Add "None" option to the beginning
        lora_selections = ["None"] + loras
        
        inputs = {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "strengths": ("STRING", {
                    "multiline": False,
                    "default": "0.8,0.9,1.0"
                }),
            },
            "optional": {}
        }
        
        # Add 10 LoRA selectors with default "None"
        for i in range(1, 11):
            inputs["optional"][f"lora_{i}"] = (lora_selections, {"default": "None"})
            
        return inputs
    
    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "metadata")
    OUTPUT_IS_LIST = (True, True, True)
    FUNCTION = "apply_loras"
    CATEGORY = "LoRA"
    
    def apply_loras(self, model, clip, strengths, lora_1="None", lora_2="None", lora_3="None", 
                    lora_4="None", lora_5="None", lora_6="None", lora_7="None", 
                    lora_8="None", lora_9="None", lora_10="None"):
        """
        Apply multiple LoRAs with different strength values.
        """
        # Collect all selected LoRAs
        lora_list = []
        for lora_name in [lora_1, lora_2, lora_3, lora_4, lora_5, lora_6, lora_7, lora_8, lora_9, lora_10]:
            if lora_name and lora_name != "None":
                lora_list.append(lora_name)
        
        # Parse strength values
        try:
            strength_list = [float(s.strip()) for s in strengths.split(",") if s.strip()]
        except ValueError as e:
            raise ValueError(f"Invalid strength values: {strengths}. Must be comma-separated floats.")
        
        if not lora_list:
            raise ValueError("At least one LoRA must be selected.")
        if not strength_list:
            raise ValueError("At least one strength value must be provided.")
        
        models_output = []
        clips_output = []
        metadata_output = []
        error_messages = []
        
        # CRITICAL: We must use the original base model/clip for each iteration
        # load_lora_for_models may modify objects in place, so we need to ensure
        # each combination starts from the same base state
        base_model = model
        base_clip = clip
        
        # Iterate over each LoRA (outer loop)
        for lora_name in lora_list:
            # Validate LoRA file exists
            lora_path = folder_paths.get_full_path("loras", lora_name)
            if not lora_path or not os.path.exists(lora_path):
                error_msg = f"LoRA file not found: {lora_name}"
                error_messages.append(error_msg)
                print(f"[LoRAPlotNode] Error: {error_msg}")
                continue
            
            try:
                # OPTIMIZATION: Check cache first, load only if not cached
                if lora_name in self._lora_cache:
                    print(f"[LoRAPlotNode] Using cached LoRA: '{lora_name}'")
                    lora_dict = self._lora_cache[lora_name]
                else:
                    print(f"[LoRAPlotNode] Loading LoRA into RAM: '{lora_name}'")
                    lora_dict = comfy.utils.load_torch_file(lora_path, safe_load=True)
                    
                    # Add to cache (with size limit to prevent memory issues)
                    if len(self._lora_cache) >= self._cache_max_size:
                        # Remove oldest entry (simple FIFO eviction)
                        oldest_key = next(iter(self._lora_cache))
                        del self._lora_cache[oldest_key]
                        print(f"[LoRAPlotNode] Evicted '{oldest_key}' from cache (max size: {self._cache_max_size})")
                    self._lora_cache[lora_name] = lora_dict
                
                # Cache sanitized filename (used for all strengths of this LoRA)
                sanitized_lora = self._sanitize_filename(lora_name)
                
                # Now iterate strengths using the already loaded dictionary
                for strength in strength_list:
                    try:
                        # Apply LoRA to model and clip using the cached lora_dict
                        model_lora, clip_lora = comfy.sd.load_lora_for_models(
                            base_model, base_clip, lora_dict, strength, strength
                        )
                        
                        models_output.append(model_lora)
                        clips_output.append(clip_lora)
                        
                        # Generate metadata string using cached sanitized name
                        metadata = f"{sanitized_lora}_{strength}"
                        metadata_output.append(metadata)
                        
                    except Exception as e_inner:
                        # Collect error details for this strength
                        error_msg = f"LoRA '{lora_name}' strength {strength}: {str(e_inner)}"
                        error_messages.append(error_msg)
                        print(f"[LoRAPlotNode] {error_msg}")
                        print(f"[LoRAPlotNode] Traceback: {traceback.format_exc()}")
                        # Continue with next strength
                        continue
                
                # MEMORY MANAGEMENT: Explicitly delete the heavy dictionary after all strengths are processed
                del lora_dict
                
            except Exception as e_outer:
                # Error loading the LoRA file itself
                error_msg = f"Failed to load LoRA file '{lora_name}': {str(e_outer)}"
                error_messages.append(error_msg)
                print(f"[LoRAPlotNode] {error_msg}")
                print(f"[LoRAPlotNode] Traceback: {traceback.format_exc()}")
                continue
        
        # Ensure we have at least one successful combination
        if not models_output:
            error_details = f"Selected LoRAs: {lora_list}, Strengths: {strength_list}"
            if error_messages:
                error_summary = "\n".join([f"  - {msg}" for msg in error_messages])
                raise ValueError(f"No LoRA-strength combinations were successfully applied.\n\n{error_details}\n\nErrors:\n{error_summary}")
            else:
                raise ValueError(f"No LoRA-strength combinations were successfully applied. {error_details}")
        
        return (models_output, clips_output, metadata_output)
    
    def _sanitize_filename(self, filename):
        """
        Sanitize a filename by removing invalid characters.
        """
        # Remove path separators and get just the filename
        basename = os.path.basename(filename)
        # Remove extension
        name_without_ext = os.path.splitext(basename)[0]
        # Replace invalid characters with underscores
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', name_without_ext)
        # Remove any leading/trailing dots or spaces
        sanitized = sanitized.strip('. ')
        return sanitized if sanitized else "lora"


class LoRAPlotImageSaver:
    """
    A ComfyUI node that saves images with text overlay showing LoRA name and strength.
    Takes images and metadata strings from LoRAPlotNode and overlays text on each image.
    """
    # Class-level cache for fonts to avoid reloading
    _font_cache = {}
    
    # Available color options for dropdowns
    COLOR_OPTIONS = [
        "white",
        "black",
        "red",
        "green",
        "blue",
        "yellow",
        "cyan",
        "magenta",
        "orange",
        "gray",
        "lightgray",
        "darkgray"
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "metadata": ("STRING",),
                "text_color": (cls.COLOR_OPTIONS, {
                    "default": "white"
                }),
                "background_color": (cls.COLOR_OPTIONS, {
                    "default": "black"
                }),
                "font_size": ("INT", {
                    "default": 24,
                    "min": 8,
                    "max": 128,
                    "step": 1
                }),
                "padding": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 50,
                    "step": 1
                }),
                "opacity": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "save_with_overlay"
    CATEGORY = "LoRA"
    
    def save_with_overlay(self, images, metadata, text_color="white", background_color="black", 
                          font_size=24, padding=10, opacity=0.8):
        """
        Overlay text on images showing LoRA name and strength.
        
        Args:
            images: List of images (tensors)
            metadata: List of metadata strings (format: "lora_name_strength")
            text_color: Text color (e.g., "white", "black", "#FFFFFF")
            background_color: Background color for text box
            font_size: Font size in pixels
            padding: Padding around text in pixels
            opacity: Opacity of the background box (0.0 to 1.0)
        """
        # Handle single image vs list
        if not isinstance(images, list):
            images = [images]
        if not isinstance(metadata, list):
            metadata = [metadata]
        
        # Ensure metadata list matches images list length
        if len(metadata) != len(images):
            # If single metadata string provided, use it for all images
            if len(metadata) == 1:
                metadata = metadata * len(images)
            else:
                raise ValueError(f"Metadata list length ({len(metadata)}) must match images list length ({len(images)})")
        
        output_images = []
        
        for img, meta in zip(images, metadata):
            # Convert ComfyUI tensor to PIL Image
            # ComfyUI images are in format: [batch, height, width, channels] with values 0-1
            img_np = img.cpu().numpy()
            if len(img_np.shape) == 4:
                img_np = img_np[0]  # Take first batch item
            # Convert from 0-1 range to 0-255 range
            img_np = (img_np * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np)
            
            # Parse metadata: format is "lora_name_strength"
            # Split on last underscore to separate name and strength
            parts = meta.rsplit('_', 1)
            if len(parts) == 2:
                lora_name, strength = parts
            else:
                lora_name = meta
                strength = ""
            
            # Format text: "LoRA Name\nStrength: 0.8"
            if strength:
                overlay_text = f"{lora_name}\nStrength: {strength}"
            else:
                overlay_text = lora_name
            
            # Create overlay
            pil_image = self._add_text_overlay(
                pil_image, overlay_text, text_color, background_color, 
                font_size, padding, opacity
            )
            
            # Convert back to ComfyUI tensor format
            img_np = np.array(pil_image).astype(np.float32) / 255.0
            # Add batch dimension: [1, height, width, channels]
            img_np = img_np[np.newaxis, ...]
            # Convert to torch tensor (ComfyUI expects torch tensors, not numpy arrays)
            img_tensor = torch.from_numpy(img_np)
            output_images.append(img_tensor)
        
        return (output_images,)
    
    def _get_font(self, font_size):
        """
        Get font, caching by size to avoid reloading.
        """
        if font_size not in self._font_cache:
            try:
                # Try to use a nice font if available
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except:
                try:
                    # Try another common font path
                    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
                except:
                    # Fallback to default font
                    font = ImageFont.load_default()
            self._font_cache[font_size] = font
        return self._font_cache[font_size]
    
    def _add_text_overlay(self, image, text, text_color, bg_color, font_size, padding, opacity):
        """
        Add text overlay to the top-right corner of an image.
        """
        # Create a copy to avoid modifying the original
        img = image.copy()
        draw = ImageDraw.Draw(img, 'RGBA')
        
        # Use cached font
        font = self._get_font(font_size)
        
        # Get text bounding box
        lines = text.split('\n')
        bboxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
        max_width = max(bbox[2] - bbox[0] for bbox in bboxes)
        total_height = sum(bbox[3] - bbox[1] for bbox in bboxes) + (len(lines) - 1) * 5  # 5px line spacing
        
        # Calculate position (top-right corner)
        img_width, img_height = img.size
        box_width = max_width + (padding * 2)
        box_height = total_height + (padding * 2)
        x = img_width - box_width - padding
        y = padding
        
        # Draw semi-transparent background box
        bg_rgba = self._color_to_rgba(bg_color, opacity)
        draw.rectangle(
            [(x, y), (x + box_width, y + box_height)],
            fill=bg_rgba
        )
        
        # Draw text
        text_rgba = self._color_to_rgba(text_color, 1.0)
        y_offset = y + padding
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_height = bbox[3] - bbox[1]
            draw.text(
                (x + padding, y_offset),
                line,
                fill=text_rgba,
                font=font
            )
            y_offset += line_height + 5  # 5px line spacing
        
        return img
    
    def _color_to_rgba(self, color_str, alpha):
        """
        Convert color string to RGBA tuple.
        Supports: "white", "black", "#FFFFFF", "#FFFFFF00", "rgb(255,255,255)"
        """
        color_str = color_str.strip().lower()
        
        # Named colors (expanded to match COLOR_OPTIONS)
        color_map = {
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "cyan": (0, 255, 255),
            "magenta": (255, 0, 255),
            "orange": (255, 165, 0),
            "gray": (128, 128, 128),
            "lightgray": (211, 211, 211),
            "darkgray": (169, 169, 169),
        }
        
        if color_str in color_map:
            r, g, b = color_map[color_str]
        elif color_str.startswith("#"):
            # Hex color
            hex_color = color_str[1:]
            if len(hex_color) == 6:
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
            elif len(hex_color) == 8:
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                alpha = int(hex_color[6:8], 16) / 255.0
            else:
                r, g, b = 255, 255, 255  # Default to white
        elif color_str.startswith("rgb"):
            # RGB format: "rgb(255,255,255)"
            match = re.search(r'rgb\((\d+),(\d+),(\d+)\)', color_str)
            if match:
                r, g, b = int(match.group(1)), int(match.group(2)), int(match.group(3))
            else:
                r, g, b = 255, 255, 255
        else:
            r, g, b = 255, 255, 255  # Default to white
        
        return (r, g, b, int(alpha * 255))
