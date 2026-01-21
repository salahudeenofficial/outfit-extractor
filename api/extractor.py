"""Outfit extraction logic with fixed prompt for consistent results.

Based on the working VTON project (try_og_pipeline).
Key: Use 3:4 aspect ratio and monkey-patch run_pipeline to inject aspect_ratio.
"""

from PIL import Image
import numpy as np
import tempfile
import os
import json
from typing import Optional, Any, Tuple


# Fixed prompt for outfit extraction
OUTFIT_EXTRACTION_PROMPT = (
    "Extract the FULL OUTFIT, not a single garment. "
    "All worn clothing must be included. "
    "STRICT: keep original colors exactly as in the input image. "
    "No color change, no hue shift, no saturation or brightness change. "
    "Preserve fabric texture and patterns. "
    "No human present. Only the outfit on white background."
)

# Target resolution for 3:4 aspect ratio (portrait)
TARGET_WIDTH = 768
TARGET_HEIGHT = 1024
TARGET_ASPECT_RATIO = "3:4"


def calculate_target_resolution(orig_width: int, orig_height: int) -> Tuple[int, int, str]:
    """
    Calculate target resolution maintaining aspect ratio.
    Uses 3:4 for portrait, 4:3 for landscape, 1:1 for square.
    
    From VTON project: dimensions must be divisible by 16.
    """
    orig_ratio = orig_width / orig_height
    
    if orig_ratio < 0.8:  # Portrait (taller than wide)
        target_aspect_ratio = "3:4"
        target_width = TARGET_WIDTH
        target_height = TARGET_HEIGHT
    elif orig_ratio > 1.2:  # Landscape (wider than tall)
        target_aspect_ratio = "4:3"
        target_width = TARGET_HEIGHT  # Swap for landscape
        target_height = TARGET_WIDTH
    else:  # Square-ish
        target_aspect_ratio = "1:1"
        target_width = 1024
        target_height = 1024
    
    # Ensure dimensions are divisible by 16
    target_width = (target_width // 16) * 16
    target_height = (target_height // 16) * 16
    
    return target_width, target_height, target_aspect_ratio


def preprocess_image(image: Image.Image, target_width: int = TARGET_WIDTH, target_height: int = TARGET_HEIGHT) -> Image.Image:
    """
    Preprocess input image for model inference.
    Resize to target dimensions (3:4 aspect ratio by default).
    
    Args:
        image: Input PIL Image.
        target_width: Target width (default 768 for 3:4).
        target_height: Target height (default 1024 for 3:4).
    
    Returns:
        Preprocessed PIL Image resized to target dimensions.
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image object")
    
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize to target dimensions (maintains consistency with VTON project)
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def postprocess_image(image: Image.Image, ensure_white_bg: bool = True) -> Image.Image:
    """
    Post-process extracted outfit image.
    
    Args:
        image: Extracted outfit image from model.
        ensure_white_bg: Whether to ensure pure white background.
    
    Returns:
        Post-processed PIL Image with white background.
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image object")
    
    if not ensure_white_bg:
        return image
    
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    img_array = np.array(image)
    white_threshold = 240
    mask = np.any(img_array < white_threshold, axis=2)
    white_bg = np.ones_like(img_array) * 255
    
    result_array = np.where(
        mask[:, :, np.newaxis],
        img_array,
        white_bg
    )
    
    return Image.fromarray(result_array.astype(np.uint8))


def extract_outfit(
    pipeline: Any,
    input_image: Image.Image,
    prompt: Optional[str] = None,
    num_inference_steps: int = 20,
    guidance_scale: float = 1.0,
    height: Optional[int] = None,
    width: Optional[int] = None
) -> Image.Image:
    """
    Extract outfit from input image using the model pipeline.
    
    Args:
        pipeline: Loaded model pipeline.
        input_image: Input PIL Image containing person with outfit.
        prompt: Custom prompt (uses default if None).
        num_inference_steps: Number of inference steps (default: 20).
            Note: For LightX2V, steps are set during generator creation, this parameter is for API compatibility.
        guidance_scale: Guidance scale for generation.
        height: Output height (uses input height if None).
        width: Output width (uses input width if None).
    
    Returns:
        Extracted outfit image on white background.
    """
    if prompt is None:
        prompt = OUTFIT_EXTRACTION_PROMPT
    
    # Calculate target resolution based on input aspect ratio
    orig_width, orig_height = input_image.size
    target_width, target_height, target_aspect_ratio = calculate_target_resolution(orig_width, orig_height)
    
    # Preprocess image to target dimensions
    processed_image = preprocess_image(input_image, target_width, target_height)
    
    # Run inference using LightX2V
    output = None
    last_error = None
    
    # Save PIL Image to temporary file for image_path parameter
    temp_input_file = None
    temp_output_file = None
    try:
        # Create temporary file for input image
        temp_fd, temp_input_path = tempfile.mkstemp(suffix='.png')
        os.close(temp_fd)
        processed_image.save(temp_input_path, format='PNG')
        temp_input_file = temp_input_path
        
        # Create temporary file for output image
        temp_fd2, temp_output_path = tempfile.mkstemp(suffix='.png')
        os.close(temp_fd2)
        temp_output_file = temp_output_path
        
        # Define negative prompt
        negative_prompt = "blurry, low quality, distorted, artifacts, deformed, bad anatomy"
        
        # Monkey-patch run_pipeline to inject aspect_ratio into input_info
        # This is needed because LightX2V's get_custom_shape() checks input_info.aspect_ratio
        # (From PROBLEMS_FACED.txt in VTON project)
        if hasattr(pipeline, 'runner') and hasattr(pipeline.runner, 'run_pipeline'):
            original_run_pipeline = pipeline.runner.run_pipeline
            
            def patched_run_pipeline(input_info):
                input_info.aspect_ratio = target_aspect_ratio
                # Also need to set _auto_resize in config (get_custom_shape checks it)
                pipeline.runner.config["_auto_resize"] = False
                return original_run_pipeline(input_info)
            
            pipeline.runner.run_pipeline = patched_run_pipeline
        
        try:
            # Generate with simple parameters
            # Resolution is controlled by pre-resizing the input image
            output = pipeline.generate(
                seed=42,
                image_path=temp_input_path,
                prompt=prompt,
                negative_prompt=negative_prompt,
                save_result_path=temp_output_path
            )
        except Exception as e:
            last_error = e
        finally:
            # Restore original method
            if hasattr(pipeline, 'runner') and hasattr(pipeline.runner, 'run_pipeline'):
                try:
                    pipeline.runner.run_pipeline = original_run_pipeline
                except:
                    pass
        
        # If output is None but save_result_path was used, load from file
        if output is None and temp_output_file and os.path.exists(temp_output_file):
            file_size = os.path.getsize(temp_output_file)
            if file_size > 0:
                output = Image.open(temp_output_file)
                
    finally:
        # Clean up temporary input file
        if temp_input_file and os.path.exists(temp_input_file):
            try:
                os.unlink(temp_input_file)
            except Exception:
                pass
        # Keep output file until we've processed it (cleanup after extraction)
    
    if output is None:
        error_msg = str(last_error) if last_error else "Unknown error"
        error_type = type(last_error).__name__ if last_error else "Unknown"
        raise RuntimeError(
            f"Failed to generate image. Tried multiple approaches. "
            f"Last error ({error_type}): {error_msg}. "
            f"Pipeline type: {type(pipeline).__name__}. "
            f"Pipeline methods available: {[m for m in dir(pipeline) if not m.startswith('_') and callable(getattr(pipeline, m, None))]}. "
            f"Please check LightX2V documentation for image editing task API."
        )
    
    # Extract image from output
    result_image = None
    
    # If output is already a PIL Image
    if isinstance(output, Image.Image):
        result_image = output
    # LightX2V generate() may return a dict with 'images' key or list
    elif isinstance(output, dict):
        result_image = output.get("images", output.get("image", output.get("result")))
        if result_image is None:
            # Try to find any image-like value in the dict
            for key, val in output.items():
                if isinstance(val, Image.Image):
                    result_image = val
                    break
                elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], Image.Image):
                    result_image = val[0]
                    break
        if isinstance(result_image, list) and len(result_image) > 0:
            result_image = result_image[0]
    elif isinstance(output, list) and len(output) > 0:
        result_image = output[0]
    elif hasattr(output, 'images'):
        result_image = output.images[0] if isinstance(output.images, list) else output.images
    elif hasattr(output, 'image'):
        result_image = output.image
    
    # If still not a PIL Image, try to load from output file
    if result_image is None or not isinstance(result_image, Image.Image):
        if temp_output_file and os.path.exists(temp_output_file):
            file_size = os.path.getsize(temp_output_file)
            if file_size > 0:
                result_image = Image.open(temp_output_file).copy()
    
    # Clean up output file now
    if temp_output_file and os.path.exists(temp_output_file):
        try:
            os.unlink(temp_output_file)
        except Exception:
            pass
    
    if result_image is None or not isinstance(result_image, Image.Image):
        raise ValueError(f"Unexpected output format from pipeline: {type(output)}. Could not extract PIL Image.")
    
    return postprocess_image(result_image, ensure_white_bg=True)
