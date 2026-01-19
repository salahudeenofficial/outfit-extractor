"""Outfit extraction logic with fixed prompt for consistent results."""

from PIL import Image
import numpy as np
import tempfile
import os
from typing import Optional, Any


# Fixed prompt for outfit extraction
OUTFIT_EXTRACTION_PROMPT = (
    "Extract the FULL OUTFIT, not a single garment. "
    "All worn clothing must be included. "
    "STRICT: keep original colors exactly as in the input image. "
    "No color change, no hue shift, no saturation or brightness change. "
    "Preserve fabric texture and patterns. "
    "No human present. Only the outfit on white background."
)


def preprocess_image(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """
    Preprocess input image for model inference.
    
    Args:
        image: Input PIL Image.
        max_size: Maximum dimension for resizing while maintaining aspect ratio.
    
    Returns:
        Preprocessed PIL Image.
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image object")
    
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    width, height = image.size
    if max(width, height) <= max_size:
        return image
    
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


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
    num_inference_steps: int = 4,
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
        num_inference_steps: Number of inference steps (4 for Lightning).
        guidance_scale: Guidance scale for generation.
        height: Output height (uses input height if None).
        width: Output width (uses input width if None).
    
    Returns:
        Extracted outfit image on white background.
    """
    if prompt is None:
        prompt = OUTFIT_EXTRACTION_PROMPT
    
    processed_image = preprocess_image(input_image)
    
    if height is None:
        height = processed_image.height
    if width is None:
        width = processed_image.width
    
    # Run inference using LightX2V
    # The pipeline config is already set during initialization, so we can call generate() directly
    # If create_generator() causes KeyError, skip it and use the pre-configured settings
    output = None
    last_error = None
    
    # Save PIL Image to temporary file for image_path parameter
    temp_input_file = None
    temp_output_file = None
    try:
        # Create temporary file for input image
        temp_fd, temp_input_path = tempfile.mkstemp(suffix='.jpg')
        os.close(temp_fd)
        processed_image.save(temp_input_path, format='JPEG', quality=95)
        temp_input_file = temp_input_path
        
        # Create temporary file for output image
        temp_fd2, temp_output_path = tempfile.mkstemp(suffix='.png')
        os.close(temp_fd2)
        temp_output_file = temp_output_path
        
        # Define negative prompt - required by LightX2V generate()
        negative_prompt = "blurry, low quality, distorted, artifacts, deformed, bad anatomy"
        
        # Pattern: Use set_infer_config_json() for config, then create_generator(), then generate()
        # set_infer_config_json accepts target_width/target_height
        # create_generator() sets up internal state (aspect_ratio, etc.)
        # generate() runs the actual inference
        if hasattr(pipeline, 'set_infer_config_json') and hasattr(pipeline, 'create_generator') and hasattr(pipeline, 'generate'):
            try:
                # Step 1: Set config via JSON (this accepts target_width/target_height)
                config = {
                    "infer_steps": num_inference_steps,
                    "target_width": width,
                    "target_height": height,
                    "sample_guide_scale": guidance_scale,
                    "sample_shift": 5.0
                }
                pipeline.set_infer_config_json(config)
                
                # Step 2: Create generator to set up internal state (aspect_ratio, etc.)
                pipeline.create_generator()
                
                # Step 3: Generate
                output = pipeline.generate(
                    seed=42,
                    image_path=temp_input_path,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    save_result_path=temp_output_path
                )
            except Exception as e:
                last_error = e
        
        # Fallback: Try with just create_generator() with minimal params
        if output is None and hasattr(pipeline, 'create_generator') and hasattr(pipeline, 'generate'):
            try:
                # Try create_generator with no args (use defaults from initialization)
                pipeline.create_generator()
                output = pipeline.generate(
                    seed=42,
                    image_path=temp_input_path,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    save_result_path=temp_output_path
                )
            except Exception as e:
                if last_error is None:
                    last_error = e
        
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
