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
    # Correct API pattern: create_generator() sets generation params, generate() takes inputs
    output = None
    last_error = None
    
    # Save PIL Image to temporary file for image_path parameter
    temp_file = None
    try:
        # Create temporary file for image
        temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg')
        os.close(temp_fd)
        processed_image.save(temp_path, format='JPEG', quality=95)
        temp_file = temp_path
        
        # Pattern 1: Correct LightX2V API - create_generator() then generate()
        # Try with target_width/target_height and sample_guide_scale (from config JSON)
        if hasattr(pipeline, 'create_generator') and hasattr(pipeline, 'generate'):
            # Try different parameter name combinations
            param_combinations = [
                # Combination 1: target_width/target_height with sample_guide_scale
                {
                    'create_gen': {
                        'infer_steps': num_inference_steps,
                        'target_width': width,
                        'target_height': height,
                        'sample_guide_scale': guidance_scale
                    },
                    'generate': {
                        'generator': None,  # Will be set
                        'prompt': prompt,
                        'image_path': temp_path,
                        'seed': 42
                    }
                },
                # Combination 2: width/height with guidance_scale
                {
                    'create_gen': {
                        'infer_steps': num_inference_steps,
                        'width': width,
                        'height': height,
                        'guidance_scale': guidance_scale
                    },
                    'generate': {
                        'generator': None,
                        'prompt': prompt,
                        'image_path': temp_path,
                        'seed': 42
                    }
                },
                # Combination 3: target_width/target_height with guidance_scale
                {
                    'create_gen': {
                        'infer_steps': num_inference_steps,
                        'target_width': width,
                        'target_height': height,
                        'guidance_scale': guidance_scale
                    },
                    'generate': {
                        'generator': None,
                        'prompt': prompt,
                        'image_path': temp_path,
                        'seed': 42
                    }
                },
            ]
            
            for combo in param_combinations:
                if output is not None:
                    break
                try:
                    gen_params = combo['create_gen'].copy()
                    generator = pipeline.create_generator(**gen_params)
                    
                    gen_params_call = combo['generate'].copy()
                    gen_params_call['generator'] = generator
                    output = pipeline.generate(**gen_params_call)
                    break
                except Exception as e:
                    last_error = e
                    # Try without generator parameter
                    try:
                        gen_params = combo['create_gen'].copy()
                        generator = pipeline.create_generator(**gen_params)
                        
                        gen_params_call = combo['generate'].copy()
                        gen_params_call.pop('generator', None)
                        output = pipeline.generate(**gen_params_call)
                        break
                    except Exception as e2:
                        last_error = e2
                        # Try with image parameter instead of image_path
                        try:
                            gen_params = combo['create_gen'].copy()
                            generator = pipeline.create_generator(**gen_params)
                            
                            gen_params_call = combo['generate'].copy()
                            gen_params_call.pop('image_path', None)
                            gen_params_call['image'] = processed_image
                            gen_params_call['generator'] = generator
                            output = pipeline.generate(**gen_params_call)
                            break
                        except Exception as e3:
                            last_error = e3
                            continue
        
        # Pattern 2: Try direct generate() without create_generator (if supported)
        if output is None and hasattr(pipeline, 'generate'):
            direct_param_combinations = [
                {
                    'prompt': prompt,
                    'image_path': temp_path,
                    'infer_steps': num_inference_steps,
                    'target_width': width,
                    'target_height': height,
                    'sample_guide_scale': guidance_scale,
                    'seed': 42
                },
                {
                    'prompt': prompt,
                    'image_path': temp_path,
                    'infer_steps': num_inference_steps,
                    'width': width,
                    'height': height,
                    'guidance_scale': guidance_scale,
                    'seed': 42
                },
                {
                    'prompt': prompt,
                    'image': processed_image,
                    'infer_steps': num_inference_steps,
                    'target_width': width,
                    'target_height': height,
                    'sample_guide_scale': guidance_scale,
                    'seed': 42
                },
                {
                    'prompt': prompt,
                    'image': processed_image,
                    'infer_steps': num_inference_steps,
                    'width': width,
                    'height': height,
                    'guidance_scale': guidance_scale,
                    'seed': 42
                },
            ]
            
            for params in direct_param_combinations:
                try:
                    output = pipeline.generate(**params)
                    break
                except Exception as e:
                    if last_error is None:
                        last_error = e
                    continue
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception:
                pass
    
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
    # LightX2V generate() typically returns a dict with 'images' key or list
    if isinstance(output, dict):
        result_image = output.get("images", output.get("image"))
        if result_image is None:
            raise ValueError("Output dict does not contain 'images' or 'image' key")
        if isinstance(result_image, list):
            result_image = result_image[0]
    elif isinstance(output, list):
        result_image = output[0]
    else:
        result_image = output
    
    # Ensure result is PIL Image
    if not isinstance(result_image, Image.Image):
        if hasattr(result_image, 'images'):
            result_image = result_image.images[0]
        elif hasattr(result_image, 'image'):
            result_image = result_image.image
        else:
            raise ValueError(f"Unexpected output format from pipeline: {type(result_image)}")
    
    return postprocess_image(result_image, ensure_white_bg=True)
