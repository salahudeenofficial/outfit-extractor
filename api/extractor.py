"""Outfit extraction logic with fixed prompt for consistent results."""

from PIL import Image
import numpy as np
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
    
    # Run inference using LightX2V generate method
    # LightX2VPipeline uses generate() method with specific parameter names
    # Try different parameter name combinations for input image
    parameter_combinations = [
        # Pattern 1: input_image parameter
        {
            'prompt': prompt,
            'input_image': processed_image,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'height': height,
            'width': width
        },
        # Pattern 2: img parameter
        {
            'prompt': prompt,
            'img': processed_image,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
        },
        # Pattern 3: input_img parameter
        {
            'prompt': prompt,
            'input_img': processed_image,
            'steps': num_inference_steps,
            'guidance': guidance_scale,
        },
        # Pattern 4: source_image parameter
        {
            'prompt': prompt,
            'source_image': processed_image,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
        },
        # Pattern 5: Just prompt and image (minimal)
        {
            'prompt': prompt,
            'image': processed_image,
        },
    ]
    
    output = None
    last_error = None
    
    if hasattr(pipeline, 'generate'):
        for i, params in enumerate(parameter_combinations, 1):
            try:
                output = pipeline.generate(**params)
                break
            except TypeError as e:
                last_error = e
                continue
            except Exception as e:
                last_error = e
                # If it's not a TypeError, might be a different issue, try next
                continue
        
        if output is None:
            raise RuntimeError(
                f"Failed to generate image. Tried {len(parameter_combinations)} parameter combinations. "
                f"Last error: {last_error}. "
                f"Please check LightX2V documentation for generate() method parameters."
            )
    else:
        raise RuntimeError("Pipeline does not have generate() method")
    
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
