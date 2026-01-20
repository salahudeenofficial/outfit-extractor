"""Outfit extraction logic with fixed prompt for consistent results.

Diffusers branch: Uses HuggingFace Diffusers pipeline API.
Key: Use 3:4 aspect ratio and Diffusers pipeline call interface.
"""

from PIL import Image
import numpy as np
import tempfile
import os
import json
import logging
import torch
from typing import Optional, Any, Tuple

logger = logging.getLogger(__name__)


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
    
    # Calculate target resolution based on input aspect ratio
    orig_width, orig_height = input_image.size
    target_width, target_height, target_aspect_ratio = calculate_target_resolution(orig_width, orig_height)
    
    # Preprocess image to target dimensions
    processed_image = preprocess_image(input_image, target_width, target_height)
    
    # Run inference using Diffusers pipeline
    output = None
    last_error = None
    
    try:
        # Define negative prompt
        negative_prompt = "blurry, low quality, distorted, artifacts, deformed, bad anatomy"
        
        # Diffusers pipeline call
        # Use __call__ method with image, prompt, num_inference_steps, guidance_scale
        logger.info(f"Running Diffusers inference: {num_inference_steps} steps, size={target_width}x{target_height}")
        
        result = pipeline(
            image=processed_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=target_height,
            width=target_width,
            generator=torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(42),
        )
        
        # Diffusers returns a dict with 'images' key containing list of PIL Images
        if isinstance(result, dict) and "images" in result:
            images = result["images"]
            if isinstance(images, list) and len(images) > 0:
                output = images[0]
            elif isinstance(images, Image.Image):
                output = images
        elif isinstance(result, Image.Image):
            output = result
        else:
            raise ValueError(f"Unexpected output format from Diffusers pipeline: {type(result)}")
            
    except Exception as e:
        last_error = e
        logger.error(f"Diffusers inference failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    if output is None:
        error_msg = str(last_error) if last_error else "Unknown error"
        error_type = type(last_error).__name__ if last_error else "Unknown"
        raise RuntimeError(
            f"Failed to generate image using Diffusers pipeline. "
            f"Error ({error_type}): {error_msg}. "
            f"Pipeline type: {type(pipeline).__name__}."
        )
    
    # Output should already be a PIL Image from Diffusers
    if not isinstance(output, Image.Image):
        raise ValueError(f"Unexpected output format from Diffusers pipeline: {type(output)}. Expected PIL Image.")
    
    return postprocess_image(output, ensure_white_bg=True)
