"""Model loader for Qwen-Image-Edit-2511 FP8 quantized model using LightX2V framework."""

import os
import torch
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def load_model(
    model_path: Optional[str] = None,
    device: str = "cuda",
    cache_dir: Optional[str] = None
):
    """
    Load Qwen-Image-Edit-2511 FP8 quantized model using LightX2V framework.
    
    Args:
        model_path: Path to model checkpoint. If None, downloads from HuggingFace Hub.
        device: Device to load model on ('cuda' or 'cpu').
        cache_dir: Directory to cache downloaded models.
    
    Returns:
        Loaded model pipeline.
    """
    if cache_dir is None:
        cache_dir = os.environ.get("MODEL_CACHE_DIR", "/workspace/models")
    
    if model_path is None:
        model_path = os.environ.get(
            "MODEL_PATH",
            "lightx2v/Qwen-Image-Edit-2511-Lightning"
        )
    
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Set device='cpu' or ensure GPU is accessible.")
    
    try:
        from lightx2v import LightX2VPipeline
        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Using device: {device}")
        
        # LightX2V uses constructor initialization
        # Try different initialization patterns
        initialization_patterns = [
            # Pattern 1: model_path as first positional argument
            lambda: LightX2VPipeline(model_path),
            # Pattern 2: model_path keyword argument
            lambda: LightX2VPipeline(model_path=model_path),
            # Pattern 3: model_id keyword argument
            lambda: LightX2VPipeline(model_id=model_path),
            # Pattern 4: With device parameter
            lambda: LightX2VPipeline(model_path, device=device),
            # Pattern 5: model_path and device as keywords
            lambda: LightX2VPipeline(model_path=model_path, device=device),
        ]
        
        pipeline = None
        last_error = None
        
        for i, init_func in enumerate(initialization_patterns, 1):
            try:
                logger.info(f"Trying initialization pattern {i}...")
                pipeline = init_func()
                logger.info(f"Successfully initialized pipeline using pattern {i}")
                break
            except TypeError as e:
                last_error = e
                logger.debug(f"Pattern {i} failed: {e}")
                continue
            except Exception as e:
                last_error = e
                logger.warning(f"Pattern {i} raised unexpected error: {e}")
                continue
        
        if pipeline is None:
            # Provide helpful error message
            error_msg = (
                f"Failed to initialize LightX2VPipeline with model_path='{model_path}'. "
                f"Tried {len(initialization_patterns)} different initialization patterns. "
                f"Last error: {last_error}. "
                f"Please check LightX2V documentation for the correct initialization API."
            )
            raise RuntimeError(error_msg)
        
        # Move to device if needed and not already set
        if device == "cuda" and torch.cuda.is_available():
            try:
                if hasattr(pipeline, 'to'):
                    pipeline = pipeline.to(device)
                elif hasattr(pipeline, 'cuda'):
                    pipeline = pipeline.cuda()
            except Exception as e:
                logger.warning(f"Could not move pipeline to {device}: {e}")
        
        return pipeline
        
    except ImportError as e:
        error_msg = (
            f"Failed to import LightX2V framework. "
            f"LightX2V is required for this model. "
            f"Please install it using one of these methods:\n"
            f"  1. pip install git+https://github.com/ModelTC/LightX2V.git\n"
            f"  2. pip install lightx2v\n"
            f"  3. Check LightX2V documentation for container-specific installation\n"
            f"Original error: {e}"
        )
        raise ImportError(error_msg)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


def get_model_info() -> dict:
    """Get information about the loaded model."""
    return {
        "model_name": "Qwen-Image-Edit-2511-Lightning",
        "quantization": "FP8",
        "framework": "LightX2V",
        "inference_steps": 4,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
