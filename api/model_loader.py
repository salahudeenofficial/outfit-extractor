"""Model loader for Qwen-Image-Edit-2511 FP8 quantized model using LightX2V framework."""

import os
import torch
from typing import Optional
from pathlib import Path


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
    
    # Check if CUDA is available
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Set device='cpu' or ensure GPU is accessible.")
    
    try:
        # Import LightX2V pipeline
        # Note: Actual import may vary based on LightX2V API
        # This is a placeholder - adjust based on actual LightX2V API
        from lightx2v import LightX2VPipeline
        
        # Load the FP8 quantized model
        pipeline = LightX2VPipeline.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,  # FP8 models may use float16 wrapper
            device=device
        )
        
        if device == "cuda":
            pipeline = pipeline.to(device)
        
        return pipeline
        
    except ImportError:
        # Fallback: try alternative import paths
        try:
            from diffusers import DiffusionPipeline
            # Alternative loading method if LightX2V uses diffusers-compatible API
            pipeline = DiffusionPipeline.from_pretrained(
                model_path,
                cache_dir=cache_dir,
                torch_dtype=torch.float16,
                device_map=device
            )
            return pipeline
        except Exception as e:
            raise ImportError(
                f"Failed to import LightX2V framework. "
                f"Ensure LightX2V is installed in the container. Error: {e}"
            )
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
