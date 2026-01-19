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
        # Check for local model first, then fall back to HuggingFace Hub
        local_paths = [
            "/workspace/models/Qwen-Image-Edit-2511",
            "/workspace/outfit-extractor/models/Qwen-Image-Edit-2511",
            "models/Qwen-Image-Edit-2511",
        ]
        
        model_path = os.environ.get("MODEL_PATH")
        if model_path is None:
            for local_path in local_paths:
                if os.path.exists(local_path):
                    model_path = local_path
                    logger.info(f"Found local model at: {model_path}")
                    break
            
            if model_path is None:
                # Fall back to HuggingFace Hub (requires internet)
                model_path = "Qwen/Qwen-Image-Edit-2511"
                logger.warning(f"No local model found. Will try to download from HuggingFace: {model_path}")
    
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Set device='cpu' or ensure GPU is accessible.")
    
    try:
        from lightx2v import LightX2VPipeline
        
        # The correct model_cls for Qwen-Image-Edit is "qwen-image-edit-2511"
        model_cls = "qwen-image-edit-2511"
        task = "i2i"
        
        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Using device: {device}")
        
        logger.info(f"Initializing LightX2VPipeline(model_path='{model_path}', model_cls='{model_cls}', task='{task}')")
        pipeline = LightX2VPipeline(
            model_path=model_path,
            model_cls=model_cls,
            task=task,
        )
        logger.info("Pipeline created successfully")
        
        # Enable CPU offload to manage GPU memory (prevents OOM errors)
        # The model needs ~44GB but offloading allows parts to stay on CPU
        logger.info("Enabling CPU offload to manage GPU memory...")
        pipeline.enable_offload()
                
        # Create generator ONCE during startup (from PROBLEMS_FACED.txt)
        # Calling create_generator() multiple times causes JSON serialization errors
        # Must pass parameters to properly initialize the runner
        logger.info("Creating generator (one-time setup with params)...")
        
        # Get attention mode based on GPU
        attn_mode = "torch_sdpa"  # Safe default
        try:
            import importlib.util
            if importlib.util.find_spec("flash_attn") is not None:
                attn_mode = "flash_attn2"
                logger.info("Using Flash Attention 2")
        except Exception:
            logger.info("Using PyTorch SDPA attention")
        
        # Create generator with required parameters (from working VTON project)
        logger.info(f"Calling create_generator(attn_mode={attn_mode}, infer_steps=4, guidance_scale=1.0, width=768, height=1024)")
        pipeline.create_generator(
            attn_mode=attn_mode,
            infer_steps=4,  # 4 steps for Lightning model
            guidance_scale=1.0,
            width=768,
            height=1024,
            aspect_ratio="3:4",
        )
        
        # Verify runner was created (critical for generate() to work)
        if hasattr(pipeline, 'runner') and pipeline.runner is not None:
            logger.info("Generator created successfully - runner initialized")
        else:
            raise RuntimeError("Pipeline runner not initialized after create_generator()")
        
        logger.info("Pipeline initialized successfully")
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
