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
        
        # LightX2V requires: task, model_path, model_cls
        # Based on working VTON project:
        # - task: "i2i" (image-to-image)
        # - model_cls: "qwen-image-edit-2511" (lowercase with dashes!)
        task_options = ["i2i"]
        model_cls_options = ["qwen-image-edit-2511", "qwen_image_edit", "QwenImageEditPipeline"]
        
        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Using device: {device}")
        
        pipeline = None
        last_error = None
        
        # Try task/model_cls combinations based on working VTON project
        for task in task_options:
            for model_cls in model_cls_options:
                try:
                    logger.info(f"Trying: LightX2VPipeline(model_path='{model_path}', model_cls='{model_cls}', task='{task}')")
                    pipeline = LightX2VPipeline(
                        model_path=model_path,
                        model_cls=model_cls,
                        task=task,
                    )
                    logger.info(f"Successfully initialized with model_cls='{model_cls}', task='{task}'")
                    
                    # Create generator ONCE during startup (from PROBLEMS_FACED.txt)
                    # Calling create_generator() multiple times causes JSON serialization errors
                    logger.info("Creating generator (one-time setup)...")
                    pipeline.create_generator()
                    logger.info("Generator created successfully")
                    
                    break
                except Exception as e:
                    last_error = e
                    logger.debug(f"Failed with model_cls='{model_cls}', task='{task}': {e}")
                    continue
            
            if pipeline is not None:
                break
        
        # If still None, try without model_cls (auto-detect)
        if pipeline is None:
            try:
                logger.info(f"Trying: LightX2VPipeline(model_path='{model_path}', task='i2i') without model_cls")
                pipeline = LightX2VPipeline(
                    model_path=model_path,
                    task="i2i",
                )
                logger.info("Successfully initialized without model_cls")
                
                # Create generator
                logger.info("Creating generator...")
                pipeline.create_generator()
                logger.info("Generator created successfully")
            except Exception as e:
                last_error = e
                logger.debug(f"Failed without model_cls: {e}")
        
        if pipeline is None:
            error_msg = (
                f"Failed to initialize LightX2VPipeline with model_path='{model_path}'. "
                f"LightX2V requires 'task' and 'model_cls' parameters. "
                f"Tried tasks: {task_options}. "
                f"Last error: {last_error}. "
                f"\n\nLightX2VPipeline signature: (task, model_path, model_cls, ...) "
                f"\nPlease check LightX2V documentation for Qwen Image Edit task and model_cls values."
            )
            raise RuntimeError(error_msg)
        
        # Device is handled automatically by LightX2V based on CUDA availability
        logger.info(f"Pipeline initialized successfully")
        
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
