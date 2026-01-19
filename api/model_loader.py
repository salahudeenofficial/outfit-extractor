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
        # Signature: LightX2VPipeline(task, model_path, model_cls, ...)
        # For Qwen Image Edit, use 'i2i' (image-to-image) task
        # Supported tasks: t2v (text-to-video), i2v (image-to-video), t2i (text-to-image), i2i (image-to-image)
        task_options = ["i2i", "t2i", "image_edit", "image-edit"]
        
        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Using device: {device}")
        
        pipeline = None
        last_error = None
        
        # Try different task names - model_cls might be optional or auto-detected
        for task in task_options:
            try:
                # Pattern 1: Try without model_cls (might be optional)
                logger.info(f"Trying: LightX2VPipeline(task='{task}', model_path='{model_path}')")
                pipeline = LightX2VPipeline(task=task, model_path=model_path)
                logger.info(f"Successfully initialized pipeline with task='{task}'")
                break
            except TypeError as e:
                # If model_cls is required, try with different model_cls values
                if "model_cls" in str(e) or "required" in str(e).lower():
                    logger.debug(f"model_cls required, trying with model_cls parameter...")
                    # Try with model_cls as string
                    model_cls_options = [
                        "QwenImageEditPipeline",
                        "QwenImageEdit",
                        "QwenImageEdit2511",
                        "QwenImageEditLightning",
                    ]
                    
                    for model_cls in model_cls_options:
                        try:
                            logger.info(f"Trying: LightX2VPipeline(task='{task}', model_path='{model_path}', model_cls='{model_cls}')")
                            pipeline = LightX2VPipeline(task=task, model_path=model_path, model_cls=model_cls)
                            logger.info(f"Successfully initialized with task='{task}', model_cls='{model_cls}'")
                            break
                        except Exception as e2:
                            last_error = e2
                            logger.debug(f"Failed with model_cls='{model_cls}': {e2}")
                            continue
                    
                    if pipeline is not None:
                        break
                else:
                    last_error = e
                    logger.debug(f"Failed with task='{task}': {e}")
                    continue
            except Exception as e:
                last_error = e
                logger.debug(f"Failed with task='{task}': {type(e).__name__}: {e}")
                continue
        
        # If still None, try importing model class directly
        if pipeline is None:
            try:
                from lightx2v.models import QwenImageEditPipeline
                logger.info("Trying with imported QwenImageEditPipeline class...")
                pipeline = LightX2VPipeline(
                    task="i2i",  # Use i2i for image-to-image editing
                    model_path=model_path,
                    model_cls=QwenImageEditPipeline
                )
                logger.info("Successfully initialized with imported model class")
            except ImportError:
                logger.debug("Could not import QwenImageEditPipeline")
            except Exception as e:
                last_error = e
                logger.debug(f"Failed with imported model class: {e}")
        
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
