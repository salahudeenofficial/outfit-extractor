"""Model loader for Qwen-Image-Edit-2511 FP8 quantized model using LightX2V framework.

Based on the working VTON project (try_og_pipeline).

Key insight: The FP8 Lightning model is NOT a standalone model - it only contains
quantized transformer weights. You need BOTH:
1. Base model (Qwen-Image-Edit-2511) - for configs, scheduler, text_encoder, vae, tokenizer
2. FP8 weights file (qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_4steps_v1.0.safetensors)
"""

import os
import gc
import torch
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def find_base_model_path() -> Optional[str]:
    """Find the base Qwen-Image-Edit-2511 model path."""
    possible_paths = [
        "/workspace/models/Qwen-Image-Edit-2511",
        "/workspace/outfit-extractor/models/Qwen-Image-Edit-2511",
        "models/Qwen-Image-Edit-2511",
        os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen-Image-Edit-2511"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            # For HuggingFace cache, need to find the actual snapshot
            if "models--Qwen--Qwen-Image-Edit-2511" in path:
                snapshots_dir = os.path.join(path, "snapshots")
                if os.path.exists(snapshots_dir):
                    snapshots = os.listdir(snapshots_dir)
                    if snapshots:
                        return os.path.join(snapshots_dir, snapshots[0])
            else:
                return path
    
    return None


def find_fp8_weights_path() -> Optional[str]:
    """Find the FP8 quantized weights file."""
    possible_paths = [
        # 4-step lightning version (preferred)
        "/workspace/models/Qwen-Image-Edit-2511-Lightning/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_4steps_v1.0.safetensors",
        "/workspace/outfit-extractor/models/Qwen-Image-Edit-2511-Lightning/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_4steps_v1.0.safetensors",
        "models/Qwen-Image-Edit-2511-Lightning/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_4steps_v1.0.safetensors",
        # Alternative paths
        "/workspace/models/Qwen-Image-Edit-2511-Lightning/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning.safetensors",
        "/workspace/models/Qwen-Image-Edit-2511-Lightning/qwen_image_edit_2511_fp8_e4m3fn_scaled.safetensors",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def find_lora_weights_path() -> Optional[str]:
    """Find the LoRA weights file (alternative to FP8)."""
    possible_paths = [
        "/workspace/models/Qwen-Image-Edit-2511-Lightning/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        "/workspace/models/Qwen-Image-Edit-2511-Lightning/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors",
        "models/Qwen-Image-Edit-2511-Lightning/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def get_attention_mode() -> str:
    """Determine the best attention mode for the current GPU."""
    attn_mode = "torch_sdpa"  # Safe default
    
    if not torch.cuda.is_available():
        return attn_mode
    
    gpu_name = torch.cuda.get_device_name(0).lower()
    
    try:
        import importlib.util
        if importlib.util.find_spec("flash_attn") is not None:
            if "l40" in gpu_name or "4090" in gpu_name or "4080" in gpu_name or "a100" in gpu_name:
                attn_mode = "flash_attn2"
                logger.info(f"GPU detected: {gpu_name} → Using Flash Attention 2")
            elif "h100" in gpu_name or "h200" in gpu_name:
                try:
                    from flash_attn_interface import flash_attn_func
                    attn_mode = "flash_attn3"
                    logger.info(f"Hopper GPU detected → Using Flash Attention 3")
                except ImportError:
                    attn_mode = "flash_attn2"
                    logger.info(f"Hopper GPU but FA3 not installed → Using Flash Attention 2")
            else:
                attn_mode = "flash_attn2"
                logger.info(f"Using Flash Attention 2")
    except Exception as e:
        logger.info(f"Flash Attention unavailable ({type(e).__name__}), using PyTorch SDPA")
    
    return attn_mode


def load_model(
    model_path: Optional[str] = None,
    device: str = "cuda",
    cache_dir: Optional[str] = None,
    mode: str = "fp8",  # "fp8", "lora", or "base"
):
    """
    Load Qwen-Image-Edit-2511 model using LightX2V framework.
    
    Args:
        model_path: Path to BASE model checkpoint (Qwen-Image-Edit-2511, not Lightning).
        device: Device to load model on ('cuda' or 'cpu').
        cache_dir: Directory to cache downloaded models.
        mode: "fp8" for FP8 quantized, "lora" for LoRA, "base" for full precision.
    
    Returns:
        Loaded model pipeline.
    """
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Set device='cpu' or ensure GPU is accessible.")
    
    # Find base model path (REQUIRED - contains all configs)
    if model_path is None:
        model_path = os.environ.get("MODEL_PATH") or find_base_model_path()
    
    if model_path is None or not os.path.exists(model_path):
        raise RuntimeError(
            "Base model (Qwen-Image-Edit-2511) not found!\n"
            "Download with: huggingface-cli download Qwen/Qwen-Image-Edit-2511 --local-dir /workspace/models/Qwen-Image-Edit-2511"
        )
    
    logger.info(f"Base model path: {model_path}")
    
    # Find FP8 weights (if using FP8 mode)
    fp8_path = None
    lora_path = None
    steps = 40  # Default for base model
    
    if mode == "fp8":
        fp8_path = find_fp8_weights_path()
        if fp8_path:
            logger.info(f"FP8 weights found: {fp8_path}")
            steps = 20  # 20 steps for better quality
        else:
            logger.warning("FP8 weights not found, falling back to LoRA mode")
            mode = "lora"
    
    if mode == "lora":
        lora_path = find_lora_weights_path()
        if lora_path:
            logger.info(f"LoRA weights found: {lora_path}")
            steps = 20  # 20 steps for better quality
        else:
            logger.warning("LoRA weights not found, falling back to base mode")
            mode = "base"
    
    if mode == "base":
        logger.warning("Using base model (40 steps) - this will be SLOW")
        steps = 40
    
    # Clear GPU memory before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"CUDA memory before load: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated")
    
    try:
        from lightx2v import LightX2VPipeline
        
        logger.info(f"Initializing LightX2VPipeline(model_path='{model_path}', model_cls='qwen-image-edit-2511', task='i2i')")
        
        # Initialize pipeline with BASE model path
        pipe = LightX2VPipeline(
            model_path=model_path,
            model_cls="qwen-image-edit-2511",
            task="i2i",
        )
        logger.info("Pipeline created successfully")
        
        # Configure based on mode
        if mode == "fp8" and fp8_path:
            logger.info(f"Enabling FP8 quantization with: {fp8_path}")
            pipe.enable_quantize(
                dit_quantized=True,
                dit_quantized_ckpt=fp8_path,
                quant_scheme="fp8-sgl"
            )
            logger.info("FP8 quantization enabled")
        
        if mode == "lora" and lora_path:
            logger.info(f"Loading 4-step Lightning LoRA: {lora_path}")
            pipe.enable_lora([
                {"path": lora_path, "strength": 1.0},
            ])
            logger.info("LoRA loaded")
        
        # Get attention mode
        attn_mode = get_attention_mode()
        
        # Create generator ONCE during startup
        # (From PROBLEMS_FACED.txt: calling create_generator() multiple times causes JSON serialization errors)
        logger.info(f"Creating generator (steps={steps}, attn_mode={attn_mode})...")
        pipe.create_generator(
            attn_mode=attn_mode,
            infer_steps=steps,
            guidance_scale=1.0,
            width=768,
            height=1024,
            aspect_ratio="3:4",
        )
        
        # Verify runner was created
        if hasattr(pipe, 'runner') and pipe.runner is not None:
            logger.info("Generator created successfully - runner initialized")
        else:
            raise RuntimeError("Pipeline runner not initialized after create_generator()")
        
        # Memory stats
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info(f"GPU Memory: {allocated:.2f} GB / {total:.2f} GB")
        
        logger.info("Pipeline initialized successfully")
        return pipe
        
    except ImportError as e:
        raise ImportError(
            f"Failed to import LightX2V framework. "
            f"Please install it: pip install lightx2v\n"
            f"Original error: {e}"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


def get_model_info() -> dict:
    """Get information about the loaded model."""
    return {
        "model_name": "Qwen-Image-Edit-2511",
        "quantization": "FP8",
        "framework": "LightX2V",
        "inference_steps": 4,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
