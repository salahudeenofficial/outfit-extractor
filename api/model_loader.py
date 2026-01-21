"""Model loader for Qwen-Image-Edit-2511 FP8 quantized model using LightX2V framework.

Based on the working VTON project (try_og_pipeline).

Configuration: FP8 base model + FP8 4-step Lightning LoRA with 20 inference steps.
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


def find_fp8_base_weights_path() -> Optional[str]:
    """Find the FP8 base quantized weights file (without Lightning)."""
    possible_paths = [
        "/workspace/models/Qwen-Image-Edit-2511-Lightning/qwen_image_edit_2511_fp8_e4m3fn_scaled.safetensors",
        "/workspace/outfit-extractor/models/Qwen-Image-Edit-2511-Lightning/qwen_image_edit_2511_fp8_e4m3fn_scaled.safetensors",
        "models/Qwen-Image-Edit-2511-Lightning/qwen_image_edit_2511_fp8_e4m3fn_scaled.safetensors",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def find_lora_weights_path() -> Optional[str]:
    """Find the 4-step Lightning LoRA weights file (BF16 or FP32, not FP8 - FP8 LoRA doesn't exist separately)."""
    possible_paths = [
        # BF16 LoRA (preferred - smaller, good quality)
        "/workspace/models/Qwen-Image-Edit-2511-Lightning/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        "/workspace/outfit-extractor/models/Qwen-Image-Edit-2511-Lightning/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        "models/Qwen-Image-Edit-2511-Lightning/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        # FP32 LoRA (fallback - larger but higher precision)
        "/workspace/models/Qwen-Image-Edit-2511-Lightning/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors",
        "/workspace/outfit-extractor/models/Qwen-Image-Edit-2511-Lightning/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors",
        "models/Qwen-Image-Edit-2511-Lightning/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors",
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
):
    """
    Load Qwen-Image-Edit-2511 model using LightX2V framework.
    
    Configuration: FP8 base model + FP8 4-step Lightning LoRA with 20 inference steps.
    
    Args:
        model_path: Path to BASE model checkpoint (Qwen-Image-Edit-2511, not Lightning).
        device: Device to load model on ('cuda' or 'cpu').
        cache_dir: Directory to cache downloaded models.
    
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
    
    # Find FP8 base weights
    fp8_base_path = find_fp8_base_weights_path()
    if not fp8_base_path:
        raise RuntimeError(
            "FP8 base weights not found!\n"
            "Download with: huggingface-cli download lightx2v/Qwen-Image-Edit-2511-Lightning "
            "qwen_image_edit_2511_fp8_e4m3fn_scaled.safetensors "
            "--local-dir /workspace/models/Qwen-Image-Edit-2511-Lightning"
        )
    logger.info(f"FP8 base weights found: {fp8_base_path}")
    
    # Find 4-step Lightning LoRA weights (BF16 or FP32 - FP8 LoRA doesn't exist separately)
    lora_path = find_lora_weights_path()
    if not lora_path:
        raise RuntimeError(
            "4-step Lightning LoRA weights not found!\n"
            "Download with: huggingface-cli download lightx2v/Qwen-Image-Edit-2511-Lightning "
            "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors "
            "--local-dir /workspace/models/Qwen-Image-Edit-2511-Lightning"
        )
    logger.info(f"4-step Lightning LoRA weights found: {lora_path}")
    logger.info("Note: Using BF16/FP32 LoRA with FP8 base (FP8 LoRA doesn't exist separately)")
    
    # Use 20 inference steps as requested
    inference_steps = 20
    logger.info(f"Mode: FP8 scaled base model + BF16/FP32 4-step Lightning LoRA with {inference_steps} inference steps")
    
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
        
        # Enable CPU offload for memory management
        logger.info("Enabling CPU offload...")
        pipe.enable_offload()
        logger.info("CPU offload enabled")
        
        # Enable FP8 quantization with base FP8 weights
        logger.info(f"Enabling FP8 quantization with base weights: {fp8_base_path}")
        pipe.enable_quantize(
            dit_quantized=True,
            dit_quantized_ckpt=fp8_base_path,
            quant_scheme="fp8-sgl"
        )
        logger.info("FP8 base quantization enabled")
        
        # Load 4-step Lightning LoRA (BF16 or FP32)
        logger.info(f"Loading 4-step Lightning LoRA: {lora_path}")
        pipe.enable_lora([
            {"path": lora_path, "strength": 1.0},
        ])
        logger.info("4-step Lightning LoRA loaded")
        
        # Get attention mode
        attn_mode = get_attention_mode()
        
        # Create generator with 20 inference steps
        logger.info(f"Creating generator (steps={inference_steps}, attn_mode={attn_mode})...")
        pipe.create_generator(
            attn_mode=attn_mode,
            infer_steps=inference_steps,
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
        
        logger.info("Pipeline initialized successfully (FP8 scaled base + BF16/FP32 4-step Lightning LoRA)")
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
        "quantization": "FP8 scaled base + BF16/FP32 4-step Lightning LoRA",
        "framework": "LightX2V",
        "inference_steps": 20,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
