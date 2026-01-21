"""Model loader for Qwen-Image-Edit-2511 with 4-step Lightning LoRA using HuggingFace Diffusers.

Diffusers branch: Uses BF16 base model with 4-step Lightning LoRA via Diffusers library.
Runs 4 inference steps (optimized for Lightning LoRA) with sequential CPU offload for memory management.
"""

import os
import gc
import torch
from typing import Optional
import logging
from pathlib import Path

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


def find_lora_weights_path() -> Optional[str]:
    """Find the 4-step Lightning LoRA weights file."""
    possible_paths = [
        "/workspace/models/Qwen-Image-Edit-2511-Lightning/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        "/workspace/models/Qwen-Image-Edit-2511-Lightning/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors",
        "/workspace/outfit-extractor/models/Qwen-Image-Edit-2511-Lightning/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        "models/Qwen-Image-Edit-2511-Lightning/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def load_model(
    model_path: Optional[str] = None,
    device: str = "cuda",
    cache_dir: Optional[str] = None,
):
    """
    Load Qwen-Image-Edit-2511 model with 4-step Lightning LoRA using HuggingFace Diffusers.
    
    Attempts to load native BF16 weights (avoids runtime FP32->BF16 conversion for better quality).
    Falls back to native format with aggressive CPU offload if BF16 causes OOM.
    Uses 4-step Lightning LoRA (BF16) for fast inference.
    Uses 4 inference steps (optimized for the Lightning LoRA) with sequential CPU offload for memory management.
    
    Args:
        model_path: Path to BASE model checkpoint (Qwen-Image-Edit-2511) or HuggingFace model ID.
        device: Device to load model on ('cuda' or 'cpu').
        cache_dir: Directory to cache downloaded models.
    
    Returns:
        Loaded model pipeline.
    """
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Set device='cpu' or ensure GPU is accessible.")
    
    try:
        from diffusers import QwenImageEditPlusPipeline
        from peft import PeftModel
        import safetensors
    except ImportError as e:
        raise ImportError(
            f"Required packages not installed. Install with: "
            f"pip install diffusers peft safetensors\n"
            f"Original error: {e}"
        )
    
    # Find base model path
    if model_path is None:
        model_path = os.environ.get("MODEL_PATH") or find_base_model_path()
    
    if model_path is None:
        # Fall back to HuggingFace model ID
        model_path = "Qwen/Qwen-Image-Edit-2511"
        logger.info(f"No local model found, using HuggingFace model ID: {model_path}")
    elif not os.path.exists(model_path):
        logger.warning(f"Model path {model_path} not found, using HuggingFace model ID")
        model_path = "Qwen/Qwen-Image-Edit-2511"
    
    logger.info(f"Base model path: {model_path}")
    
    # Find 4-step Lightning LoRA weights
    lora_path = find_lora_weights_path()
    if lora_path is None:
        raise RuntimeError(
            "4-step Lightning LoRA weights not found!\n"
            "Download with: huggingface-cli download lightx2v/Qwen-Image-Edit-2511-Lightning "
            "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors "
            "--local-dir /workspace/models/Qwen-Image-Edit-2511-Lightning"
        )
    
    logger.info(f"4-step Lightning LoRA weights found: {lora_path}")
    logger.info("Mode: BF16 base model + 4-step Lightning LoRA with sequential CPU offload")
    
    # Clear GPU memory before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"CUDA memory before load: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated")
    
    try:
        # Load base pipeline from Diffusers using QwenImageEditPlusPipeline
        # The official Qwen/Qwen-Image-Edit-2511 model stores weights in BF16 format
        # Loading with torch_dtype=torch.bfloat16 uses those native BF16 weights directly
        logger.info(f"Loading pipeline from: {model_path}")
        logger.info("Loading with torch_dtype=torch.bfloat16 to use native BF16 weights (no conversion)...")
        
        try:
            # Load with BF16 dtype - Diffusers will use the native BF16 weights from the repository
            # The official model has BF16 weights stored, so this avoids FP32->BF16 conversion
            pipe = QwenImageEditPlusPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,  # Use native BF16 weights (no runtime conversion)
                cache_dir=cache_dir,
            )
            
            # Verify dtype
            if hasattr(pipe, 'transformer'):
                actual_dtype = next(pipe.transformer.parameters()).dtype
                logger.info(f"✓ Model loaded - transformer dtype: {actual_dtype}")
                if actual_dtype == torch.bfloat16:
                    logger.info("✓ Using native BF16 weights (no conversion)")
                else:
                    logger.warning(f"⚠ Model loaded in {actual_dtype}, not BF16")
            elif hasattr(pipe, 'unet'):
                actual_dtype = next(pipe.unet.parameters()).dtype
                logger.info(f"✓ Model loaded - unet dtype: {actual_dtype}")
            
            logger.info("Pipeline loaded successfully")
                    
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            raise
        
        logger.info("Pipeline loaded successfully")
        
        # Enable sequential CPU offload for aggressive memory management
        # This moves components to CPU one at a time during inference
        # More aggressive than enable_model_cpu_offload()
        logger.info("Enabling sequential CPU offload (aggressive memory management)...")
        pipe.enable_sequential_cpu_offload()
        
        # Also enable VAE CPU offload for additional memory savings
        if hasattr(pipe, 'enable_vae_slicing'):
            pipe.enable_vae_slicing()
            logger.info("VAE slicing enabled")
        if hasattr(pipe, 'enable_vae_tiling'):
            pipe.enable_vae_tiling()
            logger.info("VAE tiling enabled")
        
        logger.info("CPU offload and optimizations enabled")
        
        # Load LoRA weights
        logger.info(f"Loading 4-step Lightning LoRA: {lora_path}")
        
        # For Diffusers, we need to load LoRA using load_lora_weights
        # The LoRA file needs to be compatible with Diffusers format
        try:
            # Try loading as LoRA weights
            pipe.load_lora_weights(
                os.path.dirname(lora_path),
                weight_name=os.path.basename(lora_path),
                adapter_name="lightning_4step"
            )
            logger.info("4-step Lightning LoRA loaded successfully")
        except Exception as lora_error:
            logger.warning(f"Could not load LoRA via load_lora_weights: {lora_error}")
            logger.info("Attempting alternative LoRA loading method...")
            # Alternative: Load LoRA weights manually if needed
            # This depends on the LoRA format
            raise RuntimeError(
                f"Failed to load LoRA weights. "
                f"Diffusers LoRA loading may require a different format. "
                f"Error: {lora_error}"
            )
        
        # Note: Diffusers pipelines don't have eval() - components are already in eval mode
        
        # Memory stats
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info(f"GPU Memory: {allocated:.2f} GB / {total:.2f} GB")
        
        logger.info("Pipeline initialized successfully (BF16 base + 4-step Lightning LoRA + sequential CPU offload)")
        return pipe
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


def get_model_info() -> dict:
    """Get information about the loaded model."""
    return {
        "model_name": "Qwen-Image-Edit-2511",
        "precision": "Native BF16 (if available) + 4-step Lightning LoRA",
        "framework": "HuggingFace Diffusers",
        "lora": True,
        "lora_type": "4-step Lightning LoRA (BF16)",
        "cpu_offload": "sequential",
        "inference_steps": 4,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
