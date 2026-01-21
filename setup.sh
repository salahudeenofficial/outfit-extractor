#!/bin/bash
# Setup script for Outfit Extractor
# Designed to run inside lightx2v/lightx2v:25101501-cu124 container

set -e  # Exit on error

echo "=========================================="
echo "Outfit Extractor Setup"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if running inside container (optional check)
if [ -z "$CUDA_VISIBLE_DEVICES" ] && [ ! -f "/.dockerenv" ]; then
    echo -e "${YELLOW}Warning: This script is designed to run inside lightx2v/lightx2v:25101501-cu124 container${NC}"
fi

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${GREEN}Current directory: $SCRIPT_DIR${NC}"

# Check Python version
echo -e "\n${GREEN}Checking Python version...${NC}"
python3 --version || {
    echo -e "${RED}Python 3 is not installed${NC}"
    exit 1
}

# Check if pip is installed
echo -e "\n${GREEN}Checking pip...${NC}"
python3 -m pip --version || {
    echo -e "${RED}pip is not installed${NC}"
    exit 1
}

# Upgrade pip
echo -e "\n${GREEN}Upgrading pip...${NC}"
python3 -m pip install --upgrade pip

# Install API dependencies
echo -e "\n${GREEN}Installing API dependencies...${NC}"
cd api
if [ -f "requirements.txt" ]; then
    python3 -m pip install -r requirements.txt
    echo -e "${GREEN}API dependencies installed successfully${NC}"
else
    echo -e "${RED}requirements.txt not found in api/ directory${NC}"
    exit 1
fi
cd ..

# Check CUDA availability
echo -e "\n${GREEN}Checking CUDA availability...${NC}"
if python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>/dev/null; then
    python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"
else
    echo -e "${YELLOW}PyTorch not installed or CUDA check failed${NC}"
fi

# Reinstall LightX2V from pinned commit to avoid breaking changes
# IMPORTANT: The container's pre-installed LightX2V may have breaking changes
# Commit 7651b0f is the last known working version for i2i tasks
echo -e "\n${GREEN}Installing LightX2V (pinned to commit 7651b0f to avoid breaking changes)...${NC}"

# Remove existing LightX2V installation
echo -e "${YELLOW}Removing existing LightX2V installation...${NC}"
pip uninstall -y lightx2v 2>/dev/null || true

# Clone and install from pinned commit
LIGHTX2V_DIR="/workspace/LightX2V"
if [ -d "$LIGHTX2V_DIR" ]; then
    echo -e "${YELLOW}Removing existing LightX2V directory...${NC}"
    rm -rf "$LIGHTX2V_DIR"
fi

echo -e "${GREEN}Cloning LightX2V repository...${NC}"
git clone https://github.com/ModelTC/LightX2V.git "$LIGHTX2V_DIR"

echo -e "${GREEN}Checking out pinned commit 7651b0f...${NC}"
cd "$LIGHTX2V_DIR"
git checkout 7651b0f

echo -e "${GREEN}Installing LightX2V from pinned commit...${NC}"
pip install -v -e .

cd "$SCRIPT_DIR"

# Verify installation
if python3 -c "import lightx2v" 2>/dev/null; then
    echo -e "${GREEN}LightX2V framework installed successfully from pinned commit${NC}"
else
    echo -e "${RED}Failed to install LightX2V${NC}"
    exit 1
fi

# ============================================
# Download required models
# ============================================
# We need THREE things:
# 1. Base model (Qwen/Qwen-Image-Edit-2511) - contains configs, scheduler, text_encoder, vae, tokenizer
# 2. FP8 base weights (qwen_image_edit_2511_fp8_e4m3fn_scaled.safetensors) - FP8 quantized base model
# 3. FP8 4-step Lightning LoRA (qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_4steps_v1.0.safetensors) - FP8 LoRA
# ============================================

MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/workspace/models}"
mkdir -p "$MODEL_CACHE_DIR"

echo -e "\n${GREEN}=========================================="
echo "Downloading Required Models"
echo "==========================================${NC}"

# 1. Download base model (Qwen/Qwen-Image-Edit-2511)
BASE_MODEL_DIR="$MODEL_CACHE_DIR/Qwen-Image-Edit-2511"
if [ -d "$BASE_MODEL_DIR" ] && [ -f "$BASE_MODEL_DIR/scheduler/scheduler_config.json" ]; then
    echo -e "${GREEN}Base model already exists at: $BASE_MODEL_DIR${NC}"
else
    echo -e "\n${GREEN}Downloading base model (Qwen/Qwen-Image-Edit-2511)...${NC}"
    echo -e "${YELLOW}This contains configs, scheduler, text_encoder, vae, tokenizer (~40GB)${NC}"
    
    huggingface-cli download Qwen/Qwen-Image-Edit-2511 \
        --local-dir "$BASE_MODEL_DIR" \
        --local-dir-use-symlinks False
    
    echo -e "${GREEN}Base model downloaded successfully${NC}"
fi

# 2. Download FP8 base weights (without Lightning)
LIGHTNING_DIR="$MODEL_CACHE_DIR/Qwen-Image-Edit-2511-Lightning"
FP8_BASE_FILE="qwen_image_edit_2511_fp8_e4m3fn_scaled.safetensors"

mkdir -p "$LIGHTNING_DIR"

if [ -f "$LIGHTNING_DIR/$FP8_BASE_FILE" ]; then
    echo -e "${GREEN}FP8 base weights already exist at: $LIGHTNING_DIR/$FP8_BASE_FILE${NC}"
else
    echo -e "\n${GREEN}Downloading FP8 base weights...${NC}"
    echo -e "${YELLOW}Downloading FP8 base weights file (~20GB)${NC}"
    
    # Download FP8 base weights
    huggingface-cli download lightx2v/Qwen-Image-Edit-2511-Lightning \
        "$FP8_BASE_FILE" \
        --local-dir "$LIGHTNING_DIR" \
        --local-dir-use-symlinks False
    
    echo -e "${GREEN}FP8 base weights downloaded successfully${NC}"
fi

# 3. Download FP8 4-step Lightning LoRA weights
FP8_LORA_FILE="qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_4steps_v1.0.safetensors"

if [ -f "$LIGHTNING_DIR/$FP8_LORA_FILE" ]; then
    echo -e "${GREEN}FP8 4-step Lightning LoRA weights already exist at: $LIGHTNING_DIR/$FP8_LORA_FILE${NC}"
else
    echo -e "\n${GREEN}Downloading FP8 4-step Lightning LoRA weights...${NC}"
    echo -e "${YELLOW}Downloading FP8 Lightning LoRA weights file (~20GB)${NC}"
    
    # Download FP8 4-step Lightning LoRA weights
    huggingface-cli download lightx2v/Qwen-Image-Edit-2511-Lightning \
        "$FP8_LORA_FILE" \
        --local-dir "$LIGHTNING_DIR" \
        --local-dir-use-symlinks False
    
    echo -e "${GREEN}FP8 4-step Lightning LoRA weights downloaded successfully${NC}"
fi

# Verify downloads
echo -e "\n${GREEN}Verifying downloaded models...${NC}"

if [ -f "$BASE_MODEL_DIR/scheduler/scheduler_config.json" ]; then
    echo -e "${GREEN}✓ Base model scheduler config found${NC}"
else
    echo -e "${RED}✗ Base model scheduler config missing${NC}"
    exit 1
fi

if [ -f "$BASE_MODEL_DIR/transformer/config.json" ]; then
    echo -e "${GREEN}✓ Base model transformer config found${NC}"
else
    echo -e "${RED}✗ Base model transformer config missing${NC}"
    exit 1
fi

if [ -f "$LIGHTNING_DIR/$FP8_BASE_FILE" ]; then
    echo -e "${GREEN}✓ FP8 base weights file found${NC}"
else
    echo -e "${RED}✗ FP8 base weights file missing${NC}"
    exit 1
fi

if [ -f "$LIGHTNING_DIR/$FP8_LORA_FILE" ]; then
    echo -e "${GREEN}✓ FP8 4-step Lightning LoRA weights file found${NC}"
else
    echo -e "${RED}✗ FP8 4-step Lightning LoRA weights file missing${NC}"
    exit 1
fi

# Create necessary directories
echo -e "\n${GREEN}Creating necessary directories...${NC}"
mkdir -p logs

# Set permissions
chmod +x setup.sh

echo -e "\n${GREEN}=========================================="
echo "Setup completed successfully!"
echo "==========================================${NC}"
echo ""
echo "Models downloaded to: $MODEL_CACHE_DIR"
echo "  - Base model: $BASE_MODEL_DIR"
echo "  - FP8 base weights: $LIGHTNING_DIR/$FP8_BASE_FILE"
echo "  - FP8 4-step Lightning LoRA: $LIGHTNING_DIR/$FP8_LORA_FILE"
echo ""
echo "Next steps:"
echo "1. Navigate to the api directory: cd api"
echo "2. Start the FastAPI server:"
echo "   uvicorn main:app --host 0.0.0.0 --port 8000"
echo ""
echo "The server will be accessible at: http://0.0.0.0:8000"
echo ""
echo "To check if the server is running:"
echo "   curl http://localhost:8000/health"
echo ""
