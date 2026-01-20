#!/bin/bash
# Setup script for Outfit Extractor (BF16 branch)
# Uses BF16 model with CPU offload - no FP8 quantization
# Designed to run inside lightx2v/lightx2v:25101501-cu124 container

set -e  # Exit on error

echo "=========================================="
echo "Outfit Extractor Setup (BF16 Branch)"
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
# Download required models (BF16 branch)
# ============================================
# For BF16 branch we need:
# 1. Base model (Qwen/Qwen-Image-Edit-2511) - full BF16 model
# 2. LoRA weights (optional, for faster 4-step inference)
# NO FP8 weights needed for this branch
# ============================================

MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/workspace/models}"
mkdir -p "$MODEL_CACHE_DIR"

echo -e "\n${GREEN}=========================================="
echo "Downloading Required Models (BF16 Branch)"
echo "==========================================${NC}"

# 1. Download base model (Qwen/Qwen-Image-Edit-2511)
BASE_MODEL_DIR="$MODEL_CACHE_DIR/Qwen-Image-Edit-2511"
if [ -d "$BASE_MODEL_DIR" ] && [ -f "$BASE_MODEL_DIR/scheduler/scheduler_config.json" ]; then
    echo -e "${GREEN}Base model already exists at: $BASE_MODEL_DIR${NC}"
else
    echo -e "\n${GREEN}Downloading base model (Qwen/Qwen-Image-Edit-2511)...${NC}"
    echo -e "${YELLOW}This is the full BF16 model (~40GB)${NC}"
    
    huggingface-cli download Qwen/Qwen-Image-Edit-2511 \
        --local-dir "$BASE_MODEL_DIR" \
        --local-dir-use-symlinks False
    
    echo -e "${GREEN}Base model downloaded successfully${NC}"
fi

# 2. Download LoRA weights (BF16 version for 4-step inference)
LIGHTNING_DIR="$MODEL_CACHE_DIR/Qwen-Image-Edit-2511-Lightning"
LORA_WEIGHTS_FILE="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"

mkdir -p "$LIGHTNING_DIR"

if [ -f "$LIGHTNING_DIR/$LORA_WEIGHTS_FILE" ]; then
    echo -e "${GREEN}LoRA weights already exist at: $LIGHTNING_DIR/$LORA_WEIGHTS_FILE${NC}"
else
    echo -e "\n${GREEN}Downloading BF16 LoRA weights for 4-step inference...${NC}"
    echo -e "${YELLOW}Downloading only the BF16 LoRA weights file (~800MB)${NC}"
    
    # Download only the BF16 LoRA weights file
    huggingface-cli download lightx2v/Qwen-Image-Edit-2511-Lightning \
        "$LORA_WEIGHTS_FILE" \
        --local-dir "$LIGHTNING_DIR" \
        --local-dir-use-symlinks False
    
    echo -e "${GREEN}LoRA weights downloaded successfully${NC}"
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

if [ -f "$LIGHTNING_DIR/$LORA_WEIGHTS_FILE" ]; then
    echo -e "${GREEN}✓ BF16 LoRA weights file found${NC}"
else
    echo -e "${YELLOW}⚠ BF16 LoRA weights not found - will use base model (slower)${NC}"
fi

# Create necessary directories
echo -e "\n${GREEN}Creating necessary directories...${NC}"
mkdir -p logs

# Set permissions
chmod +x setup.sh

echo -e "\n${GREEN}=========================================="
echo "Setup completed successfully! (BF16 Branch)"
echo "==========================================${NC}"
echo ""
echo "Models downloaded to: $MODEL_CACHE_DIR"
echo "  - Base model (BF16): $BASE_MODEL_DIR"
echo "  - LoRA weights: $LIGHTNING_DIR/$LORA_WEIGHTS_FILE"
echo ""
echo "This branch uses BF16 with CPU offload (no FP8 quantization)"
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
