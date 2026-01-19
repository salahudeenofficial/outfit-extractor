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

# Check if model needs to be downloaded
echo -e "\n${GREEN}Checking model availability...${NC}"
MODEL_PATH="${MODEL_PATH:-lightx2v/Qwen-Image-Edit-2511-Lightning}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/workspace/models}"

echo "Model path: $MODEL_PATH"
echo "Cache directory: $MODEL_CACHE_DIR"

# Create cache directory if it doesn't exist
mkdir -p "$MODEL_CACHE_DIR"

# Check if huggingface-hub is available for model download
if python3 -c "import huggingface_hub" 2>/dev/null; then
    echo -e "\n${GREEN}huggingface-hub is available${NC}"
    echo -e "${YELLOW}Note: Model will be downloaded automatically on first use if not already cached${NC}"
    echo -e "${YELLOW}To manually download the model, run:${NC}"
    echo "  python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('$MODEL_PATH', cache_dir='$MODEL_CACHE_DIR')\""
else
    echo -e "\n${YELLOW}huggingface-hub not found. Model will be downloaded on first API call.${NC}"
fi

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

# Create necessary directories
echo -e "\n${GREEN}Creating necessary directories...${NC}"
mkdir -p "$MODEL_CACHE_DIR"
mkdir -p logs

# Set permissions
chmod +x setup.sh

echo -e "\n${GREEN}=========================================="
echo "Setup completed successfully!"
echo "==========================================${NC}"
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
echo "Note: The model will be downloaded automatically on first API call"
echo "      if it's not already cached in: $MODEL_CACHE_DIR"
echo ""
