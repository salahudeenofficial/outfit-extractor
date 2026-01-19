# LightX2V Installation Guide

## Issue
The LightX2V framework is not pre-installed in the `lightx2v/lightx2v:25101501-cu124` container and needs to be installed manually.

## Installation Methods

### Method 1: Install from GitHub (Recommended)
```bash
pip install git+https://github.com/ModelTC/LightX2V.git
```

### Method 2: Install from PyPI (if available)
```bash
pip install lightx2v
```

### Method 3: Manual Installation
If the above methods don't work, check the official LightX2V repository:
- GitHub: https://github.com/ModelTC/LightX2V
- Follow the installation instructions in their README

## After Installation

1. Verify installation:
   ```bash
   python3 -c "import lightx2v; print('LightX2V installed successfully')"
   ```

2. Start the API server:
   ```bash
   cd api
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

## Troubleshooting

- If installation fails, ensure you have git installed: `apt-get install git`
- Check Python version compatibility (requires Python 3.8+)
- Ensure you have sufficient disk space for the installation
- If you encounter permission errors, you may need to use `--user` flag or run as root

## Updated Setup Script

The `setup.sh` script has been updated to automatically attempt LightX2V installation if it's not found. You can run:

```bash
bash setup.sh
```

This will try to install LightX2V automatically.
