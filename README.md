# Outfit Extractor

AI-powered outfit extraction tool using Qwen-Image-Edit-2511 FP8 quantized model via LightX2V framework. Extract full outfits from images with precise color preservation and white background.

## Features

- **Full Outfit Extraction**: Extracts complete outfits, not just single garments
- **Color Preservation**: Maintains original colors exactly as in input images
- **White Background**: Automatically places extracted outfits on pure white background
- **Fast Inference**: Uses FP8 quantized Lightning model (4 inference steps)
- **Web Interface**: User-friendly web client for easy image upload and viewing
- **RESTful API**: FastAPI-based server for programmatic access

## Architecture

```
┌─────────────────┐         HTTP POST         ┌──────────────────────────────┐
│   Web Client    │ ────────────────────────> │   Vast AI Instance           │
│   (app/)        │ <──────────────────────── │   ┌────────────────────────┐ │
│   Local Machine │      Image Response        │   │  lightx2v/lightx2v:    │ │
│                 │                            │   │  25101501-cu124        │ │
└─────────────────┘                            │   │  ┌──────────────────┐  │ │
                                               │   │  │ FastAPI Server   │  │ │
                                               │   │  │ (api/)            │  │ │
                                               │   │  └──────────────────┘  │ │
                                               │   │         │               │ │
                                               │   │         ▼               │ │
                                               │   │  ┌──────────────────┐  │ │
                                               │   │  │ LightX2V Model   │  │ │
                                               │   │  │ Qwen-Image-Edit  │  │ │
                                               │   │  │ 2511 FP8         │  │ │
                                               │   │  └──────────────────┘  │ │
                                               │   └────────────────────────┘ │
                                               └──────────────────────────────┘
```

## Project Structure

```
outfit-extractor/
├── api/                          # FastAPI server (runs in container)
│   ├── main.py                   # FastAPI app with /extract endpoint
│   ├── model_loader.py           # Load Qwen model via LightX2V
│   ├── extractor.py              # Outfit extraction logic with fixed prompt
│   └── requirements.txt          # Python dependencies
├── app/                          # Web client (runs locally)
│   ├── index.html                # Web interface
│   ├── client.js                 # Frontend JavaScript
│   ├── config.json               # Server IP configuration
│   └── README.md                 # Client usage instructions
├── setup.sh                      # Setup script (runs inside container)
└── README.md                     # This file
```

## Prerequisites

### For API Server (Vast AI Instance)

- Vast AI instance with GPU (recommended: 24GB+ VRAM)
- Docker container: `lightx2v/lightx2v:25101501-cu124`
- CUDA 12.4 support
- Internet connection for model download

### For Web Client (Local Machine)

- Modern web browser (Chrome, Firefox, Safari, Edge)
- No additional dependencies required

## Setup Instructions

### On Vast AI Instance (Inside Container)

1. **Start Vast AI Instance**
   - Select a GPU instance with sufficient VRAM (24GB+ recommended)
   - Use container: `lightx2v/lightx2v:25101501-cu124`
   - Note the instance's public IP address

2. **SSH into the Container**
   ```bash
   ssh root@<vast-ai-instance-ip>
   ```

3. **Clone the Repository**
   ```bash
   git clone <your-repo-url> outfit-extractor
   cd outfit-extractor
   ```

4. **Run Setup Script**
   ```bash
   chmod +x setup.sh
   bash setup.sh
   ```

5. **Start the FastAPI Server**
   ```bash
   cd api
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

6. **Verify Server is Running**
   ```bash
   curl http://localhost:8000/health
   ```

7. **Note the Public IP and Port**
   - The server will be accessible at: `http://<vast-ai-instance-ip>:8000`
   - Ensure port 8000 is exposed in Vast AI instance settings

### On Local Machine (Web Client)

1. **Clone the Repository**
   ```bash
   git clone <your-repo-url> outfit-extractor
   cd outfit-extractor/app
   ```

2. **Configure Server URL**
   - Open `config.json` and set `server_url` to your Vast AI instance IP
   - Or configure via the web interface after opening `index.html`

3. **Open Web Interface**
   - Simply open `index.html` in your web browser
   - Or use a local web server:
     ```bash
     python3 -m http.server 8080
     # Then open http://localhost:8080/index.html
     ```

## Usage

### Web Interface

1. Open `app/index.html` in your browser
2. Configure the server URL (if not already set)
3. Upload an image by clicking the upload area or dragging and dropping
4. Click "Extract Outfit" to process the image
5. View the extracted outfit in the result preview

### API Endpoints

#### Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_info": {
    "model_name": "Qwen-Image-Edit-2511-Lightning",
    "quantization": "FP8",
    "framework": "LightX2V",
    "inference_steps": 4,
    "device": "cuda"
  }
}
```

#### Extract Outfit
```bash
POST /extract
Content-Type: multipart/form-data

file: <image_file>
num_inference_steps: 4 (optional)
guidance_scale: 1.0 (optional)
```

Response: PNG image stream

Example using curl:
```bash
curl -X POST \
  http://<server-ip>:8000/extract \
  -F "file=@image.jpg" \
  --output extracted_outfit.png
```

## Model Details

- **Model**: `lightx2v/Qwen-Image-Edit-2511-Lightning` (HuggingFace)
- **Checkpoint**: FP8 quantized version (`qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning.safetensors`)
- **Framework**: LightX2V (required for proper FP8 support)
- **Inference Steps**: 4 (Lightning distillation)
- **Memory**: ~20GB VRAM (FP8 quantization reduces memory usage)
- **Prompt**: Fixed prompt ensures consistent outfit extraction with color preservation

## Fixed Extraction Prompt

The extraction uses a fixed prompt to ensure consistent results:

```
Extract the FULL OUTFIT, not a single garment. All worn clothing must be included.
STRICT: keep original colors exactly as in the input image. No color change, 
no hue shift, no saturation or brightness change. Preserve fabric texture and patterns.
No human present. Only the outfit on white background.
```

## Environment Variables

- `MODEL_PATH`: Path to model checkpoint or HuggingFace model ID (default: `lightx2v/Qwen-Image-Edit-2511-Lightning`)
- `MODEL_CACHE_DIR`: Directory to cache downloaded models (default: `/workspace/models`)
- `CUDA_VISIBLE_DEVICES`: GPU device IDs to use (optional)

## Troubleshooting

### Server Issues

- **LightX2V not found**: If LightX2V framework is not installed, run:
  ```bash
  pip install git+https://github.com/ModelTC/LightX2V.git
  ```
  Or try: `pip install lightx2v`
- **Model not loading**: Ensure LightX2V framework is installed and CUDA is available
- **CUDA errors**: Verify GPU is accessible and CUDA is properly configured
- **Out of memory**: Use a GPU with more VRAM or reduce image size
- **Connection refused**: Check that port 8000 is exposed and firewall allows connections

### Client Issues

- **CORS errors**: Ensure server has CORS enabled (already configured in the API)
- **Connection timeout**: Verify server IP and port are correct
- **Image not displaying**: Check browser console for errors

## Development

### API Development

The API uses FastAPI with async/await patterns. Key files:

- `api/main.py`: FastAPI application and endpoints
- `api/model_loader.py`: Model loading logic
- `api/extractor.py`: Extraction pipeline with preprocessing/postprocessing

### Client Development

The web client is vanilla JavaScript with no build step required. Key files:

- `app/index.html`: HTML structure and styles
- `app/client.js`: Client-side logic and API communication
- `app/config.json`: Default server configuration

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Acknowledgments

- LightX2V framework for efficient model inference
- Qwen team for the Image-Edit-2511 model
- HuggingFace for model hosting
