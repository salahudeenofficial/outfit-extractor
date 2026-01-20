"""FastAPI server for outfit extraction using Qwen-Image-Edit-2511 with Diffusers."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import logging

from model_loader import load_model, get_model_info
from extractor import extract_outfit, OUTFIT_EXTRACTION_PROMPT
from schemas import HealthResponse, RootResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model pipeline
pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    global pipeline
    
    # Startup: Load model
    logger.info("Loading Qwen-Image-Edit-2511 model with Diffusers...")
    try:
        pipeline = load_model(device="cuda")
        logger.info("Model loaded successfully")
        logger.info(f"Model info: {get_model_info()}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}")
    
    yield
    
    # Shutdown: Cleanup
    logger.info("Shutting down...")
    pipeline = None


app = FastAPI(
    title="Outfit Extractor API",
    description="Extract outfits from images using Qwen-Image-Edit-2511 with Diffusers (full precision + 4-step Lightning LoRA)",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware to allow requests from web client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return HealthResponse(
        status="healthy",
        model_loaded=pipeline is not None,
        model_info=get_model_info()
    )


@app.get("/", response_model=RootResponse)
async def root() -> RootResponse:
    """Root endpoint with API information."""
    return RootResponse(
        name="Outfit Extractor API",
        version="1.0.0",
        endpoints={
            "health": "/health",
            "extract": "/extract"
        }
    )


@app.post("/extract")
async def extract_outfit_endpoint(
    file: UploadFile = File(...),
    num_inference_steps: int = Query(default=20, ge=1, le=50),
    guidance_scale: float = Query(default=1.0, ge=0.0, le=20.0)
) -> StreamingResponse:
    """
    Extract outfit from uploaded image.
    
    Args:
        file: Image file to process.
        num_inference_steps: Number of inference steps (default: 20).
        guidance_scale: Guidance scale (default: 1.0).
    
    Returns:
        Extracted outfit image as PNG stream.
    """
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image file."
        )
    
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    
    try:
        input_image = Image.open(io.BytesIO(contents))
    except Exception as e:
        logger.error(f"Failed to open image: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to open image: {str(e)}"
        )
    
    logger.info(f"Processing image: {file.filename}")
    try:
        result_image = extract_outfit(
            pipeline=pipeline,
            input_image=input_image,
            prompt=OUTFIT_EXTRACTION_PROMPT,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract outfit: {str(e)}"
        )
    
    output_buffer = io.BytesIO()
    result_image.save(output_buffer, format="PNG")
    output_buffer.seek(0)
    
    logger.info(f"Successfully processed image: {file.filename}")
    
    return StreamingResponse(
        output_buffer,
        media_type="image/png",
        headers={
            "Content-Disposition": f'attachment; filename="extracted_outfit.png"'
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
