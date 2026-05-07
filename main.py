import os
import uuid
import shutil
import logging
import asyncio

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models.schemas import AnalysisResponse
from services.ui_detector import UIDetector
from services.code_generator import CodeGenerator
from utils.image_processor import ImageProcessor

# ==========================================
# LOGGING CONFIG
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# ==========================================
# FASTAPI APP
# ==========================================

app = FastAPI(
    title="Code From Design AI",
    version="1.0.0"
)

# ==========================================
# CORS CONFIG
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# INITIALIZE SERVICES
# ==========================================

image_processor = ImageProcessor()
detector = UIDetector()
generator = CodeGenerator()

# ==========================================
# TEMP DIRECTORY
# ==========================================

TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

# ==========================================
# ROOT ENDPOINT
# ==========================================

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Code From Design API Running",
        "engine": "HuggingFace LLaVA Free API"
    }

# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
async def health():
    return {
        "success": True,
        "server": "running"
    }

# ==========================================
# ANALYZE IMAGE ENDPOINT
# ==========================================

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):

    temp_path = None

    try:
        logger.info("New image upload received")

        # ==========================================
        # VALIDATE FILE
        # ==========================================

        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="No file uploaded"
            )

        allowed_extensions = [
            ".png",
            ".jpg",
            ".jpeg",
            ".webp"
        ]

        file_ext = os.path.splitext(
            file.filename
        )[1].lower()

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail="Unsupported image format"
            )

        # ==========================================
        # SAVE IMAGE
        # ==========================================

        temp_filename = f"{uuid.uuid4()}{file_ext}"

        temp_path = os.path.join(
            TEMP_DIR,
            temp_filename
        )

        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Image saved: {temp_path}")

        # ==========================================
        # LOAD IMAGE
        # ==========================================

        with open(temp_path, "rb") as f:
            contents = f.read()

        image_cv = image_processor.load_image_from_bytes(
            contents
        )

        # ==========================================
        # DETECT UI ELEMENTS
        # ==========================================

        logger.info("Detecting UI elements")

        elements = detector.detect_elements(image_cv)

        logger.info(
            f"Detected {len(elements)} UI elements"
        )

        # ==========================================
        # GENERATE CODE
        # ==========================================

        logger.info("Generating frontend code")

        # Run blocking AI task in thread
        generated_code = await asyncio.to_thread(
            generator.generate,
            temp_path,
            elements
        )

        # ==========================================
        # CHECK AI RESPONSE
        # ==========================================

        if generated_code.get("error"):
            logger.error(
                f"AI Generation Error: {generated_code['error']}"
            )

        # ==========================================
        # PREVIEW DATA
        # ==========================================

        preview_data = {
            "element_count": len(elements),
            "engine": "LLaVA Free API",
            "success": generated_code.get("error") is None
        }

        # ==========================================
        # CLEANUP
        # ==========================================

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        logger.info("Processing completed successfully")

        # ==========================================
        # RESPONSE
        # ==========================================

        return AnalysisResponse(
            valid=True,
            message="UI Analysis Completed Successfully",
            elements=elements,
            generated_code=generated_code,
            preview_data=preview_data
        )

    except HTTPException as http_error:

        logger.error(
            f"HTTP Error: {str(http_error.detail)}"
        )

        raise http_error

    except Exception as e:

        logger.error(f"Server Error: {str(e)}")

        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

        raise HTTPException(
            status_code=500,
            detail=f"Internal Server Error: {str(e)}"
        )

# ==========================================
# START SERVER
# ==========================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )