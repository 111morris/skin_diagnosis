from fastapi import HTTPException, Header, UploadFile
from typing import Optional
import logging
from config import settings
from PIL import Image
import io

# Setup Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("skin_diagnosis_api")

async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if x_api_key != settings.API_KEY:
        logger.warning(f"Unauthorized access attempt with key: {x_api_key}")
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key

async def validate_image(file: UploadFile) -> Image.Image:
    # Check file size (approximation as we read chunks, but for now read all)
    # In a real high-throughput scenario, we'd check content-length header first or read in chunks
    contents = await file.read()
    
    if len(contents) > settings.MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Max size is 5MB.")
    
    try:
        image = Image.open(io.BytesIO(contents))
        image.verify() # Verify it's an image
        image = Image.open(io.BytesIO(contents)) # Re-open after verify
        
        if image.format.lower() not in settings.ALLOWED_EXTENSIONS:
             raise HTTPException(status_code=400, detail=f"Invalid image format. Allowed: {settings.ALLOWED_EXTENSIONS}")
             
        return image.convert("RGB")
    except Exception as e:
        logger.error(f"Image validation failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file.")
