from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import json
import os
import logging
from typing import List
from pydantic import BaseModel

# LLM imports
import google.generativeai as genai

# Local imports
from config import settings
from utils import verify_api_key, validate_image, logger

app = FastAPI(title="Skin Diagnosis API", version="2.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error. Please contact support."},
    )

# Load Resources
logger.info("Loading resources...")

# 1. Load Skin Model
try:
    model = load_model(settings.MODEL_PATH)
    logger.info("✅ Skin Model loaded")
except Exception as e:
    logger.error(f"❌ Failed to load Skin Model: {e}")
    model = None

# 2. Load Labels
try:
    if os.path.exists(settings.CLASS_INDEX_PATH):
        with open(settings.CLASS_INDEX_PATH, "r") as f:
            CLASSES = json.load(f)
            # If it's a dict (old format), convert to list, else assume list
            if isinstance(CLASSES, dict):
                 CLASSES = [cls for cls, idx in sorted(CLASSES.items(), key=lambda x: x[1])]
    else:
        logger.warning("⚠️ Labels file not found, using default.")
        CLASSES = ["Acne", "Hairloss", "Nail Fungus", "Normal", "Skin Allergy"]
    logger.info(f"✅ Classes loaded: {CLASSES}")
except Exception as e:
    logger.error(f"❌ Failed to load classes: {e}")
    CLASSES = ["Acne", "Hairloss", "Nail Fungus", "Normal", "Skin Allergy"]

# 3. Configure Gemini
try:
    if settings.GEMINI_API_KEY:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        llm_model = genai.GenerativeModel(settings.LLM_MODEL_NAME)
        logger.info(f"✅ Gemini configured with model: {settings.LLM_MODEL_NAME}")
    else:
        logger.warning("⚠️ GEMINI_API_KEY not found. Chat features will be disabled.")
        llm_model = None
except Exception as e:
    logger.error(f"❌ Failed to configure Gemini: {e}")
    llm_model = None

SYSTEM_PROMPT = (
    "You are a helpful AI dermatology assistant. "
    "Only answer questions about skin, hair or nails. "
    "Keep your answers concise and helpful. "
    "If the question is off-topic, politely redirect."
)

@app.get("/")
def root():
    return {"message": "Skin Diagnosis API v2.0 is running!", "docs": "/docs"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    image = await validate_image(file)
    
    # Preprocess
    image = image.resize((224, 224))
    image_arr = img_to_array(image) / 255.0
    image_arr = np.expand_dims(image_arr, axis=0)
    
    # Predict
    preds = model.predict(image_arr)[0]
    idx = np.argmax(preds)
    confidence = float(preds[idx])
    
    # Responsible AI: Uncertainty / Low Confidence Check
    # If confidence is low (e.g. < 0.6), we might want to flag it
    warning = None
    if confidence < 0.6:
        warning = "Low confidence prediction. Please consult a specialist."
        
    result = {
        "disease": CLASSES[idx],
        "confidence": confidence,
        "warning": warning
    }

    logger.info(f"Prediction: {result}")
    return result

class ChatIn(BaseModel):
    user_msg: str


@app.post("/chat", dependencies=[Depends(verify_api_key)])
async def chat(payload: ChatIn):
    if llm_model is None:
        if not settings.GEMINI_API_KEY:
             raise HTTPException(status_code=503, detail="Chat service unavailable. Missing GEMINI_API_KEY.")
        raise HTTPException(status_code=503, detail="Chat service unavailable")

    # safety filter (unchanged)
    unsafe = ["kill", "suicide", "bomb", "weapon"]
    if any(k in payload.user_msg.lower() for k in unsafe):
        return {"reply": "I cannot answer that query. If you are in danger, please call emergency services."}

    try:
        chat_session = llm_model.start_chat(
            history=[
                {"role": "user", "parts": [SYSTEM_PROMPT]},
                {"role": "model", "parts": ["Understood. I am ready to assist with skin, hair, and nail questions."]},
            ]
        )
        response = chat_session.send_message(payload.user_msg)
        reply = response.text.strip()
    except Exception as e:
        logger.error(f"Gemini generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating response.\nGemini Error:{e}")

    return {"reply": reply}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)