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

# LLM imports
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

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

# 3. Load LLM
try:
    logger.info(f"⏳ Loading LLM: {settings.LLM_MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(settings.LLM_MODEL_NAME)
    llm = AutoModelForSeq2SeqLM.from_pretrained(settings.LLM_MODEL_NAME, torch_dtype=torch.float32)
    logger.info("✅ LLM loaded")
except Exception as e:
    logger.error(f"❌ Failed to load LLM: {e}")
    tok = None
    llm = None

SYSTEM_PROMPT = (
    "You are a helpful AI dermatology assistant. "
    "Only answer questions about skin, hair or nails. "
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

@app.post("/chat")
async def chat(user_msg: str):
    if llm is None or tok is None:
        raise HTTPException(status_code=503, detail="Chat service unavailable")

    # Basic Safety Filter (Keyword based for now, can be improved)
    unsafe_keywords = ["kill", "suicide", "bomb", "weapon"]
    if any(k in user_msg.lower() for k in unsafe_keywords):
        return {"reply": "I cannot answer that query. If you are in danger, please call emergency services."}

    prompt = f"{SYSTEM_PROMPT}\nPatient: {user_msg}\nDermatologist:"
    inputs = tok(prompt, return_tensors="pt")
    
    with torch.no_grad():
        out = llm.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.4,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tok.eos_token_id,
        )
    
    reply = tok.decode(out[0], skip_special_tokens=True)
    reply = reply.split("Dermatologist:")[-1].strip()
    
    return {"reply": reply}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)