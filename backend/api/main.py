from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import json
import os
import io

# Load trained model
MODEL_PATH = "../model/Skin_Model.h5"
model = load_model(MODEL_PATH)

# Load class names from saved JSON 
CLASS_INDEX_PATH = "../model/class_indices.json"

if os.path.exists(CLASS_INDEX_PATH):
 with open(CLASS_INDEX_PATH) as f:
  class_indices = json.load(f)
  # Reverse mapping: index → class name
  CLASSES = [cls for cls, idx in sorted(class_indices.items(), key=lambda x: x[1])]
else:
 # fallback (hardcoded)
 CLASSES = ["Acne", "Hairloss", "Nail Fungus", "Normal", "Skin Allergy"]




# Define classes (same order as your training dataset folders)
CLASSES = ["Acne", "Hairloss", "Nail Fungus", "Normal", "Skin Allergy"]

app = FastAPI()

# Allow frontend (Flutter) to call API
app.add_middleware(
 CORSMiddleware,
 allow_origins=["*"],
 allow_credentials=True,
 allow_methods=["*"],
 allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
 contents = await file.read()
 image = Image.open(io.BytesIO(contents)).convert("RGB")
 image = image.resize((224, 224))
 image = img_to_array(image) / 255.0
 image = np.expand_dims(image, axis=0)
 preds = model.predict(image)[0]
 idx = np.argmax(preds)
 result = {
  "disease": CLASSES[idx],
  "confidence": float(preds[idx])
 }
 return result


if __name__ == "__main__":
 uvicorn.run(app, host="0.0.0.0", port=8000)
