# Skin Diagnosis App 

A machine learning–powered app that helps users diagnose common skin conditions from images.  
Built with **FastAPI (backend)** + **Flutter (frontend)**.

---

## Features
-  Upload or capture an image of a skin condition
-  AI model predicts condition (Acne, Hairloss, Nail Fungus, Normal, Skin Allergy)
-  Confidence score shown with prediction
-  Deployed backend with REST API (FastAPI + Render)
-  Cross-platform mobile app (Flutter)

---

## Tech Stack
- **Backend**: FastAPI, TensorFlow/Keras, Uvicorn  
- **Frontend**: Flutter, Dart, http, image_picker  
- **Deployment**: Render (Backend), Android (.apk for distribution)  

---


# Skin Diagnosis App – Setup & Run Guide

A step‑by‑step guide to train the model, run the API, and connect the frontend.


## 0) Prerequisites

* Python 3.9+ (3.12 works)
* Git (optional)

## 0.1) Cloning the project
```bash
git clone https://github.com/111morris/skin_diagnosis.git
```

change the directory to skin_diagnosis 

```bash
cd skin_diagnosis
```

activate the vertual environment

```bash
# from project root
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt # or install packages listed below
```

**Minimal requirements (if you don’t have a file yet):**

```
fastapi
uvicorn
numpy
pillow
tensorflow
scikit-learn
imutils
matplotlib
python-multipart
```

---


## 1) Project Structure 

```
skin_diagnosis/
├── backend/
│   ├── api/
│   │   └── main.py
│   └── model/
│       ├── train.py
│       ├── Skin_Model.h5         # ← created after training
│       └── labels.json           # ← class names saved during training
├── dataset/                      # not committed to git
│   ├── train/
│   │   ├── Acne/
│   │   ├── Hairloss/
│   │   ├── Nail Fungus/
│   │   ├── Normal/
│   │   └── Skin Allergy/
│   └── test/...
├── frontend/                     # Flutter app (or other)
├── requirements.txt
└── README.md
```

---

## 2) How to train the Model

> You can run training from **project root** or from **`backend/model/`**. Ensure the dataset path is correct for your working directory.

### Option A: Run from **project root**

```bash
python backend/model/train.py \
  --dataset dataset/train \
  --model backend/model/Skin_Model.h5 \
  --plot backend/model/plot.png
```

### Option B: Run from **backend/model/**

```bash
cd backend/model
python train.py --dataset ../../dataset/train --model Skin_Model.h5 --plot plot.png
```

### What you should expect

* Console prints: `[INFO] loading images...`, training progress per epoch
* A saved model file: `backend/model/Skin_Model.h5`
* A training plot: `plot.png`
* A classification report printed in terminal

### Save your class names (IMPORTANT)

Add these lines **in ************`train.py`************ right after** you one‑hot encode labels with `LabelBinarizer` (or after you have `lb.classes_`):

```python
# save class list next to the model
import json, os
labels_path = os.path.join(os.path.dirname(__file__), "labels.json")
with open(labels_path, "w") as f:
    json.dump(lb.classes_.tolist(), f)
```

This produces `backend/model/labels.json` like:

```json
["Acne", "Hairloss", "Nail Fungus", "Normal", "Skin Allergy"]
```

[//]: # (> The API will load this file to map prediction indices → class names.)

---

## 3) Scan the Image

Using `backend/model/ScanImage.py` inside the model folder:

Run:
```bash
 python ScanImage.py --image "[input_image_name.png]"
```

You should see something like this or familiar

```
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 877ms/step
[D] Skin Allergy
[P] 74.75%
```


## Frontend (Flutter App)

1. Navigate to the frontend:
```bash
cd skin_diagnosis/frontend
```

2. Get dependencies:
```bash
flutter pub get
```

3. Run the app:

```bash
flutter run
```

## API Endpoints
*GET /* Head check 
*POST /predict* Upload image and get prediction 
- Request: form-data with file (image)

- Response:

```bash
{
  "disease": "Acne",
  "confidence": 0.93
}
```

## APK Installation

Download the **.apk** file from 📱 [Releases_v1]((https://drive.google.com/file/d/1V6z062Dfc2t0664eJVdtxdRREbK0seUk/view?usp=drive_link))


Install on your Android device (allow unknown sources if needed)


## Roadmap

- Add more skin conditions
- Improve model accuracy
- iOS support
- Secure user authentication
