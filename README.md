# Skin Diagnosis App

A machine learning–powered app that helps users diagnose common skin conditions from images.
Built with **FastAPI (backend)** + **Flutter (frontend)**.

---

## Features

- **Image Diagnosis**: Upload or capture an image to predict skin conditions (Acne, Hairloss, Nail Fungus, Normal, Skin Allergy).
- **AI Chatbot**: Ask questions about skin, hair, and nails using the integrated Google Gemini AI.
- **Responsible AI**: Includes a low-confidence warning system for uncertain predictions.
- **Cross-Platform**: Mobile app built with Flutter for Android (and iOS ready).

---

## Tech Stack

- **Backend**: FastAPI, TensorFlow/Keras, Uvicorn, Google Gemini AI
- **Frontend**: Flutter, Dart, http, image_picker
- **Deployment**: Render (Backend), Android (.apk)

---


## Project Structure

```
skin_diagnosis/
├── backend/
│   ├── api/
│   │   ├── main.py           # FastAPI application & endpoints
│   │   ├── config.py         # Configuration settings
│   │   ├── utils.py          # Helper functions
│   │   └── requirements.txt  # Backend dependencies
│   └── model/
│       ├── train.py          # Training script
│       ├── Skin_Model.h5     # Trained model
│       └── labels.json       # Class names
├── frontend/
│   └── hope/                 # Flutter application source code
├── dataset/                  # Training dataset (not in git)
└── README.md
```

---

## Setup & Run Guide

### 1. Backend Setup

**Prerequisites**: Python 3.9+

1.  Navigate to the backend API directory:

    ```bash
    cd backend/api
    ```

2.  Create and activate a virtual environment:

    ```bash
    python3 -m venv venv
    source venv/bin/activate        # Windows: venv\Scripts\activate
    ```

3.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Configuration**:

    - Create a `.env` file in `backend/api/` (optional, or set env vars).
    - **Required for Chat**: Set `GEMINI_API_KEY=your_api_key_here` to enable the AI chatbot.

5.  Run the Server:
    ```bash
    python main.py
    # OR
    uvicorn main:app --reload
    ```
    The API will be available at `http://localhost:8000`.

### 2. Frontend Setup (Flutter)

**Prerequisites**: Flutter SDK

1.  Navigate to the Flutter app directory:

    ```bash
    cd frontend/hope
    ```

2.  Install dependencies:

    ```bash
    flutter pub get
    ```

3.  Run the app:
    ```bash
    flutter run
    ```

---

## API Endpoints

### `GET /`

Health check. Returns API status.

### `POST /predict`

Upload an image to get a diagnosis.

- **Body**: `multipart/form-data` with `file` field.
- **Response**:
  ```json
  {
    "disease": "Acne",
    "confidence": 0.93,
    "warning": null
  }
  ```

### `POST /chat`

Chat with the AI dermatologist.

- **Headers**: `x-api-key` (if configured)
- **Body**:
  ```json
  {
    "user_msg": "What is the best treatment for dry skin?"
  }
  ```
- **Response**:
  ```json
  {
    "reply": "For dry skin, it is recommended to..."
  }
  ```

---

## Model Training

To retrain the model, navigate to `backend/model/` and run:

```bash
python train.py --dataset ../../dataset/train --model Skin_Model.h5 --plot plot.png
```

---

## Roadmap

- [ ] Add more skin conditions
- [ ] Improve model accuracy
- [ ] iOS support
- [ ] Secure user authentication
