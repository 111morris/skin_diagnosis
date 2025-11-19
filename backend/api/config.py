import os

class Settings:
    # Security
    API_KEY = os.getenv("API_KEY", "1234567890") 
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

    # Model Paths
    MODEL_PATH = "../model/Skin_Model.h5"
    CLASS_INDEX_PATH = "../model/labels.json"
    
    # LLM Settings
    LLM_MODEL_NAME = "google/flan-t5-small"
    
    # Validation
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB``
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
settings = Settings()
