from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    # Security
    API_KEY = os.getenv("API_KEY")
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

    # Model Paths
    MODEL_PATH = "../model/Skin_Model.h5"
    CLASS_INDEX_PATH = "../model/labels.json"
    
    # LLM Settings
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    LLM_MODEL_NAME = "gemini-2.0-flash-lite"
    
    # Validation
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB``
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
settings = Settings()
