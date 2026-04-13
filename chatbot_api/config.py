import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Mock mode configuration
    USE_MOCK_MODE: bool = os.getenv("USE_MOCK_MODE", "False").lower() == "true"
    
    # API configuration
    LLM_API_URL: str = os.getenv("LLM_API_URL", "http://localhost:11434/api/generate")
    
    # Database configuration
    DB_PATH: str = "chatbot_api/cskh.db"

settings = Settings()
