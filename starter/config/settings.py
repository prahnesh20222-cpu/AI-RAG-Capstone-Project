"""
Configuration Settings.
This module uses Pydantic Settings to load and validate environment variables from the .env file.
It centralizes all configuration management, ensuring type safety for API keys,
database paths, and other system-wide constants.
"""
import os
from pydantic_settings import BaseSettings, SettingsConfigDict

# Get the absolute path to the root of the project (going up one level from config folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE_PATH = os.path.join(BASE_DIR, ".env")

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE_PATH, 
                                      env_file_encoding="utf-8",
                                        extra="ignore")

    OPENAI_API_KEY: str = ""
    OPENAI_API_BASE_URL: str = ""
    OPENAI_API_EMBEDDING_MODEL: str = ""
    OPENAI_LLM_MODEL: str = ""
    OPENAI_TEMPERATURE: float = 0.0
    #DB_PATH: str = "data/db/eval_runs.db"
    #DEFAULT_EMBEDDING_MODEL: str = "text-embedding-ada-002"
    TARGET_CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    VECTOR_STORE_TYPE: str = ""
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = ""
    TOP_K: int = 5
    

settings = Settings()
#print(settings.model_dump())