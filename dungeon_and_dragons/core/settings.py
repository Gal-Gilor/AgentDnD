from google.cloud.exceptions import NotFound
from google.cloud.storage import Bucket, Client
from pydantic import Field, PrivateAttr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Required environment and configuration settings."""

    GOOGLE_API_KEY: str = Field(description="Google API Key")
    GOOGLE_PROJECT_ID: str = Field(description="Google Cloud Project ID")
    GOOGLE_APPLICATION_CREDENTIALS: str = Field(
        description="Google Cloud Application Credentials"
    )
    GOOGLE_CLOUD_REGION: str = Field(
        default="us-central1", description="Google Cloud Region"
    )
    GOOGLE_CLOUD_STORAGE_BUCKET: str = Field(description="Google Cloud Storage Bucket")
    GROQ_API_KEY: str = Field(description="GROQ API Key")
    PINECONE_API_KEY: str = Field(description="Pinecone API Key")
    HF_TOKEN: str = Field(description="HuggingFace Token")

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="allow"
    )
