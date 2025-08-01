"""
Application configuration settings
"""
import os
from pathlib import Path
from typing import Optional
try:
    from pydantic_settings import BaseSettings
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings"""
    
    # API Keys
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    
    # Server Configuration
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8000, env="PORT")
    debug: bool = Field(True, env="DEBUG")
    
    # Vector Store Configuration
    faiss_index_path: Path = Field(
        Path("./data/embeddings/faiss_index"), 
        env="FAISS_INDEX_PATH"
    )
    embedding_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL"
    )
    
    # Cache Configuration
    cache_ttl: int = Field(3600, env="CACHE_TTL")
    max_cache_size: int = Field(1000, env="MAX_CACHE_SIZE")
    
    # Topology Configuration
    topology_depth: int = Field(3, env="TOPOLOGY_DEPTH")
    gravity_strength: float = Field(1.0, env="GRAVITY_STRENGTH")
    
    # Performance Settings
    max_workers: int = Field(4, env="MAX_WORKERS")
    batch_size: int = Field(32, env="BATCH_SIZE")
    
    # Paths
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = base_dir / "data"
    companies_dir: Path = data_dir / "companies"
    embeddings_dir: Path = data_dir / "embeddings"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
try:
    settings = Settings()
except Exception:
    # Create settings with defaults when .env is missing
    class DefaultSettings:
        openai_api_key = "sk-test-key"
        host = "0.0.0.0"
        port = 8000
        debug = True
        faiss_index_path = Path("./data/embeddings/faiss_index")
        embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        cache_ttl = 3600
        max_cache_size = 1000
        topology_depth = 3
        gravity_strength = 1.0
        max_workers = 4
        batch_size = 32
        base_dir = Path(__file__).parent.parent
        data_dir = base_dir / "data"
        companies_dir = data_dir / "companies"
        embeddings_dir = data_dir / "embeddings"
    
    settings = DefaultSettings()