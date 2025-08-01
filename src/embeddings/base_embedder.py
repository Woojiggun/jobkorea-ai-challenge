"""
Base embedder interface for all embedding implementations
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path
import logging

try:
    from config.settings import settings
except ImportError:
    # Fallback for when running without full config
    class Settings:
        embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        batch_size = 32
    settings = Settings()

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Abstract base class for all embedders"""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedder with a specific model
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name or settings.embedding_model
        self.model = self._load_model()
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
    def _load_model(self) -> SentenceTransformer:
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            return SentenceTransformer(self.model_name)
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    @abstractmethod
    def prepare_text(self, data: Dict[str, Any]) -> str:
        """
        Prepare text from data for embedding
        
        Args:
            data: Raw data dictionary
            
        Returns:
            Prepared text string
        """
        pass
    
    @abstractmethod
    def extract_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract numerical features from data
        
        Args:
            data: Raw data dictionary
            
        Returns:
            Dictionary of feature names to values
        """
        pass
    
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text
        
        Args:
            text: Single text or list of texts
            
        Returns:
            Embedding vector(s)
        """
        if isinstance(text, str):
            text = [text]
            
        embeddings = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        return embeddings[0] if len(text) == 1 else embeddings
    
    def embed(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> np.ndarray:
        """
        Generate embeddings for data
        
        Args:
            data: Single data dict or list of data dicts
            
        Returns:
            Embedding vector(s)
        """
        # Handle single data point
        if isinstance(data, dict):
            data = [data]
            single = True
        else:
            single = False
            
        # Prepare texts
        texts = [self.prepare_text(d) for d in data]
        
        # Generate text embeddings
        text_embeddings = self.embed_text(texts)
        
        # Extract features
        features_list = [self.extract_features(d) for d in data]
        
        # Combine text embeddings with features
        combined_embeddings = self._combine_embeddings_and_features(
            text_embeddings, features_list
        )
        
        return combined_embeddings[0] if single else combined_embeddings
    
    def _combine_embeddings_and_features(
        self, 
        text_embeddings: np.ndarray, 
        features_list: List[Dict[str, float]]
    ) -> np.ndarray:
        """
        Combine text embeddings with numerical features
        
        Args:
            text_embeddings: Text embedding vectors
            features_list: List of feature dictionaries
            
        Returns:
            Combined embedding vectors
        """
        if not features_list or not features_list[0]:
            return text_embeddings
            
        # Convert features to numpy array
        feature_keys = list(features_list[0].keys())
        feature_matrix = np.array([
            [f.get(k, 0.0) for k in feature_keys] 
            for f in features_list
        ])
        
        # Normalize features
        feature_matrix = self._normalize_features(feature_matrix)
        
        # Ensure text_embeddings is 2D
        if len(text_embeddings.shape) == 1:
            text_embeddings = text_embeddings.reshape(1, -1)
        
        # Ensure feature_matrix matches shape
        if len(feature_matrix.shape) == 1:
            feature_matrix = feature_matrix.reshape(1, -1)
        
        # Combine with text embeddings
        combined = np.hstack([text_embeddings, feature_matrix])
        
        return combined
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize feature values to [0, 1] range"""
        if features.shape[1] == 0:
            return features
            
        # Min-max normalization per feature
        min_vals = features.min(axis=0, keepdims=True)
        max_vals = features.max(axis=0, keepdims=True)
        
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        
        normalized = (features - min_vals) / range_vals
        
        return normalized
    
    def save_embeddings(self, embeddings: np.ndarray, path: Path):
        """Save embeddings to file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(embeddings, f)
            
        logger.info(f"Saved embeddings to {path}")
    
    def load_embeddings(self, path: Path) -> np.ndarray:
        """Load embeddings from file"""
        with open(path, 'rb') as f:
            embeddings = pickle.load(f)
            
        logger.info(f"Loaded embeddings from {path}")
        return embeddings
    
    def batch_embed(
        self, 
        data_list: List[Dict[str, Any]], 
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Embed data in batches for efficiency
        
        Args:
            data_list: List of data dictionaries
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings
        """
        batch_size = batch_size or settings.batch_size
        embeddings = []
        
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i + batch_size]
            batch_embeddings = self.embed(batch)
            embeddings.append(batch_embeddings)
            
        return np.vstack(embeddings)