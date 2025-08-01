"""
Vector store for efficient similarity search
"""
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import faiss
import pickle
from pathlib import Path
import logging
from datetime import datetime

from config.settings import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """
    FAISS-based vector store for embeddings
    """
    
    def __init__(
        self, 
        dimension: int,
        index_type: str = "L2",
        use_gpu: bool = False
    ):
        """
        Initialize vector store
        
        Args:
            dimension: Embedding dimension
            index_type: Type of index (L2, IP, etc.)
            use_gpu: Whether to use GPU acceleration
        """
        self.dimension = dimension
        self.index_type = index_type
        self.use_gpu = use_gpu and self._check_gpu_available()
        
        # Initialize FAISS index
        self.index = self._create_index()
        
        # Metadata storage
        self.metadata: Dict[int, Dict[str, Any]] = {}
        self.id_to_idx: Dict[str, int] = {}
        self.idx_to_id: Dict[int, str] = {}
        self.current_idx = 0
        
    def _check_gpu_available(self) -> bool:
        """Check if GPU is available for FAISS"""
        try:
            import faiss
            return faiss.get_num_gpus() > 0
        except:
            return False
    
    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on configuration"""
        if self.index_type == "L2":
            index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IP":
            index = faiss.IndexFlatIP(self.dimension)
        else:
            # Default to L2 with IVF for large-scale
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            
        if self.use_gpu:
            # Move index to GPU
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            
        logger.info(f"Created {self.index_type} index with dimension {self.dimension}")
        return index
    
    def add(
        self, 
        embeddings: np.ndarray, 
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Add embeddings to the store
        
        Args:
            embeddings: Numpy array of embeddings
            ids: List of unique identifiers
            metadata: Optional metadata for each embedding
        """
        if embeddings.shape[0] != len(ids):
            raise ValueError("Number of embeddings must match number of IDs")
            
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} doesn't match "
                f"index dimension {self.dimension}"
            )
            
        # Normalize embeddings if using inner product
        if self.index_type == "IP":
            faiss.normalize_L2(embeddings)
            
        # Add to index
        start_idx = self.current_idx
        self.index.add(embeddings)
        
        # Update mappings and metadata
        for i, (emb_id, meta) in enumerate(zip(ids, metadata or [{}])):
            idx = start_idx + i
            self.id_to_idx[emb_id] = idx
            self.idx_to_id[idx] = emb_id
            self.metadata[idx] = {
                **meta,
                "id": emb_id,
                "added_at": datetime.now().isoformat()
            }
            
        self.current_idx += len(ids)
        logger.info(f"Added {len(ids)} embeddings to store")
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 10,
        filter_fn: Optional[callable] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar embeddings
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filter_fn: Optional function to filter results
            
        Returns:
            List of (id, distance, metadata) tuples
        """
        if query_embedding.shape[0] != self.dimension:
            raise ValueError(
                f"Query dimension {query_embedding.shape[0]} doesn't match "
                f"index dimension {self.dimension}"
            )
            
        # Reshape for FAISS
        query = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Normalize if using inner product
        if self.index_type == "IP":
            faiss.normalize_L2(query)
            
        # Search
        distances, indices = self.index.search(query, k * 2)  # Get extra for filtering
        
        # Prepare results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
                
            # Get ID and metadata
            if idx in self.idx_to_id:
                result_id = self.idx_to_id[idx]
                result_meta = self.metadata.get(idx, {})
                
                # Apply filter if provided
                if filter_fn and not filter_fn(result_meta):
                    continue
                    
                results.append((result_id, float(dist), result_meta))
                
                if len(results) >= k:
                    break
                    
        return results
    
    def batch_search(
        self, 
        query_embeddings: np.ndarray, 
        k: int = 10
    ) -> List[List[Tuple[str, float, Dict[str, Any]]]]:
        """
        Batch search for multiple queries
        
        Args:
            query_embeddings: Array of query embeddings
            k: Number of results per query
            
        Returns:
            List of search results for each query
        """
        if query_embeddings.shape[1] != self.dimension:
            raise ValueError("Query dimensions don't match index dimension")
            
        # Normalize if needed
        if self.index_type == "IP":
            faiss.normalize_L2(query_embeddings)
            
        # Batch search
        distances, indices = self.index.search(query_embeddings, k)
        
        # Prepare results for each query
        all_results = []
        for query_dists, query_indices in zip(distances, indices):
            results = []
            for dist, idx in zip(query_dists, query_indices):
                if idx == -1:
                    continue
                    
                if idx in self.idx_to_id:
                    result_id = self.idx_to_id[idx]
                    result_meta = self.metadata.get(idx, {})
                    results.append((result_id, float(dist), result_meta))
                    
            all_results.append(results)
            
        return all_results
    
    def get_by_id(self, emb_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata by ID"""
        idx = self.id_to_idx.get(emb_id)
        if idx is not None:
            return self.metadata.get(idx)
        return None
    
    def remove(self, ids: List[str]):
        """
        Remove embeddings by ID (note: FAISS doesn't support direct removal,
        so this marks them as deleted)
        """
        for emb_id in ids:
            if emb_id in self.id_to_idx:
                idx = self.id_to_idx[emb_id]
                # Mark as deleted in metadata
                if idx in self.metadata:
                    self.metadata[idx]["deleted"] = True
                # Remove from mappings
                del self.id_to_idx[emb_id]
                del self.idx_to_id[idx]
                
        logger.info(f"Marked {len(ids)} embeddings as deleted")
    
    def save(self, path: Path):
        """Save index and metadata to disk"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = path / "index.faiss"
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata and mappings
        meta_path = path / "metadata.pkl"
        with open(meta_path, 'wb') as f:
            pickle.dump({
                "metadata": self.metadata,
                "id_to_idx": self.id_to_idx,
                "idx_to_id": self.idx_to_id,
                "current_idx": self.current_idx,
                "dimension": self.dimension,
                "index_type": self.index_type
            }, f)
            
        logger.info(f"Saved vector store to {path}")
    
    def load(self, path: Path):
        """Load index and metadata from disk"""
        path = Path(path)
        
        # Load FAISS index
        index_path = path / "index.faiss"
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata and mappings
        meta_path = path / "metadata.pkl"
        with open(meta_path, 'rb') as f:
            data = pickle.load(f)
            self.metadata = data["metadata"]
            self.id_to_idx = data["id_to_idx"]
            self.idx_to_id = data["idx_to_id"]
            self.current_idx = data["current_idx"]
            self.dimension = data["dimension"]
            self.index_type = data["index_type"]
            
        logger.info(f"Loaded vector store from {path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        total_embeddings = self.index.ntotal
        active_embeddings = len(self.id_to_idx)
        deleted_embeddings = total_embeddings - active_embeddings
        
        return {
            "total_embeddings": total_embeddings,
            "active_embeddings": active_embeddings,
            "deleted_embeddings": deleted_embeddings,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "use_gpu": self.use_gpu
        }
    
    def rebuild_index(self):
        """
        Rebuild index without deleted embeddings
        (useful for periodic maintenance)
        """
        # Get all active embeddings
        active_embeddings = []
        active_ids = []
        active_metadata = []
        
        for emb_id, idx in self.id_to_idx.items():
            if idx in self.metadata and not self.metadata[idx].get("deleted", False):
                # Reconstruct embedding (would need to store or regenerate)
                # For now, we'll skip this as it requires storing embeddings
                logger.warning("Index rebuild requires stored embeddings")
                return
                
        logger.info("Index rebuild completed")