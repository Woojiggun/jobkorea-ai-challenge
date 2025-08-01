"""
Topology-aware retriever for RAG system
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging
from dataclasses import dataclass

from .vector_store import VectorStore
from src.topology.topology_mapper import TopologyMapper
from src.topology.boundary_validator import BoundaryValidator

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from retrieval operation"""
    documents: List[Dict[str, Any]]
    scores: List[float]
    filtered_count: int
    strategy_used: str


class TopologyAwareRetriever:
    """
    Retriever that considers topological boundaries and relationships
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        topology: TopologyMapper,
        boundary_validator: BoundaryValidator
    ):
        """
        Initialize retriever
        
        Args:
            vector_store: Vector store instance
            topology: Topology mapper instance
            boundary_validator: Boundary validator instance
        """
        self.vector_store = vector_store
        self.topology = topology
        self.boundary_validator = boundary_validator
        
    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        company_id: Optional[str] = None,
        k: int = 10,
        strategy: str = "hybrid"
    ) -> RetrievalResult:
        """
        Retrieve relevant documents with topology awareness
        
        Args:
            query: Query text
            query_embedding: Query embedding vector
            company_id: Optional company ID for context
            k: Number of documents to retrieve
            strategy: Retrieval strategy (hybrid, strict, exploration)
            
        Returns:
            RetrievalResult
        """
        if strategy == "strict":
            return self._strict_retrieval(query_embedding, company_id, k)
        elif strategy == "exploration":
            return self._exploration_retrieval(query_embedding, company_id, k)
        else:  # hybrid
            return self._hybrid_retrieval(query, query_embedding, company_id, k)
    
    def _strict_retrieval(
        self,
        query_embedding: np.ndarray,
        company_id: Optional[str],
        k: int
    ) -> RetrievalResult:
        """
        Strict retrieval within topological boundaries
        
        Args:
            query_embedding: Query embedding
            company_id: Company ID for boundary determination
            k: Number of results
            
        Returns:
            RetrievalResult
        """
        # Determine region if company_id provided
        allowed_regions = set()
        if company_id and company_id in self.topology.nodes:
            region = self.topology._get_node_region(company_id)
            if region:
                allowed_regions.add(region)
                
                # Add immediately adjacent regions
                boundaries = self.topology.get_region_boundaries(region)
                for boundary_node in boundaries["boundary_nodes"]:
                    neighbors = self.topology.nodes[boundary_node].neighbors
                    for neighbor in neighbors:
                        neighbor_region = self.topology._get_node_region(neighbor)
                        if neighbor_region:
                            allowed_regions.add(neighbor_region)
        
        # Define filter function
        def region_filter(metadata: Dict[str, Any]) -> bool:
            if not allowed_regions:  # No restrictions
                return True
                
            doc_region = metadata.get("region")
            return doc_region in allowed_regions
        
        # Search with filter
        results = self.vector_store.search(
            query_embedding,
            k=k * 2,  # Get extra to account for filtering
            filter_fn=region_filter
        )
        
        # Take top k after filtering
        results = results[:k]
        
        return RetrievalResult(
            documents=[r[2] for r in results],
            scores=[r[1] for r in results],
            filtered_count=len(results),
            strategy_used="strict"
        )
    
    def _exploration_retrieval(
        self,
        query_embedding: np.ndarray,
        company_id: Optional[str],
        k: int
    ) -> RetrievalResult:
        """
        Exploration retrieval that looks beyond immediate boundaries
        
        Args:
            query_embedding: Query embedding
            company_id: Company ID for context
            k: Number of results
            
        Returns:
            RetrievalResult
        """
        # Get initial results without filtering
        initial_results = self.vector_store.search(query_embedding, k=k * 3)
        
        # Score results based on topological distance
        scored_results = []
        
        for doc_id, distance, metadata in initial_results:
            # Calculate topological bonus/penalty
            topo_score = 1.0
            
            if company_id and company_id in self.topology.nodes:
                doc_node_id = metadata.get("node_id")
                if doc_node_id and doc_node_id in self.topology.nodes:
                    # Calculate topological distance
                    topo_distance = self.topology.calculate_topological_distance(
                        company_id, doc_node_id
                    )
                    
                    # Convert to score (closer = higher score)
                    if topo_distance != float('inf'):
                        topo_score = 1.0 / (1.0 + topo_distance)
                    else:
                        topo_score = 0.1  # Penalty for disconnected nodes
            
            # Combine vector distance and topological score
            combined_score = (1.0 / (1.0 + distance)) * topo_score
            
            scored_results.append((
                metadata,
                combined_score,
                f"vec_dist: {distance:.3f}, topo_score: {topo_score:.3f}"
            ))
        
        # Sort by combined score
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Take top k
        final_results = scored_results[:k]
        
        return RetrievalResult(
            documents=[r[0] for r in final_results],
            scores=[r[1] for r in final_results],
            filtered_count=len(final_results),
            strategy_used="exploration"
        )
    
    def _hybrid_retrieval(
        self,
        query: str,
        query_embedding: np.ndarray,
        company_id: Optional[str],
        k: int
    ) -> RetrievalResult:
        """
        Hybrid retrieval balancing relevance and boundaries
        
        Args:
            query: Query text
            query_embedding: Query embedding
            company_id: Company ID for context
            k: Number of results
            
        Returns:
            RetrievalResult
        """
        # Get results from both strategies
        strict_results = self._strict_retrieval(
            query_embedding, company_id, k // 2
        )
        exploration_results = self._exploration_retrieval(
            query_embedding, company_id, k // 2
        )
        
        # Combine results, deduplicating
        seen_ids = set()
        combined_documents = []
        combined_scores = []
        
        # Add strict results first (higher priority)
        for doc, score in zip(strict_results.documents, strict_results.scores):
            doc_id = doc.get("id")
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                combined_documents.append(doc)
                combined_scores.append(score * 1.2)  # Boost for being in boundary
        
        # Add exploration results
        for doc, score in zip(exploration_results.documents, exploration_results.scores):
            doc_id = doc.get("id")
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                combined_documents.append(doc)
                combined_scores.append(score)
        
        # Validate results against boundaries
        validated_documents = []
        validated_scores = []
        
        for doc, score in zip(combined_documents, combined_scores):
            # Check if document respects boundaries
            if self._validate_document(doc, company_id):
                validated_documents.append(doc)
                validated_scores.append(score)
        
        return RetrievalResult(
            documents=validated_documents[:k],
            scores=validated_scores[:k],
            filtered_count=len(validated_documents),
            strategy_used="hybrid"
        )
    
    def _validate_document(
        self,
        document: Dict[str, Any],
        company_id: Optional[str]
    ) -> bool:
        """
        Validate if a document respects topological boundaries
        
        Args:
            document: Document metadata
            company_id: Company ID for context
            
        Returns:
            True if document is valid
        """
        if not company_id:
            return True  # No context to validate against
            
        # Get company data
        company_node = self.topology.nodes.get(company_id)
        if not company_node:
            return True  # Can't validate without company data
            
        # Simple validation: check if document is from compatible region
        doc_region = document.get("region")
        company_region = self.topology._get_node_region(company_id)
        
        if not doc_region or not company_region:
            return True  # Can't validate without regions
            
        # Check if regions are compatible
        if doc_region == company_region:
            return True
            
        # Check if regions are connected
        region_boundaries = self.topology.get_region_boundaries(company_region)
        boundary_regions = set()
        
        for boundary_node in region_boundaries["boundary_nodes"]:
            for neighbor in self.topology.nodes[boundary_node].neighbors:
                neighbor_region = self.topology._get_node_region(neighbor)
                if neighbor_region:
                    boundary_regions.add(neighbor_region)
                    
        return doc_region in boundary_regions
    
    def retrieve_with_context(
        self,
        query: str,
        query_embedding: np.ndarray,
        context: Dict[str, Any],
        k: int = 10
    ) -> RetrievalResult:
        """
        Retrieve with rich context information
        
        Args:
            query: Query text
            query_embedding: Query embedding
            context: Context dictionary with company_id, region, etc.
            k: Number of results
            
        Returns:
            RetrievalResult
        """
        company_id = context.get("company_id")
        preferred_regions = context.get("preferred_regions", [])
        required_attributes = context.get("required_attributes", {})
        
        # Create filter function based on context
        def context_filter(metadata: Dict[str, Any]) -> bool:
            # Check preferred regions
            if preferred_regions:
                doc_region = metadata.get("region")
                if doc_region not in preferred_regions:
                    return False
                    
            # Check required attributes
            for attr, required_value in required_attributes.items():
                doc_value = metadata.get(attr)
                if doc_value != required_value:
                    return False
                    
            return True
        
        # Search with context filter
        results = self.vector_store.search(
            query_embedding,
            k=k * 2,
            filter_fn=context_filter
        )
        
        # Apply topological scoring
        scored_results = []
        for doc_id, distance, metadata in results:
            # Calculate context-aware score
            score = self._calculate_context_score(
                metadata, context, distance
            )
            scored_results.append((metadata, score))
        
        # Sort by score
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Take top k
        final_results = scored_results[:k]
        
        return RetrievalResult(
            documents=[r[0] for r in final_results],
            scores=[r[1] for r in final_results],
            filtered_count=len(final_results),
            strategy_used="context_aware"
        )
    
    def _calculate_context_score(
        self,
        metadata: Dict[str, Any],
        context: Dict[str, Any],
        distance: float
    ) -> float:
        """
        Calculate context-aware relevance score
        
        Args:
            metadata: Document metadata
            context: Query context
            distance: Vector distance
            
        Returns:
            Relevance score
        """
        # Base score from vector similarity
        base_score = 1.0 / (1.0 + distance)
        
        # Apply context modifiers
        score_multiplier = 1.0
        
        # Region match bonus
        if "preferred_regions" in context:
            doc_region = metadata.get("region")
            if doc_region in context["preferred_regions"]:
                score_multiplier *= 1.5
                
        # Attribute match bonus
        if "preferred_attributes" in context:
            matches = 0
            for attr, preferred_value in context["preferred_attributes"].items():
                if metadata.get(attr) == preferred_value:
                    matches += 1
            score_multiplier *= (1.0 + 0.1 * matches)
            
        # Recency bonus
        if "prefer_recent" in context and context["prefer_recent"]:
            added_at = metadata.get("added_at", "")
            # Simple recency check (would need proper date parsing)
            if "2024" in added_at:
                score_multiplier *= 1.2
                
        return base_score * score_multiplier
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about retrieval performance"""
        vector_stats = self.vector_store.get_stats()
        topology_stats = {
            "total_nodes": len(self.topology.nodes),
            "total_regions": len(self.topology.regions),
            "total_edges": len(self.topology.edges)
        }
        
        return {
            "vector_store": vector_stats,
            "topology": topology_stats
        }