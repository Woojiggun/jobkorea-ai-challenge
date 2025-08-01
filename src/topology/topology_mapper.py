"""
Topology mapper for creating and managing topological spaces
"""
from typing import Dict, List, Tuple, Set, Optional, Any
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import networkx as nx
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class TopologyNode:
    """Represents a node in the topological space"""
    id: str
    name: str
    attributes: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    neighbors: Set[str] = None
    
    def __post_init__(self):
        if self.neighbors is None:
            self.neighbors = set()


@dataclass
class TopologyEdge:
    """Represents an edge in the topological space"""
    source: str
    target: str
    weight: float
    attributes: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


class TopologyMapper:
    """
    Maps entities (companies, candidates) into a topological space
    """
    
    def __init__(self):
        self.nodes: Dict[str, TopologyNode] = {}
        self.edges: List[TopologyEdge] = []
        self.graph = nx.Graph()
        self.regions: Dict[str, Set[str]] = defaultdict(set)
        
    def add_node(
        self, 
        node_id: str, 
        name: str, 
        attributes: Dict[str, Any],
        embedding: Optional[np.ndarray] = None
    ) -> TopologyNode:
        """
        Add a node to the topological space
        
        Args:
            node_id: Unique identifier
            name: Node name
            attributes: Node attributes
            embedding: Optional embedding vector
            
        Returns:
            Created TopologyNode
        """
        node = TopologyNode(
            id=node_id,
            name=name,
            attributes=attributes,
            embedding=embedding
        )
        
        self.nodes[node_id] = node
        self.graph.add_node(node_id, **attributes)
        
        # Assign to region based on attributes
        region = self._determine_region(attributes)
        self.regions[region].add(node_id)
        
        logger.debug(f"Added node {node_id} to region {region}")
        
        return node
    
    def add_edge(
        self, 
        source: str, 
        target: str, 
        weight: float = 1.0,
        attributes: Optional[Dict[str, Any]] = None
    ) -> TopologyEdge:
        """
        Add an edge between nodes
        
        Args:
            source: Source node ID
            target: Target node ID
            weight: Edge weight (connection strength)
            attributes: Optional edge attributes
            
        Returns:
            Created TopologyEdge
        """
        if source not in self.nodes or target not in self.nodes:
            raise ValueError(f"Both nodes must exist: {source}, {target}")
            
        edge = TopologyEdge(
            source=source,
            target=target,
            weight=weight,
            attributes=attributes or {}
        )
        
        self.edges.append(edge)
        self.graph.add_edge(source, target, weight=weight, **edge.attributes)
        
        # Update neighbors
        self.nodes[source].neighbors.add(target)
        self.nodes[target].neighbors.add(source)
        
        return edge
    
    def _determine_region(self, attributes: Dict[str, Any]) -> str:
        """
        Determine which region a node belongs to based on attributes
        
        Args:
            attributes: Node attributes
            
        Returns:
            Region identifier
        """
        # Company regions
        if "employee_count" in attributes:
            size = attributes["employee_count"]
            if size < 50:
                base_region = "startup"
            elif size < 200:
                base_region = "scaleup"
            elif size < 1000:
                base_region = "midsize"
            else:
                base_region = "enterprise"
                
            # Add industry modifier
            industry = attributes.get("industry", "general")
            return f"{base_region}_{industry}"
            
        # Candidate regions  
        elif "years_experience" in attributes:
            years = attributes["years_experience"]
            if years < 2:
                base_region = "junior"
            elif years < 5:
                base_region = "mid"
            elif years < 10:
                base_region = "senior"
            else:
                base_region = "expert"
                
            # Add field modifier
            field = attributes.get("field", "general")
            return f"{base_region}_{field}"
            
        return "unknown"
    
    def find_neighbors(
        self, 
        node_id: str, 
        depth: int = 1
    ) -> Dict[int, Set[str]]:
        """
        Find neighbors at various depths
        
        Args:
            node_id: Starting node
            depth: Maximum depth to search
            
        Returns:
            Dictionary mapping depth to set of node IDs
        """
        if node_id not in self.nodes:
            return {}
            
        neighbors_by_depth = {}
        current_level = {node_id}
        visited = {node_id}
        
        for d in range(1, depth + 1):
            next_level = set()
            
            for current_node in current_level:
                for neighbor in self.nodes[current_node].neighbors:
                    if neighbor not in visited:
                        next_level.add(neighbor)
                        visited.add(neighbor)
                        
            neighbors_by_depth[d] = next_level
            current_level = next_level
            
            if not current_level:
                break
                
        return neighbors_by_depth
    
    def calculate_topological_distance(
        self, 
        source: str, 
        target: str
    ) -> float:
        """
        Calculate topological distance between nodes
        
        Args:
            source: Source node ID
            target: Target node ID
            
        Returns:
            Topological distance (infinity if not connected)
        """
        if source not in self.nodes or target not in self.nodes:
            return float('inf')
            
        try:
            # Shortest path length
            path_length = nx.shortest_path_length(
                self.graph, 
                source, 
                target,
                weight='weight'
            )
            return path_length
        except nx.NetworkXNoPath:
            return float('inf')
    
    def find_path(
        self, 
        source: str, 
        target: str
    ) -> Optional[List[str]]:
        """
        Find path between nodes
        
        Args:
            source: Source node ID
            target: Target node ID
            
        Returns:
            List of node IDs forming the path, or None
        """
        try:
            return nx.shortest_path(self.graph, source, target)
        except nx.NetworkXNoPath:
            return None
    
    def get_region_boundaries(self, region: str) -> Dict[str, Any]:
        """
        Get the boundaries of a region
        
        Args:
            region: Region identifier
            
        Returns:
            Dictionary describing region boundaries
        """
        if region not in self.regions:
            return {}
            
        region_nodes = self.regions[region]
        
        # Collect all attributes from nodes in region
        attributes_ranges = defaultdict(lambda: {"min": float('inf'), "max": float('-inf')})
        
        for node_id in region_nodes:
            node = self.nodes[node_id]
            for attr, value in node.attributes.items():
                if isinstance(value, (int, float)):
                    attributes_ranges[attr]["min"] = min(
                        attributes_ranges[attr]["min"], value
                    )
                    attributes_ranges[attr]["max"] = max(
                        attributes_ranges[attr]["max"], value
                    )
                    
        # Find boundary nodes (connected to other regions)
        boundary_nodes = set()
        for node_id in region_nodes:
            for neighbor in self.nodes[node_id].neighbors:
                neighbor_region = self._get_node_region(neighbor)
                if neighbor_region != region:
                    boundary_nodes.add(node_id)
                    break
                    
        return {
            "region": region,
            "size": len(region_nodes),
            "attribute_ranges": dict(attributes_ranges),
            "boundary_nodes": list(boundary_nodes),
            "core_nodes": list(region_nodes - boundary_nodes)
        }
    
    def _get_node_region(self, node_id: str) -> Optional[str]:
        """Get the region a node belongs to"""
        for region, nodes in self.regions.items():
            if node_id in nodes:
                return region
        return None
    
    def detect_clusters(self) -> Dict[str, List[str]]:
        """
        Detect natural clusters in the topology
        
        Returns:
            Dictionary mapping cluster ID to list of node IDs
        """
        # Use community detection
        communities = nx.community.louvain_communities(self.graph)
        
        clusters = {}
        for i, community in enumerate(communities):
            clusters[f"cluster_{i}"] = list(community)
            
        return clusters
    
    def calculate_density(self, region: str) -> float:
        """
        Calculate the density of connections in a region
        
        Args:
            region: Region identifier
            
        Returns:
            Connection density (0-1)
        """
        if region not in self.regions:
            return 0.0
            
        region_nodes = self.regions[region]
        if len(region_nodes) < 2:
            return 0.0
            
        # Count internal connections
        internal_edges = 0
        for node_id in region_nodes:
            for neighbor in self.nodes[node_id].neighbors:
                if neighbor in region_nodes:
                    internal_edges += 1
                    
        # Each edge is counted twice
        internal_edges //= 2
        
        # Maximum possible edges
        max_edges = len(region_nodes) * (len(region_nodes) - 1) // 2
        
        return internal_edges / max_edges if max_edges > 0 else 0.0
    
    def find_bridges(self) -> List[Tuple[str, str]]:
        """
        Find bridge connections between regions
        
        Returns:
            List of (source, target) tuples representing bridges
        """
        bridges = []
        
        for edge in self.edges:
            source_region = self._get_node_region(edge.source)
            target_region = self._get_node_region(edge.target)
            
            if source_region != target_region:
                bridges.append((edge.source, edge.target))
                
        return bridges
    
    def create_topology_embedding(self, node_id: str) -> np.ndarray:
        """
        Create an embedding that includes topological information
        
        Args:
            node_id: Node ID
            
        Returns:
            Enhanced embedding vector
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")
            
        node = self.nodes[node_id]
        
        # Start with base embedding
        if node.embedding is not None:
            base_embedding = node.embedding.copy()
        else:
            base_embedding = np.zeros(settings.embedding_dim)
            
        # Add topological features
        topo_features = []
        
        # Degree centrality
        degree = len(node.neighbors)
        topo_features.append(degree / len(self.nodes))
        
        # Clustering coefficient
        clustering = nx.clustering(self.graph, node_id)
        topo_features.append(clustering)
        
        # Betweenness centrality (cached for efficiency)
        if not hasattr(self, '_betweenness'):
            self._betweenness = nx.betweenness_centrality(self.graph)
        topo_features.append(self._betweenness.get(node_id, 0))
        
        # Region density
        region = self._get_node_region(node_id)
        if region:
            topo_features.append(self.calculate_density(region))
        else:
            topo_features.append(0.0)
            
        # Combine embeddings
        topo_vector = np.array(topo_features)
        enhanced_embedding = np.concatenate([base_embedding, topo_vector])
        
        return enhanced_embedding