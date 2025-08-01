"""
Gravity field implementation for topological space
"""
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
import logging

from .topology_mapper import TopologyMapper, TopologyNode

logger = logging.getLogger(__name__)


@dataclass
class GravityCenter:
    """Represents a center of gravity in the job market"""
    id: str
    name: str
    position: np.ndarray
    mass: float
    attributes: Dict[str, Any]
    influence_radius: float = 10.0


class GravityField:
    """
    Implements gravity field effects in the topological job market space
    """
    
    def __init__(self, topology: TopologyMapper):
        self.topology = topology
        self.gravity_centers: Dict[str, GravityCenter] = {}
        self.field_strength = 1.0
        
    def identify_mass_centers(self) -> List[GravityCenter]:
        """
        Identify major gravity centers based on node properties
        
        Returns:
            List of identified gravity centers
        """
        centers = []
        
        # Group nodes by region
        for region, node_ids in self.topology.regions.items():
            if len(node_ids) < 3:  # Skip small regions
                continue
                
            # Calculate center of mass for region
            total_mass = 0
            weighted_position = np.zeros(3)  # Using 3D for visualization
            
            for node_id in node_ids:
                node = self.topology.nodes[node_id]
                mass = self._calculate_node_mass(node)
                position = self._get_node_position(node)
                
                total_mass += mass
                weighted_position += mass * position
                
            if total_mass > 0:
                center_position = weighted_position / total_mass
                
                # Create gravity center
                center = GravityCenter(
                    id=f"center_{region}",
                    name=f"{region} cluster",
                    position=center_position,
                    mass=total_mass,
                    attributes={"region": region, "node_count": len(node_ids)}
                )
                
                centers.append(center)
                self.gravity_centers[center.id] = center
                
        logger.info(f"Identified {len(centers)} gravity centers")
        return centers
    
    def _calculate_node_mass(self, node: TopologyNode) -> float:
        """
        Calculate the 'mass' of a node based on its properties
        
        Args:
            node: TopologyNode
            
        Returns:
            Mass value
        """
        mass = 1.0  # Base mass
        
        # For companies
        if "employee_count" in node.attributes:
            mass *= np.log(node.attributes["employee_count"] + 1)
            
        if "revenue" in node.attributes:
            mass *= np.log(node.attributes["revenue"] + 1) / 10
            
        if "market_position" in node.attributes:
            mass *= node.attributes["market_position"]
            
        # For candidates
        if "years_experience" in node.attributes:
            mass *= (node.attributes["years_experience"] / 10 + 1)
            
        if "skill_count" in node.attributes:
            mass *= (node.attributes["skill_count"] / 20 + 1)
            
        return mass
    
    def _get_node_position(self, node: TopologyNode) -> np.ndarray:
        """
        Get or calculate position for a node
        
        Args:
            node: TopologyNode
            
        Returns:
            3D position vector
        """
        if node.embedding is not None and len(node.embedding) >= 3:
            # Use first 3 dimensions of embedding
            return node.embedding[:3]
        else:
            # Generate position based on attributes
            x = node.attributes.get("employee_count", 0) / 1000
            y = node.attributes.get("years_experience", 0) / 20
            z = node.attributes.get("satisfaction_score", 0.5)
            return np.array([x, y, z])
    
    def calculate_gravitational_force(
        self, 
        position: np.ndarray, 
        center: GravityCenter
    ) -> np.ndarray:
        """
        Calculate gravitational force at a position due to a center
        
        Args:
            position: Current position
            center: Gravity center
            
        Returns:
            Force vector
        """
        # Vector from position to center
        direction = center.position - position
        distance = np.linalg.norm(direction)
        
        if distance < 0.01:  # Avoid division by zero
            return np.zeros_like(position)
            
        # Normalize direction
        direction = direction / distance
        
        # Force magnitude (inverse square law with cutoff)
        if distance > center.influence_radius:
            force_magnitude = 0
        else:
            force_magnitude = self.field_strength * center.mass / (distance ** 2)
            
        return force_magnitude * direction
    
    def calculate_total_force(self, position: np.ndarray) -> np.ndarray:
        """
        Calculate total gravitational force at a position
        
        Args:
            position: Current position
            
        Returns:
            Total force vector
        """
        total_force = np.zeros_like(position)
        
        for center in self.gravity_centers.values():
            force = self.calculate_gravitational_force(position, center)
            total_force += force
            
        return total_force
    
    def predict_trajectory(
        self, 
        start_position: np.ndarray,
        initial_velocity: np.ndarray,
        time_steps: int = 10,
        dt: float = 0.1
    ) -> List[np.ndarray]:
        """
        Predict trajectory under gravitational influence
        
        Args:
            start_position: Starting position
            initial_velocity: Initial velocity vector
            time_steps: Number of time steps to simulate
            dt: Time step size
            
        Returns:
            List of positions along trajectory
        """
        trajectory = [start_position.copy()]
        position = start_position.copy()
        velocity = initial_velocity.copy()
        
        for _ in range(time_steps):
            # Calculate force
            force = self.calculate_total_force(position)
            
            # Update velocity (F = ma, assuming m = 1)
            velocity += force * dt
            
            # Add some damping to prevent runaway
            velocity *= 0.98
            
            # Update position
            position += velocity * dt
            
            trajectory.append(position.copy())
            
        return trajectory
    
    def find_equilibrium_points(self) -> List[np.ndarray]:
        """
        Find equilibrium points in the gravity field
        
        Returns:
            List of equilibrium positions
        """
        equilibrium_points = []
        
        # Check between pairs of gravity centers
        centers = list(self.gravity_centers.values())
        
        for i, center1 in enumerate(centers):
            for center2 in centers[i+1:]:
                # Line between centers
                direction = center2.position - center1.position
                distance = np.linalg.norm(direction)
                
                if distance < 0.01:
                    continue
                    
                # Check points along the line
                for t in np.linspace(0.1, 0.9, 9):
                    test_point = center1.position + t * direction
                    force = self.calculate_total_force(test_point)
                    
                    # If force is very small, it's near equilibrium
                    if np.linalg.norm(force) < 0.01:
                        equilibrium_points.append(test_point)
                        
        return equilibrium_points
    
    def calculate_orbital_velocity(
        self, 
        position: np.ndarray, 
        center: GravityCenter
    ) -> float:
        """
        Calculate orbital velocity needed at a position
        
        Args:
            position: Current position
            center: Gravity center to orbit
            
        Returns:
            Required orbital velocity magnitude
        """
        distance = np.linalg.norm(position - center.position)
        
        if distance < 0.01:
            return 0.0
            
        # Circular orbit: v = sqrt(GM/r)
        return np.sqrt(self.field_strength * center.mass / distance)
    
    def classify_career_trajectory(
        self, 
        trajectory: List[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Classify a career trajectory based on its path
        
        Args:
            trajectory: List of positions
            
        Returns:
            Classification results
        """
        if len(trajectory) < 2:
            return {"type": "stationary", "characteristics": {}}
            
        # Calculate trajectory characteristics
        velocities = []
        accelerations = []
        
        for i in range(1, len(trajectory)):
            velocity = trajectory[i] - trajectory[i-1]
            velocities.append(np.linalg.norm(velocity))
            
            if i > 1:
                acceleration = velocity - (trajectory[i-1] - trajectory[i-2])
                accelerations.append(np.linalg.norm(acceleration))
                
        avg_velocity = np.mean(velocities)
        velocity_variance = np.var(velocities)
        
        # Classify based on patterns
        if avg_velocity < 0.1:
            trajectory_type = "stable"
        elif velocity_variance < 0.01:
            trajectory_type = "steady_growth"
        elif avg_velocity > 0.5:
            trajectory_type = "rapid_advancement"
        else:
            trajectory_type = "dynamic"
            
        # Find closest gravity center
        final_position = trajectory[-1]
        closest_center = None
        min_distance = float('inf')
        
        for center in self.gravity_centers.values():
            distance = np.linalg.norm(final_position - center.position)
            if distance < min_distance:
                min_distance = distance
                closest_center = center
                
        return {
            "type": trajectory_type,
            "characteristics": {
                "average_velocity": avg_velocity,
                "velocity_variance": velocity_variance,
                "total_distance": sum(velocities),
                "closest_center": closest_center.name if closest_center else None,
                "distance_to_center": min_distance
            }
        }
    
    def recommend_next_move(
        self, 
        current_position: np.ndarray,
        desired_direction: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Recommend next career move based on gravity field
        
        Args:
            current_position: Current position in space
            desired_direction: Optional preferred direction
            
        Returns:
            Recommendations with rationale
        """
        # Calculate forces at current position
        total_force = self.calculate_total_force(current_position)
        
        # Find nearby gravity centers
        nearby_centers = []
        for center in self.gravity_centers.values():
            distance = np.linalg.norm(current_position - center.position)
            if distance < center.influence_radius:
                nearby_centers.append((center, distance))
                
        nearby_centers.sort(key=lambda x: x[1])
        
        # Generate recommendations
        recommendations = []
        
        # Natural path (follow the force)
        if np.linalg.norm(total_force) > 0.01:
            natural_direction = total_force / np.linalg.norm(total_force)
            recommendations.append({
                "type": "natural_progression",
                "direction": natural_direction,
                "rationale": "Following market forces and demand",
                "effort": "low"
            })
            
        # Orbital paths around nearby centers
        for center, distance in nearby_centers[:3]:
            orbital_velocity = self.calculate_orbital_velocity(
                current_position, center
            )
            recommendations.append({
                "type": "orbital",
                "target": center.name,
                "velocity_required": orbital_velocity,
                "rationale": f"Stable position near {center.name}",
                "effort": "medium"
            })
            
        # Escape trajectory (if desired_direction provided)
        if desired_direction is not None:
            escape_velocity = self._calculate_escape_velocity(current_position)
            recommendations.append({
                "type": "escape",
                "direction": desired_direction,
                "velocity_required": escape_velocity,
                "rationale": "Breaking into new territory",
                "effort": "high"
            })
            
        return {
            "current_forces": total_force.tolist(),
            "nearby_centers": [(c.name, d) for c, d in nearby_centers],
            "recommendations": recommendations
        }
    
    def _calculate_escape_velocity(self, position: np.ndarray) -> float:
        """Calculate velocity needed to escape current gravity well"""
        # Find dominant gravity center
        max_force = 0
        dominant_center = None
        
        for center in self.gravity_centers.values():
            force = self.calculate_gravitational_force(position, center)
            force_magnitude = np.linalg.norm(force)
            
            if force_magnitude > max_force:
                max_force = force_magnitude
                dominant_center = center
                
        if dominant_center is None:
            return 0.0
            
        distance = np.linalg.norm(position - dominant_center.position)
        
        # Escape velocity: v = sqrt(2GM/r)
        return np.sqrt(2 * self.field_strength * dominant_center.mass / distance)