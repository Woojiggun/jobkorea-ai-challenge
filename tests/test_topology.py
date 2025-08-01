"""
Tests for topology system
"""
import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.topology import TopologyMapper, GravityField, BoundaryValidator


class TestTopologyMapper:
    """Test topology mapper functionality"""
    
    @pytest.fixture
    def topology(self):
        """Create topology mapper instance"""
        return TopologyMapper()
    
    @pytest.fixture
    def sample_nodes(self):
        """Sample nodes for testing"""
        return [
            {
                "id": "company_1",
                "name": "스타트업 A",
                "attributes": {"employee_count": 30, "industry": "tech"}
            },
            {
                "id": "company_2", 
                "name": "스타트업 B",
                "attributes": {"employee_count": 45, "industry": "tech"}
            },
            {
                "id": "company_3",
                "name": "대기업 C",
                "attributes": {"employee_count": 5000, "industry": "tech"}
            }
        ]
    
    def test_add_node(self, topology):
        """Test adding nodes to topology"""
        node = topology.add_node(
            "test_1",
            "Test Company",
            {"employee_count": 100, "industry": "tech"}
        )
        
        assert node.id == "test_1"
        assert node.name == "Test Company"
        assert "test_1" in topology.nodes
        assert topology.graph.has_node("test_1")
    
    def test_add_edge(self, topology, sample_nodes):
        """Test adding edges between nodes"""
        # Add nodes first
        for node_data in sample_nodes[:2]:
            topology.add_node(
                node_data["id"],
                node_data["name"],
                node_data["attributes"]
            )
        
        # Add edge
        edge = topology.add_edge("company_1", "company_2", weight=0.8)
        
        assert edge.source == "company_1"
        assert edge.target == "company_2"
        assert edge.weight == 0.8
        assert topology.graph.has_edge("company_1", "company_2")
        assert "company_2" in topology.nodes["company_1"].neighbors
    
    def test_region_assignment(self, topology, sample_nodes):
        """Test automatic region assignment"""
        for node_data in sample_nodes:
            topology.add_node(
                node_data["id"],
                node_data["name"],
                node_data["attributes"]
            )
        
        # Check regions
        assert "startup_tech" in topology.regions
        assert "enterprise_tech" in topology.regions
        assert "company_1" in topology.regions["startup_tech"]
        assert "company_3" in topology.regions["enterprise_tech"]
    
    def test_find_neighbors(self, topology, sample_nodes):
        """Test finding neighbors at various depths"""
        # Create a simple chain: 1 -> 2 -> 3
        for node_data in sample_nodes:
            topology.add_node(
                node_data["id"],
                node_data["name"],
                node_data["attributes"]
            )
        
        topology.add_edge("company_1", "company_2")
        topology.add_edge("company_2", "company_3")
        
        # Find neighbors
        neighbors = topology.find_neighbors("company_1", depth=2)
        
        assert 1 in neighbors
        assert "company_2" in neighbors[1]
        assert 2 in neighbors
        assert "company_3" in neighbors[2]
    
    def test_topological_distance(self, topology, sample_nodes):
        """Test topological distance calculation"""
        # Create nodes and edges
        for node_data in sample_nodes:
            topology.add_node(
                node_data["id"],
                node_data["name"],
                node_data["attributes"]
            )
        
        topology.add_edge("company_1", "company_2", weight=1.0)
        topology.add_edge("company_2", "company_3", weight=2.0)
        
        # Calculate distances
        dist_12 = topology.calculate_topological_distance("company_1", "company_2")
        dist_13 = topology.calculate_topological_distance("company_1", "company_3")
        dist_11 = topology.calculate_topological_distance("company_1", "company_1")
        
        assert dist_12 == 1.0
        assert dist_13 == 3.0  # 1 + 2
        assert dist_11 == 0  # Distance to self
    
    def test_region_boundaries(self, topology, sample_nodes):
        """Test getting region boundaries"""
        # Add nodes
        for node_data in sample_nodes:
            topology.add_node(
                node_data["id"],
                node_data["name"],
                node_data["attributes"]
            )
        
        # Get boundaries
        boundaries = topology.get_region_boundaries("startup_tech")
        
        assert "region" in boundaries
        assert boundaries["region"] == "startup_tech"
        assert "size" in boundaries
        assert boundaries["size"] >= 2
        assert "attribute_ranges" in boundaries
        assert "employee_count" in boundaries["attribute_ranges"]


class TestGravityField:
    """Test gravity field functionality"""
    
    @pytest.fixture
    def topology_with_nodes(self):
        """Create topology with sample nodes"""
        topology = TopologyMapper()
        
        # Add diverse nodes
        nodes = [
            ("big_corp_1", "대기업 1", {"employee_count": 10000, "revenue": 1000000}),
            ("big_corp_2", "대기업 2", {"employee_count": 8000, "revenue": 800000}),
            ("startup_1", "스타트업 1", {"employee_count": 20, "revenue": 1000}),
            ("startup_2", "스타트업 2", {"employee_count": 30, "revenue": 2000})
        ]
        
        for node_id, name, attrs in nodes:
            topology.add_node(node_id, name, attrs, 
                            embedding=np.random.randn(3))  # 3D for testing
        
        return topology
    
    @pytest.fixture
    def gravity_field(self, topology_with_nodes):
        """Create gravity field instance"""
        return GravityField(topology_with_nodes)
    
    def test_identify_mass_centers(self, gravity_field):
        """Test identification of gravity centers"""
        centers = gravity_field.identify_mass_centers()
        
        assert len(centers) > 0
        assert all(hasattr(c, "mass") for c in centers)
        assert all(hasattr(c, "position") for c in centers)
        
        # Large companies should have more mass
        large_corp_center = next(
            (c for c in centers if "enterprise" in c.id), None
        )
        startup_center = next(
            (c for c in centers if "startup" in c.id), None
        )
        
        if large_corp_center and startup_center:
            assert large_corp_center.mass > startup_center.mass
    
    def test_gravitational_force(self, gravity_field):
        """Test gravitational force calculation"""
        # Create a gravity center
        center = gravity_field.GravityCenter(
            id="test_center",
            name="Test Center",
            position=np.array([0, 0, 0]),
            mass=1000,
            attributes={}
        )
        gravity_field.gravity_centers["test_center"] = center
        
        # Test force at different positions
        pos1 = np.array([1, 0, 0])
        force1 = gravity_field.calculate_gravitational_force(pos1, center)
        
        pos2 = np.array([2, 0, 0])
        force2 = gravity_field.calculate_gravitational_force(pos2, center)
        
        # Force should decrease with distance
        assert np.linalg.norm(force1) > np.linalg.norm(force2)
        
        # Force should point toward center
        assert force1[0] < 0  # Negative x direction (toward origin)
    
    def test_trajectory_prediction(self, gravity_field):
        """Test trajectory prediction"""
        # Add a gravity center
        center = gravity_field.GravityCenter(
            id="test_center",
            name="Test Center",
            position=np.array([5, 5, 0]),
            mass=1000,
            attributes={}
        )
        gravity_field.gravity_centers["test_center"] = center
        
        # Predict trajectory
        start_pos = np.array([0, 0, 0])
        initial_velocity = np.array([1, 1, 0])
        
        trajectory = gravity_field.predict_trajectory(
            start_pos, initial_velocity, time_steps=5
        )
        
        assert len(trajectory) == 6  # Initial + 5 steps
        assert isinstance(trajectory[0], np.ndarray)
        
        # Should curve toward gravity center
        final_pos = trajectory[-1]
        assert np.linalg.norm(final_pos - center.position) < \
               np.linalg.norm(start_pos - center.position)
    
    def test_career_trajectory_classification(self, gravity_field):
        """Test career trajectory classification"""
        # Create sample trajectory
        trajectory = [
            np.array([0, 0, 0]),
            np.array([0.1, 0.1, 0]),
            np.array([0.2, 0.2, 0]),
            np.array([0.3, 0.3, 0])
        ]
        
        classification = gravity_field.classify_career_trajectory(trajectory)
        
        assert "type" in classification
        assert "characteristics" in classification
        assert "average_velocity" in classification["characteristics"]
        assert classification["type"] in [
            "stable", "steady_growth", "rapid_advancement", "dynamic"
        ]


class TestBoundaryValidator:
    """Test boundary validator functionality"""
    
    @pytest.fixture
    def topology(self):
        """Create topology for validator"""
        topology = TopologyMapper()
        topology.add_node(
            "startup_1",
            "Test Startup",
            {"employee_count": 30, "industry": "tech"}
        )
        return topology
    
    @pytest.fixture
    def validator(self, topology):
        """Create boundary validator instance"""
        return BoundaryValidator(topology)
    
    @pytest.fixture
    def sample_content(self):
        """Sample generated content"""
        return """
        테스트 스타트업 채용공고
        
        우리는 30명의 직원과 함께 성장하는 기술 스타트업입니다.
        2020년에 설립되어 빠르게 성장하고 있습니다.
        
        복지:
        - 스톡옵션 제공
        - 유연근무제
        - 재택근무 가능
        """
    
    @pytest.fixture
    def sample_company_data(self):
        """Sample company data"""
        return {
            "id": "startup_1",
            "name": "테스트 스타트업",
            "employee_count": 30,
            "founded_year": 2020,
            "industry": "tech"
        }
    
    def test_validate_content_valid(self, validator, sample_content, sample_company_data):
        """Test validation of valid content"""
        result = validator.validate_content(sample_content, sample_company_data)
        
        assert result.is_valid
        assert result.confidence > 0.8
        assert len(result.violations) == 0
    
    def test_validate_content_with_hallucination(self, validator, sample_company_data):
        """Test validation of content with hallucinations"""
        bad_content = """
        우리는 업계 최고의 기업입니다!
        국내 유일의 기술을 보유하고 있습니다.
        300% 성장률을 기록했습니다.
        직원 수는 300명입니다.
        """
        
        result = validator.validate_content(bad_content, sample_company_data)
        
        assert not result.is_valid
        assert result.confidence < 0.5
        assert len(result.violations) > 0
        assert any("최고" in v for v in result.violations)
        assert any("Employee count" in v for v in result.violations)
    
    def test_check_forbidden_patterns(self, validator):
        """Test forbidden pattern detection"""
        content = "우리는 업계 최초이자 유일한 기업입니다. 여성 우대합니다."
        
        violations = validator._check_forbidden_patterns(content)
        
        assert len(violations) > 0
        assert any("최초" in v for v in violations)
        assert any("여성" in v for v in violations)
    
    def test_check_consistency(self, validator, topology):
        """Test topological consistency check"""
        content = "우리 스타트업은 대기업 수준의 복지를 제공합니다."
        
        result = validator.check_consistency(content, "startup_1")
        
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.confidence, float)
        
        # May have violations due to inconsistent claims
        if not result.is_valid:
            assert len(result.violations) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])