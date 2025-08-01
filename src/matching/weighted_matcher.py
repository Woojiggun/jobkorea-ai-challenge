"""
Weighted matching system for job recommendations
"""
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
import logging

from src.embeddings.company_embedder import CompanyEmbedder
from src.embeddings.candidate_embedder import CandidateEmbedder
from src.topology.topology_mapper import TopologyMapper
from src.topology.gravity_field import GravityField

logger = logging.getLogger(__name__)


@dataclass
class MatchScore:
    """Detailed match score with breakdown"""
    total_score: float
    spec_match: float
    potential_match: float
    culture_match: float
    trajectory_match: float
    breakdown: Dict[str, float]
    explanation: str


@dataclass
class MatchResult:
    """Result of matching operation"""
    company_id: str
    candidate_id: str
    score: MatchScore
    compatibility_factors: Dict[str, Any]
    recommendations: List[str]


class WeightedMatcher:
    """
    Implements weighted matching between companies and candidates
    """
    
    def __init__(
        self,
        company_embedder: CompanyEmbedder,
        candidate_embedder: CandidateEmbedder,
        topology: TopologyMapper,
        gravity_field: GravityField
    ):
        """
        Initialize weighted matcher
        
        Args:
            company_embedder: Company embedder instance
            candidate_embedder: Candidate embedder instance
            topology: Topology mapper instance
            gravity_field: Gravity field instance
        """
        self.company_embedder = company_embedder
        self.candidate_embedder = candidate_embedder
        self.topology = topology
        self.gravity_field = gravity_field
        
        # Default weight configurations
        self.weights = {
            "spec_match": 0.3,
            "potential_match": 0.3,
            "culture_match": 0.2,
            "trajectory_match": 0.2
        }
        
    def match(
        self,
        company_data: Dict[str, Any],
        candidate_data: Dict[str, Any],
        custom_weights: Optional[Dict[str, float]] = None
    ) -> MatchResult:
        """
        Calculate match between company and candidate
        
        Args:
            company_data: Company information
            candidate_data: Candidate information
            custom_weights: Optional custom weight configuration
            
        Returns:
            MatchResult with detailed scoring
        """
        # Use custom weights if provided
        weights = custom_weights or self.weights
        
        # Generate embeddings
        company_embedding = self.company_embedder.embed(company_data)
        candidate_embedding = self.candidate_embedder.embed(candidate_data)
        
        # Calculate individual match components
        spec_match = self._calculate_spec_match(company_data, candidate_data)
        potential_match = self._calculate_potential_match(company_data, candidate_data)
        culture_match = self._calculate_culture_match(
            company_embedding, candidate_embedding, company_data, candidate_data
        )
        trajectory_match = self._calculate_trajectory_match(
            company_data, candidate_data
        )
        
        # Calculate weighted total score
        total_score = (
            weights["spec_match"] * spec_match +
            weights["potential_match"] * potential_match +
            weights["culture_match"] * culture_match +
            weights["trajectory_match"] * trajectory_match
        )
        
        # Create detailed breakdown
        breakdown = {
            "spec_components": self._get_spec_breakdown(company_data, candidate_data),
            "potential_components": self._get_potential_breakdown(candidate_data),
            "culture_components": self._get_culture_breakdown(company_data, candidate_data),
            "trajectory_components": self._get_trajectory_breakdown(company_data, candidate_data)
        }
        
        # Generate explanation
        explanation = self._generate_match_explanation(
            spec_match, potential_match, culture_match, trajectory_match, breakdown
        )
        
        # Create match score
        score = MatchScore(
            total_score=total_score,
            spec_match=spec_match,
            potential_match=potential_match,
            culture_match=culture_match,
            trajectory_match=trajectory_match,
            breakdown=breakdown,
            explanation=explanation
        )
        
        # Identify compatibility factors
        compatibility_factors = self._identify_compatibility_factors(
            company_data, candidate_data, score
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            company_data, candidate_data, score
        )
        
        return MatchResult(
            company_id=company_data.get("id", "unknown"),
            candidate_id=candidate_data.get("id", "unknown"),
            score=score,
            compatibility_factors=compatibility_factors,
            recommendations=recommendations
        )
    
    def _calculate_spec_match(
        self,
        company_data: Dict[str, Any],
        candidate_data: Dict[str, Any]
    ) -> float:
        """Calculate specification match score"""
        score_components = []
        
        # Experience match
        required_exp = company_data.get("required_experience", 0)
        candidate_exp = candidate_data.get("years_experience", 0)
        
        if required_exp > 0:
            exp_ratio = candidate_exp / required_exp
            if exp_ratio >= 1.0:
                exp_score = 1.0
            elif exp_ratio >= 0.8:
                exp_score = 0.9
            elif exp_ratio >= 0.6:
                exp_score = 0.7
            else:
                exp_score = exp_ratio
            score_components.append(exp_score)
        
        # Skills match
        required_skills = set(company_data.get("required_skills", []))
        candidate_skills = set(candidate_data.get("skills", []))
        
        if required_skills:
            skill_overlap = len(required_skills & candidate_skills)
            skill_score = skill_overlap / len(required_skills)
            score_components.append(skill_score)
        
        # Education match
        required_edu = company_data.get("required_education", "bachelor")
        candidate_edu = candidate_data.get("education_level", "bachelor")
        
        edu_levels = {
            "high_school": 1,
            "associate": 2,
            "bachelor": 3,
            "master": 4,
            "phd": 5
        }
        
        if candidate_edu in edu_levels and required_edu in edu_levels:
            if edu_levels[candidate_edu] >= edu_levels[required_edu]:
                edu_score = 1.0
            else:
                edu_score = edu_levels[candidate_edu] / edu_levels[required_edu]
            score_components.append(edu_score)
        
        # Average all components
        return np.mean(score_components) if score_components else 0.5
    
    def _calculate_potential_match(
        self,
        company_data: Dict[str, Any],
        candidate_data: Dict[str, Any]
    ) -> float:
        """Calculate potential match score"""
        # Get candidate's potential score
        candidate_potential = self.candidate_embedder.calculate_potential_score(
            candidate_data
        )
        
        # Adjust based on company's growth stage
        company_stage = company_data.get("growth_stage", "stable")
        
        stage_multipliers = {
            "startup": 1.5,      # High potential valued more
            "growth": 1.3,
            "stable": 1.0,
            "mature": 0.8        # Experience valued more than potential
        }
        
        multiplier = stage_multipliers.get(company_stage, 1.0)
        
        # Consider company's innovation focus
        if company_data.get("innovation_focused", False):
            multiplier *= 1.2
            
        return min(candidate_potential * multiplier, 1.0)
    
    def _calculate_culture_match(
        self,
        company_embedding: np.ndarray,
        candidate_embedding: np.ndarray,
        company_data: Dict[str, Any],
        candidate_data: Dict[str, Any]
    ) -> float:
        """Calculate cultural fit score"""
        # Embedding similarity (cosine similarity)
        embedding_similarity = np.dot(company_embedding, candidate_embedding) / (
            np.linalg.norm(company_embedding) * np.linalg.norm(candidate_embedding)
        )
        
        # Normalize to [0, 1]
        embedding_score = (embedding_similarity + 1) / 2
        
        # Work style preferences
        work_style_score = 0.5  # Default
        
        company_remote = company_data.get("remote_work", False)
        candidate_remote_pref = candidate_data.get("remote_preference", 0.5)
        
        if company_remote:
            work_style_score = candidate_remote_pref
        else:
            work_style_score = 1.0 - candidate_remote_pref * 0.5
            
        # Company size preference
        size_score = self._calculate_size_preference_match(
            company_data, candidate_data
        )
        
        # Combine scores
        culture_score = (
            embedding_score * 0.5 +
            work_style_score * 0.3 +
            size_score * 0.2
        )
        
        return culture_score
    
    def _calculate_size_preference_match(
        self,
        company_data: Dict[str, Any],
        candidate_data: Dict[str, Any]
    ) -> float:
        """Calculate match based on company size preferences"""
        company_size = company_data.get("employee_count", 100)
        preferred_size = candidate_data.get("preferred_company_size", "medium")
        
        size_ranges = {
            "startup": (1, 50),
            "small": (50, 200),
            "medium": (200, 1000),
            "large": (1000, 5000),
            "enterprise": (5000, float('inf'))
        }
        
        if preferred_size in size_ranges:
            min_size, max_size = size_ranges[preferred_size]
            if min_size <= company_size <= max_size:
                return 1.0
            elif company_size < min_size:
                return company_size / min_size
            else:
                return max_size / company_size
        
        return 0.5  # Default if preference unknown
    
    def _calculate_trajectory_match(
        self,
        company_data: Dict[str, Any],
        candidate_data: Dict[str, Any]
    ) -> float:
        """Calculate career trajectory alignment"""
        # Check if we have topology nodes for both
        company_id = company_data.get("id")
        candidate_id = candidate_data.get("id")
        
        if not (company_id in self.topology.nodes and 
                candidate_id in self.topology.nodes):
            # Fallback to simple calculation
            return self._simple_trajectory_match(company_data, candidate_data)
        
        # Get current positions in topological space
        company_node = self.topology.nodes[company_id]
        candidate_node = self.topology.nodes[candidate_id]
        
        # Calculate topological distance
        topo_distance = self.topology.calculate_topological_distance(
            candidate_id, company_id
        )
        
        if topo_distance == float('inf'):
            # Not connected in topology
            return 0.2
        
        # Convert distance to score (closer = better)
        distance_score = 1.0 / (1.0 + topo_distance)
        
        # Check if company is on candidate's likely path
        if company_node.embedding is not None and candidate_node.embedding is not None:
            # Predict candidate's trajectory
            trajectory = self.gravity_field.predict_trajectory(
                candidate_node.embedding[:3],
                np.zeros(3),  # Initial velocity
                time_steps=5
            )
            
            # Check if trajectory passes near company
            min_distance = float('inf')
            company_pos = company_node.embedding[:3]
            
            for pos in trajectory:
                distance = np.linalg.norm(pos - company_pos)
                min_distance = min(min_distance, distance)
            
            # Convert to score
            trajectory_score = 1.0 / (1.0 + min_distance)
            
            # Combine scores
            return (distance_score + trajectory_score) / 2
        
        return distance_score
    
    def _simple_trajectory_match(
        self,
        company_data: Dict[str, Any],
        candidate_data: Dict[str, Any]
    ) -> float:
        """Simple trajectory match without topology"""
        # Check career progression alignment
        candidate_level = candidate_data.get("seniority_level", "mid")
        position_level = company_data.get("position_level", "mid")
        
        level_map = {
            "entry": 0,
            "junior": 1,
            "mid": 2,
            "senior": 3,
            "lead": 4,
            "executive": 5
        }
        
        candidate_num = level_map.get(candidate_level, 2)
        position_num = level_map.get(position_level, 2)
        
        # Best match is next level up
        if position_num == candidate_num + 1:
            return 1.0
        elif position_num == candidate_num:
            return 0.8
        elif position_num == candidate_num + 2:
            return 0.6
        elif position_num < candidate_num:
            return 0.4  # Overqualified
        else:
            return 0.2  # Too big a jump
    
    def _get_spec_breakdown(
        self,
        company_data: Dict[str, Any],
        candidate_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Get detailed spec match breakdown"""
        breakdown = {}
        
        # Experience
        required_exp = company_data.get("required_experience", 0)
        candidate_exp = candidate_data.get("years_experience", 0)
        if required_exp > 0:
            breakdown["experience_match"] = min(candidate_exp / required_exp, 1.0)
        
        # Skills
        required_skills = set(company_data.get("required_skills", []))
        candidate_skills = set(candidate_data.get("skills", []))
        if required_skills:
            breakdown["skill_coverage"] = len(required_skills & candidate_skills) / len(required_skills)
            breakdown["extra_skills"] = len(candidate_skills - required_skills)
        
        return breakdown
    
    def _get_potential_breakdown(
        self,
        candidate_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Get detailed potential breakdown"""
        return {
            "learning_velocity": candidate_data.get("learning_velocity", 0.5),
            "career_progression": candidate_data.get("promotion_rate", 0.5),
            "skill_diversity": min(candidate_data.get("skill_count", 5) / 10, 1.0),
            "ambition_score": candidate_data.get("ambition_score", 0.5)
        }
    
    def _get_culture_breakdown(
        self,
        company_data: Dict[str, Any],
        candidate_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Get detailed culture match breakdown"""
        breakdown = {}
        
        # Work style
        if company_data.get("remote_work"):
            breakdown["remote_alignment"] = candidate_data.get("remote_preference", 0.5)
        else:
            breakdown["office_alignment"] = 1.0 - candidate_data.get("remote_preference", 0.5)
        
        # Company size preference
        breakdown["size_preference"] = self._calculate_size_preference_match(
            company_data, candidate_data
        )
        
        # Values alignment (simplified)
        company_values = set(company_data.get("values", []))
        candidate_values = set(candidate_data.get("values", []))
        if company_values and candidate_values:
            breakdown["values_overlap"] = len(company_values & candidate_values) / len(company_values)
        
        return breakdown
    
    def _get_trajectory_breakdown(
        self,
        company_data: Dict[str, Any],
        candidate_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Get detailed trajectory breakdown"""
        breakdown = {}
        
        # Career level progression
        candidate_level = candidate_data.get("seniority_level", "mid")
        position_level = company_data.get("position_level", "mid")
        
        level_map = {"entry": 0, "junior": 1, "mid": 2, "senior": 3, "lead": 4, "executive": 5}
        
        candidate_num = level_map.get(candidate_level, 2)
        position_num = level_map.get(position_level, 2)
        
        breakdown["level_progression"] = 1.0 - abs(position_num - candidate_num - 1) * 0.2
        breakdown["growth_alignment"] = company_data.get("growth_rate", 0) / 100
        
        return breakdown
    
    def _generate_match_explanation(
        self,
        spec_match: float,
        potential_match: float,
        culture_match: float,
        trajectory_match: float,
        breakdown: Dict[str, Any]
    ) -> str:
        """Generate human-readable match explanation"""
        explanations = []
        
        # Spec match explanation
        if spec_match >= 0.8:
            explanations.append("강력한 기술 및 경력 매칭")
        elif spec_match >= 0.6:
            explanations.append("적절한 기술 및 경력 수준")
        else:
            explanations.append("일부 기술 격차 존재")
        
        # Potential match explanation
        if potential_match >= 0.8:
            explanations.append("뛰어난 성장 잠재력")
        elif potential_match >= 0.6:
            explanations.append("양호한 성장 가능성")
        
        # Culture match explanation
        if culture_match >= 0.8:
            explanations.append("기업 문화에 매우 적합")
        elif culture_match >= 0.6:
            explanations.append("기업 문화와 양호한 궁합")
        
        # Trajectory match explanation
        if trajectory_match >= 0.8:
            explanations.append("커리어 경로상 이상적인 다음 단계")
        elif trajectory_match >= 0.6:
            explanations.append("커리어 발전에 적합한 기회")
        
        return " | ".join(explanations)
    
    def _identify_compatibility_factors(
        self,
        company_data: Dict[str, Any],
        candidate_data: Dict[str, Any],
        score: MatchScore
    ) -> Dict[str, Any]:
        """Identify key compatibility factors"""
        factors = {
            "strengths": [],
            "gaps": [],
            "opportunities": []
        }
        
        # Identify strengths
        if score.spec_match >= 0.8:
            factors["strengths"].append("기술 요구사항 충족")
        if score.culture_match >= 0.8:
            factors["strengths"].append("문화적 적합성 우수")
        if score.potential_match >= 0.8:
            factors["strengths"].append("높은 성장 잠재력")
        
        # Identify gaps
        if score.spec_match < 0.6:
            missing_skills = set(company_data.get("required_skills", [])) - set(candidate_data.get("skills", []))
            if missing_skills:
                factors["gaps"].append(f"필요 기술: {', '.join(list(missing_skills)[:3])}")
        
        # Identify opportunities
        if score.trajectory_match >= 0.7:
            factors["opportunities"].append("자연스러운 커리어 성장 경로")
        if candidate_data.get("skill_count", 0) > len(company_data.get("required_skills", [])):
            factors["opportunities"].append("추가 역량 활용 가능")
        
        return factors
    
    def _generate_recommendations(
        self,
        company_data: Dict[str, Any],
        candidate_data: Dict[str, Any],
        score: MatchScore
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # For company
        if score.spec_match < 0.7:
            recommendations.append("온보딩 프로그램 강화 필요")
        if score.potential_match >= 0.8:
            recommendations.append("성장 기회 및 챌린지 제공 중요")
        
        # For candidate
        if score.culture_match < 0.6:
            recommendations.append("기업 문화 사전 파악 권장")
        if score.trajectory_match >= 0.8:
            recommendations.append("장기적 커리어 목표와 부합")
        
        return recommendations
    
    def batch_match(
        self,
        companies: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        top_k: int = 10
    ) -> Dict[str, List[MatchResult]]:
        """
        Batch match multiple companies and candidates
        
        Args:
            companies: List of company data
            candidates: List of candidate data
            top_k: Number of top matches per entity
            
        Returns:
            Dictionary with top matches for each company and candidate
        """
        results = {
            "company_matches": {},
            "candidate_matches": {}
        }
        
        # Calculate all matches
        all_matches = []
        for company in companies:
            company_id = company.get("id", f"company_{companies.index(company)}")
            results["company_matches"][company_id] = []
            
            for candidate in candidates:
                match_result = self.match(company, candidate)
                all_matches.append(match_result)
                
        # Sort matches for each company
        for company in companies:
            company_id = company.get("id", f"company_{companies.index(company)}")
            company_matches = [m for m in all_matches if m.company_id == company_id]
            company_matches.sort(key=lambda x: x.score.total_score, reverse=True)
            results["company_matches"][company_id] = company_matches[:top_k]
        
        # Sort matches for each candidate
        for candidate in candidates:
            candidate_id = candidate.get("id", f"candidate_{candidates.index(candidate)}")
            candidate_matches = [m for m in all_matches if m.candidate_id == candidate_id]
            candidate_matches.sort(key=lambda x: x.score.total_score, reverse=True)
            results["candidate_matches"][candidate_id] = candidate_matches[:top_k]
        
        return results