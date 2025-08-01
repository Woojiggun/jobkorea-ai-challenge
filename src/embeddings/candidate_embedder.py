"""
Candidate-specific embedder implementation
"""
from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime
import logging

from .base_embedder import BaseEmbedder

logger = logging.getLogger(__name__)


class CandidateEmbedder(BaseEmbedder):
    """Embedder for candidate/job seeker information"""
    
    def prepare_text(self, data: Dict[str, Any]) -> str:
        """
        Prepare candidate data for text embedding
        
        Args:
            data: Candidate data dictionary
            
        Returns:
            Formatted text string
        """
        text_parts = []
        
        # Basic information
        if "title" in data:
            text_parts.append(f"현재 직무: {data['title']}")
            
        if "desired_position" in data:
            text_parts.append(f"희망 직무: {data['desired_position']}")
            
        # Skills and expertise
        if "skills" in data:
            if isinstance(data["skills"], list):
                skills_text = ", ".join(data["skills"])
            else:
                skills_text = data["skills"]
            text_parts.append(f"보유 기술: {skills_text}")
            
        # Experience summary
        if "experience_summary" in data:
            text_parts.append(f"경력 요약: {data['experience_summary']}")
            
        # Education
        if "education" in data:
            text_parts.append(f"학력: {data['education']}")
            
        # Career interests
        if "interests" in data:
            if isinstance(data["interests"], list):
                interests_text = ", ".join(data["interests"])
            else:
                interests_text = data["interests"]
            text_parts.append(f"관심 분야: {interests_text}")
            
        # Work preferences
        if "work_preferences" in data:
            text_parts.append(f"근무 선호사항: {data['work_preferences']}")
            
        return " ".join(text_parts)
    
    def extract_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract numerical features from candidate data
        
        Args:
            data: Candidate data dictionary
            
        Returns:
            Dictionary of feature values
        """
        features = {}
        
        # Experience features
        if "years_experience" in data:
            features["experience_years"] = float(data["years_experience"])
            features["seniority_score"] = self._calculate_seniority_score(
                data["years_experience"]
            )
            
        # Education level
        if "education_level" in data:
            features["education_score"] = self._education_to_score(
                data["education_level"]
            )
            
        # Skills proficiency
        if "skill_count" in data:
            features["skill_diversity"] = float(data["skill_count"])
            
        if "primary_skill_years" in data:
            features["expertise_depth"] = float(data["primary_skill_years"])
            
        # Career progression (potential indicators)
        if "job_changes" in data:
            features["mobility_score"] = self._calculate_mobility_score(
                data["job_changes"], 
                data.get("years_experience", 5)
            )
            
        if "promotion_count" in data:
            features["growth_rate"] = float(data["promotion_count"]) / max(
                data.get("years_experience", 1), 1
            )
            
        # Project complexity
        if "project_scale" in data:
            features["project_complexity"] = self._project_scale_to_score(
                data["project_scale"]
            )
            
        # Certifications and achievements
        if "certification_count" in data:
            features["certification_score"] = min(
                float(data["certification_count"]) / 5, 1.0
            )
            
        # Salary expectations (normalized)
        if "expected_salary" in data and "current_salary" in data:
            features["salary_growth_expectation"] = (
                data["expected_salary"] - data["current_salary"]
            ) / max(data["current_salary"], 1)
            
        # Remote work preference
        if "remote_preference" in data:
            features["remote_preference_score"] = float(data["remote_preference"])
            
        return features
    
    def _calculate_seniority_score(self, years: float) -> float:
        """
        Calculate seniority score based on years of experience
        
        Args:
            years: Years of experience
            
        Returns:
            Seniority score (0-1)
        """
        if years < 1:
            return 0.0  # Entry level
        elif years < 3:
            return 0.2  # Junior
        elif years < 5:
            return 0.4  # Mid-level
        elif years < 8:
            return 0.6  # Senior
        elif years < 12:
            return 0.8  # Lead/Principal
        else:
            return 1.0  # Executive/Expert
    
    def _education_to_score(self, education_level: str) -> float:
        """
        Convert education level to numerical score
        
        Args:
            education_level: Education level string
            
        Returns:
            Education score (0-1)
        """
        education_scores = {
            "high_school": 0.2,
            "associate": 0.4,
            "bachelor": 0.6,
            "master": 0.8,
            "phd": 1.0,
            "bootcamp": 0.5,
            "self_taught": 0.3
        }
        
        return education_scores.get(education_level.lower(), 0.5)
    
    def _calculate_mobility_score(self, job_changes: int, years: float) -> float:
        """
        Calculate job mobility score (normalized job changes)
        
        Args:
            job_changes: Number of job changes
            years: Total years of experience
            
        Returns:
            Mobility score (0-1)
        """
        # Average job tenure
        avg_tenure = years / max(job_changes + 1, 1)
        
        # Normalize: 1-2 years = high mobility, 5+ years = low mobility
        if avg_tenure < 1:
            return 1.0  # Very high mobility
        elif avg_tenure < 2:
            return 0.8
        elif avg_tenure < 3:
            return 0.6
        elif avg_tenure < 5:
            return 0.4
        else:
            return 0.2  # Low mobility
    
    def _project_scale_to_score(self, scale: str) -> float:
        """
        Convert project scale to complexity score
        
        Args:
            scale: Project scale descriptor
            
        Returns:
            Complexity score (0-1)
        """
        scale_scores = {
            "personal": 0.1,
            "team": 0.3,
            "department": 0.5,
            "company": 0.7,
            "enterprise": 0.9,
            "global": 1.0
        }
        
        return scale_scores.get(scale.lower(), 0.5)
    
    def calculate_potential_score(self, data: Dict[str, Any]) -> float:
        """
        Calculate candidate's potential score based on growth indicators
        
        Args:
            data: Candidate data
            
        Returns:
            Potential score (0-1)
        """
        potential_factors = []
        
        # Learning velocity
        if "years_experience" in data and "skill_count" in data:
            learning_rate = data["skill_count"] / max(data["years_experience"], 1)
            potential_factors.append(min(learning_rate / 5, 1.0))
            
        # Career progression speed
        if "promotion_count" in data and "years_experience" in data:
            promotion_rate = data["promotion_count"] / max(data["years_experience"], 1)
            potential_factors.append(min(promotion_rate * 2, 1.0))
            
        # Education ambition
        if "continuing_education" in data:
            potential_factors.append(1.0 if data["continuing_education"] else 0.5)
            
        # Project complexity growth
        if "project_complexity_trend" in data:
            potential_factors.append(data["project_complexity_trend"])
            
        # Average all factors
        if potential_factors:
            return sum(potential_factors) / len(potential_factors)
        else:
            return 0.5  # Default middle score
    
    def create_candidate_profile(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a comprehensive candidate profile with embeddings
        
        Args:
            data: Raw candidate data
            
        Returns:
            Candidate profile with embeddings and metadata
        """
        # Generate embedding
        embedding = self.embed(data)
        
        # Extract features
        features = self.extract_features(data)
        
        # Calculate additional scores
        potential_score = self.calculate_potential_score(data)
        
        # Create profile
        profile = {
            "candidate_id": data.get("id", "unknown"),
            "current_title": data.get("title", "Unknown"),
            "desired_position": data.get("desired_position", "Unknown"),
            "experience_years": data.get("years_experience", 0),
            "embedding": embedding,
            "features": features,
            "spec_score": features.get("experience_years", 0) * 0.4 + 
                         features.get("education_score", 0) * 0.3 +
                         features.get("skill_diversity", 0) * 0.3,
            "potential_score": potential_score,
            "timestamp": datetime.now().isoformat()
        }
        
        return profile