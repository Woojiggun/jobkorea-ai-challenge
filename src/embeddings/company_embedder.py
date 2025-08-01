"""
Company-specific embedder implementation
"""
from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime
import logging

from .base_embedder import BaseEmbedder

logger = logging.getLogger(__name__)


class CompanyEmbedder(BaseEmbedder):
    """Embedder for company information"""
    
    def prepare_text(self, data: Dict[str, Any]) -> str:
        """
        Prepare company data for text embedding
        
        Args:
            data: Company data dictionary
            
        Returns:
            Formatted text string
        """
        text_parts = []
        
        # Company name and basic info
        if "name" in data:
            text_parts.append(f"회사명: {data['name']}")
            
        if "industry" in data:
            text_parts.append(f"업종: {data['industry']}")
            
        if "description" in data:
            text_parts.append(f"회사 소개: {data['description']}")
            
        # Culture and values
        if "culture" in data:
            text_parts.append(f"기업 문화: {data['culture']}")
            
        if "mission" in data:
            text_parts.append(f"미션: {data['mission']}")
            
        # Benefits and perks
        if "benefits" in data:
            if isinstance(data["benefits"], list):
                benefits_text = ", ".join(data["benefits"])
            else:
                benefits_text = data["benefits"]
            text_parts.append(f"복지: {benefits_text}")
            
        # Work environment
        if "work_style" in data:
            text_parts.append(f"근무 방식: {data['work_style']}")
            
        if "location" in data:
            text_parts.append(f"위치: {data['location']}")
            
        return " ".join(text_parts)
    
    def extract_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract numerical features from company data
        
        Args:
            data: Company data dictionary
            
        Returns:
            Dictionary of feature values
        """
        features = {}
        
        # Company size features
        if "employee_count" in data:
            features["employee_count"] = float(data["employee_count"])
            features["size_category"] = self._categorize_size(data["employee_count"])
            
        # Financial features
        if "revenue" in data:
            features["revenue"] = float(data["revenue"])
            
        if "growth_rate" in data:
            features["growth_rate"] = float(data["growth_rate"])
            
        if "funding_stage" in data:
            features["funding_score"] = self._funding_to_score(data["funding_stage"])
            
        # Age and stability
        if "founded_year" in data:
            current_year = datetime.now().year
            features["company_age"] = float(current_year - data["founded_year"])
            
        # Culture and satisfaction scores
        if "employee_satisfaction" in data:
            features["satisfaction_score"] = float(data["employee_satisfaction"])
            
        if "glassdoor_rating" in data:
            features["glassdoor_rating"] = float(data["glassdoor_rating"])
            
        # Work style features
        if "remote_work" in data:
            features["remote_work_score"] = 1.0 if data["remote_work"] else 0.0
            
        if "flexible_hours" in data:
            features["flexibility_score"] = 1.0 if data["flexible_hours"] else 0.0
            
        # Industry position
        if "market_position" in data:
            features["market_position"] = float(data["market_position"])
            
        return features
    
    def _categorize_size(self, employee_count: int) -> float:
        """
        Categorize company size into numerical score
        
        Args:
            employee_count: Number of employees
            
        Returns:
            Size category score (0-1)
        """
        if employee_count < 10:
            return 0.0  # Startup
        elif employee_count < 50:
            return 0.2  # Small
        elif employee_count < 200:
            return 0.4  # Medium
        elif employee_count < 1000:
            return 0.6  # Large
        elif employee_count < 5000:
            return 0.8  # Enterprise
        else:
            return 1.0  # Mega corp
    
    def _funding_to_score(self, funding_stage: str) -> float:
        """
        Convert funding stage to numerical score
        
        Args:
            funding_stage: Funding stage string
            
        Returns:
            Funding score (0-1)
        """
        funding_scores = {
            "pre_seed": 0.1,
            "seed": 0.2,
            "series_a": 0.3,
            "series_b": 0.4,
            "series_c": 0.5,
            "series_d": 0.6,
            "series_e+": 0.7,
            "pre_ipo": 0.8,
            "ipo": 0.9,
            "public": 1.0
        }
        
        return funding_scores.get(funding_stage.lower(), 0.5)
    
    def create_company_profile(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a comprehensive company profile with embeddings
        
        Args:
            data: Raw company data
            
        Returns:
            Company profile with embeddings and metadata
        """
        # Generate embedding
        embedding = self.embed(data)
        
        # Extract key metadata
        profile = {
            "company_id": data.get("id", "unknown"),
            "name": data.get("name", "Unknown Company"),
            "industry": data.get("industry", "Unknown"),
            "size": data.get("employee_count", 0),
            "embedding": embedding,
            "features": self.extract_features(data),
            "timestamp": datetime.now().isoformat()
        }
        
        return profile
    
    def batch_embed_companies(
        self, 
        companies: List[Dict[str, Any]], 
        save_path: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Embed multiple companies efficiently
        
        Args:
            companies: List of company data dictionaries
            save_path: Optional path to save embeddings
            
        Returns:
            Dictionary mapping company IDs to embeddings
        """
        logger.info(f"Embedding {len(companies)} companies...")
        
        # Batch embed
        embeddings = self.batch_embed(companies)
        
        # Create mapping
        company_embeddings = {}
        for i, company in enumerate(companies):
            company_id = company.get("id", f"company_{i}")
            company_embeddings[company_id] = embeddings[i]
            
        # Save if requested
        if save_path:
            self.save_embeddings(embeddings, save_path)
            
        logger.info(f"Successfully embedded {len(companies)} companies")
        
        return company_embeddings