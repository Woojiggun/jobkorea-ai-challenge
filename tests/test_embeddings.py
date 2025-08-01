"""
Tests for embedding system
"""
import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.embeddings import CompanyEmbedder, CandidateEmbedder


class TestCompanyEmbedder:
    """Test company embedder functionality"""
    
    @pytest.fixture
    def embedder(self):
        """Create company embedder instance"""
        return CompanyEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    @pytest.fixture
    def sample_company(self):
        """Sample company data"""
        return {
            "id": "test_company_1",
            "name": "테스트 기업",
            "industry": "IT/소프트웨어",
            "employee_count": 150,
            "founded_year": 2020,
            "location": "서울 강남구",
            "culture": "자율적이고 창의적인 문화",
            "benefits": ["재택근무", "유연근무", "스톡옵션", "교육지원"],
            "remote_work": True,
            "growth_rate": 45.5,
            "description": "AI 기반 솔루션을 개발하는 스타트업"
        }
    
    def test_prepare_text(self, embedder, sample_company):
        """Test text preparation from company data"""
        text = embedder.prepare_text(sample_company)
        
        assert isinstance(text, str)
        assert "테스트 기업" in text
        assert "IT/소프트웨어" in text
        assert "재택근무" in text
    
    def test_extract_features(self, embedder, sample_company):
        """Test feature extraction"""
        features = embedder.extract_features(sample_company)
        
        assert isinstance(features, dict)
        assert "employee_count" in features
        assert features["employee_count"] == 150.0
        assert "growth_rate" in features
        assert features["growth_rate"] == 45.5
        assert "company_age" in features
        assert features["company_age"] >= 4  # 2024 - 2020
    
    def test_embed_single_company(self, embedder, sample_company):
        """Test embedding generation for single company"""
        embedding = embedder.embed(sample_company)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] > 0  # Has dimensions
        assert not np.isnan(embedding).any()  # No NaN values
    
    def test_embed_multiple_companies(self, embedder, sample_company):
        """Test batch embedding"""
        companies = [sample_company, sample_company.copy()]
        companies[1]["id"] = "test_company_2"
        companies[1]["name"] = "다른 기업"
        
        embeddings = embedder.embed(companies)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 2
        assert not np.array_equal(embeddings[0], embeddings[1])  # Different embeddings
    
    def test_create_company_profile(self, embedder, sample_company):
        """Test company profile creation"""
        profile = embedder.create_company_profile(sample_company)
        
        assert "company_id" in profile
        assert profile["company_id"] == "test_company_1"
        assert "embedding" in profile
        assert "features" in profile
        assert "timestamp" in profile


class TestCandidateEmbedder:
    """Test candidate embedder functionality"""
    
    @pytest.fixture
    def embedder(self):
        """Create candidate embedder instance"""
        return CandidateEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    @pytest.fixture
    def sample_candidate(self):
        """Sample candidate data"""
        return {
            "id": "test_candidate_1",
            "title": "백엔드 개발자",
            "years_experience": 5,
            "skills": ["Python", "Django", "FastAPI", "PostgreSQL", "Docker"],
            "education_level": "bachelor",
            "desired_position": "시니어 백엔드 개발자",
            "remote_preference": 0.8,
            "expected_salary": 7000,
            "current_salary": 5500,
            "interests": ["마이크로서비스", "클라우드", "DevOps"],
            "skill_count": 12,
            "job_changes": 2,
            "promotion_count": 2
        }
    
    def test_prepare_text(self, embedder, sample_candidate):
        """Test text preparation from candidate data"""
        text = embedder.prepare_text(sample_candidate)
        
        assert isinstance(text, str)
        assert "백엔드 개발자" in text
        assert "Python" in text
        assert "마이크로서비스" in text
    
    def test_extract_features(self, embedder, sample_candidate):
        """Test feature extraction"""
        features = embedder.extract_features(sample_candidate)
        
        assert isinstance(features, dict)
        assert "experience_years" in features
        assert features["experience_years"] == 5.0
        assert "seniority_score" in features
        assert 0 <= features["seniority_score"] <= 1.0
        assert "education_score" in features
    
    def test_calculate_potential_score(self, embedder, sample_candidate):
        """Test potential score calculation"""
        score = embedder.calculate_potential_score(sample_candidate)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1.0
    
    def test_embed_candidate(self, embedder, sample_candidate):
        """Test candidate embedding generation"""
        embedding = embedder.embed(sample_candidate)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] > 0
        assert not np.isnan(embedding).any()
    
    def test_create_candidate_profile(self, embedder, sample_candidate):
        """Test candidate profile creation"""
        profile = embedder.create_candidate_profile(sample_candidate)
        
        assert "candidate_id" in profile
        assert profile["candidate_id"] == "test_candidate_1"
        assert "embedding" in profile
        assert "spec_score" in profile
        assert "potential_score" in profile
        assert 0 <= profile["spec_score"] <= 1.0
        assert 0 <= profile["potential_score"] <= 1.0


class TestEmbeddingIntegration:
    """Test integration between embedders"""
    
    @pytest.fixture
    def company_embedder(self):
        return CompanyEmbedder()
    
    @pytest.fixture
    def candidate_embedder(self):
        return CandidateEmbedder()
    
    def test_embedding_compatibility(
        self, 
        company_embedder, 
        candidate_embedder,
        sample_company,
        sample_candidate
    ):
        """Test that embeddings are compatible for matching"""
        company_emb = company_embedder.embed(sample_company)
        candidate_emb = candidate_embedder.embed(sample_candidate)
        
        # Should have compatible dimensions for similarity calculation
        assert company_emb.shape[0] == candidate_emb.shape[0]
        
        # Can calculate cosine similarity
        similarity = np.dot(company_emb, candidate_emb) / (
            np.linalg.norm(company_emb) * np.linalg.norm(candidate_emb)
        )
        
        assert -1 <= similarity <= 1  # Valid cosine similarity


if __name__ == "__main__":
    pytest.main([__file__, "-v"])