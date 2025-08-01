"""
Tests for generation and validation system
"""
import pytest
import asyncio
from pathlib import Path
import sys
from unittest.mock import Mock, patch, AsyncMock

sys.path.append(str(Path(__file__).parent.parent))

from src.generation import HallucinationGuard
from src.validation import FactChecker, ConsistencyValidator
from src.topology import BoundaryValidator, TopologyMapper


class TestHallucinationGuard:
    """Test hallucination guard functionality"""
    
    @pytest.fixture
    def boundary_validator(self):
        """Create mock boundary validator"""
        topology = TopologyMapper()
        return BoundaryValidator(topology)
    
    @pytest.fixture
    def guard(self, boundary_validator):
        """Create hallucination guard instance"""
        return HallucinationGuard(boundary_validator)
    
    @pytest.fixture
    def sample_content(self):
        """Sample generated content"""
        return """
        테크 기업 - 백엔드 개발자 채용
        
        우리 회사는 2020년에 설립된 스타트업으로 현재 50명의 직원과 함께
        연 30% 성장률을 기록하고 있습니다.
        
        주요 업무:
        - Python 기반 백엔드 개발
        - 마이크로서비스 아키텍처 설계
        
        필요 경력: 3년 이상
        """
    
    @pytest.fixture
    def sample_source_data(self):
        """Sample source data"""
        return {
            "name": "테크 기업",
            "founded_year": 2020,
            "employee_count": 50,
            "growth_rate": 30,
            "required_experience": 3
        }
    
    def test_check_hallucination_clean(self, guard, sample_content, sample_source_data):
        """Test checking clean content without hallucinations"""
        result = guard.check_hallucination(sample_content, sample_source_data)
        
        assert not result.has_hallucination
        assert result.hallucination_score < 0.2
        assert result.confidence > 0.8
    
    def test_check_hallucination_with_issues(self, guard, sample_source_data):
        """Test detecting hallucinations"""
        bad_content = """
        우리는 업계 최고의 기업입니다!
        2015년 설립 이후 500% 성장했습니다.
        현재 500명의 직원이 있습니다.
        곧 나스닥 상장 예정입니다.
        """
        
        result = guard.check_hallucination(bad_content, sample_source_data)
        
        assert result.has_hallucination
        assert result.hallucination_score > 0.5
        assert len(result.detected_issues) > 0
        
        # Check specific issues
        issue_types = [issue["type"] for issue in result.detected_issues]
        assert "pattern_hallucination" in issue_types
        assert "factual_error" in issue_types
    
    def test_pattern_detection(self, guard):
        """Test pattern-based hallucination detection"""
        content = "최고의 복지! 무한한 성장 가능성! 업계 1위!"
        
        issues = guard._detect_pattern_hallucinations(content)
        
        assert len(issues) > 0
        assert any(issue["category"] == "absolute_claims" for issue in issues)
        assert any(issue["category"] == "vague_benefits" for issue in issues)
    
    def test_fact_verification(self, guard, sample_source_data):
        """Test fact verification"""
        content = "우리 회사는 100명의 직원과 2018년에 설립되었습니다."
        
        issues = guard._verify_facts(content, sample_source_data)
        
        assert len(issues) > 0
        assert any("employee_count" in issue.get("claim", {}).get("type", "") 
                  for issue in issues)
        assert any("founded_year" in issue.get("claim", {}).get("type", "")
                  for issue in issues)
    
    def test_apply_corrections(self, guard):
        """Test applying corrections to content"""
        content = "우리 회사는 100명의 직원이 있습니다."
        corrections = [
            {
                "original": "100명",
                "corrected": "50명",
                "reason": "Incorrect employee count"
            }
        ]
        
        corrected = guard.apply_corrections(content, corrections)
        
        assert "50명" in corrected
        assert "100명" not in corrected
    
    def test_generate_safe_content(self, guard):
        """Test safe content generation"""
        template = "{company_name}는 {employee_count}명의 직원과 함께하는 {industry} 기업입니다."
        data = {
            "company_name": "테스트 기업",
            "employee_count": 50,
            "industry": "IT"
        }
        
        safe_content = guard.generate_safe_content(template, data, strict_mode=True)
        
        assert "테스트 기업" in safe_content
        assert "50명" in safe_content
        assert "IT" in safe_content
        assert "{" not in safe_content  # No remaining placeholders


class TestFactChecker:
    """Test fact checker functionality"""
    
    @pytest.fixture
    def checker(self):
        """Create fact checker instance"""
        return FactChecker()
    
    @pytest.fixture
    def sample_content(self):
        """Sample content with facts"""
        return """
        테크 기업은 2020년에 설립되어 현재 50명의 직원이 근무하고 있습니다.
        서울 강남구에 위치하며, 연 30% 성장률을 보이고 있습니다.
        5년 이상 경력자를 찾고 있으며, 팀 규모는 10명입니다.
        """
    
    @pytest.fixture
    def source_data(self):
        """Source data for verification"""
        return {
            "founded_year": 2020,
            "employee_count": 50,
            "location": "서울 강남구",
            "growth_rate": 30,
            "required_experience": 5,
            "team_size": 10
        }
    
    def test_extract_facts(self, checker, sample_content):
        """Test fact extraction"""
        facts = checker._extract_facts(sample_content)
        
        assert len(facts) > 0
        
        fact_types = [f["type"] for f in facts]
        assert "employee_count" in fact_types
        assert "founded_year" in fact_types
        assert "growth_rate" in fact_types
        assert "experience_requirement" in fact_types
    
    def test_check_facts_valid(self, checker, sample_content, source_data):
        """Test fact checking with valid content"""
        result = checker.check_facts(sample_content, source_data)
        
        assert result.is_valid
        assert result.facts_verified > 0
        assert result.facts_failed == 0
        assert result.confidence == 1.0
    
    def test_check_facts_invalid(self, checker, source_data):
        """Test fact checking with invalid facts"""
        bad_content = """
        우리 회사는 2018년에 설립되어 100명의 직원이 있습니다.
        연 50% 성장률을 기록하고 있습니다.
        """
        
        result = checker.check_facts(bad_content, source_data)
        
        assert not result.is_valid
        assert result.facts_failed > 0
        assert result.confidence < 1.0
    
    def test_cross_check_facts(self, checker):
        """Test cross-checking facts between sets"""
        facts1 = [
            {"type": "employee_count", "value": 50},
            {"type": "founded_year", "value": 2020}
        ]
        
        facts2 = [
            {"type": "employee_count", "value": 50},
            {"type": "founded_year", "value": 2021}  # Conflict
        ]
        
        result = checker.cross_check_facts(facts1, facts2)
        
        assert len(result["conflicts"]) > 0
        assert len(result["agreements"]) > 0
        assert result["conflict_rate"] > 0
    
    def test_suggest_corrections(self, checker, source_data):
        """Test fact correction suggestions"""
        failed_facts = [
            {
                "verified": False,
                "fact": {
                    "type": "employee_count",
                    "text": "100명",
                    "value": 100
                },
                "details": {"expected": 50}
            }
        ]
        
        corrections = checker.suggest_fact_corrections(failed_facts, source_data)
        
        assert len(corrections) > 0
        assert corrections[0]["original"] == "100명"
        assert corrections[0]["suggested"] == "50명"


class TestConsistencyValidator:
    """Test consistency validator functionality"""
    
    @pytest.fixture
    def validator(self):
        """Create consistency validator instance"""
        return ConsistencyValidator()
    
    @pytest.fixture
    def consistent_content(self):
        """Sample consistent content"""
        return """
        우리 스타트업은 30명의 직원과 함께 성장하고 있습니다.
        2020년 설립 이후 꾸준히 성장해왔습니다.
        재택근무와 유연근무제를 운영하고 있으며,
        스타트업 문화를 지향합니다.
        """
    
    @pytest.fixture
    def inconsistent_content(self):
        """Sample inconsistent content"""
        return """
        우리 회사는 50명의 직원이 있습니다.
        신입 환영하며 5년 이상 경력 필수입니다.
        재택근무 가능하지만 매일 9시 출근 필수입니다.
        회사는 100명 규모입니다.
        """
    
    def test_validate_consistency_valid(self, validator, consistent_content):
        """Test validation of consistent content"""
        result = validator.validate_consistency(consistent_content)
        
        assert result.is_consistent
        assert result.consistency_score > 0.8
        assert len(result.inconsistencies) == 0
    
    def test_validate_consistency_invalid(self, validator, inconsistent_content):
        """Test validation of inconsistent content"""
        result = validator.validate_consistency(inconsistent_content)
        
        assert not result.is_consistent
        assert result.consistency_score < 0.8
        assert len(result.inconsistencies) > 0
        
        # Check specific inconsistencies
        inconsistency_types = [i["type"] for i in result.inconsistencies]
        assert "numeric_conflict" in inconsistency_types
        assert "logical_contradiction" in inconsistency_types
    
    def test_numeric_consistency(self, validator):
        """Test numeric consistency checking"""
        content = "우리는 50명 회사입니다. 전체 100명의 직원이 있습니다."
        
        inconsistencies = validator._check_numeric_consistency(content, {})
        
        assert len(inconsistencies) > 0
        assert any(i["entity"] == "employee" for i in inconsistencies)
    
    def test_temporal_consistency(self, validator):
        """Test temporal consistency checking"""
        content = "2020년 설립된 회사가 2025년부터 운영을 시작했습니다."
        
        inconsistencies = validator._check_temporal_consistency(content, {})
        
        assert len(inconsistencies) > 0
    
    def test_logical_consistency(self, validator):
        """Test logical consistency checking"""
        content = "신입 환영! 10년 이상 경력 필수입니다."
        
        inconsistencies = validator._check_logical_consistency(content, {})
        
        assert len(inconsistencies) > 0
        assert any(i["type"] == "logical_contradiction" for i in inconsistencies)
    
    def test_compare_consistency(self, validator):
        """Test comparing consistency between contents"""
        content1 = "우리 회사는 50명입니다."
        content2 = "우리 회사는 100명입니다."
        
        comparison = validator.compare_consistency(content1, content2)
        
        assert "conflicts" in comparison
        assert "alignments" in comparison
        assert len(comparison["conflicts"]) > 0
        assert comparison["overall_alignment"] < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])