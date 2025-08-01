"""
Fact checking system for generated content
"""
from typing import Dict, List, Any, Optional, Tuple
import re
from dataclasses import dataclass
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class FactCheckResult:
    """Result of fact checking"""
    is_valid: bool
    facts_checked: int
    facts_verified: int
    facts_failed: int
    details: List[Dict[str, Any]]
    confidence: float


class FactChecker:
    """
    Verifies factual accuracy of generated content
    """
    
    def __init__(self):
        self.fact_patterns = self._initialize_fact_patterns()
        
    def _initialize_fact_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize patterns for fact extraction"""
        return {
            "employee_count": re.compile(r"(\d+)\s*명?\s*(직원|임직원|구성원)"),
            "founded_year": re.compile(r"(\d{4})\s*년?\s*(설립|창립|시작)"),
            "revenue": re.compile(r"(\d+)\s*(억|만)?\s*원?\s*(매출|수익)"),
            "growth_rate": re.compile(r"(\d+)\s*%\s*(성장|증가|향상)"),
            "location": re.compile(r"(서울|부산|대구|인천|광주|대전|울산|경기|강원|충북|충남|전북|전남|경북|경남|제주)"),
            "benefits_count": re.compile(r"(\d+)\s*가지?\s*(복지|혜택|제도)"),
            "experience_requirement": re.compile(r"(\d+)\s*년?\s*(이상|이하)?\s*(경력|경험)"),
            "team_size": re.compile(r"(\d+)\s*명?\s*(팀|부서|조직)"),
            "salary": re.compile(r"(\d+)\s*(만원|천만원|억원)\s*(연봉|급여|월급)"),
            "work_hours": re.compile(r"(\d+)\s*시?\s*~\s*(\d+)\s*시?")
        }
    
    def check_facts(
        self,
        content: str,
        source_data: Dict[str, Any],
        context_data: Optional[List[Dict[str, Any]]] = None
    ) -> FactCheckResult:
        """
        Check facts in content against source data
        
        Args:
            content: Content to check
            source_data: Source of truth
            context_data: Additional context for verification
            
        Returns:
            FactCheckResult
        """
        # Extract facts from content
        extracted_facts = self._extract_facts(content)
        
        # Verify each fact
        verification_results = []
        facts_verified = 0
        facts_failed = 0
        
        for fact in extracted_facts:
            is_verified, details = self._verify_fact(fact, source_data, context_data)
            
            if is_verified:
                facts_verified += 1
            else:
                facts_failed += 1
                
            verification_results.append({
                "fact": fact,
                "verified": is_verified,
                "details": details
            })
        
        # Calculate confidence
        total_facts = len(extracted_facts)
        confidence = facts_verified / total_facts if total_facts > 0 else 1.0
        
        return FactCheckResult(
            is_valid=facts_failed == 0,
            facts_checked=total_facts,
            facts_verified=facts_verified,
            facts_failed=facts_failed,
            details=verification_results,
            confidence=confidence
        )
    
    def _extract_facts(self, content: str) -> List[Dict[str, Any]]:
        """Extract facts from content"""
        facts = []
        
        for fact_type, pattern in self.fact_patterns.items():
            matches = pattern.finditer(content)
            
            for match in matches:
                fact = {
                    "type": fact_type,
                    "text": match.group(),
                    "value": self._parse_fact_value(match, fact_type),
                    "position": (match.start(), match.end())
                }
                facts.append(fact)
                
        return facts
    
    def _parse_fact_value(self, match: re.Match, fact_type: str) -> Any:
        """Parse fact value from regex match"""
        if fact_type in ["employee_count", "founded_year", "benefits_count", 
                        "experience_requirement", "team_size"]:
            return int(match.group(1))
            
        elif fact_type == "growth_rate":
            return float(match.group(1))
            
        elif fact_type == "revenue":
            value = int(match.group(1))
            unit = match.group(2)
            if unit == "억":
                value *= 100000000
            elif unit == "만":
                value *= 10000
            return value
            
        elif fact_type == "salary":
            value = int(match.group(1))
            unit = match.group(2)
            if unit == "천만원":
                value *= 10000000
            elif unit == "만원":
                value *= 10000
            elif unit == "억원":
                value *= 100000000
            return value
            
        elif fact_type == "work_hours":
            return (int(match.group(1)), int(match.group(2)))
            
        elif fact_type == "location":
            return match.group(1)
            
        return match.group()
    
    def _verify_fact(
        self,
        fact: Dict[str, Any],
        source_data: Dict[str, Any],
        context_data: Optional[List[Dict[str, Any]]]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Verify a single fact"""
        fact_type = fact["type"]
        fact_value = fact["value"]
        
        # Direct verification against source
        if fact_type == "employee_count":
            if "employee_count" in source_data:
                actual = source_data["employee_count"]
                tolerance = actual * 0.1  # 10% tolerance
                is_valid = abs(fact_value - actual) <= tolerance
                return is_valid, {
                    "expected": actual,
                    "found": fact_value,
                    "tolerance": tolerance
                }
                
        elif fact_type == "founded_year":
            if "founded_year" in source_data:
                actual = source_data["founded_year"]
                is_valid = fact_value == actual
                return is_valid, {
                    "expected": actual,
                    "found": fact_value
                }
                
        elif fact_type == "growth_rate":
            if "growth_rate" in source_data:
                actual = source_data["growth_rate"]
                tolerance = 5  # 5% tolerance for growth rates
                is_valid = abs(fact_value - actual) <= tolerance
                return is_valid, {
                    "expected": actual,
                    "found": fact_value,
                    "tolerance": tolerance
                }
                
        elif fact_type == "location":
            if "location" in source_data:
                actual = source_data["location"]
                is_valid = fact_value in actual or actual in fact_value
                return is_valid, {
                    "expected": actual,
                    "found": fact_value
                }
                
        # Context-based verification
        if context_data:
            for context in context_data:
                if self._verify_against_context(fact, context):
                    return True, {"source": "context", "context_id": context.get("id")}
                    
        # If no source to verify against
        return False, {"reason": "no_source_data"}
    
    def _verify_against_context(
        self,
        fact: Dict[str, Any],
        context: Dict[str, Any]
    ) -> bool:
        """Verify fact against context document"""
        # Simple verification - check if similar fact appears in context
        context_text = context.get("content", "")
        fact_text = fact["text"]
        
        # Check exact match
        if fact_text in context_text:
            return True
            
        # Check value match for numeric facts
        if isinstance(fact["value"], (int, float)):
            value_str = str(fact["value"])
            if value_str in context_text:
                return True
                
        return False
    
    def cross_check_facts(
        self,
        facts1: List[Dict[str, Any]],
        facts2: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Cross-check facts between two sets"""
        # Group facts by type
        facts1_by_type = {}
        facts2_by_type = {}
        
        for fact in facts1:
            fact_type = fact["type"]
            if fact_type not in facts1_by_type:
                facts1_by_type[fact_type] = []
            facts1_by_type[fact_type].append(fact)
            
        for fact in facts2:
            fact_type = fact["type"]
            if fact_type not in facts2_by_type:
                facts2_by_type[fact_type] = []
            facts2_by_type[fact_type].append(fact)
            
        # Compare facts
        conflicts = []
        agreements = []
        
        for fact_type in set(facts1_by_type.keys()) | set(facts2_by_type.keys()):
            set1_facts = facts1_by_type.get(fact_type, [])
            set2_facts = facts2_by_type.get(fact_type, [])
            
            if set1_facts and set2_facts:
                # Compare values
                values1 = [f["value"] for f in set1_facts]
                values2 = [f["value"] for f in set2_facts]
                
                if set(values1) == set(values2):
                    agreements.append({
                        "type": fact_type,
                        "values": values1
                    })
                else:
                    conflicts.append({
                        "type": fact_type,
                        "set1_values": values1,
                        "set2_values": values2
                    })
                    
        return {
            "conflicts": conflicts,
            "agreements": agreements,
            "conflict_rate": len(conflicts) / (len(conflicts) + len(agreements)) 
                           if (conflicts or agreements) else 0
        }
    
    def suggest_fact_corrections(
        self,
        failed_facts: List[Dict[str, Any]],
        source_data: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Suggest corrections for failed facts"""
        corrections = []
        
        for fact_result in failed_facts:
            if not fact_result["verified"]:
                fact = fact_result["fact"]
                details = fact_result["details"]
                
                if "expected" in details:
                    correction = {
                        "original": fact["text"],
                        "suggested": self._generate_correction(fact, details["expected"]),
                        "reason": f"Actual value: {details['expected']}"
                    }
                    corrections.append(correction)
                    
        return corrections
    
    def _generate_correction(self, fact: Dict[str, Any], correct_value: Any) -> str:
        """Generate corrected text for a fact"""
        fact_type = fact["type"]
        
        if fact_type == "employee_count":
            return f"{correct_value}명"
        elif fact_type == "founded_year":
            return f"{correct_value}년 설립"
        elif fact_type == "growth_rate":
            return f"{correct_value}% 성장"
        elif fact_type == "location":
            return str(correct_value)
        else:
            # Generic replacement
            return str(correct_value)