"""
Consistency validation for generated content
"""
from typing import Dict, List, Any, Optional, Set
import re
from dataclasses import dataclass
import logging
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class ConsistencyResult:
    """Result of consistency validation"""
    is_consistent: bool
    consistency_score: float
    inconsistencies: List[Dict[str, Any]]
    suggestions: List[str]


class ConsistencyValidator:
    """
    Validates internal consistency of generated content
    """
    
    def __init__(self):
        self.consistency_rules = self._initialize_rules()
        
    def _initialize_rules(self) -> Dict[str, Any]:
        """Initialize consistency rules"""
        return {
            "numeric_consistency": {
                "check_function": self._check_numeric_consistency,
                "weight": 0.3
            },
            "temporal_consistency": {
                "check_function": self._check_temporal_consistency,
                "weight": 0.2
            },
            "logical_consistency": {
                "check_function": self._check_logical_consistency,
                "weight": 0.3
            },
            "terminology_consistency": {
                "check_function": self._check_terminology_consistency,
                "weight": 0.2
            }
        }
    
    def validate_consistency(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConsistencyResult:
        """
        Validate internal consistency of content
        
        Args:
            content: Content to validate
            metadata: Optional metadata for context
            
        Returns:
            ConsistencyResult
        """
        all_inconsistencies = []
        weighted_scores = []
        
        # Run each consistency check
        for rule_name, rule_config in self.consistency_rules.items():
            check_function = rule_config["check_function"]
            weight = rule_config["weight"]
            
            inconsistencies = check_function(content, metadata)
            all_inconsistencies.extend(inconsistencies)
            
            # Calculate score for this rule (1 - normalized inconsistency count)
            rule_score = 1.0 / (1.0 + len(inconsistencies))
            weighted_scores.append(rule_score * weight)
        
        # Calculate overall consistency score
        consistency_score = sum(weighted_scores) / sum(
            r["weight"] for r in self.consistency_rules.values()
        )
        
        # Generate suggestions
        suggestions = self._generate_suggestions(all_inconsistencies)
        
        return ConsistencyResult(
            is_consistent=len(all_inconsistencies) == 0,
            consistency_score=consistency_score,
            inconsistencies=all_inconsistencies,
            suggestions=suggestions
        )
    
    def _check_numeric_consistency(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Check consistency of numeric values"""
        inconsistencies = []
        
        # Extract all numeric references
        numeric_refs = self._extract_numeric_references(content)
        
        # Group by entity
        entity_numbers = {}
        for ref in numeric_refs:
            entity = ref["entity"]
            if entity not in entity_numbers:
                entity_numbers[entity] = []
            entity_numbers[entity].append(ref)
        
        # Check for conflicts
        for entity, refs in entity_numbers.items():
            values = [r["value"] for r in refs]
            if len(set(values)) > 1:
                inconsistencies.append({
                    "type": "numeric_conflict",
                    "entity": entity,
                    "values": values,
                    "locations": [r["location"] for r in refs],
                    "description": f"Conflicting values for {entity}: {values}"
                })
                
        return inconsistencies
    
    def _extract_numeric_references(self, content: str) -> List[Dict[str, Any]]:
        """Extract numeric references with context"""
        references = []
        
        # Pattern to capture number with surrounding context
        pattern = re.compile(r'(\w+\s+)?(\d+)\s*(%|명|년|개|억|만)?(\s+\w+)?')
        
        for match in pattern.finditer(content):
            before = match.group(1) or ""
            number = int(match.group(2))
            unit = match.group(3) or ""
            after = match.group(4) or ""
            
            # Determine entity from context
            entity = self._determine_entity(before + after)
            
            if entity:
                references.append({
                    "entity": entity,
                    "value": number,
                    "unit": unit,
                    "text": match.group(),
                    "location": (match.start(), match.end())
                })
                
        return references
    
    def _determine_entity(self, context: str) -> Optional[str]:
        """Determine entity type from context"""
        context_lower = context.lower()
        
        entity_keywords = {
            "employee": ["직원", "임직원", "구성원", "인원"],
            "year": ["년", "연도", "설립", "창립"],
            "growth": ["성장", "증가", "향상"],
            "experience": ["경력", "경험", "년차"],
            "team": ["팀", "부서", "조직"]
        }
        
        for entity, keywords in entity_keywords.items():
            for keyword in keywords:
                if keyword in context_lower:
                    return entity
                    
        return None
    
    def _check_temporal_consistency(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Check temporal consistency"""
        inconsistencies = []
        
        # Extract temporal references
        year_pattern = re.compile(r'(\d{4})\s*년')
        years = []
        
        for match in year_pattern.finditer(content):
            year = int(match.group(1))
            years.append({
                "year": year,
                "text": match.group(),
                "location": (match.start(), match.end())
            })
        
        # Check for logical temporal order
        if years:
            # Check if founded year appears
            founded_years = [y for y in years if "설립" in content[max(0, y["location"][0]-20):y["location"][1]+20]]
            current_year = 2024  # Or get from metadata
            
            for year_ref in years:
                year = year_ref["year"]
                
                # Future years are suspicious
                if year > current_year:
                    inconsistencies.append({
                        "type": "future_year",
                        "year": year,
                        "location": year_ref["location"],
                        "description": f"Future year referenced: {year}"
                    })
                    
                # Very old years might be errors
                if year < 1900 and year not in founded_years:
                    inconsistencies.append({
                        "type": "suspicious_year",
                        "year": year,
                        "location": year_ref["location"],
                        "description": f"Suspicious year: {year}"
                    })
                    
        return inconsistencies
    
    def _check_logical_consistency(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Check logical consistency"""
        inconsistencies = []
        
        # Check for contradictory statements
        contradictions = [
            (r"신입\s*환영", r"\d+년\s*이상\s*경력"),
            (r"재택\s*근무", r"출퇴근\s*필수"),
            (r"자율\s*출퇴근", r"9\s*시\s*출근"),
            (r"스타트업", r"대기업\s*수준")
        ]
        
        for pattern1, pattern2 in contradictions:
            if re.search(pattern1, content) and re.search(pattern2, content):
                inconsistencies.append({
                    "type": "logical_contradiction",
                    "patterns": [pattern1, pattern2],
                    "description": f"Contradictory statements found"
                })
                
        # Check size-benefit consistency
        size_benefit_rules = {
            "startup": ["스톡옵션", "빠른 성장", "자율성"],
            "enterprise": ["안정성", "체계적", "복지"]
        }
        
        # Detect company size from content
        if "스타트업" in content or "작은" in content:
            company_type = "startup"
        elif "대기업" in content or "큰" in content:
            company_type = "enterprise"
        else:
            company_type = None
            
        if company_type:
            opposite_type = "enterprise" if company_type == "startup" else "startup"
            opposite_benefits = size_benefit_rules[opposite_type]
            
            for benefit in opposite_benefits:
                if benefit in content:
                    inconsistencies.append({
                        "type": "size_benefit_mismatch",
                        "company_type": company_type,
                        "benefit": benefit,
                        "description": f"{company_type} mentioning {opposite_type} benefit: {benefit}"
                    })
                    
        return inconsistencies
    
    def _check_terminology_consistency(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Check terminology consistency"""
        inconsistencies = []
        
        # Define equivalent terms
        term_groups = [
            ["회사", "기업", "조직"],
            ["직원", "임직원", "구성원"],
            ["개발자", "엔지니어", "프로그래머"],
            ["팀", "부서", "조직"],
            ["연봉", "급여", "보수"]
        ]
        
        # Check for inconsistent term usage
        for term_group in term_groups:
            used_terms = []
            for term in term_group:
                if term in content:
                    count = content.count(term)
                    used_terms.append((term, count))
                    
            if len(used_terms) > 1:
                # Multiple terms from same group used
                inconsistencies.append({
                    "type": "terminology_inconsistency",
                    "terms": [t[0] for t in used_terms],
                    "counts": {t[0]: t[1] for t in used_terms},
                    "description": f"Inconsistent terminology: {', '.join(t[0] for t in used_terms)}"
                })
                
        return inconsistencies
    
    def _generate_suggestions(
        self,
        inconsistencies: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate suggestions to fix inconsistencies"""
        suggestions = []
        
        # Group by type
        by_type = {}
        for inc in inconsistencies:
            inc_type = inc["type"]
            if inc_type not in by_type:
                by_type[inc_type] = []
            by_type[inc_type].append(inc)
            
        # Generate type-specific suggestions
        if "numeric_conflict" in by_type:
            suggestions.append("통일된 수치 사용: 같은 항목에 대해 일관된 숫자를 사용하세요")
            
        if "logical_contradiction" in by_type:
            suggestions.append("모순된 내용 수정: 서로 상충하는 내용을 확인하고 수정하세요")
            
        if "terminology_inconsistency" in by_type:
            suggestions.append("용어 통일: 같은 의미의 용어는 하나로 통일하여 사용하세요")
            
        if "future_year" in by_type:
            suggestions.append("연도 확인: 미래 연도가 올바른지 확인하세요")
            
        return suggestions
    
    def compare_consistency(
        self,
        content1: str,
        content2: str
    ) -> Dict[str, Any]:
        """Compare consistency between two pieces of content"""
        # Validate each independently
        result1 = self.validate_consistency(content1)
        result2 = self.validate_consistency(content2)
        
        # Extract facts from both
        facts1 = self._extract_all_facts(content1)
        facts2 = self._extract_all_facts(content2)
        
        # Compare facts
        conflicts = []
        alignments = []
        
        for fact_type in set(facts1.keys()) | set(facts2.keys()):
            if fact_type in facts1 and fact_type in facts2:
                if facts1[fact_type] == facts2[fact_type]:
                    alignments.append({
                        "type": fact_type,
                        "value": facts1[fact_type]
                    })
                else:
                    conflicts.append({
                        "type": fact_type,
                        "value1": facts1[fact_type],
                        "value2": facts2[fact_type]
                    })
                    
        return {
            "individual_scores": {
                "content1": result1.consistency_score,
                "content2": result2.consistency_score
            },
            "conflicts": conflicts,
            "alignments": alignments,
            "overall_alignment": len(alignments) / (len(alignments) + len(conflicts))
                               if (alignments or conflicts) else 1.0
        }
    
    def _extract_all_facts(self, content: str) -> Dict[str, Any]:
        """Extract all facts from content"""
        facts = {}
        
        # Extract numeric facts
        numeric_refs = self._extract_numeric_references(content)
        for ref in numeric_refs:
            entity = ref["entity"]
            if entity:
                facts[entity] = ref["value"]
                
        # Extract other facts (simplified)
        if "재택" in content:
            facts["remote_work"] = True
        elif "출퇴근" in content:
            facts["remote_work"] = False
            
        return facts