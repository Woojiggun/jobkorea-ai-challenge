"""
Boundary validator for preventing hallucinations
"""
from typing import Dict, List, Any, Optional, Set, Tuple
import numpy as np
from dataclasses import dataclass
import re
import logging

from .topology_mapper import TopologyMapper

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of boundary validation"""
    is_valid: bool
    violations: List[str]
    confidence: float
    suggestions: List[str]


class BoundaryValidator:
    """
    Validates that generated content stays within topological boundaries
    """
    
    def __init__(self, topology: TopologyMapper):
        self.topology = topology
        self.forbidden_patterns = self._load_forbidden_patterns()
        self.boundary_rules = self._initialize_boundary_rules()
        
    def _load_forbidden_patterns(self) -> List[re.Pattern]:
        """Load patterns that indicate hallucination"""
        patterns = [
            # Absolute superlatives without data
            r"업계\s*(최고|최대|최초|유일)",
            r"국내\s*(최고|최대|최초|유일)",
            r"세계\s*(최고|최대|최초|유일)",
            
            # Unverifiable claims
            r"(\d+)%\s*이상의\s*성장",
            r"평균\s*연봉.*억",
            
            # Discriminatory patterns
            r"(남자|여자|남성|여성)\s*(우대|선호|환영)",
            r"(\d+)세\s*(이하|미만)",
            r"(기혼|미혼)\s*(우대|선호)",
            
            # Vague quantifiers
            r"다수의\s*기회",
            r"무한한\s*성장",
            r"최상의\s*대우"
        ]
        
        return [re.compile(pattern) for pattern in patterns]
    
    def _initialize_boundary_rules(self) -> Dict[str, Any]:
        """Initialize rules for different regions"""
        return {
            "startup": {
                "allowed_benefits": [
                    "스톡옵션", "유연근무", "자율출퇴근", "간식제공",
                    "도서지원", "교육지원", "맥북지원"
                ],
                "forbidden_claims": [
                    "안정적인 기업", "대기업 수준의 복지", "업계 최고 연봉"
                ],
                "typical_ranges": {
                    "employee_count": (1, 50),
                    "growth_rate": (0, 200),
                    "years_established": (0, 5)
                }
            },
            "enterprise": {
                "allowed_benefits": [
                    "4대보험", "퇴직금", "인센티브", "건강검진",
                    "자녀학자금", "사내대출", "복지포인트"
                ],
                "forbidden_claims": [
                    "스타트업의 속도", "무한한 성장 가능성", "자유로운 분위기"
                ],
                "typical_ranges": {
                    "employee_count": (1000, 100000),
                    "growth_rate": (-10, 30),
                    "years_established": (10, 100)
                }
            }
        }
    
    def validate_content(
        self, 
        content: str, 
        company_data: Dict[str, Any],
        region: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate generated content against boundaries
        
        Args:
            content: Generated text content
            company_data: Original company data
            region: Optional region override
            
        Returns:
            ValidationResult
        """
        violations = []
        suggestions = []
        
        # Determine region if not provided
        if region is None:
            region = self._determine_company_region(company_data)
            
        # Check forbidden patterns
        pattern_violations = self._check_forbidden_patterns(content)
        violations.extend(pattern_violations)
        
        # Check factual accuracy
        fact_violations = self._check_facts(content, company_data)
        violations.extend(fact_violations)
        
        # Check regional boundaries
        if region in self.boundary_rules:
            boundary_violations = self._check_regional_boundaries(
                content, company_data, region
            )
            violations.extend(boundary_violations)
            
        # Check quantitative claims
        quant_violations = self._check_quantitative_claims(content, company_data)
        violations.extend(quant_violations)
        
        # Generate suggestions for violations
        if violations:
            suggestions = self._generate_suggestions(violations, company_data)
            
        # Calculate confidence score
        confidence = 1.0 - (len(violations) * 0.1)
        confidence = max(0.0, confidence)
        
        return ValidationResult(
            is_valid=len(violations) == 0,
            violations=violations,
            confidence=confidence,
            suggestions=suggestions
        )
    
    def _determine_company_region(self, company_data: Dict[str, Any]) -> str:
        """Determine which region a company belongs to"""
        employee_count = company_data.get("employee_count", 0)
        
        if employee_count < 50:
            return "startup"
        elif employee_count < 200:
            return "scaleup"
        elif employee_count < 1000:
            return "midsize"
        else:
            return "enterprise"
    
    def _check_forbidden_patterns(self, content: str) -> List[str]:
        """Check for forbidden patterns in content"""
        violations = []
        
        for pattern in self.forbidden_patterns:
            matches = pattern.findall(content)
            if matches:
                violations.append(
                    f"Forbidden pattern found: {matches[0]}"
                )
                
        return violations
    
    def _check_facts(
        self, 
        content: str, 
        company_data: Dict[str, Any]
    ) -> List[str]:
        """Check factual accuracy against company data"""
        violations = []
        
        # Extract numbers from content
        numbers_in_content = re.findall(r'\d+', content)
        
        # Check employee count
        if "employee_count" in company_data:
            actual_count = company_data["employee_count"]
            for num in numbers_in_content:
                if "명" in content[content.find(num):content.find(num)+10]:
                    stated_count = int(num)
                    if abs(stated_count - actual_count) > actual_count * 0.1:
                        violations.append(
                            f"Employee count mismatch: stated {stated_count}, "
                            f"actual {actual_count}"
                        )
                        
        # Check year established
        if "founded_year" in company_data:
            actual_year = company_data["founded_year"]
            for num in numbers_in_content:
                if len(num) == 4 and 1900 < int(num) < 2030:
                    stated_year = int(num)
                    if stated_year != actual_year:
                        violations.append(
                            f"Founded year mismatch: stated {stated_year}, "
                            f"actual {actual_year}"
                        )
                        
        return violations
    
    def _check_regional_boundaries(
        self, 
        content: str, 
        company_data: Dict[str, Any],
        region: str
    ) -> List[str]:
        """Check if content respects regional boundaries"""
        violations = []
        
        if region not in self.boundary_rules:
            return violations
            
        rules = self.boundary_rules[region]
        
        # Check for forbidden claims
        for forbidden in rules.get("forbidden_claims", []):
            if forbidden.lower() in content.lower():
                violations.append(
                    f"Inappropriate claim for {region}: '{forbidden}'"
                )
                
        # Check if benefits are appropriate
        mentioned_benefits = self._extract_benefits(content)
        allowed_benefits = set(rules.get("allowed_benefits", []))
        
        for benefit in mentioned_benefits:
            if benefit not in allowed_benefits:
                # Check if it's a completely made-up benefit
                if not self._is_generic_benefit(benefit):
                    violations.append(
                        f"Unusual benefit for {region}: '{benefit}'"
                    )
                    
        return violations
    
    def _check_quantitative_claims(
        self, 
        content: str, 
        company_data: Dict[str, Any]
    ) -> List[str]:
        """Check quantitative claims against data"""
        violations = []
        
        # Check growth rate claims
        growth_match = re.search(r'(\d+)%\s*성장', content)
        if growth_match:
            claimed_growth = int(growth_match.group(1))
            actual_growth = company_data.get("growth_rate", 0)
            
            if abs(claimed_growth - actual_growth) > 10:
                violations.append(
                    f"Growth rate mismatch: claimed {claimed_growth}%, "
                    f"actual {actual_growth}%"
                )
                
        # Check salary claims
        salary_match = re.search(r'(\d+)만원', content)
        if salary_match and "salary_range" not in company_data:
            violations.append(
                "Salary mentioned without data support"
            )
            
        return violations
    
    def _extract_benefits(self, content: str) -> Set[str]:
        """Extract mentioned benefits from content"""
        benefits = set()
        
        # Common benefit keywords
        benefit_keywords = [
            "스톡옵션", "유연근무", "자율출퇴근", "재택근무",
            "4대보험", "퇴직금", "인센티브", "성과급",
            "건강검진", "복지포인트", "간식", "도서지원",
            "교육비", "맥북", "식대", "교통비"
        ]
        
        for keyword in benefit_keywords:
            if keyword in content:
                benefits.add(keyword)
                
        return benefits
    
    def _is_generic_benefit(self, benefit: str) -> bool:
        """Check if a benefit is generic enough to be acceptable"""
        generic_benefits = {
            "성장기회", "교육기회", "네트워킹", "멘토링",
            "팀워크", "소통", "자기계발"
        }
        
        return benefit in generic_benefits
    
    def _generate_suggestions(
        self, 
        violations: List[str], 
        company_data: Dict[str, Any]
    ) -> List[str]:
        """Generate suggestions to fix violations"""
        suggestions = []
        
        for violation in violations:
            if "Employee count mismatch" in violation:
                suggestions.append(
                    f"Use actual employee count: {company_data.get('employee_count')}명"
                )
            elif "Growth rate mismatch" in violation:
                suggestions.append(
                    f"Use actual growth rate: {company_data.get('growth_rate')}%"
                )
            elif "Forbidden pattern" in violation:
                suggestions.append(
                    "Remove absolute claims and use factual descriptions"
                )
            elif "Inappropriate claim" in violation:
                region = self._determine_company_region(company_data)
                suggestions.append(
                    f"Use descriptions appropriate for {region} companies"
                )
                
        return suggestions
    
    def check_consistency(
        self, 
        content: str, 
        node_id: str
    ) -> ValidationResult:
        """
        Check if content is consistent with node's topological position
        
        Args:
            content: Generated content
            node_id: Node ID in topology
            
        Returns:
            ValidationResult
        """
        if node_id not in self.topology.nodes:
            return ValidationResult(
                is_valid=False,
                violations=["Node not found in topology"],
                confidence=0.0,
                suggestions=["Ensure node exists in topology"]
            )
            
        node = self.topology.nodes[node_id]
        region = self.topology._get_node_region(node_id)
        
        # Get region boundaries
        boundaries = self.topology.get_region_boundaries(region)
        
        violations = []
        
        # Check if content mentions attributes outside boundaries
        for attr, ranges in boundaries["attribute_ranges"].items():
            # Look for numbers near attribute mentions
            attr_pattern = re.compile(
                rf'{attr}.*?(\d+)', 
                re.IGNORECASE | re.DOTALL
            )
            matches = attr_pattern.findall(content[:100])  # Check first 100 chars
            
            for match in matches:
                value = float(match)
                if value < ranges["min"] or value > ranges["max"]:
                    violations.append(
                        f"{attr} value {value} outside region bounds "
                        f"[{ranges['min']}, {ranges['max']}]"
                    )
                    
        # Check topological consistency
        mentioned_neighbors = self._extract_mentioned_entities(content)
        valid_neighbors = set()
        
        # Get neighbors at different depths
        neighbors_by_depth = self.topology.find_neighbors(node_id, depth=2)
        for neighbors in neighbors_by_depth.values():
            valid_neighbors.update(neighbors)
            
        # Check if mentioned entities are topologically close
        for entity in mentioned_neighbors:
            if entity not in valid_neighbors and entity != node_id:
                violations.append(
                    f"Mentioned entity '{entity}' is not topologically "
                    f"connected to {node_id}"
                )
                
        confidence = 1.0 - (len(violations) * 0.15)
        
        return ValidationResult(
            is_valid=len(violations) == 0,
            violations=violations,
            confidence=max(0.0, confidence),
            suggestions=self._generate_consistency_suggestions(violations)
        )
    
    def _extract_mentioned_entities(self, content: str) -> Set[str]:
        """Extract entity references from content"""
        entities = set()
        
        # Look for company names (simplified)
        company_pattern = re.compile(r'[가-힣]+(?:테크|소프트|시스템즈?|그룹)')
        entities.update(company_pattern.findall(content))
        
        return entities
    
    def _generate_consistency_suggestions(
        self, 
        violations: List[str]
    ) -> List[str]:
        """Generate suggestions for consistency violations"""
        suggestions = []
        
        for violation in violations:
            if "outside region bounds" in violation:
                suggestions.append(
                    "Adjust values to match the company's actual scale"
                )
            elif "not topologically connected" in violation:
                suggestions.append(
                    "Only reference companies in similar categories"
                )
                
        return suggestions