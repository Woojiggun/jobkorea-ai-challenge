"""
Hallucination prevention and detection system
"""
from typing import Dict, List, Any, Optional, Set, Tuple
import re
import numpy as np
from dataclasses import dataclass
import logging
from difflib import SequenceMatcher

from src.topology.boundary_validator import BoundaryValidator, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class HallucinationCheck:
    """Result of hallucination check"""
    has_hallucination: bool
    hallucination_score: float
    detected_issues: List[Dict[str, Any]]
    corrections: List[Dict[str, str]]
    confidence: float


class HallucinationGuard:
    """
    Guards against hallucinations in generated content
    """
    
    def __init__(self, boundary_validator: BoundaryValidator):
        """
        Initialize hallucination guard
        
        Args:
            boundary_validator: BoundaryValidator instance
        """
        self.boundary_validator = boundary_validator
        self.hallucination_patterns = self._initialize_patterns()
        
    def _initialize_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Initialize hallucination detection patterns"""
        return {
            "absolute_claims": [
                re.compile(r"(최고|최초|유일|최대)의?\s+\w+", re.IGNORECASE),
                re.compile(r"업계\s*(1위|선두|리더)", re.IGNORECASE),
                re.compile(r"국내\s*(최초|유일)", re.IGNORECASE)
            ],
            "unverifiable_numbers": [
                re.compile(r"\d+\s*%\s*(이상의?|가량의?)\s*(성장|증가|향상)", re.IGNORECASE),
                re.compile(r"(평균|최대)\s*\d+억\s*(매출|수익)", re.IGNORECASE),
                re.compile(r"\d+\s*개국\s*(진출|서비스)", re.IGNORECASE)
            ],
            "vague_benefits": [
                re.compile(r"(최고|최상)의?\s*(대우|복지|환경)", re.IGNORECASE),
                re.compile(r"(무한한?|무제한)\s*(성장|기회|가능성)", re.IGNORECASE),
                re.compile(r"업계\s*(최고|최상위)\s*(연봉|급여)", re.IGNORECASE)
            ],
            "temporal_claims": [
                re.compile(r"(곧|조만간|머지않아)\s*\w+\s*(예정|계획)", re.IGNORECASE),
                re.compile(r"(빠른|급속한)\s*(성장|확장)\s*중", re.IGNORECASE)
            ]
        }
    
    def check_hallucination(
        self,
        generated_content: str,
        source_data: Dict[str, Any],
        retrieval_context: Optional[List[Dict[str, Any]]] = None
    ) -> HallucinationCheck:
        """
        Check for hallucinations in generated content
        
        Args:
            generated_content: Generated text
            source_data: Original source data
            retrieval_context: Retrieved context used for generation
            
        Returns:
            HallucinationCheck result
        """
        detected_issues = []
        
        # 1. Pattern-based detection
        pattern_issues = self._detect_pattern_hallucinations(generated_content)
        detected_issues.extend(pattern_issues)
        
        # 2. Fact verification
        fact_issues = self._verify_facts(generated_content, source_data)
        detected_issues.extend(fact_issues)
        
        # 3. Numeric consistency check
        numeric_issues = self._check_numeric_consistency(generated_content, source_data)
        detected_issues.extend(numeric_issues)
        
        # 4. Context alignment check
        if retrieval_context:
            context_issues = self._check_context_alignment(
                generated_content, retrieval_context
            )
            detected_issues.extend(context_issues)
        
        # 5. Boundary validation
        validation_result = self.boundary_validator.validate_content(
            generated_content, source_data
        )
        
        for violation in validation_result.violations:
            detected_issues.append({
                "type": "boundary_violation",
                "severity": "high",
                "description": violation,
                "location": None
            })
        
        # Calculate hallucination score
        hallucination_score = self._calculate_hallucination_score(detected_issues)
        
        # Generate corrections
        corrections = self._generate_corrections(detected_issues, generated_content, source_data)
        
        # Calculate confidence
        confidence = 1.0 - hallucination_score
        
        return HallucinationCheck(
            has_hallucination=len(detected_issues) > 0,
            hallucination_score=hallucination_score,
            detected_issues=detected_issues,
            corrections=corrections,
            confidence=confidence
        )
    
    def _detect_pattern_hallucinations(self, content: str) -> List[Dict[str, Any]]:
        """Detect hallucinations based on patterns"""
        issues = []
        
        for category, patterns in self.hallucination_patterns.items():
            for pattern in patterns:
                matches = pattern.finditer(content)
                for match in matches:
                    issues.append({
                        "type": "pattern_hallucination",
                        "category": category,
                        "severity": "medium",
                        "description": f"Detected {category}: '{match.group()}'",
                        "location": (match.start(), match.end()),
                        "matched_text": match.group()
                    })
                    
        return issues
    
    def _verify_facts(
        self,
        content: str,
        source_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Verify factual claims against source data"""
        issues = []
        
        # Extract claims from content
        claims = self._extract_claims(content)
        
        for claim in claims:
            if not self._verify_claim(claim, source_data):
                issues.append({
                    "type": "factual_error",
                    "severity": "high",
                    "description": f"Unverifiable claim: {claim['text']}",
                    "location": claim.get("location"),
                    "claim": claim
                })
                
        return issues
    
    def _extract_claims(self, content: str) -> List[Dict[str, Any]]:
        """Extract factual claims from content"""
        claims = []
        
        # Company size claims
        size_pattern = re.compile(r"(\d+)\s*명?\s*(직원|임직원|구성원)")
        for match in size_pattern.finditer(content):
            claims.append({
                "type": "employee_count",
                "value": int(match.group(1)),
                "text": match.group(),
                "location": (match.start(), match.end())
            })
        
        # Year claims
        year_pattern = re.compile(r"(\d{4})\s*년?\s*(설립|창립|시작)")
        for match in year_pattern.finditer(content):
            claims.append({
                "type": "founded_year",
                "value": int(match.group(1)),
                "text": match.group(),
                "location": (match.start(), match.end())
            })
        
        # Growth rate claims
        growth_pattern = re.compile(r"(\d+)\s*%\s*(성장|증가)")
        for match in growth_pattern.finditer(content):
            claims.append({
                "type": "growth_rate",
                "value": int(match.group(1)),
                "text": match.group(),
                "location": (match.start(), match.end())
            })
        
        return claims
    
    def _verify_claim(self, claim: Dict[str, Any], source_data: Dict[str, Any]) -> bool:
        """Verify a single claim against source data"""
        claim_type = claim["type"]
        claim_value = claim["value"]
        
        if claim_type == "employee_count":
            actual_value = source_data.get("employee_count")
            if actual_value:
                # Allow 10% margin
                return abs(claim_value - actual_value) <= actual_value * 0.1
                
        elif claim_type == "founded_year":
            actual_value = source_data.get("founded_year")
            if actual_value:
                return claim_value == actual_value
                
        elif claim_type == "growth_rate":
            actual_value = source_data.get("growth_rate")
            if actual_value:
                # Allow 5% margin for growth rates
                return abs(claim_value - actual_value) <= 5
                
        # If no data to verify against, mark as unverifiable
        return False
    
    def _check_numeric_consistency(
        self,
        content: str,
        source_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check consistency of numeric values"""
        issues = []
        
        # Extract all numbers
        numbers = re.findall(r'\d+', content)
        
        for num_str in numbers:
            num = int(num_str)
            
            # Check if number appears in source data
            found_in_source = False
            for key, value in source_data.items():
                if isinstance(value, (int, float)):
                    if abs(num - value) < value * 0.1:  # 10% tolerance
                        found_in_source = True
                        break
                        
            # Large numbers not in source are suspicious
            if not found_in_source and num > 100:
                issues.append({
                    "type": "suspicious_number",
                    "severity": "medium",
                    "description": f"Large number {num} not found in source data",
                    "location": None,
                    "value": num
                })
                
        return issues
    
    def _check_context_alignment(
        self,
        content: str,
        retrieval_context: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Check if content aligns with retrieved context"""
        issues = []
        
        # Extract key phrases from content
        content_phrases = self._extract_key_phrases(content)
        
        # Check if phrases appear in context
        context_text = " ".join([
            doc.get("content", "") for doc in retrieval_context
        ])
        
        for phrase in content_phrases:
            if len(phrase) > 10 and phrase not in context_text:
                # Check similarity to context
                similarity = self._calculate_phrase_similarity(phrase, context_text)
                
                if similarity < 0.3:  # Low similarity
                    issues.append({
                        "type": "context_deviation",
                        "severity": "low",
                        "description": f"Phrase not aligned with context: '{phrase[:50]}...'",
                        "location": None,
                        "phrase": phrase,
                        "similarity": similarity
                    })
                    
        return issues
    
    def _extract_key_phrases(self, content: str) -> List[str]:
        """Extract key phrases from content"""
        # Simple extraction of phrases between punctuation
        sentences = re.split(r'[.!?]', content)
        phrases = []
        
        for sentence in sentences:
            # Extract noun phrases (simplified)
            phrase_pattern = re.compile(r'[\w\s]{10,50}')
            phrases.extend(phrase_pattern.findall(sentence))
            
        return phrases
    
    def _calculate_phrase_similarity(self, phrase: str, context: str) -> float:
        """Calculate similarity between phrase and context"""
        # Use sequence matcher for simple similarity
        matcher = SequenceMatcher(None, phrase.lower(), context.lower())
        
        # Find best matching substring
        match = matcher.find_longest_match(0, len(phrase), 0, len(context))
        
        # Calculate similarity ratio
        return match.size / len(phrase)
    
    def _calculate_hallucination_score(
        self,
        detected_issues: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall hallucination score"""
        if not detected_issues:
            return 0.0
            
        # Weight by severity
        severity_weights = {
            "high": 0.3,
            "medium": 0.2,
            "low": 0.1
        }
        
        total_score = 0.0
        for issue in detected_issues:
            severity = issue.get("severity", "medium")
            weight = severity_weights.get(severity, 0.2)
            total_score += weight
            
        # Normalize to [0, 1]
        return min(total_score, 1.0)
    
    def _generate_corrections(
        self,
        issues: List[Dict[str, Any]],
        content: str,
        source_data: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Generate corrections for detected issues"""
        corrections = []
        
        for issue in issues:
            if issue["type"] == "factual_error":
                claim = issue.get("claim", {})
                if claim.get("type") == "employee_count":
                    actual_count = source_data.get("employee_count")
                    if actual_count:
                        corrections.append({
                            "original": claim["text"],
                            "corrected": f"{actual_count}명",
                            "reason": "Incorrect employee count"
                        })
                        
            elif issue["type"] == "pattern_hallucination":
                if issue["category"] == "absolute_claims":
                    corrections.append({
                        "original": issue["matched_text"],
                        "corrected": "[구체적인 데이터로 교체 필요]",
                        "reason": "Unverifiable absolute claim"
                    })
                    
        return corrections
    
    def apply_corrections(
        self,
        content: str,
        corrections: List[Dict[str, str]]
    ) -> str:
        """Apply corrections to content"""
        corrected_content = content
        
        # Sort corrections by position (reverse) to maintain positions
        corrections_with_pos = []
        for correction in corrections:
            original = correction["original"]
            pos = content.find(original)
            if pos != -1:
                corrections_with_pos.append((pos, correction))
                
        corrections_with_pos.sort(key=lambda x: x[0], reverse=True)
        
        # Apply corrections
        for pos, correction in corrections_with_pos:
            original = correction["original"]
            corrected = correction["corrected"]
            
            before = corrected_content[:pos]
            after = corrected_content[pos + len(original):]
            corrected_content = before + corrected + after
            
        return corrected_content
    
    def generate_safe_content(
        self,
        template: str,
        data: Dict[str, Any],
        strict_mode: bool = True
    ) -> str:
        """
        Generate content using safe templates
        
        Args:
            template: Content template with placeholders
            data: Data to fill template
            strict_mode: If True, only use exact data matches
            
        Returns:
            Safe generated content
        """
        # Replace placeholders with actual data
        safe_content = template
        
        for key, value in data.items():
            placeholder = f"{{{key}}}"
            if placeholder in safe_content:
                # Ensure value is safe
                if isinstance(value, (int, float)):
                    safe_value = str(value)
                elif isinstance(value, list):
                    safe_value = ", ".join(str(v) for v in value)
                else:
                    safe_value = str(value)
                    
                safe_content = safe_content.replace(placeholder, safe_value)
        
        if strict_mode:
            # Remove any remaining placeholders
            safe_content = re.sub(r'\{[^}]+\}', '', safe_content)
            
        return safe_content.strip()