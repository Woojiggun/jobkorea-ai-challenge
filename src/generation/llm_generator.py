"""
LLM-based job posting generator
"""
from typing import Dict, List, Any, Optional
import openai
import yaml
from pathlib import Path
import logging
from dataclasses import dataclass
import asyncio
import json

from config.settings import settings
from src.rag.retriever import TopologyAwareRetriever, RetrievalResult
from src.topology.boundary_validator import BoundaryValidator, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of generation operation"""
    content: str
    metadata: Dict[str, Any]
    validation_result: ValidationResult
    retrieval_context: List[Dict[str, Any]]
    prompt_used: str
    model_used: str
    tokens_used: Dict[str, int]


class LLMGenerator:
    """
    Generates job postings using LLM with topological constraints
    """
    
    def __init__(
        self,
        retriever: TopologyAwareRetriever,
        boundary_validator: BoundaryValidator,
        prompts_path: Optional[Path] = None
    ):
        """
        Initialize generator
        
        Args:
            retriever: TopologyAwareRetriever instance
            boundary_validator: BoundaryValidator instance
            prompts_path: Path to prompts configuration
        """
        self.retriever = retriever
        self.boundary_validator = boundary_validator
        
        # Initialize OpenAI client
        openai.api_key = settings.openai_api_key
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        
        # Load prompts
        self.prompts = self._load_prompts(prompts_path)
        
    def _load_prompts(self, prompts_path: Optional[Path]) -> Dict[str, Any]:
        """Load prompt templates"""
        if prompts_path is None:
            prompts_path = Path(__file__).parent.parent.parent / "config" / "prompts.yaml"
            
        try:
            with open(prompts_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load prompts: {e}")
            return self._get_default_prompts()
    
    def _get_default_prompts(self) -> Dict[str, Any]:
        """Get default prompts if file loading fails"""
        return {
            "job_posting_generation": {
                "system": "You are a job posting writer. Only use provided information.",
                "template": "{company_info}\n{job_info}\nWrite a job posting."
            }
        }
    
    async def generate_job_posting(
        self,
        company_data: Dict[str, Any],
        position_data: Dict[str, Any],
        target_candidate_profile: Optional[Dict[str, Any]] = None,
        style: str = "professional",
        max_retries: int = 3
    ) -> GenerationResult:
        """
        Generate a job posting with hallucination prevention
        
        Args:
            company_data: Company information
            position_data: Position requirements
            target_candidate_profile: Optional target candidate profile
            style: Writing style
            max_retries: Maximum generation attempts
            
        Returns:
            GenerationResult
        """
        # Retrieve relevant context
        retrieval_result = await self._retrieve_context(
            company_data, position_data, target_candidate_profile
        )
        
        # Generate with retries
        for attempt in range(max_retries):
            try:
                # Create prompt
                prompt = self._create_generation_prompt(
                    company_data, position_data, 
                    target_candidate_profile, retrieval_result,
                    style
                )
                
                # Generate content
                generated_content, tokens_used = await self._generate_with_llm(prompt)
                
                # Validate generated content
                validation_result = self.boundary_validator.validate_content(
                    generated_content, company_data
                )
                
                if validation_result.is_valid:
                    return GenerationResult(
                        content=generated_content,
                        metadata={
                            "company_id": company_data.get("id"),
                            "position": position_data.get("title"),
                            "style": style,
                            "attempt": attempt + 1
                        },
                        validation_result=validation_result,
                        retrieval_context=retrieval_result.documents,
                        prompt_used=prompt,
                        model_used="gpt-4",
                        tokens_used=tokens_used
                    )
                else:
                    # Log violations and retry with stricter constraints
                    logger.warning(
                        f"Validation failed on attempt {attempt + 1}: "
                        f"{validation_result.violations}"
                    )
                    
                    # Add violations to context for next attempt
                    company_data["_previous_violations"] = validation_result.violations
                    
            except Exception as e:
                logger.error(f"Generation failed on attempt {attempt + 1}: {e}")
                
        # All attempts failed
        raise Exception(f"Failed to generate valid content after {max_retries} attempts")
    
    async def _retrieve_context(
        self,
        company_data: Dict[str, Any],
        position_data: Dict[str, Any],
        target_profile: Optional[Dict[str, Any]]
    ) -> RetrievalResult:
        """Retrieve relevant context for generation"""
        # Create query combining company and position info
        query_parts = []
        
        if company_data.get("industry"):
            query_parts.append(f"{company_data['industry']} 기업")
        if position_data.get("title"):
            query_parts.append(f"{position_data['title']} 채용")
        if target_profile and target_profile.get("desired_position"):
            query_parts.append(f"{target_profile['desired_position']} 지원자")
            
        query = " ".join(query_parts)
        
        # Create query embedding (simplified - would use actual embedder)
        query_embedding = np.random.randn(384)  # Placeholder
        
        # Retrieve with context
        context = {
            "company_id": company_data.get("id"),
            "preferred_regions": [company_data.get("region", "general")],
            "required_attributes": {
                "industry": company_data.get("industry")
            }
        }
        
        return self.retriever.retrieve_with_context(
            query, query_embedding, context, k=5
        )
    
    def _create_generation_prompt(
        self,
        company_data: Dict[str, Any],
        position_data: Dict[str, Any],
        target_profile: Optional[Dict[str, Any]],
        retrieval_result: RetrievalResult,
        style: str
    ) -> str:
        """Create generation prompt with constraints"""
        # Get template
        prompt_config = self.prompts.get("job_posting_generation", {})
        system_prompt = prompt_config.get("system", "")
        constraints = prompt_config.get("constraints", "")
        template = prompt_config.get("template", "")
        
        # Prepare company info
        company_info = self._format_company_info(company_data)
        
        # Prepare job info
        job_info = self._format_job_info(position_data)
        
        # Prepare target profile if provided
        target_info = ""
        if target_profile:
            target_info = self._format_target_profile(target_profile)
            
        # Include previous violations if any
        violation_warning = ""
        if "_previous_violations" in company_data:
            violations = company_data["_previous_violations"]
            violation_warning = f"\n[이전 생성 시 문제점]\n" + "\n".join(violations)
            violation_warning += "\n위 문제를 반드시 수정하세요.\n"
            
        # Include relevant examples from retrieval
        examples = ""
        if retrieval_result.documents:
            examples = "\n[참고할 수 있는 유사 사례]\n"
            for doc in retrieval_result.documents[:3]:
                if "content" in doc:
                    examples += f"- {doc['content'][:200]}...\n"
                    
        # Style instructions
        style_instructions = self._get_style_instructions(style)
        
        # Combine into final prompt
        prompt = f"""
{system_prompt}

{constraints}

{violation_warning}

[회사 정보]
{company_info}

[직무 정보]  
{job_info}

{target_info}

{examples}

{style_instructions}

위 정보를 바탕으로 채용공고를 작성하세요. 제공된 정보에 없는 내용은 절대 추가하지 마세요.
"""
        
        return prompt
    
    def _format_company_info(self, company_data: Dict[str, Any]) -> str:
        """Format company information for prompt"""
        info_parts = []
        
        # Essential information
        if "name" in company_data:
            info_parts.append(f"회사명: {company_data['name']}")
        if "industry" in company_data:
            info_parts.append(f"업종: {company_data['industry']}")
        if "employee_count" in company_data:
            info_parts.append(f"직원수: {company_data['employee_count']}명")
        if "founded_year" in company_data:
            info_parts.append(f"설립연도: {company_data['founded_year']}년")
            
        # Culture and benefits
        if "culture" in company_data:
            info_parts.append(f"기업문화: {company_data['culture']}")
        if "benefits" in company_data:
            benefits = company_data["benefits"]
            if isinstance(benefits, list):
                benefits = ", ".join(benefits)
            info_parts.append(f"복지: {benefits}")
            
        # Work environment
        if "location" in company_data:
            info_parts.append(f"위치: {company_data['location']}")
        if "remote_work" in company_data:
            remote_status = "가능" if company_data["remote_work"] else "불가"
            info_parts.append(f"재택근무: {remote_status}")
            
        return "\n".join(info_parts)
    
    def _format_job_info(self, position_data: Dict[str, Any]) -> str:
        """Format position information for prompt"""
        info_parts = []
        
        if "title" in position_data:
            info_parts.append(f"포지션: {position_data['title']}")
        if "department" in position_data:
            info_parts.append(f"부서: {position_data['department']}")
        if "level" in position_data:
            info_parts.append(f"직급: {position_data['level']}")
            
        # Requirements
        if "required_experience" in position_data:
            info_parts.append(f"필요 경력: {position_data['required_experience']}년 이상")
        if "required_skills" in position_data:
            skills = ", ".join(position_data["required_skills"])
            info_parts.append(f"필수 기술: {skills}")
        if "preferred_skills" in position_data:
            skills = ", ".join(position_data["preferred_skills"])
            info_parts.append(f"우대 기술: {skills}")
            
        # Job details
        if "responsibilities" in position_data:
            resp = position_data["responsibilities"]
            if isinstance(resp, list):
                resp = "\n  - " + "\n  - ".join(resp)
            info_parts.append(f"주요 업무:\n{resp}")
            
        return "\n".join(info_parts)
    
    def _format_target_profile(self, target_profile: Dict[str, Any]) -> str:
        """Format target candidate profile for prompt"""
        info_parts = ["\n[타겟 구직자 프로필]"]
        
        if "seniority_level" in target_profile:
            info_parts.append(f"경력 수준: {target_profile['seniority_level']}")
        if "interests" in target_profile:
            interests = ", ".join(target_profile["interests"])
            info_parts.append(f"주요 관심사: {interests}")
        if "career_goals" in target_profile:
            info_parts.append(f"커리어 목표: {target_profile['career_goals']}")
            
        return "\n".join(info_parts)
    
    def _get_style_instructions(self, style: str) -> str:
        """Get style-specific instructions"""
        style_map = {
            "professional": "전문적이고 신뢰감 있는 톤으로 작성하세요.",
            "casual": "친근하고 편안한 톤으로 작성하세요.",
            "innovative": "혁신적이고 도전적인 톤으로 작성하세요.",
            "technical": "기술적 세부사항을 강조하여 작성하세요."
        }
        
        return f"\n[작성 스타일]\n{style_map.get(style, style_map['professional'])}"
    
    async def _generate_with_llm(self, prompt: str) -> Tuple[str, Dict[str, int]]:
        """Generate content using LLM"""
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.prompts["job_posting_generation"]["system"]},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            tokens_used = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            return content, tokens_used
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
    
    def generate_variations(
        self,
        base_result: GenerationResult,
        variations: List[str] = ["casual", "technical", "innovative"],
        company_data: Dict[str, Any] = None
    ) -> List[GenerationResult]:
        """
        Generate variations of a job posting
        
        Args:
            base_result: Base generation result
            variations: List of style variations
            company_data: Company data for regeneration
            
        Returns:
            List of GenerationResult
        """
        results = []
        
        for style in variations:
            try:
                # Modify prompt to create variation
                variation_prompt = f"""
아래 채용공고를 {style} 스타일로 다시 작성하세요:

[원본 채용공고]
{base_result.content}

[스타일 가이드]
{self._get_style_instructions(style)}

핵심 정보는 모두 유지하되, 톤과 표현 방식만 변경하세요.
"""
                
                # Generate variation
                content, tokens = asyncio.run(self._generate_with_llm(variation_prompt))
                
                # Validate if company data provided
                if company_data:
                    validation_result = self.boundary_validator.validate_content(
                        content, company_data
                    )
                else:
                    validation_result = ValidationResult(
                        is_valid=True,
                        violations=[],
                        confidence=0.9,
                        suggestions=[]
                    )
                
                # Create result
                result = GenerationResult(
                    content=content,
                    metadata={
                        **base_result.metadata,
                        "style": style,
                        "variation_of": base_result.metadata.get("id", "base")
                    },
                    validation_result=validation_result,
                    retrieval_context=base_result.retrieval_context,
                    prompt_used=variation_prompt,
                    model_used="gpt-4",
                    tokens_used=tokens
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to generate {style} variation: {e}")
                
        return results