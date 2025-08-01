"""
FastAPI server for job matching system
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import asyncio

from config.settings import settings

# Import system components (lazy loading in practice)
# from src.embeddings import CompanyEmbedder, CandidateEmbedder
# from src.topology import TopologyMapper, GravityField, BoundaryValidator
# from src.rag import VectorStore, TopologyAwareRetriever
# from src.matching import WeightedMatcher, BidirectionalOptimizer
# from src.generation import LLMGenerator, HallucinationGuard
# from src.validation import FactChecker, ConsistencyValidator

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="JobKorea AI Challenge - Topological Job Matching",
    description="Advanced job matching system with hallucination prevention",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class CompanyData(BaseModel):
    """Company information model"""
    id: Optional[str] = None
    name: str
    industry: str
    employee_count: int = Field(gt=0)
    founded_year: Optional[int] = None
    location: Optional[str] = None
    culture: Optional[str] = None
    benefits: Optional[List[str]] = None
    remote_work: bool = False
    growth_rate: Optional[float] = None
    description: Optional[str] = None


class CandidateData(BaseModel):
    """Candidate information model"""
    id: Optional[str] = None
    title: str
    years_experience: float = Field(ge=0)
    skills: List[str]
    education_level: Optional[str] = None
    desired_position: Optional[str] = None
    remote_preference: float = Field(ge=0, le=1)
    expected_salary: Optional[int] = None
    interests: Optional[List[str]] = None


class PositionData(BaseModel):
    """Position requirements model"""
    title: str
    department: Optional[str] = None
    level: Optional[str] = None
    required_experience: int = Field(ge=0)
    required_skills: List[str]
    preferred_skills: Optional[List[str]] = None
    responsibilities: Optional[List[str]] = None
    min_salary: Optional[int] = None
    max_salary: Optional[int] = None


class GenerationRequest(BaseModel):
    """Job posting generation request"""
    company: CompanyData
    position: PositionData
    target_profile: Optional[CandidateData] = None
    style: str = "professional"
    use_rag: bool = True
    strict_mode: bool = True


class MatchingRequest(BaseModel):
    """Matching request"""
    companies: List[CompanyData]
    candidates: List[CandidateData]
    strategy: str = "bidirectional"  # or "weighted"
    top_k: int = Field(default=10, gt=0, le=50)


class EmbeddingRequest(BaseModel):
    """Embedding generation request"""
    data_type: str = Field(..., pattern="^(company|candidate)$")
    data: Dict[str, Any]
    save_to_store: bool = False


# Response models
class GenerationResponse(BaseModel):
    """Job posting generation response"""
    content: str
    validation_passed: bool
    confidence: float
    violations: List[str]
    metadata: Dict[str, Any]
    tokens_used: Dict[str, int]


class MatchingResponse(BaseModel):
    """Matching response"""
    matches: List[Dict[str, Any]]
    total_matches: int
    avg_match_score: float
    processing_time: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    components: Dict[str, str]


# System state (in production, use proper state management)
system_state = {
    "initialized": False,
    "components": {},
    "stats": {
        "requests_processed": 0,
        "generations_completed": 0,
        "matches_computed": 0
    }
}


@app.on_event("startup")
async def startup_event():
    """Initialize system components on startup"""
    logger.info("Starting JobKorea AI Challenge API server...")
    
    # In production, initialize all components here
    # system_state["components"]["embedders"] = initialize_embedders()
    # system_state["components"]["topology"] = initialize_topology()
    # etc.
    
    system_state["initialized"] = True
    logger.info("API server started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API server...")
    # Cleanup resources


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        components={
            "api": "running",
            "system": "initialized" if system_state["initialized"] else "not_initialized"
        }
    )


@app.post("/api/v1/generate", response_model=GenerationResponse)
async def generate_job_posting(
    request: GenerationRequest,
    background_tasks: BackgroundTasks
):
    """Generate a job posting with hallucination prevention"""
    try:
        # Track stats
        system_state["stats"]["requests_processed"] += 1
        
        # Mock response for now (would use actual LLMGenerator)
        mock_content = f"""
{request.company.name} - {request.position.title} 채용

회사 소개:
{request.company.name}는 {request.company.industry} 분야의 선도 기업입니다.
현재 {request.company.employee_count}명의 직원과 함께 성장하고 있습니다.

모집 포지션: {request.position.title}
필요 경력: {request.position.required_experience}년 이상
필수 기술: {', '.join(request.position.required_skills)}

우대사항:
{', '.join(request.position.preferred_skills or [])}

주요 업무:
{chr(10).join('- ' + r for r in (request.position.responsibilities or []))}

지원 방법:
이메일로 이력서를 보내주세요.
"""
        
        # Mock validation
        validation_passed = True
        violations = []
        
        if request.strict_mode and request.company.employee_count > 10000:
            violations.append("Large employee count requires verification")
            
        response = GenerationResponse(
            content=mock_content,
            validation_passed=validation_passed,
            confidence=0.95 if validation_passed else 0.7,
            violations=violations,
            metadata={
                "company_id": request.company.id,
                "position": request.position.title,
                "generated_at": datetime.now().isoformat()
            },
            tokens_used={
                "prompt_tokens": 500,
                "completion_tokens": 300,
                "total_tokens": 800
            }
        )
        
        # Background task to update stats
        background_tasks.add_task(
            update_generation_stats,
            request.company.id,
            len(mock_content)
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/match", response_model=MatchingResponse)
async def compute_matches(request: MatchingRequest):
    """Compute optimal matches between companies and candidates"""
    try:
        start_time = datetime.now()
        
        # Mock matching (would use actual WeightedMatcher/BidirectionalOptimizer)
        matches = []
        
        for i, company in enumerate(request.companies[:request.top_k]):
            for j, candidate in enumerate(request.candidates[:request.top_k]):
                # Mock match score
                score = 0.8 - (abs(i - j) * 0.1)
                
                matches.append({
                    "company_id": company.id or f"company_{i}",
                    "company_name": company.name,
                    "candidate_id": candidate.id or f"candidate_{j}",
                    "candidate_title": candidate.title,
                    "match_score": score,
                    "match_type": request.strategy
                })
                
        # Sort by score
        matches.sort(key=lambda x: x["match_score"], reverse=True)
        matches = matches[:request.top_k]
        
        # Calculate metrics
        avg_score = sum(m["match_score"] for m in matches) / len(matches) if matches else 0
        processing_time = (datetime.now() - start_time).total_seconds()
        
        system_state["stats"]["matches_computed"] += len(matches)
        
        return MatchingResponse(
            matches=matches,
            total_matches=len(matches),
            avg_match_score=avg_score,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Matching failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/embed")
async def generate_embedding(request: EmbeddingRequest):
    """Generate embeddings for company or candidate data"""
    try:
        # Mock embedding (would use actual embedders)
        mock_embedding = [0.1] * 384  # Standard embedding size
        
        response = {
            "data_type": request.data_type,
            "embedding_dim": len(mock_embedding),
            "embedding": mock_embedding[:10] + ["..."],  # Truncated for response
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "saved": request.save_to_store
            }
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/stats")
async def get_statistics():
    """Get system statistics"""
    return {
        "stats": system_state["stats"],
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/v1/validate")
async def validate_content(content: str, company_data: CompanyData):
    """Validate generated content for hallucinations"""
    try:
        # Mock validation (would use actual validators)
        is_valid = True
        violations = []
        
        # Simple checks
        if "최고" in content or "유일" in content:
            violations.append("Absolute claims detected")
            is_valid = False
            
        if str(company_data.employee_count) not in content:
            violations.append("Employee count mismatch")
            
        return {
            "is_valid": is_valid,
            "violations": violations,
            "confidence": 0.9 if is_valid else 0.6,
            "suggestions": [
                "Remove absolute claims",
                "Verify all numeric values"
            ] if violations else []
        }
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background tasks
async def update_generation_stats(company_id: str, content_length: int):
    """Update generation statistics"""
    system_state["stats"]["generations_completed"] += 1
    logger.info(f"Generated content for {company_id}: {content_length} chars")


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return HTTPException(status_code=400, detail=str(exc))


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)