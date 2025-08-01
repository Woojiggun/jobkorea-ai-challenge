"""
JobKorea AI Challenge - Complete System Showcase
Demonstrates all key features of the topological job matching system
"""
import sys
import os
from pathlib import Path
import numpy as np
import json
import time

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Mock settings to avoid pydantic issues
class MockSettings:
    openai_api_key = "sk-test"
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size = 32
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"

# Setup mock modules
import types
mock_config = types.ModuleType('config')
mock_config.settings = MockSettings()
sys.modules['config'] = mock_config
sys.modules['config.settings'] = mock_config

# Mock sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
    REAL_EMBEDDINGS = True
except ImportError:
    REAL_EMBEDDINGS = False
    
    class MockSentenceTransformer:
        def __init__(self, model_name=None):
            self.model_name = model_name or "mock-model"
            self.embedding_dimension = 384
            
        def encode(self, texts, **kwargs):
            if isinstance(texts, str):
                texts = [texts]
            embeddings = []
            for text in texts:
                seed = abs(hash(text)) % 10000
                np.random.seed(seed)
                embedding = np.random.randn(self.embedding_dimension)
                embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding)
            return np.array(embeddings)
        
        def get_sentence_embedding_dimension(self):
            return self.embedding_dimension
    
    mock_st = types.ModuleType('sentence_transformers')
    mock_st.SentenceTransformer = MockSentenceTransformer
    sys.modules['sentence_transformers'] = mock_st

# Import modules
from src.embeddings import CompanyEmbedder, CandidateEmbedder
from src.validation import FactChecker, ConsistencyValidator

# Demo data
demo_companies = [
    {
        "id": "kakao",
        "name": "카카오",
        "industry": "IT",
        "employee_count": 5000,
        "founded_year": 2014,
        "growth_rate": 15.5,
        "description": "국내 대표 IT 플랫폼 기업으로 메신저, 금융, 콘텐츠 등 다양한 서비스 제공",
        "location": "제주",
        "benefits": ["재택근무", "자율출퇴근", "안식휴가", "사내카페"],
        "tech_stack": ["Java", "Kotlin", "Python", "React", "Kubernetes"]
    },
    {
        "id": "naver",
        "name": "네이버",
        "industry": "IT",
        "employee_count": 4500,
        "founded_year": 1999,
        "growth_rate": 12.3,
        "description": "국내 최대 검색 포털 및 AI, 클라우드, 콘텐츠 플랫폼 운영",
        "location": "성남",
        "benefits": ["사내어린이집", "건강검진", "자기개발비", "해외연수"],
        "tech_stack": ["Java", "Python", "C++", "TensorFlow", "Docker"]
    },
    {
        "id": "coupang",
        "name": "쿠팡",
        "industry": "이커머스",
        "employee_count": 50000,
        "founded_year": 2010,
        "growth_rate": 25.7,
        "description": "로켓배송으로 유명한 국내 최대 이커머스 플랫폼",
        "location": "서울",
        "benefits": ["스톡옵션", "식사제공", "통근버스", "헬스케어"],
        "tech_stack": ["Java", "Python", "AWS", "React", "Spring"]
    },
    {
        "id": "toss",
        "name": "토스",
        "industry": "핀테크",
        "employee_count": 2000,
        "founded_year": 2015,
        "growth_rate": 45.2,
        "description": "간편송금으로 시작한 종합 금융 플랫폼",
        "location": "서울",
        "benefits": ["무제한휴가", "재택근무", "최신장비", "점심지원"],
        "tech_stack": ["TypeScript", "React", "Node.js", "Kotlin", "AWS"]
    }
]

demo_candidates = [
    {
        "id": "cand_1",
        "title": "백엔드 개발자",
        "years_experience": 5,
        "skills": ["Java", "Spring", "MySQL", "AWS", "Docker"],
        "education_level": "bachelor",
        "skill_count": 15,
        "preferred_industries": ["IT", "핀테크"],
        "career_trajectory": ["주니어 개발자", "백엔드 개발자", "시니어 개발자"],
        "desired_benefits": ["재택근무", "자율출퇴근"]
    },
    {
        "id": "cand_2", 
        "title": "프론트엔드 개발자",
        "years_experience": 3,
        "skills": ["React", "TypeScript", "CSS", "Redux", "Webpack"],
        "education_level": "bachelor",
        "skill_count": 12,
        "preferred_industries": ["IT", "이커머스"],
        "career_trajectory": ["웹 퍼블리셔", "프론트엔드 개발자"],
        "desired_benefits": ["자기개발비", "재택근무"]
    },
    {
        "id": "cand_3",
        "title": "데이터 사이언티스트",
        "years_experience": 7,
        "skills": ["Python", "TensorFlow", "SQL", "Spark", "Statistics"],
        "education_level": "master",
        "skill_count": 20,
        "preferred_industries": ["IT", "핀테크", "이커머스"],
        "career_trajectory": ["데이터 분석가", "ML 엔지니어", "데이터 사이언티스트"],
        "desired_benefits": ["해외연수", "자율출퇴근"]
    },
    {
        "id": "cand_4",
        "title": "DevOps 엔지니어",
        "years_experience": 4,
        "skills": ["Kubernetes", "Docker", "AWS", "Jenkins", "Python"],
        "education_level": "bachelor",
        "skill_count": 18,
        "preferred_industries": ["IT", "이커머스"],
        "career_trajectory": ["시스템 관리자", "DevOps 엔지니어"],
        "desired_benefits": ["재택근무", "최신장비"]
    }
]

def print_section(title):
    """Print section header"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}\n")

def demonstrate_embeddings():
    """Demonstrate embedding generation and features"""
    print_section("1. 임베딩 시스템 데모")
    
    company_embedder = CompanyEmbedder()
    candidate_embedder = CandidateEmbedder()
    
    # Single embedding
    company = demo_companies[0]
    print(f"회사: {company['name']} ({company['industry']})")
    
    start_time = time.time()
    embedding = company_embedder.embed(company)
    embed_time = time.time() - start_time
    
    features = company_embedder.extract_features(company)
    
    print(f"- 임베딩 차원: {embedding.shape}")
    print(f"- 생성 시간: {embed_time*1000:.1f}ms")
    print(f"- 추출된 특성: {len(features)}개")
    print(f"- 주요 특성: {list(features.keys())[:5]}")
    
    # Batch embedding
    print(f"\n배치 임베딩 테스트:")
    start_time = time.time()
    batch_embeddings = company_embedder.batch_embed(demo_companies)
    batch_time = time.time() - start_time
    
    print(f"- {len(demo_companies)}개 회사 임베딩 생성")
    print(f"- 총 시간: {batch_time:.2f}초")
    print(f"- 평균 시간: {batch_time/len(demo_companies)*1000:.1f}ms/회사")
    
    return company_embedder, candidate_embedder

def demonstrate_similarity_matching():
    """Demonstrate similarity-based matching"""
    print_section("2. 유사도 기반 매칭")
    
    company_embedder = CompanyEmbedder()
    candidate_embedder = CandidateEmbedder()
    
    # Generate embeddings
    company_embeddings = company_embedder.batch_embed(demo_companies)
    candidate_embeddings = candidate_embedder.batch_embed(demo_candidates)
    
    # Calculate similarity matrix
    similarity_matrix = np.dot(candidate_embeddings, company_embeddings.T)
    
    print("매칭 결과 (유사도 점수):")
    print(f"{'후보자':<20} {'최적 회사':<15} {'유사도':<10} {'2순위':<15} {'유사도':<10}")
    print("-" * 80)
    
    for i, candidate in enumerate(demo_candidates):
        scores = similarity_matrix[i]
        sorted_indices = np.argsort(scores)[::-1]
        
        best_company = demo_companies[sorted_indices[0]]
        second_company = demo_companies[sorted_indices[1]]
        
        print(f"{candidate['title']:<20} {best_company['name']:<15} {scores[sorted_indices[0]]:.3f}      "
              f"{second_company['name']:<15} {scores[sorted_indices[1]]:.3f}")

def demonstrate_feature_analysis():
    """Demonstrate feature extraction and analysis"""
    print_section("3. 특성 분석 시스템")
    
    company_embedder = CompanyEmbedder()
    candidate_embedder = CandidateEmbedder()
    
    # Analyze company features
    print("회사별 주요 특성:")
    for company in demo_companies[:3]:
        features = company_embedder.extract_features(company)
        print(f"\n{company['name']}:")
        print(f"  - 규모: {features.get('employee_count', 0):.0f}명 (카테고리: {features.get('size_category', 0):.0f})")
        print(f"  - 성장률: {features.get('growth_rate', 0):.1f}%")
        print(f"  - 업력: {2024 - company['founded_year']}년")
    
    # Analyze candidate potential
    print("\n\n후보자 잠재력 분석:")
    for candidate in demo_candidates:
        potential = candidate_embedder.calculate_potential_score(candidate)
        features = candidate_embedder.extract_features(candidate)
        
        print(f"\n{candidate['title']}:")
        print(f"  - 경력: {features.get('years_experience', 0):.0f}년")
        print(f"  - 스킬 수: {len(candidate['skills'])}개")
        print(f"  - 학력 점수: {features.get('education_score', 0):.1f}")
        print(f"  - 잠재력 점수: {potential:.3f}")

def demonstrate_validation():
    """Demonstrate content validation and hallucination detection"""
    print_section("4. 검증 시스템 데모")
    
    fact_checker = FactChecker()
    consistency_validator = ConsistencyValidator()
    
    company = demo_companies[0]  # 카카오
    
    # Good content
    good_content = f"""
    {company['name']} 백엔드 개발자 채용
    
    {company['description']}
    
    우리는 {company['founded_year']}년에 설립되어 현재 {company['employee_count']}명의 
    직원과 함께 연 {company['growth_rate']}% 성장하고 있습니다.
    
    근무지: {company['location']}
    기술스택: {', '.join(company['tech_stack'][:3])}
    """
    
    # Bad content (with hallucinations)
    bad_content = f"""
    {company['name']} - 세계 최고의 기업!
    
    1990년 설립된 우리 회사는 10만명의 직원과 함께 
    연 500% 성장을 기록하고 있습니다.
    
    미국 실리콘밸리에 본사를 두고 있으며, 
    구글과 애플을 인수한 경험이 있습니다.
    """
    
    print("정상 콘텐츠 검증:")
    fact_result = fact_checker.check_facts(good_content, company)
    consistency_result = consistency_validator.validate_consistency(good_content)
    
    print(f"- 팩트 체크: {fact_result.facts_verified}/{fact_result.facts_checked} 검증됨")
    print(f"- 일관성 점수: {consistency_result.consistency_score:.3f}")
    if consistency_result.inconsistencies:
        for issue in consistency_result.inconsistencies:
            print(f"  ! {issue}")
    
    print("\n환각 콘텐츠 검증:")
    fact_result = fact_checker.check_facts(bad_content, company)
    consistency_result = consistency_validator.validate_consistency(bad_content)
    
    print(f"- 팩트 체크: {fact_result.facts_verified}/{fact_result.facts_checked} 검증됨")
    print(f"- 일관성 점수: {consistency_result.consistency_score:.3f}")
    if consistency_result.inconsistencies:
        for issue in consistency_result.inconsistencies[:3]:
            print(f"  ! {issue}")

def demonstrate_performance_metrics():
    """Show performance metrics"""
    print_section("5. 성능 메트릭")
    
    company_embedder = CompanyEmbedder()
    candidate_embedder = CandidateEmbedder()
    
    # Test different batch sizes
    test_sizes = [10, 50, 100]
    
    print("배치 크기별 처리 시간:")
    print(f"{'크기':<10} {'회사 임베딩':<15} {'후보자 임베딩':<15} {'평균 시간':<15}")
    print("-" * 60)
    
    for size in test_sizes:
        # Generate test data
        test_companies = demo_companies * (size // len(demo_companies) + 1)
        test_companies = test_companies[:size]
        
        test_candidates = demo_candidates * (size // len(demo_candidates) + 1)
        test_candidates = test_candidates[:size]
        
        # Time company embeddings
        start = time.time()
        company_embedder.batch_embed(test_companies)
        company_time = time.time() - start
        
        # Time candidate embeddings
        start = time.time()
        candidate_embedder.batch_embed(test_candidates)
        candidate_time = time.time() - start
        
        avg_time = (company_time + candidate_time) / (2 * size) * 1000
        
        print(f"{size:<10} {company_time:.3f}초        {candidate_time:.3f}초         {avg_time:.1f}ms/항목")

def main():
    """Run all demonstrations"""
    print("\n" + "="*60)
    print("JobKorea AI Challenge - 위상정보 기반 채용 매칭 시스템")
    print("="*60)
    print(f"\n임베딩 모드: {'실제' if REAL_EMBEDDINGS else '모의'}")
    
    # Run demonstrations
    demonstrate_embeddings()
    demonstrate_similarity_matching()
    demonstrate_feature_analysis()
    demonstrate_validation()
    demonstrate_performance_metrics()
    
    # Summary
    print_section("시스템 요약")
    print("주요 기능:")
    print("1. 텍스트-수치 하이브리드 임베딩 시스템")
    print("2. 다차원 특성 추출 및 분석")
    print("3. 유사도 기반 양방향 매칭")
    print("4. 환각 방지 검증 시스템")
    print("5. 실시간 처리 가능한 성능")
    
    print("\n핵심 혁신:")
    print("- 위상정보 시스템을 통한 자연스러운 클러스터링")
    print("- 중력장 모델링으로 관련성 높은 매칭")
    print("- 다층 검증으로 환각 문제 해결")
    print("- 클라이언트 임베딩으로 응답 지연 최소화")
    
    if not REAL_EMBEDDINGS:
        print("\n[참고] 현재 모의 임베딩으로 실행 중입니다.")
        print("실제 임베딩을 사용하려면 sentence-transformers를 설치하세요:")
        print("  pip install sentence-transformers==2.2.2")

if __name__ == "__main__":
    main()