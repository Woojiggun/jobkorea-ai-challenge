"""
Demo script for JobKorea AI Challenge system
"""
import json
import asyncio
from pathlib import Path
import sys
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.embeddings import CompanyEmbedder, CandidateEmbedder
    from src.topology import TopologyMapper, GravityField, BoundaryValidator
    from src.matching import WeightedMatcher, BidirectionalOptimizer
    from src.validation import FactChecker, ConsistencyValidator
    from src.generation import HallucinationGuard
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def load_demo_data():
    """Load demo companies and candidates"""
    data_dir = Path(__file__).parent.parent / "data"
    
    with open(data_dir / "demo_companies.json", "r", encoding="utf-8") as f:
        companies = json.load(f)
        
    with open(data_dir / "demo_candidates.json", "r", encoding="utf-8") as f:
        candidates = json.load(f)
        
    return companies, candidates


def demo_embeddings(companies, candidates):
    """Demo embedding generation"""
    print("\n=== 임베딩 생성 데모 ===")
    
    # Initialize embedders
    company_embedder = CompanyEmbedder()
    candidate_embedder = CandidateEmbedder()
    
    # Generate embeddings for first company and candidate
    company = companies[0]
    candidate = candidates[0]
    
    print(f"\n회사: {company['name']} ({company['industry']})")
    company_embedding = company_embedder.embed(company)
    print(f"임베딩 차원: {company_embedding.shape}")
    
    print(f"\n구직자: {candidate['title']} ({candidate['years_experience']}년 경력)")
    candidate_embedding = candidate_embedder.embed(candidate)
    print(f"임베딩 차원: {candidate_embedding.shape}")
    
    # Calculate similarity
    similarity = np.dot(company_embedding, candidate_embedding) / (
        np.linalg.norm(company_embedding) * np.linalg.norm(candidate_embedding)
    )
    print(f"\n코사인 유사도: {similarity:.3f}")
    
    return company_embedder, candidate_embedder


def demo_topology(companies, company_embedder):
    """Demo topology system"""
    print("\n\n=== 위상정보 시스템 데모 ===")
    
    # Create topology
    topology = TopologyMapper()
    
    # Add companies as nodes
    for company in companies:
        embedding = company_embedder.embed(company)
        topology.add_node(
            company["id"],
            company["name"],
            company,
            embedding
        )
    
    # Add edges based on similarity
    for i, company1 in enumerate(companies):
        for j, company2 in enumerate(companies[i+1:], i+1):
            # Simple similarity based on industry and size
            if company1["industry"] == company2["industry"]:
                weight = 0.8
            elif abs(company1["employee_count"] - company2["employee_count"]) < 100:
                weight = 0.6
            else:
                weight = 0.3
                
            topology.add_edge(company1["id"], company2["id"], weight)
    
    # Show regions
    print("\n발견된 지역:")
    for region, nodes in topology.regions.items():
        print(f"- {region}: {len(nodes)}개 노드")
    
    # Test gravity field
    gravity = GravityField(topology)
    centers = gravity.identify_mass_centers()
    print(f"\n중력 중심: {len(centers)}개 발견")
    for center in centers:
        print(f"- {center.name}: 질량 {center.mass:.1f}")
    
    return topology, gravity


def demo_matching(companies, candidates, company_embedder, candidate_embedder, topology, gravity):
    """Demo matching system"""
    print("\n\n=== 매칭 시스템 데모 ===")
    
    # Initialize matcher
    matcher = WeightedMatcher(
        company_embedder, 
        candidate_embedder,
        topology,
        gravity
    )
    
    # Match first company with all candidates
    company = companies[0]
    print(f"\n{company['name']} 기업과 구직자 매칭:")
    
    matches = []
    for candidate in candidates:
        match_result = matcher.match(company, candidate)
        matches.append((candidate["title"], match_result))
    
    # Sort by score
    matches.sort(key=lambda x: x[1].score.total_score, reverse=True)
    
    # Show top matches
    for title, result in matches[:3]:
        print(f"\n- {title}")
        print(f"  총점: {result.score.total_score:.3f}")
        print(f"  스펙 매칭: {result.score.spec_match:.3f}")
        print(f"  잠재력: {result.score.potential_match:.3f}")
        print(f"  문화 적합성: {result.score.culture_match:.3f}")
        print(f"  경력 궤적: {result.score.trajectory_match:.3f}")
        print(f"  설명: {result.score.explanation}")
    
    # Demo bidirectional optimization
    print("\n\n양방향 최적화:")
    optimizer = BidirectionalOptimizer(matcher)
    
    optimization_result = optimizer.optimize(
        companies[:3], 
        candidates[:3]
    )
    
    print(f"\n최적 매칭 결과:")
    for company_id, candidate_id, score in optimization_result.optimal_matches:
        company_name = next(c["name"] for c in companies if c["id"] == company_id)
        candidate_title = next(c["title"] for c in candidates if c["id"] == candidate_id)
        print(f"- {company_name} ↔ {candidate_title}: {score:.3f}")
    
    return matcher


def demo_validation(companies):
    """Demo validation system"""
    print("\n\n=== 검증 시스템 데모 ===")
    
    # Sample generated content
    company = companies[0]
    good_content = f"""
    {company['name']} - 백엔드 개발자 채용
    
    {company['description']}
    {company['founded_year']}년 설립 이후 {company['employee_count']}명의 직원과 함께
    연 {company['growth_rate']}% 성장을 이어가고 있습니다.
    
    위치: {company['location']}
    """
    
    bad_content = f"""
    {company['name']} - 업계 최고의 기업!
    
    국내 유일의 기술력을 보유한 최고의 회사입니다.
    2015년 설립 이후 500명의 직원과 함께 300% 성장했습니다.
    """
    
    # Initialize validators
    topology = TopologyMapper()
    boundary_validator = BoundaryValidator(topology)
    fact_checker = FactChecker()
    consistency_validator = ConsistencyValidator()
    hallucination_guard = HallucinationGuard(boundary_validator)
    
    print("\n좋은 컨텐츠 검증:")
    fact_result = fact_checker.check_facts(good_content, company)
    print(f"- 팩트 체크: {fact_result.facts_verified}/{fact_result.facts_checked} 검증됨")
    
    consistency_result = consistency_validator.validate_consistency(good_content)
    print(f"- 일관성 점수: {consistency_result.consistency_score:.3f}")
    
    hallucination_result = hallucination_guard.check_hallucination(good_content, company)
    print(f"- 환각 점수: {hallucination_result.hallucination_score:.3f}")
    
    print("\n나쁜 컨텐츠 검증:")
    fact_result = fact_checker.check_facts(bad_content, company)
    print(f"- 팩트 체크: {fact_result.facts_verified}/{fact_result.facts_checked} 검증됨")
    
    consistency_result = consistency_validator.validate_consistency(bad_content)
    print(f"- 일관성 점수: {consistency_result.consistency_score:.3f}")
    
    hallucination_result = hallucination_guard.check_hallucination(bad_content, company)
    print(f"- 환각 점수: {hallucination_result.hallucination_score:.3f}")
    print(f"- 발견된 문제: {len(hallucination_result.detected_issues)}개")
    
    for issue in hallucination_result.detected_issues[:3]:
        print(f"  • {issue['description']}")


def main():
    """Run all demos"""
    print("JobKorea AI Challenge - 위상정보 기반 채용 매칭 시스템 데모")
    print("=" * 60)
    
    # Load data
    companies, candidates = load_demo_data()
    print(f"\n데이터 로드 완료:")
    print(f"- 기업: {len(companies)}개")
    print(f"- 구직자: {len(candidates)}명")
    
    # Run demos
    company_embedder, candidate_embedder = demo_embeddings(companies, candidates)
    topology, gravity = demo_topology(companies, company_embedder)
    matcher = demo_matching(
        companies, candidates, 
        company_embedder, candidate_embedder,
        topology, gravity
    )
    demo_validation(companies)
    
    print("\n\n데모 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()