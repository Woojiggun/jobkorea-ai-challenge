"""
Performance testing script
"""
import time
import json
import numpy as np
from pathlib import Path
import sys
import psutil
import os
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, skipping plots")

sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.embeddings import CompanyEmbedder, CandidateEmbedder
    from src.topology import TopologyMapper, GravityField
    from src.rag import VectorStore
    from src.matching import WeightedMatcher
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure to install all dependencies: pip install -r requirements-minimal.txt")
    sys.exit(1)


class PerformanceTester:
    """Test system performance"""
    
    def __init__(self):
        self.results = {
            "embedding_times": [],
            "matching_times": [],
            "retrieval_times": [],
            "validation_times": [],
            "memory_usage": []
        }
        self.process = psutil.Process(os.getpid())
        
    def generate_test_data(self, n_companies=100, n_candidates=100):
        """Generate test data"""
        companies = []
        for i in range(n_companies):
            companies.append({
                "id": f"company_{i}",
                "name": f"회사 {i}",
                "industry": np.random.choice(["IT", "금융", "제조", "서비스"]),
                "employee_count": np.random.randint(10, 1000),
                "founded_year": np.random.randint(2000, 2024),
                "growth_rate": np.random.uniform(0, 50),
                "benefits": ["benefit1", "benefit2", "benefit3"]
            })
            
        candidates = []
        for i in range(n_candidates):
            candidates.append({
                "id": f"candidate_{i}",
                "title": np.random.choice(["개발자", "디자이너", "마케터", "기획자"]),
                "years_experience": np.random.uniform(0, 15),
                "skills": ["skill1", "skill2", "skill3"],
                "education_level": np.random.choice(["bachelor", "master", "phd"]),
                "skill_count": np.random.randint(5, 20)
            })
            
        return companies, candidates
    
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def test_embedding_performance(self, companies, candidates):
        """Test embedding generation performance"""
        print("\n=== 임베딩 성능 테스트 ===")
        
        # Memory before
        mem_before = self.get_memory_usage()
        
        company_embedder = CompanyEmbedder()
        candidate_embedder = CandidateEmbedder()
        
        # Test company embeddings
        start_time = time.time()
        company_embeddings = company_embedder.batch_embed(companies)
        company_time = time.time() - start_time
        
        print(f"기업 임베딩 ({len(companies)}개): {company_time:.2f}초")
        print(f"평균: {company_time/len(companies)*1000:.2f}ms/기업")
        
        # Test candidate embeddings
        start_time = time.time()
        candidate_embeddings = candidate_embedder.batch_embed(candidates)
        candidate_time = time.time() - start_time
        
        print(f"구직자 임베딩 ({len(candidates)}명): {candidate_time:.2f}초")
        print(f"평균: {candidate_time/len(candidates)*1000:.2f}ms/구직자")
        
        # Memory after
        mem_after = self.get_memory_usage()
        print(f"메모리 사용량: {mem_after - mem_before:.2f}MB 증가")
        
        self.results["embedding_times"].append({
            "n_companies": len(companies),
            "n_candidates": len(candidates),
            "company_time": company_time,
            "candidate_time": candidate_time
        })
        
        self.results["memory_usage"].append({
            "operation": "embeddings",
            "memory_increase": mem_after - mem_before
        })
        
        return company_embeddings, candidate_embeddings
    
    def test_vector_store_performance(self, embeddings):
        """Test vector store performance"""
        print("\n=== 벡터 스토어 성능 테스트 ===")
        
        dim = embeddings.shape[1]
        vector_store = VectorStore(dimension=dim)
        
        # Test adding
        ids = [f"id_{i}" for i in range(len(embeddings))]
        start_time = time.time()
        vector_store.add(embeddings, ids)
        add_time = time.time() - start_time
        
        print(f"벡터 추가 ({len(embeddings)}개): {add_time:.2f}초")
        
        # Test search
        query = embeddings[0]
        start_time = time.time()
        results = vector_store.search(query, k=10)
        search_time = time.time() - start_time
        
        print(f"검색 시간 (k=10): {search_time*1000:.2f}ms")
        
        # Test batch search
        queries = embeddings[:10]
        start_time = time.time()
        batch_results = vector_store.batch_search(queries, k=10)
        batch_time = time.time() - start_time
        
        print(f"배치 검색 (10개 쿼리): {batch_time*1000:.2f}ms")
        print(f"평균: {batch_time/10*1000:.2f}ms/쿼리")
        
        self.results["retrieval_times"].append({
            "n_vectors": len(embeddings),
            "add_time": add_time,
            "search_time": search_time,
            "batch_search_time": batch_time
        })
    
    def test_matching_performance(self, companies, candidates):
        """Test matching performance"""
        print("\n=== 매칭 성능 테스트 ===")
        
        # Initialize components
        company_embedder = CompanyEmbedder()
        candidate_embedder = CandidateEmbedder()
        topology = TopologyMapper()
        gravity = GravityField(topology)
        
        matcher = WeightedMatcher(
            company_embedder,
            candidate_embedder,
            topology,
            gravity
        )
        
        # Test single match
        start_time = time.time()
        match_result = matcher.match(companies[0], candidates[0])
        single_time = time.time() - start_time
        
        print(f"단일 매칭: {single_time*1000:.2f}ms")
        
        # Test batch matching
        n_matches = min(10, len(companies), len(candidates))
        start_time = time.time()
        
        for i in range(n_matches):
            matcher.match(companies[i], candidates[i])
            
        batch_time = time.time() - start_time
        
        print(f"배치 매칭 ({n_matches}개): {batch_time:.2f}초")
        print(f"평균: {batch_time/n_matches*1000:.2f}ms/매칭")
        
        self.results["matching_times"].append({
            "n_matches": n_matches,
            "single_time": single_time,
            "batch_time": batch_time
        })
    
    def test_hallucination_check_performance(self):
        """Test hallucination checking performance"""
        print("\n=== 환각 검증 성능 테스트 ===")
        
        # Sample content
        content = """
        테크 기업은 100명의 직원과 함께 성장하고 있습니다.
        2020년 설립 이후 매년 30% 성장률을 기록하고 있으며,
        서울 강남구에 위치해 있습니다.
        """ * 10  # Make it longer
        
        source_data = {
            "employee_count": 100,
            "founded_year": 2020,
            "growth_rate": 30,
            "location": "서울 강남구"
        }
        
        from src.topology import BoundaryValidator
        from src.generation import HallucinationGuard
        
        topology = TopologyMapper()
        boundary_validator = BoundaryValidator(topology)
        guard = HallucinationGuard(boundary_validator)
        
        start_time = time.time()
        result = guard.check_hallucination(content, source_data)
        check_time = time.time() - start_time
        
        print(f"환각 검증 시간: {check_time*1000:.2f}ms")
        print(f"콘텐츠 길이: {len(content)}자")
        
        self.results["validation_times"].append({
            "content_length": len(content),
            "check_time": check_time
        })
    
    def plot_results(self):
        """Plot performance results"""
        if not MATPLOTLIB_AVAILABLE:
            print("\nmatplotlib not available, skipping plot generation")
            return
            
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Performance Test Results', fontsize=16)
        
        # Embedding times
        if self.results["embedding_times"]:
            data = self.results["embedding_times"][-1]
            ax = axes[0, 0]
            ax.bar(['Companies', 'Candidates'], 
                  [data['company_time'], data['candidate_time']])
            ax.set_title('Embedding Generation Time')
            ax.set_ylabel('Time (seconds)')
        
        # Retrieval times
        if self.results["retrieval_times"]:
            data = self.results["retrieval_times"][-1]
            ax = axes[0, 1]
            ax.bar(['Add', 'Search', 'Batch Search'], 
                  [data['add_time'], data['search_time'], data['batch_search_time']])
            ax.set_title('Vector Store Operations')
            ax.set_ylabel('Time (seconds)')
        
        # Matching times
        if self.results["matching_times"]:
            data = self.results["matching_times"][-1]
            ax = axes[1, 0]
            ax.bar(['Single', 'Batch'], 
                  [data['single_time'], data['batch_time']])
            ax.set_title('Matching Performance')
            ax.set_ylabel('Time (seconds)')
        
        # Validation times
        if self.results["validation_times"]:
            data = self.results["validation_times"][-1]
            ax = axes[1, 1]
            ax.bar(['Hallucination Check'], [data['check_time']])
            ax.set_title('Validation Performance')
            ax.set_ylabel('Time (seconds)')
        
        plt.tight_layout()
        plt.savefig('performance_results.png')
        print("\n성능 결과 그래프 저장: performance_results.png")
    
    def generate_report(self):
        """Generate performance report"""
        report = """
# 성능 테스트 보고서

## 요약
- 임베딩 생성: 평균 {:.2f}ms/항목
- 벡터 검색: 평균 {:.2f}ms/쿼리
- 매칭 계산: 평균 {:.2f}ms/매칭
- 환각 검증: 평균 {:.2f}ms/문서

## 세부 결과
"""
        
        # Calculate averages
        if self.results["embedding_times"]:
            data = self.results["embedding_times"][-1]
            avg_embedding = (data['company_time'] + data['candidate_time']) / (
                data['n_companies'] + data['n_candidates']) * 1000
        else:
            avg_embedding = 0
            
        if self.results["retrieval_times"]:
            avg_search = self.results["retrieval_times"][-1]['search_time'] * 1000
        else:
            avg_search = 0
            
        if self.results["matching_times"]:
            data = self.results["matching_times"][-1]
            avg_matching = data['batch_time'] / data['n_matches'] * 1000
        else:
            avg_matching = 0
            
        if self.results["validation_times"]:
            avg_validation = self.results["validation_times"][-1]['check_time'] * 1000
        else:
            avg_validation = 0
        
        report = report.format(
            avg_embedding, avg_search, avg_matching, avg_validation
        )
        
        # Add detailed results
        report += json.dumps(self.results, indent=2, ensure_ascii=False)
        
        # Save report
        with open("performance_report.md", "w", encoding="utf-8") as f:
            f.write(report)
            
        print("\n성능 보고서 저장: performance_report.md")


def main():
    """Run performance tests"""
    print("JobKorea AI Challenge - 성능 테스트")
    print("=" * 60)
    
    tester = PerformanceTester()
    
    # Generate test data
    print("\n테스트 데이터 생성 중...")
    companies, candidates = tester.generate_test_data(
        n_companies=50,
        n_candidates=50
    )
    
    # Run tests
    embeddings, _ = tester.test_embedding_performance(companies, candidates)
    tester.test_vector_store_performance(embeddings)
    tester.test_matching_performance(companies, candidates)
    tester.test_hallucination_check_performance()
    
    # Generate results
    tester.plot_results()
    tester.generate_report()
    
    print("\n\n성능 테스트 완료!")


if __name__ == "__main__":
    main()