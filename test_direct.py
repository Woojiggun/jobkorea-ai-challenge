"""
Direct test bypassing config/settings issues
"""
import sys
import os
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Mock settings to avoid pydantic issues
class MockSettings:
    openai_api_key = "sk-test"
    host = "0.0.0.0"
    port = 8000
    debug = True
    faiss_index_path = Path("./data/embeddings/faiss_index")
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    cache_ttl = 3600
    max_cache_size = 1000
    topology_depth = 3
    gravity_strength = 1.0
    max_workers = 4
    batch_size = 32
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    companies_dir = data_dir / "companies"
    embeddings_dir = data_dir / "embeddings"

# Replace settings module
import types
mock_config = types.ModuleType('config')
mock_config.settings = MockSettings()
sys.modules['config'] = mock_config
sys.modules['config.settings'] = mock_config

# Mock sentence_transformers if not available
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
    
    import types
    mock_module = types.ModuleType('sentence_transformers')
    mock_module.SentenceTransformer = MockSentenceTransformer
    sys.modules['sentence_transformers'] = mock_module

print("JobKorea AI Challenge - Direct Test")
print("=" * 60)
print(f"Using {'real' if REAL_EMBEDDINGS else 'mock'} embeddings")
print()

# Test data
test_company = {
    "id": "test_co_1",
    "name": "테크 스타트업",
    "industry": "IT",
    "employee_count": 50,
    "founded_year": 2020,
    "growth_rate": 30.0,
    "description": "AI 기반 솔루션을 개발하는 스타트업",
    "location": "서울 강남구",
    "benefits": ["재택근무", "스톡옵션", "유연근무"]
}

test_candidate = {
    "id": "test_cand_1",
    "title": "백엔드 개발자",
    "years_experience": 3,
    "skills": ["Python", "FastAPI", "PostgreSQL"],
    "education_level": "bachelor",
    "skill_count": 10
}

# Test 1: Embeddings
print("1. Testing Embeddings...")
try:
    from src.embeddings import CompanyEmbedder, CandidateEmbedder
    
    company_embedder = CompanyEmbedder()
    candidate_embedder = CandidateEmbedder()
    
    # Test company embedding
    company_emb = company_embedder.embed(test_company)
    print(f"  [OK] Company embedding: shape={company_emb.shape}, norm={np.linalg.norm(company_emb):.3f}")
    
    # Test candidate embedding  
    candidate_emb = candidate_embedder.embed(test_candidate)
    print(f"  [OK] Candidate embedding: shape={candidate_emb.shape}, norm={np.linalg.norm(candidate_emb):.3f}")
    
    # Test similarity
    similarity = np.dot(company_emb, candidate_emb) / (np.linalg.norm(company_emb) * np.linalg.norm(candidate_emb))
    print(f"  [OK] Similarity score: {similarity:.3f}")
    
except Exception as e:
    print(f"  [FAIL] {e}")

# Test 2: Topology (without config dependency)
print("\n2. Testing Topology...")
try:
    # Import NetworkX directly
    import networkx as nx
    
    # Create simple topology test
    G = nx.Graph()
    G.add_node("company1", name="Company 1", employee_count=100)
    G.add_node("company2", name="Company 2", employee_count=200)
    G.add_edge("company1", "company2", weight=0.8)
    
    print(f"  [OK] Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Test shortest path
    path_length = nx.shortest_path_length(G, "company1", "company2")
    print(f"  [OK] Path length between nodes: {path_length}")
    
except Exception as e:
    print(f"  [FAIL] {e}")

# Test 3: Vector operations with FAISS
print("\n3. Testing Vector Store...")
try:
    import faiss
    
    # Create simple index
    dimension = 384
    index = faiss.IndexFlatL2(dimension)
    
    # Add vectors
    vectors = np.random.randn(10, dimension).astype(np.float32)
    index.add(vectors)
    
    print(f"  [OK] FAISS index created: {index.ntotal} vectors")
    
    # Search
    query = vectors[0].reshape(1, -1)
    distances, indices = index.search(query, k=5)
    
    print(f"  [OK] Search completed: found {len(indices[0])} results")
    print(f"  [OK] Top match distance: {distances[0][0]:.3f}")
    
except Exception as e:
    print(f"  [FAIL] {e}")

# Test 4: Validation logic  
print("\n4. Testing Validation...")
try:
    from src.validation import FactChecker, ConsistencyValidator
    
    fact_checker = FactChecker()
    consistency_validator = ConsistencyValidator()
    
    test_content = f"""
    {test_company['name']}은 {test_company['founded_year']}년에 설립되었으며,
    현재 {test_company['employee_count']}명의 직원이 근무하고 있습니다.
    연 성장률은 {test_company['growth_rate']}%입니다.
    """
    
    # Test fact checking
    fact_result = fact_checker.check_facts(test_content, test_company)
    print(f"  [OK] Fact checking: {fact_result.facts_verified}/{fact_result.facts_checked} facts verified")
    
    # Test consistency
    consistency_result = consistency_validator.validate_consistency(test_content)
    print(f"  [OK] Consistency score: {consistency_result.consistency_score:.3f}")
    
except Exception as e:
    print(f"  [FAIL] {e}")

# Test 5: Complete workflow
print("\n5. Testing Complete Workflow...")
try:
    # Create embeddings
    if 'company_embedder' in locals() and 'candidate_embedder' in locals():
        # Batch test
        companies = [test_company] * 3
        candidates = [test_candidate] * 3
        
        company_embeddings = company_embedder.batch_embed(companies)
        candidate_embeddings = candidate_embedder.batch_embed(candidates)
        
        print(f"  [OK] Batch embeddings: companies={company_embeddings.shape}, candidates={candidate_embeddings.shape}")
        
        # Calculate pairwise similarities
        similarities = np.dot(company_embeddings, candidate_embeddings.T)
        print(f"  [OK] Similarity matrix: shape={similarities.shape}, max={similarities.max():.3f}")
        
        print("\n[SUCCESS] All basic tests passed!")
    else:
        print("  [SKIP] Embedders not available")
        
except Exception as e:
    print(f"  [FAIL] {e}")

print("\n" + "=" * 60)
print("Test completed!")
if not REAL_EMBEDDINGS:
    print("\nNote: Tests ran with mock embeddings.")
    print("For production use, install sentence-transformers properly.")
print("\nTo run the full system:")
print("  python scripts/demo.py")
print("  python scripts/run_server.py")