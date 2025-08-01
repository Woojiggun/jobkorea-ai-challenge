"""
Simple test to verify basic functionality
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

print("Testing imports...")

try:
    from src.embeddings import CompanyEmbedder, CandidateEmbedder
    print("[OK] Embeddings module imported successfully")
except ImportError as e:
    print(f"[FAIL] Failed to import embeddings: {e}")

try:
    from src.topology import TopologyMapper, GravityField, BoundaryValidator
    print("[OK] Topology module imported successfully")
except ImportError as e:
    print(f"[FAIL] Failed to import topology: {e}")

try:
    from src.matching import WeightedMatcher
    print("[OK] Matching module imported successfully")
except ImportError as e:
    print(f"[FAIL] Failed to import matching: {e}")

print("\nTesting basic functionality...")

try:
    # Test company embedder
    company_embedder = CompanyEmbedder()
    test_company = {
        "name": "Test Company",
        "industry": "IT",
        "employee_count": 50,
        "description": "A test company"
    }
    
    embedding = company_embedder.embed(test_company)
    print(f"[OK] Company embedding generated: shape {embedding.shape}")
    
    # Test candidate embedder
    candidate_embedder = CandidateEmbedder()
    test_candidate = {
        "title": "Software Engineer",
        "years_experience": 5,
        "skills": ["Python", "JavaScript"],
        "education_level": "bachelor"
    }
    
    embedding = candidate_embedder.embed(test_candidate)
    print(f"[OK] Candidate embedding generated: shape {embedding.shape}")
    
    print("\n[SUCCESS] Basic tests passed!")
    
except Exception as e:
    print(f"\n[ERROR] Test failed: {e}")
    import traceback
    traceback.print_exc()

print("\nFor full tests, run: pytest tests/")
print("For demo, run: python scripts/demo.py")
print("For performance test, run: python scripts/performance_test.py")