"""
Embedding system for companies and candidates
"""

from .base_embedder import BaseEmbedder
from .company_embedder import CompanyEmbedder
from .candidate_embedder import CandidateEmbedder

__all__ = ["BaseEmbedder", "CompanyEmbedder", "CandidateEmbedder"]