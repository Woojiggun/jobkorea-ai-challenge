"""
Bidirectional optimization for mutual benefit matching
"""
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
import logging
from scipy.optimize import linear_sum_assignment

from .weighted_matcher import WeightedMatcher, MatchResult

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of bidirectional optimization"""
    optimal_matches: List[Tuple[str, str, float]]
    total_satisfaction: float
    company_satisfaction: Dict[str, float]
    candidate_satisfaction: Dict[str, float]
    unmatched_companies: List[str]
    unmatched_candidates: List[str]
    metrics: Dict[str, Any]


class BidirectionalOptimizer:
    """
    Optimizes matches considering both company and candidate preferences
    """
    
    def __init__(self, weighted_matcher: WeightedMatcher):
        """
        Initialize optimizer
        
        Args:
            weighted_matcher: WeightedMatcher instance
        """
        self.weighted_matcher = weighted_matcher
        
    def optimize(
        self,
        companies: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        company_preferences: Optional[Dict[str, List[str]]] = None,
        candidate_preferences: Optional[Dict[str, List[str]]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> OptimizationResult:
        """
        Find optimal matching considering mutual preferences
        
        Args:
            companies: List of company data
            candidates: List of candidate data
            company_preferences: Optional preference rankings by companies
            candidate_preferences: Optional preference rankings by candidates
            constraints: Optional matching constraints
            
        Returns:
            OptimizationResult with optimal assignments
        """
        # Build preference matrices
        company_matrix, candidate_matrix = self._build_preference_matrices(
            companies, candidates, company_preferences, candidate_preferences
        )
        
        # Create combined satisfaction matrix
        satisfaction_matrix = self._compute_satisfaction_matrix(
            company_matrix, candidate_matrix, constraints
        )
        
        # Solve assignment problem
        optimal_assignment = self._solve_assignment(
            satisfaction_matrix, constraints
        )
        
        # Extract results
        result = self._extract_results(
            optimal_assignment, companies, candidates, 
            satisfaction_matrix, company_matrix, candidate_matrix
        )
        
        return result
    
    def _build_preference_matrices(
        self,
        companies: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        company_preferences: Optional[Dict[str, List[str]]],
        candidate_preferences: Optional[Dict[str, List[str]]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build preference matrices for companies and candidates"""
        n_companies = len(companies)
        n_candidates = len(candidates)
        
        # Initialize matrices
        company_matrix = np.zeros((n_companies, n_candidates))
        candidate_matrix = np.zeros((n_companies, n_candidates))
        
        # Fill company preferences
        for i, company in enumerate(companies):
            company_id = company.get("id", f"company_{i}")
            
            for j, candidate in enumerate(candidates):
                candidate_id = candidate.get("id", f"candidate_{j}")
                
                # Use provided preferences if available
                if company_preferences and company_id in company_preferences:
                    pref_list = company_preferences[company_id]
                    if candidate_id in pref_list:
                        # Higher score for higher preference
                        rank = pref_list.index(candidate_id)
                        company_matrix[i, j] = 1.0 - (rank / len(pref_list))
                    else:
                        company_matrix[i, j] = 0.0
                else:
                    # Calculate match score
                    match_result = self.weighted_matcher.match(company, candidate)
                    company_matrix[i, j] = match_result.score.total_score
        
        # Fill candidate preferences
        for i, company in enumerate(companies):
            company_id = company.get("id", f"company_{i}")
            
            for j, candidate in enumerate(candidates):
                candidate_id = candidate.get("id", f"candidate_{j}")
                
                # Use provided preferences if available
                if candidate_preferences and candidate_id in candidate_preferences:
                    pref_list = candidate_preferences[candidate_id]
                    if company_id in pref_list:
                        rank = pref_list.index(company_id)
                        candidate_matrix[i, j] = 1.0 - (rank / len(pref_list))
                    else:
                        candidate_matrix[i, j] = 0.0
                else:
                    # Use same match score (symmetric for now)
                    candidate_matrix[i, j] = company_matrix[i, j]
        
        return company_matrix, candidate_matrix
    
    def _compute_satisfaction_matrix(
        self,
        company_matrix: np.ndarray,
        candidate_matrix: np.ndarray,
        constraints: Optional[Dict[str, Any]]
    ) -> np.ndarray:
        """Compute combined satisfaction matrix"""
        # Default: equal weight to both sides
        alpha = 0.5
        if constraints and "company_weight" in constraints:
            alpha = constraints["company_weight"]
        
        # Combine matrices
        satisfaction_matrix = (
            alpha * company_matrix + 
            (1 - alpha) * candidate_matrix
        )
        
        # Apply constraints
        if constraints:
            satisfaction_matrix = self._apply_constraints(
                satisfaction_matrix, constraints
            )
        
        return satisfaction_matrix
    
    def _apply_constraints(
        self,
        matrix: np.ndarray,
        constraints: Dict[str, Any]
    ) -> np.ndarray:
        """Apply matching constraints to satisfaction matrix"""
        modified_matrix = matrix.copy()
        
        # Minimum score threshold
        if "min_score" in constraints:
            min_score = constraints["min_score"]
            modified_matrix[modified_matrix < min_score] = -1
        
        # Forbidden matches
        if "forbidden_matches" in constraints:
            for company_idx, candidate_idx in constraints["forbidden_matches"]:
                modified_matrix[company_idx, candidate_idx] = -1
        
        # Required matches
        if "required_matches" in constraints:
            for company_idx, candidate_idx in constraints["required_matches"]:
                # Boost score to ensure matching
                modified_matrix[company_idx, candidate_idx] = 2.0
        
        return modified_matrix
    
    def _solve_assignment(
        self,
        satisfaction_matrix: np.ndarray,
        constraints: Optional[Dict[str, Any]]
    ) -> List[Tuple[int, int]]:
        """Solve the assignment problem"""
        # Handle rectangular matrices (different number of companies and candidates)
        n_companies, n_candidates = satisfaction_matrix.shape
        
        if n_companies != n_candidates:
            # Pad with dummy entries
            max_dim = max(n_companies, n_candidates)
            padded_matrix = np.full((max_dim, max_dim), -1)
            padded_matrix[:n_companies, :n_candidates] = satisfaction_matrix
            satisfaction_matrix = padded_matrix
        
        # Convert to cost matrix (we want to maximize satisfaction)
        cost_matrix = -satisfaction_matrix
        
        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Filter out dummy assignments
        assignments = []
        for row, col in zip(row_indices, col_indices):
            if row < n_companies and col < n_candidates:
                if satisfaction_matrix[row, col] > 0:  # Valid match
                    assignments.append((row, col))
        
        return assignments
    
    def _extract_results(
        self,
        assignments: List[Tuple[int, int]],
        companies: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        satisfaction_matrix: np.ndarray,
        company_matrix: np.ndarray,
        candidate_matrix: np.ndarray
    ) -> OptimizationResult:
        """Extract and format optimization results"""
        # Extract matches with scores
        optimal_matches = []
        company_satisfaction = {}
        candidate_satisfaction = {}
        
        matched_companies = set()
        matched_candidates = set()
        
        for company_idx, candidate_idx in assignments:
            company_id = companies[company_idx].get("id", f"company_{company_idx}")
            candidate_id = candidates[candidate_idx].get("id", f"candidate_{candidate_idx}")
            
            score = satisfaction_matrix[company_idx, candidate_idx]
            optimal_matches.append((company_id, candidate_id, score))
            
            company_satisfaction[company_id] = company_matrix[company_idx, candidate_idx]
            candidate_satisfaction[candidate_id] = candidate_matrix[company_idx, candidate_idx]
            
            matched_companies.add(company_id)
            matched_candidates.add(candidate_id)
        
        # Identify unmatched entities
        all_companies = {c.get("id", f"company_{i}") for i, c in enumerate(companies)}
        all_candidates = {c.get("id", f"candidate_{i}") for i, c in enumerate(candidates)}
        
        unmatched_companies = list(all_companies - matched_companies)
        unmatched_candidates = list(all_candidates - matched_candidates)
        
        # Calculate total satisfaction
        total_satisfaction = sum(score for _, _, score in optimal_matches)
        
        # Calculate metrics
        metrics = {
            "match_rate": len(optimal_matches) / min(len(companies), len(candidates)),
            "avg_satisfaction": total_satisfaction / len(optimal_matches) if optimal_matches else 0,
            "satisfaction_variance": np.var([score for _, _, score in optimal_matches]) if optimal_matches else 0,
            "company_match_rate": len(matched_companies) / len(companies),
            "candidate_match_rate": len(matched_candidates) / len(candidates)
        }
        
        return OptimizationResult(
            optimal_matches=optimal_matches,
            total_satisfaction=total_satisfaction,
            company_satisfaction=company_satisfaction,
            candidate_satisfaction=candidate_satisfaction,
            unmatched_companies=unmatched_companies,
            unmatched_candidates=unmatched_candidates,
            metrics=metrics
        )
    
    def stable_matching(
        self,
        companies: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        company_capacity: Optional[Dict[str, int]] = None
    ) -> OptimizationResult:
        """
        Implement stable matching (Gale-Shapley algorithm variant)
        
        Args:
            companies: List of company data
            candidates: List of candidate data
            company_capacity: Optional capacity for each company
            
        Returns:
            Stable matching result
        """
        # Build preference rankings
        company_prefs = self._build_preference_rankings(companies, candidates, "company")
        candidate_prefs = self._build_preference_rankings(companies, candidates, "candidate")
        
        # Initialize capacities
        if company_capacity is None:
            company_capacity = {
                c.get("id", f"company_{i}"): 1 
                for i, c in enumerate(companies)
            }
        
        # Run deferred acceptance algorithm
        matches = self._deferred_acceptance(
            company_prefs, candidate_prefs, company_capacity
        )
        
        # Convert to OptimizationResult format
        return self._convert_stable_matching_result(
            matches, companies, candidates
        )
    
    def _build_preference_rankings(
        self,
        companies: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        perspective: str
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Build preference rankings from match scores"""
        rankings = {}
        
        if perspective == "company":
            for company in companies:
                company_id = company.get("id", f"company_{companies.index(company)}")
                preferences = []
                
                for candidate in candidates:
                    candidate_id = candidate.get("id", f"candidate_{candidates.index(candidate)}")
                    match_result = self.weighted_matcher.match(company, candidate)
                    preferences.append((candidate_id, match_result.score.total_score))
                
                # Sort by score (descending)
                preferences.sort(key=lambda x: x[1], reverse=True)
                rankings[company_id] = preferences
        else:  # candidate perspective
            for candidate in candidates:
                candidate_id = candidate.get("id", f"candidate_{candidates.index(candidate)}")
                preferences = []
                
                for company in companies:
                    company_id = company.get("id", f"company_{companies.index(company)}")
                    match_result = self.weighted_matcher.match(company, candidate)
                    preferences.append((company_id, match_result.score.total_score))
                
                preferences.sort(key=lambda x: x[1], reverse=True)
                rankings[candidate_id] = preferences
        
        return rankings
    
    def _deferred_acceptance(
        self,
        company_prefs: Dict[str, List[Tuple[str, float]]],
        candidate_prefs: Dict[str, List[Tuple[str, float]]],
        company_capacity: Dict[str, int]
    ) -> Dict[str, List[str]]:
        """Implement deferred acceptance algorithm"""
        # Initialize
        company_matches = {c: [] for c in company_prefs}
        candidate_match = {}
        free_companies = set(company_prefs.keys())
        
        # Track proposals
        proposals_made = {c: 0 for c in company_prefs}
        
        while free_companies:
            # Pick a free company
            company = free_companies.pop()
            
            # Check if company has capacity and candidates to propose to
            if (len(company_matches[company]) < company_capacity.get(company, 1) and
                proposals_made[company] < len(company_prefs[company])):
                
                # Get next candidate to propose to
                candidate, score = company_prefs[company][proposals_made[company]]
                proposals_made[company] += 1
                
                # If candidate is free, accept
                if candidate not in candidate_match:
                    candidate_match[candidate] = company
                    company_matches[company].append(candidate)
                else:
                    # Candidate already matched, check preferences
                    current_company = candidate_match[candidate]
                    
                    # Find scores in candidate's preference list
                    candidate_pref_list = [c for c, _ in candidate_prefs[candidate]]
                    
                    if (company in candidate_pref_list and 
                        current_company in candidate_pref_list):
                        
                        company_rank = candidate_pref_list.index(company)
                        current_rank = candidate_pref_list.index(current_company)
                        
                        # If new company is preferred
                        if company_rank < current_rank:
                            # Remove from current company
                            company_matches[current_company].remove(candidate)
                            free_companies.add(current_company)
                            
                            # Match with new company
                            candidate_match[candidate] = company
                            company_matches[company].append(candidate)
                        else:
                            # Candidate prefers current match
                            free_companies.add(company)
                    else:
                        # Keep current match if preference unknown
                        free_companies.add(company)
                
                # Check if company still has capacity
                if (len(company_matches[company]) < company_capacity.get(company, 1) and
                    proposals_made[company] < len(company_prefs[company])):
                    free_companies.add(company)
        
        return company_matches
    
    def _convert_stable_matching_result(
        self,
        matches: Dict[str, List[str]],
        companies: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]]
    ) -> OptimizationResult:
        """Convert stable matching result to OptimizationResult"""
        optimal_matches = []
        company_satisfaction = {}
        candidate_satisfaction = {}
        
        # Create lookup dictionaries
        company_lookup = {c.get("id", f"company_{i}"): c for i, c in enumerate(companies)}
        candidate_lookup = {c.get("id", f"candidate_{i}"): c for i, c in enumerate(candidates)}
        
        matched_candidates = set()
        
        for company_id, candidate_ids in matches.items():
            for candidate_id in candidate_ids:
                if company_id in company_lookup and candidate_id in candidate_lookup:
                    # Calculate match score
                    match_result = self.weighted_matcher.match(
                        company_lookup[company_id],
                        candidate_lookup[candidate_id]
                    )
                    
                    score = match_result.score.total_score
                    optimal_matches.append((company_id, candidate_id, score))
                    
                    company_satisfaction[company_id] = score
                    candidate_satisfaction[candidate_id] = score
                    
                    matched_candidates.add(candidate_id)
        
        # Identify unmatched
        all_companies = set(company_lookup.keys())
        all_candidates = set(candidate_lookup.keys())
        
        unmatched_companies = [c for c in all_companies if c not in matches or not matches[c]]
        unmatched_candidates = list(all_candidates - matched_candidates)
        
        # Calculate metrics
        total_satisfaction = sum(score for _, _, score in optimal_matches)
        
        metrics = {
            "match_rate": len(optimal_matches) / min(len(companies), len(candidates)),
            "avg_satisfaction": total_satisfaction / len(optimal_matches) if optimal_matches else 0,
            "stability": 1.0,  # Stable matching guarantees stability
            "company_match_rate": len([c for c in matches if matches[c]]) / len(companies),
            "candidate_match_rate": len(matched_candidates) / len(candidates)
        }
        
        return OptimizationResult(
            optimal_matches=optimal_matches,
            total_satisfaction=total_satisfaction,
            company_satisfaction=company_satisfaction,
            candidate_satisfaction=candidate_satisfaction,
            unmatched_companies=unmatched_companies,
            unmatched_candidates=unmatched_candidates,
            metrics=metrics
        )