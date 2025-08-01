"""
Client-specific handling for different platforms
"""
from typing import Dict, Any, Optional
from enum import Enum
import hashlib
import json
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class ClientType(Enum):
    """Client platform types"""
    MOBILE_APP = "mobile_app"
    MODERN_BROWSER = "modern_browser"
    LEGACY_BROWSER = "legacy_browser"
    API = "api"


class ClientCapabilities:
    """Client capability detection and management"""
    
    def __init__(self):
        self.capability_cache = {}
        self.embedding_strategy_map = {
            ClientType.MOBILE_APP: "local",
            ClientType.MODERN_BROWSER: "hybrid",
            ClientType.LEGACY_BROWSER: "server",
            ClientType.API: "server"
        }
        
    def detect_client_type(self, user_agent: str, headers: Dict[str, str]) -> ClientType:
        """
        Detect client type from user agent and headers
        
        Args:
            user_agent: User agent string
            headers: Request headers
            
        Returns:
            ClientType
        """
        user_agent_lower = user_agent.lower()
        
        # Check for mobile app
        if "jobkorea-app" in user_agent_lower or headers.get("X-App-Version"):
            return ClientType.MOBILE_APP
            
        # Check for modern browsers
        modern_browsers = ["chrome/9", "chrome/10", "firefox/9", "firefox/10", "safari/15", "safari/16"]
        if any(browser in user_agent_lower for browser in modern_browsers):
            return ClientType.MODERN_BROWSER
            
        # Check for API client
        if "python" in user_agent_lower or "curl" in user_agent_lower:
            return ClientType.API
            
        # Default to legacy browser
        return ClientType.LEGACY_BROWSER
    
    def get_client_capabilities(self, client_type: ClientType) -> Dict[str, Any]:
        """
        Get capabilities for client type
        
        Args:
            client_type: Type of client
            
        Returns:
            Dictionary of capabilities
        """
        capabilities = {
            ClientType.MOBILE_APP: {
                "local_embedding": True,
                "max_embedding_size": 100,  # MB
                "offline_capable": True,
                "cache_size": 500,  # MB
                "supports_wasm": False,
                "supports_workers": True,
                "batch_size": 50
            },
            ClientType.MODERN_BROWSER: {
                "local_embedding": True,
                "max_embedding_size": 50,  # MB
                "offline_capable": False,
                "cache_size": 100,  # MB
                "supports_wasm": True,
                "supports_workers": True,
                "batch_size": 20
            },
            ClientType.LEGACY_BROWSER: {
                "local_embedding": False,
                "max_embedding_size": 0,
                "offline_capable": False,
                "cache_size": 50,  # MB
                "supports_wasm": False,
                "supports_workers": False,
                "batch_size": 10
            },
            ClientType.API: {
                "local_embedding": False,
                "max_embedding_size": 0,
                "offline_capable": False,
                "cache_size": 0,
                "supports_wasm": False,
                "supports_workers": False,
                "batch_size": 100
            }
        }
        
        return capabilities.get(client_type, capabilities[ClientType.LEGACY_BROWSER])
    
    def get_embedding_strategy(self, client_type: ClientType) -> str:
        """
        Get embedding strategy for client type
        
        Args:
            client_type: Type of client
            
        Returns:
            Embedding strategy (local, hybrid, server)
        """
        return self.embedding_strategy_map.get(client_type, "server")


class ClientCache:
    """Client-side cache management"""
    
    def __init__(self):
        self.cache_config = {
            "max_age": timedelta(hours=24),
            "max_size_mb": 100,
            "eviction_policy": "lru"
        }
        
    def generate_cache_key(self, data: Dict[str, Any]) -> str:
        """
        Generate cache key for data
        
        Args:
            data: Data to cache
            
        Returns:
            Cache key
        """
        # Sort keys for consistent hashing
        sorted_data = json.dumps(data, sort_keys=True)
        return hashlib.sha256(sorted_data.encode()).hexdigest()[:16]
    
    def get_cache_headers(self, client_type: ClientType, content_type: str) -> Dict[str, str]:
        """
        Get appropriate cache headers for client
        
        Args:
            client_type: Type of client
            content_type: Type of content
            
        Returns:
            Cache headers
        """
        headers = {}
        
        if client_type == ClientType.MOBILE_APP:
            # Longer cache for mobile apps
            headers["Cache-Control"] = "private, max-age=86400"  # 24 hours
            headers["ETag"] = self._generate_etag()
            
        elif client_type == ClientType.MODERN_BROWSER:
            # Moderate cache for modern browsers
            headers["Cache-Control"] = "private, max-age=3600"  # 1 hour
            headers["ETag"] = self._generate_etag()
            
        else:
            # Minimal cache for legacy/API
            headers["Cache-Control"] = "no-cache"
            
        return headers
    
    def _generate_etag(self) -> str:
        """Generate ETag for content"""
        timestamp = datetime.now().isoformat()
        return f'W/"{hashlib.md5(timestamp.encode()).hexdigest()[:8]}"'


class ClientOptimizer:
    """Optimize responses for different clients"""
    
    def __init__(self):
        self.compression_thresholds = {
            ClientType.MOBILE_APP: 1024,  # 1KB
            ClientType.MODERN_BROWSER: 2048,  # 2KB
            ClientType.LEGACY_BROWSER: 4096,  # 4KB
            ClientType.API: 512  # 0.5KB
        }
        
    def optimize_response(
        self,
        data: Dict[str, Any],
        client_type: ClientType,
        include_embeddings: bool = True
    ) -> Dict[str, Any]:
        """
        Optimize response for client
        
        Args:
            data: Response data
            client_type: Type of client
            include_embeddings: Whether to include embeddings
            
        Returns:
            Optimized response
        """
        optimized = data.copy()
        
        # Remove embeddings for certain clients
        if not include_embeddings and "embeddings" in optimized:
            optimized["embedding_url"] = "/api/v1/embeddings/" + data.get("id", "unknown")
            del optimized["embeddings"]
            
        # Truncate large fields for mobile
        if client_type == ClientType.MOBILE_APP:
            for key, value in optimized.items():
                if isinstance(value, str) and len(value) > 1000:
                    optimized[key] = value[:1000] + "..."
                    optimized[f"{key}_truncated"] = True
                    
        # Add client-specific metadata
        optimized["_client_optimized"] = {
            "client_type": client_type.value,
            "optimization_applied": True,
            "timestamp": datetime.now().isoformat()
        }
        
        return optimized
    
    def get_batch_config(self, client_type: ClientType) -> Dict[str, int]:
        """
        Get batching configuration for client
        
        Args:
            client_type: Type of client
            
        Returns:
            Batch configuration
        """
        configs = {
            ClientType.MOBILE_APP: {
                "max_batch_size": 50,
                "batch_timeout_ms": 100,
                "max_concurrent_requests": 3
            },
            ClientType.MODERN_BROWSER: {
                "max_batch_size": 20,
                "batch_timeout_ms": 200,
                "max_concurrent_requests": 6
            },
            ClientType.LEGACY_BROWSER: {
                "max_batch_size": 10,
                "batch_timeout_ms": 500,
                "max_concurrent_requests": 2
            },
            ClientType.API: {
                "max_batch_size": 100,
                "batch_timeout_ms": 50,
                "max_concurrent_requests": 10
            }
        }
        
        return configs.get(client_type, configs[ClientType.LEGACY_BROWSER])


class ClientMetrics:
    """Track client-specific metrics"""
    
    def __init__(self):
        self.metrics = {
            "requests_by_type": {},
            "embedding_strategy_usage": {},
            "cache_hit_rates": {},
            "response_times": {}
        }
        
    def record_request(
        self,
        client_type: ClientType,
        endpoint: str,
        response_time: float,
        cache_hit: bool
    ):
        """Record client request metrics"""
        type_key = client_type.value
        
        # Increment request count
        if type_key not in self.metrics["requests_by_type"]:
            self.metrics["requests_by_type"][type_key] = 0
        self.metrics["requests_by_type"][type_key] += 1
        
        # Update response times
        if type_key not in self.metrics["response_times"]:
            self.metrics["response_times"][type_key] = []
        self.metrics["response_times"][type_key].append(response_time)
        
        # Update cache hit rate
        if type_key not in self.metrics["cache_hit_rates"]:
            self.metrics["cache_hit_rates"][type_key] = {"hits": 0, "total": 0}
        self.metrics["cache_hit_rates"][type_key]["total"] += 1
        if cache_hit:
            self.metrics["cache_hit_rates"][type_key]["hits"] += 1
            
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of client metrics"""
        summary = {
            "requests_by_type": self.metrics["requests_by_type"],
            "avg_response_times": {},
            "cache_hit_rates": {}
        }
        
        # Calculate average response times
        for client_type, times in self.metrics["response_times"].items():
            if times:
                summary["avg_response_times"][client_type] = sum(times) / len(times)
                
        # Calculate cache hit rates
        for client_type, stats in self.metrics["cache_hit_rates"].items():
            if stats["total"] > 0:
                summary["cache_hit_rates"][client_type] = (
                    stats["hits"] / stats["total"]
                )
                
        return summary