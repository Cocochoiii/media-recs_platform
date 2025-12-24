"""
Caching Utilities for Recommendation System

Redis-based caching for fast inference and feature serving.
"""

import json
import pickle
import hashlib
import logging
from typing import Any, Dict, List, Optional, Union
from functools import wraps
import time
from dataclasses import dataclass
import asyncio

try:
    import redis
    from redis import asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Cache configuration."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    default_ttl: int = 3600  # 1 hour
    max_connections: int = 50
    socket_timeout: int = 5
    prefix: str = "recommender:"


class RedisCache:
    """
    Redis cache wrapper for recommendation data.
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self._client: Optional[redis.Redis] = None
        self._connected = False
    
    def connect(self) -> bool:
        """Establish Redis connection."""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available. Caching disabled.")
            return False
        
        try:
            self._client = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                socket_timeout=self.config.socket_timeout,
                decode_responses=False
            )
            self._client.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {self.config.host}:{self.config.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            return False
    
    def _make_key(self, key: str) -> str:
        """Create prefixed cache key."""
        return f"{self.config.prefix}{key}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self._connected:
            return None
        
        try:
            data = self._client.get(self._make_key(key))
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache."""
        if not self._connected:
            return False
        
        try:
            ttl = ttl or self.config.default_ttl
            data = pickle.dumps(value)
            self._client.setex(
                self._make_key(key),
                ttl,
                data
            )
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self._connected:
            return False
        
        try:
            self._client.delete(self._make_key(key))
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache."""
        if not self._connected:
            return {}
        
        try:
            prefixed_keys = [self._make_key(k) for k in keys]
            values = self._client.mget(prefixed_keys)
            
            result = {}
            for key, value in zip(keys, values):
                if value:
                    result[key] = pickle.loads(value)
            
            return result
        except Exception as e:
            logger.error(f"Cache get_many error: {e}")
            return {}
    
    def set_many(
        self, 
        mapping: Dict[str, Any], 
        ttl: Optional[int] = None
    ) -> bool:
        """Set multiple values in cache."""
        if not self._connected:
            return False
        
        try:
            ttl = ttl or self.config.default_ttl
            pipe = self._client.pipeline()
            
            for key, value in mapping.items():
                data = pickle.dumps(value)
                pipe.setex(self._make_key(key), ttl, data)
            
            pipe.execute()
            return True
        except Exception as e:
            logger.error(f"Cache set_many error: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern."""
        if not self._connected:
            return 0
        
        try:
            full_pattern = self._make_key(pattern)
            keys = self._client.keys(full_pattern)
            if keys:
                return self._client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Cache clear_pattern error: {e}")
            return 0


class RecommendationCache:
    """
    Specialized cache for recommendation results.
    """
    
    def __init__(self, cache: RedisCache):
        self.cache = cache
        self.user_recs_ttl = 300  # 5 minutes
        self.item_embeddings_ttl = 3600  # 1 hour
        self.popular_items_ttl = 600  # 10 minutes
    
    def _user_recs_key(self, user_id: int, model: str = "hybrid") -> str:
        """Generate key for user recommendations."""
        return f"recs:user:{user_id}:model:{model}"
    
    def _item_embedding_key(self, item_id: int) -> str:
        """Generate key for item embedding."""
        return f"embedding:item:{item_id}"
    
    def get_user_recommendations(
        self, 
        user_id: int, 
        model: str = "hybrid"
    ) -> Optional[List[tuple]]:
        """Get cached recommendations for user."""
        return self.cache.get(self._user_recs_key(user_id, model))
    
    def set_user_recommendations(
        self, 
        user_id: int, 
        recommendations: List[tuple],
        model: str = "hybrid"
    ):
        """Cache recommendations for user."""
        self.cache.set(
            self._user_recs_key(user_id, model),
            recommendations,
            ttl=self.user_recs_ttl
        )
    
    def invalidate_user_recommendations(self, user_id: int):
        """Invalidate all cached recommendations for user."""
        pattern = f"recs:user:{user_id}:*"
        self.cache.clear_pattern(pattern)
    
    def get_item_embedding(self, item_id: int) -> Optional[np.ndarray]:
        """Get cached item embedding."""
        return self.cache.get(self._item_embedding_key(item_id))
    
    def set_item_embedding(self, item_id: int, embedding: np.ndarray):
        """Cache item embedding."""
        self.cache.set(
            self._item_embedding_key(item_id),
            embedding,
            ttl=self.item_embeddings_ttl
        )
    
    def get_item_embeddings_batch(
        self, 
        item_ids: List[int]
    ) -> Dict[int, np.ndarray]:
        """Get multiple item embeddings."""
        keys = [self._item_embedding_key(i) for i in item_ids]
        results = self.cache.get_many(keys)
        
        return {
            int(k.split(":")[-1]): v 
            for k, v in results.items()
        }
    
    def get_popular_items(self, category: Optional[str] = None) -> Optional[List[tuple]]:
        """Get cached popular items."""
        key = f"popular:{category or 'all'}"
        return self.cache.get(key)
    
    def set_popular_items(
        self, 
        items: List[tuple], 
        category: Optional[str] = None
    ):
        """Cache popular items."""
        key = f"popular:{category or 'all'}"
        self.cache.set(key, items, ttl=self.popular_items_ttl)


class InMemoryCache:
    """
    Simple in-memory cache for when Redis is unavailable.
    """
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._cache: Dict[str, tuple] = {}  # key -> (value, expiry_time)
        self._access_order: List[str] = []
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self._cache:
            return None
        
        value, expiry = self._cache[key]
        
        # Check expiry
        if expiry and time.time() > expiry:
            del self._cache[key]
            return None
        
        # Update access order
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        
        return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache."""
        # Evict if at capacity
        while len(self._cache) >= self.max_size:
            oldest_key = self._access_order.pop(0)
            if oldest_key in self._cache:
                del self._cache[oldest_key]
        
        expiry = time.time() + ttl if ttl else None
        self._cache[key] = (value, expiry)
        self._access_order.append(key)
    
    def delete(self, key: str):
        """Delete key from cache."""
        if key in self._cache:
            del self._cache[key]
        if key in self._access_order:
            self._access_order.remove(key)
    
    def clear(self):
        """Clear all cache."""
        self._cache.clear()
        self._access_order.clear()


def cached(
    cache: Union[RedisCache, InMemoryCache],
    key_prefix: str,
    ttl: int = 3600
):
    """
    Decorator for caching function results.
    
    Usage:
        @cached(cache, "my_function", ttl=300)
        def my_function(arg1, arg2):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from arguments
            key_parts = [key_prefix]
            key_parts.extend(str(a) for a in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(key_parts)
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Compute and cache
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl=ttl)
            
            return result
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # Example usage
    config = CacheConfig(host="localhost", port=6379)
    cache = RedisCache(config)
    
    if cache.connect():
        rec_cache = RecommendationCache(cache)
        
        # Cache some recommendations
        recs = [(1, 0.95), (2, 0.87), (3, 0.82)]
        rec_cache.set_user_recommendations(user_id=123, recommendations=recs)
        
        # Retrieve
        cached_recs = rec_cache.get_user_recommendations(user_id=123)
        print(f"Cached recommendations: {cached_recs}")
    else:
        # Fall back to in-memory cache
        mem_cache = InMemoryCache()
        mem_cache.set("test", "value", ttl=60)
        print(f"In-memory cache: {mem_cache.get('test')}")
