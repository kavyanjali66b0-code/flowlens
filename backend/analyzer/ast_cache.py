"""
AST caching system for improving incremental analysis performance.

This module provides disk-based caching of parsed ASTs to avoid re-parsing
unchanged files across multiple analysis runs.
"""

import os
import pickle
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import time
import json


@dataclass
class CacheEntry:
    """Represents a cached AST entry."""
    file_path: str
    file_hash: str
    mtime: float
    ast_data: Any
    metadata: Dict[str, Any]
    cached_at: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization (excluding ast_data)."""
        return {
            'file_path': self.file_path,
            'file_hash': self.file_hash,
            'mtime': self.mtime,
            'metadata': self.metadata,
            'cached_at': self.cached_at
        }


class ASTCache:
    """Disk-based cache for parsed ASTs with invalidation and size limits."""
    
    def __init__(self, 
                 cache_dir: Optional[Path] = None,
                 max_size_mb: int = 500,
                 enable_compression: bool = True):
        """Initialize AST cache.
        
        Args:
            cache_dir: Directory for cache storage (default: .flowlens-cache/ast/)
            max_size_mb: Maximum cache size in MB (default: 500)
            enable_compression: Whether to compress cached data (default: True)
        """
        if cache_dir is None:
            # Default to .flowlens-cache/ast in current working directory
            cache_dir = Path.cwd() / '.flowlens-cache' / 'ast'
        
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.enable_compression = enable_compression
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache statistics
        self.hits = 0
        self.misses = 0
        self.invalidations = 0
        
        # In-memory index for fast lookups
        self.index_file = self.cache_dir / 'cache_index.json'
        self.index: Dict[str, Dict[str, Any]] = self._load_index()
        
        logging.info(f"Initialized ASTCache at {self.cache_dir} (max_size={max_size_mb}MB)")
    
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load cache index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load cache index, starting fresh: {e}")
        return {}
    
    def _save_index(self):
        """Save cache index to disk."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save cache index: {e}")
    
    def _compute_file_hash(self, file_path: Path, content: Optional[str] = None) -> str:
        """Compute MD5 hash of file content.
        
        Args:
            file_path: Path to the file
            content: Optional pre-loaded content (to avoid re-reading)
            
        Returns:
            MD5 hash string
        """
        try:
            if content is None:
                content = file_path.read_text(encoding='utf-8')
            return hashlib.md5(content.encode('utf-8')).hexdigest()
        except Exception as e:
            logging.warning(f"Failed to compute hash for {file_path}: {e}")
            return ""
    
    def _get_cache_key(self, file_path: Path) -> str:
        """Generate cache key for a file.
        
        Uses file path hash to create a safe filename.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Cache key string
        """
        path_str = str(file_path.resolve())
        return hashlib.md5(path_str.encode('utf-8')).hexdigest()
    
    def _get_cache_file_path(self, cache_key: str) -> Path:
        """Get full path to cache file.
        
        Args:
            cache_key: Cache key from _get_cache_key()
            
        Returns:
            Path to cache file
        """
        return self.cache_dir / f"{cache_key}.pkl"
    
    def get(self, 
            file_path: Path, 
            current_content: Optional[str] = None) -> Optional[Tuple[Any, Dict[str, Any]]]:
        """Retrieve cached AST if valid.
        
        Args:
            file_path: Path to the file
            current_content: Optional current content (to avoid re-reading)
            
        Returns:
            Tuple of (ast_data, metadata) if cache hit, None if miss
        """
        try:
            cache_key = self._get_cache_key(file_path)
            
            # Check in-memory index first
            if cache_key not in self.index:
                self.misses += 1
                return None
            
            index_entry = self.index[cache_key]
            
            # Validate file hasn't changed
            if not file_path.exists():
                self._invalidate(cache_key)
                self.misses += 1
                return None
            
            current_mtime = file_path.stat().st_mtime
            current_hash = self._compute_file_hash(file_path, current_content)
            
            # Check if file changed based on mtime and hash
            if (index_entry['mtime'] != current_mtime or 
                index_entry['file_hash'] != current_hash):
                self._invalidate(cache_key)
                self.misses += 1
                logging.debug(f"Cache invalidated for {file_path} (changed)")
                return None
            
            # Load from disk
            cache_file = self._get_cache_file_path(cache_key)
            if not cache_file.exists():
                self._invalidate(cache_key)
                self.misses += 1
                return None
            
            with open(cache_file, 'rb') as f:
                entry = pickle.load(f)
            
            self.hits += 1
            logging.debug(f"Cache hit for {file_path}")
            return entry.ast_data, entry.metadata
            
        except Exception as e:
            logging.warning(f"Cache get failed for {file_path}: {e}")
            self.misses += 1
            return None
    
    def put(self, 
            file_path: Path,
            ast_data: Any,
            metadata: Optional[Dict[str, Any]] = None,
            content: Optional[str] = None):
        """Store AST in cache.
        
        Args:
            file_path: Path to the file
            ast_data: Parsed AST to cache
            metadata: Optional metadata about the AST
            content: Optional file content (to avoid re-reading)
        """
        try:
            cache_key = self._get_cache_key(file_path)
            
            # Prepare cache entry
            mtime = file_path.stat().st_mtime
            file_hash = self._compute_file_hash(file_path, content)
            
            entry = CacheEntry(
                file_path=str(file_path),
                file_hash=file_hash,
                mtime=mtime,
                ast_data=ast_data,
                metadata=metadata or {},
                cached_at=time.time()
            )
            
            # Write to disk
            cache_file = self._get_cache_file_path(cache_key)
            with open(cache_file, 'wb') as f:
                pickle.dump(entry, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Update index
            self.index[cache_key] = {
                'file_path': str(file_path),
                'file_hash': file_hash,
                'mtime': mtime,
                'metadata': metadata or {},
                'cached_at': entry.cached_at,
                'size_bytes': cache_file.stat().st_size
            }
            self._save_index()
            
            # Check cache size and cleanup if needed
            self._enforce_size_limit()
            
            logging.debug(f"Cached AST for {file_path}")
            
        except Exception as e:
            logging.error(f"Failed to cache AST for {file_path}: {e}")
    
    def _invalidate(self, cache_key: str):
        """Invalidate a cache entry.
        
        Args:
            cache_key: Cache key to invalidate
        """
        try:
            # Remove from index
            if cache_key in self.index:
                del self.index[cache_key]
                self._save_index()
            
            # Remove file
            cache_file = self._get_cache_file_path(cache_key)
            if cache_file.exists():
                cache_file.unlink()
            
            self.invalidations += 1
            
        except Exception as e:
            logging.warning(f"Failed to invalidate cache entry {cache_key}: {e}")
    
    def invalidate_file(self, file_path: Path):
        """Invalidate cache entry for a specific file.
        
        Args:
            file_path: Path to the file
        """
        cache_key = self._get_cache_key(file_path)
        self._invalidate(cache_key)
        logging.info(f"Invalidated cache for {file_path}")
    
    def _enforce_size_limit(self):
        """Enforce maximum cache size by removing oldest entries."""
        try:
            total_size = sum(entry['size_bytes'] for entry in self.index.values())
            
            if total_size <= self.max_size_bytes:
                return
            
            logging.info(f"Cache size ({total_size / (1024*1024):.2f}MB) exceeds limit, cleaning up...")
            
            # Sort by cached_at (oldest first)
            sorted_entries = sorted(
                self.index.items(),
                key=lambda x: x[1]['cached_at']
            )
            
            # Remove oldest entries until under limit
            removed_count = 0
            for cache_key, entry in sorted_entries:
                if total_size <= self.max_size_bytes * 0.8:  # Target 80% of limit
                    break
                
                size = entry['size_bytes']
                self._invalidate(cache_key)
                total_size -= size
                removed_count += 1
            
            logging.info(f"Removed {removed_count} oldest cache entries")
            
        except Exception as e:
            logging.error(f"Failed to enforce size limit: {e}")
    
    def clear(self):
        """Clear entire cache."""
        try:
            # Remove all cache files
            for cache_file in self.cache_dir.glob('*.pkl'):
                cache_file.unlink()
            
            # Clear index
            self.index = {}
            self._save_index()
            
            # Reset statistics
            self.hits = 0
            self.misses = 0
            self.invalidations = 0
            
            logging.info("Cache cleared successfully")
            
        except Exception as e:
            logging.error(f"Failed to clear cache: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        total_size = sum(entry['size_bytes'] for entry in self.index.values())
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'invalidations': self.invalidations,
            'hit_rate_percent': round(hit_rate, 2),
            'total_entries': len(self.index),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'max_size_mb': self.max_size_bytes / (1024 * 1024),
            'cache_dir': str(self.cache_dir)
        }
    
    def print_statistics(self):
        """Print cache statistics to log."""
        stats = self.get_statistics()
        logging.info(
            f"AST Cache Stats: {stats['hits']} hits, {stats['misses']} misses "
            f"({stats['hit_rate_percent']}% hit rate), "
            f"{stats['total_entries']} entries, "
            f"{stats['total_size_mb']}MB / {stats['max_size_mb']}MB"
        )
