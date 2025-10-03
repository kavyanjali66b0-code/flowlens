"""Memory monitoring and management for the analyzer."""

import psutil
import logging
import gc
from typing import Optional, Callable
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a point in time."""
    timestamp: datetime
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    percent: float  # Percentage of system memory
    available_mb: float


class MemoryMonitor:
    """Monitors and manages memory usage during analysis."""
    
    def __init__(self, max_memory_mb: int = 512, warning_threshold: float = 0.8):
        """
        Initialize memory monitor.
        
        Args:
            max_memory_mb: Maximum allowed memory in MB
            warning_threshold: Trigger warning at this percentage of max
        """
        self.max_memory_mb = max_memory_mb
        self.warning_threshold = warning_threshold
        self.process = psutil.Process()
        self.snapshots: list[MemorySnapshot] = []
        self.cleanup_callbacks: list[Callable] = []
        
    def get_current_usage(self) -> MemorySnapshot:
        """Get current memory usage snapshot."""
        mem_info = self.process.memory_info()
        sys_mem = psutil.virtual_memory()
        
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            rss_mb=mem_info.rss / 1024 / 1024,
            vms_mb=mem_info.vms / 1024 / 1024,
            percent=self.process.memory_percent(),
            available_mb=sys_mem.available / 1024 / 1024
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def check_memory_threshold(self) -> tuple[bool, Optional[str]]:
        """
        Check if memory usage exceeds thresholds.
        
        Returns:
            Tuple of (exceeded, message)
        """
        snapshot = self.get_current_usage()
        
        if snapshot.rss_mb > self.max_memory_mb:
            msg = f"Memory limit exceeded: {snapshot.rss_mb:.1f}MB > {self.max_memory_mb}MB"
            logging.error(msg)
            return True, msg
            
        warning_limit = self.max_memory_mb * self.warning_threshold
        if snapshot.rss_mb > warning_limit:
            msg = f"Memory warning: {snapshot.rss_mb:.1f}MB (threshold: {warning_limit:.1f}MB)"
            logging.warning(msg)
            self.trigger_cleanup()
            
        return False, None
    
    def trigger_cleanup(self):
        """Trigger all registered cleanup callbacks and force garbage collection."""
        logging.info("Triggering memory cleanup")
        
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logging.error(f"Cleanup callback failed: {e}")
        
        # Force garbage collection
        gc.collect()
        
        # Log memory after cleanup
        after = self.get_current_usage()
        logging.info(f"Memory after cleanup: {after.rss_mb:.1f}MB")
    
    def register_cleanup(self, callback: Callable):
        """Register a cleanup callback to be called when memory is high."""
        self.cleanup_callbacks.append(callback)
    
    def get_peak_usage(self) -> Optional[MemorySnapshot]:
        """Get the snapshot with highest memory usage."""
        if not self.snapshots:
            return None
        return max(self.snapshots, key=lambda s: s.rss_mb)
    
    def get_statistics(self) -> dict:
        """Get memory usage statistics."""
        if not self.snapshots:
            return {}
            
        rss_values = [s.rss_mb for s in self.snapshots]
        return {
            "peak_mb": max(rss_values),
            "average_mb": sum(rss_values) / len(rss_values),
            "current_mb": rss_values[-1],
            "snapshots_count": len(self.snapshots)
        }
