"""
Incremental analyzer for tracking and re-analyzing only changed files.

This module provides intelligent change detection using git diff or file mtimes,
allowing FlowLens to skip unchanged files and only re-analyze what's necessary.
"""

import os
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import time
import hashlib


@dataclass
class FileMetadata:
    """Metadata for tracking file changes."""
    path: str
    mtime: float
    size: int
    hash: str
    last_analyzed: float


@dataclass
class AnalysisSnapshot:
    """Snapshot of previous analysis state."""
    timestamp: float
    project_path: str
    files: Dict[str, FileMetadata]
    total_files: int
    total_nodes: int
    total_edges: int
    project_type: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp,
            'project_path': self.project_path,
            'files': {k: asdict(v) for k, v in self.files.items()},
            'total_files': self.total_files,
            'total_nodes': self.total_nodes,
            'total_edges': self.total_edges,
            'project_type': self.project_type
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisSnapshot':
        """Create from dictionary."""
        files = {
            k: FileMetadata(**v) for k, v in data.get('files', {}).items()
        }
        return cls(
            timestamp=data['timestamp'],
            project_path=data['project_path'],
            files=files,
            total_files=data['total_files'],
            total_nodes=data['total_nodes'],
            total_edges=data['total_edges'],
            project_type=data.get('project_type', 'unknown')
        )


class IncrementalAnalyzer:
    """Tracks file changes and determines what needs re-analysis."""
    
    def __init__(self, project_path: Path, cache_dir: Optional[Path] = None):
        """Initialize incremental analyzer.
        
        Args:
            project_path: Root path of the project
            cache_dir: Directory for cache storage (default: .flowlens-cache/)
        """
        self.project_path = Path(project_path)
        
        if cache_dir is None:
            cache_dir = self.project_path / '.flowlens-cache'
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.snapshot_file = self.cache_dir / 'last-analysis.json'
        self.previous_snapshot: Optional[AnalysisSnapshot] = self._load_snapshot()
        
        # Git detection
        self.is_git_repo = self._detect_git_repo()
        
        logging.info(f"Initialized IncrementalAnalyzer (git_repo={self.is_git_repo})")
    
    def _detect_git_repo(self) -> bool:
        """Check if project is a git repository."""
        git_dir = self.project_path / '.git'
        return git_dir.exists() and git_dir.is_dir()
    
    def _load_snapshot(self) -> Optional[AnalysisSnapshot]:
        """Load previous analysis snapshot from disk."""
        if not self.snapshot_file.exists():
            logging.info("No previous analysis snapshot found")
            return None
        
        try:
            with open(self.snapshot_file, 'r') as f:
                data = json.load(f)
            
            snapshot = AnalysisSnapshot.from_dict(data)
            logging.info(
                f"Loaded previous snapshot: {snapshot.total_files} files, "
                f"{snapshot.total_nodes} nodes, {snapshot.total_edges} edges "
                f"(analyzed {time.time() - snapshot.timestamp:.1f}s ago)"
            )
            return snapshot
            
        except Exception as e:
            logging.warning(f"Failed to load previous snapshot: {e}")
            return None
    
    def save_snapshot(self, 
                     files: List[Path],
                     total_nodes: int,
                     total_edges: int,
                     project_type: str):
        """Save current analysis state as snapshot.
        
        Args:
            files: List of analyzed file paths
            total_nodes: Total nodes in analysis result
            total_edges: Total edges in analysis result
            project_type: Detected project type
        """
        try:
            # Build file metadata
            file_metadata: Dict[str, FileMetadata] = {}
            
            for file_path in files:
                if not file_path.exists():
                    continue
                
                try:
                    relative_path = str(file_path.relative_to(self.project_path))
                    stat = file_path.stat()
                    content = file_path.read_text(encoding='utf-8')
                    file_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
                    
                    file_metadata[relative_path] = FileMetadata(
                        path=relative_path,
                        mtime=stat.st_mtime,
                        size=stat.st_size,
                        hash=file_hash,
                        last_analyzed=time.time()
                    )
                except Exception as e:
                    logging.warning(f"Failed to create metadata for {file_path}: {e}")
            
            # Create snapshot
            snapshot = AnalysisSnapshot(
                timestamp=time.time(),
                project_path=str(self.project_path),
                files=file_metadata,
                total_files=len(file_metadata),
                total_nodes=total_nodes,
                total_edges=total_edges,
                project_type=project_type
            )
            
            # Save to disk
            with open(self.snapshot_file, 'w') as f:
                json.dump(snapshot.to_dict(), f, indent=2)
            
            logging.info(
                f"Saved analysis snapshot: {snapshot.total_files} files, "
                f"{snapshot.total_nodes} nodes, {snapshot.total_edges} edges"
            )
            
            self.previous_snapshot = snapshot
            
        except Exception as e:
            logging.error(f"Failed to save analysis snapshot: {e}")
    
    def detect_changes(self, current_files: List[Path]) -> Tuple[Set[Path], Set[Path], Set[Path]]:
        """Detect which files have changed, been added, or deleted.
        
        Args:
            current_files: List of current files to analyze
            
        Returns:
            Tuple of (added_files, modified_files, deleted_files)
        """
        # If no previous snapshot, all files are "added"
        if not self.previous_snapshot:
            logging.info("No previous snapshot - treating all files as new")
            return set(current_files), set(), set()
        
        # Try git-based detection first
        if self.is_git_repo:
            try:
                git_changes = self._detect_changes_git()
                if git_changes is not None:
                    added, modified, deleted = git_changes
                    logging.info(
                        f"Git change detection: {len(added)} added, "
                        f"{len(modified)} modified, {len(deleted)} deleted"
                    )
                    return added, modified, deleted
            except Exception as e:
                logging.warning(f"Git detection failed, falling back to mtime: {e}")
        
        # Fallback to mtime-based detection
        return self._detect_changes_mtime(current_files)
    
    def _detect_changes_git(self) -> Optional[Tuple[Set[Path], Set[Path], Set[Path]]]:
        """Detect changes using git diff.
        
        Returns:
            Tuple of (added_files, modified_files, deleted_files) or None if git fails
        """
        try:
            # Get list of changed files since last commit
            result = subprocess.run(
                ['git', 'diff', '--name-status', 'HEAD'],
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                logging.warning(f"Git diff failed: {result.stderr}")
                return None
            
            added = set()
            modified = set()
            deleted = set()
            
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                
                parts = line.split('\t', 1)
                if len(parts) != 2:
                    continue
                
                status, file_path = parts
                full_path = self.project_path / file_path
                
                if status.startswith('A'):
                    added.add(full_path)
                elif status.startswith('M'):
                    modified.add(full_path)
                elif status.startswith('D'):
                    deleted.add(full_path)
                elif status.startswith('R'):  # Renamed
                    # Handle rename: R100 old_name -> new_name
                    if '->' in file_path:
                        old_name, new_name = file_path.split('->')
                        deleted.add(self.project_path / old_name.strip())
                        added.add(self.project_path / new_name.strip())
                    else:
                        modified.add(full_path)
            
            # Also check for untracked files
            result = subprocess.run(
                ['git', 'ls-files', '--others', '--exclude-standard'],
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        added.add(self.project_path / line)
            
            return added, modified, deleted
            
        except subprocess.TimeoutExpired:
            logging.warning("Git command timed out")
            return None
        except Exception as e:
            logging.warning(f"Git detection error: {e}")
            return None
    
    def _detect_changes_mtime(self, current_files: List[Path]) -> Tuple[Set[Path], Set[Path], Set[Path]]:
        """Detect changes using file modification times and hashes.
        
        Args:
            current_files: List of current files to check
            
        Returns:
            Tuple of (added_files, modified_files, deleted_files)
        """
        if not self.previous_snapshot:
            return set(current_files), set(), set()
        
        added = set()
        modified = set()
        
        # Check current files
        for file_path in current_files:
            try:
                relative_path = str(file_path.relative_to(self.project_path))
                
                # Check if file existed in previous snapshot
                if relative_path not in self.previous_snapshot.files:
                    added.add(file_path)
                    continue
                
                prev_metadata = self.previous_snapshot.files[relative_path]
                
                # Quick check: mtime
                current_mtime = file_path.stat().st_mtime
                if abs(current_mtime - prev_metadata.mtime) > 0.001:  # Allow small float differences
                    # Verify with hash
                    content = file_path.read_text(encoding='utf-8')
                    current_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
                    
                    if current_hash != prev_metadata.hash:
                        modified.add(file_path)
                
            except Exception as e:
                logging.warning(f"Failed to check file {file_path}: {e}")
                # Treat as modified to be safe
                modified.add(file_path)
        
        # Check for deleted files
        deleted = set()
        current_relative_paths = {
            str(f.relative_to(self.project_path)) for f in current_files
        }
        
        for prev_relative_path in self.previous_snapshot.files.keys():
            if prev_relative_path not in current_relative_paths:
                deleted.add(self.project_path / prev_relative_path)
        
        logging.info(
            f"Mtime change detection: {len(added)} added, "
            f"{len(modified)} modified, {len(deleted)} deleted"
        )
        
        return added, modified, deleted
    
    def get_files_to_analyze(self, 
                            all_files: List[Path],
                            include_dependents: bool = True) -> Tuple[List[Path], Dict[str, Any]]:
        """Determine which files need to be analyzed.
        
        Args:
            all_files: Complete list of files in project
            include_dependents: Whether to include files that depend on changed files
            
        Returns:
            Tuple of (files_to_analyze, change_summary)
        """
        # Detect changes
        added, modified, deleted = self.detect_changes(all_files)
        
        # Files that need analysis
        files_to_analyze = list(added | modified)
        
        # TODO: Implement dependency tracking to include dependent files
        # For now, we skip this and just re-analyze changed files
        # In a full implementation, you would:
        # 1. Build a dependency graph from previous snapshot
        # 2. For each modified file, find all files that import it
        # 3. Add those dependent files to files_to_analyze
        
        change_summary = {
            'added': len(added),
            'modified': len(modified),
            'deleted': len(deleted),
            'to_analyze': len(files_to_analyze),
            'unchanged': len(all_files) - len(added) - len(modified),
            'total': len(all_files)
        }
        
        if files_to_analyze:
            logging.info(
                f"Incremental analysis: {change_summary['to_analyze']} files to analyze "
                f"({change_summary['added']} added, {change_summary['modified']} modified) "
                f"out of {change_summary['total']} total"
            )
        else:
            logging.info("No changes detected - can skip re-analysis")
        
        return files_to_analyze, change_summary
    
    def should_use_incremental(self, 
                              all_files: List[Path],
                              threshold_percent: float = 30.0) -> bool:
        """Determine if incremental analysis is worth using.
        
        Args:
            all_files: Complete list of files in project
            threshold_percent: Only use incremental if changes are below this % (default: 30%)
            
        Returns:
            True if incremental analysis should be used
        """
        if not self.previous_snapshot:
            return False
        
        added, modified, deleted = self.detect_changes(all_files)
        total_changes = len(added) + len(modified)
        total_files = len(all_files)
        
        if total_files == 0:
            return False
        
        change_percent = (total_changes / total_files) * 100
        
        use_incremental = change_percent < threshold_percent
        
        logging.info(
            f"Incremental analysis decision: {change_percent:.1f}% changed "
            f"(threshold: {threshold_percent}%) - {'USE' if use_incremental else 'SKIP'} incremental"
        )
        
        return use_incremental
    
    def clear_snapshot(self):
        """Clear saved snapshot to force full re-analysis."""
        try:
            if self.snapshot_file.exists():
                self.snapshot_file.unlink()
                logging.info("Cleared analysis snapshot")
            self.previous_snapshot = None
        except Exception as e:
            logging.error(f"Failed to clear snapshot: {e}")
