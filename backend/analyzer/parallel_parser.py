"""
Parallel parser for processing multiple files concurrently using multiprocessing.

This module provides significant performance improvements for large codebases
by distributing parsing work across multiple CPU cores.
"""

import logging
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass
import traceback

from .models import Node, Edge
from .exceptions import ParsingError, MemoryLimitExceeded


@dataclass
class ParseTask:
    """Represents a single file parsing task."""
    file_path: Path
    relative_path: str
    is_entry: bool
    content: str
    
    
@dataclass
class ParseResult:
    """Result of parsing a single file."""
    relative_path: str
    nodes: List[Node]
    symbols: Dict[str, Any]
    success: bool
    error: Optional[str] = None


def _parse_file_worker(task: ParseTask, project_path: str, plugin_class_name: str) -> ParseResult:
    """Worker function that runs in separate process.
    
    This function is the target for multiprocessing workers. It must be
    at module level (not nested) for pickle serialization.
    
    Args:
        task: ParseTask containing file info and content
        project_path: Root path of the project
        plugin_class_name: Name of the language plugin to use
        
    Returns:
        ParseResult with nodes, symbols, and success status
    """
    try:
        # Import here to avoid issues with multiprocessing and imports
        from .plugins.languages.javascript_plugin import JavaScriptPlugin
        from .plugins.languages.python_plugin import PythonPlugin
        from .plugins.languages.java_plugin import JavaPlugin
        
        # Map plugin names to classes
        plugin_map = {
            'JavaScriptPlugin': JavaScriptPlugin,
            'PythonPlugin': PythonPlugin,
            'JavaPlugin': JavaPlugin
        }
        
        plugin_class = plugin_map.get(plugin_class_name)
        if not plugin_class:
            return ParseResult(
                relative_path=task.relative_path,
                nodes=[],
                symbols={},
                success=False,
                error=f"Unknown plugin class: {plugin_class_name}"
            )
        
        # Instantiate plugin
        plugin = plugin_class(project_path)
        
        # Parse the file
        nodes, symbols = plugin.parse(task.file_path, task.content, task.is_entry)
        
        return ParseResult(
            relative_path=task.relative_path,
            nodes=nodes,
            symbols=symbols,
            success=True
        )
        
    except Exception as e:
        error_msg = f"Failed to parse {task.relative_path}: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_msg)
        return ParseResult(
            relative_path=task.relative_path,
            nodes=[],
            symbols={},
            success=False,
            error=error_msg
        )


class ParallelParser:
    """Orchestrates parallel parsing of multiple files using multiprocessing."""
    
    def __init__(self, 
                 project_path: Path,
                 worker_count: Optional[int] = None,
                 chunk_size: int = 10,
                 progress_callback: Optional[Callable[[int, int], None]] = None):
        """Initialize parallel parser.
        
        Args:
            project_path: Root path of the project
            worker_count: Number of worker processes (default: cpu_count - 1)
            chunk_size: Number of files per chunk for workers (default: 10)
            progress_callback: Optional callback(completed, total) for progress updates
        """
        self.project_path = Path(project_path)
        
        # Determine worker count (leave 1 core free for OS)
        if worker_count is None:
            cpu_count = mp.cpu_count()
            self.worker_count = max(1, cpu_count - 1)
        else:
            self.worker_count = max(1, worker_count)
            
        self.chunk_size = chunk_size
        self.progress_callback = progress_callback
        
        logging.info(f"Initialized ParallelParser with {self.worker_count} workers, chunk_size={self.chunk_size}")
    
    def parse_files(self, 
                   tasks: List[ParseTask],
                   plugin_selector: Callable[[Path], Optional[str]]) -> Dict[str, ParseResult]:
        """Parse multiple files in parallel.
        
        Args:
            tasks: List of ParseTask objects to process
            plugin_selector: Function that returns plugin class name for a file path
            
        Returns:
            Dict mapping relative_path to ParseResult
        """
        if not tasks:
            logging.warning("No tasks provided to ParallelParser")
            return {}
        
        total_tasks = len(tasks)
        logging.info(f"Starting parallel parsing of {total_tasks} files across {self.worker_count} workers")
        
        # Group tasks by plugin type for better efficiency
        tasks_by_plugin: Dict[str, List[Tuple[ParseTask, str]]] = {}
        
        for task in tasks:
            plugin_name = plugin_selector(task.file_path)
            if plugin_name:
                if plugin_name not in tasks_by_plugin:
                    tasks_by_plugin[plugin_name] = []
                tasks_by_plugin[plugin_name].append((task, plugin_name))
        
        logging.info(f"Grouped tasks: {[(plugin, len(tasks)) for plugin, tasks in tasks_by_plugin.items()]}")
        
        # Process all tasks
        results: Dict[str, ParseResult] = {}
        completed = 0
        
        try:
            # Create process pool
            with mp.Pool(processes=self.worker_count) as pool:
                # Process each plugin group
                for plugin_name, plugin_tasks in tasks_by_plugin.items():
                    # Prepare arguments for starmap
                    worker_args = [
                        (task, str(self.project_path), plugin_name)
                        for task, _ in plugin_tasks
                    ]
                    
                    # Process in chunks with progress tracking
                    for i in range(0, len(worker_args), self.chunk_size):
                        chunk = worker_args[i:i + self.chunk_size]
                        
                        # Process chunk
                        chunk_results = pool.starmap(_parse_file_worker, chunk)
                        
                        # Aggregate results
                        for result in chunk_results:
                            results[result.relative_path] = result
                            completed += 1
                            
                            # Report progress
                            if self.progress_callback and completed % 10 == 0:
                                self.progress_callback(completed, total_tasks)
                        
                        logging.debug(f"Processed chunk {i//self.chunk_size + 1}: {len(chunk_results)} files")
                
                # Final progress update
                if self.progress_callback:
                    self.progress_callback(completed, total_tasks)
        
        except Exception as e:
            logging.error(f"Parallel parsing failed: {e}")
            raise ParsingError(f"Parallel parsing failed: {str(e)}", details={"total_tasks": total_tasks})
        
        # Log statistics
        successful = sum(1 for r in results.values() if r.success)
        failed = sum(1 for r in results.values() if not r.success)
        
        logging.info(f"Parallel parsing complete: {successful} successful, {failed} failed out of {total_tasks} files")
        
        if failed > 0:
            # Log first few failures
            failed_results = [r for r in results.values() if not r.success]
            for result in failed_results[:5]:
                logging.error(f"Failed: {result.relative_path} - {result.error}")
        
        return results
    
    def parse_files_sequential_fallback(self,
                                       tasks: List[ParseTask],
                                       plugin_selector: Callable[[Path], Optional[str]]) -> Dict[str, ParseResult]:
        """Fallback to sequential parsing if parallel fails.
        
        This is used as a safety mechanism when parallel parsing encounters issues.
        
        Args:
            tasks: List of ParseTask objects to process
            plugin_selector: Function that returns plugin class name for a file path
            
        Returns:
            Dict mapping relative_path to ParseResult
        """
        logging.warning("Using sequential fallback for parsing")
        
        results: Dict[str, ParseResult] = {}
        
        for idx, task in enumerate(tasks):
            plugin_name = plugin_selector(task.file_path)
            if not plugin_name:
                continue
            
            result = _parse_file_worker(task, str(self.project_path), plugin_name)
            results[result.relative_path] = result
            
            # Report progress
            if self.progress_callback and (idx + 1) % 10 == 0:
                self.progress_callback(idx + 1, len(tasks))
        
        return results
    
    @staticmethod
    def get_optimal_worker_count() -> int:
        """Calculate optimal number of workers based on CPU count.
        
        Returns:
            Recommended worker count (cpu_count - 1, minimum 1)
        """
        cpu_count = mp.cpu_count()
        return max(1, cpu_count - 1)
    
    @staticmethod
    def estimate_chunk_size(total_files: int, worker_count: int) -> int:
        """Estimate optimal chunk size based on file count and workers.
        
        Args:
            total_files: Total number of files to parse
            worker_count: Number of worker processes
            
        Returns:
            Recommended chunk size
        """
        # Aim for ~20 chunks per worker to balance overhead vs granularity
        target_chunks = worker_count * 20
        chunk_size = max(1, total_files // target_chunks)
        
        # Cap at reasonable limits
        chunk_size = min(50, max(5, chunk_size))
        
        return chunk_size


def create_parse_task(file_path: Path, 
                     project_path: Path,
                     is_entry: bool = False) -> Optional[ParseTask]:
    """Helper to create a ParseTask from a file path.
    
    Args:
        file_path: Path to the file
        project_path: Root path of the project
        is_entry: Whether this is an entry point file
        
    Returns:
        ParseTask if file is readable, None otherwise
    """
    try:
        content = file_path.read_text(encoding='utf-8')
        relative_path = str(file_path.relative_to(project_path))
        
        return ParseTask(
            file_path=file_path,
            relative_path=relative_path,
            is_entry=is_entry,
            content=content
        )
    except Exception as e:
        logging.warning(f"Could not create parse task for {file_path}: {e}")
        return None
