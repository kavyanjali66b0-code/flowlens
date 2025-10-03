"""
Base plugin classes for the analyzer system.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from pathlib import Path

from ..models import Node, Edge, NodeType, EdgeType, ProjectType

if TYPE_CHECKING:
    from ..symbol_table import SymbolTable


class LanguagePlugin(ABC):
    """Base class for language-specific parsing plugins."""
    
    def __init__(self, project_path: str, user_config: Optional[Dict[str, Any]] = None, 
                 symbol_table: Optional['SymbolTable'] = None):
        self.project_path = Path(project_path)
        self.user_config = user_config or {}
        self.symbol_table = symbol_table
        self.queries = self._load_queries()
    
    @abstractmethod
    def can_parse(self, file_extension: str) -> bool:
        """Check if this plugin can parse files with the given extension."""
        pass
    
    @abstractmethod
    def parse(self, file_path: Path, content: str, is_entry: bool = False) -> tuple[List[Node], Dict[str, Any]]:
        """Parse a file and return a list of nodes and symbols."""
        pass
    
    def parse_with_ast(self, file_path: Path, content: str, tree_sitter_ast: Any, is_entry: bool = False) -> tuple[List[Node], Dict[str, Any]]:
        """Parse a file using pre-parsed AST from tree-sitter.
        
        This method allows plugins to leverage existing AST from parser
        instead of re-parsing. Default implementation falls back to parse().
        
        Args:
            file_path: Path to the file being parsed
            content: File content as string
            tree_sitter_ast: Pre-parsed tree-sitter AST object
            is_entry: Whether this file is an entry point
            
        Returns:
            tuple: (list of Node objects, dict of symbols)
        """
        # Default implementation: ignore AST and call parse()
        # Subclasses can override to use AST directly
        return self.parse(file_path, content, is_entry)
    
    def _load_queries(self) -> Dict[str, List[str]]:
        """Load tree-sitter queries from JSON files."""
        import json
        import os
        
        query_file = os.path.join(os.path.dirname(__file__), '..', 'queries', f'{self.language_name}.json')
        try:
            with open(query_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load queries for {self.language_name}: {e}")
            return {}
    
    @property
    @abstractmethod
    def language_name(self) -> str:
        """Return the language name for query file lookup."""
        pass


class ScannerPlugin(ABC):
    """Base class for project type detection plugins."""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
    
    @abstractmethod
    def is_applicable(self, files: List[Path]) -> bool:
        """Check if this plugin applies to the given project files."""
        pass
    
    @abstractmethod
    def get_project_type(self) -> ProjectType:
        """Return the project type this plugin detects."""
        pass
    
    @abstractmethod
    def get_config_files(self) -> List[Dict[str, Any]]:
        """Return configuration files found by this plugin."""
        pass
class BasePlugin:
    """Base class for a framework-specific analyzer plugin."""

    name: str = "Base Plugin"
    project_type: ProjectType = ProjectType.UNKNOWN

    def __init__(self, identifier):
        self.identifier = identifier
        self.project_path = identifier.project_path
        self.config_files = identifier.config_files
        self.entry_points: List[Dict] = []

    def is_applicable(self) -> bool:
        return self.identifier.project_type == self.project_type

    def find_entry_points(self) -> List[Dict]:
        raise NotImplementedError("Each plugin must implement find_entry_points.")
