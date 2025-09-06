"""
Python parsing plugin.
"""

import ast
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..base import LanguagePlugin
from ...models import Node, NodeType


class PythonPlugin(LanguagePlugin):
    """Plugin for parsing Python files."""
    
    @property
    def language_name(self) -> str:
        return "python"
    
    def can_parse(self, file_extension: str) -> bool:
        return file_extension.lower() == '.py'
    
    def parse(self, file_path: Path, content: str, is_entry: bool = False) -> tuple[List[Node], Dict[str, Any]]:
        """Parse Python file and return nodes and symbols."""
        nodes = []
        symbols = {}
        relative_path = str(file_path.relative_to(self.project_path))
        
        try:
            # Create module node for the file
            module_name = file_path.stem
            module_id = self._generate_node_id(relative_path, 'module')
            c4_level = self._determine_c4_level(NodeType.MODULE, relative_path, module_name, is_entry)
            module_node = Node(
                id=module_id,
                type=NodeType.MODULE,
                file=relative_path,
                name=module_name,
                metadata={"is_entry": is_entry, "c4_level": c4_level}
            )
            nodes.append(module_node)
            
            # Parse with Python AST
            tree = ast.parse(content)
            
            # Process AST nodes
            function_nodes = self._process_ast(tree, content, relative_path, is_entry)
            nodes.extend(function_nodes)
            
            # Create symbols dictionary
            symbols = {
                'declared': [node.name for node in function_nodes if node.name],
                'imports': self._extract_imports(tree)
            }
            
        except SyntaxError as e:
            logging.warning(f"Syntax error in Python file {relative_path}: {e}")
        except Exception as e:
            logging.error(f"Failed to parse Python file {relative_path}: {e}")
        
        return nodes, symbols
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements from AST."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        imports.append(f"{node.module}.{alias.name}")
        return imports
    
    def _process_ast(self, tree: ast.AST, content: str, relative_path: str, is_entry: bool) -> List[Node]:
        """Process Python AST and return nodes."""
        nodes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                node_id = self._generate_node_id(relative_path, node.name)
                c4_level = self._determine_c4_level(NodeType.CLASS, relative_path, node.name, is_entry)
                graph_node = Node(
                    id=node_id, type=NodeType.CLASS, file=relative_path, name=node.name,
                    metadata={"is_entry": is_entry, "c4_level": c4_level}
                )
                nodes.append(graph_node)
                
            elif isinstance(node, ast.FunctionDef):
                node_type = NodeType.VIEW if 'view' in node.name.lower() else NodeType.FUNCTION
                node_id = self._generate_node_id(relative_path, node.name)
                c4_level = self._determine_c4_level(node_type, relative_path, node.name, is_entry)
                graph_node = Node(
                    id=node_id, type=node_type, file=relative_path, name=node.name,
                    metadata={"is_entry": is_entry, "c4_level": c4_level}
                )
                nodes.append(graph_node)
        
        return nodes
    
    def _generate_node_id(self, file_path: str, name: str) -> str:
        """Generate a unique node ID using the same format as the main parser."""
        content = f"{file_path}:{name}"
        hash_suffix = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{file_path.replace('/', '_').replace('.', '_')}_{name}_{hash_suffix}"
    
    def _determine_c4_level(self, node_type: NodeType, file_path: str, name: str, is_entry: bool = False) -> str:
        """Determine C4 model level for a node."""
        # User overrides from .devscope.yml
        try:
            overrides = (self.user_config or {}).get('c4_overrides', {})
            # Path-based overrides
            for level, patterns in (overrides.get('path_contains', {}) or {}).items():
                for pat in patterns:
                    if pat and pat.lower() in file_path.lower():
                        return level
            # Type-based overrides
            for level, types in (overrides.get('node_types', {}) or {}).items():
                if node_type.value in types:
                    return level
        except Exception:
            pass
        
        # Default heuristics
        if node_type == NodeType.API_ENDPOINT:
            return "system"
        
        if is_entry or node_type == NodeType.MODULE:
            file_lower = file_path.lower()
            name_lower = name.lower()
            if (name_lower in ['main', 'app', 'index'] or
                'main.' in file_lower or
                file_lower.endswith(('main.py', 'app.py', 'urls.py'))):
                return "container"
        
        if node_type in [NodeType.COMPONENT, NodeType.VIEW, NodeType.CONTROLLER, NodeType.SERVICE]:
            return "component"
        
        if node_type == NodeType.MODULE:
            if any(segment in file_path.lower() for segment in ['pages', 'components', 'views', 'controllers', 'services']):
                return "component"
        
        if node_type in [NodeType.FUNCTION, NodeType.CLASS, NodeType.MODEL, NodeType.TEMPLATE]:
            return "code"
        
        return "code"
