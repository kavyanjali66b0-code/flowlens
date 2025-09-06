"""
Java parsing plugin.
"""

import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..base import LanguagePlugin
from ...models import Node, NodeType


class JavaPlugin(LanguagePlugin):
    """Plugin for parsing Java files."""
    
    @property
    def language_name(self) -> str:
        return "java"
    
    def can_parse(self, file_extension: str) -> bool:
        return file_extension.lower() == '.java'
    
    def parse(self, file_path: Path, content: str, is_entry: bool = False) -> tuple[List[Node], Dict[str, Any]]:
        """Parse Java file and return nodes and symbols."""
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
            
            # Load tree-sitter language
            lang = self._get_java_language()
            if not lang:
                logging.warning(f"Could not load Tree-sitter language for {file_path.suffix}")
                return nodes, symbols
            
            # Parse with tree-sitter
            tree = lang.parse(bytes(content, 'utf-8'))
            
            # Execute queries
            captures = self._execute_queries(lang, tree, content)
            
            # Process captures into nodes
            class_nodes = self._process_captures(captures, content, relative_path, is_entry)
            nodes.extend(class_nodes)
            
            # Create symbols dictionary
            symbols = {
                'declared': [node.name for node in class_nodes if node.name],
                'imports': []  # Java imports would need separate parsing
            }
            
        except Exception as e:
            logging.error(f"Failed to parse Java file {relative_path}: {e}")
        
        return nodes, symbols
    
    def _get_java_language(self):
        """Get Java language for tree-sitter."""
        try:
            import tree_sitter_java
            return tree_sitter_java.language()
        except ImportError:
            logging.warning("tree-sitter-java not available")
            return None
    
    def _execute_queries(self, lang, tree, content: str) -> Dict[str, List]:
        """Execute tree-sitter queries and return captures."""
        captures = {
            'class_name': [],
            'ann_name': [],
            'map_ann': [],
            'map_path': []
        }
        
        try:
            core_queries = self.queries.get('core', [])
            for query_str in core_queries:
                query = lang.query(query_str)
                query_captures = query.captures(tree.root_node)
                for capture_name, node in query_captures:
                    if capture_name in captures:
                        captures[capture_name].append(node)
                        
        except Exception as e:
            logging.warning(f"Query execution failed: {e}")
        
        return captures
    
    def _process_captures(self, captures: Dict[str, List], content: str, 
                         relative_path: str, is_entry: bool) -> List[Node]:
        """Process tree-sitter captures into Node objects."""
        nodes = []
        
        # Process classes
        for node in captures['class_name']:
            class_name = self._capture_text(content, node)
            if not class_name:
                continue
            
            # Determine if it's a controller or service based on annotations
            is_controller = any('Controller' in self._capture_text(content, ann) 
                              for ann in captures['ann_name'])
            is_service = any('Service' in self._capture_text(content, ann) 
                           for ann in captures['ann_name'])
            
            node_type = NodeType.CONTROLLER if is_controller else (NodeType.SERVICE if is_service else NodeType.CLASS)
            c4_level = self._determine_c4_level(node_type, relative_path, class_name, is_entry)
            node_id = self._generate_node_id(relative_path, class_name)
            graph_node = Node(
                id=node_id, type=node_type, file=relative_path, name=class_name,
                metadata={"is_entry": is_entry, "c4_level": c4_level}
            )
            nodes.append(graph_node)
        
        # Process API endpoints
        for ann_node, path_node in zip(captures.get('map_ann', []), captures.get('map_path', [])):
            ann_name = self._capture_text(content, ann_node)
            if ann_name not in {'GetMapping', 'PostMapping', 'PutMapping', 'DeleteMapping', 'RequestMapping'}:
                continue
            raw = self._capture_text(content, path_node)
            endpoint_path = raw.strip().strip('"\'')
            if not endpoint_path:
                continue
            api_node_id = f"api_{hashlib.md5(endpoint_path.encode()).hexdigest()[:8]}"
            c4_level = self._determine_c4_level(NodeType.API_ENDPOINT, relative_path, endpoint_path, False)
            api_node = Node(
                id=api_node_id, type=NodeType.API_ENDPOINT, file=relative_path, name=endpoint_path,
                metadata={"endpoint": endpoint_path, "c4_level": c4_level}
            )
            nodes.append(api_node)
        
        return nodes
    
    def _capture_text(self, content: str, node) -> str:
        """Extract text content from a tree-sitter node."""
        return content[node.start_byte:node.end_byte]
    
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
                'springbootapplication' in name_lower):
                return "container"
        
        if node_type in [NodeType.COMPONENT, NodeType.VIEW, NodeType.CONTROLLER, NodeType.SERVICE]:
            return "component"
        
        if node_type == NodeType.MODULE:
            if any(segment in file_path.lower() for segment in ['pages', 'components', 'views', 'controllers', 'services']):
                return "component"
        
        if node_type in [NodeType.FUNCTION, NodeType.CLASS, NodeType.MODEL, NodeType.TEMPLATE]:
            return "code"
        
        return "code"
