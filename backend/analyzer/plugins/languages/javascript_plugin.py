"""
JavaScript/TypeScript parsing plugin (Fixed version v2).
"""

import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

from ..base import LanguagePlugin
from ...models import Node, NodeType


class JavaScriptPlugin(LanguagePlugin):
    """Plugin for parsing JavaScript/TypeScript files."""
    
    @property
    def language_name(self) -> str:
        return "javascript"
    
    def can_parse(self, file_extension: str) -> bool:
        return file_extension.lower() in ['.js', '.jsx', '.ts', '.tsx']
    
    def parse(self, file_path: Path, content: str, is_entry: bool = False) -> tuple[List[Node], Dict[str, Any]]:
        """Parse JavaScript/TypeScript file and return nodes and symbols."""
        nodes = []
        symbols = {
            'declared': [],
            'imports': [],
            'jsx_components': []
        }
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
            lang = self._get_ts_language(file_path.suffix.lower())
            if not lang:
                logging.warning(f"Could not load Tree-sitter language for {file_path.suffix}, falling back to regex parsing")
                # Fallback to regex-based parsing
                function_nodes, file_symbols = self._parse_with_regex(content, relative_path, file_path.suffix, is_entry)
                nodes.extend(function_nodes)
                symbols.update(file_symbols)
                return nodes, symbols
            
            # Parse with tree-sitter
            from tree_sitter import Parser
            parser = Parser()
            parser.set_language(lang)
            tree = parser.parse(bytes(content, 'utf-8'))
            
            # Extract symbols and nodes
            function_nodes, file_symbols = self._parse_with_tree_sitter(tree, content, relative_path, file_path.suffix, is_entry)
            nodes.extend(function_nodes)
            symbols.update(file_symbols)
            
        except Exception as e:
            logging.error(f"Failed to parse JavaScript file {relative_path}: {e}")
            # Fallback to regex parsing
            try:
                function_nodes, file_symbols = self._parse_with_regex(content, relative_path, file_path.suffix, is_entry)
                nodes.extend(function_nodes)
                symbols.update(file_symbols)
            except Exception as fallback_error:
                logging.error(f"Fallback parsing also failed for {relative_path}: {fallback_error}")
        
        return nodes, symbols
    
    def _get_ts_language(self, ext: str):
        """Get TypeScript/JavaScript language for tree-sitter with multiple fallbacks."""
        try:
            if ext in ['.js', '.jsx']:
                # Try JavaScript language
                try:
                    import tree_sitter_javascript
                    # Try different attribute names
                    if hasattr(tree_sitter_javascript, 'language'):
                        return tree_sitter_javascript.language()
                    elif hasattr(tree_sitter_javascript, 'LANGUAGE'):
                        return tree_sitter_javascript.LANGUAGE
                    else:
                        # Check what attributes are available
                        attrs = [attr for attr in dir(tree_sitter_javascript) if not attr.startswith('_')]
                        logging.debug(f"tree_sitter_javascript attributes: {attrs}")
                        if 'language_javascript' in attrs:
                            return getattr(tree_sitter_javascript, 'language_javascript')()
                except ImportError:
                    logging.debug("tree_sitter_javascript not available")
            
            elif ext in ['.ts', '.tsx']:
                # Try TypeScript language
                try:
                    import tree_sitter_typescript
                    # Try different attribute names
                    if hasattr(tree_sitter_typescript, 'language'):
                        return tree_sitter_typescript.language()
                    elif hasattr(tree_sitter_typescript, 'language_typescript'):
                        return tree_sitter_typescript.language_typescript()
                    elif hasattr(tree_sitter_typescript, 'language_tsx'):
                        return tree_sitter_typescript.language_tsx()
                    elif hasattr(tree_sitter_typescript, 'LANGUAGE'):
                        return tree_sitter_typescript.LANGUAGE
                    else:
                        # Check what attributes are available
                        attrs = [attr for attr in dir(tree_sitter_typescript) if not attr.startswith('_')]
                        logging.debug(f"tree_sitter_typescript attributes: {attrs}")
                        # Try common attribute patterns
                        for attr_name in ['language_typescript', 'language_tsx', 'typescript', 'tsx']:
                            if attr_name in attrs:
                                attr = getattr(tree_sitter_typescript, attr_name)
                                if callable(attr):
                                    return attr()
                                else:
                                    return attr
                except ImportError:
                    logging.debug("tree_sitter_typescript not available")
            
            # Fallback to tree-sitter-languages package
            try:
                from tree_sitter_languages import get_language
                if ext in ['.js', '.jsx']:
                    return get_language('javascript')
                elif ext in ['.ts', '.tsx']:
                    return get_language('typescript')
            except ImportError:
                logging.debug("tree_sitter_languages not available")
                
            logging.warning(f"No tree-sitter language available for {ext}")
            return None
            
        except Exception as e:
            logging.warning(f"Failed to load tree-sitter language for {ext}: {e}")
            return None
    
    def _parse_with_regex(self, content: str, relative_path: str, ext: str, is_entry: bool) -> tuple[List[Node], Dict[str, Any]]:
        """Fallback parsing using regex when tree-sitter is not available."""
        import re
        
        nodes = []
        symbols = {
            'declared': [],
            'imports': [],
            'jsx_components': []
        }
        
        # Extract imports using regex
        import_patterns = [
            r"import\s+.*?\s+from\s+['\"]([^'\"]+)['\"]",
            r"import\s+['\"]([^'\"]+)['\"]",
            r"import\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            symbols['imports'].extend(matches)
        
        # Extract function declarations
        func_patterns = [
            r"function\s+(\w+)\s*\(",
            r"const\s+(\w+)\s*=\s*\([^)]*\)\s*=>\s*{",
            r"let\s+(\w+)\s*=\s*\([^)]*\)\s*=>\s*{",
            r"var\s+(\w+)\s*=\s*\([^)]*\)\s*=>\s*{",
            r"const\s+(\w+)\s*=\s*function\s*\(",
            r"export\s+function\s+(\w+)\s*\("
        ]
        
        declared_names = set()
        for pattern in func_patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            for match in matches:
                if match and match not in declared_names:
                    declared_names.add(match)
                    is_component = match[0].isupper() and ext in ['.tsx', '.jsx']
                    node_type = NodeType.COMPONENT if is_component else NodeType.FUNCTION
                    c4_level = self._determine_c4_level(node_type, relative_path, match, is_entry)
                    node_id = self._generate_node_id(relative_path, match)
                    graph_node = Node(
                        id=node_id, type=node_type, file=relative_path, name=match,
                        metadata={"is_entry": is_entry, "c4_level": c4_level}
                    )
                    nodes.append(graph_node)
        
        # Extract class declarations
        class_pattern = r"class\s+(\w+)\s*(?:extends\s+\w+)?\s*{"
        class_matches = re.findall(class_pattern, content, re.MULTILINE)
        for class_name in class_matches:
            if class_name and class_name not in declared_names:
                declared_names.add(class_name)
                c4_level = self._determine_c4_level(NodeType.CLASS, relative_path, class_name, is_entry)
                node_id = self._generate_node_id(relative_path, class_name)
                graph_node = Node(
                    id=node_id, type=NodeType.CLASS, file=relative_path, name=class_name,
                    metadata={"is_entry": is_entry, "c4_level": c4_level}
                )
                nodes.append(graph_node)
        
        # Extract JSX components for .tsx/.jsx files
        if ext in ['.tsx', '.jsx']:
            jsx_pattern = r"<(\w+)(?:\s+[^>]*)?"
            jsx_matches = re.findall(jsx_pattern, content)
            jsx_components = []
            for jsx_name in jsx_matches:
                if jsx_name and jsx_name[0].isupper() and jsx_name not in ['div', 'span', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'button']:
                    jsx_components.append(jsx_name)
            symbols['jsx_components'] = list(set(jsx_components))
        
        symbols['declared'] = list(declared_names)
        
        logging.debug(f"Regex parsing found: {len(nodes)} nodes, {len(symbols['imports'])} imports, {len(symbols['jsx_components'])} JSX components")
        
        return nodes, symbols
    
    def _parse_with_tree_sitter(self, tree, content: str, relative_path: str, ext: str, is_entry: bool) -> tuple[List[Node], Dict[str, Any]]:
        """Parse using tree-sitter queries."""
        nodes = []
        symbols = {
            'declared': [],
            'imports': [],
            'jsx_components': []
        }
        
        try:
            # Extract functions and classes
            declared_names = self._extract_functions_and_classes(tree, content, relative_path, ext, is_entry, nodes)
            
            # Extract imports
            imports = self._extract_imports(tree, content)
            
            # Extract JSX components if it's a JSX file
            jsx_components = []
            if ext in ['.tsx', '.jsx']:
                jsx_components = self._extract_jsx_components(tree, content)
            
            symbols = {
                'declared': list(declared_names),
                'imports': imports,
                'jsx_components': jsx_components
            }
            
        except Exception as e:
            logging.error(f"Failed to parse tree-sitter content for {relative_path}: {e}")
        
        return nodes, symbols
    
    def _extract_functions_and_classes(self, tree, content: str, relative_path: str, ext: str, is_entry: bool, nodes: List[Node]) -> Set[str]:
        """Extract functions and classes from the AST."""
        declared_names = set()
        
        def walk_node(node):
            # Function declarations
            if node.type == 'function_declaration':
                name_node = None
                for child in node.children:
                    if child.type == 'identifier':
                        name_node = child
                        break
                
                if name_node:
                    func_name = self._capture_text(content, name_node)
                    if func_name:
                        declared_names.add(func_name)
                        is_component = func_name[0].isupper() and ext in ['.tsx', '.jsx']
                        node_type = NodeType.COMPONENT if is_component else NodeType.FUNCTION
                        c4_level = self._determine_c4_level(node_type, relative_path, func_name, is_entry)
                        node_id = self._generate_node_id(relative_path, func_name)
                        graph_node = Node(
                            id=node_id, type=node_type, file=relative_path, name=func_name,
                            metadata={"is_entry": is_entry, "c4_level": c4_level}
                        )
                        nodes.append(graph_node)
            
            # Class declarations
            elif node.type == 'class_declaration':
                name_node = None
                for child in node.children:
                    if child.type == 'identifier':
                        name_node = child
                        break
                
                if name_node:
                    class_name = self._capture_text(content, name_node)
                    if class_name:
                        declared_names.add(class_name)
                        c4_level = self._determine_c4_level(NodeType.CLASS, relative_path, class_name, is_entry)
                        node_id = self._generate_node_id(relative_path, class_name)
                        graph_node = Node(
                            id=node_id, type=NodeType.CLASS, file=relative_path, name=class_name,
                            metadata={"is_entry": is_entry, "c4_level": c4_level}
                        )
                        nodes.append(graph_node)
            
            # Variable declarations (arrow functions, function expressions)
            elif node.type == 'variable_declaration':
                for child in node.children:
                    if child.type == 'variable_declarator':
                        var_name = None
                        is_function = False
                        
                        for grandchild in child.children:
                            if grandchild.type == 'identifier':
                                var_name = self._capture_text(content, grandchild)
                            elif grandchild.type in ['arrow_function', 'function_expression']:
                                is_function = True
                        
                        if var_name and is_function:
                            declared_names.add(var_name)
                            is_component = var_name[0].isupper() and ext in ['.tsx', '.jsx']
                            node_type = NodeType.COMPONENT if is_component else NodeType.FUNCTION
                            c4_level = self._determine_c4_level(node_type, relative_path, var_name, is_entry)
                            node_id = self._generate_node_id(relative_path, var_name)
                            graph_node = Node(
                                id=node_id, type=node_type, file=relative_path, name=var_name,
                                metadata={"is_entry": is_entry, "c4_level": c4_level}
                            )
                            nodes.append(graph_node)
            
            # Recursively walk children
            for child in node.children:
                walk_node(child)
        
        walk_node(tree.root_node)
        return declared_names
    
    def _extract_imports(self, tree, content: str) -> List[str]:
        """Extract import statements."""
        imports = []
        
        def walk_node(node):
            if node.type == 'import_statement':
                # Find the source string
                for child in node.children:
                    if child.type == 'string':
                        import_path = self._capture_text(content, child).strip().strip('"\'')
                        if import_path:
                            imports.append(import_path)
                        break
            
            # Recursively walk children
            for child in node.children:
                walk_node(child)
        
        walk_node(tree.root_node)
        return imports
    
    def _extract_jsx_components(self, tree, content: str) -> List[str]:
        """Extract JSX component usage."""
        jsx_components = []
        
        def walk_node(node):
            # JSX opening elements
            if node.type in ['jsx_opening_element', 'jsx_self_closing_element']:
                for child in node.children:
                    if child.type == 'identifier':
                        jsx_name = self._capture_text(content, child)
                        if jsx_name and jsx_name[0].isupper():  # Component names start with uppercase
                            jsx_components.append(jsx_name)
                        break
            
            # Recursively walk children
            for child in node.children:
                walk_node(child)
        
        walk_node(tree.root_node)
        return list(set(jsx_components))  # Remove duplicates
    
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
                file_lower.endswith(('main.tsx', 'main.ts', 'main.js', 'app.tsx'))):
                return "container"
        
        if node_type in [NodeType.COMPONENT, NodeType.VIEW, NodeType.CONTROLLER, NodeType.SERVICE]:
            return "component"
        
        if node_type == NodeType.MODULE:
            if any(segment in file_path.lower() for segment in ['pages', 'components', 'views', 'controllers', 'services']):
                return "component"
        
        if node_type in [NodeType.FUNCTION, NodeType.CLASS, NodeType.MODEL, NodeType.TEMPLATE]:
            return "code"
        
        return "code"