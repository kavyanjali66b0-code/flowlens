"""
JavaScript/TypeScript parsing plugin (Fixed version v2).
"""

import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

from ..base import LanguagePlugin
from ...models import Node, NodeType
from ...symbol_table import Symbol, SymbolType, SymbolReference


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
        # Normalize path to use forward slashes for cross-platform consistency
        relative_path = self._normalize_path(str(file_path.relative_to(self.project_path)))
        
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
        
        # DEBUG: Log what we're returning for ALL files
        jsx_count = len(symbols.get('jsx_components', []))
        logging.info(f"DEBUG PLUGIN: {relative_path} - {len(nodes)} nodes, {jsx_count} JSX components")
        if jsx_count > 0:
            logging.info(f"DEBUG PLUGIN: JSX list: {symbols['jsx_components']}")
        
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
                    import tree_sitter
                    import tree_sitter_typescript
                    # CRITICAL FIX: tree_sitter_typescript.language_typescript() returns a raw C pointer (integer),
                    # not a Language object. We must wrap it with tree_sitter.Language(pointer, name) to create
                    # a proper Language instance that set_language() expects. This fixes the "Argument to 
                    # set_language must be a Language" error that was blocking all TypeScript/TSX parsing.
                    lang_ptr = tree_sitter_typescript.language_typescript()
                    return tree_sitter.Language(lang_ptr, 'typescript')
                except ImportError:
                    logging.debug("tree_sitter_typescript not available")
                except (AttributeError, TypeError) as e:
                    logging.debug(f"tree_sitter_typescript error: {e}")
            
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
        
        # Import Symbol and SymbolType if symbol_table is available
        if self.symbol_table:
            from ...symbol_table import Symbol, SymbolType
        
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
            (r"export\s+function\s+(\w+)\s*\(", True, False),  # export function
            (r"export\s+default\s+function\s+(\w+)\s*\(", True, True),  # export default function
            (r"function\s+(\w+)\s*\(", False, False),  # function
            (r"export\s+const\s+(\w+)\s*=\s*\([^)]*\)\s*=>\s*{", True, False),  # export const arrow
            (r"const\s+(\w+)\s*=\s*\([^)]*\)\s*=>\s*{", False, False),  # const arrow
            (r"let\s+(\w+)\s*=\s*\([^)]*\)\s*=>\s*{", False, False),  # let arrow
            (r"var\s+(\w+)\s*=\s*\([^)]*\)\s*=>\s*{", False, False),  # var arrow
            (r"const\s+(\w+)\s*=\s*function\s*\(", False, False),  # const function expression
        ]
        
        declared_names = set()
        for pattern, is_exported, is_default in func_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match_obj in matches:
                func_name = match_obj.group(1)
                if func_name and func_name not in declared_names:
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
                    
                    # Add to symbol table
                    if self.symbol_table:
                        line_number = content[:match_obj.start()].count('\n') + 1
                        symbol = Symbol(
                            name=func_name,
                            symbol_type=SymbolType.FUNCTION,
                            file_path=relative_path,
                            line_number=line_number,
                            scope="module",
                            is_exported=is_exported,
                            is_default_export=is_default,
                            metadata={"is_component": is_component, "node_type": node_type.value}
                        )
                        self.symbol_table.add_symbol(symbol)
        
        # Extract class declarations
        class_patterns = [
            (r"export\s+class\s+(\w+)\s*(?:extends\s+\w+)?\s*{", True, False),  # export class
            (r"export\s+default\s+class\s+(\w+)\s*(?:extends\s+\w+)?\s*{", True, True),  # export default class
            (r"class\s+(\w+)\s*(?:extends\s+\w+)?\s*{", False, False),  # class
        ]
        
        for pattern, is_exported, is_default in class_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match_obj in matches:
                class_name = match_obj.group(1)
                if class_name and class_name not in declared_names:
                    declared_names.add(class_name)
                    c4_level = self._determine_c4_level(NodeType.CLASS, relative_path, class_name, is_entry)
                    node_id = self._generate_node_id(relative_path, class_name)
                    graph_node = Node(
                        id=node_id, type=NodeType.CLASS, file=relative_path, name=class_name,
                        metadata={"is_entry": is_entry, "c4_level": c4_level}
                    )
                    nodes.append(graph_node)
                    
                    # Add to symbol table
                    if self.symbol_table:
                        line_number = content[:match_obj.start()].count('\n') + 1
                        symbol = Symbol(
                            name=class_name,
                            symbol_type=SymbolType.CLASS,
                            file_path=relative_path,
                            line_number=line_number,
                            scope="module",
                            is_exported=is_exported,
                            is_default_export=is_default,
                            metadata={"node_type": NodeType.CLASS.value}
                        )
                        self.symbol_table.add_symbol(symbol)
        
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
                
                # FALLBACK: If tree-sitter didn't find JSX (common with TypeScript), use regex
                if not jsx_components:
                    import re
                    jsx_pattern = r'<([A-Z][a-zA-Z0-9]*)'  # Matches <ComponentName
                    jsx_matches = re.findall(jsx_pattern, content)
                    jsx_components = list(set(jsx_matches))  # Remove duplicates
                    if jsx_components:
                        logging.info(f"DEBUG JSX REGEX: Found {len(jsx_components)} components via regex: {jsx_components[:5]}")
            
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
        
        # Import Symbol and SymbolType if symbol_table is available
        if self.symbol_table:
            from ...symbol_table import Symbol, SymbolType, SymbolReference
        
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
                        
                        # Add to symbol table
                        if self.symbol_table:
                            # Check if it's exported
                            is_exported = self._is_exported(node, content, tree)
                            is_default = self._is_default_export(node, content, tree)
                            
                            symbol = Symbol(
                                name=func_name,
                                symbol_type=SymbolType.FUNCTION,
                                file_path=relative_path,
                                line_number=name_node.start_point[0] + 1,
                                scope="module",
                                is_exported=is_exported,
                                is_default_export=is_default,
                                metadata={"is_component": is_component, "node_type": node_type.value}
                            )
                            self.symbol_table.add_symbol(symbol)
            
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
                        
                        # Add to symbol table
                        if self.symbol_table:
                            is_exported = self._is_exported(node, content, tree)
                            is_default = self._is_default_export(node, content, tree)
                            
                            symbol = Symbol(
                                name=class_name,
                                symbol_type=SymbolType.CLASS,
                                file_path=relative_path,
                                line_number=name_node.start_point[0] + 1,
                                scope="module",
                                is_exported=is_exported,
                                is_default_export=is_default,
                                metadata={"node_type": NodeType.CLASS.value}
                            )
                            self.symbol_table.add_symbol(symbol)
                
                # NEW: Track class heritage (extends/implements)
                if self.symbol_table:
                    self._track_class_heritage(node, content, relative_path)
            
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
                            
                            # Add to symbol table
                            if self.symbol_table:
                                is_exported = self._is_exported(node.parent, content, tree)
                                is_default = self._is_default_export(node.parent, content, tree)
                                
                                symbol = Symbol(
                                    name=var_name,
                                    symbol_type=SymbolType.FUNCTION,
                                    file_path=relative_path,
                                    line_number=child.start_point[0] + 1,
                                    scope="module",
                                    is_exported=is_exported,
                                    is_default_export=is_default,
                                    metadata={"is_component": is_component, "node_type": node_type.value}
                                )
                                self.symbol_table.add_symbol(symbol)
                        elif var_name and not is_function:
                            # Add constants and variables to symbol table
                            if self.symbol_table:
                                # Check if it's const or let to determine type
                                is_const = node.type == 'const'
                                parent_text = self._capture_text(content, node.parent) if node.parent else ""
                                is_exported = 'export' in parent_text
                                
                                symbol = Symbol(
                                    name=var_name,
                                    symbol_type=SymbolType.CONSTANT if is_const else SymbolType.VARIABLE,
                                    file_path=relative_path,
                                    line_number=child.start_point[0] + 1,
                                    scope="module",
                                    is_exported=is_exported,
                                    is_default_export=False,
                                    metadata={}
                                )
                                self.symbol_table.add_symbol(symbol)
            
            # NEW: Track function calls
            elif node.type == 'call_expression' and self.symbol_table:
                self._track_function_call(node, content, relative_path)
            
            # NEW: Track class instantiation
            elif node.type == 'new_expression' and self.symbol_table:
                self._track_instantiation(node, content, relative_path)
            
            # Recursively walk children
            for child in node.children:
                walk_node(child)
        
        walk_node(tree.root_node)
        return declared_names
    
    def _extract_imports(self, tree, content: str) -> List[dict]:
        """Extract import statements with metadata."""
        imports = []
        
        def walk_node(node):
            if node.type == 'import_statement':
                # Get line number
                line_number = node.start_point[0] + 1
                
                # Extract import path and imported names
                import_path = None
                imported_names = []
                is_default = False
                
                for child in node.children:
                    # Get import path from string literal
                    if child.type == 'string':
                        import_path = self._capture_text(content, child).strip().strip('"\'')
                    
                    # Get imported names from import clause
                    elif child.type == 'import_clause':
                        for subchild in child.children:
                            if subchild.type == 'identifier':
                                # Default import
                                imported_names.append(self._capture_text(content, subchild))
                                is_default = True
                            elif subchild.type == 'named_imports':
                                # Named imports
                                for name_node in subchild.children:
                                    if name_node.type == 'import_specifier':
                                        for id_node in name_node.children:
                                            if id_node.type == 'identifier':
                                                imported_names.append(self._capture_text(content, id_node))
                
                if import_path:
                    imports.append({
                        'path': import_path,
                        'line': line_number,
                        'names': imported_names,
                        'is_default': is_default
                    })
            
            # Recursively walk children
            for child in node.children:
                walk_node(child)
        
        walk_node(tree.root_node)
        return imports
    
    def _extract_jsx_components(self, tree, content: str) -> List[str]:
        """Extract JSX component usage."""
        jsx_components = []
        node_types_seen = set()
        
        def walk_node(node):
            node_types_seen.add(node.type)
            # JSX opening elements
            if node.type in ['jsx_opening_element', 'jsx_self_closing_element']:
                for child in node.children:
                    if child.type == 'identifier':
                        jsx_name = self._capture_text(content, child).strip()
                        if jsx_name and jsx_name[0].isupper():  # Component names start with uppercase
                            jsx_components.append(jsx_name)
                        break
            
            # Recursively walk children
            for child in node.children:
                walk_node(child)
        
        walk_node(tree.root_node)
        result = list(set(jsx_components))
        if not result:
            logging.info(f"DEBUG JSX: No components found. Node types seen: {sorted(list(node_types_seen))[:20]}")
        return result  # Remove duplicates
    
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
    
    def _is_exported(self, node, content: str, tree) -> bool:
        """Check if a node is exported."""
        if not node:
            return False
        
        # Check if parent is export_statement
        current = node
        while current:
            if current.type in ['export_statement', 'export_declaration']:
                return True
            # Check if text around node contains 'export'
            if current.parent and current.parent.type == 'program':
                # Get a small window around the node
                start = max(0, current.start_byte - 50)
                end = min(len(content), current.start_byte + 10)
                window = content[start:end]
                if 'export' in window:
                    return True
            current = current.parent if hasattr(current, 'parent') else None
        
        return False
    
    def _is_default_export(self, node, content: str, tree) -> bool:
        """Check if a node is a default export."""
        if not node:
            return False
        
        # Check parent chain for default export
        current = node
        while current:
            if current.type in ['export_statement', 'export_declaration']:
                # Check if it contains 'default'
                export_text = self._capture_text(content, current)
                if 'default' in export_text:
                    return True
            current = current.parent if hasattr(current, 'parent') else None
        
        return False
    
    def _track_function_call(self, node, content: str, relative_path: str):
        """Track function call references for edge creation.
        
        Handles:
        - Regular function calls: foo()
        - Method calls: obj.method()
        - Async calls: await foo()
        """
        if not self.symbol_table:
            return
        
        # Extract function name from call_expression
        function_node = node.child_by_field_name('function')
        if not function_node:
            return
        
        # Determine if this is an async call
        is_async = False
        parent = node.parent if hasattr(node, 'parent') else None
        while parent:
            if parent.type == 'await_expression':
                is_async = True
                break
            # Also check if parent is async function
            if parent.type in ['arrow_function', 'function_declaration', 'method_definition']:
                # Check if parent has async modifier
                parent_text = self._capture_text(content, parent)
                if parent_text.strip().startswith('async '):
                    is_async = True
                break
            parent = parent.parent if hasattr(parent, 'parent') else None
        
        # Extract target symbol name
        target_name = None
        if function_node.type == 'identifier':
            # Simple call: foo()
            target_name = self._capture_text(content, function_node)
        elif function_node.type == 'member_expression':
            # Method call: obj.method()
            property_node = function_node.child_by_field_name('property')
            if property_node:
                target_name = self._capture_text(content, property_node)
        
        if not target_name:
            return
        
        # Create reference with context
        context_info = {
            'is_async': is_async,
            'call_type': 'async_call' if is_async else 'function_call',
            'line': node.start_point[0] + 1
        }
        
        # Add reference to symbol table
        reference = SymbolReference(
            symbol_name=target_name,
            file_path=relative_path,
            line_number=node.start_point[0] + 1,
            context='async_call' if is_async else 'call',
            metadata=context_info
        )
        self.symbol_table.add_reference(reference)
        logging.debug(f"DEBUG: Added function_call reference: {target_name} in {relative_path} (context: {'async_call' if is_async else 'call'})")
    
    def _track_instantiation(self, node, content: str, relative_path: str):
        """Track class instantiation references for edge creation.
        
        Handles:
        - new expressions: new ClassName()
        - Constructor calls with arguments
        """
        if not self.symbol_table:
            return
        
        # Extract class name from new_expression
        # new_expression: new <constructor> ( <arguments> )
        constructor_node = None
        for child in node.children:
            if child.type == 'identifier' or child.type == 'member_expression':
                constructor_node = child
                break
        
        if not constructor_node:
            return
        
        # Extract class name
        class_name = None
        if constructor_node.type == 'identifier':
            # Simple: new Foo()
            class_name = self._capture_text(content, constructor_node)
        elif constructor_node.type == 'member_expression':
            # Namespaced: new module.Foo()
            property_node = constructor_node.child_by_field_name('property')
            if property_node:
                class_name = self._capture_text(content, property_node)
        
        if not class_name:
            return
        
        # Create reference with context
        context_info = {
            'instantiation': True,
            'line': node.start_point[0] + 1
        }
        
        # Add reference to symbol table
        reference = SymbolReference(
            symbol_name=class_name,
            file_path=relative_path,
            line_number=node.start_point[0] + 1,
            context='instantiate',
            metadata=context_info
        )
        self.symbol_table.add_reference(reference)
    
    def _track_class_heritage(self, node, content: str, relative_path: str):
        """Track class inheritance and interface implementation.
        
        Handles:
        - extends clause: class Foo extends Bar
        - implements clause: class Foo implements IBar
        """
        if not self.symbol_table:
            return
        
        # Find heritage clause (extends/implements)
        heritage_clause = node.child_by_field_name('heritage')
        if not heritage_clause:
            return
        
        # Process each heritage type
        for child in heritage_clause.children:
            if child.type == 'extends_clause':
                # Track extends relationship
                for identifier in child.children:
                    if identifier.type == 'identifier':
                        parent_class = self._capture_text(content, identifier)
                        
                        context_info = {
                            'relationship': 'extends',
                            'line': child.start_point[0] + 1
                        }
                        
                        reference = SymbolReference(
                            symbol_name=parent_class,
                            file_path=relative_path,
                            line_number=child.start_point[0] + 1,
                            context='extends',
                            metadata=context_info
                        )
                        self.symbol_table.add_reference(reference)
            
            elif child.type == 'implements_clause' or child.type == 'class_heritage':
                # Track implements relationship
                for identifier in child.children:
                    if identifier.type == 'identifier' or identifier.type == 'type_identifier':
                        interface_name = self._capture_text(content, identifier)
                        
                        # Skip keywords
                        if interface_name in ['implements', 'extends']:
                            continue
                        
                        context_info = {
                            'relationship': 'implements',
                            'line': child.start_point[0] + 1
                        }
                        
                        reference = SymbolReference(
                            symbol_name=interface_name,
                            file_path=relative_path,
                            line_number=child.start_point[0] + 1,
                            context='implements',
                            metadata=context_info
                        )
                        self.symbol_table.add_reference(reference)