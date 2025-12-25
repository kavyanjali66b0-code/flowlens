"""
Language parser for analyzing source code files (Fixed version).
"""

import os
import ast
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, TYPE_CHECKING

import tree_sitter
from tree_sitter import Language, Parser

from .models import Node, Edge, NodeType, EdgeType, ProjectType, PathUtils
from .plugins.loader import discover_language_plugins

if TYPE_CHECKING:
    from .symbol_table import SymbolTable

# -------------------------------------------------------------------
# Tree-sitter language setup (package-first, compiled fallback)
# -------------------------------------------------------------------
_LANGUAGE_CACHE: Dict[str, Any] = {}


def _compiled_lib_path() -> Optional[str]:
    ext = "dll" if os.name == "nt" else "so"
    lib_path = Path(__file__).resolve().parent.parent / "build" / f"languages.{ext}"
    return str(lib_path) if lib_path.exists() else None


def _load_from_compiled(name: str) -> Optional[Any]:
    """Load language from a compiled shared library if available."""
    try:
        lib_path = _compiled_lib_path()
        if not lib_path:
            return None
        cache_key = f"compiled::{name}::{lib_path}"
        if cache_key in _LANGUAGE_CACHE:
            return _LANGUAGE_CACHE[cache_key]
        # Updated for new API
        lang = Language(lib_path, name)
        _LANGUAGE_CACHE[cache_key] = lang
        logging.debug(f"Loaded tree-sitter language '{name}' from compiled library: {lib_path}")
        return lang
    except Exception as e:
        logging.debug(f"Failed compiled load for '{name}': {e}")
        return None


class LanguageParser:
    """Parses source code files and extracts semantic information."""

    def __init__(
        self, 
        project_path: str, 
        user_config: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[callable] = None,
        symbol_table: Optional['SymbolTable'] = None
    ):
        """Initialize the language parser.

        Args:
            project_path: Path to the project directory
            user_config: Optional user configuration for C4 level overrides
            progress_callback: Optional callback for progress updates (current, total)
            symbol_table: Optional SymbolTable instance for tracking symbols across files
        """
        self.project_path = Path(project_path)
        self.user_config = user_config or {}
        self.progress_callback = progress_callback
        self.symbol_table = symbol_table
        self.file_asts: Dict[str, Any] = {}
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []
        self.node_registry: Dict[str, Node] = {}
        self.file_symbols: Dict[str, Dict[str, Any]] = {}
        self.module_index: Dict[str, str] = {}
        self.parsed_files: Set[str] = set()
        # Incremental parsing cache: relative_path -> md5 hash
        self.file_cache: Dict[str, str] = {}
        # Parse errors collection
        self.parse_errors: List = []  # Will store ParseError objects
        
        # Auto-discover and initialize language plugins
        self.language_plugins = discover_language_plugins(
            project_path=str(project_path),
            user_config=user_config,
            symbol_table=symbol_table
        )
        logging.info(f"Loaded {len(self.language_plugins)} language plugins")
        
        # NEW: Load path aliases from TypeScript config
        from .config_reader import TsConfigReader
        self.path_aliases = TsConfigReader.read_path_aliases(self.project_path)
        if self.path_aliases:
            logging.info(f"Loaded {len(self.path_aliases)} TypeScript path aliases")
        
        # NEW: Add memory monitoring
        from .memory_monitor import MemoryMonitor
        self.memory_monitor = MemoryMonitor(max_memory_mb=512)
        self.memory_monitor.register_cleanup(self._cleanup_caches)
        
        # Progress tracking
        self.files_to_parse_count = 0
        self.files_parsed_count = 0

    def _get_ts_language(self, ext: str):
        """Return a Tree-sitter language for JS/TS/TSX using updated API."""
        try:
            if ext in ['.js', '.jsx']:
                try:
                    import tree_sitter_javascript
                    return tree_sitter_javascript.language()
                except ImportError:
                    try:
                        from tree_sitter_languages import get_language
                        return get_language('javascript')
                    except ImportError:
                        return _load_from_compiled('javascript')
            
            if ext in ['.ts', '.tsx']:
                try:
                    import tree_sitter_typescript
                    return tree_sitter_typescript.language()
                except ImportError:
                    try:
                        from tree_sitter_languages import get_language
                        return get_language('typescript')
                    except ImportError:
                        return _load_from_compiled('typescript') or _load_from_compiled('tsx')
                        
        except Exception as e:
            logging.warning(f"Failed to load Tree-sitter language for {ext}: {e}")
        return None

    def _get_java_language(self):
        """Return a Tree-sitter language object for Java using updated API."""
        try:
            try:
                import tree_sitter_java
                return tree_sitter_java.language()
            except ImportError:
                try:
                    from tree_sitter_languages import get_language
                    return get_language('java')
                except ImportError:
                    return _load_from_compiled('java')
        except Exception as e:
            logging.warning(f"Failed to load Tree-sitter Java language: {e}")
        return None

    def _capture_text(self, source_code: str, node: Any) -> str:
        """Extract text content from a tree-sitter node."""
        return source_code[node.start_byte:node.end_byte]

    def _determine_c4_level(self, node_type: NodeType, file_path: str, name: str, is_entry: bool = False) -> str:
        """Determine C4 model level for a node based on type, file, and context."""
        # User overrides from .devscope.yml
        try:
            overrides = (self.user_config or {}).get('c4_overrides', {})
            # Path-based overrides: e.g., { "system": ["src/api/"] }
            for level, patterns in (overrides.get('path_contains', {}) or {}).items():
                for pat in patterns:
                    if pat and pat.lower() in file_path.lower():
                        return level
            # Type-based overrides: e.g., { "component": ["component", "view"] }
            for level, types in (overrides.get('node_types', {}) or {}).items():
                if node_type.value in types:
                    return level
        except Exception:
            pass
        
        # System level: API endpoints and external boundaries
        if node_type == NodeType.API_ENDPOINT:
            return "system"
        
        # Container level: Entry modules, main applications, root services
        if is_entry or node_type == NodeType.MODULE:
            # Check if it's a main entry file
            file_lower = file_path.lower()
            name_lower = name.lower()
            
            # Entry points and main modules
            if (name_lower in ['main', 'app', 'index'] or 
                'main.' in file_lower or 
                file_lower.endswith(('main.tsx', 'main.ts', 'main.js', 'app.tsx', 'app.py', 'urls.py')) or
                'springbootapplication' in name_lower):
                return "container"
        
        # Component level: React components, views, controllers, services
        if node_type in [NodeType.COMPONENT, NodeType.VIEW, NodeType.CONTROLLER, NodeType.SERVICE]:
            return "component"
        
        # Component level for modules that represent feature boundaries
        if node_type == NodeType.MODULE:
            # Check if it's a feature module (pages, components directories)
            if any(segment in file_path.lower() for segment in ['pages', 'components', 'views', 'controllers', 'services']):
                return "component"
        
        # Code level: Functions, classes, utilities
        if node_type in [NodeType.FUNCTION, NodeType.CLASS, NodeType.MODEL]:
            return "code"
        
        # Default to code level for unspecified types
        return "code"

    def parse_project(self, entry_points: List[Dict], project_type: ProjectType):
        """Parse project files starting from entry points."""
        logging.info(f"Starting parse for project: {self.project_path}")
        
        # NEW: Check memory before starting
        exceeded, msg = self.memory_monitor.check_memory_threshold()
        if exceeded:
            raise MemoryError(f"Insufficient memory to start parsing: {msg}")
        
        # Build index for module resolution
        self._build_module_index()
        
        # Start with entry points
        for entry in entry_points:
            file_path = self.project_path / entry['value']
            logging.debug(f"Checking entry point: {file_path} (exists: {file_path.exists()})")
            if file_path.exists() and file_path.is_file():
                self._parse_file(file_path, is_entry=True)
                
                # NEW: Check memory after each entry point
                self.memory_monitor.check_memory_threshold()
        
        # Parse other relevant files based on project type
        self._parse_additional_files(project_type)
        
        # Perform semantic analysis
        self._analyze_relationships()
        
        # NEW: Log memory statistics
        stats = self.memory_monitor.get_statistics()
        logging.info(f"Parse complete. Memory stats: {stats}")
        logging.info(f"Found {len(self.nodes)} nodes and {len(self.edges)} edges")
        
        # NEW: Final cleanup
        self._cleanup_caches()
    
    def _cleanup_caches(self):
        """Clean up parser caches to free memory."""
        logging.info(f"Cleaning parser caches. Current items: ASTs={len(self.file_asts)}, Cache={len(self.file_cache)}")
        
        # Clear AST cache (CRITICAL FIX for memory leak)
        self.file_asts.clear()
        
        # Keep only recent cache entries (last 1000 files)
        if len(self.file_cache) > 1000:
            # Keep most recently accessed
            sorted_cache = sorted(
                self.file_cache.items(),
                key=lambda x: hash(x[0]),  # Simple hash-based selection
                reverse=True
            )
            self.file_cache = dict(sorted_cache[:1000])
        
        logging.info(f"Caches cleaned. Remaining: ASTs={len(self.file_asts)}, Cache={len(self.file_cache)}")
        
    def _parse_file(self, file_path: Path, is_entry: bool = False):
        """Parse individual file using appropriate language plugin."""
        ext = file_path.suffix.lower()
        # Keep Windows-style paths with backslashes
        relative_path = str(file_path.relative_to(self.project_path))

        # Skip node_modules and TypeScript declaration files
        if "node_modules" in relative_path or relative_path.endswith(".d.ts"):
            logging.debug(f"Skipping external/declaration file: {relative_path}")
            return

        # Skip if already parsed
        if relative_path in self.parsed_files:
            return

        try:
            # --- Caching logic ---
            content = file_path.read_text(encoding='utf-8')
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            if self.file_cache.get(relative_path) == content_hash:
                logging.debug(f"Cache hit for {relative_path}. Skipping parse.")
                self.parsed_files.add(relative_path)
                return

            # Mark as parsed before parsing to avoid recursion issues
            self.parsed_files.add(relative_path)

            # Find the correct plugin
            plugin = self._find_plugin_for_file(file_path)
            if plugin:
                # Call the plugin's parse method
                plugin_nodes, plugin_symbols = plugin.parse(file_path, content, is_entry)

                # Integrate the results
                if plugin_nodes:
                    self.nodes.extend(plugin_nodes)
                    for node in plugin_nodes:
                        self.node_registry[node.id] = node

                if plugin_symbols:
                    self.file_symbols[relative_path] = plugin_symbols
                    
                    # DEBUG: Log JSX component storage
                    jsx_comps = plugin_symbols.get('jsx_components', [])
                    if jsx_comps:
                        logging.info(f"DEBUG PARSER: Stored JSX components for {relative_path}: {jsx_comps}")
                    
                    # Create import edges from plugin symbols
                    for imp in plugin_symbols.get('imports', []):
                        # Handle both old string format and new dict format
                        if isinstance(imp, dict):
                            self._add_import_edge(
                                relative_path, 
                                imp['path'], 
                                line_number=imp.get('line'),
                                imported_names=imp.get('names', [])
                            )
                        else:
                            # Old string format (backward compatibility)
                            self._add_import_edge(relative_path, imp)

                logging.debug(f"Parsed {relative_path}: {len(plugin_nodes)} nodes created")
            else:
                # Fallback for HTML files and others not yet pluginized
                if ext == '.html':
                    self._parse_html_file(file_path, relative_path, is_entry)

            # Update cache on successful parse
            self.file_cache[relative_path] = content_hash
        except Exception as e:
            logging.error(f"Failed to parse file {relative_path}: {e}")
    
    def _find_plugin_for_file(self, file_path: Path):
        """Find the correct plugin for a file."""
        ext = file_path.suffix.lower()
        for plugin in self.language_plugins:
            if plugin.can_parse(ext):
                return plugin
        return None
    
    def _create_js_import_edges(self, relative_path: str, content: str):
        """Create import edges for JavaScript/TypeScript files using updated API."""
        try:
            lang = self._get_ts_language(Path(relative_path).suffix.lower())
            if not lang:
                return
                
            parser = Parser()
            parser.set_language(lang)  # Fixed: use set_language instead of parser.language =
            tree = parser.parse(bytes(content, 'utf-8'))
            root = tree.root_node
            
            source_module_id = self._generate_node_id(relative_path, 'module')
            
            # Simple import extraction without queries for now
            imports = self._extract_js_imports_simple(tree, content)
            
            for import_info in imports:
                import_path = import_info['path']
                if import_path and import_path.startswith('.'):
                    target_rel = self._resolve_import_target(relative_path, import_path)
                    if target_rel:
                        target_module_id = self._generate_node_id(target_rel, 'module')
                        
                        # Check if target module exists
                        if target_module_id in self.node_registry:
                            # Create edge with enriched metadata
                            edge = Edge(
                                source=source_module_id,
                                target=target_module_id,
                                type=EdgeType.IMPORTS,
                                metadata={
                                    'reason': 'import_statement',
                                    'import_path': import_path,
                                    'line_number': import_info['line'],
                                    'source_file': str(relative_path).replace('/', '\\'),
                                    'target_file': str(target_rel).replace('/', '\\'),
                                    'imported_names': import_info['imported_names'],
                                    'is_default': import_info['is_default']
                                }
                            )
                            self.edges.append(edge)
        except Exception as e:
            logging.error(f"Failed to create JS import edges for {relative_path}: {e}")
    
    def _extract_js_imports_simple(self, tree, content: str) -> List[dict]:
        """Simple extraction of import paths and line numbers by walking the tree."""
        imports = []
        
        def walk_node(node):
            if node.type == 'import_statement':
                # Get line number (0-indexed, convert to 1-indexed)
                line_number = node.start_point[0] + 1
                
                # Extract imported names
                imported_names = []
                import_path = None
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
                        'imported_names': imported_names,
                        'is_default': is_default
                    })
            
            # Recursively walk children
            for child in node.children:
                walk_node(child)
        
        walk_node(tree.root_node)
        return imports
    
    def _create_python_import_edges(self, relative_path: str, content: str):
        """Create import edges for Python files."""
        try:
            tree = ast.parse(content)
            source_module_id = self._generate_node_id(relative_path, 'module')
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        target_rel = self._resolve_import_target(relative_path, alias.name)
                        if target_rel:
                            target_module_id = self._generate_node_id(target_rel, 'module')
                            # Check if target module exists
                            if target_module_id in self.node_registry:
                                edge = Edge(
                                    source=source_module_id,
                                    target=target_module_id,
                                    type=EdgeType.IMPORTS,
                                    metadata={
                                        'reason': 'import_statement',
                                        'import_path': alias.name,
                                        'line_number': node.lineno,
                                        'source_file': str(relative_path).replace('/', '\\'),
                                        'target_file': str(target_rel).replace('/', '\\'),
                                        'imported_names': [alias.asname if alias.asname else alias.name],
                                        'is_default': False
                                    }
                                )
                                self.edges.append(edge)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        target_rel = self._resolve_import_target(relative_path, node.module)
                        if target_rel:
                            target_module_id = self._generate_node_id(target_rel, 'module')
                            # Check if target module exists
                            if target_module_id in self.node_registry:
                                # Collect imported names
                                imported_names = [
                                    alias.asname if alias.asname else alias.name 
                                    for alias in node.names
                                ]
                                edge = Edge(
                                    source=source_module_id,
                                    target=target_module_id,
                                    type=EdgeType.IMPORTS,
                                    metadata={
                                        'reason': 'import_statement',
                                        'import_path': node.module,
                                        'line_number': node.lineno,
                                        'source_file': str(relative_path).replace('/', '\\'),
                                        'target_file': str(target_rel).replace('/', '\\'),
                                        'imported_names': imported_names,
                                        'is_default': False
                                    }
                                )
                                self.edges.append(edge)
        except Exception as e:
            logging.error(f"Failed to create Python import edges for {relative_path}: {e}")

    def _parse_html_file(self, file_path: Path, relative_path: str, is_entry: bool):
        """Parse HTML file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            node_id = self._generate_node_id(relative_path, "template")
            c4_level = self._determine_c4_level(NodeType.TEMPLATE, relative_path, file_path.stem, is_entry)
            graph_node = Node(
                id=node_id,
                type=NodeType.TEMPLATE,
                file=relative_path,
                name=file_path.stem,
                metadata={"is_entry": is_entry, "c4_level": c4_level}
            )
            self.nodes.append(graph_node)
            self.node_registry[node_id] = graph_node
            
        except (UnicodeDecodeError, OSError) as e:
            logging.error(f"Failed to read HTML file {relative_path}: {e}")
    
    def _parse_additional_files(self, project_type: ProjectType):
        """Parse additional files based on project type."""
        extensions_to_parse = set()
        
        if project_type in [ProjectType.REACT_VITE, ProjectType.ANGULAR, ProjectType.EXPRESS_NODE]:
            extensions_to_parse.update(['.js', '.jsx', '.ts', '.tsx'])
        elif project_type in [ProjectType.DJANGO, ProjectType.PYTHON_APP]:
            extensions_to_parse.add('.py')
        elif project_type in [ProjectType.MAVEN_JAVA, ProjectType.GRADLE_JAVA, ProjectType.SPRING_BOOT]:
            extensions_to_parse.add('.java')
        
        # NEW: Count files first for progress tracking
        files_to_parse = []
        for location in ['src', 'app', 'lib', 'components', 'pages', 'api', 'server']:
            location_path = self.project_path / location
            if location_path.exists():
                for root, dirs, files in os.walk(location_path):
                    # Skip node_modules and other build directories
                    dirs[:] = [d for d in dirs if d not in {
                        'node_modules', '_pycache_', '.git', '__pycache__'
                    }]
                    
                    for file in files:
                        file_path = Path(root) / file
                        if file_path.suffix.lower() in extensions_to_parse:
                            files_to_parse.append(file_path)
        
        # NEW: Set total for progress tracking
        self.files_to_parse_count = len(files_to_parse)
        self.files_parsed_count = 0
        
        # Parse files with progress tracking and memory checks
        for idx, file_path in enumerate(files_to_parse):
            self._parse_file(file_path)
            self.files_parsed_count += 1
            
            # Report progress if callback is set
            if self.progress_callback and idx % 10 == 0:  # Every 10 files
                self.progress_callback(self.files_parsed_count, self.files_to_parse_count)
            
            # Check memory every 50 files
            if idx % 50 == 0:
                self.memory_monitor.check_memory_threshold()
    
    def _analyze_relationships(self):
        """Analyze semantic relationships between nodes."""
        logging.info("Analyzing relationships between nodes")
        
        # RENDERS: use jsx_components captured per file
        # NOTE: This is legacy code - most edge creation is now in SemanticAnalyzer
        # Keeping this for backward compatibility but with self-reference protection
        for file, meta in list(self.file_symbols.items()):
            jsx_list = (meta or {}).get('jsx_components', [])
            if not jsx_list:
                continue
            source_components = [n for n in self.nodes if n.file == file and n.type == NodeType.COMPONENT]
            if not source_components:
                continue
            for jsx_name in jsx_list:
                targets = [n for n in self.nodes if n.name == jsx_name and n.type == NodeType.COMPONENT]
                for src in source_components:
                    for tgt in targets:
                        # CRITICAL: Prevent self-referencing edges
                        if src.id == tgt.id:
                            logging.debug(f"Skipping self-referencing RENDERS edge: {src.name} -> {tgt.name}")
                            continue
                        
                        # Check if edge already exists
                        edge_exists = any(
                            e.source == src.id and 
                            e.target == tgt.id and 
                            e.type == EdgeType.RENDERS
                            for e in self.edges
                        )
                        
                        if not edge_exists:
                            edge = Edge(source=src.id, target=tgt.id, type=EdgeType.RENDERS)
                            self.edges.append(edge)
                            logging.debug(f"Created RENDERS edge: {src.name} -> {tgt.name}")

        # CALLS: JS/TS
        for n in self.nodes:
            if n.file.endswith(('.js', '.jsx', '.ts', '.tsx')) and n.type in {NodeType.FUNCTION, NodeType.COMPONENT, NodeType.CLASS}:
                self._analyze_js_calls(n.file, n)

        # CALLS: Python
        for n in self.nodes:
            if n.file.endswith('.py') and n.type in {NodeType.FUNCTION, NodeType.CLASS, NodeType.VIEW}:
                self._analyze_py_calls(n.file, n)
    
    def _add_import_edge(self, from_file: str, import_path: str, line_number: int = None, imported_names: list = None):
        """Add import edge between files with metadata.
        
        Args:
            from_file: Source file path
            import_path: Import path (e.g., './Component', 'react')
            line_number: Optional line number where import occurs
            imported_names: Optional list of imported symbol names
        """
        try:
            target_rel = self._resolve_import_target(from_file, import_path)
            if not target_rel:
                logging.debug(f"Could not resolve import target: {from_file} -> {import_path}")
                return
                
            source_module_id = self._generate_node_id(from_file, 'module')
            target_module_id = self._generate_node_id(target_rel, 'module')
            
            # Ensure source module exists
            if source_module_id not in self.node_registry:
                src_name = Path(from_file).stem
                c4_level = self._determine_c4_level(NodeType.MODULE, from_file, src_name, False)
                src_node = Node(
                    id=source_module_id, 
                    type=NodeType.MODULE, 
                    file=from_file, 
                    name=src_name,
                    metadata={"c4_level": c4_level}
                )
                self.nodes.append(src_node)
                self.node_registry[source_module_id] = src_node
                
            # Ensure target module exists
            if target_module_id not in self.node_registry:
                tgt_name = Path(target_rel).stem
                c4_level = self._determine_c4_level(NodeType.MODULE, target_rel, tgt_name, False)
                tgt_node = Node(
                    id=target_module_id, 
                    type=NodeType.MODULE, 
                    file=target_rel, 
                    name=tgt_name,
                    metadata={"c4_level": c4_level}
                )
                self.nodes.append(tgt_node)
                self.node_registry[target_module_id] = tgt_node
                
            # Create the edge with metadata
            metadata = {
                'reason': 'import_statement',
                'import_path': import_path,
                'source_file': str(from_file).replace('/', '\\\\'),
                'target_file': str(target_rel).replace('/', '\\\\')
            }
            
            # Add line number if provided
            if line_number is not None:
                metadata['line_number'] = line_number
            
            # Add imported names if provided
            if imported_names:
                metadata['imported_names'] = imported_names
                metadata['is_default'] = len(imported_names) == 1 and imported_names[0] != '*'
            else:
                metadata['imported_names'] = []
                metadata['is_default'] = False
            
            edge = Edge(source=source_module_id, target=target_module_id, type=EdgeType.IMPORTS, metadata=metadata)
            self.edges.append(edge)
            logging.debug(f"Created import edge: {from_file} -> {target_rel}")
            
        except Exception as e:
            logging.exception(f"Failed to add import edge {from_file} -> {import_path}: {e}")

    def _build_module_index(self):
        """Build index of modules for import resolution."""
        try:
            exts = ['.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs', '.py', '.java']
            src_roots = ['src', 'app', 'lib', 'components', 'pages', 'server', 'api']
            for root, dirs, files in os.walk(self.project_path):
                dirs[:] = [d for d in dirs if d not in {
                    '.git', 'node_modules', '__pycache__', 'build', 'dist', 'target'
                }]
                for fname in files:
                    fp = Path(root) / fname
                    if fp.suffix.lower() not in exts:
                        continue
                    rel = str(fp.relative_to(self.project_path))
                    self.module_index[self._normalize_module_key(rel)] = rel
                    if fp.stem == 'index' and fp.parent != self.project_path:
                        folder_key = self._normalize_module_key(str(fp.parent.relative_to(self.project_path)))
                        self.module_index[folder_key] = rel
                    for root_dir in src_roots:
                        if rel.startswith(root_dir + os.sep):
                            key = self._normalize_module_key(rel[len(root_dir) + 1:])
                            self.module_index[key] = rel
        except Exception as e:
            logging.exception(f"Module index build failed: {e}")

    def _normalize_module_key(self, s: str) -> str:
        """Normalize module key for consistent lookups."""
        return s.replace('\\\\', '/').replace('\\', '/').rstrip('/')

    def _resolve_import_target(self, from_file: str, specifier: str) -> Optional[str]:
        """Resolve import target to actual file path."""
        try:
            # NEW: Check TypeScript path aliases first
            for alias, target_path in self.path_aliases.items():
                if specifier.startswith(alias):
                    # Replace alias with actual path
                    remainder = specifier[len(alias):].lstrip('/')
                    resolved = Path(target_path) / remainder
                    
                    # Try with extensions
                    for ext in ['.ts', '.tsx', '.js', '.jsx']:
                        candidate = resolved.with_suffix(ext)
                        if candidate.exists():
                            try:
                                return str(candidate.relative_to(self.project_path))
                            except ValueError:
                                # Path might be outside project
                                pass
                    
                    # Try index files
                    for index_file in ['index.ts', 'index.tsx', 'index.js', 'index.jsx']:
                        index_path = resolved / index_file
                        if index_path.exists():
                            try:
                                return str(index_path.relative_to(self.project_path))
                            except ValueError:
                                pass
            
            # Original resolution logic
            base_dir = str((self.project_path / from_file).parent)
            candidates: List[Path] = []

            def push_with_exts(p: Path):
                if p.suffix:
                    candidates.append(p)
                    return
                for ext in ['.ts', '.tsx', '.js', '.jsx', '.mjs', '.cjs', '.py', '.java']:
                    candidates.append(p.with_suffix(ext))
                candidates.append(p / 'index.ts')
                candidates.append(p / 'index.tsx')
                candidates.append(p / 'index.js')
                candidates.append(p / 'index.jsx')
                candidates.append(p / '__init__.py')

            if specifier.startswith('./') or specifier.startswith('../'):
                push_with_exts(Path(base_dir) / specifier)
            else:
                norm = self._normalize_module_key(specifier)
                if norm in self.module_index:
                    return self.module_index[norm]
                for root in ['src', 'app', 'lib']:
                    push_with_exts(self.project_path / root / specifier)
                for k, rel in self.module_index.items():
                    if k.endswith('/' + norm) or k == norm:
                        return rel

            for cand in candidates:
                if cand.exists():
                    return str(cand.relative_to(self.project_path))
            return None
        except Exception as e:
            logging.exception(f"Import resolve failed for {specifier} from {from_file}: {e}")
            return None

    def _analyze_js_calls(self, relative_path: str, scope_node: Node):
        """Analyze function calls in JavaScript/TypeScript files."""
        try:
            file_path = self.project_path / relative_path
            if not file_path.exists():
                return
            content = file_path.read_text(encoding='utf-8')
            lang = self._get_ts_language(Path(relative_path).suffix.lower())
            if not lang:
                return
            parser = Parser()
            parser.set_language(lang)  # Fixed: use set_language
            tree = parser.parse(content.encode('utf-8'))
            root = tree.root_node
    
            # Simple call extraction
            called_names = self._extract_called_names(tree, content)
                
            # Link calls to known nodes
            for called in called_names:
                targets = [n for n in self.nodes if n.name == called]
                for tgt in targets:
                    edge = Edge(source=scope_node.id, target=tgt.id, type=EdgeType.CALLS)
                    self.edges.append(edge)
                    logging.debug(f"Created CALLS edge: {scope_node.name} -> {tgt.name}")
        except Exception as e:
            logging.exception(f"JS calls analysis failed for {relative_path}: {e}")

    def _extract_called_names(self, tree, content: str) -> Set[str]:
        """Extract function call names from tree."""
        called_names = set()
        
        def walk_node(node):
            if node.type == 'call_expression':
                # Find the function being called
                for child in node.children:
                    if child.type == 'identifier':
                        func_name = self._capture_text(content, child)
                        if func_name:
                            called_names.add(func_name)
                        break
                    elif child.type == 'member_expression':
                        # For member expressions like obj.method()
                        for grandchild in child.children:
                            if grandchild.type == 'property_identifier':
                                method_name = self._capture_text(content, grandchild)
                                if method_name:
                                    called_names.add(method_name)
                                break
            
            # Recursively walk children
            for child in node.children:
                walk_node(child)
        
        walk_node(tree.root_node)
        return called_names

    def _analyze_py_calls(self, relative_path: str, scope_node: Node):
        """Analyze function calls in Python files."""
        try:
            file_path = self.project_path / relative_path
            if not file_path.exists():
                return
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
            called_names: Set[str] = set()
            for n in ast.walk(tree):
                if isinstance(n, ast.Call):
                    if isinstance(n.func, ast.Name):
                        called_names.add(n.func.id)
                    elif isinstance(n.func, ast.Attribute):
                        called_names.add(n.func.attr)
            for called in called_names:
                targets = [n for n in self.nodes if n.name == called]
                for tgt in targets:
                    edge = Edge(source=scope_node.id, target=tgt.id, type=EdgeType.CALLS)
                    self.edges.append(edge)
                    logging.debug(f"Created CALLS edge: {scope_node.name} -> {tgt.name}")
        except Exception as e:
            logging.exception(f"Python calls analysis failed for {relative_path}: {e}")
    
    def _generate_node_id(self, file_path: str, name: str) -> str:
        """Generate unique node ID using hash for better uniqueness."""
        # Normalize path to forward slashes for cross-platform consistency
        normalized_path = PathUtils.normalize(file_path)
        
        # Use hash to avoid collisions and make IDs more stable
        content = f"{normalized_path}:{name}"
        hash_suffix = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{normalized_path.replace('/', '_').replace('.', '_')}_{name}_{hash_suffix}"