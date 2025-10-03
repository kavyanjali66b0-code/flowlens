"""
CommonJS import resolver for Node.js modules.

Handles:
- require('module')
- require('./relative')
- module.exports = X
- exports.X = Y
- Core modules (fs, path, etc.)
"""

from pathlib import Path
from typing import Optional, List, Dict, Set
from dataclasses import dataclass, field
import re
import json


@dataclass
class RequireStatement:
    """Represents a CommonJS require() statement."""
    module_path: str
    assigned_to: Optional[str] = None  # const X = require('mod')
    destructured_names: List[str] = field(default_factory=list)  # const { A, B } = require('mod')
    is_dynamic: bool = False  # require(variable)
    source_file: str = ""
    line_number: int = 0


@dataclass
class ExportsStatement:
    """Represents a CommonJS exports statement."""
    export_type: str  # 'module.exports', 'exports.X'
    export_name: Optional[str] = None  # For exports.X
    is_object: bool = False  # module.exports = { ... }
    exported_names: List[str] = field(default_factory=list)  # For object exports
    source_file: str = ""
    line_number: int = 0


class CommonJSResolver:
    """Resolves CommonJS module imports using Node.js resolution algorithm."""
    
    # Node.js core modules (built-in)
    CORE_MODULES: Set[str] = {
        'assert', 'buffer', 'child_process', 'cluster', 'crypto', 'dgram',
        'dns', 'domain', 'events', 'fs', 'http', 'https', 'net', 'os',
        'path', 'punycode', 'querystring', 'readline', 'repl', 'stream',
        'string_decoder', 'sys', 'timers', 'tls', 'tty', 'url', 'util',
        'v8', 'vm', 'zlib', 'process', 'console'
    }
    
    def __init__(self, project_root: str):
        """
        Initialize CommonJS import resolver.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.module_cache: Dict[str, Optional[str]] = {}  # Cache resolved paths
        self.package_json_cache: Dict[str, Dict] = {}  # Cache package.json files
    
    def resolve_require(
        self, 
        require_path: str, 
        from_file: str,
        node_modules_lookup: bool = True
    ) -> Optional[str]:
        """
        Resolve a require() statement to absolute file path.
        
        Implements Node.js module resolution algorithm:
        1. Core modules (return None - they're built-in)
        2. Relative paths (./foo, ../bar)
        3. Absolute paths (/foo)
        4. Node modules (node_modules/)
        
        Args:
            require_path: The path in require()
            from_file: The file containing the require
            node_modules_lookup: Whether to search node_modules
            
        Returns:
            Absolute path to the resolved module file, or None if not found
        """
        # Check cache
        cache_key = f"{from_file}::{require_path}"
        if cache_key in self.module_cache:
            return self.module_cache[cache_key]
        
        resolved = None
        
        # 1. Core modules - return None (built-in)
        if require_path in self.CORE_MODULES:
            self.module_cache[cache_key] = None
            return None
        
        # 2. Relative paths
        if require_path.startswith('./') or require_path.startswith('../'):
            resolved = self._resolve_relative_require(require_path, from_file)
        
        # 3. Absolute paths
        elif require_path.startswith('/'):
            resolved = self._resolve_absolute_require(require_path)
        
        # 4. Node modules
        elif node_modules_lookup:
            resolved = self._resolve_node_modules_require(require_path, from_file)
        
        # Cache result
        if resolved:
            self.module_cache[cache_key] = resolved
        
        return resolved
    
    def _resolve_relative_require(self, require_path: str, from_file: str) -> Optional[str]:
        """Resolve relative require('./foo', '../bar')."""
        from_dir = Path(from_file).parent
        target = from_dir / require_path
        
        # Try as file with extensions
        for ext in ['', '.js', '.json', '.node']:
            file_with_ext = Path(str(target) + ext)
            if file_with_ext.is_file():
                return str(file_with_ext.resolve())
        
        # Try as directory with package.json
        if target.is_dir():
            package_json = target / 'package.json'
            if package_json.exists():
                main_file = self._get_package_main(str(target))
                if main_file:
                    return main_file
            
            # Try index files
            for index_name in ['index.js', 'index.json', 'index.node']:
                index_file = target / index_name
                if index_file.exists():
                    return str(index_file.resolve())
        
        return None
    
    def _resolve_absolute_require(self, require_path: str) -> Optional[str]:
        """Resolve absolute require('/foo')."""
        target = Path(require_path)
        
        # Try as file with extensions
        for ext in ['', '.js', '.json', '.node']:
            file_with_ext = Path(str(target) + ext)
            if file_with_ext.is_file():
                return str(file_with_ext.resolve())
        
        # Try as directory
        if target.is_dir():
            for index_name in ['index.js', 'index.json', 'index.node']:
                index_file = target / index_name
                if index_file.exists():
                    return str(index_file.resolve())
        
        return None
    
    def _resolve_node_modules_require(self, module_name: str, from_file: str) -> Optional[str]:
        """Resolve node_modules require('lodash')."""
        # Start from the directory containing from_file
        current_dir = Path(from_file).parent
        
        # Walk up the directory tree looking for node_modules
        while True:
            node_modules = current_dir / 'node_modules' / module_name
            
            if node_modules.exists():
                # Check package.json for main entry
                package_json = node_modules / 'package.json'
                if package_json.exists():
                    main_file = self._get_package_main(str(node_modules))
                    if main_file:
                        return main_file
                
                # Try index files
                for index_name in ['index.js', 'index.json', 'index.node']:
                    index_file = node_modules / index_name
                    if index_file.exists():
                        return str(index_file.resolve())
                
                # Try as file with extensions
                for ext in ['.js', '.json', '.node']:
                    file_with_ext = Path(str(node_modules) + ext)
                    if file_with_ext.is_file():
                        return str(file_with_ext.resolve())
            
            # Move up one directory
            parent = current_dir.parent
            if parent == current_dir:  # Reached root
                break
            current_dir = parent
        
        return None
    
    def _get_package_main(self, package_dir: str) -> Optional[str]:
        """Get main entry point from package.json."""
        package_json_path = Path(package_dir) / 'package.json'
        
        # Check cache
        cache_key = str(package_json_path)
        if cache_key in self.package_json_cache:
            package_data = self.package_json_cache[cache_key]
        else:
            try:
                with open(package_json_path, 'r', encoding='utf-8') as f:
                    package_data = json.load(f)
                    self.package_json_cache[cache_key] = package_data
            except (json.JSONDecodeError, IOError):
                return None
        
        # Get main field (default to index.js)
        main = package_data.get('main', 'index.js')
        main_path = Path(package_dir) / main
        
        # Try with extensions if no extension
        if not main_path.suffix:
            for ext in ['.js', '.json', '.node']:
                file_with_ext = Path(str(main_path) + ext)
                if file_with_ext.exists():
                    return str(file_with_ext.resolve())
        
        if main_path.exists():
            return str(main_path.resolve())
        
        return None
    
    def parse_requires(self, content: str, source_file: str = "") -> List[RequireStatement]:
        """
        Parse require() statements from JavaScript code.
        
        Handles:
        - const X = require('mod')
        - const { A, B } = require('mod')
        - require('mod')  (side-effect)
        - require(variable)  (dynamic)
        
        Args:
            content: JavaScript source code
            source_file: Path to the source file
            
        Returns:
            List of RequireStatement objects
        """
        requires = []
        
        # Pattern 1: const X = require('module')
        pattern1 = r"(?:const|let|var)\s+(\w+)\s*=\s*require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"
        for match in re.finditer(pattern1, content):
            assigned_to, module_path = match.groups()
            line_number = content[:match.start()].count('\n') + 1
            requires.append(RequireStatement(
                module_path=module_path,
                assigned_to=assigned_to,
                source_file=source_file,
                line_number=line_number
            ))
        
        # Pattern 2: const { A, B } = require('module')
        pattern2 = r"(?:const|let|var)\s*\{\s*([^}]+)\s*\}\s*=\s*require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"
        for match in re.finditer(pattern2, content):
            destructured, module_path = match.groups()
            names = [name.strip() for name in destructured.split(',')]
            line_number = content[:match.start()].count('\n') + 1
            requires.append(RequireStatement(
                module_path=module_path,
                destructured_names=names,
                source_file=source_file,
                line_number=line_number
            ))
        
        # Pattern 3: require('module') - side-effect
        pattern3 = r"(?<!const\s)(?<!let\s)(?<!var\s)(?<!=\s)require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"
        for match in re.finditer(pattern3, content):
            module_path = match.group(1)
            line_number = content[:match.start()].count('\n') + 1
            
            # Skip if already captured by pattern1 or pattern2
            if any(r.module_path == module_path and r.line_number == line_number for r in requires):
                continue
            
            requires.append(RequireStatement(
                module_path=module_path,
                source_file=source_file,
                line_number=line_number
            ))
        
        # Pattern 4: require(variable) - dynamic
        pattern4 = r"require\s*\(\s*([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\)"
        for match in re.finditer(pattern4, content):
            variable = match.group(1)
            line_number = content[:match.start()].count('\n') + 1
            requires.append(RequireStatement(
                module_path=variable,
                is_dynamic=True,
                source_file=source_file,
                line_number=line_number
            ))
        
        return requires
    
    def parse_exports(self, content: str, source_file: str = "") -> List[ExportsStatement]:
        """
        Parse module.exports and exports statements.
        
        Handles:
        - module.exports = X
        - module.exports = { A, B }
        - exports.X = Y
        
        Args:
            content: JavaScript source code
            source_file: Path to the source file
            
        Returns:
            List of ExportsStatement objects
        """
        exports = []
        
        # Pattern 1: module.exports = { ... }
        pattern1 = r"module\.exports\s*=\s*\{([^}]+)\}"
        for match in re.finditer(pattern1, content):
            exported_obj = match.group(1)
            # Extract property names
            names = []
            for prop_match in re.finditer(r'(\w+)\s*:', exported_obj):
                names.append(prop_match.group(1))
            
            line_number = content[:match.start()].count('\n') + 1
            exports.append(ExportsStatement(
                export_type='module.exports',
                is_object=True,
                exported_names=names,
                source_file=source_file,
                line_number=line_number
            ))
        
        # Pattern 2: module.exports = X (single value)
        pattern2 = r"module\.exports\s*=\s*(\w+)"
        for match in re.finditer(pattern2, content):
            # Skip if it's an object literal (handled by pattern1)
            if match.group(1) != '{':
                line_number = content[:match.start()].count('\n') + 1
                exports.append(ExportsStatement(
                    export_type='module.exports',
                    export_name=match.group(1),
                    source_file=source_file,
                    line_number=line_number
                ))
        
        # Pattern 3: exports.X = Y
        pattern3 = r"exports\.(\w+)\s*="
        for match in re.finditer(pattern3, content):
            export_name = match.group(1)
            line_number = content[:match.start()].count('\n') + 1
            exports.append(ExportsStatement(
                export_type='exports',
                export_name=export_name,
                source_file=source_file,
                line_number=line_number
            ))
        
        return exports
