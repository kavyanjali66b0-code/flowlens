"""
ES6 import resolver for JavaScript and TypeScript modules.

Handles:
- Named imports: import { A, B } from 'module'
- Default imports: import X from 'module'
- Namespace imports: import * as X from 'module'
- Dynamic imports: import('module')
- Re-exports: export { A } from 'module'
- Side-effect imports: import 'module'
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field


@dataclass
class ImportStatement:
    """Represents a parsed ES6 import statement."""
    module_path: str  # The module being imported from
    imported_names: List[str] = field(default_factory=list)  # Named imports
    default_import: Optional[str] = None  # Default import name
    namespace_import: Optional[str] = None  # Namespace import name (import * as X)
    is_dynamic: bool = False  # Dynamic import()
    is_type_only: bool = False  # TypeScript type-only import
    source_file: str = ""  # File where import appears
    line_number: int = 0


@dataclass
class ExportStatement:
    """Represents a parsed ES6 export statement."""
    exported_names: List[str] = field(default_factory=list)  # Named exports
    default_export: Optional[str] = None  # Default export
    re_export_from: Optional[str] = None  # Re-export from module
    is_type_only: bool = False  # TypeScript type-only export


class ES6ImportResolver:
    """Resolves ES6 module imports to absolute file paths."""
    
    def __init__(self, project_root: str, tsconfig_paths: Optional[Dict[str, List[str]]] = None):
        """
        Initialize ES6 import resolver.
        
        Args:
            project_root: Root directory of the project
            tsconfig_paths: TypeScript path mappings from tsconfig.json
        """
        self.project_root = Path(project_root)
        self.tsconfig_paths = tsconfig_paths or {}
        self.module_cache: Dict[str, str] = {}  # module_name -> resolved_path
        self.package_json_cache: Dict[str, Dict] = {}  # directory -> package.json
        
    def resolve_import(
        self, 
        import_path: str, 
        from_file: str,
        node_modules_lookup: bool = True
    ) -> Optional[str]:
        """
        Resolve an import path to an absolute file path.
        
        Args:
            import_path: The import path (e.g., './utils', '@/components', 'lodash')
            from_file: The file containing the import
            node_modules_lookup: Whether to search node_modules
            
        Returns:
            Absolute path to the resolved module file, or None if not found
        """
        # Check cache
        cache_key = f"{from_file}::{import_path}"
        if cache_key in self.module_cache:
            return self.module_cache[cache_key]
        
        resolved = None
        
        # Determine import type and resolve
        if import_path.startswith('.'):
            # Relative import
            resolved = self._resolve_relative_import(import_path, from_file)
        elif import_path.startswith('@/'):
            # Alias import (common in Vite, Webpack configs)
            resolved = self._resolve_alias_import(import_path)
        elif self.tsconfig_paths and any(import_path.startswith(pattern.rstrip('/*')) for pattern in self.tsconfig_paths):
            # TypeScript path mapping
            resolved = self._resolve_tsconfig_path(import_path)
        elif node_modules_lookup:
            # Node modules import
            resolved = self._resolve_node_modules_import(import_path, from_file)
        
        # Cache result
        if resolved:
            self.module_cache[cache_key] = resolved
            
        return resolved
    
    def _resolve_relative_import(self, import_path: str, from_file: str) -> Optional[str]:
        """Resolve a relative import (./foo, ../bar)."""
        from_dir = Path(from_file).parent
        target = from_dir / import_path
        
        # Try exact path
        if target.exists() and target.is_file():
            return str(target.resolve())
        
        # Try with extensions
        for ext in ['.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs']:
            file_with_ext = target.with_suffix(ext)
            if file_with_ext.exists():
                return str(file_with_ext.resolve())
        
        # Try as directory with index file
        if target.is_dir():
            for index_name in ['index.js', 'index.jsx', 'index.ts', 'index.tsx', 'index.mjs']:
                index_file = target / index_name
                if index_file.exists():
                    return str(index_file.resolve())
        
        return None
    
    def _resolve_alias_import(self, import_path: str) -> Optional[str]:
        """Resolve alias imports like @/components."""
        # Common alias: @/ maps to src/
        if import_path.startswith('@/'):
            relative_path = import_path[2:]  # Remove '@/'
            src_path = self.project_root / 'src' / relative_path
            
            # Try with extensions
            for ext in ['', '.js', '.jsx', '.ts', '.tsx']:
                file_path = Path(str(src_path) + ext)
                if file_path.is_file():
                    return str(file_path.resolve())
            
            # Try as directory
            if src_path.is_dir():
                for index_name in ['index.js', 'index.jsx', 'index.ts', 'index.tsx', 'index.mjs']:
                    index_file = src_path / index_name
                    if index_file.exists():
                        return str(index_file.resolve())
        
        return None
    
    def _resolve_tsconfig_path(self, import_path: str) -> Optional[str]:
        """Resolve TypeScript path mapping from tsconfig.json."""
        for pattern, paths in self.tsconfig_paths.items():
            # Match pattern (e.g., "@components/*" matches "@components/Button")
            pattern_regex = pattern.replace('*', '(.*)')
            match = re.match(pattern_regex, import_path)
            
            if match:
                # Get the wildcard match
                wildcard = match.group(1) if match.groups() else ''
                
                # Try each path replacement
                for path_template in paths:
                    resolved_path = path_template.replace('*', wildcard)
                    full_path = self.project_root / resolved_path
                    
                    # Try with extensions
                    for ext in ['', '.js', '.jsx', '.ts', '.tsx']:
                        file_path = Path(str(full_path) + ext)
                        if file_path.exists():
                            return str(file_path.resolve())
                    
                    # Try as directory
                    if full_path.is_dir():
                        for index_name in ['index.ts', 'index.tsx', 'index.js', 'index.jsx']:
                            index_file = full_path / index_name
                            if index_file.exists():
                                return str(index_file.resolve())
        
        return None
    
    def _resolve_node_modules_import(self, import_path: str, from_file: str) -> Optional[str]:
        """Resolve import from node_modules."""
        # Start from the directory of from_file and walk up
        current_dir = Path(from_file).parent
        
        while current_dir >= self.project_root:
            node_modules = current_dir / 'node_modules'
            
            if node_modules.exists():
                # Try to resolve from this node_modules
                module_path = node_modules / import_path
                
                # Check if it's a file
                if module_path.exists() and module_path.is_file():
                    return str(module_path.resolve())
                
                # Check if it's a package directory
                if module_path.is_dir():
                    # Try to read package.json
                    package_json = module_path / 'package.json'
                    if package_json.exists():
                        try:
                            with open(package_json, 'r', encoding='utf-8') as f:
                                pkg_data = json.load(f)
                                
                            # Check for "module" field (ES6 modules)
                            if 'module' in pkg_data:
                                entry = module_path / pkg_data['module']
                                if entry.exists():
                                    return str(entry.resolve())
                            
                            # Check for "main" field
                            if 'main' in pkg_data:
                                entry = module_path / pkg_data['main']
                                if entry.exists():
                                    return str(entry.resolve())
                        except (json.JSONDecodeError, IOError):
                            pass
                    
                    # Fallback to index.js
                    for index_name in ['index.js', 'index.mjs', 'index.ts']:
                        index_file = module_path / index_name
                        if index_file.exists():
                            return str(index_file.resolve())
            
            # Move up one directory
            if current_dir == current_dir.parent:
                break
            current_dir = current_dir.parent
        
        return None
    
    def parse_imports(self, file_content: str, file_path: str) -> List[ImportStatement]:
        """
        Parse all import statements from a JavaScript/TypeScript file.
        
        Args:
            file_content: Content of the file
            file_path: Path to the file (for context)
            
        Returns:
            List of parsed import statements
        """
        imports = []
        
        # Regular import patterns
        patterns = [
            # Named imports: import { A, B } from 'module'
            r"import\s+\{([^}]+)\}\s+from\s+['\"]([^'\"]+)['\"]",
            
            # Default import: import X from 'module'
            r"import\s+(\w+)\s+from\s+['\"]([^'\"]+)['\"]",
            
            # Namespace import: import * as X from 'module'
            r"import\s+\*\s+as\s+(\w+)\s+from\s+['\"]([^'\"]+)['\"]",
            
            # Side-effect import: import 'module'
            r"import\s+['\"]([^'\"]+)['\"]",
            
            # TypeScript type-only: import type { A } from 'module'
            r"import\s+type\s+\{([^}]+)\}\s+from\s+['\"]([^'\"]+)['\"]",
        ]
        
        lines = file_content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Skip comments
            if line.strip().startswith('//') or line.strip().startswith('/*'):
                continue
            
            # Named imports
            match = re.search(patterns[0], line)
            if match:
                names = [n.strip() for n in match.group(1).split(',')]
                imports.append(ImportStatement(
                    module_path=match.group(2),
                    imported_names=names,
                    source_file=file_path,
                    line_number=line_num
                ))
                continue
            
            # Default import
            match = re.search(patterns[1], line)
            if match:
                imports.append(ImportStatement(
                    module_path=match.group(2),
                    default_import=match.group(1),
                    source_file=file_path,
                    line_number=line_num
                ))
                continue
            
            # Namespace import
            match = re.search(patterns[2], line)
            if match:
                imports.append(ImportStatement(
                    module_path=match.group(2),
                    namespace_import=match.group(1),
                    source_file=file_path,
                    line_number=line_num
                ))
                continue
            
            # Side-effect import
            match = re.search(patterns[3], line)
            if match:
                imports.append(ImportStatement(
                    module_path=match.group(1),
                    source_file=file_path,
                    line_number=line_num
                ))
                continue
            
            # Type-only import
            match = re.search(patterns[4], line)
            if match:
                names = [n.strip() for n in match.group(1).split(',')]
                imports.append(ImportStatement(
                    module_path=match.group(2),
                    imported_names=names,
                    is_type_only=True,
                    source_file=file_path,
                    line_number=line_num
                ))
                continue
            
            # Dynamic import
            if 'import(' in line:
                dynamic_match = re.search(r"import\(['\"]([^'\"]+)['\"]\)", line)
                if dynamic_match:
                    imports.append(ImportStatement(
                        module_path=dynamic_match.group(1),
                        is_dynamic=True,
                        source_file=file_path,
                        line_number=line_num
                    ))
        
        return imports
    
    def parse_exports(self, file_content: str) -> List[ExportStatement]:
        """
        Parse all export statements from a JavaScript/TypeScript file.
        
        Args:
            file_content: Content of the file
            
        Returns:
            List of parsed export statements
        """
        exports = []
        
        lines = file_content.split('\n')
        
        for line in lines:
            # Skip comments
            if line.strip().startswith('//') or line.strip().startswith('/*'):
                continue
            
            # Named export: export { A, B }
            match = re.search(r"export\s+\{([^}]+)\}", line)
            if match:
                names = [n.strip() for n in match.group(1).split(',')]
                
                # Check for re-export
                re_export_match = re.search(r"from\s+['\"]([^'\"]+)['\"]", line)
                re_export_from = re_export_match.group(1) if re_export_match else None
                
                exports.append(ExportStatement(
                    exported_names=names,
                    re_export_from=re_export_from
                ))
                continue
            
            # Default export
            if 'export default' in line:
                exports.append(ExportStatement(
                    default_export='default'
                ))
                continue
            
            # Named declaration export: export const X = ...
            declaration_match = re.search(r"export\s+(?:const|let|var|function|class)\s+(\w+)", line)
            if declaration_match:
                exports.append(ExportStatement(
                    exported_names=[declaration_match.group(1)]
                ))
        
        return exports
