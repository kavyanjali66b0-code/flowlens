"""
Configuration file readers for build tools.

This module provides utilities to read configuration from various
build tools and frameworks to enhance code analysis.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, List


class TsConfigReader:
    """Read TypeScript configuration for path aliases."""
    
    @staticmethod
    def read_path_aliases(project_path: Path) -> Dict[str, str]:
        """
        Extract path aliases from tsconfig.json.
        
        Args:
            project_path: Root path of the project
            
        Returns:
            Dict mapping alias prefix to actual path
            e.g., {'@components': '/abs/path/to/src/components'}
        """
        tsconfig_path = project_path / 'tsconfig.json'
        if not tsconfig_path.exists():
            logging.debug(f"No tsconfig.json found at {tsconfig_path}")
            return {}
        
        try:
            with open(tsconfig_path, 'r', encoding='utf-8') as f:
                # Handle comments in JSON (common in tsconfig.json)
                content = f.read()
                # Simple comment removal (not perfect but works for most cases)
                lines = []
                for line in content.split('\n'):
                    # Remove // comments
                    if '//' in line:
                        line = line[:line.index('//')]
                    lines.append(line)
                content = '\n'.join(lines)
                
                config = json.loads(content)
            
            compiler_options = config.get('compilerOptions', {})
            paths = compiler_options.get('paths', {})
            base_url = compiler_options.get('baseUrl', '.')
            
            # Normalize paths
            aliases = {}
            for alias, targets in paths.items():
                if targets and len(targets) > 0:
                    # Remove trailing /* from alias and target
                    clean_alias = alias.rstrip('/*')
                    clean_target = targets[0].rstrip('/*')
                    
                    # Resolve relative to baseUrl
                    base_path = project_path / base_url
                    full_target = (base_path / clean_target).resolve()
                    
                    aliases[clean_alias] = str(full_target)
            
            if aliases:
                logging.info(f"Found {len(aliases)} TypeScript path aliases: {list(aliases.keys())}")
            
            return aliases
            
        except json.JSONDecodeError as e:
            logging.warning(f"Failed to parse tsconfig.json: {e}")
            return {}
        except Exception as e:
            logging.warning(f"Failed to read tsconfig.json: {e}")
            return {}


class ViteConfigReader:
    """Read Vite configuration for aliases."""
    
    @staticmethod
    def read_aliases(project_path: Path) -> Dict[str, str]:
        """
        Extract aliases from vite.config.ts/js.
        
        Note: Vite configs are JavaScript/TypeScript, so this is limited.
        We look for common patterns but can't execute the config.
        
        Args:
            project_path: Root path of the project
            
        Returns:
            Dict mapping alias to path
        """
        # Check for vite.config.ts and vite.config.js
        for config_name in ['vite.config.ts', 'vite.config.js']:
            config_path = project_path / config_name
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Look for resolve.alias patterns
                    # This is a simple regex-based approach - not perfect
                    # Example: alias: { '@': path.resolve(__dirname, './src') }
                    
                    # For now, just log that we found it
                    # Full implementation would need a JS parser
                    logging.debug(f"Found {config_name} but alias extraction not yet implemented")
                    
                except Exception as e:
                    logging.debug(f"Error reading {config_name}: {e}")
        
        return {}


class PackageJsonReader:
    """Read package.json for project information."""
    
    @staticmethod
    def read_package_json(project_path: Path) -> Optional[Dict]:
        """
        Read package.json file.
        
        Args:
            project_path: Root path of the project
            
        Returns:
            Parsed package.json content or None
        """
        package_json_path = project_path / 'package.json'
        if not package_json_path.exists():
            return None
        
        try:
            with open(package_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to read package.json: {e}")
            return None
    
    @staticmethod
    def get_dependencies(project_path: Path) -> Dict[str, List[str]]:
        """
        Extract all dependencies from package.json.
        
        Args:
            project_path: Root path of the project
            
        Returns:
            Dict with 'dependencies' and 'devDependencies' lists
        """
        pkg = PackageJsonReader.read_package_json(project_path)
        if not pkg:
            return {'dependencies': [], 'devDependencies': []}
        
        return {
            'dependencies': list(pkg.get('dependencies', {}).keys()),
            'devDependencies': list(pkg.get('devDependencies', {}).keys())
        }
