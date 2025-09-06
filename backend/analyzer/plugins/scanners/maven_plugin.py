"""
Maven project scanner plugin.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any

from ..base import ScannerPlugin
from ...models import ProjectType, ConfigFile


class MavenPlugin(ScannerPlugin):
    """Plugin for detecting Maven projects."""
    
    def is_applicable(self, files: List[Path]) -> bool:
        """Check if this is a Maven project."""
        file_names = [f.name for f in files]
        return 'pom.xml' in file_names
    
    def get_project_type(self) -> ProjectType:
        return ProjectType.MAVEN_JAVA
    
    def get_config_files(self) -> List[Dict[str, Any]]:
        """Return Maven configuration files."""
        config_files = []
        
        # Check for pom.xml
        pom_path = self.project_path / 'pom.xml'
        if pom_path.exists():
            try:
                with open(pom_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                config_files.append({
                    'type': 'pom.xml',
                    'path': 'pom.xml',
                    'content': content
                })
            except (OSError, UnicodeDecodeError):
                pass
        
        return config_files
