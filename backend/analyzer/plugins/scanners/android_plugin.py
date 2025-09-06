"""
Android project scanner plugin.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any

from ..base import ScannerPlugin
from ...models import ProjectType, ConfigFile


class AndroidPlugin(ScannerPlugin):
    """Plugin for detecting Android projects."""
    
    def is_applicable(self, files: List[Path]) -> bool:
        """Check if this is an Android project."""
        file_names = [f.name for f in files]
        
        # Check for Android-specific files
        android_files = ['AndroidManifest.xml', 'build.gradle', 'gradle.properties']
        if any(android_file in file_names for android_file in android_files):
            return True
        
        # Check for Android directory structure
        android_dirs = ['app', 'src', 'res']
        if all((self.project_path / dir_name).exists() for dir_name in android_dirs):
            return True
        
        return False
    
    def get_project_type(self) -> ProjectType:
        return ProjectType.ANDROID
    
    def get_config_files(self) -> List[Dict[str, Any]]:
        """Return Android configuration files."""
        config_files = []
        
        # Check for AndroidManifest.xml
        manifest_path = self.project_path / 'app' / 'src' / 'main' / 'AndroidManifest.xml'
        if manifest_path.exists():
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                config_files.append({
                    'type': 'AndroidManifest.xml',
                    'path': 'app/src/main/AndroidManifest.xml',
                    'content': content
                })
            except (OSError, UnicodeDecodeError):
                pass
        
        # Check for build.gradle (app level)
        app_gradle_path = self.project_path / 'app' / 'build.gradle'
        if app_gradle_path.exists():
            try:
                with open(app_gradle_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                config_files.append({
                    'type': 'build.gradle',
                    'path': 'app/build.gradle',
                    'content': content
                })
            except (OSError, UnicodeDecodeError):
                pass
        
        # Check for build.gradle (project level)
        project_gradle_path = self.project_path / 'build.gradle'
        if project_gradle_path.exists():
            try:
                with open(project_gradle_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                config_files.append({
                    'type': 'build.gradle',
                    'path': 'build.gradle',
                    'content': content
                })
            except (OSError, UnicodeDecodeError):
                pass
        
        # Check for gradle.properties
        gradle_props_path = self.project_path / 'gradle.properties'
        if gradle_props_path.exists():
            try:
                with open(gradle_props_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                config_files.append({
                    'type': 'gradle.properties',
                    'path': 'gradle.properties',
                    'content': content
                })
            except (OSError, UnicodeDecodeError):
                pass
        
        return config_files
