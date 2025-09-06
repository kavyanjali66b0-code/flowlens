#!/usr/bin/env python3
"""
Test to verify tree-sitter language loading works.
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_tree_sitter_imports():
    """Test that we can import tree-sitter modules."""
    try:
        import tree_sitter_javascript
        print("✓ Successfully imported tree_sitter_javascript")
        return True
    except ImportError as e:
        print(f"✗ Failed to import tree_sitter_javascript: {e}")
        return False

def test_tree_sitter_language():
    """Test that we can get the language object."""
    try:
        import tree_sitter_javascript
        lang = tree_sitter_javascript.LANGUAGE
        print(f"✓ Successfully got language object: {type(lang)}")
        return True
    except Exception as e:
        print(f"✗ Failed to get language object: {e}")
        return False

def test_plugin_language_loading():
    """Test that the plugin can load the language."""
    try:
        from analyzer.plugins.languages.javascript_plugin import JavaScriptPlugin
        import tempfile
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as temp_dir:
            plugin = JavaScriptPlugin(temp_dir)
            lang = plugin._get_ts_language()
            if lang:
                print(f"✓ Plugin successfully loaded language: {type(lang)}")
                return True
            else:
                print("✗ Plugin failed to load language")
                return False
    except Exception as e:
        print(f"✗ Plugin language loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Testing tree-sitter language loading...")
    
    tests = [
        test_tree_sitter_imports,
        test_tree_sitter_language,
        test_plugin_language_loading
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Tests passed: {passed}/{total}")
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
