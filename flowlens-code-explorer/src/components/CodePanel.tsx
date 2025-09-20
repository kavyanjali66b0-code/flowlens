import React, { useEffect, useState } from 'react';
import Editor from '@monaco-editor/react';
import { Copy, ExternalLink, Sparkles, Code } from 'lucide-react';
import { useFlowLensStore } from '../store/useFlowLensStore';
import { toast } from 'react-toastify';

const CodePanel: React.FC = () => {
  const {
    files,
    selectedFileId,
    selectedClassId,
    selectedMethodId,
    darkMode
  } = useFlowLensStore();

  const [editorReady, setEditorReady] = useState(false);

  const getDisplayContent = () => {
    if (selectedMethodId) {
      // Show specific method
      const file = files.find(f => f.classes.some(c => c.methods.some(m => m.id === selectedMethodId)));
      const selectedClass = file?.classes.find(c => c.methods.some(m => m.id === selectedMethodId));
      const method = selectedClass?.methods.find(m => m.id === selectedMethodId);
      
      if (method) {
        return {
          title: `Method: ${method.name}`,
          subtitle: `in ${selectedClass?.name} (${file?.path.split('/').pop()})`,
          code: method.code,
          description: method.description,
          language: 'javascript',
          startLine: method.startLine,
          endLine: method.endLine,
          githubLink: `#L${method.startLine}-L${method.endLine}`
        };
      }
    }

    if (selectedClassId) {
      // Show specific class
      const file = files.find(f => f.classes.some(c => c.id === selectedClassId));
      const selectedClass = file?.classes.find(c => c.id === selectedClassId);
      
      if (selectedClass && file) {
        // Extract class code from file content
        const lines = file.content?.split('\n') || [];
        const classCode = lines.slice(selectedClass.startLine - 1, selectedClass.endLine).join('\n');
        
        return {
          title: `Class: ${selectedClass.name}`,
          subtitle: file.path,
          code: classCode || `// Class ${selectedClass.name}\n// Code content would be extracted from the file`,
          description: `Class with ${selectedClass.methods.length} methods`,
          language: 'javascript',
          startLine: selectedClass.startLine,
          endLine: selectedClass.endLine,
          githubLink: `#L${selectedClass.startLine}-L${selectedClass.endLine}`
        };
      }
    }

    if (selectedFileId) {
      // Show full file
      const file = files.find(f => f.id === selectedFileId);
      
      if (file) {
        return {
          title: file.path.split('/').pop() || 'File',
          subtitle: file.path,
          code: file.content || '// File content would be loaded here',
          description: `${file.classes.length} classes, ${file.classes.reduce((sum, c) => sum + c.methods.length, 0)} methods`,
          language: 'javascript',
          startLine: 1,
          endLine: file.endLine,
          githubLink: ''
        };
      }
    }

    return null;
  };

  const content = getDisplayContent();

  const copyToClipboard = async () => {
    if (content?.code) {
      try {
        await navigator.clipboard.writeText(content.code);
        toast.success('Code copied to clipboard!');
      } catch (error) {
        toast.error('Failed to copy code');
      }
    }
  };

  const generateSummary = () => {
    // Mock functionality for now
    toast.info('AI summary generation coming soon!');
  };

  const openOnGitHub = () => {
    // Mock functionality for now
    toast.info('GitHub integration coming soon!');
  };

  // Handle Monaco editor highlighting for methods
  useEffect(() => {
    if (editorReady && selectedMethodId && content) {
      // This would typically highlight specific lines
      // Monaco editor line highlighting would be implemented here
    }
  }, [editorReady, selectedMethodId, content]);

  if (!content) {
    return (
      <div className="w-96 flowlens-panel m-4 p-6 rounded-lg">
        <div className="h-full flex flex-col items-center justify-center text-center space-y-4">
          <Code className="h-16 w-16 text-muted-foreground opacity-50" />
          <div>
            <h3 className="text-lg font-medium text-foreground mb-2">No component selected</h3>
            <p className="text-sm text-muted-foreground">
              Click a node in the diagram to view its code
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="w-96 flowlens-panel m-4 rounded-lg flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-border">
        <div className="flex items-start justify-between mb-3">
          <div className="flex-1 min-w-0">
            <h3 className="text-lg font-semibold text-foreground truncate">{content.title}</h3>
            <p className="text-sm text-muted-foreground truncate">{content.subtitle}</p>
          </div>
        </div>
        
        {content.description && (
          <p className="text-sm text-muted-foreground mb-3">{content.description}</p>
        )}

        {/* Action buttons */}
        <div className="flex items-center space-x-2">
          <button
            onClick={copyToClipboard}
            className="flowlens-button-ghost px-3 py-1.5 text-xs"
            title="Copy code"
          >
            <Copy className="h-3 w-3 mr-1" />
            Copy
          </button>
          
          <button
            onClick={openOnGitHub}
            className="flowlens-button-ghost px-3 py-1.5 text-xs"
            title="Open on GitHub"
          >
            <ExternalLink className="h-3 w-3 mr-1" />
            GitHub
          </button>
          
          <button
            onClick={generateSummary}
            className="flowlens-button-ghost px-3 py-1.5 text-xs"
            title="Generate AI summary"
          >
            <Sparkles className="h-3 w-3 mr-1" />
            Summary
          </button>
        </div>
      </div>

      {/* Code editor */}
      <div className="flex-1 monaco-editor-container">
        <Editor
          height="100%"
          language={content.language}
          value={content.code}
          theme={darkMode ? 'vs-dark' : 'vs-light'}
          options={{
            readOnly: true,
            minimap: { enabled: false },
            scrollBeyondLastLine: false,
            fontSize: 12,
            lineNumbers: 'on',
            wordWrap: 'on',
            automaticLayout: true,
            contextmenu: false,
            lineDecorationsWidth: 0,
            lineNumbersMinChars: 3,
            glyphMargin: false,
            folding: false,
            selectOnLineNumbers: false,
            selectionHighlight: false,
            cursorStyle: 'line-thin',
            renderLineHighlight: 'none',
          }}
          onMount={() => setEditorReady(true)}
        />
      </div>
    </div>
  );
};

export default CodePanel;