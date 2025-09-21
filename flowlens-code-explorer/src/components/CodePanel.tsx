import React, { useEffect, useState, useMemo } from 'react';
import Editor from '@monaco-editor/react';
import { Copy, ExternalLink, Sparkles, Code, Eye, Navigation } from 'lucide-react';
import { useFlowLensStore } from '../store/useFlowLensStore';
import { toast } from 'react-toastify';

const CodePanel: React.FC = () => {
  const {
    graphNodes,
    darkMode,
    inspectedNodeId,
    getNodeById
  } = useFlowLensStore();

  console.log('CodePanel render - inspectedNodeId:', inspectedNodeId);

  // Get the currently inspected node
  const inspectedNode = useMemo(() => {
    if (!inspectedNodeId) return null;
    const node = getNodeById(inspectedNodeId);
    console.log('Found inspected node:', node);
    return node;
  }, [inspectedNodeId, getNodeById]);

  // Generate placeholder code for any node
  const generatePlaceholderCode = (node: any): string => {
    console.log('Generating placeholder code for:', node);
    
    const parentFile = node.data.fileId ? getNodeById(node.data.fileId) : null;
    const parentClass = node.data.classId ? getNodeById(node.data.classId) : null;

    switch (node.type) {
      case 'method': {
        return `// Method: ${node.data.label}
// File: ${parentFile?.data.label || 'Unknown'}
// Class: ${parentClass?.data.label || 'Unknown'}
// Node ID: ${node.id}

/**
 * ${node.data.description || 'Method implementation'}
 */
function ${node.data.label}() {
  // This is a placeholder for the actual method code
  // In a full implementation, this would show the real method body
  
  console.log('Executing method: ${node.data.label}');
  
  // TODO: Load actual method implementation from repository
  throw new Error('Method implementation not yet loaded');
}

export { ${node.data.label} };`;
      }

      case 'class': {
        const relatedMethods = graphNodes.filter(n => n.type === 'method' && n.data.classId === node.id);
        
        return `// Class: ${node.data.label}
// File: ${parentFile?.data.label || 'Unknown'}
// Methods: ${relatedMethods.length}
// Node ID: ${node.id}

/**
 * ${node.data.description || 'Class implementation'}
 */
class ${node.data.label} {
${relatedMethods.map(method => `  
  /**
   * ${method.data.description || 'Method'}
   */
  ${method.data.label}() {
    // TODO: Load actual method implementation
    return null;
  }`).join('\n')}

  constructor() {
    // TODO: Load actual constructor implementation
    console.log('Creating instance of ${node.data.label}');
  }
}

export default ${node.data.label};`;
      }

      case 'file': {
        const relatedClasses = graphNodes.filter(n => n.type === 'class' && n.data.fileId === node.id);
        const relatedMethods = graphNodes.filter(n => n.type === 'method' && n.data.fileId === node.id);

        return `// File: ${node.data.label}
// Classes: ${relatedClasses.length}
// Methods: ${relatedMethods.length}
// Node ID: ${node.id}

/**
 * ${node.data.description || 'File contents'}
 * 
 * This file contains the following components:
 */

${relatedClasses.map(cls => `// class ${cls.data.label} { ... }`).join('\n')}
${relatedMethods.map(method => `// function ${method.data.label}() { ... }`).join('\n')}

/**
 * File structure and exports
 * In a full implementation, this would show the actual file content
 */

// TODO: Load actual file content from repository
export {};`;
      }

      default:
        return `// Unknown node type: ${node.type}
// Label: ${node.data.label}
// Description: ${node.data.description}
// Node ID: ${node.id}

console.log('Node details:', ${JSON.stringify(node.data, null, 2)});`;
    }
  };

  // Get display content based on inspected node
  const displayContent = useMemo(() => {
    if (!inspectedNode) {
      console.log('No inspected node, returning null');
      return null;
    }

    console.log('Creating display content for:', inspectedNode);

    const parentFile = inspectedNode.data.fileId ? getNodeById(inspectedNode.data.fileId) : null;
    const parentClass = inspectedNode.data.classId ? getNodeById(inspectedNode.data.classId) : null;

    const getLanguage = (fileName?: string) => {
      if (!fileName) return 'typescript';
      const ext = fileName.split('.').pop()?.toLowerCase();
      switch (ext) {
        case 'js': return 'javascript';
        case 'ts': return 'typescript';
        case 'tsx': return 'typescript';
        case 'jsx': return 'javascript';
        case 'py': return 'python';
        case 'java': return 'java';
        case 'cpp': case 'c': return 'cpp';
        case 'css': return 'css';
        case 'html': return 'html';
        case 'json': return 'json';
        default: return 'typescript';
      }
    };

    const baseContent = {
      language: getLanguage(parentFile?.data.label),
      filePath: inspectedNode.data.description?.split('•')[1]?.trim(),
      code: generatePlaceholderCode(inspectedNode),
    };

    switch (inspectedNode.type) {
      case 'method':
        return {
          ...baseContent,
          title: `Method: ${inspectedNode.data.label}`,
          subtitle: `in ${parentClass?.data.label || 'Unknown Class'} (${parentFile?.data.label || 'Unknown File'})`,
          description: inspectedNode.data.description || 'Method implementation',
        };

      case 'class':
        const relatedMethods = graphNodes.filter(n => n.type === 'method' && n.data.classId === inspectedNode.id);
        return {
          ...baseContent,
          title: `Class: ${inspectedNode.data.label}`,
          subtitle: `in ${parentFile?.data.label || 'Unknown File'}`,
          description: `Class with ${relatedMethods.length} methods`,
        };

      case 'file':
        const fileClasses = graphNodes.filter(n => n.type === 'class' && n.data.fileId === inspectedNode.id);
        const fileMethods = graphNodes.filter(n => n.type === 'method' && n.data.fileId === inspectedNode.id);
        return {
          ...baseContent,
          title: inspectedNode.data.label || 'File',
          subtitle: inspectedNode.data.description || 'File contents',
          description: `${fileClasses.length} classes, ${fileMethods.length} methods`,
        };

      default:
        return {
          ...baseContent,
          title: `Unknown: ${inspectedNode.data.label}`,
          subtitle: inspectedNode.data.description || 'Unknown node type',
          description: `Type: ${inspectedNode.type}`,
        };
    }
  }, [inspectedNode, graphNodes, getNodeById]);

  console.log('Display content:', displayContent);

  const copyToClipboard = async () => {
    if (displayContent?.code) {
      try {
        await navigator.clipboard.writeText(displayContent.code);
        toast.success('Code copied to clipboard!');
      } catch (error) {
        toast.error('Failed to copy code');
      }
    }
  };

  const generateSummary = () => {
    if (inspectedNode) {
      const summaries = {
        method: `This method "${inspectedNode.data.label}" is part of the codebase structure.`,
        class: `This class "${inspectedNode.data.label}" contains multiple methods and represents a component.`,
        file: `This file "${inspectedNode.data.label}" contains classes and functions.`
      };
      
      toast.info(summaries[inspectedNode.type as keyof typeof summaries] || 'Code analysis complete.');
    }
  };

  const openOnGitHub = () => {
    toast.info('GitHub integration coming soon!');
  };

  // Show empty state when no node is inspected
  if (!displayContent) {
    return (
      <div className="w-96 flowlens-panel m-4 p-6 rounded-lg">
        <div className="h-full flex flex-col items-center justify-center text-center space-y-4">
          <Eye className="h-16 w-16 text-muted-foreground opacity-50" />
          <div>
            <h3 className="text-lg font-medium text-foreground mb-2">Code Inspector</h3>
            <p className="text-sm text-muted-foreground mb-4">
              Hover over any node in the diagram to inspect its code
            </p>
            <div className="text-xs text-gray-400 space-y-1">
              <p>• Hover to inspect code instantly</p>
              <p>• Double-click to navigate into components</p>
              <p>• Click empty space to clear inspection</p>
            </div>
            {inspectedNodeId && (
              <div className="mt-4 p-2 bg-red-50 border border-red-200 rounded text-xs text-red-700">
                Debug: inspectedNodeId = {inspectedNodeId} but node not found
              </div>
            )}
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
            <div className="flex items-center space-x-2 mb-1">
              <Eye className="h-4 w-4 text-green-500" />
              <h3 className="text-lg font-semibold text-foreground truncate">
                {displayContent.title}
              </h3>
            </div>
            <p className="text-sm text-muted-foreground truncate">
              {displayContent.subtitle}
            </p>
          </div>
          <div className="flex items-center space-x-1">
            <Navigation className="h-4 w-4 text-gray-400" />
          </div>
        </div>
        
        <p className="text-sm text-muted-foreground mb-3">
          {displayContent.description}
        </p>

        {/* Debug info */}
        <div className="bg-blue-50 border border-blue-200 rounded p-2 mb-3 text-xs text-blue-700">
          Inspecting: {inspectedNode?.id} | Type: {inspectedNode?.type}
        </div>

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
            title="AI analysis"
          >
            <Sparkles className="h-3 w-3 mr-1" />
            Analyze
          </button>
        </div>
      </div>

      {/* Code editor */}
      <div className="flex-1 monaco-editor-container">
        <Editor
          height="100%"
          language={displayContent.language}
          value={displayContent.code}
          theme={darkMode ? 'vs-dark' : 'vs-light'}
          options={{
            readOnly: true,
            minimap: { enabled: true },
            scrollBeyondLastLine: false,
            fontSize: 12,
            lineNumbers: 'on',
            wordWrap: 'on',
            automaticLayout: true,
            contextmenu: false,
            lineDecorationsWidth: 0,
            lineNumbersMinChars: 3,
            glyphMargin: false,
            folding: true,
            selectOnLineNumbers: false,
            selectionHighlight: false,
            cursorStyle: 'line-thin',
            renderLineHighlight: 'none',
            bracketPairColorization: { enabled: true },
            guides: { bracketPairs: true },
          }}
        />
      </div>

      {/* Footer */}
      <div className="px-4 py-2 border-t border-border bg-gray-50 text-xs text-gray-500">
        <div className="flex justify-between items-center">
          <span>
            {displayContent.filePath || 'Generated structure'}
          </span>
          <span className="flex items-center space-x-2">
            <span>{displayContent.language}</span>
            <span className="text-green-600">Inspecting</span>
          </span>
        </div>
      </div>
    </div>
  );
};

export default CodePanel;