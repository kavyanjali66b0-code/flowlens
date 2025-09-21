import React, { useEffect, useRef, useMemo } from 'react';
import { Search, Files, Box, Zap, ChevronRight, ArrowLeft } from 'lucide-react';
import { useFlowLensStore } from '../store/useFlowLensStore';

const Overview: React.FC = () => {
  const {
    projectMeta,
    selectedFileId,
    selectedClassId,
    selectedMethodId,
    searchQuery,
    setSearchQuery,
    filterType,
    setFilterType,
    files,
    graphNodes,
    viewLevel,
    setSelectedFile,
    setSelectedClass,
    setSelectedMethod,
    clearSelection
  } = useFlowLensStore();

  const searchInputRef = useRef<HTMLInputElement>(null);

  // Handle global keyboard shortcut for search
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === '/' && !e.ctrlKey && !e.metaKey) {
        e.preventDefault();
        searchInputRef.current?.focus();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Memoized node lists to prevent recalculation
  const nodeLists = useMemo(() => {
    const fileNodes = graphNodes
      .filter(node => node.type === 'file')
      .map(node => ({
        id: node.id,
        name: node.data.label,
        path: node.data.description || node.data.label
      }))
      .sort((a, b) => a.name.localeCompare(b.name));

    const classNodes = graphNodes
      .filter(node => node.type === 'class')
      .map(node => ({
        id: node.id,
        name: node.data.label,
        path: node.data.description || node.data.label,
        fileId: node.data.fileId
      }))
      .sort((a, b) => a.name.localeCompare(b.name));

    const methodNodes = graphNodes
      .filter(node => node.type === 'method')
      .map(node => ({
        id: node.id,
        name: node.data.label,
        path: node.data.description || node.data.label,
        classId: node.data.classId,
        fileId: node.data.fileId
      }))
      .sort((a, b) => a.name.localeCompare(b.name));

    return { fileNodes, classNodes, methodNodes };
  }, [graphNodes]);

  // Calculate contextual statistics based on current selection
  const getDisplayData = useMemo(() => {
    if (selectedMethodId) {
      // Method view - show method details
      const method = nodeLists.methodNodes.find(m => m.id === selectedMethodId);
      const parentClass = nodeLists.classNodes.find(c => c.id === method?.classId);
      const parentFile = nodeLists.fileNodes.find(f => f.id === method?.fileId);
      
      return {
        title: method?.name || 'Method',
        subtitle: `in ${parentClass?.name || 'Unknown Class'} (${parentFile?.name || 'Unknown File'})`,
        stats: [
          { 
            label: 'Method Details', 
            count: 1, 
            icon: Zap, 
            type: 'methods' as const 
          }
        ]
      };
    }

    if (selectedClassId) {
      // Class view - show methods in this class
      const selectedClass = nodeLists.classNodes.find(c => c.id === selectedClassId);
      const classMethodsCount = nodeLists.methodNodes.filter(m => m.classId === selectedClassId).length;
      const parentFile = nodeLists.fileNodes.find(f => f.id === selectedClass?.fileId);
      
      return {
        title: selectedClass?.name || 'Class',
        subtitle: `in ${parentFile?.name || 'Unknown File'}`,
        stats: [
          { 
            label: 'Methods', 
            count: classMethodsCount, 
            icon: Zap, 
            type: 'methods' as const 
          }
        ]
      };
    }

    if (selectedFileId) {
      // File view - show classes and methods in this file
      const selectedFile = nodeLists.fileNodes.find(f => f.id === selectedFileId);
      const fileClassesCount = nodeLists.classNodes.filter(c => c.fileId === selectedFileId).length;
      const fileMethodsCount = nodeLists.methodNodes.filter(m => m.fileId === selectedFileId).length;
      
      return {
        title: selectedFile?.name || 'File',
        subtitle: selectedFile?.path || '',
        stats: [
          { 
            label: 'Classes', 
            count: fileClassesCount, 
            icon: Box, 
            type: 'classes' as const 
          },
          { 
            label: 'Methods', 
            count: fileMethodsCount, 
            icon: Zap, 
            type: 'methods' as const 
          }
        ]
      };
    }

    // Repository overview - show total counts
    return {
      title: projectMeta?.name || 'Repository',
      subtitle: 'Project Overview',
      stats: [
        { 
          label: 'Files', 
          count: nodeLists.fileNodes.length, 
          icon: Files, 
          type: 'files' as const 
        },
        { 
          label: 'Classes', 
          count: nodeLists.classNodes.length, 
          icon: Box, 
          type: 'classes' as const 
        },
        { 
          label: 'Methods', 
          count: nodeLists.methodNodes.length, 
          icon: Zap, 
          type: 'methods' as const 
        }
      ]
    };
  }, [selectedFileId, selectedClassId, selectedMethodId, projectMeta, nodeLists]);

  // Get filtered lists based on current context
  const getFilteredLists = useMemo(() => {
    let filesToShow = nodeLists.fileNodes;
    let classesToShow = nodeLists.classNodes;
    let methodsToShow = nodeLists.methodNodes;

    // Context-based filtering
    if (selectedFileId) {
      classesToShow = nodeLists.classNodes.filter(c => c.fileId === selectedFileId);
      methodsToShow = nodeLists.methodNodes.filter(m => m.fileId === selectedFileId);
    }
    
    if (selectedClassId) {
      methodsToShow = nodeLists.methodNodes.filter(m => m.classId === selectedClassId);
    }

    return {
      files: filesToShow.slice(0, 15),
      classes: classesToShow.slice(0, 15),
      methods: methodsToShow.slice(0, 15),
      totalCounts: {
        files: filesToShow.length,
        classes: classesToShow.length,
        methods: methodsToShow.length
      }
    };
  }, [nodeLists, selectedFileId, selectedClassId]);

  if (!projectMeta) {
    return (
      <div className="w-80 bg-white m-4 p-6 rounded-lg border">
        <div className="text-center text-gray-500">
          <Files className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p>No project loaded</p>
          <p className="text-sm mt-2">Upload a project or connect a GitHub repository to get started</p>
        </div>
      </div>
    );
  }

  const displayData = getDisplayData;
  const filteredLists = getFilteredLists;

  return (
    <div className="w-80 bg-white m-4 p-6 rounded-lg border space-y-6">
      {/* Header with navigation */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-lg font-semibold text-gray-900 truncate">{displayData.title}</h2>
          {(selectedFileId || selectedClassId || selectedMethodId) && (
            <button
              onClick={clearSelection}
              className="flex items-center text-sm text-gray-500 hover:text-gray-700"
              title="Back to overview"
            >
              <ArrowLeft className="h-4 w-4" />
            </button>
          )}
        </div>
        <p className="text-sm text-gray-500 truncate">{displayData.subtitle}</p>
        
        {/* Breadcrumb */}
        <div className="text-xs text-gray-400 mt-1">
          View: {viewLevel} | Context: {
            selectedMethodId ? 'Method' :
            selectedClassId ? 'Class' :
            selectedFileId ? 'File' : 'Repository'
          }
        </div>
      </div>

      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
        <input
          ref={searchInputRef}
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search nodes... (Press '/' to focus)"
          className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
        />
      </div>

      {/* Stats Cards */}
      <div className="space-y-3">
        <h3 className="text-sm font-medium text-gray-500 uppercase tracking-wide">
          Statistics
        </h3>
        <div className="space-y-2">
          {displayData.stats.map((stat) => (
            <button
              key={stat.type}
              onClick={() => setFilterType(stat.type)}
              className={`w-full flex items-center justify-between p-3 rounded-md transition-colors ${
                filterType === stat.type
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-50 hover:bg-gray-100 text-gray-700'
              }`}
            >
              <div className="flex items-center space-x-3">
                <stat.icon className="h-5 w-5" />
                <div className="text-left">
                  <p className="font-medium">{stat.label}</p>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <span className="text-lg font-bold">{stat.count}</span>
                <ChevronRight className="h-4 w-4" />
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Context-aware Browser */}
      <div className="space-y-3">
        <h3 className="text-sm font-medium text-gray-500 uppercase tracking-wide">
          {filterType === 'files' ? 'Files' : filterType === 'classes' ? 'Classes' : 'Methods'}
        </h3>
        
        <div className="max-h-60 overflow-y-auto space-y-1">
          {filterType === 'files' && filteredLists.files.map((file) => (
            <button
              key={file.id}
              onClick={() => setSelectedFile(file.id)}
              className={`w-full text-left text-sm p-2 rounded hover:bg-gray-100 truncate ${
                selectedFileId === file.id ? 'bg-blue-50 text-blue-700' : 'text-gray-700'
              }`}
              title={file.path}
            >
              <Files className="h-3 w-3 inline mr-2" />
              {file.name}
            </button>
          ))}
          
          {filterType === 'classes' && filteredLists.classes.map((cls) => (
            <button
              key={cls.id}
              onClick={() => setSelectedClass(cls.id)}
              className={`w-full text-left text-sm p-2 rounded hover:bg-gray-100 truncate ${
                selectedClassId === cls.id ? 'bg-blue-50 text-blue-700' : 'text-gray-700'
              }`}
              title={cls.path}
            >
              <Box className="h-3 w-3 inline mr-2" />
              {cls.name}
            </button>
          ))}
          
          {filterType === 'methods' && filteredLists.methods.map((method) => (
            <button
              key={method.id}
              onClick={() => setSelectedMethod(method.id)}
              className={`w-full text-left text-sm p-2 rounded hover:bg-gray-100 truncate ${
                selectedMethodId === method.id ? 'bg-blue-50 text-blue-700' : 'text-gray-700'
              }`}
              title={method.path}
            >
              <Zap className="h-3 w-3 inline mr-2" />
              {method.name}
            </button>
          ))}
        </div>
        
        {/* Show count if truncated */}
        {filterType === 'files' && filteredLists.totalCounts.files > 15 && (
          <div className="text-xs text-gray-400 p-2">
            ... and {filteredLists.totalCounts.files - 15} more files
          </div>
        )}
        {filterType === 'classes' && filteredLists.totalCounts.classes > 15 && (
          <div className="text-xs text-gray-400 p-2">
            ... and {filteredLists.totalCounts.classes - 15} more classes
          </div>
        )}
        {filterType === 'methods' && filteredLists.totalCounts.methods > 15 && (
          <div className="text-xs text-gray-400 p-2">
            ... and {filteredLists.totalCounts.methods - 15} more methods
          </div>
        )}
      </div>
    </div>
  );
};

export default Overview;