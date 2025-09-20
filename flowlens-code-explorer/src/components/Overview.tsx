import React, { useEffect, useRef } from 'react';
import { Search, Files, Box, Zap, ChevronRight } from 'lucide-react';
import { useFlowLensStore } from '../store/useFlowLensStore';

const Overview: React.FC = () => {
  const {
    projectMeta,
    selectedFileId,
    selectedClassId,
    searchQuery,
    setSearchQuery,
    filterType,
    setFilterType,
    files
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

  const getDisplayData = () => {
    if (selectedClassId) {
      // Show class details
      const file = files.find(f => f.classes.some(c => c.id === selectedClassId));
      const selectedClass = file?.classes.find(c => c.id === selectedClassId);
      return {
        title: selectedClass?.name || 'Class',
        subtitle: `in ${file?.path.split('/').pop()}`,
        stats: [
          { label: 'Methods', count: selectedClass?.methods.length || 0, icon: Zap, type: 'methods' as const }
        ]
      };
    }

    if (selectedFileId) {
      // Show file details
      const file = files.find(f => f.id === selectedFileId);
      return {
        title: file?.path.split('/').pop() || 'File',
        subtitle: file?.path || '',
        stats: [
          { label: 'Classes', count: file?.classes.length || 0, icon: Box, type: 'classes' as const },
          { label: 'Methods', count: file?.classes.reduce((sum, c) => sum + c.methods.length, 0) || 0, icon: Zap, type: 'methods' as const }
        ]
      };
    }

    // Show repo overview
    return {
      title: projectMeta?.name || 'Repository',
      subtitle: 'Project Overview',
      stats: [
        { label: 'Files', count: projectMeta?.fileCount || 0, icon: Files, type: 'files' as const },
        { label: 'Classes', count: projectMeta?.classCount || 0, icon: Box, type: 'classes' as const },
        { label: 'Methods', count: projectMeta?.methodCount || 0, icon: Zap, type: 'methods' as const }
      ]
    };
  };

  const displayData = getDisplayData();

  if (!projectMeta) {
    return (
      <div className="w-80 flowlens-panel m-4 p-6 rounded-lg">
        <div className="text-center text-muted-foreground">
          <Files className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p>No project loaded</p>
          <p className="text-sm mt-2">Upload a project or connect a GitHub repository to get started</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-80 flowlens-panel m-4 p-6 rounded-lg space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-lg font-semibold text-foreground truncate">{displayData.title}</h2>
        <p className="text-sm text-muted-foreground truncate">{displayData.subtitle}</p>
      </div>

      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
        <input
          ref={searchInputRef}
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search nodes... (Press '/' to focus)"
          className="flowlens-input pl-10 pr-4"
        />
      </div>

      {/* Stats Cards */}
      <div className="space-y-3">
        <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wide">
          Statistics
        </h3>
        <div className="space-y-2">
          {displayData.stats.map((stat) => (
            <button
              key={stat.type}
              onClick={() => setFilterType(stat.type)}
              className={`w-full flex items-center justify-between p-3 rounded-md transition-colors ${
                filterType === stat.type
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-accent hover:bg-accent/80 text-accent-foreground'
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

      {/* File Tree (simplified for now) */}
      {!selectedFileId && files.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wide">
            Files
          </h3>
          <div className="space-y-1 max-h-60 overflow-y-auto">
            {files.slice(0, 10).map((file) => (
              <div
                key={file.id}
                className="text-sm p-2 rounded hover:bg-accent text-accent-foreground cursor-pointer truncate"
                title={file.path}
              >
                {file.path.split('/').pop()}
              </div>
            ))}
            {files.length > 10 && (
              <div className="text-xs text-muted-foreground p-2">
                ... and {files.length - 10} more files
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default Overview;