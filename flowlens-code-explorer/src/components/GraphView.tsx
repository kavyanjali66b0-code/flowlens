import React, { useCallback, useEffect, useMemo } from 'react';

import {
  ReactFlow,
  Node,
  Edge,
  addEdge,
  useNodesState,
  useEdgesState,
  useReactFlow,
  Background,
  Controls,
  MiniMap,
  ConnectionMode,
  NodeTypes,
  Handle,
  Position,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { useFlowLensStore } from '../store/useFlowLensStore';
import { getNodesForView, getEdgesForView, filterNodesByQuery, filterNodesByType } from '../services/dataService';
import { FileIcon, Box, Zap, GitBranch, ArrowRight } from 'lucide-react';

// Custom node component with improved interaction
const CustomNode: React.FC<{ data: any; selected: boolean }> = ({ data, selected }) => {
  const { setInspectedNode, inspectedNodeId, setSelectedFile, setSelectedClass, setSelectedMethod } = useFlowLensStore();
  
  const isInspected = inspectedNodeId === data.nodeId;

  const getIcon = () => {
    switch (data.type) {
      case 'file':
        return <FileIcon className="h-4 w-4" />;
      case 'class':
        return <Box className="h-4 w-4" />;
      case 'method':
        return <Zap className="h-4 w-4" />;
      default:
        return <GitBranch className="h-4 w-4" />;
    }
  };

  const getNodeStyle = () => {
    const baseStyle = 'px-4 py-3 rounded-lg border-2 bg-white min-w-[120px] transition-all duration-200 relative shadow-sm cursor-pointer group';
    
    if (selected) {
      return `${baseStyle} border-blue-500 bg-blue-50 shadow-lg ring-2 ring-blue-200`;
    }
    
    if (isInspected) {
      return `${baseStyle} border-green-400 bg-green-50 shadow-md ring-2 ring-green-200`;
    }
    
    if (data.dimmed) {
      return `${baseStyle} border-gray-200 bg-gray-50 text-gray-400 opacity-50`;
    }
    
    return `${baseStyle} border-gray-300 hover:border-blue-400 hover:bg-gray-50 hover:shadow-md`;
  };

  const getNodeColor = () => {
    if (selected) return '#3b82f6';
    if (isInspected) return '#22c55e';
    
    switch (data.type) {
      case 'file':
        return '#6366f1';
      case 'class':
        return '#8b5cf6';
      case 'method':
        return '#06b6d4';
      default:
        return '#64748b';
    }
  };

  const handleMouseEnter = () => {
    setInspectedNode(data.nodeId);
  };

  const handleDoubleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    // Double-click to navigate (drill down)
    switch (data.type) {
      case 'file':
        setSelectedFile(data.nodeId);
        break;
      case 'class':
        const classFileId = data.fileId as string | undefined;
        if (classFileId) setSelectedFile(classFileId);
        setSelectedClass(data.nodeId);
        break;
      case 'method':
        const methodFileId = data.fileId as string | undefined;
        const methodClassId = data.classId as string | undefined;
        if (methodFileId) setSelectedFile(methodFileId);
        if (methodClassId) setSelectedClass(methodClassId);
        setSelectedMethod(data.nodeId);
        break;
    }
  };

  return (
    <>
      {/* Target handle (input) */}
      <Handle
        type="target"
        position={Position.Left}
        style={{
          background: getNodeColor(),
          border: '2px solid #fff',
          width: 8,
          height: 8,
        }}
      />
      
      <div 
        className={getNodeStyle()}
        onMouseEnter={handleMouseEnter}
        onDoubleClick={handleDoubleClick}
        title="Hover to inspect • Double-click to navigate"
      >
        <div className="flex items-center space-x-2">
          {getIcon()}
          <div className="flex-1 min-w-0">
            <div className="font-medium text-sm text-gray-900 truncate">{data.label}</div>
            {data.description && (
              <div className="text-xs text-gray-500 truncate">
                {data.description}
              </div>
            )}
          </div>
          {/* Navigation indicator */}
          <div className="opacity-0 group-hover:opacity-100 transition-opacity">
            <ArrowRight className="h-3 w-3 text-gray-400" />
          </div>
        </div>
        
        {/* Inspection indicator */}
        {isInspected && (
          <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full border-2 border-white shadow-sm"></div>
        )}
      </div>

      {/* Source handle (output) */}
      <Handle
        type="source"
        position={Position.Right}
        style={{
          background: getNodeColor(),
          border: '2px solid #fff',
          width: 8,
          height: 8,
        }}
      />
    </>
  );
};

const nodeTypes: NodeTypes = {
  custom: CustomNode,
};

const GraphView: React.FC = () => {
  const {
    graphNodes,
    graphEdges,
    viewLevel,
    selectedFileId,
    selectedClassId,
    selectedMethodId,
    searchQuery,
    filterType,
    projectMeta,
    setInspectedNode,
  } = useFlowLensStore();

  const { fitView, zoomIn, zoomOut } = useReactFlow();
  
  // Get filtered nodes based on current view and filters
  const filteredNodes = useMemo(() => {
    console.log('Computing filtered nodes...', { 
      totalNodes: graphNodes.length, 
      viewLevel, 
      selectedFileId, 
      selectedClassId,
      filterType,
      searchQuery 
    });
    
    // Start with view-level filtering
    let nodes = getNodesForView(graphNodes, viewLevel, selectedFileId, selectedClassId);
    
    // Apply type filtering if not 'files' (which is default for repo view)
    if (filterType !== 'files' || viewLevel !== 'repo') {
      const targetType = viewLevel === 'file' ? 'classes' : 
                        viewLevel === 'class' || viewLevel === 'method' ? 'methods' : 
                        filterType;
      
      if (targetType !== 'files') {
        nodes = filterNodesByType(nodes, targetType as 'files' | 'classes' | 'methods');
      }
    }
    
    // Apply search query filtering
    if (searchQuery) {
      const matchingNodes = filterNodesByQuery(nodes, searchQuery);
      const matchingIds = new Set(matchingNodes.map(n => n.id));
      
      // Mark non-matching nodes as dimmed instead of hiding them
      nodes = nodes.map(node => ({
        ...node,
        data: {
          ...node.data,
          dimmed: !matchingIds.has(node.id)
        }
      }));
    }
    
    // Convert to React Flow format with proper selection state
    const reactFlowNodes = nodes.map(node => ({
      id: node.id,
      type: 'custom',
      position: node.position,
      data: {
        ...node.data,
        type: node.type,
        nodeId: node.id, // Pass the actual node ID for inspection
      },
      parentNode: node.parentNode,
      extent: node.extent,
      selected: 
        (node.type === 'file' && node.id === selectedFileId) ||
        (node.type === 'class' && node.id === selectedClassId) ||
        (node.type === 'method' && node.id === selectedMethodId)
    }));
    
    console.log('Final React Flow nodes:', reactFlowNodes.length);
    return reactFlowNodes;
  }, [graphNodes, viewLevel, selectedFileId, selectedClassId, selectedMethodId, searchQuery, filterType]);

  const filteredEdges = useMemo(() => {
    const visibleNodeIds = new Set(filteredNodes.map(n => n.id));
    const edges = getEdgesForView(graphEdges, visibleNodeIds);
    
    const reactFlowEdges = edges.map(edge => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      type: 'default',
      animated: edge.animated || false,
      label: edge.label,
      style: edge.style || { strokeWidth: 2, stroke: '#94a3b8' },
    }));
    
    console.log('Final React Flow edges:', reactFlowEdges.length);
    return reactFlowEdges;
  }, [graphEdges, filteredNodes]);

  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  // Update nodes and edges when filtered data changes
  useEffect(() => {
    console.log('Updating nodes and edges...', { 
      nodesLength: filteredNodes.length, 
      edgesLength: filteredEdges.length 
    });
    setNodes(filteredNodes);
    setEdges(filteredEdges);
  }, [filteredNodes, filteredEdges, setNodes, setEdges]);

  // Handle clicks on empty space to clear inspection
  const handlePaneClick = useCallback(() => {
    setInspectedNode(null);
  }, [setInspectedNode]);

  const onConnect = useCallback((params: any) => setEdges((eds) => addEdge(params, eds)), [setEdges]);

  // Listen for external events from toolbar
  useEffect(() => {
    const handleZoomIn = () => {
      try {
        zoomIn();
      } catch (error) {
        console.error('Error zooming in:', error);
      }
    };
    
    const handleZoomOut = () => {
      try {
        zoomOut();
      } catch (error) {
        console.error('Error zooming out:', error);
      }
    };
    
    const handleFitView = () => {
      try {
        fitView({ padding: 0.2 });
      } catch (error) {
        console.error('Error fitting view:', error);
      }
    };

    window.addEventListener('flowlens:zoom-in', handleZoomIn);
    window.addEventListener('flowlens:zoom-out', handleZoomOut);
    window.addEventListener('flowlens:fit-view', handleFitView);
    window.addEventListener('flowlens:reset-view', handleFitView);

    return () => {
      window.removeEventListener('flowlens:zoom-in', handleZoomIn);
      window.removeEventListener('flowlens:zoom-out', handleZoomOut);
      window.removeEventListener('flowlens:fit-view', handleFitView);
      window.removeEventListener('flowlens:reset-view', handleFitView);
    };
  }, [zoomIn, zoomOut, fitView]);

  // Auto-fit view when nodes change, but only if there are nodes
  useEffect(() => {
    if (nodes.length > 0) {
      console.log('Auto-fitting view for', nodes.length, 'nodes');
      setTimeout(() => {
        try {
          fitView({ padding: 0.2, duration: 800 });
        } catch (error) {
          console.error('Error auto-fitting view:', error);
        }
      }, 300);
    }
  }, [nodes.length, fitView]);

  // Show empty state if no project is loaded
  if (!projectMeta) {
    return (
      <div className="flex-1 bg-white m-4 rounded-lg border overflow-hidden" style={{ minHeight: '600px' }}>
        <div className="w-full h-full flex items-center justify-center" style={{ height: '600px' }}>
          <div className="text-center text-gray-500">
            <GitBranch className="h-16 w-16 mx-auto mb-4 opacity-50" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No Project Loaded</h3>
            <p className="text-sm">Upload a project or connect a GitHub repository to visualize its structure</p>
          </div>
        </div>
      </div>
    );
  }

  // Show empty graph state if no nodes to display
  if (nodes.length === 0) {
    const contextMessage = 
      selectedMethodId ? 'No methods found in the selected context' :
      selectedClassId ? 'No items found in the selected class' :
      selectedFileId ? 'No items found in the selected file' :
      searchQuery ? `No nodes match "${searchQuery}"` : 
      'No nodes found for the current filters';
    
    return (
      <div className="flex-1 bg-white m-4 rounded-lg border overflow-hidden" style={{ minHeight: '600px' }}>
        <div className="w-full h-full flex items-center justify-center" style={{ height: '600px' }}>
          <div className="text-center text-gray-500">
            <GitBranch className="h-16 w-16 mx-auto mb-4 opacity-50" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No Nodes to Display</h3>
            <p className="text-sm">{contextMessage}</p>
            <div className="text-xs mt-2 space-y-1">
              <p>View: {viewLevel} | Filter: {filterType}</p>
              <p>Total available: {graphNodes.length} nodes, {graphEdges.length} edges</p>
              <p className="text-blue-500 mt-2">Hover nodes to inspect • Double-click to navigate</p>
            </div>
          </div>
        </div>
      </div>
    );
  }
  
  return (
    <div className="flex-1 bg-white m-4 rounded-lg border overflow-hidden" style={{ minHeight: '600px' }}>
      {/* Enhanced status bar with interaction hints */}
      <div className="px-4 py-2 border-b bg-gray-50 text-xs text-gray-600 flex justify-between items-center">
        <div>
          Showing {nodes.length} nodes, {edges.length} edges | View: {viewLevel} | Filter: {filterType}
          {searchQuery && ` | Search: "${searchQuery}"`}
        </div>
        <div className="flex space-x-4 items-center">
          <span className="text-green-600">🔍 Hover to inspect</span>
          <span className="text-blue-600">⚡ Double-click to navigate</span>
          {selectedFileId && <span className="text-blue-600">File: {selectedFileId.split('_').pop()}</span>}
          {selectedClassId && <span className="text-purple-600">Class: {selectedClassId.split('_').pop()}</span>}
          {selectedMethodId && <span className="text-cyan-600">Method: {selectedMethodId.split('_').pop()}</span>}
        </div>
      </div>
      
      <div 
        id="graph-root" 
        className="w-full relative"
        style={{ 
          height: '560px', // Account for status bar
          width: '100%',
          backgroundColor: '#fafafa'
        }}
      >
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onPaneClick={handlePaneClick}
          nodeTypes={nodeTypes}
          connectionMode={ConnectionMode.Loose}
          fitView
          attributionPosition="bottom-right"
          style={{ width: '100%', height: '100%' }}
          defaultEdgeOptions={{
            type: 'default',
            animated: false,
          }}
          minZoom={0.2}
          maxZoom={4}
          defaultViewport={{ x: 0, y: 0, zoom: 0.8 }}
        >
          <Background />
          <Controls showInteractive={false} />
          <MiniMap 
            zoomable 
            pannable
            nodeColor={(node) => {
              if (node.selected) return '#3b82f6';
              // Check if this node is inspected
              const isInspected = node.data?.nodeId === useFlowLensStore.getState().inspectedNodeId;
              if (isInspected) return '#22c55e';
              
              switch (node.data?.type) {
                case 'file':
                  return '#6366f1';
                case 'class':
                  return '#8b5cf6';
                case 'method':
                  return '#06b6d4';
                default:
                  return '#64748b';
              }
            }}
            style={{
              backgroundColor: '#f8fafc',
            }}
          />
        </ReactFlow>
      </div>
    </div>
  );
};

export default GraphView;