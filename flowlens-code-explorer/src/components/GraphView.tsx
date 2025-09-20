import React, { useCallback, useEffect, useMemo, useState } from 'react';
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
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { useFlowLensStore } from '../store/useFlowLensStore';
import { getNodesForView, getEdgesForView, filterNodesByQuery } from '../services/dataService';
import { FileIcon, Box, Zap } from 'lucide-react';

// Custom node component
const CustomNode: React.FC<{ data: any; selected: boolean }> = ({ data, selected }) => {
  const getIcon = () => {
    switch (data.type) {
      case 'file':
        return <FileIcon className="h-4 w-4" />;
      case 'class':
        return <Box className="h-4 w-4" />;
      case 'method':
        return <Zap className="h-4 w-4" />;
      default:
        return null;
    }
  };

  const getNodeStyle = () => {
    const baseStyle = 'px-4 py-3 rounded-lg border-2 bg-node-bg min-w-[120px] transition-all duration-200';
    
    if (selected) {
      return `${baseStyle} border-primary bg-node-selected shadow-glow`;
    }
    
    return `${baseStyle} border-border hover:border-primary/50 hover:bg-node-hover hover:shadow-md`;
  };

  return (
    <div className={getNodeStyle()}>
      <div className="flex items-center space-x-2">
        {getIcon()}
        <div>
          <div className="font-medium text-sm text-foreground">{data.label}</div>
          {data.description && (
            <div className="text-xs text-muted-foreground truncate max-w-[100px]">
              {data.description}
            </div>
          )}
        </div>
      </div>
    </div>
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
    setSelectedFile,
    setSelectedClass,
    setSelectedMethod,
  } = useFlowLensStore();

  const { fitView, zoomIn, zoomOut } = useReactFlow();
  
  // Get filtered nodes and edges based on current view
  const filteredNodes = useMemo(() => {
    let nodes = getNodesForView(graphNodes, viewLevel, selectedFileId, selectedClassId);
    
    if (searchQuery) {
      const matchingNodes = filterNodesByQuery(nodes, searchQuery);
      const matchingIds = new Set(matchingNodes.map(n => n.id));
      
      // Mark non-matching nodes as dimmed
      nodes = nodes.map(node => ({
        ...node,
        data: {
          ...node.data,
          dimmed: !matchingIds.has(node.id)
        }
      }));
    }
    
    // Convert to React Flow format
    return nodes.map(node => ({
      id: node.id,
      type: 'custom',
      position: node.position,
      data: {
        ...node.data,
        type: node.type
      },
      parentNode: node.parentNode,
      extent: node.extent,
      className: node.data.dimmed ? 'dimmed' : '',
      selected: 
        (node.type === 'file' && node.id === selectedFileId) ||
        (node.type === 'class' && node.data.classId === selectedClassId) ||
        (node.type === 'method' && node.data.methodId === selectedMethodId)
    }));
  }, [graphNodes, viewLevel, selectedFileId, selectedClassId, selectedMethodId, searchQuery]);

  const filteredEdges = useMemo(() => {
    const visibleNodeIds = new Set(filteredNodes.map(n => n.id));
    const edges = getEdgesForView(graphEdges, visibleNodeIds);
    
    return edges.map(edge => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      type: edge.type || 'default',
      animated: edge.animated || false,
      label: edge.label,
      style: { strokeWidth: 2 },
    }));
  }, [graphEdges, filteredNodes]);

  const [nodes, setNodes, onNodesChange] = useNodesState(filteredNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(filteredEdges);

  // Update nodes and edges when filtered data changes
  useEffect(() => {
    setNodes(filteredNodes);
    setEdges(filteredEdges);
  }, [filteredNodes, filteredEdges, setNodes, setEdges]);

  // Handle node clicks
  const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    const { data } = node;
    
    // Emit custom event for node selection
    window.dispatchEvent(new CustomEvent('flowlens:node-selected', {
      detail: { id: node.id, type: data.type }
    }));
    
    switch (data.type) {
      case 'file':
        setSelectedFile((data.fileId as string) || node.id);
        break;
      case 'class':
        setSelectedClass((data.classId as string) || node.id);
        break;
      case 'method':
        setSelectedMethod((data.methodId as string) || node.id);
        break;
    }
  }, [setSelectedFile, setSelectedClass, setSelectedMethod]);

  const onConnect = useCallback((params: any) => setEdges((eds) => addEdge(params, eds)), [setEdges]);

  // Listen for external events
  useEffect(() => {
    const handleZoomIn = () => zoomIn();
    const handleZoomOut = () => zoomOut();
    const handleFitView = () => fitView();
    const handleResetView = () => fitView();

    window.addEventListener('flowlens:zoom-in', handleZoomIn);
    window.addEventListener('flowlens:zoom-out', handleZoomOut);
    window.addEventListener('flowlens:fit-view', handleFitView);
    window.addEventListener('flowlens:reset-view', handleResetView);

    return () => {
      window.removeEventListener('flowlens:zoom-in', handleZoomIn);
      window.removeEventListener('flowlens:zoom-out', handleZoomOut);
      window.removeEventListener('flowlens:fit-view', handleFitView);
      window.removeEventListener('flowlens:reset-view', handleResetView);
    };
  }, [zoomIn, zoomOut, fitView]);

  // Auto-fit view when nodes change
  useEffect(() => {
    if (nodes.length > 0) {
      setTimeout(() => fitView(), 100);
    }
  }, [viewLevel, fitView, nodes.length]);

  return (
    <div className="flex-1 flowlens-panel m-4 rounded-lg overflow-hidden">
      <div id="graph-root" className="h-full bg-graph-bg">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodeClick={onNodeClick}
          nodeTypes={nodeTypes}
          connectionMode={ConnectionMode.Loose}
          fitView
          attributionPosition="bottom-right"
        >
          <Background />
          <Controls showInteractive={false} />
          <MiniMap 
            zoomable 
            pannable
            nodeColor={(node) => {
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
          />
        </ReactFlow>
      </div>
    </div>
  );
};

export default GraphView;