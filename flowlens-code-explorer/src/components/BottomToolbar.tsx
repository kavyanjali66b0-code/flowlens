import React from 'react';
import { ZoomIn, ZoomOut, Maximize, RotateCcw, Download } from 'lucide-react';
import { exportDiagramAsPNG, zoomIn, zoomOut, fitToView, resetView } from '../utils/exportUtils';
import { toast } from 'react-toastify';

const BottomToolbar: React.FC = () => {
  const handleExportPNG = async () => {
    try {
      await exportDiagramAsPNG('graph-root', 'flowlens-diagram.png');
      toast.success('Diagram exported successfully!');
    } catch (error) {
      toast.error('Failed to export diagram');
    }
  };

  const buttons = [
    {
      icon: ZoomOut,
      label: 'Zoom Out',
      onClick: zoomOut,
    },
    {
      icon: ZoomIn,
      label: 'Zoom In',
      onClick: zoomIn,
    },
    {
      icon: Maximize,
      label: 'Fit View',
      onClick: fitToView,
    },
    {
      icon: RotateCcw,
      label: 'Reset View',
      onClick: resetView,
    },
    {
      icon: Download,
      label: 'Export PNG',
      onClick: handleExportPNG,
    },
  ];

  return (
    <div className="fixed bottom-6 left-1/2 transform -translate-x-1/2 z-10">
      <div className="flowlens-panel rounded-full px-4 py-2 shadow-lg">
        <div className="flex items-center space-x-2">
          {buttons.map((button, index) => (
            <button
              key={index}
              onClick={button.onClick}
              className="flowlens-button-ghost h-10 w-10 p-0 rounded-full hover:bg-accent"
              title={button.label}
              aria-label={button.label}
            >
              <button.icon className="h-5 w-5" />
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default BottomToolbar;