import { toPng } from 'html-to-image';

/**
 * Export diagram as PNG image
 */
export async function exportDiagramAsPNG(elementId: string, filename: string = 'flowlens-diagram.png'): Promise<void> {
  try {
    const element = document.getElementById(elementId);
    if (!element) {
      throw new Error(`Element with ID "${elementId}" not found`);
    }

    const dataUrl = await toPng(element, {
      quality: 1,
      pixelRatio: 2, // Higher quality
      backgroundColor: '#ffffff',
      style: {
        transform: 'scale(1)',
        transformOrigin: 'top left',
      },
    });

    // Create download link
    const link = document.createElement('a');
    link.download = filename;
    link.href = dataUrl;
    link.click();
  } catch (error) {
    console.error('Failed to export diagram:', error);
    throw new Error('Failed to export diagram. Please try again.');
  }
}

/**
 * Custom event emitter for graph controls
 */
export function emitGraphEvent(eventName: string, detail?: any): void {
  const event = new CustomEvent(eventName, { detail });
  window.dispatchEvent(event);
}

/**
 * Fit view utility for React Flow
 */
export function fitToView(): void {
  emitGraphEvent('flowlens:fit-view');
}

/**
 * Zoom controls
 */
export function zoomIn(): void {
  emitGraphEvent('flowlens:zoom-in');
}

export function zoomOut(): void {
  emitGraphEvent('flowlens:zoom-out');
}

/**
 * Reset view
 */
export function resetView(): void {
  emitGraphEvent('flowlens:reset-view');
}