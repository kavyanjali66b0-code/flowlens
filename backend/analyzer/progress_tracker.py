"""Progress tracking for analysis operations."""

from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class AnalysisPhase(Enum):
    """Phases of analysis process."""
    INITIALIZING = "initializing"
    SCANNING = "scanning_project"
    DETECTING_TYPE = "detecting_type"
    IDENTIFYING_ENTRIES = "identifying_entries"
    PARSING_FILES = "parsing_files"
    ANALYZING_RELATIONSHIPS = "analyzing_relationships"
    GENERATING_GRAPH = "generating_graph"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProgressUpdate:
    """Progress update snapshot."""
    phase: AnalysisPhase
    progress: float  # 0.0 to 1.0
    current: int
    total: int
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    errors: list[str] = field(default_factory=list)


class ProgressTracker:
    """Tracks and reports analysis progress."""
    
    def __init__(self, session_id: str):
        """Initialize progress tracker."""
        self.session_id = session_id
        self.current_phase = AnalysisPhase.INITIALIZING
        self.updates: list[ProgressUpdate] = []
        self.start_time = datetime.now()
        
        # Phase weights for overall progress calculation
        self.phase_weights = {
            AnalysisPhase.INITIALIZING: 0.05,
            AnalysisPhase.SCANNING: 0.10,
            AnalysisPhase.DETECTING_TYPE: 0.05,
            AnalysisPhase.IDENTIFYING_ENTRIES: 0.05,
            AnalysisPhase.PARSING_FILES: 0.50,
            AnalysisPhase.ANALYZING_RELATIONSHIPS: 0.20,
            AnalysisPhase.GENERATING_GRAPH: 0.05,
        }
        
    def update(
        self,
        phase: AnalysisPhase,
        current: int,
        total: int,
        message: str = "",
        errors: Optional[list[str]] = None
    ):
        """Update progress."""
        progress = current / total if total > 0 else 0.0
        
        update = ProgressUpdate(
            phase=phase,
            progress=progress,
            current=current,
            total=total,
            message=message or f"{phase.value}: {current}/{total}",
            errors=errors or []
        )
        
        self.current_phase = phase
        self.updates.append(update)
        
    def get_overall_progress(self) -> float:
        """Calculate overall progress across all phases."""
        if self.current_phase == AnalysisPhase.COMPLETED:
            return 1.0
        if self.current_phase == AnalysisPhase.FAILED:
            return 0.0
            
        # Calculate weighted progress
        completed_weight = 0.0
        
        # Get phase ordering
        phase_order = list(self.phase_weights.keys())
        
        # Find current phase index
        try:
            current_phase_index = phase_order.index(self.current_phase)
        except ValueError:
            # Phase not in weights (e.g., COMPLETED, FAILED)
            return 0.0
        
        # Add weights for all completed phases
        for idx, phase in enumerate(phase_order):
            if idx < current_phase_index:
                # Phase already completed
                completed_weight += self.phase_weights[phase]
        
        # Add partial progress for current phase
        if self.updates and self.current_phase in self.phase_weights:
            latest = self.updates[-1]
            if latest.phase == self.current_phase:
                phase_weight = self.phase_weights[self.current_phase]
                completed_weight += phase_weight * latest.progress
        
        return completed_weight
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current status summary."""
        latest = self.updates[-1] if self.updates else None
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "session_id": self.session_id,
            "phase": self.current_phase.value,
            "overall_progress": self.get_overall_progress(),
            "current_progress": latest.progress if latest else 0.0,
            "current": latest.current if latest else 0,
            "total": latest.total if latest else 0,
            "message": latest.message if latest else "",
            "elapsed_seconds": elapsed,
            "errors": latest.errors if latest else []
        }
    
    def mark_completed(self):
        """Mark analysis as completed."""
        self.current_phase = AnalysisPhase.COMPLETED
        self.update(
            AnalysisPhase.COMPLETED,
            current=1,
            total=1,
            message="Analysis completed successfully"
        )
    
    def mark_failed(self, error: str):
        """Mark analysis as failed."""
        self.current_phase = AnalysisPhase.FAILED
        self.update(
            AnalysisPhase.FAILED,
            current=0,
            total=1,
            message=f"Analysis failed: {error}",
            errors=[error]
        )
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return (datetime.now() - self.start_time).total_seconds()
    
    def get_estimated_remaining_time(self) -> Optional[float]:
        """Estimate remaining time based on current progress."""
        progress = self.get_overall_progress()
        if progress <= 0 or progress >= 1.0:
            return None
            
        elapsed = self.get_elapsed_time()
        estimated_total = elapsed / progress
        return estimated_total - elapsed
