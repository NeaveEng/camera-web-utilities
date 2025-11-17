"""
Abstract base class for workflows.

Workflows provide step-by-step guided processes for tasks like
camera setup, calibration, etc.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import json
from pathlib import Path
from datetime import datetime


class WorkflowStep:
    """Represents a single step in a workflow."""

    def __init__(self, step_id: str, title: str, description: str,
                 ui_schema: Dict[str, Any], validation_schema: Optional[Dict[str, Any]] = None):
        """
        Initialize a workflow step.

        Args:
            step_id: Unique step identifier
            title: Step title
            description: Step description/instructions
            ui_schema: JSON schema for UI rendering
            validation_schema: Optional JSON schema for validating step data
        """
        self.step_id = step_id
        self.title = title
        self.description = description
        self.ui_schema = ui_schema
        self.validation_schema = validation_schema or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary."""
        return {
            'step_id': self.step_id,
            'title': self.title,
            'description': self.description,
            'ui_schema': self.ui_schema,
            'validation_schema': self.validation_schema
        }


class Workflow(ABC):
    """Abstract base class for workflows."""

    def __init__(self):
        """Initialize the workflow."""
        self.state: Dict[str, Any] = {
            'current_step': 0,
            'completed_steps': [],
            'step_data': {},
            'result': None,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get workflow metadata.

        Returns:
            Dictionary containing:
                - name: Workflow name
                - description: Workflow description
                - version: Workflow version
                - category: Workflow category
                - requires_cameras: Number of cameras required
                - estimated_time: Estimated completion time
        """
        pass

    @abstractmethod
    def get_steps(self) -> List[WorkflowStep]:
        """
        Get ordered list of workflow steps.

        Returns:
            List of WorkflowStep instances
        """
        pass

    @abstractmethod
    def validate_step(self, step_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate data for a specific step.

        Args:
            step_id: Step identifier
            data: Data to validate

        Returns:
            Dictionary containing:
                - valid: Boolean indicating if data is valid
                - errors: List of validation error messages (if any)
        """
        pass

    @abstractmethod
    def execute_step(self, step_id: str, data: Dict[str, Any],
                    camera_backend, group_manager) -> Dict[str, Any]:
        """
        Execute a workflow step.

        Args:
            step_id: Step identifier
            data: Step data from user
            camera_backend: CameraBackend instance
            group_manager: CameraGroupManager instance

        Returns:
            Dictionary containing:
                - success: Boolean indicating if step executed successfully
                - message: Optional message for user
                - next_step: Optional next step ID (for conditional branching)
                - result: Optional step result data
        """
        pass

    def get_current_step(self) -> int:
        """Get current step index."""
        return self.state.get('current_step', 0)

    def get_step_data(self, step_id: str) -> Optional[Dict[str, Any]]:
        """Get data for a specific step."""
        return self.state.get('step_data', {}).get(step_id)

    def set_step_data(self, step_id: str, data: Dict[str, Any]):
        """Store data for a step."""
        if 'step_data' not in self.state:
            self.state['step_data'] = {}
        self.state['step_data'][step_id] = data
        self.state['updated_at'] = datetime.now().isoformat()

    def mark_step_completed(self, step_id: str):
        """Mark a step as completed."""
        if 'completed_steps' not in self.state:
            self.state['completed_steps'] = []
        if step_id not in self.state['completed_steps']:
            self.state['completed_steps'].append(step_id)

    def advance_step(self):
        """Advance to the next step."""
        self.state['current_step'] = self.state.get('current_step', 0) + 1
        self.state['updated_at'] = datetime.now().isoformat()

    def go_to_step(self, step_index: int):
        """Go to a specific step by index."""
        steps = self.get_steps()
        if 0 <= step_index < len(steps):
            self.state['current_step'] = step_index
            self.state['updated_at'] = datetime.now().isoformat()

    def is_completed(self) -> bool:
        """Check if workflow is completed."""
        steps = self.get_steps()
        return self.state.get('current_step', 0) >= len(steps)

    def get_state(self) -> Dict[str, Any]:
        """
        Get complete workflow state.

        Returns:
            Dictionary containing workflow state
        """
        return {
            **self.state,
            'metadata': self.get_metadata(),
            'steps': [step.to_dict() for step in self.get_steps()],
            'is_completed': self.is_completed()
        }

    def save_state(self, filepath: Path):
        """
        Save workflow state to JSON file.

        Args:
            filepath: Path to save state file
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.get_state(), f, indent=2)

    def load_state(self, filepath: Path):
        """
        Load workflow state from JSON file.

        Args:
            filepath: Path to state file
        """
        if not filepath.exists():
            return
        
        with open(filepath, 'r') as f:
            loaded_state = json.load(f)
            # Restore state
            self.state = {
                'current_step': loaded_state.get('current_step', 0),
                'completed_steps': loaded_state.get('completed_steps', []),
                'step_data': loaded_state.get('step_data', {}),
                'result': loaded_state.get('result'),
                'created_at': loaded_state.get('created_at', datetime.now().isoformat()),
                'updated_at': loaded_state.get('updated_at', datetime.now().isoformat())
            }

    def reset(self):
        """Reset workflow to initial state."""
        self.state = {
            'current_step': 0,
            'completed_steps': [],
            'step_data': {},
            'result': None,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
