"""
Workflow manager.

Discovers workflows from features and global templates, manages instances.
"""

import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from .base import Workflow


class WorkflowManager:
    """Manages workflow discovery, instances, and state."""

    def __init__(self, template_dir: Optional[Path] = None, state_dir: Optional[Path] = None):
        """
        Initialize the workflow manager.

        Args:
            template_dir: Directory containing global workflow templates
            state_dir: Directory for storing workflow instance states
        """
        self.template_dir = template_dir or (Path(__file__).parent / 'templates')
        self.state_dir = state_dir or Path('data/workflows/state')
        
        self.template_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        self.workflow_templates: Dict[str, type] = {}
        self.workflow_instances: Dict[str, Workflow] = {}
        
        self._discover_workflows()

    def _discover_workflows(self):
        """Discover workflow templates from global templates directory."""
        # Create __init__.py if it doesn't exist
        init_file = self.template_dir / '__init__.py'
        if not init_file.exists():
            init_file.write_text('"""Workflow templates."""\n')
        
        # Scan for Python files
        for workflow_file in self.template_dir.glob('*.py'):
            if workflow_file.name.startswith('_'):
                continue
            
            try:
                # Import the module
                module_name = f'backend.workflows.templates.{workflow_file.stem}'
                module = importlib.import_module(module_name)
                
                # Find Workflow subclasses
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, Workflow) and obj is not Workflow:
                        # Store template class (don't instantiate yet)
                        workflow_instance = obj()
                        metadata = workflow_instance.get_metadata()
                        workflow_name = metadata.get('name', name)
                        
                        self.workflow_templates[workflow_name] = obj
                        print(f"Loaded workflow template: {workflow_name}")
                        
            except Exception as e:
                print(f"Error loading workflow {workflow_file}: {e}")

    def register_plugin_workflows(self, feature_manager):
        """
        Register workflows from feature plugins.

        Args:
            feature_manager: FeatureManager instance
        """
        plugin_workflows = feature_manager.get_workflows()
        for plugin_name, workflows in plugin_workflows.items():
            for workflow in workflows:
                metadata = workflow.get_metadata()
                workflow_name = f"{plugin_name}:{metadata.get('name')}"
                
                # Store the workflow class
                self.workflow_templates[workflow_name] = type(workflow)
                print(f"Loaded plugin workflow: {workflow_name}")

    def list_workflows(self) -> List[Dict[str, Any]]:
        """
        List all available workflow templates.

        Returns:
            List of workflow metadata dictionaries
        """
        workflows = []
        for name, workflow_class in self.workflow_templates.items():
            instance = workflow_class()
            metadata = instance.get_metadata()
            metadata['workflow_name'] = name
            workflows.append(metadata)
        return workflows

    def start_workflow(self, workflow_name: str, instance_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Start a new workflow instance.

        Args:
            workflow_name: Name of the workflow template
            instance_id: Optional instance ID (generated if not provided)

        Returns:
            Dictionary containing:
                - instance_id: Unique workflow instance ID
                - state: Initial workflow state
        """
        if workflow_name not in self.workflow_templates:
            return {
                'success': False,
                'message': f'Workflow {workflow_name} not found'
            }
        
        # Generate instance ID if not provided
        if not instance_id:
            timestamp = int(datetime.now().timestamp())
            instance_id = f"{workflow_name}_{timestamp}"
        
        # Create workflow instance
        workflow_class = self.workflow_templates[workflow_name]
        workflow = workflow_class()
        
        # Try to load existing state
        state_file = self.state_dir / f'{instance_id}.json'
        if state_file.exists():
            workflow.load_state(state_file)
        
        # Store instance
        self.workflow_instances[instance_id] = workflow
        
        return {
            'success': True,
            'instance_id': instance_id,
            'state': workflow.get_state()
        }

    def get_workflow_state(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """Get state of a workflow instance."""
        workflow = self.workflow_instances.get(instance_id)
        if workflow:
            return workflow.get_state()
        
        # Try to load from disk
        state_file = self.state_dir / f'{instance_id}.json'
        if state_file.exists():
            # Determine workflow template from instance_id
            workflow_name = instance_id.rsplit('_', 1)[0]
            if workflow_name in self.workflow_templates:
                workflow_class = self.workflow_templates[workflow_name]
                workflow = workflow_class()
                workflow.load_state(state_file)
                self.workflow_instances[instance_id] = workflow
                return workflow.get_state()
        
        return None

    def validate_step(self, instance_id: str, step_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate data for a workflow step.

        Args:
            instance_id: Workflow instance ID
            step_id: Step identifier
            data: Data to validate

        Returns:
            Validation result dictionary
        """
        workflow = self.workflow_instances.get(instance_id)
        if not workflow:
            return {
                'valid': False,
                'errors': ['Workflow instance not found']
            }
        
        return workflow.validate_step(step_id, data)

    def execute_step(self, instance_id: str, step_id: str, data: Dict[str, Any],
                    camera_backend, group_manager) -> Dict[str, Any]:
        """
        Execute a workflow step.

        Args:
            instance_id: Workflow instance ID
            step_id: Step identifier
            data: Step data
            camera_backend: CameraBackend instance
            group_manager: CameraGroupManager instance

        Returns:
            Execution result dictionary
        """
        workflow = self.workflow_instances.get(instance_id)
        if not workflow:
            return {
                'success': False,
                'message': 'Workflow instance not found'
            }
        
        # Validate first
        validation = workflow.validate_step(step_id, data)
        if not validation.get('valid', False):
            return {
                'success': False,
                'message': 'Validation failed',
                'errors': validation.get('errors', [])
            }
        
        # Execute step
        result = workflow.execute_step(step_id, data, camera_backend, group_manager)
        
        if result.get('success', False):
            # Store step data
            workflow.set_step_data(step_id, data)
            workflow.mark_step_completed(step_id)
            
            # Advance to next step if no explicit next_step
            if 'next_step' not in result:
                workflow.advance_step()
            
            # Save state
            state_file = self.state_dir / f'{instance_id}.json'
            workflow.save_state(state_file)
        
        return result

    def reset_workflow(self, instance_id: str) -> bool:
        """Reset a workflow instance to initial state."""
        workflow = self.workflow_instances.get(instance_id)
        if not workflow:
            return False
        
        workflow.reset()
        
        # Save reset state
        state_file = self.state_dir / f'{instance_id}.json'
        workflow.save_state(state_file)
        
        return True

    def delete_workflow(self, instance_id: str) -> bool:
        """Delete a workflow instance."""
        # Remove from memory
        if instance_id in self.workflow_instances:
            del self.workflow_instances[instance_id]
        
        # Remove state file
        state_file = self.state_dir / f'{instance_id}.json'
        if state_file.exists():
            state_file.unlink()
        
        return True

    def list_instances(self) -> List[Dict[str, Any]]:
        """List all active workflow instances."""
        instances = []
        for instance_id, workflow in self.workflow_instances.items():
            state = workflow.get_state()
            instances.append({
                'instance_id': instance_id,
                'workflow_name': state['metadata']['name'],
                'current_step': state['current_step'],
                'is_completed': state['is_completed'],
                'updated_at': state['updated_at']
            })
        return instances
