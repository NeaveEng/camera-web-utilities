"""
Abstract base class for feature plugins.

Feature plugins extend the camera platform with additional functionality
like calibration, object detection, etc.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import numpy as np


class FeaturePlugin(ABC):
    """Abstract base class for feature plugins."""

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get plugin metadata.

        Returns:
            Dictionary containing:
                - name: Plugin name
                - version: Plugin version
                - description: Brief description
                - author: Plugin author
                - requires_cameras: Number of cameras required (1, 2, 3, or "any")
                - requires_high_res: Whether plugin needs full-resolution frames
                - supports_groups: Whether plugin can work with camera groups
        """
        pass

    @abstractmethod
    def get_ui_schema(self) -> Dict[str, Any]:
        """
        Get UI schema for frontend rendering.

        Returns:
            JSON schema describing the plugin's UI:
            {
                "controls": [
                    {
                        "name": "threshold",
                        "type": "range",
                        "label": "Threshold",
                        "min": 0,
                        "max": 255,
                        "default": 128
                    },
                    {
                        "name": "mode",
                        "type": "select",
                        "label": "Mode",
                        "options": ["fast", "accurate"],
                        "default": "fast"
                    }
                ],
                "buttons": [
                    {
                        "name": "capture",
                        "label": "Capture Image",
                        "action": "capture_calibration_image"
                    }
                ],
                "displays": [
                    {
                        "name": "result",
                        "type": "text",
                        "label": "Result"
                    }
                ]
            }
        """
        pass

    def required_cameras(self) -> int:
        """
        Get number of cameras required.

        Returns:
            Number of cameras (1, 2, 3) or 0 for "any number"
        """
        metadata = self.get_metadata()
        cameras = metadata.get('requires_cameras', 1)
        if cameras == "any":
            return 0
        return int(cameras)

    def requires_high_res(self) -> bool:
        """Check if plugin requires high-resolution frames."""
        return self.get_metadata().get('requires_high_res', False)

    def supports_groups(self) -> bool:
        """Check if plugin supports camera groups."""
        return self.get_metadata().get('supports_groups', False)

    @abstractmethod
    def process_frames(self, camera_backend, camera_ids: List[str], 
                      params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process frames from one or more cameras.

        The plugin pulls frames from the camera backend as needed using:
            - camera_backend.get_full_frame(camera_id) for high-res frames
            - camera_backend.get_preview_frame(camera_id) for preview frames

        Args:
            camera_backend: CameraBackend instance to pull frames from
            camera_ids: List of camera IDs to process
            params: Optional parameters from UI controls

        Returns:
            Dictionary containing:
                - success: Boolean indicating if processing succeeded
                - result: Processing result (type depends on plugin)
                - message: Optional message for user
                - overlay: Optional overlay data for visualization
        """
        pass

    def process_group(self, camera_backend, group, 
                     params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process frames from a camera group.

        Default implementation calls process_frames with group's camera IDs.

        Args:
            camera_backend: CameraBackend instance
            group: CameraGroup instance
            params: Optional parameters

        Returns:
            Processing result dictionary
        """
        if not self.supports_groups():
            return {
                'success': False,
                'message': 'Plugin does not support camera groups'
            }
        
        return self.process_frames(camera_backend, group.camera_ids, params)

    def get_workflows(self) -> List['Workflow']:
        """
        Get workflows provided by this plugin.

        Override this method to provide plugin-specific workflows.

        Returns:
            List of Workflow instances
        """
        return []

    def register_routes(self, app):
        """
        Register plugin-specific API routes.

        Override this method to add custom endpoints for the plugin.

        Args:
            app: Flask/FastAPI application instance
        """
        pass

    def initialize(self):
        """
        Initialize the plugin.

        Called once when plugin is loaded. Use for setup tasks.
        """
        pass

    def cleanup(self):
        """
        Cleanup plugin resources.

        Called when plugin is unloaded or application shuts down.
        """
        pass
