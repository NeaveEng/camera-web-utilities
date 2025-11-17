"""
Luxonis DepthAI camera backend.

This is a stub implementation for future Luxonis DepthAI support.
"""

from typing import Dict, List, Optional, Any
import numpy as np

from .base import CameraBackend


class LuxonisCameraBackend(CameraBackend):
    """Camera backend for Luxonis DepthAI cameras."""

    def __init__(self):
        """Initialize the Luxonis camera backend."""
        self.platform_name = "luxonis"
        raise NotImplementedError(
            "Luxonis DepthAI backend not yet implemented. "
            "This will use the DepthAI SDK for camera access and stereo depth."
        )

    def enumerate_cameras(self) -> List[Dict[str, Any]]:
        """Enumerate available cameras."""
        raise NotImplementedError()

    def get_capabilities(self, camera_id: str) -> Dict[str, Any]:
        """Get camera capabilities."""
        raise NotImplementedError()

    def start_stream(self, camera_id: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """Start camera stream."""
        raise NotImplementedError()

    def stop_stream(self, camera_id: str) -> bool:
        """Stop camera stream."""
        raise NotImplementedError()

    def get_full_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """Get full resolution frame."""
        raise NotImplementedError()

    def get_preview_frame(self, camera_id: str) -> Optional[bytes]:
        """Get preview frame."""
        raise NotImplementedError()

    def get_controls(self, camera_id: str) -> Dict[str, Dict[str, Any]]:
        """Get available controls."""
        raise NotImplementedError()

    def set_control(self, camera_id: str, control_name: str, value: Any) -> bool:
        """Set a control value."""
        raise NotImplementedError()

    def get_control(self, camera_id: str, control_name: str) -> Optional[Any]:
        """Get current control value."""
        raise NotImplementedError()

    def get_profiles(self) -> List[str]:
        """Get available profiles."""
        raise NotImplementedError()

    def get_profile(self, profile_name: str) -> Optional[Dict[str, Any]]:
        """Get a profile."""
        raise NotImplementedError()

    def save_profile(self, profile_name: str, settings: Dict[str, Any]) -> bool:
        """Save a profile."""
        raise NotImplementedError()

    def delete_profile(self, profile_name: str) -> bool:
        """Delete a profile."""
        raise NotImplementedError()

    def apply_profile(self, camera_id: str, profile_name: str) -> bool:
        """Apply a profile to camera."""
        raise NotImplementedError()

    def is_streaming(self, camera_id: str) -> bool:
        """Check if camera is streaming."""
        raise NotImplementedError()

    def get_platform_name(self) -> str:
        """Get platform name."""
        return self.platform_name
