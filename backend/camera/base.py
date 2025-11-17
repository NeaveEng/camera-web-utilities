"""
Abstract base class for camera backend implementations.

This module defines the interface that all hardware-specific camera backends
must implement to provide a unified API across different platforms.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class CameraBackend(ABC):
    """Abstract base class for camera backend implementations."""

    @abstractmethod
    def enumerate_cameras(self) -> List[Dict[str, Any]]:
        """
        Enumerate all available cameras on the system.

        Returns:
            List of camera dictionaries with the following keys:
                - id: Unique camera identifier (str or int)
                - name: Human-readable camera name
                - type: Camera type (e.g., "CSI", "USB", "DepthAI")
                - available: Whether camera is currently available
        """
        pass

    @abstractmethod
    def get_capabilities(self, camera_id: str) -> Dict[str, Any]:
        """
        Get detailed capabilities for a specific camera.

        Args:
            camera_id: Unique camera identifier

        Returns:
            Dictionary containing:
                - resolutions: List of supported resolutions [(width, height), ...]
                - formats: List of supported pixel formats
                - frame_rates: List of supported frame rates
                - controls: Available camera controls (see get_controls)
        """
        pass

    @abstractmethod
    def start_stream(self, camera_id: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Start streaming from a specific camera.

        Args:
            camera_id: Unique camera identifier
            config: Optional configuration dictionary with:
                - width: Desired frame width
                - height: Desired frame height
                - fps: Desired frame rate
                - format: Desired pixel format
                - preview_width: Width for preview stream (lower than full res)
                - preview_height: Height for preview stream
                - preview_quality: JPEG quality for preview (0-100)

        Returns:
            True if stream started successfully, False otherwise
        """
        pass

    @abstractmethod
    def stop_stream(self, camera_id: str) -> bool:
        """
        Stop streaming from a specific camera.

        Args:
            camera_id: Unique camera identifier

        Returns:
            True if stream stopped successfully, False otherwise
        """
        pass

    @abstractmethod
    def get_full_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """
        Get the latest full-resolution frame from camera (non-blocking).

        This provides high-resolution frames for feature processing like
        calibration and object detection.

        Args:
            camera_id: Unique camera identifier

        Returns:
            NumPy array (H, W, C) in BGR format, or None if unavailable
        """
        pass

    @abstractmethod
    def get_preview_frame(self, camera_id: str) -> Optional[bytes]:
        """
        Get the latest preview frame as compressed JPEG (non-blocking).

        This provides lower-resolution/quality frames optimized for web streaming.

        Args:
            camera_id: Unique camera identifier

        Returns:
            JPEG-encoded bytes, or None if unavailable
        """
        pass

    @abstractmethod
    def get_controls(self, camera_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Get available controls for a camera with metadata.

        Returns normalized control names mapped to platform-specific controls.

        Args:
            camera_id: Unique camera identifier

        Returns:
            Dictionary mapping control names to metadata:
            {
                "exposure": {
                    "type": "range" | "bool" | "menu",
                    "min": 0,
                    "max": 100000,
                    "step": 1,
                    "default": 10000,
                    "current": 10000,
                    "unit": "microseconds",
                    "dependencies": ["auto_exposure"],  # Controls this depends on
                    "platform_name": "exposuretimerange"  # Platform-specific name
                },
                "auto_exposure": {
                    "type": "bool",
                    "default": True,
                    "current": True,
                    "disables": ["exposure"]  # Controls disabled when this is True
                },
                "white_balance": {
                    "type": "menu",
                    "options": ["auto", "incandescent", "fluorescent", "daylight"],
                    "default": "auto",
                    "current": "auto"
                },
                ...
            }
        """
        pass

    @abstractmethod
    def set_control(self, camera_id: str, control_name: str, value: Any) -> bool:
        """
        Set a camera control value (works during active streaming).

        Args:
            camera_id: Unique camera identifier
            control_name: Normalized control name (e.g., "exposure", "white_balance")
            value: Control value (type depends on control)

        Returns:
            True if control was set successfully, False otherwise
        """
        pass

    @abstractmethod
    def get_control(self, camera_id: str, control_name: str) -> Optional[Any]:
        """
        Get current value of a camera control.

        Args:
            camera_id: Unique camera identifier
            control_name: Normalized control name

        Returns:
            Current control value, or None if unavailable
        """
        pass

    @abstractmethod
    def get_profiles(self) -> List[str]:
        """
        Get list of available camera profile names.

        Returns:
            List of profile names (e.g., ["indoor", "outdoor", "low_light"])
        """
        pass

    @abstractmethod
    def get_profile(self, profile_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific camera profile.

        Args:
            profile_name: Name of the profile

        Returns:
            Dictionary of control settings, or None if profile doesn't exist
        """
        pass

    @abstractmethod
    def save_profile(self, profile_name: str, settings: Dict[str, Any]) -> bool:
        """
        Save a camera profile.

        Args:
            profile_name: Name for the profile
            settings: Dictionary of control settings

        Returns:
            True if saved successfully, False otherwise
        """
        pass

    @abstractmethod
    def delete_profile(self, profile_name: str) -> bool:
        """
        Delete a camera profile.

        Args:
            profile_name: Name of the profile to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        pass

    @abstractmethod
    def apply_profile(self, camera_id: str, profile_name: str) -> bool:
        """
        Apply a profile to a camera.

        Args:
            camera_id: Unique camera identifier
            profile_name: Name of the profile to apply

        Returns:
            True if applied successfully, False otherwise
        """
        pass

    @abstractmethod
    def is_streaming(self, camera_id: str) -> bool:
        """
        Check if a camera is currently streaming.

        Args:
            camera_id: Unique camera identifier

        Returns:
            True if camera is streaming, False otherwise
        """
        pass

    @abstractmethod
    def get_platform_name(self) -> str:
        """
        Get the name of this backend platform.

        Returns:
            Platform name (e.g., "jetson", "raspberry_pi", "luxonis", "webcam")
        """
        pass
