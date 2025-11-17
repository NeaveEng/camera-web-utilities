"""
Platform detection and camera backend factory.

This module automatically detects the hardware platform and returns
the appropriate camera backend implementation.
"""

import os
import platform
from pathlib import Path
from typing import Optional

from .base import CameraBackend


def detect_platform() -> str:
    """
    Detect the current hardware platform.
    
    Returns:
        Platform identifier: "jetson", "raspberry_pi", "luxonis", or "generic"
    """
    # Check for Jetson
    device_tree = Path('/proc/device-tree/model')
    if device_tree.exists():
        try:
            model = device_tree.read_text().strip('\x00')
            if 'NVIDIA Jetson' in model:
                return 'jetson'
        except:
            pass
    
    # Check for Raspberry Pi
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.exists():
        try:
            cpu_text = cpuinfo.read_text()
            if 'Raspberry Pi' in cpu_text or 'BCM' in cpu_text:
                return 'raspberry_pi'
        except:
            pass
    
    # Check for Luxonis devices (DepthAI)
    # This would typically check for USB devices or DepthAI SDK
    # For now, we'll skip auto-detection and require manual selection
    
    # Default to generic webcam backend
    return 'generic'


def get_camera_backend(platform_override: Optional[str] = None) -> CameraBackend:
    """
    Get the appropriate camera backend for the current platform.
    
    Args:
        platform_override: Optional platform name to force specific backend
        
    Returns:
        CameraBackend instance for the detected or specified platform
        
    Raises:
        ImportError: If required dependencies for the platform are missing
        ValueError: If platform is unknown
    """
    platform_name = platform_override or detect_platform()
    
    if platform_name == 'jetson':
        from .jetson import JetsonCameraBackend
        return JetsonCameraBackend()
    
    elif platform_name == 'raspberry_pi':
        from .raspberry_pi import RaspberryPiCameraBackend
        return RaspberryPiCameraBackend()
    
    elif platform_name == 'luxonis':
        from .luxonis import LuxonisCameraBackend
        return LuxonisCameraBackend()
    
    elif platform_name == 'generic' or platform_name == 'webcam':
        from .webcam import WebcamCameraBackend
        return WebcamCameraBackend()
    
    else:
        raise ValueError(f"Unknown platform: {platform_name}")


def list_available_platforms() -> list:
    """
    List all camera backend platforms that can be loaded.
    
    Returns:
        List of available platform names
    """
    available = []
    
    # Try to import each backend
    try:
        from .jetson import JetsonCameraBackend
        available.append('jetson')
    except ImportError:
        pass
    
    try:
        from .raspberry_pi import RaspberryPiCameraBackend
        available.append('raspberry_pi')
    except ImportError:
        pass
    
    try:
        from .luxonis import LuxonisCameraBackend
        available.append('luxonis')
    except ImportError:
        pass
    
    try:
        from .webcam import WebcamCameraBackend
        available.append('webcam')
    except ImportError:
        pass
    
    return available
