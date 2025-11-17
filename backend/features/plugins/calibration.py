"""
Camera calibration feature plugin.

Provides single camera and stereo camera calibration using ChArUco boards.
Uses calibration_utils module adapted from ne-calib-utils.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from backend.features.base import FeaturePlugin
from backend.calibration_utils import CalibrationCore


class CalibrationPlugin(FeaturePlugin):
    """Camera calibration feature plugin."""

    def __init__(self):
        """Initialize calibration plugin."""
        super().__init__()
        
        # Use CalibrationCore for all calibration operations
        self.calib_core = CalibrationCore()
        
        # Storage for calibration images
        self.calibration_data_dir = Path('data/calibration')
        self.calibration_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Session state
        self.capture_sessions: Dict[str, Dict] = {}

    def get_metadata(self) -> Dict[str, Any]:
        """Get plugin metadata."""
        return {
            'name': 'calibration',
            'version': '1.0.0',
            'description': 'Camera calibration using ChArUco boards',
            'author': 'Camera Web Utilities',
            'requires_cameras': 'any',  # Can calibrate 1 or 2 cameras
            'requires_high_res': True,
            'supports_groups': True
        }

    def get_ui_schema(self) -> Dict[str, Any]:
        """Get UI schema for frontend."""
        return {
            'controls': [
                {
                    'name': 'calibration_type',
                    'type': 'select',
                    'label': 'Calibration Type',
                    'options': ['single', 'stereo'],
                    'default': 'single'
                },
                {
                    'name': 'min_images',
                    'type': 'range',
                    'label': 'Minimum Images',
                    'min': 10,
                    'max': 50,
                    'default': 20,
                    'step': 5
                },
                {
                    'name': 'capture_interval',
                    'type': 'range',
                    'label': 'Capture Interval (seconds)',
                    'min': 1,
                    'max': 10,
                    'default': 2,
                    'step': 1
                },
                {
                    'name': 'fisheye_model',
                    'type': 'checkbox',
                    'label': 'Use Fisheye Model',
                    'default': False
                }
            ],
            'buttons': [
                {
                    'name': 'start_session',
                    'label': 'Start Capture Session',
                    'action': 'start_calibration_session'
                },
                {
                    'name': 'capture_image',
                    'label': 'Capture Image',
                    'action': 'capture_calibration_image'
                },
                {
                    'name': 'stop_session',
                    'label': 'Stop Session',
                    'action': 'stop_calibration_session'
                },
                {
                    'name': 'calibrate',
                    'label': 'Run Calibration',
                    'action': 'run_calibration'
                }
            ],
            'displays': [
                {
                    'name': 'marker_detection',
                    'type': 'overlay',
                    'label': 'Marker Detection'
                },
                {
                    'name': 'session_status',
                    'type': 'text',
                    'label': 'Session Status'
                },
                {
                    'name': 'calibration_result',
                    'type': 'json',
                    'label': 'Calibration Result'
                }
            ]
        }

    def initialize(self):
        """Initialize the plugin."""
        print(f"Calibration plugin initialized")
        print(f"Markers total: {self.calib_core.markers_total}, required: {self.calib_core.markers_required}")

    def process_frames(self, camera_backend, camera_ids: List[str], 
                      params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process frames for calibration.

        Args:
            camera_backend: CameraBackend instance
            camera_ids: List of camera IDs
            params: Parameters including action to perform

        Returns:
            Processing result
        """
        if not params:
            params = {}
        
        action = params.get('action', 'detect_markers')
        
        if action == 'detect_markers':
            return self._detect_markers(camera_backend, camera_ids)
        elif action == 'start_session':
            return self._start_session(camera_backend, camera_ids, params)
        elif action == 'capture_image':
            return self._capture_image(camera_backend, camera_ids, params)
        elif action == 'stop_session':
            return self._stop_session(camera_ids, params)
        elif action == 'run_calibration':
            return self._run_calibration(camera_ids, params)
        else:
            return {
                'success': False,
                'message': f'Unknown action: {action}'
            }

    def _detect_markers(self, camera_backend, camera_ids: List[str]) -> Dict[str, Any]:
        """
        Detect ChArUco markers in current frames.

        Args:
            camera_backend: CameraBackend instance
            camera_ids: List of camera IDs

        Returns:
            Detection results with overlay data
        """
        results = {}
        
        for camera_id in camera_ids:
            try:
                # Get high-res frame
                frame = camera_backend.get_full_frame(camera_id)
                if frame is None:
                    results[camera_id] = {
                        'success': False,
                        'message': 'No frame available'
                    }
                    continue
                
                # Detect markers using CalibrationCore
                corners, ids, rejected = self.calib_core.detect_markers(frame)
                
                # Create overlay data
                overlay_frame = frame.copy()
                detected_count = len(corners) if corners else 0
                
                if corners and detected_count > 0:
                    # Show markers using CalibrationCore
                    overlay_frame = self.calib_core.show_markers(overlay_frame, corners, ids, scale=1)
                    
                    # Add status text
                    status_text = f"Detected: {detected_count}/{self.calib_core.markers_total}"
                    color = (0, 255, 0) if detected_count >= self.calib_core.markers_required else (0, 165, 255)
                    cv2.putText(overlay_frame, status_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                results[camera_id] = {
                    'success': True,
                    'markers_detected': detected_count,
                    'markers_required': self.calib_core.markers_required,
                    'sufficient': detected_count >= self.calib_core.markers_required,
                    'overlay': overlay_frame
                }
                
            except Exception as e:
                results[camera_id] = {
                    'success': False,
                    'message': str(e)
                }
        
        return {
            'success': True,
            'cameras': results
        }

    def _start_session(self, camera_backend, camera_ids: List[str], 
                      params: Dict[str, Any]) -> Dict[str, Any]:
        """Start a calibration capture session."""
        session_id = params.get('session_id', f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        calibration_type = params.get('calibration_type', 'single')
        
        # Validate camera count
        if calibration_type == 'stereo' and len(camera_ids) != 2:
            return {
                'success': False,
                'message': 'Stereo calibration requires exactly 2 cameras'
            }
        
        # Create session
        session_dir = self.calibration_data_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        self.capture_sessions[session_id] = {
            'camera_ids': camera_ids,
            'calibration_type': calibration_type,
            'min_images': params.get('min_images', 20),
            'capture_interval': params.get('capture_interval', 2),
            'fisheye_model': params.get('fisheye_model', False),
            'session_dir': str(session_dir),
            'images_captured': {cam_id: 0 for cam_id in camera_ids},
            'last_capture_time': None,
            'created_at': datetime.now().isoformat()
        }
        
        # Create subdirectories for each camera
        for camera_id in camera_ids:
            (session_dir / camera_id).mkdir(exist_ok=True)
        
        return {
            'success': True,
            'session_id': session_id,
            'message': f'Started {calibration_type} calibration session',
            'session': self.capture_sessions[session_id]
        }

    def _capture_image(self, camera_backend, camera_ids: List[str], 
                      params: Dict[str, Any]) -> Dict[str, Any]:
        """Capture calibration images."""
        session_id = params.get('session_id')
        
        if not session_id or session_id not in self.capture_sessions:
            return {
                'success': False,
                'message': 'Invalid or missing session_id'
            }
        
        session = self.capture_sessions[session_id]
        results = {}
        
        for camera_id in camera_ids:
            try:
                # Get frame
                frame = camera_backend.get_full_frame(camera_id)
                if frame is None:
                    results[camera_id] = {
                        'success': False,
                        'message': 'No frame available'
                    }
                    continue
                
                # Detect markers
                corners, ids, _ = self.calib_core.detect_markers(frame)
                
                detected_count = len(corners) if corners else 0
                
                if detected_count >= self.calib_core.markers_required:
                    # Save image
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                    filename = f"{session['session_dir']}/{camera_id}/capture_{timestamp}.png"
                    cv2.imwrite(filename, frame)
                    
                    session['images_captured'][camera_id] += 1
                    session['last_capture_time'] = datetime.now().isoformat()
                    
                    results[camera_id] = {
                        'success': True,
                        'filename': filename,
                        'markers_detected': detected_count,
                        'total_captured': session['images_captured'][camera_id]
                    }
                else:
                    results[camera_id] = {
                        'success': False,
                        'message': f'Insufficient markers: {detected_count}/{self.calib_core.markers_required}',
                        'markers_detected': detected_count
                    }
                    
            except Exception as e:
                results[camera_id] = {
                    'success': False,
                    'message': str(e)
                }
        
        return {
            'success': True,
            'cameras': results,
            'session': session
        }

    def _stop_session(self, camera_ids: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Stop a calibration session."""
        session_id = params.get('session_id')
        
        if not session_id or session_id not in self.capture_sessions:
            return {
                'success': False,
                'message': 'Invalid or missing session_id'
            }
        
        session = self.capture_sessions[session_id]
        del self.capture_sessions[session_id]
        
        return {
            'success': True,
            'message': 'Session stopped',
            'session_summary': {
                'session_id': session_id,
                'images_captured': session['images_captured'],
                'session_dir': session['session_dir']
            }
        }

    def _run_calibration(self, camera_ids: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run camera calibration on captured images.

        Args:
            camera_ids: List of camera IDs
            params: Parameters including session_id

        Returns:
            Calibration results
        """
        session_id = params.get('session_id')
        
        if not session_id:
            return {
                'success': False,
                'message': 'Missing session_id'
            }
        
        session_dir = self.calibration_data_dir / session_id
        if not session_dir.exists():
            return {
                'success': False,
                'message': f'Session directory not found: {session_id}'
            }
        
        # Load session metadata if available
        calibration_type = params.get('calibration_type', 'single')
        fisheye_model = params.get('fisheye_model', False)
        
        if calibration_type == 'single' or len(camera_ids) == 1:
            return self._calibrate_single_camera(camera_ids[0], session_dir, fisheye_model)
        elif calibration_type == 'stereo' and len(camera_ids) == 2:
            return self._calibrate_stereo(camera_ids, session_dir, fisheye_model)
        else:
            return {
                'success': False,
                'message': 'Invalid calibration configuration'
            }

    def _calibrate_single_camera(self, camera_id: str, session_dir: Path, 
                                 fisheye: bool = False) -> Dict[str, Any]:
        """
        Calibrate a single camera.

        Args:
            camera_id: Camera ID
            session_dir: Session directory path
            fisheye: Use fisheye camera model

        Returns:
            Calibration results
        """
        # Use CalibrationCore to perform calibration from directory
        result = self.calib_core.calibrate_from_directory(str(session_dir / camera_id), fisheye)
        
        if not result.get('success'):
            return result
        
        # Save calibration
        calibration_file = session_dir / f'{camera_id}_calibration.json'
        self.calib_core.save_calibration(result, str(calibration_file))
        
        return {
            'success': True,
            'camera_id': camera_id,
            'reprojection_error': result['reprojection_error'],
            'num_images': result['num_images'],
            'calibration_file': str(calibration_file),
            'calibration_data': result
        }

    def _calibrate_stereo(self, camera_ids: List[str], session_dir: Path, 
                         fisheye: bool = False) -> Dict[str, Any]:
        """
        Calibrate stereo camera pair.

        Args:
            camera_ids: List of two camera IDs
            session_dir: Session directory path
            fisheye: Use fisheye camera model

        Returns:
            Stereo calibration results
        """
        # First calibrate individual cameras
        results = {}
        for camera_id in camera_ids:
            result = self._calibrate_single_camera(camera_id, session_dir, fisheye)
            if not result['success']:
                return {
                    'success': False,
                    'message': f'Failed to calibrate {camera_id}: {result.get("message")}'
                }
            results[camera_id] = result
        
        # TODO: Implement stereo calibration
        # This requires synchronized image pairs and stereo calibration algorithm
        
        return {
            'success': True,
            'message': 'Individual camera calibrations completed. Stereo calibration pending implementation.',
            'cameras': results
        }

    def cleanup(self):
        """Cleanup plugin resources."""
        self.capture_sessions.clear()
