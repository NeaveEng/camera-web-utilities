"""
Camera calibration plugin using ChArUco boards.

This plugin provides camera calibration functionality using ChArUco boards
(combination of checkerboard and ArUco markers) for both monocular and stereo
camera calibration.
"""

import cv2
import numpy as np
import os
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path

from backend.features.base import FeaturePlugin


class CalibrationPlugin(FeaturePlugin):
    """Camera calibration using ChArUco boards."""

    def __init__(self):
        """Initialize calibration plugin."""
        # ArUco dictionary options
        self.available_dictionaries = {
            'DICT_4X4_50': cv2.aruco.DICT_4X4_50,
            'DICT_4X4_100': cv2.aruco.DICT_4X4_100,
            'DICT_4X4_250': cv2.aruco.DICT_4X4_250,
            'DICT_4X4_1000': cv2.aruco.DICT_4X4_1000,
            'DICT_5X5_50': cv2.aruco.DICT_5X5_50,
            'DICT_5X5_100': cv2.aruco.DICT_5X5_100,
            'DICT_5X5_250': cv2.aruco.DICT_5X5_250,
            'DICT_5X5_1000': cv2.aruco.DICT_5X5_1000,
            'DICT_6X6_50': cv2.aruco.DICT_6X6_50,
            'DICT_6X6_100': cv2.aruco.DICT_6X6_100,
            'DICT_6X6_250': cv2.aruco.DICT_6X6_250,
            'DICT_6X6_1000': cv2.aruco.DICT_6X6_1000,
            'DICT_7X7_50': cv2.aruco.DICT_7X7_50,
            'DICT_7X7_100': cv2.aruco.DICT_7X7_100,
            'DICT_7X7_250': cv2.aruco.DICT_7X7_250,
            'DICT_7X7_1000': cv2.aruco.DICT_7X7_1000,
            'DICT_ARUCO_ORIGINAL': cv2.aruco.DICT_ARUCO_ORIGINAL
        }
        
        # Default dictionary
        self.dictionary_name = 'DICT_6X6_100'
        self.dictionary = cv2.aruco.getPredefinedDictionary(self.available_dictionaries[self.dictionary_name])
        self.parameters = cv2.aruco.DetectorParameters()
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        
        # Default ChArUco board configuration
        self.board_size = (8, 5)  # (width, height) in markers
        self.square_length = 0.05  # 50mm
        self.marker_length = 0.037  # 37mm
        self.board = None
        self._init_board()
        
        # Calibration data storage - use absolute path
        # Get the absolute path to this file, then navigate to data/calibration
        self.data_dir = Path(__file__).resolve().parent.parent.parent.parent / 'data' / 'calibration'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Session storage
        self.current_session = None
        self.captured_images = {}  # {camera_id: [image_paths]}
        self.calibration_results = {}  # {camera_id: calibration_data}

    def _init_board(self):
        """Initialize ChArUco board with current configuration."""
        self.board = cv2.aruco.CharucoBoard(
            self.board_size,
            self.square_length,
            self.marker_length,
            self.dictionary
        )
        self.markers_total = self.board_size[0] * self.board_size[1] // 2
        self.markers_required = int(self.markers_total * 0.9)

    def get_metadata(self) -> Dict[str, Any]:
        """Get plugin metadata."""
        return {
            "name": "Camera Calibration",
            "version": "1.0.0",
            "description": "Calibrate cameras using ChArUco boards for accurate distortion correction",
            "author": "NeaveEng",
            "requires_cameras": 1,
            "requires_high_res": True,
            "supports_groups": False
        }

    def get_ui_schema(self) -> Dict[str, Any]:
        """Get UI schema for frontend rendering."""
        return {
            "controls": [
                {
                    "name": "board_width",
                    "type": "range",
                    "label": "Board Width (markers)",
                    "min": 4,
                    "max": 12,
                    "default": 8,
                    "step": 1
                },
                {
                    "name": "board_height",
                    "type": "range",
                    "label": "Board Height (markers)",
                    "min": 3,
                    "max": 10,
                    "default": 5,
                    "step": 1
                },
                {
                    "name": "square_length",
                    "type": "number",
                    "label": "Square Length (mm)",
                    "min": 10,
                    "max": 200,
                    "default": 50,
                    "step": 0.1
                },
                {
                    "name": "marker_length",
                    "type": "number",
                    "label": "Marker Length (mm)",
                    "min": 5,
                    "max": 150,
                    "default": 37,
                    "step": 0.1
                }
            ],
            "actions": [
                "start_session",
                "capture_image",
                "detect_pattern",
                "run_calibration",
                "save_calibration",
                "load_calibration"
            ]
        }

    def configure_board(self, width: int, height: int, 
                       square_length: float, marker_length: float,
                       dictionary: Optional[str] = None) -> Dict[str, Any]:
        """
        Configure ChArUco board parameters.
        
        Args:
            width: Number of markers horizontally
            height: Number of markers vertically
            square_length: Length of each square in meters
            marker_length: Length of ArUco markers in meters
            dictionary: ArUco dictionary name (e.g., 'DICT_6X6_100')
            
        Returns:
            Success status and board configuration
        """
        # Update dictionary if specified
        if dictionary and dictionary in self.available_dictionaries:
            self.dictionary_name = dictionary
            self.dictionary = cv2.aruco.getPredefinedDictionary(self.available_dictionaries[dictionary])
            self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        
        self.board_size = (width, height)
        self.square_length = square_length / 1000.0  # Convert mm to meters
        self.marker_length = marker_length / 1000.0
        self._init_board()
        
        return {
            "success": True,
            "board_config": {
                "size": self.board_size,
                "square_length_m": self.square_length,
                "marker_length_m": self.marker_length,
                "dictionary": self.dictionary_name,
                "markers_total": self.markers_total,
                "markers_required": self.markers_required
            }
        }

    def start_calibration_session(self, camera_id: str, 
                                  config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Start a new calibration session.
        
        Args:
            camera_id: Camera identifier
            config: Optional board configuration
            
        Returns:
            Session information
        """
        # Create session directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.data_dir / f"session_{timestamp}"
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Create camera-specific subdirectory
        camera_dir = session_dir / camera_id
        camera_dir.mkdir(exist_ok=True)
        
        self.current_session = {
            "id": timestamp,
            "camera_id": camera_id,
            "session_dir": str(session_dir),
            "camera_dir": str(camera_dir),
            "start_time": datetime.now().isoformat(),
            "board_config": {
                "size": self.board_size,
                "square_length": self.square_length,
                "marker_length": self.marker_length,
                "dictionary": self.dictionary_name
            }
        }
        
        # Initialize image storage for this camera
        self.captured_images[camera_id] = []
        
        # Save session metadata
        session_file = session_dir / "session.json"
        with open(session_file, 'w') as f:
            json.dump(self.current_session, f, indent=2)
        
        return {
            "success": True,
            "session": self.current_session
        }

    def detect_charuco_pattern(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect ChArUco pattern in an image.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Detection results with corners and IDs
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect ArUco markers
        marker_corners, marker_ids, rejected = self.detector.detectMarkers(gray)
        
        if marker_ids is None or len(marker_corners) == 0:
            return {
                "detected": False,
                "markers_detected": 0,
                "markers_found": 0,
                "markers_required": self.markers_required,
                "corners_detected": 0,
                "quality": "No markers detected"
            }
        
        # Interpolate ChArUco corners (OpenCV 4.7+ compatible)
        charuco_detector = cv2.aruco.CharucoDetector(self.board)
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
        retval = len(charuco_corners) if charuco_corners is not None else 0
        
        detected = retval > 0 and len(marker_corners) >= self.markers_required
        quality = len(marker_corners) / self.markers_total if marker_corners is not None else 0.0
        
        result = {
            "detected": detected,
            "markers_detected": len(marker_corners) if marker_corners is not None else 0,
            "markers_found": len(marker_corners) if marker_corners is not None else 0,  # Keep for backwards compat
            "markers_required": self.markers_required,
            "corners_detected": retval if retval > 0 else 0,
            "charuco_corners": retval if retval > 0 else 0,  # Keep for backwards compat
            "quality": f"{quality:.2f}" if quality > 0 else "No pattern",
            "marker_corners": marker_corners,
            "marker_ids": marker_ids,
            "charuco_corners_array": charuco_corners if retval > 0 else None,
            "charuco_ids_array": charuco_ids if retval > 0 else None
        }
        
        return result

    def capture_calibration_image(self, camera_backend, camera_id: str,
                                  image: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Capture and save a calibration image.
        
        Args:
            camera_backend: Camera backend instance
            camera_id: Camera identifier
            image: Optional pre-captured image (if None, will grab from camera)
            
        Returns:
            Capture result with detection information
        """
        if self.current_session is None:
            return {
                "success": False,
                "error": "No active calibration session. Call start_calibration_session first."
            }
        
        # Get image from camera if not provided
        if image is None:
            frame_data = camera_backend.get_frame(camera_id)
            if frame_data is None:
                return {
                    "success": False,
                    "error": "Failed to capture frame from camera"
                }
            image = frame_data
        
        # Detect pattern
        detection = self.detect_charuco_pattern(image)
        
        if not detection["detected"]:
            return {
                "success": False,
                "error": "ChArUco pattern not detected or insufficient markers",
                "detection": detection
            }
        
        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"calibration_{timestamp}.png"
        filepath = Path(self.current_session["camera_dir"]) / filename
        
        cv2.imwrite(str(filepath), image)
        
        # Store image info
        image_info = {
            "path": str(filepath),
            "filename": filename,
            "timestamp": timestamp,
            "detection": {
                "markers_found": detection["markers_found"],
                "charuco_corners": detection["charuco_corners"],
                "quality": detection["quality"]
            }
        }
        
        if camera_id not in self.captured_images:
            self.captured_images[camera_id] = []
        self.captured_images[camera_id].append(image_info)
        
        return {
            "success": True,
            "image": image_info,
            "total_images": len(self.captured_images[camera_id]),
            "detection": detection
        }

    def calibrate_camera(self, camera_id: str, fisheye: bool = False,
                        image_paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run camera calibration using captured images.
        
        Args:
            camera_id: Camera identifier
            fisheye: Whether to use fisheye calibration model
            image_paths: Optional list of image paths (uses session images if None)
            
        Returns:
            Calibration results
        """
        # Get images to process
        if image_paths is None:
            if camera_id not in self.captured_images or len(self.captured_images[camera_id]) == 0:
                return {
                    "success": False,
                    "error": "No captured images available for calibration"
                }
            image_paths = [img["path"] for img in self.captured_images[camera_id]]
        
        # Process images to extract corners
        all_charuco_corners = []
        all_charuco_ids = []
        valid_images = []
        ignored_count = 0
        
        for image_path in image_paths:
            image = cv2.imread(image_path)
            if image is None:
                ignored_count += 1
                continue
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            marker_corners, marker_ids, _ = self.detector.detectMarkers(gray)
            
            if marker_ids is None or len(marker_corners) < self.markers_required:
                ignored_count += 1
                continue
            
            # Use CharucoDetector for OpenCV 4.7+ compatibility
            charuco_detector = cv2.aruco.CharucoDetector(self.board)
            charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)
            retval = len(charuco_corners) if charuco_corners is not None else 0
            
            if retval > 0:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
                valid_images.append(image_path)
            else:
                ignored_count += 1
        
        if len(all_charuco_corners) < 3:
            return {
                "success": False,
                "error": f"Insufficient valid images. Found {len(all_charuco_corners)}, need at least 3",
                "ignored_images": ignored_count
            }
        
        # Get image size from first valid image
        sample_image = cv2.imread(valid_images[0], cv2.IMREAD_GRAYSCALE)
        image_size = sample_image.shape[:2][::-1]  # (width, height)
        
        # Perform calibration
        if fisheye:
            # Fisheye calibration
            K = np.zeros((3, 3))
            D = np.zeros((4, 1))
            
            object_points = []
            image_points = []
            
            for corners, ids in zip(all_charuco_corners, all_charuco_ids):
                if len(corners) > 0:
                    obj_pts = self.board.getChessboardCorners()[ids.flatten()]
                    object_points.append(obj_pts)
                    image_points.append(corners)
            
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            image_count = len(all_charuco_corners)
            rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(image_count)]
            tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(image_count)]
            
            calibration_flags = (
                cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
                cv2.fisheye.CALIB_CHECK_COND +
                cv2.fisheye.CALIB_FIX_SKEW
            )
            
            retval, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                object_points,
                image_points,
                image_size,
                K,
                D,
                rvecs,
                tvecs,
                calibration_flags,
                criteria
            )
            
            result = {
                "success": True,
                "model": "fisheye",
                "reprojection_error": float(retval),
                "camera_matrix": K.tolist(),
                "distortion_coeffs": D.flatten().tolist(),
                "image_size": image_size,
                "images_used": len(all_charuco_corners),
                "images_ignored": ignored_count,
                "rvecs": [r.tolist() for r in rvecs],
                "tvecs": [t.tolist() for t in tvecs]
            }
            
        else:
            # Standard pinhole calibration (OpenCV 4.7+ compatible)
            # Get all possible ChArUco corner positions
            all_obj_corners = self.board.getChessboardCorners()
            
            obj_points = []
            img_points = []
            for corners, ids in zip(all_charuco_corners, all_charuco_ids):
                # Map detected corner IDs to their 3D positions
                ids_flat = ids.flatten()
                obj_pts = all_obj_corners[ids_flat]
                obj_points.append(obj_pts)
                img_points.append(corners)
            
            retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                obj_points,
                img_points,
                image_size,
                None,
                None
            )
            
            result = {
                "success": True,
                "model": "pinhole",
                "reprojection_error": float(retval),
                "camera_matrix": camera_matrix.tolist(),
                "distortion_coeffs": dist_coeffs.flatten().tolist(),
                "image_size": image_size,
                "images_used": len(all_charuco_corners),
                "images_ignored": ignored_count,
                "rvecs": [r.tolist() for r in rvecs],
                "tvecs": [t.tolist() for t in tvecs]
            }
        
        # Store results
        self.calibration_results[camera_id] = result
        
        return result

    def save_calibration(self, camera_id: str, name: Optional[str] = None,
                        include_images: bool = False) -> Dict[str, Any]:
        """
        Save calibration results to disk.
        
        Args:
            camera_id: Camera identifier
            name: Optional custom name for calibration file
            include_images: Whether to keep calibration images
            
        Returns:
            Save status
        """
        if camera_id not in self.calibration_results:
            return {
                "success": False,
                "error": "No calibration results available for this camera"
            }
        
        result = self.calibration_results[camera_id]
        
        # Generate filename
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"{camera_id}_calibration_{timestamp}"
        
        # Save as NPZ (NumPy format)
        npz_path = self.data_dir / f"{name}.npz"
        np.savez(
            npz_path,
            camera_matrix=np.array(result["camera_matrix"]),
            distortion_coeffs=np.array(result["distortion_coeffs"]),
            image_size=np.array(result["image_size"]),
            reprojection_error=result["reprojection_error"],
            model=result["model"]
        )
        
        # Save as JSON (human-readable)
        json_path = self.data_dir / f"{name}.json"
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Optionally clean up images
        if not include_images and self.current_session:
            camera_dir = Path(self.current_session["camera_dir"])
            if camera_dir.exists():
                import shutil
                shutil.rmtree(camera_dir)
        
        return {
            "success": True,
            "npz_path": str(npz_path),
            "json_path": str(json_path),
            "calibration_name": name
        }

    def load_calibration(self, name: str) -> Dict[str, Any]:
        """
        Load calibration from disk.
        
        Args:
            name: Calibration file name (with or without extension) or path like "session_xxx/camera_id" or "panorama_xxx/file.json"
            
        Returns:
            Calibration data
        """
        # Check if name contains a path separator (session/camera format or panorama/file format)
        if '/' in name:
            # First try the exact path (for panorama session calibration files)
            direct_path = self.data_dir / name
            print(f"[CalibrationPlugin] Loading calibration: name={name}")
            print(f"[CalibrationPlugin] data_dir={self.data_dir}")
            print(f"[CalibrationPlugin] direct_path={direct_path}")
            print(f"[CalibrationPlugin] exists={direct_path.exists()}")
            
            if direct_path.exists():
                try:
                    with open(direct_path, 'r') as f:
                        return {
                            "success": True,
                            **json.load(f)
                        }
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Error loading calibration from {direct_path}: {str(e)}"
                    }
            
            # Handle session/camera_id format (session_xxx/camera_id/calibration_results.json)
            session_path = self.data_dir / name / "calibration_results.json"
            if session_path.exists():
                try:
                    with open(session_path, 'r') as f:
                        return {
                            "success": True,
                            **json.load(f)
                        }
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Error loading calibration from {session_path}: {str(e)}"
                    }
            else:
                return {
                    "success": False,
                    "error": f"Calibration file not found: {direct_path} or {session_path}"
                }
        
        # Try NPZ first
        npz_path = self.data_dir / f"{name}.npz"
        json_path = self.data_dir / f"{name}.json"
        
        if npz_path.exists():
            data = np.load(npz_path, allow_pickle=True)
            return {
                "success": True,
                "camera_matrix": data["camera_matrix"].tolist(),
                "distortion_coeffs": data["distortion_coeffs"].tolist(),
                "image_size": data["image_size"].tolist() if "image_size" in data else None,
                "reprojection_error": float(data["reprojection_error"]) if "reprojection_error" in data else None,
                "model": str(data["model"]) if "model" in data else "unknown"
            }
        elif json_path.exists():
            with open(json_path, 'r') as f:
                return {
                    "success": True,
                    **json.load(f)
                }
        else:
            return {
                "success": False,
                "error": f"Calibration file not found: {name}"
            }

    def undistort_image(self, image: np.ndarray, camera_id: str,
                       balance: float = 1.0) -> Optional[np.ndarray]:
        """
        Undistort an image using calibration results.
        
        Args:
            image: Input distorted image
            camera_id: Camera identifier
            balance: Balance factor (0=preserve all pixels, 1=maximize FOV)
            
        Returns:
            Undistorted image or None if calibration not available
        """
        if camera_id not in self.calibration_results:
            return None
        
        result = self.calibration_results[camera_id]
        K = np.array(result["camera_matrix"])
        D = np.array(result["distortion_coeffs"])
        
        img_size = image.shape[:2][::-1]
        
        if result["model"] == "fisheye":
            # Fisheye undistortion
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D, img_size, np.eye(3), balance=balance
            )
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K, D, np.eye(3), new_K, img_size, cv2.CV_16SC2
            )
            undistorted = cv2.remap(image, map1, map2, 
                                   interpolation=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT)
        else:
            # Standard undistortion
            undistorted = cv2.undistort(image, K, D)
        
        return undistorted

    def process_frames(self, camera_backend, camera_ids: List[str],
                      params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process frames for calibration feature.
        
        Args:
            camera_backend: Camera backend instance
            camera_ids: List of camera IDs
            params: Optional parameters (action, config, etc.)
            
        Returns:
            Processing results
        """
        if params is None:
            params = {}
        
        action = params.get("action", "detect")
        camera_id = camera_ids[0] if camera_ids else None
        
        if action == "detect":
            # Real-time pattern detection
            frame = camera_backend.get_frame(camera_id)
            if frame is None:
                return {"success": False, "error": "Failed to get frame"}
            return self.detect_charuco_pattern(frame)
        
        elif action == "capture":
            return self.capture_calibration_image(camera_backend, camera_id)
        
        elif action == "calibrate":
            fisheye = params.get("fisheye", False)
            return self.calibrate_camera(camera_id, fisheye)
        
        else:
            return {"success": False, "error": f"Unknown action: {action}"}
    
    # Utility methods
    
    @staticmethod
    def load_calibration(calibration_path: str) -> Optional[Dict[str, Any]]:
        """
        Load calibration data from a JSON file.
        
        Args:
            calibration_path: Path to calibration JSON file (can be relative like "session_xxx/0")
        
        Returns:
            Dictionary with calibration data or None on error
        """
        try:
            from pathlib import Path
            
            # Get the absolute path to data/calibration directory
            calibration_dir = Path(__file__).resolve().parent.parent.parent.parent / 'data' / 'calibration'
            
            # If path contains /, treat it as relative to data/calibration directory
            if '/' in calibration_path:
                if calibration_path.endswith('.json'):
                    # Direct path to a JSON file in a subdirectory (e.g., panorama_xxx/file.json)
                    full_path = calibration_dir / calibration_path
                    print(f"[CalibrationPlugin] Loading calibration from session file: {full_path}")
                else:
                    # Path to session/camera_id format (needs calibration_results.json appended)
                    full_path = calibration_dir / calibration_path / 'calibration_results.json'
                    print(f"[CalibrationPlugin] Loading calibration from session format: {full_path}")
            else:
                # No slash - treat as absolute path
                full_path = Path(calibration_path).resolve()
                print(f"[CalibrationPlugin] Loading calibration from absolute path: {full_path}")
            
            with open(full_path, 'r') as f:
                calibration_data = json.load(f)
            
            # Convert lists back to numpy arrays for use with OpenCV
            calibration_data['camera_matrix'] = np.array(calibration_data['camera_matrix'])
            calibration_data['distortion_coeffs'] = np.array(calibration_data['distortion_coeffs'])
            
            # Return in the expected format with success flag
            return {
                'success': True,
                **calibration_data
            }
            
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    @staticmethod
    def undistort_image(image: np.ndarray, camera_matrix: np.ndarray, 
                       dist_coeffs: np.ndarray, crop: bool = False) -> np.ndarray:
        """
        Undistort an image using calibration parameters.
        
        Args:
            image: Input image (numpy array)
            camera_matrix: Camera matrix (3x3)
            dist_coeffs: Distortion coefficients
            crop: If True, crop the image to remove black borders
        
        Returns:
            Undistorted image
        """
        h, w = image.shape[:2]
        
        if crop:
            # Get optimal camera matrix that crops the image
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                camera_matrix, dist_coeffs, (w, h), 1, (w, h)
            )
            
            # Undistort
            dst = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
            
            # Crop the image
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
        else:
            # Undistort without cropping
            dst = cv2.undistort(image, camera_matrix, dist_coeffs, None, camera_matrix)
        
        return dst
    
    @staticmethod
    def scale_calibration_for_resolution(camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                                        original_size: Tuple[int, int], 
                                        target_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale camera calibration parameters from one resolution to another.
        
        When calibration is performed at one resolution but needs to be applied to a different resolution,
        the camera matrix (focal lengths and principal point) must be scaled accordingly.
        Distortion coefficients remain the same as they are dimensionless.
        
        Args:
            camera_matrix: Original camera matrix (3x3 numpy array)
            dist_coeffs: Distortion coefficients (numpy array, unchanged by scaling)
            original_size: (width, height) tuple of the calibration resolution
            target_size: (width, height) tuple of the target resolution
        
        Returns:
            Tuple of (scaled_camera_matrix, dist_coeffs)
        """
        orig_width, orig_height = original_size
        target_width, target_height = target_size
        
        # Calculate scaling factors
        scale_x = target_width / orig_width
        scale_y = target_height / orig_height
        
        # Create a copy of the camera matrix to avoid modifying the original
        scaled_matrix = camera_matrix.copy()
        
        # Scale focal lengths (fx, fy)
        scaled_matrix[0, 0] *= scale_x  # fx
        scaled_matrix[1, 1] *= scale_y  # fy
        
        # Scale principal point (cx, cy)
        scaled_matrix[0, 2] *= scale_x  # cx
        scaled_matrix[1, 2] *= scale_y  # cy
        
        # Distortion coefficients are dimensionless and don't need scaling
        return scaled_matrix, dist_coeffs
    
    @staticmethod
    def save_calibration_results(results: Dict[str, Any], output_path: str, 
                                calibration_name: str) -> Dict[str, Any]:
        """
        Save calibration results to a file.
        
        Args:
            results: Calibration results dictionary
            output_path: Directory path to save results
            calibration_name: Name for the calibration file
        
        Returns:
            Dictionary with success status and file path
        """
        try:
            os.makedirs(output_path, exist_ok=True)
            file_path = os.path.join(output_path, f'{calibration_name}.json')
            
            with open(file_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            return {
                'success': True,
                'file_path': file_path
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def stereo_calibrate_from_sessions(self, camera1_path: str, camera2_path: str, 
                                      board_config: Dict[str, Any],
                                      camera1_calib: Dict[str, Any], 
                                      camera2_calib: Dict[str, Any],
                                      flags: Optional[int] = None,
                                      progress_callback = None) -> Dict[str, Any]:
        """
        Perform stereo calibration using synchronized ChArUco board captures from two cameras.
        
        This computes the extrinsic relationship (rotation R and translation T) between two cameras
        using the same ChArUco board visible in both cameras simultaneously.
        
        Args:
            camera1_path: Path to camera 1 calibration session directory
            camera2_path: Path to camera 2 calibration session directory
            board_config: ChArUco board configuration dict
            camera1_calib: Individual calibration for camera 1 (camera_matrix, distortion_coeffs)
            camera2_calib: Individual calibration for camera 2 (camera_matrix, distortion_coeffs)
            flags: Stereo calibration flags (default: cv2.CALIB_FIX_INTRINSIC for fixed intrinsics)
            progress_callback: Optional callback for progress updates
        
        Returns:
            Dictionary with stereo calibration results including R, T, E, F matrices
        """
        def log_progress(message):
            if progress_callback:
                progress_callback(message)
        
        try:
            log_progress('Loading synchronized image pairs...')
            
            # Load matching images from both cameras (assuming same filenames)
            cam1_images = sorted(Path(camera1_path).glob("image_*.jpg"))
            cam2_images = sorted(Path(camera2_path).glob("image_*.jpg"))
            
            # Match images by filename
            cam1_dict = {img.name: img for img in cam1_images}
            cam2_dict = {img.name: img for img in cam2_images}
            common_names = set(cam1_dict.keys()) & set(cam2_dict.keys())
            
            if len(common_names) < 5:
                return {
                    'success': False,
                    'error': f'Insufficient synchronized image pairs. Found {len(common_names)}, need at least 5.',
                    'cam1_images': len(cam1_images),
                    'cam2_images': len(cam2_images),
                    'matched_pairs': len(common_names)
                }
            
            log_progress(f'Found {len(common_names)} synchronized image pairs')
            
            # Create ChArUco board
            log_progress('Creating ChArUco board definition...')
            aruco_dict = cv2.aruco.getPredefinedDictionary(
                getattr(cv2.aruco, board_config['dictionary'])
            )
            
            board = cv2.aruco.CharucoBoard(
                (board_config['width'], board_config['height']),
                board_config['square_length'],
                board_config['marker_length'],
                aruco_dict
            )
            
            # Detect ChArUco corners in all synchronized pairs
            log_progress('Detecting ChArUco patterns in synchronized pairs...')
            object_points = []  # 3D points in board coordinate system
            image_points1 = []  # 2D points in camera 1
            image_points2 = []  # 2D points in camera 2
            image_size = None
            processed_pairs = []
            
            detector_params = cv2.aruco.DetectorParameters()
            
            for i, img_name in enumerate(sorted(common_names)):
                log_progress(f'Processing pair {i+1}/{len(common_names)}: {img_name}')
                
                img1 = cv2.imread(str(cam1_dict[img_name]))
                img2 = cv2.imread(str(cam2_dict[img_name]))
                
                if img1 is None or img2 is None:
                    continue
                
                if image_size is None:
                    image_size = img1.shape[:2][::-1]  # (width, height)
                
                gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                
                # Detect in camera 1 (OpenCV 4.7+ compatible)
                aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
                corners1, ids1, _ = aruco_detector.detectMarkers(gray1)
                charuco_corners1, charuco_ids1 = None, None
                if ids1 is not None and len(ids1) > 0:
                    charuco_detector = cv2.aruco.CharucoDetector(board)
                    charuco_corners1, charuco_ids1, _, _ = charuco_detector.detectBoard(gray1)
                ret1 = len(charuco_corners1) if charuco_corners1 is not None else 0
                
                # Detect in camera 2
                corners2, ids2, _ = aruco_detector.detectMarkers(gray2)
                charuco_corners2, charuco_ids2 = None, None
                if ids2 is not None and len(ids2) > 0:
                    charuco_corners2, charuco_ids2, _, _ = charuco_detector.detectBoard(gray2)
                ret2 = len(charuco_corners2) if charuco_corners2 is not None else 0
                
                # Only use pairs where both cameras detected sufficient corners
                if (ret1 and ret2 and charuco_corners1 is not None and charuco_corners2 is not None and
                    len(charuco_corners1) >= 4 and len(charuco_corners2) >= 4):
                    
                    # Find common corner IDs between both cameras
                    ids1_flat = charuco_ids1.flatten()
                    ids2_flat = charuco_ids2.flatten()
                    common_ids = np.intersect1d(ids1_flat, ids2_flat)
                    
                    if len(common_ids) >= 4:  # Need at least 4 common points
                        # Extract matching corners
                        matched_corners1 = []
                        matched_corners2 = []
                        matched_obj_points = []
                        
                        for corner_id in common_ids:
                            idx1 = np.where(ids1_flat == corner_id)[0][0]
                            idx2 = np.where(ids2_flat == corner_id)[0][0]
                            matched_corners1.append(charuco_corners1[idx1])
                            matched_corners2.append(charuco_corners2[idx2])
                            # Get 3D position of this corner on the board
                            matched_obj_points.append(board.getChessboardCorners()[corner_id])
                        
                        object_points.append(np.array(matched_obj_points, dtype=np.float32))
                        image_points1.append(np.array(matched_corners1, dtype=np.float32))
                        image_points2.append(np.array(matched_corners2, dtype=np.float32))
                        
                        processed_pairs.append({
                            'filename': img_name,
                            'common_corners': len(common_ids),
                            'cam1_total': len(charuco_corners1),
                            'cam2_total': len(charuco_corners2)
                        })
            
            if len(object_points) < 3:
                return {
                    'success': False,
                    'error': f'Not enough valid stereo pairs for calibration. Found {len(object_points)} pairs with common corners, need at least 3.',
                    'pairs_processed': len(object_points),
                    'pairs_total': len(common_names)
                }
            
            log_progress(f'Found {len(object_points)} valid stereo pairs with matching corners')
            
            # Extract camera matrices and distortion coefficients
            camera_matrix1 = np.array(camera1_calib['camera_matrix'])
            dist_coeffs1 = np.array(camera1_calib['distortion_coeffs'])
            camera_matrix2 = np.array(camera2_calib['camera_matrix'])
            dist_coeffs2 = np.array(camera2_calib['distortion_coeffs'])
            
            # Set stereo calibration flags
            if flags is None:
                # Default: fix intrinsics since we already have individual calibrations
                # For divergent cameras with minimal overlap, we want to trust individual calibrations
                flags = cv2.CALIB_FIX_INTRINSIC
            
            log_progress('Computing stereo calibration (rotation and translation between cameras)...')
            
            # Perform stereo calibration
            retval, camera_matrix1_out, dist_coeffs1_out, camera_matrix2_out, dist_coeffs2_out, R, T, E, F = \
                cv2.stereoCalibrate(
                    object_points,
                    image_points1,
                    image_points2,
                    camera_matrix1,
                    dist_coeffs1,
                    camera_matrix2,
                    dist_coeffs2,
                    image_size,
                    flags=flags
                )
            
            if not retval or retval > 100:  # High error indicates failure
                return {
                    'success': False,
                    'error': f'Stereo calibration failed with high reprojection error: {retval:.2f} pixels',
                    'pairs_used': len(object_points)
                }
            
            log_progress('Computing rectification transforms...')
            
            # Compute rectification for aligned epipolar lines
            # This is useful for stereo matching but optional for panorama stitching
            R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                camera_matrix1,
                dist_coeffs1,
                camera_matrix2,
                dist_coeffs2,
                image_size,
                R,
                T,
                flags=cv2.CALIB_ZERO_DISPARITY,
                alpha=0  # 0 = crop to valid pixels only
            )
            
            log_progress('Stereo calibration complete!')
            
            # Return comprehensive results
            return {
                'success': True,
                'reprojection_error': float(retval),
                'rotation_matrix': R.tolist(),
                'translation_vector': T.flatten().tolist(),
                'essential_matrix': E.tolist(),
                'fundamental_matrix': F.tolist(),
                'rectification': {
                    'R1': R1.tolist(),
                    'R2': R2.tolist(),
                    'P1': P1.tolist(),
                    'P2': P2.tolist(),
                    'Q': Q.tolist(),
                    'roi1': roi1,
                    'roi2': roi2
                },
                'camera_matrices': {
                    'camera1': camera_matrix1.tolist(),
                    'camera2': camera_matrix2.tolist()
                },
                'distortion_coeffs': {
                    'camera1': dist_coeffs1.flatten().tolist(),
                    'camera2': dist_coeffs2.flatten().tolist()
                },
                'pairs_used': len(object_points),
                'pairs_total': len(common_names),
                'image_size': list(image_size),
                'pair_details': processed_pairs,
                'board_config': board_config
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': f'Stereo calibration error: {str(e)}'
            }
    
    @staticmethod
    def compute_optimal_homography_from_stereo(stereo_calib: Dict[str, Any], 
                                              image_size: Tuple[int, int]) -> np.ndarray:
        """
        Compute optimal homography for panorama stitching from stereo calibration.
        
        For side-by-side cameras, the homography can be computed from the rotation matrix
        and camera intrinsics, providing a more geometrically accurate transformation than
        feature-based homography estimation.
        
        Args:
            stereo_calib: Stereo calibration results dict with R and camera matrices
            image_size: (width, height) tuple
        
        Returns:
            3x3 homography matrix as numpy array
        """
        R = np.array(stereo_calib['rotation_matrix'])
        K1 = np.array(stereo_calib['camera_matrices']['camera1'])
        K2 = np.array(stereo_calib['camera_matrices']['camera2'])
        
        # For pure rotation (panorama case), H = K2 * R * K1^-1
        # This assumes cameras are at the same position (rotation only)
        # For translation, this is an approximation that works well when scene is far
        H = K2 @ R @ np.linalg.inv(K1)
        
        # Normalize
        H = H / H[2, 2]
        
        return H

