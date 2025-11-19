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
        
        # Calibration data storage
        self.data_dir = Path("data/calibration")
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
        
        # Interpolate ChArUco corners
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, self.board
        )
        
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
            
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, self.board
            )
            
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
            # Standard pinhole calibration
            retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                all_charuco_corners,
                all_charuco_ids,
                self.board,
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
            name: Calibration file name (with or without extension)
            
        Returns:
            Calibration data
        """
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
