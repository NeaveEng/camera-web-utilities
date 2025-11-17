#!/usr/bin/env python3
"""
Camera calibration utilities.

Platform-agnostic calibration module adapted from ne-calib-utils.
Uses CameraBackend abstraction instead of picamera2 for multi-platform support.
"""

import cv2
import numpy as np
import time
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import fnmatch


class CalibrationCore:
    """Core calibration functionality using ChArUco boards."""
    
    def __init__(self):
        """Initialize calibration with ChArUco board configuration."""
        self.init_aruco_board()
        
        self.marker_colour = (0, 0, 255)  # BGR so red
        
        # Calibration results
        self.left_calibration = None
        self.right_calibration = None
        self.left_fisheye_calibration = None
        self.right_fisheye_calibration = None
        self.stereo_calibration = None
        
        self.rectification_alpha = 0.95
        
    def init_aruco_board(self):
        """Initialize ArUco board detector and parameters."""
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)
        self.parameters = cv2.aruco.DetectorParameters()
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        
        self.marker_dimensions = (8, 5)
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        self.board = cv2.aruco.CharucoBoard(
            self.marker_dimensions, 
            0.05,  # Square size in meters
            0.037,  # Marker size in meters
            self.dictionary
        )
        
        self.markers_total = self.marker_dimensions[0] * self.marker_dimensions[1] // 2
        self.markers_required = int(self.markers_total * 0.9)
        print(f"Markers total: {self.markers_total}, markers required: {self.markers_required}")
    
    def show_markers(self, frame: np.ndarray, corners, ids, scale: int = 1) -> np.ndarray:
        """
        Draw detected markers on frame.
        
        Args:
            frame: Input frame
            corners: Detected marker corners
            ids: Detected marker IDs
            scale: Scale factor for resizing
            
        Returns:
            Frame with markers drawn
        """
        if corners is not None and len(corners) > self.markers_required:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids, borderColor=self.marker_colour)
        
        if scale > 1:
            frame = cv2.resize(frame, (frame.shape[1] // scale, frame.shape[0] // scale))
        return frame
    
    def detect_markers(self, frame: np.ndarray) -> Tuple:
        """
        Detect ArUco markers in frame.
        
        Args:
            frame: Input frame (BGR or RGB)
            
        Returns:
            Tuple of (corners, ids, rejected_points)
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        corners, ids, rejected = self.detector.detectMarkers(gray)
        return corners, ids, rejected
    
    def calibrate_camera(self, images: List[np.ndarray], fisheye: bool = False) -> Dict[str, Any]:
        """
        Calibrate a single camera using captured images.
        
        Args:
            images: List of calibration images
            fisheye: Use fisheye camera model
            
        Returns:
            Dictionary containing calibration results
        """
        all_charuco_corners = []
        all_charuco_ids = []
        ignored_images = 0
        image_size = None
        
        for idx, image in enumerate(images):
            if image_size is None:
                image_size = image.shape[:2][::-1]
            
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            marker_corners, marker_ids, _ = self.detector.detectMarkers(gray)
            
            if marker_ids is not None:
                print(f"Image {idx}: Found {len(marker_corners)} corners and {len(marker_ids)} IDs")
            
            if marker_corners is not None and len(marker_corners) >= self.markers_required:
                if len(marker_corners) != len(marker_ids):
                    print(f"Error: Marker corners and IDs are not the same length")
                    ignored_images += 1
                    continue
                
                retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    marker_corners, marker_ids, gray, self.board
                )
                
                if retval > 0:
                    all_charuco_corners.append(charuco_corners)
                    all_charuco_ids.append(charuco_ids)
                else:
                    print(f"No corners interpolated in image {idx}, detected {len(marker_corners)}, ret: {retval}")
                    ignored_images += 1
            else:
                ignored_images += 1
        
        print(f"Ignored {ignored_images} images of {len(images)}")
        print(f"Calibrating using {len(all_charuco_corners)} valid images")
        
        if len(all_charuco_corners) < 10:
            return {
                'success': False,
                'message': f'Insufficient valid images: {len(all_charuco_corners)} (minimum 10 required)'
            }
        
        if fisheye:
            return self._calibrate_fisheye(all_charuco_corners, all_charuco_ids, image_size)
        else:
            return self._calibrate_standard(all_charuco_corners, all_charuco_ids, image_size)
    
    def _calibrate_standard(self, all_charuco_corners, all_charuco_ids, image_size) -> Dict[str, Any]:
        """Standard pinhole camera calibration."""
        retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            all_charuco_corners, all_charuco_ids, self.board, image_size, None, None
        )
        
        print(f"Reprojection error: {retval}")
        print(f"Camera matrix:\n{camera_matrix}")
        print(f"Distortion coefficients:\n{dist_coeffs}")
        
        return {
            'success': True,
            'camera_matrix': camera_matrix,
            'dist_coeffs': dist_coeffs,
            'rvecs': rvecs,
            'tvecs': tvecs,
            'reprojection_error': float(retval),
            'image_size': image_size,
            'num_images': len(all_charuco_corners),
            'fisheye': False
        }
    
    def _calibrate_fisheye(self, all_charuco_corners, all_charuco_ids, image_size) -> Dict[str, Any]:
        """Fisheye camera calibration."""
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        
        object_points = []
        image_points = []
        
        for corners, ids in zip(all_charuco_corners, all_charuco_ids):
            if len(corners) > 0:
                object_points.append(self.board.getChessboardCorners()[ids])
                image_points.append(corners)
        
        image_count = len(all_charuco_corners)
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(image_count)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(image_count)]
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        retval, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            object_points,
            image_points,
            image_size,
            K,
            D,
            rvecs,
            tvecs,
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + 
            cv2.fisheye.CALIB_CHECK_COND + 
            cv2.fisheye.CALIB_FIX_SKEW,
            criteria
        )
        
        print(f"Fisheye reprojection error: {retval}")
        print(f"K:\n{K}")
        print(f"Distortion coefficients:\n{D}")
        
        return {
            'success': True,
            'camera_matrix': K,
            'dist_coeffs': D,
            'rvecs': rvecs,
            'tvecs': tvecs,
            'reprojection_error': float(retval),
            'image_size': image_size,
            'num_images': len(all_charuco_corners),
            'fisheye': True
        }
    
    def calibrate_from_directory(self, folder: str, fisheye: bool = False) -> Dict[str, Any]:
        """
        Calibrate camera from images in a directory.
        
        Args:
            folder: Path to directory containing calibration images
            fisheye: Use fisheye camera model
            
        Returns:
            Calibration results
        """
        print(f"Opening folder: {folder}")
        image_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".png")]
        image_files.sort()
        print(f"{len(image_files)} images found in {folder}")
        
        if len(image_files) == 0:
            return {
                'success': False,
                'message': 'No images found in directory'
            }
        
        # Load images
        images = []
        for image_file in image_files:
            image = cv2.imread(image_file)
            if image is not None:
                images.append(image)
        
        return self.calibrate_camera(images, fisheye)
    
    def stereo_calibrate(self, left_images: List[np.ndarray], right_images: List[np.ndarray],
                        left_calibration: Dict, right_calibration: Dict) -> Dict[str, Any]:
        """
        Perform stereo calibration on image pairs.
        
        Args:
            left_images: List of left camera images
            right_images: List of right camera images
            left_calibration: Left camera calibration data
            right_calibration: Right camera calibration data
            
        Returns:
            Stereo calibration results
        """
        if len(left_images) != len(right_images):
            return {
                'success': False,
                'message': 'Left and right image counts do not match'
            }
        
        all_corners = {'left': [], 'right': []}
        all_ids = {'left': [], 'right': []}
        
        print(f"Processing {len(left_images)} stereo image pairs")
        
        for idx, (left_image, right_image) in enumerate(zip(left_images, right_images)):
            left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY) if len(left_image.shape) == 3 else left_image
            right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY) if len(right_image.shape) == 3 else right_image
            
            # Detect markers
            left_corners, left_ids, _ = self.detector.detectMarkers(left_gray)
            right_corners, right_ids, _ = self.detector.detectMarkers(right_gray)
            
            print(f"Pair {idx}: Left corners: {len(left_corners) if left_corners is not None else 0}, " +
                  f"Right corners: {len(right_corners) if right_corners is not None else 0}")
            
            # Interpolate ChArUco corners
            if left_ids is not None:
                retval_left, charuco_corners_left, charuco_ids_left = cv2.aruco.interpolateCornersCharuco(
                    left_corners, left_ids, left_gray, self.board
                )
                if retval_left > 0:
                    all_corners['left'].append(charuco_corners_left)
                    all_ids['left'].append(charuco_ids_left)
            
            if right_ids is not None:
                retval_right, charuco_corners_right, charuco_ids_right = cv2.aruco.interpolateCornersCharuco(
                    right_corners, right_ids, right_gray, self.board
                )
                if retval_right > 0:
                    all_corners['right'].append(charuco_corners_right)
                    all_ids['right'].append(charuco_ids_right)
        
        # Match points between left and right
        matched_object_points = []
        matched_corners_left = []
        matched_corners_right = []
        
        for i in range(min(len(all_corners['left']), len(all_corners['right']))):
            # Find common IDs in both images
            common_ids = np.intersect1d(all_ids['left'][i], all_ids['right'][i])
            
            if len(common_ids) > 0:
                indices_left = np.isin(all_ids['left'][i], common_ids).flatten()
                indices_right = np.isin(all_ids['right'][i], common_ids).flatten()
                
                matched_object_points.append(self.board.getChessboardCorners()[common_ids, :])
                matched_corners_left.append(all_corners['left'][i][indices_left])
                matched_corners_right.append(all_corners['right'][i][indices_right])
        
        print(f"Matched {len(matched_corners_left)} image pairs for stereo calibration")
        
        if len(matched_corners_left) == 0:
            return {
                'success': False,
                'message': 'No matching corners found between left and right images'
            }
        
        # Get image size
        image_size = left_gray.shape[::-1]
        
        # Perform stereo calibration
        ret, camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right, R, T, E, F = cv2.stereoCalibrate(
            objectPoints=matched_object_points,
            imagePoints1=matched_corners_left,
            imagePoints2=matched_corners_right,
            cameraMatrix1=left_calibration['camera_matrix'],
            distCoeffs1=left_calibration['dist_coeffs'],
            cameraMatrix2=right_calibration['camera_matrix'],
            distCoeffs2=right_calibration['dist_coeffs'],
            imageSize=image_size,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
            flags=0
        )
        
        print("Stereo Calibration Reprojection Error:", ret)
        print("Rotation Matrix:\n", R)
        print("Translation Vector:\n", T)
        
        return {
            'success': True,
            'camera_matrix_left': camera_matrix_left,
            'dist_coeffs_left': dist_coeffs_left,
            'camera_matrix_right': camera_matrix_right,
            'dist_coeffs_right': dist_coeffs_right,
            'R': R,
            'T': T,
            'E': E,
            'F': F,
            'reprojection_error': float(ret),
            'num_pairs': len(matched_corners_left),
            'baseline': float(np.linalg.norm(T))
        }
    
    def save_calibration(self, calibration_data: Dict, filename: str):
        """
        Save calibration data to file.
        
        Args:
            calibration_data: Calibration results
            filename: Output filename (.json or .npz)
        """
        if filename.endswith('.npz'):
            # Save as numpy archive
            np.savez(filename, **{k: v for k, v in calibration_data.items() if isinstance(v, np.ndarray)})
        else:
            # Save as JSON
            json_data = {}
            for k, v in calibration_data.items():
                if isinstance(v, np.ndarray):
                    json_data[k] = v.tolist()
                else:
                    json_data[k] = v
            
            with open(filename, 'w') as f:
                json.dump(json_data, f, indent=2)
        
        print(f"Calibration saved to {filename}")
    
    def load_calibration(self, filename: str) -> Dict:
        """
        Load calibration data from file.
        
        Args:
            filename: Input filename (.json or .npz)
            
        Returns:
            Calibration data dictionary
        """
        if filename.endswith('.npz'):
            data = np.load(filename)
            return dict(data)
        else:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Convert lists back to numpy arrays
            for k, v in data.items():
                if isinstance(v, list):
                    data[k] = np.array(v)
            
            return data
    
    def rectify_images(self, left_image: np.ndarray, right_image: np.ndarray,
                      stereo_calibration: Dict, alpha: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rectify stereo image pair.
        
        Args:
            left_image: Left camera image
            right_image: Right camera image
            stereo_calibration: Stereo calibration data
            alpha: Rectification alpha (0=no invalid pixels, 1=all pixels retained)
            
        Returns:
            Tuple of (left_rectified, right_rectified)
        """
        left_camera_matrix = stereo_calibration['camera_matrix_left']
        left_dist_coeffs = stereo_calibration['dist_coeffs_left']
        right_camera_matrix = stereo_calibration['camera_matrix_right']
        right_dist_coeffs = stereo_calibration['dist_coeffs_right']
        R = stereo_calibration['R']
        T = stereo_calibration['T']
        
        image_size = left_image.shape[:2][::-1]
        
        # Stereo rectification
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            left_camera_matrix, left_dist_coeffs,
            right_camera_matrix, right_dist_coeffs,
            image_size, R, T, alpha=alpha
        )
        
        # Compute undistortion and rectification maps
        left_map1, left_map2 = cv2.initUndistortRectifyMap(
            left_camera_matrix, left_dist_coeffs, R1, P1, image_size, cv2.CV_16SC2
        )
        right_map1, right_map2 = cv2.initUndistortRectifyMap(
            right_camera_matrix, right_dist_coeffs, R2, P2, image_size, cv2.CV_16SC2
        )
        
        # Apply rectification
        left_rectified = cv2.remap(left_image, left_map1, left_map2, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_image, right_map1, right_map2, cv2.INTER_LINEAR)
        
        return left_rectified, right_rectified
    
    def undistort_image(self, image: np.ndarray, calibration: Dict, balance: float = 1.0) -> np.ndarray:
        """
        Undistort image using calibration data.
        
        Args:
            image: Input image
            calibration: Calibration data
            balance: Balance parameter for fisheye (0-1)
            
        Returns:
            Undistorted image
        """
        if calibration.get('fisheye', False):
            return self._undistort_fisheye(image, calibration, balance)
        else:
            return self._undistort_standard(image, calibration)
    
    def _undistort_standard(self, image: np.ndarray, calibration: Dict) -> np.ndarray:
        """Undistort using standard pinhole model."""
        camera_matrix = calibration['camera_matrix']
        dist_coeffs = calibration['dist_coeffs']
        
        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )
        
        undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
        
        # Crop to ROI
        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]
        
        return undistorted
    
    def _undistort_fisheye(self, image: np.ndarray, calibration: Dict, balance: float = 1.0) -> np.ndarray:
        """Undistort using fisheye model."""
        img_dim = image.shape[:2][::-1]
        K = calibration['camera_matrix']
        D = calibration['dist_coeffs']
        
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, img_dim, np.eye(3), balance=balance
        )
        
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), new_K, img_dim, cv2.CV_16SC2
        )
        
        undistorted = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
        
        return undistorted
    
    def generate_board_image(self, output_file: str = "charuco_board.png", 
                            width: int = 800, height: int = 500):
        """
        Generate ChArUco board image for printing.
        
        Args:
            output_file: Output filename
            width: Image width in pixels
            height: Image height in pixels
        """
        img = self.board.generateImage((width, height))
        cv2.imwrite(output_file, img)
        print(f"ChArUco board saved to {output_file}")
        return img
