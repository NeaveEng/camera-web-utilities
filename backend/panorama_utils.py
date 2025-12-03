"""
Panorama calibration utilities for multi-camera stitching.

Supports ChArUco-based alignment for precise correspondence matching.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def detect_charuco_for_panorama(image: np.ndarray, board_config: dict) -> Optional[dict]:
    """
    Detect ChArUco board in image and return corner data with marker IDs.
    
    Args:
        image: Input image (BGR or grayscale)
        board_config: Dictionary with board parameters (width, height, square_length, 
                     marker_length, dictionary)
    
    Returns:
        Dictionary containing:
        - corners: Detected corner positions (Nx1x2 array)
        - ids: Corner IDs (Nx1 array)
        - marker_corners: Detected marker corners
        - marker_ids: Detected marker IDs
        Or None if detection failed
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Get ArUco dictionary
    aruco_dict_name = board_config.get('dictionary', 'DICT_6X6_100')
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict_name))
    
    # Create ChArUco board
    board = cv2.aruco.CharucoBoard(
        (board_config['width'], board_config['height']),
        board_config['square_length'],
        board_config['marker_length'],
        aruco_dict
    )
    
    # Detect markers (OpenCV 4.7+ compatible)
    detector_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
    corners, ids, rejected = aruco_detector.detectMarkers(gray)
    
    if ids is None or len(ids) < 4:
        return None
    
    # Interpolate ChArUco corners (OpenCV 4.7+ compatible)
    charuco_detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)
    num_corners = len(charuco_corners) if charuco_corners is not None else 0
    
    if num_corners < 4:
        return None
    
    return {
        'corners': charuco_corners,
        'ids': charuco_ids,
        'marker_corners': corners,
        'marker_ids': ids,
        'num_corners': num_corners
    }


def match_charuco_corners(detection1: dict, detection2: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match ChArUco corners between two images based on corner IDs.
    
    Args:
        detection1: Detection result from first image
        detection2: Detection result from second image
    
    Returns:
        Tuple of (points1, points2) - matched corner positions as Nx2 arrays
    """
    ids1 = detection1['ids'].flatten()
    ids2 = detection2['ids'].flatten()
    corners1 = detection1['corners'].reshape(-1, 2)
    corners2 = detection2['corners'].reshape(-1, 2)
    
    # Find common IDs
    common_ids = np.intersect1d(ids1, ids2)
    
    if len(common_ids) < 4:
        raise ValueError(f"Insufficient matching corners: only {len(common_ids)} found, need at least 4")
    
    # Extract matched points
    points1 = []
    points2 = []
    
    for corner_id in common_ids:
        idx1 = np.where(ids1 == corner_id)[0][0]
        idx2 = np.where(ids2 == corner_id)[0][0]
        points1.append(corners1[idx1])
        points2.append(corners2[idx2])
    
    return np.array(points1, dtype=np.float32), np.array(points2, dtype=np.float32)


def compute_panorama_homography(points1: np.ndarray, points2: np.ndarray, 
                                 ransac_threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute homography between two sets of matched points.
    
    Args:
        points1: Points from first image (Nx2)
        points2: Points from second image (Nx2)
        ransac_threshold: RANSAC reprojection threshold in pixels
    
    Returns:
        Tuple of (homography_matrix, inlier_mask, quality_metrics)
    """
    if len(points1) < 4 or len(points2) < 4:
        raise ValueError("Need at least 4 point correspondences")
    
    # Compute homography with RANSAC
    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, ransac_threshold)
    
    if H is None:
        raise ValueError("Failed to compute homography")
    
    # Calculate quality metrics
    inliers = np.sum(mask)
    inlier_ratio = inliers / len(points1)
    
    # Compute reprojection error for inliers
    points1_transformed = cv2.perspectiveTransform(points1.reshape(-1, 1, 2), H)
    errors = np.linalg.norm(points1_transformed.reshape(-1, 2) - points2, axis=1)
    inlier_errors = errors[mask.ravel() == 1]
    
    metrics = {
        'total_matches': len(points1),
        'inliers': int(inliers),
        'inlier_ratio': float(inlier_ratio),
        'mean_error': float(np.mean(inlier_errors)),
        'max_error': float(np.max(inlier_errors)),
        'rmse': float(np.sqrt(np.mean(inlier_errors**2)))
    }
    
    return H, mask, metrics


def calibrate_panorama_pair(image1: np.ndarray, image2: np.ndarray, 
                            board_config: dict,
                            camera1_calibration: Optional[dict] = None,
                            camera2_calibration: Optional[dict] = None) -> dict:
    """
    Calibrate panorama alignment between two cameras using ChArUco board.
    Single capture version - use calibrate_panorama_multiple for better results.
    
    Args:
        image1: Image from first camera
        image2: Image from second camera
        board_config: ChArUco board configuration
        camera1_calibration: Optional individual calibration for camera 1
        camera2_calibration: Optional individual calibration for camera 2
    
    Returns:
        Dictionary with calibration results:
        - homography: 3x3 transformation matrix
        - metrics: Quality metrics
        - matches: Number of matched corners
        - success: Boolean indicating success
    """
    # Undistort images if calibrations provided
    if camera1_calibration:
        camera_matrix1 = np.array(camera1_calibration['camera_matrix'])
        dist_coeffs1 = np.array(camera1_calibration['distortion_coeffs'])
        h, w = image1.shape[:2]
        new_camera_matrix1, roi1 = cv2.getOptimalNewCameraMatrix(
            camera_matrix1, dist_coeffs1, (w, h), alpha=1.0
        )
        image1 = cv2.undistort(image1, camera_matrix1, dist_coeffs1, None, new_camera_matrix1)
    
    if camera2_calibration:
        camera_matrix2 = np.array(camera2_calibration['camera_matrix'])
        dist_coeffs2 = np.array(camera2_calibration['distortion_coeffs'])
        h, w = image2.shape[:2]
        new_camera_matrix2, roi2 = cv2.getOptimalNewCameraMatrix(
            camera_matrix2, dist_coeffs2, (w, h), alpha=1.0
        )
        image2 = cv2.undistort(image2, camera_matrix2, dist_coeffs2, None, new_camera_matrix2)
    
    # Detect ChArUco in both images
    detection1 = detect_charuco_for_panorama(image1, board_config)
    detection2 = detect_charuco_for_panorama(image2, board_config)
    
    if detection1 is None:
        return {
            'success': False,
            'error': 'Failed to detect ChArUco board in first camera image',
            'camera': 1
        }
    
    if detection2 is None:
        return {
            'success': False,
            'error': 'Failed to detect ChArUco board in second camera image',
            'camera': 2
        }
    
    # Match corners
    try:
        points1, points2 = match_charuco_corners(detection1, detection2)
    except ValueError as e:
        return {
            'success': False,
            'error': str(e),
            'detected_corners_cam1': detection1['num_corners'],
            'detected_corners_cam2': detection2['num_corners']
        }
    
    # Compute homography
    try:
        H, mask, metrics = compute_panorama_homography(points1, points2)
    except ValueError as e:
        return {
            'success': False,
            'error': str(e),
            'matches': len(points1)
        }
    
    return {
        'success': True,
        'homography': H.tolist(),
        'metrics': metrics,
        'matches': len(points1),
        'detected_corners_cam1': detection1['num_corners'],
        'detected_corners_cam2': detection2['num_corners']
    }


def stitch_images(image1: np.ndarray, image2: np.ndarray, homography: np.ndarray,
                  blend_width: int = 50) -> np.ndarray:
    """
    Stitch two images using computed homography.
    
    Args:
        image1: First image (reference)
        image2: Second image (to be warped)
        homography: 3x3 homography matrix
        blend_width: Width of blending zone in pixels
    
    Returns:
        Stitched panorama image
    """
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    # Find corners of second image in first image coordinate system
    corners2 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], dtype=np.float32).reshape(-1, 1, 2)
    corners2_transformed = cv2.perspectiveTransform(corners2, homography)
    
    # Compute output canvas size
    all_corners = np.concatenate([
        np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]], dtype=np.float32).reshape(-1, 1, 2),
        corners2_transformed
    ], axis=0)
    
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    # Translation to keep everything in frame
    translation = np.array([[1, 0, -x_min],
                           [0, 1, -y_min],
                           [0, 0, 1]])
    
    # Warp second image
    output_size = (x_max - x_min, y_max - y_min)
    warped2 = cv2.warpPerspective(image2, translation @ homography, output_size)
    
    # Place first image
    result = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
    result[-y_min:-y_min+h1, -x_min:-x_min+w1] = image1
    
    # Simple blending (just overlay for now - can improve later)
    mask = (warped2 > 0).any(axis=2)
    result[mask] = warped2[mask]
    
    return result


def save_panorama_calibration(calibration_data: dict, camera1_id: str, camera2_id: str,
                              output_dir: str = "backend/camera/settings/panorama") -> str:
    """
    Save panorama calibration to disk.
    
    Args:
        calibration_data: Calibration result dictionary
        camera1_id: ID of first camera
        camera2_id: ID of second camera
        output_dir: Directory to save calibration
    
    Returns:
        Path to saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create filename
    filename = f"{camera1_id}_{camera2_id}.json"
    filepath = output_path / filename
    
    # Add metadata
    save_data = {
        'camera1_id': camera1_id,
        'camera2_id': camera2_id,
        'calibration': calibration_data,
        'version': '1.0'
    }
    
    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    logger.info(f"Saved panorama calibration to {filepath}")
    return str(filepath)


def save_stereo_calibration(stereo_calib: dict, camera1_id: str, camera2_id: str,
                            output_dir: str = "backend/camera/settings/stereo") -> str:
    """
    Save stereo calibration (extrinsics) to disk.
    
    Args:
        stereo_calib: Stereo calibration result dictionary
        camera1_id: ID of first camera
        camera2_id: ID of second camera
        output_dir: Directory to save calibration
    
    Returns:
        Path to saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create filename
    filename = f"{camera1_id}_{camera2_id}.json"
    filepath = output_path / filename
    
    # Add metadata
    save_data = {
        'camera1_id': camera1_id,
        'camera2_id': camera2_id,
        'stereo_calibration': stereo_calib,
        'version': '1.0',
        'description': 'Stereo calibration containing rotation, translation, and epipolar geometry'
    }
    
    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    logger.info(f"Saved stereo calibration to {filepath}")
    return str(filepath)


def load_stereo_calibration(camera1_id: str, camera2_id: str,
                            input_dir: str = "backend/camera/settings/stereo") -> Optional[dict]:
    """
    Load stereo calibration from disk.
    
    Args:
        camera1_id: ID of first camera
        camera2_id: ID of second camera
        input_dir: Directory containing calibrations
    
    Returns:
        Stereo calibration data or None if not found
    """
    input_path = Path(input_dir)
    
    # Try both camera order combinations
    filenames = [
        f"{camera1_id}_{camera2_id}.json",
        f"{camera2_id}_{camera1_id}.json"
    ]
    
    for filename in filenames:
        filepath = input_path / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded stereo calibration from {filepath}")
            return data
    
    return None


def calibrate_panorama_multiple(image_pairs: List[Tuple[np.ndarray, np.ndarray]],
                                board_config: dict,
                                camera1_calibration: Optional[dict] = None,
                                camera2_calibration: Optional[dict] = None,
                                stereo_flags: Optional[int] = None,
                                calibration_method: str = 'extrinsics_only') -> dict:
    """
    Calibrate panorama alignment using multiple image pairs for robustness.
    
    Args:
        image_pairs: List of (image1, image2) tuples from both cameras
        board_config: ChArUco board configuration
        camera1_calibration: Optional individual calibration for camera 1
        camera2_calibration: Optional individual calibration for camera 2
        stereo_flags: Optional OpenCV stereo calibration flags (e.g., cv2.CALIB_FIX_INTRINSIC)
        calibration_method: Either 'extrinsics_only' (PnP-based) or 'full_stereo' (stereoCalibrate)
    
    Returns:
        Dictionary with calibration results including averaged homography
    """
    if len(image_pairs) == 0:
        return {
            'success': False,
            'error': 'No image pairs provided'
        }
    
    all_points1 = []
    all_points2 = []
    capture_results = []
    
    # Process each image pair - detect on ORIGINAL images, don't undistort first!
    # The calibrations will be used in stereoCalibrate with CALIB_FIX_INTRINSIC flag
    for idx, (image1, image2) in enumerate(image_pairs):
        # Detect ChArUco on original (distorted) images
        detection1 = detect_charuco_for_panorama(image1, board_config)
        detection2 = detect_charuco_for_panorama(image2, board_config)
        
        if detection1 is None or detection2 is None:
            error_msg = 'Board not detected in both cameras'
            if detection1 is None and detection2 is None:
                error_msg = 'Board not detected in either camera'
            elif detection1 is None:
                error_msg = f'Board not detected in camera 0 (camera 1: {detection2["num_corners"]} corners)'
            else:
                error_msg = f'Board not detected in camera 1 (camera 0: {detection1["num_corners"]} corners)'
            
            print(f"Capture {idx}: {error_msg}")
            capture_results.append({
                'capture_idx': idx,
                'success': False,
                'error': error_msg
            })
            continue
        
        # Match corners
        try:
            points1, points2 = match_charuco_corners(detection1, detection2)
            all_points1.append(points1)
            all_points2.append(points2)
            
            capture_results.append({
                'capture_idx': idx,
                'success': True,
                'matches': len(points1),
                'corners_cam1': detection1['num_corners'],
                'corners_cam2': detection2['num_corners']
            })
        except ValueError as e:
            capture_results.append({
                'capture_idx': idx,
                'success': False,
                'error': str(e)
            })
    
    # Check if we have enough successful captures
    successful_captures = [r for r in capture_results if r.get('success', False)]
    if len(successful_captures) == 0:
        return {
            'success': False,
            'error': 'No successful captures - board not detected in both cameras',
            'capture_results': capture_results
        }
    
    # Combine all points from successful captures
    combined_points1 = np.vstack(all_points1)
    combined_points2 = np.vstack(all_points2)
    
    # Compute homography from all points
    try:
        H, mask, metrics = compute_panorama_homography(combined_points1, combined_points2)
    except ValueError as e:
        return {
            'success': False,
            'error': str(e),
            'capture_results': capture_results
        }
    
    result = {
        'success': True,
        'homography': H.tolist(),
        'metrics': metrics,
        'total_captures': len(image_pairs),
        'successful_captures': len(successful_captures),
        'total_matches': len(combined_points1),
        'capture_results': capture_results
    }
    
    # If calibrations provided, compute extrinsics (R, T) using 3D board points
    # This gives the relative pose between cameras with correct metric scale
    if camera1_calibration is not None and camera2_calibration is not None:
        try:
            # Extract calibration matrices
            camera_matrix1 = np.array(camera1_calibration['camera_matrix'])
            dist_coeffs1 = np.array(camera1_calibration['distortion_coeffs'])
            camera_matrix2 = np.array(camera2_calibration['camera_matrix'])
            dist_coeffs2 = np.array(camera2_calibration['distortion_coeffs'])
            
            # Get board parameters
            squares_x = board_config.get('squares_x', board_config.get('width'))
            squares_y = board_config.get('squares_y', board_config.get('height'))
            square_length = board_config['square_length']
            marker_length = board_config['marker_length']
            dictionary_name = board_config.get('dictionary', 'DICT_6X6_100')
            
            # Create ChArUco board to get 3D object points
            aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
            charuco_board = cv2.aruco.CharucoBoard(
                (squares_x, squares_y),
                square_length,
                marker_length,
                aruco_dict
            )
            
            # Get 3D positions of all ChArUco corners on the board (in mm)
            board_3d_points = charuco_board.getChessboardCorners()
            
            if calibration_method == 'extrinsics_only':
                # Method 1: Extrinsics-only using PnP (recommended for divergent cameras)
                # Collect board poses from all captures
                R_list_cam1 = []
                T_list_cam1 = []
                R_list_cam2 = []
                T_list_cam2 = []
                
                for idx, (image1, image2) in enumerate(image_pairs):
                    if not capture_results[idx].get('success', False):
                        continue
                    
                    # Detect ChArUco corners in both images
                    detection1 = detect_charuco_for_panorama(image1, board_config)
                    detection2 = detect_charuco_for_panorama(image2, board_config)
                    
                    if detection1 is None or detection2 is None:
                        continue
                    
                    # For camera 1: Get board pose using solvePnP
                    if len(detection1['ids']) >= 4:
                        ids1 = detection1['ids'].flatten()
                        corners1 = detection1['corners'].reshape(-1, 2)
                        
                        # Get 3D object points for detected corners
                        obj_pts1 = np.array([board_3d_points[int(id)] for id in ids1], dtype=np.float32)
                        
                        # Solve PnP to get board pose in camera 1 frame
                        success1, rvec1, tvec1 = cv2.solvePnP(
                            obj_pts1,
                            corners1,
                            camera_matrix1,
                            dist_coeffs1,
                            flags=cv2.SOLVEPNP_ITERATIVE
                        )
                        
                        if success1:
                            R1, _ = cv2.Rodrigues(rvec1)
                            R_list_cam1.append(R1)
                            T_list_cam1.append(tvec1)
                    
                    # For camera 2: Get board pose using solvePnP
                    if len(detection2['ids']) >= 4:
                        ids2 = detection2['ids'].flatten()
                        corners2 = detection2['corners'].reshape(-1, 2)
                        
                        # Get 3D object points for detected corners
                        obj_pts2 = np.array([board_3d_points[int(id)] for id in ids2], dtype=np.float32)
                        
                        # Solve PnP to get board pose in camera 2 frame
                        success2, rvec2, tvec2 = cv2.solvePnP(
                            obj_pts2,
                            corners2,
                            camera_matrix2,
                            dist_coeffs2,
                            flags=cv2.SOLVEPNP_ITERATIVE
                        )
                        
                        if success2:
                            R2, _ = cv2.Rodrigues(rvec2)
                            R_list_cam2.append(R2)
                            T_list_cam2.append(tvec2)
                
                if len(R_list_cam1) > 0 and len(R_list_cam2) > 0:
                    # Average the board poses across all captures for stability
                    R_board_cam1 = np.mean(R_list_cam1, axis=0)
                    T_board_cam1 = np.mean(T_list_cam1, axis=0)
                    R_board_cam2 = np.mean(R_list_cam2, axis=0)
                    T_board_cam2 = np.mean(T_list_cam2, axis=0)
                    
                    # Compute relative transformation from camera 1 to camera 2
                    R_board_cam1_inv = R_board_cam1.T
                    R_board_cam2_inv = R_board_cam2.T
                    
                    # Relative rotation: R_2to1 = R1 * R2^T
                    R_rel = R_board_cam1 @ R_board_cam2_inv
                    
                    # Relative translation: T_2to1 = T1 - R_2to1 * T2
                    T_rel = T_board_cam1 - R_rel @ T_board_cam2
                    
                    # Compute baseline distance
                    baseline = float(np.linalg.norm(T_rel))
                    
                    # Also compute fundamental matrix for reference
                    F, mask_F = cv2.findFundamentalMat(
                        combined_points1,
                        combined_points2,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=3.0,
                        confidence=0.99
                    )
                    
                    result['extrinsics'] = {
                        'rotation_matrix': R_rel.tolist(),
                        'translation_vector': T_rel.flatten().tolist(),
                        'baseline_distance_mm': baseline,
                        'fundamental_matrix': F.tolist() if F is not None else None,
                        'method': 'PnP-based pose estimation using ChArUco 3D points',
                        'calibration_type': 'extrinsics_only',
                        'extrinsics_only': True,
                        'board_poses_used': min(len(R_list_cam1), len(R_list_cam2)),
                        'note': 'Translation is in millimeters based on board square_length parameter. Extrinsics computed from individual camera calibrations using PnP, not full stereo calibration.'
                    }
                else:
                    result['extrinsics'] = {
                        'success': False,
                        'error': 'Could not compute board poses - need at least 4 corners detected in each camera'
                    }
            
            else:  # calibration_method == 'full_stereo'
                # Method 2: Full stereo calibration using cv2.stereoCalibrate
                # Prepare object points and image points for stereo calibration
                object_points = []
                image_points1 = []
                image_points2 = []
                
                for idx, (image1, image2) in enumerate(image_pairs):
                    if not capture_results[idx].get('success', False):
                        continue
                    
                    # Detect ChArUco corners in both images
                    detection1 = detect_charuco_for_panorama(image1, board_config)
                    detection2 = detect_charuco_for_panorama(image2, board_config)
                    
                    if detection1 is None or detection2 is None:
                        continue
                    
                    # Find common corner IDs between both cameras
                    ids1_flat = detection1['ids'].flatten()
                    ids2_flat = detection2['ids'].flatten()
                    common_ids = np.intersect1d(ids1_flat, ids2_flat)
                    
                    if len(common_ids) >= 4:  # Need at least 4 common points
                        # Extract matching corners and object points
                        matched_obj_points = []
                        matched_corners1 = []
                        matched_corners2 = []
                        
                        for corner_id in common_ids:
                            idx1 = np.where(ids1_flat == corner_id)[0][0]
                            idx2 = np.where(ids2_flat == corner_id)[0][0]
                            matched_corners1.append(detection1['corners'][idx1])
                            matched_corners2.append(detection2['corners'][idx2])
                            matched_obj_points.append(board_3d_points[corner_id])
                        
                        object_points.append(np.array(matched_obj_points, dtype=np.float32))
                        image_points1.append(np.array(matched_corners1, dtype=np.float32))
                        image_points2.append(np.array(matched_corners2, dtype=np.float32))
                
                if len(object_points) >= 3:
                    # Get image size
                    image_size = image_pairs[0][0].shape[:2][::-1]  # (width, height)
                    
                    # Perform stereo calibration
                    # Default to fixing intrinsics if no flags provided
                    if stereo_flags is None:
                        stereo_flags = cv2.CALIB_FIX_INTRINSIC
                    
                    retval, K1_out, d1_out, K2_out, d2_out, R, T, E, F = cv2.stereoCalibrate(
                        object_points,
                        image_points1,
                        image_points2,
                        camera_matrix1,
                        dist_coeffs1,
                        camera_matrix2,
                        dist_coeffs2,
                        image_size,
                        flags=stereo_flags
                    )
                    
                    # Compute baseline distance
                    baseline = float(np.linalg.norm(T))
                    
                    result['extrinsics'] = {
                        'rotation_matrix': R.tolist(),
                        'translation_vector': T.flatten().tolist(),
                        'baseline_distance_mm': baseline,
                        'essential_matrix': E.tolist(),
                        'fundamental_matrix': F.tolist(),
                        'reprojection_error': float(retval),
                        'method': 'Full stereo calibration using cv2.stereoCalibrate',
                        'calibration_type': 'full_stereo',
                        'extrinsics_only': False,
                        'pairs_used': len(object_points),
                        'note': 'Full stereo calibration jointly optimizing intrinsics and extrinsics. Translation is in millimeters based on board square_length parameter.'
                    }
                else:
                    result['extrinsics'] = {
                        'success': False,
                        'error': f'Not enough valid stereo pairs for calibration. Found {len(object_points)} pairs with common corners, need at least 3.'
                    }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            result['extrinsics'] = {
                'success': False,
                'error': f'Extrinsics computation failed: {str(e)}'
            }
    
    return result


def load_panorama_calibration(camera1_id: str, camera2_id: str,
                              input_dir: str = "backend/camera/settings/panorama") -> Optional[dict]:
    """
    Load panorama calibration from disk.
    
    Args:
        camera1_id: ID of first camera
        camera2_id: ID of second camera
        input_dir: Directory containing calibrations
    
    Returns:
        Calibration data or None if not found
    """
    input_path = Path(input_dir)
    
    # Try both camera order combinations
    filenames = [
        f"{camera1_id}_{camera2_id}.json",
        f"{camera2_id}_{camera1_id}.json"
    ]
    
    for filename in filenames:
        filepath = input_path / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded panorama calibration from {filepath}")
            return data
    
    return None


def cylindrical_warp(image: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Warp image to cylindrical coordinates.
    
    Args:
        image: Input image
        K: Camera intrinsic matrix
        
    Returns:
        Cylindrically warped image
    """
    h, w = image.shape[:2]
    
    # Get focal length and center from K
    f = K[0, 0]  # focal length
    cx = K[0, 2]  # principal point x
    cy = K[1, 2]  # principal point y
    
    # Create coordinate meshgrid for output image
    y_i, x_i = np.indices((h, w))
    
    # Convert to cylindrical coordinates
    # x' = f * tan((x - cx) / f)
    # y' = (y - cy) * sqrt(x'^2 + f^2) / f
    
    X = (x_i - cx) / f
    Y = (y_i - cy) / f
    
    # Cylindrical projection
    theta = X
    h_cyl = Y * np.sqrt(X**2 + 1)
    
    # Map back to image coordinates
    x_src = f * np.tan(theta) + cx
    y_src = h_cyl * f + cy
    
    # Remap the image
    warped = cv2.remap(image, x_src.astype(np.float32), y_src.astype(np.float32), 
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    return warped


def stitch_panorama_cylindrical(image_left: np.ndarray, image_right: np.ndarray,
                                rotation_matrix: np.ndarray,
                                K_left: np.ndarray, K_right: np.ndarray,
                                blend_width: int = 50) -> np.ndarray:
    """
    Create panorama using cylindrical projection with symmetric camera rotation.
    
    This approach:
    1. Projects both images to cylindrical coordinates
    2. Rotates each camera away from center (symmetric)
    3. Places them side-by-side with blending
    
    Args:
        image_left: Left camera image
        image_right: Right camera image
        rotation_matrix: 3x3 rotation from left to right camera
        K_left: Left camera intrinsic matrix
        K_right: Right camera intrinsic matrix
        blend_width: Blending region width in pixels
        
    Returns:
        Cylindrical panorama
    """
    # Convert rotation to axis-angle to get rotation angle
    rvec, _ = cv2.Rodrigues(rotation_matrix)
    rotation_angle = np.linalg.norm(rvec)
    
    # Each camera rotates half the total angle
    half_angle = rotation_angle / 2.0
    
    print(f"  Cylindrical projection: total rotation {np.degrees(rotation_angle):.1f}°, half angle {np.degrees(half_angle):.1f}°")
    
    # Warp both images to cylindrical projection
    cyl_left = cylindrical_warp(image_left, K_left)
    cyl_right = cylindrical_warp(image_right, K_right)
    
    h, w = cyl_left.shape[:2]
    
    # Calculate horizontal shift based on rotation
    # Each camera sees roughly half_angle * f worth of scene
    f_left = K_left[0, 0]
    f_right = K_right[0, 0]
    
    # Shift amount (in pixels) for each camera
    # This creates the panoramic effect
    shift_left = int(f_left * np.tan(half_angle))
    shift_right = int(f_right * np.tan(half_angle))
    
    # Create output canvas
    # Width = left image + shift_left + shift_right + right image (minus overlap)
    overlap = blend_width * 2
    canvas_width = w + shift_left + shift_right
    canvas_height = h
    
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    
    # Place left image (shifted to the left)
    # The left camera sees the left part of the panorama
    canvas[:h, :w] = cyl_left
    
    # Place right image (shifted to the right)  
    # The right camera sees the right part of the panorama
    right_start = shift_left + shift_right
    canvas[:h, right_start:right_start + w] = cyl_right
    
    # Blend in overlap region
    if blend_width > 0:
        blend_start = w - blend_width
        blend_end = right_start + blend_width
        
        if blend_start < blend_end and blend_start >= 0:
            blend_region_width = min(blend_end - blend_start, canvas_width - blend_start)
            
            for i in range(blend_region_width):
                x = blend_start + i
                if x < canvas_width:
                    alpha = i / blend_region_width
                    
                    # Get pixels from both images at this x position
                    left_pixel = canvas[:h, x].copy()
                    
                    right_x = x - right_start
                    if 0 <= right_x < w:
                        right_pixel = cyl_right[:h, right_x]
                        
                        # Blend where both have valid data
                        mask = (left_pixel.sum(axis=1) > 0) & (right_pixel.sum(axis=1) > 0)
                        canvas[mask, x] = (left_pixel[mask] * (1 - alpha) + 
                                         right_pixel[mask] * alpha).astype(np.uint8)
    
    return canvas


def compute_symmetric_rectification(rotation_matrix: np.ndarray, homography: np.ndarray,
                                    image_shape: Tuple[int, int],
                                    K_left: Optional[np.ndarray] = None,
                                    K_right: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute symmetric rectification transforms for panorama stitching.
    
    Uses the rotation matrix from extrinsics to create balanced transforms that
    warp both cameras equally toward a virtual center camera. This creates a
    symmetric panorama where the baseline between cameras becomes the centerline.
    
    Args:
        rotation_matrix: 3x3 rotation matrix from camera 1 to camera 2 (from extrinsics)
        homography: 3x3 homography matrix (for fallback/validation)
        image_shape: (height, width) of input images
        K_left: Optional 3x3 camera intrinsic matrix for left camera
        K_right: Optional 3x3 camera intrinsic matrix for right camera
        
    Returns:
        (H_left, H_right): Homography matrices to apply to left and right cameras
    """
    import cv2
    
    height, width = image_shape
    
    # Use provided intrinsics or estimate from image size
    if K_left is None:
        # Estimate: focal length ≈ image width
        focal_length = width
        cx, cy = width / 2.0, height / 2.0
        K_left = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float64)
    
    if K_right is None:
        K_right = K_left.copy()
    
    # Convert rotation matrix to Rodrigues rotation vector
    rvec, _ = cv2.Rodrigues(rotation_matrix)
    
    # Create half-rotation for symmetric split
    # Each camera rotates halfway toward the other
    rvec_half = rvec / 2.0
    R_half, _ = cv2.Rodrigues(rvec_half)
    R_half_inv, _ = cv2.Rodrigues(-rvec_half)
    
    # Create rectification homographies
    # H = K * R * K_inv transforms the image
    K_left_inv = np.linalg.inv(K_left)
    K_right_inv = np.linalg.inv(K_right)
    
    # Left camera: rotate by -half (toward center)
    H_left = K_left @ R_half_inv @ K_left_inv
    
    # Right camera: rotate by +half (toward center), then apply relative transformation
    H_right = K_right @ R_half @ K_right_inv @ homography
    
    return H_left, H_right


def stitch_panorama_symmetric_rectified(image_left: np.ndarray, image_right: np.ndarray,
                                        rotation_matrix: np.ndarray, homography: np.ndarray,
                                        blend_width: int = 50,
                                        K_left: Optional[np.ndarray] = None,
                                        K_right: Optional[np.ndarray] = None,
                                        calibration_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Stitch two images using symmetric rectification based on extrinsics.
    
    This creates a geometrically optimal panorama where both cameras are warped
    equally toward a virtual center camera, resulting in a natural symmetric layout.
    
    Args:
        image_left: Left camera image
        image_right: Right camera image
        rotation_matrix: 3x3 rotation from extrinsics (camera 1 to camera 2)
        homography: 3x3 homography matrix (cam2 to cam1 coordinates)
        blend_width: Width of blending region in pixels
        K_left: Optional 3x3 intrinsic matrix for left camera
        K_right: Optional 3x3 intrinsic matrix for right camera
        calibration_size: Optional (width, height) at which calibration was performed
        
    Returns:
        Stitched panorama with symmetric rectification
    """
    h, w = image_left.shape[:2]
    
    # Scale intrinsics and homography if current resolution differs from calibration
    if calibration_size is not None and K_left is not None and K_right is not None:
        calib_w, calib_h = calibration_size
        scale_x = w / calib_w
        scale_y = h / calib_h
        
        if abs(scale_x - 1.0) > 0.01 or abs(scale_y - 1.0) > 0.01:
            # Scale camera matrices
            S = np.array([
                [scale_x, 0, 0],
                [0, scale_y, 0],
                [0, 0, 1]
            ], dtype=np.float64)
            
            K_left = S @ K_left
            K_right = S @ K_right
            
            # Scale homography: H' = S * H * S^-1
            S_inv = np.linalg.inv(S)
            homography = S @ homography @ S_inv
    
    # Compute symmetric rectification transforms
    H_left, H_right = compute_symmetric_rectification(
        rotation_matrix, homography, (h, w), K_left, K_right
    )
    
    # Find output canvas size by warping image corners
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
    
    corners_left_warped = cv2.perspectiveTransform(corners, H_left)
    corners_right_warped = cv2.perspectiveTransform(corners, H_right)
    
    all_corners = np.concatenate([corners_left_warped, corners_right_warped], axis=0)
    
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    # Translation to keep everything visible
    translation = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ], dtype=np.float64)
    
    output_width = x_max - x_min
    output_height = y_max - y_min
    
    # Warp both images
    warped_left = cv2.warpPerspective(
        image_left,
        translation @ H_left,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR
    )
    
    warped_right = cv2.warpPerspective(
        image_right,
        translation @ H_right,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR
    )
    
    # Alpha blending in overlap region
    if blend_width > 0:
        mask_left = (warped_left.sum(axis=2) > 0).astype(np.uint8)
        mask_right = (warped_right.sum(axis=2) > 0).astype(np.uint8)
        
        overlap = (mask_left & mask_right) > 0
        
        if overlap.any():
            # Distance transform for smooth blending
            dist_left = cv2.distanceTransform(mask_left, cv2.DIST_L2, 5).astype(np.float32)
            dist_right = cv2.distanceTransform(mask_right, cv2.DIST_L2, 5).astype(np.float32)
            
            # Compute blending weights
            total_dist = dist_left + dist_right + 1e-10
            alpha_left = (dist_left / total_dist)[:, :, np.newaxis]
            alpha_right = (dist_right / total_dist)[:, :, np.newaxis]
            
            # Blend
            overlap_mask = overlap[:, :, np.newaxis]
            panorama = (
                warped_left.astype(np.float32) * alpha_left * overlap_mask +
                warped_right.astype(np.float32) * alpha_right * overlap_mask +
                warped_left.astype(np.float32) * (1 - overlap_mask) * (mask_left[:, :, np.newaxis] > 0) +
                warped_right.astype(np.float32) * (1 - overlap_mask) * (mask_right[:, :, np.newaxis] > 0)
            )
            panorama = panorama.astype(np.uint8)
        else:
            panorama = np.maximum(warped_left, warped_right)
    else:
        panorama = np.maximum(warped_left, warped_right)
    
    return panorama


def stitch_panorama_symmetric(image_left: np.ndarray, image_right: np.ndarray,
                              homography: np.ndarray, blend_width: int = 50) -> np.ndarray:
    """
    Stitch two images into a symmetric panorama where both images are warped equally.
    
    This creates a balanced panorama with the centerline between cameras in the middle.
    Both cameras contribute equally to the final result, unlike asymmetric stitching
    where one camera is the anchor.
    
    Args:
        image_left: Left camera image
        image_right: Right camera image  
        homography: 3x3 homography matrix mapping right to left coordinates
        blend_width: Width of the blending region in pixels
        
    Returns:
        Stitched panorama with symmetric layout
    """
    h_left, w_left = image_left.shape[:2]
    h_right, w_right = image_right.shape[:2]
    
    # Find where right image corners map to in left image space
    corners_right = np.array([[0, 0], [w_right, 0], [w_right, h_right], [0, h_right]], 
                            dtype=np.float32).reshape(-1, 1, 2)
    corners_right_in_left = cv2.perspectiveTransform(corners_right, homography)
    
    # Get all corners in left coordinate system
    corners_left = np.array([[0, 0], [w_left, 0], [w_left, h_left], [0, h_left]], 
                           dtype=np.float32).reshape(-1, 1, 2)
    all_corners = np.concatenate([corners_left, corners_right_in_left], axis=0)
    
    # Find bounding box
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    # Calculate center point in the combined space
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    
    # Calculate how much to shift to center the panorama
    # We want the midpoint between the two cameras to be in the center
    output_width = x_max - x_min
    output_height = y_max - y_min
    
    # Create symmetric transformation: shift so center is at origin, then shift to canvas center
    center_offset_x = output_width / 2 - x_center
    center_offset_y = output_height / 2 - y_center
    
    # Translation matrix to center the panorama
    translation = np.array([[1, 0, -x_min + center_offset_x],
                           [0, 1, -y_min + center_offset_y],
                           [0, 0, 1]], dtype=np.float64)
    
    # Warp both images with the centering translation
    warped_left = cv2.warpPerspective(
        image_left,
        translation,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR
    )
    
    warped_right = cv2.warpPerspective(
        image_right,
        translation @ homography,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR
    )
    
    # Create alpha blending
    if blend_width > 0:
        # Create masks
        mask_left = (warped_left.sum(axis=2) > 0).astype(np.uint8)
        mask_right = (warped_right.sum(axis=2) > 0).astype(np.uint8)
        
        # Find overlap
        overlap = (mask_left & mask_right) > 0
        
        if overlap.any():
            # Distance transforms for smooth blending
            dist_left = cv2.distanceTransform(mask_left, cv2.DIST_L2, 5).astype(np.float32)
            dist_right = cv2.distanceTransform(mask_right, cv2.DIST_L2, 5).astype(np.float32)
            
            # Normalize in overlap region
            total_dist = dist_left + dist_right + 1e-10
            alpha_left = (dist_left / total_dist)[:, :, np.newaxis]
            alpha_right = (dist_right / total_dist)[:, :, np.newaxis]
            
            # Blend
            overlap_3ch = overlap[:, :, np.newaxis]
            panorama = (
                warped_left.astype(np.float32) * alpha_left * overlap_3ch +
                warped_right.astype(np.float32) * alpha_right * overlap_3ch +
                warped_left.astype(np.float32) * (1 - overlap_3ch) * (mask_left[:, :, np.newaxis] > 0) +
                warped_right.astype(np.float32) * (1 - overlap_3ch) * (mask_right[:, :, np.newaxis] > 0)
            )
            panorama = panorama.astype(np.uint8)
        else:
            # No overlap, just combine
            panorama = np.maximum(warped_left, warped_right)
    else:
        # No blending
        panorama = np.maximum(warped_left, warped_right)
    
    return panorama


def stitch_panorama(image1: np.ndarray, image2: np.ndarray, 
                    homography: np.ndarray, blend_width: int = 50) -> np.ndarray:
    """
    Stitch two images into a panorama using a homography matrix.
    
    Args:
        image1: First image (left/base image)
        image2: Second image (right image to be warped)
        homography: 3x3 homography matrix that maps image2 to image1 coordinates
        blend_width: Width of the blending region in pixels
    
    Returns:
        Stitched panorama image
    """
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    # Get corners of image2 in its own coordinate system
    corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    
    # Transform corners to image1 coordinate system
    corners2_transformed = cv2.perspectiveTransform(corners2, homography)
    
    # Combine with image1 corners to get the full panorama bounds
    corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    all_corners = np.concatenate([corners1, corners2_transformed], axis=0)
    
    # Get bounding box
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    # Translation to keep all pixels in view
    translation = np.array([[1, 0, -x_min],
                           [0, 1, -y_min],
                           [0, 0, 1]])
    
    # Output size
    output_width = x_max - x_min
    output_height = y_max - y_min
    
    # Warp image2 to panorama coordinates
    warped_image2 = cv2.warpPerspective(
        image2,
        translation @ homography,
        (output_width, output_height)
    )
    
    # Place image1 in panorama coordinates
    panorama = cv2.warpPerspective(
        image1,
        translation,
        (output_width, output_height)
    )
    
    # Create alpha blending mask
    if blend_width > 0:
        # Create mask for warped_image2
        mask2 = (warped_image2.sum(axis=2) > 0).astype(np.float32)
        
        # Create mask for panorama (image1)
        mask1 = (panorama.sum(axis=2) > 0).astype(np.float32)
        
        # Find overlap region
        overlap = (mask1 * mask2) > 0
        
        if overlap.any():
            # Create distance transforms for alpha blending
            # Distance from edge of each image
            dist1 = cv2.distanceTransform((mask1 > 0).astype(np.uint8), cv2.DIST_L2, 5)
            dist2 = cv2.distanceTransform((mask2 > 0).astype(np.uint8), cv2.DIST_L2, 5)
            
            # Normalize distances in overlap region
            alpha1 = dist1 / (dist1 + dist2 + 1e-10)
            alpha2 = dist2 / (dist1 + dist2 + 1e-10)
            
            # Apply feathering only in overlap region
            alpha1 = np.expand_dims(alpha1, axis=2)
            alpha2 = np.expand_dims(alpha2, axis=2)
            
            # Blend in overlap region
            overlap_mask = np.expand_dims(overlap.astype(np.float32), axis=2)
            panorama = (panorama * alpha1 * overlap_mask + 
                       warped_image2 * alpha2 * overlap_mask +
                       panorama * (1 - overlap_mask) * mask1[:,:,np.newaxis] +
                       warped_image2 * (1 - overlap_mask) * mask2[:,:,np.newaxis])
            panorama = panorama.astype(np.uint8)
        else:
            # No overlap, just combine
            mask2_3ch = np.stack([mask2, mask2, mask2], axis=2) > 0
            panorama[mask2_3ch] = warped_image2[mask2_3ch]
    else:
        # No blending, simple overlay
        mask2 = (warped_image2.sum(axis=2) > 0)
        mask2_3ch = np.stack([mask2, mask2, mask2], axis=2)
        panorama[mask2_3ch] = warped_image2[mask2_3ch]
    
    return panorama
