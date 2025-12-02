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
                                stereo_flags: Optional[int] = None) -> dict:
    """
    Calibrate panorama alignment using multiple image pairs for robustness.
    
    Args:
        image_pairs: List of (image1, image2) tuples from both cameras
        board_config: ChArUco board configuration
        camera1_calibration: Optional individual calibration for camera 1
        camera2_calibration: Optional individual calibration for camera 2
        stereo_flags: Optional OpenCV stereo calibration flags (e.g., cv2.CALIB_FIX_INTRINSIC)
    
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
    
    # If stereo flags provided and both cameras have calibration, also compute stereo parameters
    # This provides R, T matrices which can be useful for understanding camera geometry
    if stereo_flags is not None and camera1_calibration is not None and camera2_calibration is not None:
        try:
            # Prepare object points and image points for stereo calibration
            # Use the same detected corners from all successful captures
            object_points_list = []
            image_points1_list = []
            image_points2_list = []
            
            # Get board parameters
            squares_x = board_config.get('squares_x', board_config.get('width'))
            squares_y = board_config.get('squares_y', board_config.get('height'))
            square_length = board_config['square_length']
            marker_length = board_config['marker_length']
            dictionary_name = board_config.get('dictionary', 'DICT_6X6_100')
            
            # Create ChArUco board for object point generation
            aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
            charuco_board = cv2.aruco.CharucoBoard(
                (squares_x, squares_y),
                square_length,
                marker_length,
                aruco_dict
            )
            
            # Re-process each successful image pair to get corner IDs and object points
            for idx, (image1, image2) in enumerate(image_pairs):
                if not capture_results[idx].get('success', False):
                    continue
                
                # Undistort images
                camera_matrix1 = np.array(camera1_calibration['camera_matrix'])
                dist_coeffs1 = np.array(camera1_calibration['distortion_coeffs'])
                image1_undist = cv2.undistort(image1, camera_matrix1, dist_coeffs1)
                
                camera_matrix2 = np.array(camera2_calibration['camera_matrix'])
                dist_coeffs2 = np.array(camera2_calibration['distortion_coeffs'])
                image2_undist = cv2.undistort(image2, camera_matrix2, dist_coeffs2)
                
                # Detect corners with IDs
                detection1 = detect_charuco_for_panorama(image1_undist, board_config)
                detection2 = detect_charuco_for_panorama(image2_undist, board_config)
                
                if detection1 is None or detection2 is None:
                    continue
                
                # Find common corner IDs
                ids1 = detection1['ids'].flatten()
                ids2 = detection2['ids'].flatten()
                common_ids = np.intersect1d(ids1, ids2)
                
                if len(common_ids) < 4:
                    continue
                
                # Extract matching corners
                corners1 = []
                corners2 = []
                obj_pts = []
                
                for corner_id in common_ids:
                    idx1 = np.where(ids1 == corner_id)[0][0]
                    idx2 = np.where(ids2 == corner_id)[0][0]
                    corners1.append(detection1['corners'][idx1])
                    corners2.append(detection2['corners'][idx2])
                    
                    # Get 3D object point for this corner ID
                    obj_pts.append(charuco_board.getChessboardCorners()[corner_id])
                
                object_points_list.append(np.array(obj_pts, dtype=np.float32))
                image_points1_list.append(np.array(corners1, dtype=np.float32))
                image_points2_list.append(np.array(corners2, dtype=np.float32))
            
            if len(object_points_list) >= 3:
                # Get image size from first image
                image_size = (image_pairs[0][0].shape[1], image_pairs[0][0].shape[0])
                
                # Run stereo calibration
                retval, cam_mat1, dist1, cam_mat2, dist2, R, T, E, F = cv2.stereoCalibrate(
                    object_points_list,
                    image_points1_list,
                    image_points2_list,
                    camera_matrix1,
                    dist_coeffs1,
                    camera_matrix2,
                    dist_coeffs2,
                    image_size,
                    flags=stereo_flags
                )
                
                result['stereo_calibration'] = {
                    'reprojection_error': float(retval),
                    'rotation_matrix': R.tolist(),
                    'translation_vector': T.flatten().tolist(),
                    'essential_matrix': E.tolist(),
                    'fundamental_matrix': F.tolist(),
                    'flags_used': stereo_flags,
                    'pairs_used': len(object_points_list)
                }
            else:
                result['stereo_calibration'] = {
                    'success': False,
                    'error': f'Not enough valid pairs for stereo calibration (found {len(object_points_list)}, need 3)'
                }
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            result['stereo_calibration'] = {
                'success': False,
                'error': f'Stereo calibration failed: {str(e)}'
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
