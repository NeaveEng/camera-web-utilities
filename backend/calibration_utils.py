"""
Camera calibration utilities using ChArUco boards.
Handles the calibration process from captured images to camera parameters.
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path


def calibrate_camera_from_session(session_path, board_config, progress_callback=None):
    """
    Calibrate camera using images from a calibration session.
    
    Args:
        session_path: Path to calibration session directory containing images
        board_config: Dictionary with board configuration (dictionary, width, height, square_length, marker_length)
        progress_callback: Optional callback function(message: str) for progress updates
    
    Returns:
        Dictionary with calibration results or error information
    """
    def log_progress(message):
        if progress_callback:
            progress_callback(message)
    
    try:
        log_progress('Loading calibration images...')
        # Load images from session
        image_files = sorted(Path(session_path).glob("image_*.jpg"))
        
        if len(image_files) < 5:
            return {
                'success': False,
                'error': f'Insufficient images for calibration. Found {len(image_files)}, need at least 5.',
                'images_found': len(image_files)
            }
        
        log_progress(f'Found {len(image_files)} images to process')
        
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
        
        # Detect ChArUco corners in all images
        log_progress('Detecting ChArUco patterns in images...')
        all_charuco_corners = []
        all_charuco_ids = []
        image_size = None
        processed_images = []
        
        # Use ArUco detector parameters
        detector_params = cv2.aruco.DetectorParameters()
        
        for i, img_path in enumerate(image_files):
            log_progress(f'Processing image {i+1}/{len(image_files)}: {img_path.name}')
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            if image_size is None:
                image_size = img.shape[:2][::-1]  # (width, height)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect ArUco markers first
            marker_corners, marker_ids, rejected = cv2.aruco.detectMarkers(
                gray, aruco_dict, parameters=detector_params
            )
            
            # If markers found, interpolate ChArUco corners
            if marker_ids is not None and len(marker_ids) > 0:
                ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    marker_corners, marker_ids, gray, board
                )
                
                # Only use images with sufficient corners detected
                if ret and charuco_corners is not None and len(charuco_corners) >= 10:
                    all_charuco_corners.append(charuco_corners)
                    all_charuco_ids.append(charuco_ids)
                    processed_images.append({
                        'filename': img_path.name,
                        'corners_detected': len(charuco_corners)
                    })
        
        if len(all_charuco_corners) < 5:
            return {
                'success': False,
                'error': f'Not enough valid images for calibration. Found {len(all_charuco_corners)} images with sufficient corners, need at least 5.',
                'images_processed': len(all_charuco_corners),
                'images_total': len(image_files)
            }
        
        log_progress(f'Found {len(all_charuco_corners)} valid images with detected patterns')
        
        # Calibrate camera
        log_progress('Computing camera parameters...')
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            all_charuco_corners,
            all_charuco_ids,
            board,
            image_size,
            None,
            None
        )
        
        if not ret:
            return {
                'success': False,
                'error': 'Calibration failed - unable to compute camera parameters',
                'images_used': len(all_charuco_corners)
            }
        
        log_progress('Calculating reprojection errors...')
        # Calculate reprojection error
        total_error = 0
        num_points = 0
        
        for i in range(len(all_charuco_corners)):
            # Project 3D points to 2D
            obj_points = board.getChessboardCorners()[all_charuco_ids[i].flatten()]
            img_points, _ = cv2.projectPoints(
                obj_points,
                rvecs[i],
                tvecs[i],
                camera_matrix,
                dist_coeffs
            )
            
            # Calculate error
            error = cv2.norm(all_charuco_corners[i], img_points, cv2.NORM_L2)
            total_error += error
            num_points += len(all_charuco_corners[i])
        
        mean_error = total_error / num_points if num_points > 0 else 0
        
        # Calculate individual image errors for analysis
        image_errors = []
        for i in range(len(all_charuco_corners)):
            obj_points = board.getChessboardCorners()[all_charuco_ids[i].flatten()]
            img_points, _ = cv2.projectPoints(
                obj_points,
                rvecs[i],
                tvecs[i],
                camera_matrix,
                dist_coeffs
            )
            error = cv2.norm(all_charuco_corners[i], img_points, cv2.NORM_L2) / len(all_charuco_corners[i])
            image_errors.append({
                'filename': processed_images[i]['filename'],
                'error': float(error),
                'corners': processed_images[i]['corners_detected']
            })
        
        # Return calibration results
        return {
            'success': True,
            'camera_matrix': camera_matrix.tolist(),
            'distortion_coeffs': dist_coeffs.flatten().tolist(),
            'reprojection_error': float(mean_error),
            'images_used': len(all_charuco_corners),
            'images_total': len(image_files),
            'image_size': list(image_size),
            'image_errors': image_errors,
            'board_config': board_config
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Calibration error: {str(e)}'
        }


def save_calibration_results(results, output_path, calibration_name):
    """
    Save calibration results to a JSON file.
    
    Args:
        results: Calibration results dictionary
        output_path: Directory to save the calibration file
        calibration_name: Name for the calibration file
    
    Returns:
        Path to saved calibration file or None on error
    """
    try:
        os.makedirs(output_path, exist_ok=True)
        
        filename = f"{calibration_name}.json"
        filepath = os.path.join(output_path, filename)
        
        # Prepare calibration data for saving
        calibration_data = {
            'name': calibration_name,
            'camera_matrix': results['camera_matrix'],
            'distortion_coeffs': results['distortion_coeffs'],
            'reprojection_error': results['reprojection_error'],
            'image_size': results['image_size'],
            'images_used': results['images_used'],
            'board_config': results['board_config'],
            'timestamp': results.get('timestamp', None)
        }
        
        with open(filepath, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        return filepath
        
    except Exception as e:
        print(f"Error saving calibration: {e}")
        return None


def load_calibration(calibration_path):
    """
    Load calibration data from a JSON file.
    
    Args:
        calibration_path: Path to calibration JSON file
    
    Returns:
        Dictionary with calibration data or None on error
    """
    try:
        with open(calibration_path, 'r') as f:
            calibration_data = json.load(f)
        
        # Convert lists back to numpy arrays for use with OpenCV
        calibration_data['camera_matrix'] = np.array(calibration_data['camera_matrix'])
        calibration_data['distortion_coeffs'] = np.array(calibration_data['distortion_coeffs'])
        
        return calibration_data
        
    except Exception as e:
        print(f"Error loading calibration: {e}")
        return None


def undistort_image(image, camera_matrix, dist_coeffs, crop=False):
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


def scale_calibration_for_resolution(camera_matrix, dist_coeffs, original_size, target_size):
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


def stereo_calibrate_from_sessions(camera1_path, camera2_path, board_config, 
                                   camera1_calib, camera2_calib,
                                   flags=None, progress_callback=None):
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
            
            # Detect in camera 1
            corners1, ids1, _ = cv2.aruco.detectMarkers(gray1, aruco_dict, parameters=detector_params)
            ret1, charuco_corners1, charuco_ids1 = None, None, None
            if ids1 is not None and len(ids1) > 0:
                ret1, charuco_corners1, charuco_ids1 = cv2.aruco.interpolateCornersCharuco(
                    corners1, ids1, gray1, board
                )
            
            # Detect in camera 2
            corners2, ids2, _ = cv2.aruco.detectMarkers(gray2, aruco_dict, parameters=detector_params)
            ret2, charuco_corners2, charuco_ids2 = None, None, None
            if ids2 is not None and len(ids2) > 0:
                ret2, charuco_corners2, charuco_ids2 = cv2.aruco.interpolateCornersCharuco(
                    corners2, ids2, gray2, board
                )
            
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


def compute_optimal_homography_from_stereo(stereo_calib, image_size):
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
