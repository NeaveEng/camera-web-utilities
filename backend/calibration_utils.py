"""
Camera calibration utilities using ChArUco boards.
Handles the calibration process from captured images to camera parameters.
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path


def calibrate_camera_from_session(session_path, board_config):
    """
    Calibrate camera using images from a calibration session.
    
    Args:
        session_path: Path to calibration session directory containing images
        board_config: Dictionary with board configuration (dictionary, width, height, square_length, marker_length)
    
    Returns:
        Dictionary with calibration results or error information
    """
    try:
        # Load images from session
        image_files = sorted(Path(session_path).glob("image_*.jpg"))
        
        if len(image_files) < 5:
            return {
                'success': False,
                'error': f'Insufficient images for calibration. Found {len(image_files)}, need at least 5.',
                'images_found': len(image_files)
            }
        
        # Create ChArUco board
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
        all_charuco_corners = []
        all_charuco_ids = []
        image_size = None
        processed_images = []
        
        # Use ArUco detector parameters
        detector_params = cv2.aruco.DetectorParameters()
        
        for i, img_path in enumerate(image_files):
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
        
        # Calibrate camera
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
