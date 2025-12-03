"""
Main Flask application integrating camera backends, features, and workflows.
"""

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from pathlib import Path
import os
import json
import threading
import time

from backend.camera.factory import get_camera_backend, detect_platform
from backend.camera.groups import CameraGroupManager
from backend.camera.sync_pair import SynchronizedPairManager
from backend.features.manager import FeatureManager
from backend.workflows.manager import WorkflowManager
from backend.panorama_utils import calibrate_panorama_pair, save_panorama_calibration, load_panorama_calibration, stitch_images


# Initialize Flask app
# Use absolute path for frontend folder
frontend_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
app = Flask(__name__, static_folder=frontend_folder, static_url_path='')
CORS(app)

# Initialize managers
camera_backend = None
group_manager = None
sync_pair_manager = None
feature_manager = None
workflow_manager = None

# Calibration overlay state
calibration_overlay_enabled = {}
calibration_board_config = {}

# Image processing state
image_processing_enabled = {}
image_processing_config = {}


def initialize():
    """Initialize all backend systems."""
    global camera_backend, group_manager, sync_pair_manager, feature_manager, workflow_manager
    
    print("Detecting platform...")
    platform = detect_platform()
    print(f"Platform detected: {platform}")
    
    print("Initializing camera backend...")
    camera_backend = get_camera_backend()
    
    print("Initializing group manager...")
    group_manager = CameraGroupManager()
    
    print("Initializing synchronized pair manager...")
    sync_pair_manager = SynchronizedPairManager()
    
    print("Initializing feature manager...")
    feature_manager = FeatureManager()
    
    print("Initializing workflow manager...")
    workflow_manager = WorkflowManager()
    workflow_manager.register_plugin_workflows(feature_manager)
    
    # Register plugin routes
    feature_manager.register_plugin_routes(app)
    
    print("Initialization complete!")


# ============================================================================
# Calibration Helper Functions
# ============================================================================

def classify_board_pose(marker_corners, image_width, image_height):
    """
    Classify the pose of a calibration board based on marker positions.
    
    Args:
        marker_corners: List of marker corner arrays
        image_width: Width of the image
        image_height: Height of the image
    
    Returns:
        List of pose classifications (e.g., ['center', 'close'])
    """
    import numpy as np
    
    poses = []
    
    if not marker_corners or len(marker_corners) == 0:
        return poses
    
    # Calculate bounding box of all markers
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    
    for marker in marker_corners:
        # marker is already a numpy array of shape (1, 4, 2) or (4, 2)
        corners = marker.reshape(-1, 2)
        for corner in corners:
            min_x = min(min_x, corner[0])
            min_y = min(min_y, corner[1])
            max_x = max(max_x, corner[0])
            max_y = max(max_y, corner[1])
    
    # Calculate properties
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    width = max_x - min_x
    height = max_y - min_y
    area = width * height
    image_area = image_width * image_height
    scale = area / image_area
    
    # Position classification (divide image into 9 regions)
    margin_x = image_width * 0.25
    margin_y = image_height * 0.25
    
    # Check corners (board center in corner region)
    if center_x < margin_x and center_y < margin_y:
        poses.append('topLeft')
    elif center_x > image_width - margin_x and center_y < margin_y:
        poses.append('topRight')
    elif center_x < margin_x and center_y > image_height - margin_y:
        poses.append('bottomLeft')
    elif center_x > image_width - margin_x and center_y > image_height - margin_y:
        poses.append('bottomRight')
    # Check edges (board center in edge region but not corner)
    elif center_y < margin_y:
        poses.append('topEdge')
    elif center_y > image_height - margin_y:
        poses.append('bottomEdge')
    elif center_x < margin_x:
        poses.append('leftEdge')
    elif center_x > image_width - margin_x:
        poses.append('rightEdge')
    # Center region
    else:
        poses.append('center')
    
    # Scale classification (close vs far)
    if scale > 0.15:
        poses.append('close')
    elif scale < 0.05:
        poses.append('far')
    
    # Tilt classification (check if board appears rotated/tilted)
    aspect_ratio = width / height if height > 0 else 1
    if aspect_ratio < 0.7 or aspect_ratio > 1.43:  # Significant deviation from square
        poses.append('tilted')
    
    return poses


# ============================================================================
# Camera API Routes
# ============================================================================

@app.route('/api/cameras', methods=['GET'])
def list_cameras():
    """List all available cameras."""
    try:
        cameras = camera_backend.enumerate_cameras()
        return jsonify({
            'success': True,
            'cameras': cameras,
            'platform': camera_backend.get_platform_name()
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/cameras/<camera_id>/capabilities', methods=['GET'])
def get_camera_capabilities(camera_id):
    """Get capabilities for a specific camera."""
    try:
        capabilities = camera_backend.get_capabilities(camera_id)
        return jsonify({
            'success': True,
            'capabilities': capabilities
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/cameras/<camera_id>/start', methods=['POST'])
def start_camera(camera_id):
    """Start streaming from a camera."""
    try:
        config = request.json or {}
        success = camera_backend.start_stream(camera_id, config)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Camera {camera_id} started'
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Failed to start camera {camera_id}'
            }), 500
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/cameras/<camera_id>/stop', methods=['POST'])
def stop_camera(camera_id):
    """Stop streaming from a camera."""
    try:
        success = camera_backend.stop_stream(camera_id)
        return jsonify({
            'success': success,
            'message': f'Camera {camera_id} {"stopped" if success else "failed to stop"}'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/sync-pairs/stop-all', methods=['POST'])
def stop_all_sync_pairs():
    """Stop and remove all synchronized camera pairs."""
    try:
        pairs = list(sync_pair_manager.get_all_pairs().keys())
        removed_count = 0
        
        for camera1_id, camera2_id in pairs:
            if sync_pair_manager.remove_pair(camera1_id, camera2_id):
                removed_count += 1
        
        return jsonify({
            'success': True,
            'message': f'Stopped {removed_count} synchronized pair(s)',
            'pairs_stopped': removed_count
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/cameras/<camera_id>/stream')
def camera_stream(camera_id):
    """MJPEG stream endpoint for a camera."""
    import cv2
    import numpy as np
    
    print(f"[Stream] Stream requested for camera {camera_id}")
    
    def generate():
        """Generate MJPEG stream."""
        # Stream directly from camera backend (no sync pairs)
        while camera_backend.is_streaming(camera_id):
            frame_jpeg = camera_backend.get_preview_frame(camera_id)
            
            if not frame_jpeg:
                time.sleep(0.033)
                continue
            
            # If calibration overlay is enabled, decode, draw markers, re-encode
            if frame_jpeg and calibration_overlay_enabled.get(camera_id, False):
                try:
                    # Decode JPEG to numpy array
                    nparr = np.frombuffer(frame_jpeg, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        # Get calibration plugin
                        calibration_plugin = feature_manager.get_plugin('Camera Calibration')
                        if calibration_plugin:
                            # Configure board if needed
                            board_config = calibration_board_config.get(camera_id, {})
                            if board_config:
                                calibration_plugin.configure_board(
                                    width=board_config.get('width', 8),
                                    height=board_config.get('height', 5),
                                    square_length=board_config.get('square_length', 50.0),
                                    marker_length=board_config.get('marker_length', 37.0),
                                    dictionary=board_config.get('dictionary', 'DICT_6X6_100')
                                )
                            
                            # Detect pattern directly on preview frame (faster)
                            detection = calibration_plugin.detect_charuco_pattern(frame)
                            
                            # Draw markers on preview frame
                            if detection.get('detected'):
                                try:
                                    # Draw ArUco markers using OpenCV's built-in function
                                    if detection.get('marker_corners') is not None and detection.get('marker_ids') is not None:
                                        marker_corners = detection['marker_corners']
                                        marker_ids = detection['marker_ids']
                                        
                                        # Draw detected markers directly (no scaling needed)
                                        cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
                                    
                                    # Draw ChArUco corners
                                    if detection.get('charuco_corners_array') is not None:
                                        corners = detection['charuco_corners_array']
                                        # Reshape to ensure it's 2D array of points
                                        if corners.ndim == 3:
                                            corners = corners.reshape(-1, 2)
                                        for corner in corners:
                                            # Extract x, y and convert to Python scalars (no scaling needed)
                                            x = int(corner.flatten()[0])
                                            y = int(corner.flatten()[1])
                                            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                                    
                                    # Draw info overlay
                                    info_text = f"Markers: {detection.get('markers_detected', 0)} | Corners: {detection.get('corners_detected', 0)} | Q: {detection.get('quality', 'N/A')}"
                                    cv2.rectangle(frame, (5, 5), (frame.shape[1] - 5, 35), (0, 0, 0), -1)
                                    cv2.putText(frame, info_text, (10, 25),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                except Exception as draw_error:
                                    import traceback
                                    print(f"Error drawing calibration overlay: {draw_error}")
                                    traceback.print_exc()
                    
                    # Re-encode to JPEG
                    _, frame_jpeg = cv2.imencode('.jpg', frame)
                    frame_jpeg = frame_jpeg.tobytes()
                except Exception as e:
                    print(f"Error drawing calibration overlay: {e}")
                    # Fall through to use original frame
            
            # Apply image processing if enabled
            if frame_jpeg and image_processing_enabled.get(camera_id, False):
                try:
                    # Decode JPEG to numpy array
                    nparr = np.frombuffer(frame_jpeg, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        processing_config = image_processing_config.get(camera_id, {})
                        processing_type = processing_config.get('type', 'undistort')
                        
                        if processing_type == 'undistort':
                            # Load calibration data
                            calibration_plugin = feature_manager.get_plugin('Camera Calibration')
                            calibration_file = processing_config.get('calibration_file')
                            if calibration_file and os.path.exists(calibration_file) and calibration_plugin:
                                calibration_data = calibration_plugin.load_calibration(calibration_file)
                                if calibration_data:
                                    camera_matrix = np.array(calibration_data['camera_matrix'])
                                    dist_coeffs = np.array(calibration_data['distortion_coeffs'])
                                    
                                    # Scale calibration for preview resolution
                                    # Calibration is done at full resolution, but preview is lower resolution
                                    h, w = frame.shape[:2]
                                    original_size = tuple(calibration_data.get('image_size', [w, h]))
                                    target_size = (w, h)
                                    
                                    # Only scale if resolutions differ
                                    if original_size != target_size:
                                        camera_matrix, dist_coeffs = calibration_plugin.scale_calibration_for_resolution(
                                            camera_matrix, dist_coeffs, original_size, target_size
                                        )
                                    
                                    # Get alpha parameter (0 = crop all invalid pixels, 1 = keep all pixels)
                                    alpha = processing_config.get('alpha', 0.0)
                                    
                                    # Optionally use optimal camera matrix
                                    if alpha > 0:
                                        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                                            camera_matrix, dist_coeffs, (w, h), alpha
                                        )
                                        frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
                                    else:
                                        frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
                        
                        elif processing_type == 'perspective':
                            # Apply perspective transformation
                            src_points = np.array(processing_config.get('src_points', []), dtype=np.float32)
                            dst_points = np.array(processing_config.get('dst_points', []), dtype=np.float32)
                            
                            if len(src_points) == 4 and len(dst_points) == 4:
                                h, w = frame.shape[:2]
                                matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                                frame = cv2.warpPerspective(frame, matrix, (w, h))
                        
                        elif processing_type == 'affine':
                            # Apply affine transformation
                            src_points = np.array(processing_config.get('src_points', []), dtype=np.float32)
                            dst_points = np.array(processing_config.get('dst_points', []), dtype=np.float32)
                            
                            if len(src_points) == 3 and len(dst_points) == 3:
                                h, w = frame.shape[:2]
                                matrix = cv2.getAffineTransform(src_points, dst_points)
                                frame = cv2.warpAffine(frame, matrix, (w, h))
                        
                        elif processing_type == 'rotation':
                            # Apply rotation
                            angle = processing_config.get('angle', 0.0)
                            scale = processing_config.get('scale', 1.0)
                            h, w = frame.shape[:2]
                            center = (w // 2, h // 2)
                            matrix = cv2.getRotationMatrix2D(center, angle, scale)
                            frame = cv2.warpAffine(frame, matrix, (w, h))
                        
                        elif processing_type == 'custom_matrix':
                            # Apply custom transformation matrix
                            matrix_data = processing_config.get('matrix', [])
                            if len(matrix_data) == 9:  # 3x3 perspective
                                matrix = np.array(matrix_data, dtype=np.float32).reshape(3, 3)
                                h, w = frame.shape[:2]
                                frame = cv2.warpPerspective(frame, matrix, (w, h))
                            elif len(matrix_data) == 6:  # 2x3 affine
                                matrix = np.array(matrix_data, dtype=np.float32).reshape(2, 3)
                                h, w = frame.shape[:2]
                                frame = cv2.warpAffine(frame, matrix, (w, h))
                        
                        # Re-encode to JPEG
                        _, frame_jpeg = cv2.imencode('.jpg', frame)
                        frame_jpeg = frame_jpeg.tobytes()
                except Exception as e:
                    print(f"Error applying image processing: {e}")
                    import traceback
                    traceback.print_exc()
                    # Fall through to use original frame
            
            if frame_jpeg:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_jpeg + b'\r\n')
            time.sleep(0.033)  # ~30 FPS
    
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/cameras/composite/<camera0_id>/<camera1_id>/stream')
def composite_camera_stream(camera0_id, camera1_id):
    """MJPEG composite stream for two synchronized cameras side-by-side."""
    import cv2
    import numpy as np
    
    print(f"[Composite Stream] Stream requested for cameras {camera0_id} and {camera1_id}")
    
    # Get/create and start synchronized pair BEFORE creating the response
    # This avoids blocking inside the generator which would prevent HTTP headers from being sent
    sync_pair = sync_pair_manager.get_pair(camera0_id, camera1_id)
    
    if sync_pair is None:
        # If no sync pair, try to create one
        print(f"[Composite Stream] No existing pair, creating new one")
        sync_pair = sync_pair_manager.create_pair(camera0_id, camera1_id)
        if sync_pair:
            print(f"[Composite Stream] Starting new sync pair...")
            sync_pair.start()
    elif not sync_pair.is_running:
        # Pair exists but not running - start it
        print(f"[Composite Stream] Sync pair exists but not running, starting it")
        sync_pair.start()
    
    if sync_pair is None:
        print(f"[Composite Stream] Failed to get or create sync pair")
        return jsonify({"success": False, "error": "Failed to create camera pair"}), 500
    
    print(f"[Composite Stream] Sync pair ready, is_running={sync_pair.is_running}")
    
    def generate():
        """Generate composite MJPEG stream."""
        frame_count = 0
        
        while sync_pair.is_running:
            # Get composite preview frame (already JPEG encoded and side-by-side)
            composite_jpeg = sync_pair.get_preview_frame()
            
            if composite_jpeg:
                frame_count += 1
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + composite_jpeg + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
    
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/cameras/composite/<camera0_id>/<camera1_id>/stream-with-overlay')
def composite_camera_stream_with_overlay(camera0_id, camera1_id):
    """MJPEG composite stream with ChArUco detection overlay for panorama calibration."""
    import cv2
    import numpy as np
    
    print(f"[Composite Stream Overlay] Stream requested for cameras {camera0_id} and {camera1_id}")
    
    # Get board config from query parameters
    board_width = int(request.args.get('board_width', 8))
    board_height = int(request.args.get('board_height', 5))
    square_length = float(request.args.get('square_length', 50))
    marker_length = float(request.args.get('marker_length', 37))
    dictionary = request.args.get('dictionary', 'DICT_6X6_100')
    
    board_config = {
        'width': board_width,
        'height': board_height,
        'square_length': square_length,
        'marker_length': marker_length,
        'dictionary': dictionary
    }
    
    # Get/create and start synchronized pair
    sync_pair = sync_pair_manager.get_pair(camera0_id, camera1_id)
    
    if sync_pair is None:
        print(f"[Composite Stream Overlay] No existing pair, creating new one")
        sync_pair = sync_pair_manager.create_pair(camera0_id, camera1_id)
        if sync_pair:
            sync_pair.start()
    elif not sync_pair.is_running:
        print(f"[Composite Stream Overlay] Sync pair exists but not running, starting it")
        sync_pair.start()
    
    if sync_pair is None:
        print(f"[Composite Stream Overlay] Failed to get or create sync pair")
        return jsonify({"success": False, "error": "Failed to create camera pair"}), 500
    
    print(f"[Composite Stream Overlay] Sync pair ready, is_running={sync_pair.is_running}")
    
    def generate():
        """Generate composite MJPEG stream with detection overlay."""
        from backend.panorama_utils import detect_charuco_for_panorama
        
        while sync_pair.is_running:
            # Get composite preview frame (JPEG encoded)
            composite_jpeg = sync_pair.get_preview_frame()
            
            if composite_jpeg:
                try:
                    # Decode JPEG to draw overlay
                    nparr = np.frombuffer(composite_jpeg, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        # Split side-by-side composite into two frames
                        height, width = frame.shape[:2]
                        half_width = width // 2
                        frame1 = frame[:, :half_width]
                        frame2 = frame[:, half_width:]
                        
                        # Detect in both frames
                        detection1 = detect_charuco_for_panorama(frame1, board_config)
                        detection2 = detect_charuco_for_panorama(frame2, board_config)
                        
                        # Draw overlays
                        if detection1:
                            cv2.aruco.drawDetectedCornersCharuco(
                                frame1, 
                                detection1['corners'],
                                detection1['ids']
                            )
                            # Add text indicator
                            cv2.putText(frame1, f"{detection1['num_corners']} corners", 
                                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        else:
                            cv2.putText(frame1, "No board", 
                                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        if detection2:
                            cv2.aruco.drawDetectedCornersCharuco(
                                frame2, 
                                detection2['corners'],
                                detection2['ids']
                            )
                            cv2.putText(frame2, f"{detection2['num_corners']} corners", 
                                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        else:
                            cv2.putText(frame2, "No board", 
                                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Recombine frames
                        frame[:, :half_width] = frame1
                        frame[:, half_width:] = frame2
                        
                        # Re-encode to JPEG
                        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if success:
                            jpeg_data = buffer.tobytes()
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg_data + b'\r\n')
                    else:
                        # Fallback to original frame if decode fails
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + composite_jpeg + b'\r\n')
                        
                except Exception as e:
                    print(f"[Composite Stream Overlay] Error processing frame: {e}")
                    # Fallback to original frame on error
                    if composite_jpeg:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + composite_jpeg + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
    
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/cameras/<camera_id>/controls', methods=['GET'])
def get_camera_controls(camera_id):
    """Get available controls for a camera."""
    try:
        controls = camera_backend.get_controls(camera_id)
        
        # Add resolution info if camera is streaming
        resolution_info = None
        if camera_backend.is_streaming(camera_id):
            if hasattr(camera_backend, 'active_cameras') and camera_id in camera_backend.active_cameras:
                config = camera_backend.active_cameras[camera_id].get('config', {})
                resolution_info = {
                    'capture_width': config.get('width', 0),
                    'capture_height': config.get('height', 0),
                    'capture_fps': config.get('fps', 0),
                    'preview_width': config.get('preview_width', 0),
                    'preview_height': config.get('preview_height', 0)
                }
        
        return jsonify({
            'success': True,
            'controls': controls,
            'resolution_info': resolution_info
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/cameras/<camera_id>/control/<control_name>', methods=['GET'])
def get_control_value(camera_id, control_name):
    """Get current value of a control."""
    try:
        value = camera_backend.get_control(camera_id, control_name)
        return jsonify({
            'success': True,
            'value': value
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/cameras/<camera_id>/control/<control_name>', methods=['PUT'])
def set_control_value(camera_id, control_name):
    """Set a control value."""
    try:
        data = request.json
        value = data.get('value')
        
        success = camera_backend.set_control(camera_id, control_name, value)
        return jsonify({
            'success': success,
            'message': f'Control {control_name} {"set" if success else "failed"}'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/cameras/<camera_id>/profiles', methods=['GET'])
def list_profiles(camera_id):
    """List available profiles."""
    try:
        profiles = camera_backend.get_profiles()
        return jsonify({
            'success': True,
            'profiles': profiles
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/cameras/<camera_id>/profiles/<profile_name>', methods=['GET'])
def get_profile(camera_id, profile_name):
    """Get a specific profile."""
    try:
        profile = camera_backend.get_profile(profile_name)
        if profile:
            return jsonify({
                'success': True,
                'profile': profile
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Profile not found'
            }), 404
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/cameras/<camera_id>/profiles/<profile_name>', methods=['POST', 'PUT'])
def save_profile_endpoint(camera_id, profile_name):
    """Save a camera profile."""
    try:
        settings = request.json
        success = camera_backend.save_profile(profile_name, settings)
        return jsonify({
            'success': success,
            'message': f'Profile {profile_name} {"saved" if success else "failed to save"}'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/cameras/<camera_id>/profiles/<profile_name>', methods=['DELETE'])
def delete_profile_endpoint(camera_id, profile_name):
    """Delete a profile."""
    try:
        success = camera_backend.delete_profile(profile_name)
        return jsonify({
            'success': success,
            'message': f'Profile {profile_name} {"deleted" if success else "failed to delete"}'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/cameras/<camera_id>/profile', methods=['POST'])
def apply_profile_endpoint(camera_id):
    """Apply a profile to a camera."""
    try:
        data = request.json
        profile_name = data.get('profile_name')
        
        success = camera_backend.apply_profile(camera_id, profile_name)
        return jsonify({
            'success': success,
            'message': f'Profile {profile_name} {"applied" if success else "failed to apply"}'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# ============================================================================
# Camera Group API Routes
# ============================================================================

@app.route('/api/camera-groups', methods=['GET'])
def list_camera_groups():
    """List all camera groups."""
    try:
        groups = [g.to_dict() for g in group_manager.list_groups()]
        return jsonify({
            'success': True,
            'groups': groups
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/camera-groups', methods=['POST'])
def create_camera_group():
    """Create a new camera group."""
    try:
        data = request.json
        group = group_manager.create_group(
            name=data['name'],
            camera_ids=data['camera_ids'],
            group_type=data.get('group_type', 'custom'),
            metadata=data.get('metadata')
        )
        return jsonify({
            'success': True,
            'group': group.to_dict()
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/camera-groups/<group_id>', methods=['GET'])
def get_camera_group(group_id):
    """Get a specific camera group."""
    try:
        group = group_manager.get_group(group_id)
        if group:
            return jsonify({
                'success': True,
                'group': group.to_dict()
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Group not found'
            }), 404
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/camera-groups/<group_id>', methods=['PUT'])
def update_camera_group(group_id):
    """Update a camera group."""
    try:
        data = request.json
        success = group_manager.update_group(
            group_id,
            name=data.get('name'),
            camera_ids=data.get('camera_ids'),
            metadata=data.get('metadata')
        )
        return jsonify({
            'success': success,
            'message': f'Group {"updated" if success else "failed to update"}'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/camera-groups/<group_id>', methods=['DELETE'])
def delete_camera_group(group_id):
    """Delete a camera group."""
    try:
        success = group_manager.delete_group(group_id)
        return jsonify({
            'success': success,
            'message': f'Group {"deleted" if success else "failed to delete"}'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/camera-groups/<group_id>/start', methods=['POST'])
def start_camera_group(group_id):
    """Start all cameras in a group synchronously."""
    try:
        group = group_manager.get_group(group_id)
        if not group:
            return jsonify({'success': False, 'message': 'Group not found'}), 404
        
        config = request.json or {}
        results = {}
        
        for camera_id in group.camera_ids:
            results[camera_id] = camera_backend.start_stream(camera_id, config)
        
        all_success = all(results.values())
        return jsonify({
            'success': all_success,
            'results': results,
            'message': f'Group {"started" if all_success else "partially started"}'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/camera-groups/<group_id>/calibration', methods=['GET'])
def get_group_calibration(group_id):
    """Get calibration data for a group."""
    try:
        calibration = group_manager.get_calibration(group_id)
        if calibration:
            return jsonify({
                'success': True,
                'calibration': calibration
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No calibration data found'
            }), 404
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/camera-groups/<group_id>/calibration', methods=['POST'])
def save_group_calibration(group_id):
    """Save calibration data for a group."""
    try:
        calibration_data = request.json
        success = group_manager.save_calibration(group_id, calibration_data)
        return jsonify({
            'success': success,
            'message': f'Calibration {"saved" if success else "failed to save"}'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# ============================================================================
# Feature Plugin API Routes
# ============================================================================

@app.route('/api/features', methods=['GET'])
def list_features():
    """List all available feature plugins."""
    try:
        features = feature_manager.list_plugins()
        return jsonify({
            'success': True,
            'features': features
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/features/<plugin_name>/process', methods=['POST'])
def process_with_feature(plugin_name):
    """Process frames with a feature plugin."""
    try:
        data = request.json
        camera_ids = data.get('camera_ids', [])
        params = data.get('params', {})
        
        result = feature_manager.process_with_plugin(
            plugin_name, camera_backend, camera_ids, params
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/features/<plugin_name>/process-group', methods=['POST'])
def process_group_with_feature(plugin_name):
    """Process a camera group with a feature plugin."""
    try:
        data = request.json
        group_id = data.get('group_id')
        params = data.get('params', {})
        
        group = group_manager.get_group(group_id)
        if not group:
            return jsonify({'success': False, 'message': 'Group not found'}), 404
        
        result = feature_manager.process_group_with_plugin(
            plugin_name, camera_backend, group, params
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# ============================================================================
# Workflow API Routes
# ============================================================================

@app.route('/api/workflows', methods=['GET'])
def list_workflows():
    """List all available workflows."""
    try:
        workflows = workflow_manager.list_workflows()
        return jsonify({
            'success': True,
            'workflows': workflows
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/workflows/<workflow_name>/start', methods=['POST'])
def start_workflow(workflow_name):
    """Start a new workflow instance."""
    try:
        result = workflow_manager.start_workflow(workflow_name)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/workflows/<instance_id>/state', methods=['GET'])
def get_workflow_state(instance_id):
    """Get workflow instance state."""
    try:
        state = workflow_manager.get_workflow_state(instance_id)
        if state:
            return jsonify({
                'success': True,
                'state': state
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Workflow instance not found'
            }), 404
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/workflows/<instance_id>/step/<step_id>', methods=['POST'])
def execute_workflow_step(instance_id, step_id):
    """Execute a workflow step."""
    try:
        data = request.json or {}
        result = workflow_manager.execute_step(
            instance_id, step_id, data, camera_backend, group_manager
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/workflows/<instance_id>/reset', methods=['POST'])
def reset_workflow(instance_id):
    """Reset a workflow instance."""
    try:
        success = workflow_manager.reset_workflow(instance_id)
        return jsonify({
            'success': success,
            'message': f'Workflow {"reset" if success else "failed to reset"}'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/workflows/<instance_id>', methods=['DELETE'])
def delete_workflow(instance_id):
    """Delete a workflow instance."""
    try:
        success = workflow_manager.delete_workflow(instance_id)
        return jsonify({
            'success': success,
            'message': f'Workflow {"deleted" if success else "failed to delete"}'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# ============================================================================
# Profile File Serving
# ============================================================================

@app.route('/profiles/<platform>/<filename>')
def serve_profile(platform, filename):
    """Serve profile files for download/viewing."""
    try:
        profile_dir = Path(__file__).parent / 'backend' / 'camera' / 'profiles' / platform
        return send_from_directory(profile_dir, filename)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 404


# ============================================================================
# Sensor Configuration Routes
# ============================================================================

@app.route('/api/sensor-config', methods=['GET'])
def get_sensor_config():
    """Get current sensor FPS configuration"""
    try:
        # Convert tuple keys to strings for JSON
        fps_map = {f"{w}x{h}": fps for (w, h), fps in camera_backend.RESOLUTION_FPS_MAP.items()}
        
        return jsonify({
            'success': True,
            'fps_map': fps_map,
            'default_fps': camera_backend.DEFAULT_FPS,
            'current_sensor': camera_backend.current_sensor,
            'available_sensors': camera_backend._get_available_sensors()
        })
    except Exception as e:
        logger.error(f"Error getting sensor config: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/sensor-config/<sensor>', methods=['GET'])
def get_specific_sensor_config(sensor):
    """Get configuration for a specific sensor"""
    try:
        import os
        
        # Check if sensor has a saved config
        sensor_file = os.path.join(camera_backend.sensor_config_dir, f'{sensor}.json')
        
        if os.path.exists(sensor_file):
            with open(sensor_file, 'r') as f:
                data = json.load(f)
                return jsonify({
                    'success': True,
                    'fps_map': data.get('custom_fps_map', {}),
                    'default_fps': data.get('custom_default_fps', 30)
                })
        elif sensor in camera_backend.SENSOR_CONFIGS:
            # Return predefined config
            config = camera_backend.SENSOR_CONFIGS[sensor]
            # Convert tuple keys to strings
            fps_map = {f"{w}x{h}": fps for (w, h), fps in config['fps_map'].items()}
            return jsonify({
                'success': True,
                'fps_map': fps_map,
                'default_fps': config['default_fps']
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Sensor {sensor} not found'
            }), 404
    except Exception as e:
        logger.error(f"Error getting sensor {sensor} config: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/api/sensor-config', methods=['PUT'])
def update_sensor_config():
    """Update sensor FPS configuration"""
    try:
        data = request.json
        sensor = data.get('sensor')
        fps_map = data.get('fps_map', {})  # Already in string format from frontend
        default_fps = data.get('default_fps', 30)
        
        # Save the configuration
        camera_backend._save_sensor_config(sensor, fps_map, default_fps)
        
        return jsonify({
            'success': True,
            'message': 'Sensor configuration updated successfully'
        })
    except Exception as e:
        logger.error(f"Error updating sensor config: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/sensor-config/<sensor>', methods=['DELETE'])
def delete_sensor_config(sensor):
    """Delete a sensor configuration"""
    try:
        import os
        
        # Don't allow deleting predefined sensors
        if sensor in ['IMX219', 'IMX477']:
            return jsonify({
                'success': False,
                'message': f'Cannot delete predefined sensor {sensor}'
            }), 400
        
        # Delete the sensor config file
        sensor_file = os.path.join(camera_backend.sensor_config_dir, f'{sensor}.json')
        if os.path.exists(sensor_file):
            os.remove(sensor_file)
        
        # If this was the active sensor, switch to IMX219
        active_file = os.path.join(camera_backend.sensor_config_dir, 'active_sensor.json')
        if os.path.exists(active_file):
            with open(active_file, 'r') as f:
                active_config = json.load(f)
                if active_config.get('sensor') == sensor:
                    # Reset to IMX219 default
                    camera_backend.current_sensor = 'IMX219'
                    camera_backend.RESOLUTION_FPS_MAP = camera_backend.SENSOR_CONFIGS['IMX219']['fps_map'].copy()
                    camera_backend.DEFAULT_FPS = camera_backend.SENSOR_CONFIGS['IMX219']['default_fps']
                    camera_backend._save_sensor_config('IMX219', 
                                                      camera_backend.RESOLUTION_FPS_MAP, 
                                                      camera_backend.DEFAULT_FPS)
        
        return jsonify({
            'success': True,
            'message': f'Sensor {sensor} deleted successfully'
        })
    except Exception as e:
        logger.error(f"Error deleting sensor config: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


# ============================================================================
# Calibration API Endpoints
# ============================================================================

@app.route('/api/calibration/overlay/<camera_id>/enable', methods=['POST'])
def enable_calibration_overlay(camera_id):
    """Enable calibration marker overlay on camera stream."""
    try:
        global calibration_overlay_enabled, calibration_board_config
        
        data = request.json or {}
        board_config = data.get('board_config', {})
        
        calibration_overlay_enabled[camera_id] = True
        calibration_board_config[camera_id] = board_config
        
        return jsonify({
            'success': True,
            'message': f'Calibration overlay enabled for camera {camera_id}'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/api/calibration/overlay/<camera_id>/disable', methods=['POST'])
def disable_calibration_overlay(camera_id):
    """Disable calibration marker overlay on camera stream."""
    try:
        global calibration_overlay_enabled, calibration_board_config
        
        calibration_overlay_enabled[camera_id] = False
        if camera_id in calibration_board_config:
            del calibration_board_config[camera_id]
        
        return jsonify({
            'success': True,
            'message': f'Calibration overlay disabled for camera {camera_id}'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/api/calibration/detect-pattern', methods=['POST'])
def detect_calibration_pattern():
    """Detect ChArUco pattern in current camera frame."""
    try:
        data = request.json
        camera_id = data.get('camera_id')
        board_config = data.get('board_config', {})
        
        if not camera_id:
            return jsonify({
                'success': False,
                'message': 'Camera ID is required'
            }), 400
        
        # Get calibration plugin
        calibration_plugin = feature_manager.get_plugin('Camera Calibration')
        if not calibration_plugin:
            return jsonify({
                'success': False,
                'message': 'Calibration plugin not available'
            }), 500
        
        # Configure board if parameters provided
        if board_config:
            calibration_plugin.configure_board(
                width=board_config.get('width', 8),
                height=board_config.get('height', 5),
                square_length=board_config.get('square_length', 50.0),
                marker_length=board_config.get('marker_length', 37.0),
                dictionary=board_config.get('dictionary', 'DICT_6X6_100')
            )
        
        # Get current frame from camera
        frame = camera_backend.get_full_frame(camera_id)
        if frame is None:
            # Try to get the camera info and ensure it's active
            camera_info = camera_backend.get_camera(camera_id)
            if camera_info and camera_info.get('active'):
                return jsonify({
                    'success': False,
                    'message': f'Camera {camera_id} is active but failed to get frame. Check camera connection.'
                }), 500
            else:
                return jsonify({
                    'success': False,
                    'message': f'Camera {camera_id} is not streaming. Please start the camera preview first.'
                }), 400
        
        # Detect pattern
        detection = calibration_plugin.detect_charuco_pattern(frame)
        
        # Prepare response with detection results
        response_data = {
            'success': True,
            'detection': {
                'detected': detection.get('detected', False),
                'markers_detected': detection.get('markers_detected', 0),
                'corners_detected': detection.get('corners_detected', 0),
                'quality': detection.get('quality', 'Unknown'),
                'image_width': frame.shape[1] if frame is not None else 0,
                'image_height': frame.shape[0] if frame is not None else 0,
            }
        }
        
        # Add marker corners if detected
        if detection.get('marker_corners') is not None:
            # Convert numpy arrays to lists for JSON serialization
            marker_corners = detection['marker_corners']
            response_data['detection']['marker_corners'] = [
                corners.reshape(-1, 2).tolist() for corners in marker_corners
            ]
        
        # Add marker IDs if available
        if detection.get('marker_ids') is not None:
            response_data['detection']['marker_ids'] = detection['marker_ids'].flatten().tolist()
        
        # Add ChArUco corners if detected
        if detection.get('charuco_corners_array') is not None:
            corners = detection['charuco_corners_array']
            if corners is not None and len(corners) > 0:
                response_data['detection']['charuco_corners'] = corners.reshape(-1, 2).tolist()
        
        # Classify pose of detected pattern
        if detection.get('detected') and detection.get('marker_corners') is not None:
            pose_classification = classify_board_pose(
                detection['marker_corners'],
                frame.shape[1],  # width
                frame.shape[0]   # height
            )
            response_data['detection']['poses'] = pose_classification
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/api/calibration/capture-image', methods=['POST'])
def capture_calibration_image():
    """Capture and save a calibration image."""
    try:
        import cv2
        from datetime import datetime
        
        data = request.json
        camera_id = data.get('camera_id')
        board_config = data.get('board_config', {})
        session_name = data.get('session_name')
        
        if not camera_id:
            return jsonify({
                'success': False,
                'message': 'Camera ID is required'
            }), 400
        
        # Get current frame from camera
        frame = camera_backend.get_full_frame(camera_id)
        if frame is None:
            return jsonify({
                'success': False,
                'message': f'Failed to get frame from camera {camera_id}'
            }), 400
        
        # Create session directory if needed
        if not session_name:
            session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        calibration_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'calibration')
        session_dir = os.path.join(calibration_dir, session_name, camera_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Save session metadata if it doesn't exist
        metadata_file = os.path.join(session_dir, 'session_info.json')
        if not os.path.exists(metadata_file):
            import json
            metadata = {
                'camera_id': camera_id,
                'board_config': board_config,
                'model': data.get('model', 'pinhole'),
                'target_images': data.get('target_images', 20),
                'created_at': datetime.now().isoformat()
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Count existing images to determine filename
        import glob
        existing_images = glob.glob(os.path.join(session_dir, 'image_*.jpg'))
        image_num = len(existing_images) + 1
        
        # Save image
        image_filename = f"image_{image_num:04d}.jpg"
        image_path = os.path.join(session_dir, image_filename)
        cv2.imwrite(image_path, frame)
        
        return jsonify({
            'success': True,
            'session_name': session_name,
            'image_path': image_path,
            'image_filename': image_filename,
            'image_number': image_num
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/api/calibration/analyze-poses', methods=['POST'])
def analyze_calibration_poses():
    """Analyze pose diversity of captured calibration images."""
    try:
        data = request.json
        detections = data.get('detections', [])
        image_width = data.get('image_width', 1920)
        image_height = data.get('image_height', 1080)
        
        if not detections:
            return jsonify({
                'success': False,
                'message': 'No detections provided'
            }), 400
        
        # Initialize pose counts
        pose_counts = {
            'center': 0,
            'topLeft': 0,
            'topRight': 0,
            'bottomLeft': 0,
            'bottomRight': 0,
            'topEdge': 0,
            'bottomEdge': 0,
            'leftEdge': 0,
            'rightEdge': 0,
            'tilted': 0,
            'close': 0,
            'far': 0
        }
        
        # Analyze each detection
        for detection in detections:
            marker_corners = detection.get('marker_corners', [])
            if not marker_corners:
                continue
            
            # Calculate bounding box of all markers
            min_x = min_y = float('inf')
            max_x = max_y = float('-inf')
            
            for marker in marker_corners:
                for corner in marker:
                    min_x = min(min_x, corner[0])
                    min_y = min(min_y, corner[1])
                    max_x = max(max_x, corner[0])
                    max_y = max(max_y, corner[1])
            
            # Calculate properties
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            width = max_x - min_x
            height = max_y - min_y
            area = width * height
            image_area = image_width * image_height
            scale = area / image_area
            
            # Position classification
            margin_x = image_width * 0.25
            margin_y = image_height * 0.25
            
            # Check corners
            if center_x < margin_x and center_y < margin_y:
                pose_counts['topLeft'] += 1
            elif center_x > image_width - margin_x and center_y < margin_y:
                pose_counts['topRight'] += 1
            elif center_x < margin_x and center_y > image_height - margin_y:
                pose_counts['bottomLeft'] += 1
            elif center_x > image_width - margin_x and center_y > image_height - margin_y:
                pose_counts['bottomRight'] += 1
            # Check edges
            elif center_y < margin_y:
                pose_counts['topEdge'] += 1
            elif center_y > image_height - margin_y:
                pose_counts['bottomEdge'] += 1
            elif center_x < margin_x:
                pose_counts['leftEdge'] += 1
            elif center_x > image_width - margin_x:
                pose_counts['rightEdge'] += 1
            # Center region
            else:
                pose_counts['center'] += 1
            
            # Scale classification
            if scale > 0.15:
                pose_counts['close'] += 1
            elif scale < 0.05:
                pose_counts['far'] += 1
            
            # Tilt classification
            aspect_ratio = width / height if height > 0 else 1
            if aspect_ratio < 0.7 or aspect_ratio > 1.43:
                pose_counts['tilted'] += 1
        
        # Calculate diversity metrics
        total_pose_types = 12
        captured_pose_types = sum(1 for count in pose_counts.values() if count > 0)
        diversity_percent = round((captured_pose_types / total_pose_types) * 100)
        
        # Recommended counts
        recommended = {
            'center': 3,
            'corners': 8,  # 2 per corner
            'edges': 8,    # 2 per edge
            'tilted': 4,
            'close': 2,
            'far': 2
        }
        
        corner_count = (pose_counts['topLeft'] + pose_counts['topRight'] + 
                       pose_counts['bottomLeft'] + pose_counts['bottomRight'])
        edge_count = (pose_counts['topEdge'] + pose_counts['bottomEdge'] + 
                     pose_counts['leftEdge'] + pose_counts['rightEdge'])
        
        # Calculate completeness
        completeness = {
            'center': pose_counts['center'] >= recommended['center'],
            'corners': corner_count >= recommended['corners'],
            'edges': edge_count >= recommended['edges'],
            'tilted': pose_counts['tilted'] >= recommended['tilted'],
            'close': pose_counts['close'] >= recommended['close'],
            'far': pose_counts['far'] >= recommended['far']
        }
        
        return jsonify({
            'success': True,
            'pose_counts': pose_counts,
            'diversity_percent': diversity_percent,
            'captured_pose_types': captured_pose_types,
            'total_pose_types': total_pose_types,
            'recommended': recommended,
            'completeness': completeness,
            'summary': {
                'corner_count': corner_count,
                'edge_count': edge_count
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/api/calibration/sessions', methods=['GET'])
def list_calibration_sessions():
    """List all available calibration sessions."""
    try:
        import glob
        from datetime import datetime
        
        # Get all calibration session directories (both regular and panorama sessions)
        calibration_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'calibration')
        session_dirs = glob.glob(os.path.join(calibration_dir, 'session_*'))
        panorama_dirs = glob.glob(os.path.join(calibration_dir, 'panorama_*'))
        
        # Also check for old-format panorama sessions (camera_id pairs like "0_1_timestamp")
        # These have session_info.json at the root level
        all_dirs = glob.glob(os.path.join(calibration_dir, '*'))
        for dir_path in all_dirs:
            if os.path.isdir(dir_path) and dir_path not in session_dirs and dir_path not in panorama_dirs:
                # Check if it has session_info.json (indicates panorama session)
                session_info_path = os.path.join(dir_path, 'session_info.json')
                if os.path.exists(session_info_path):
                    panorama_dirs.append(dir_path)
        
        all_session_dirs = session_dirs + panorama_dirs
        
        sessions = []
        for session_dir in sorted(all_session_dirs, reverse=True):  # Most recent first
            try:
                session_name = os.path.basename(session_dir)
                
                # Parse timestamp from session name (format: session_YYYYMMDD_HHMMSS or panorama_YYYYMMDD_HHMMSS)
                timestamp_str = session_name.replace('session_', '').replace('panorama_', '')
                try:
                    timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                    date_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    date_str = timestamp_str
                
                # Check for camera subdirectories
                camera_dirs = glob.glob(os.path.join(session_dir, '*'))
                camera_dirs = [d for d in camera_dirs if os.path.isdir(d)]
                
                for camera_dir in camera_dirs:
                    camera_id = os.path.basename(camera_dir)
                    
                    # Count images in this camera's directory
                    image_files = glob.glob(os.path.join(camera_dir, '*.jpg')) + \
                                 glob.glob(os.path.join(camera_dir, '*.png'))
                    image_count = len(image_files)
                    
                    # Check if calibration results exist
                    calibration_file = os.path.join(camera_dir, 'calibration_results.json')
                    calibrated = os.path.exists(calibration_file)
                    
                    # Load calibration data if it exists
                    reprojection_error = None
                    num_images_used = None
                    if calibrated:
                        try:
                            import json
                            with open(calibration_file, 'r') as f:
                                calib_data = json.load(f)
                                reprojection_error = calib_data.get('reprojection_error')
                                num_images_used = calib_data.get('num_images_used')
                        except:
                            pass
                    
                    if image_count > 0:  # Only include sessions with images
                        session_info = {
                            'path': session_name,
                            'camera_id': camera_id,
                            'date': date_str,
                            'image_count': image_count,
                            'num_images': image_count,
                            'calibrated': calibrated,
                            'timestamp': date_str
                        }
                        
                        if reprojection_error is not None:
                            session_info['reprojection_error'] = reprojection_error
                        if num_images_used is not None:
                            session_info['num_images_used'] = num_images_used
                        
                        sessions.append(session_info)
            except Exception as e:
                print(f"Error processing session {session_dir}: {e}")
                continue
        
        # Also create a simple list of unique session names for dropdowns
        session_names = list(set([s['path'] for s in sessions]))
        session_names.sort(reverse=True)
        
        return jsonify({
            'success': True,
            'sessions': session_names,  # Simple list for dropdown
            'session_details': sessions  # Detailed info for session management
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/api/calibration/session/<path:session_path>', methods=['GET'])
def get_calibration_session(session_path):
    """Load a specific calibration session."""
    try:
        import glob
        import json
        import cv2
        
        # Sanitize path to prevent directory traversal
        session_path = os.path.basename(session_path)
        
        # Build full path
        calibration_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'calibration')
        session_dir = os.path.join(calibration_dir, session_path)
        
        if not os.path.exists(session_dir):
            return jsonify({
                'success': False,
                'message': 'Session not found'
            }), 404
        
        # Find camera directories
        camera_dirs = glob.glob(os.path.join(session_dir, '*'))
        camera_dirs = [d for d in camera_dirs if os.path.isdir(d)]
        
        if not camera_dirs:
            return jsonify({
                'success': False,
                'message': 'No camera data found in session'
            }), 404
        
        # Use the first camera directory (in future could support multiple)
        camera_dir = camera_dirs[0]
        camera_id = os.path.basename(camera_dir)
        
        # Load session metadata if it exists
        metadata_file = os.path.join(camera_dir, 'session_info.json')
        board_config = None
        model = 'pinhole'
        target_images = 20
        
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                board_config = metadata.get('board_config')
                model = metadata.get('model', 'pinhole')
                target_images = metadata.get('target_images', 20)
        
        # Get calibration plugin for pattern detection
        calibration_plugin = feature_manager.get_plugin('Camera Calibration')
        
        # Configure board if we have config
        if board_config and calibration_plugin:
            calibration_plugin.configure_board(
                width=board_config.get('width', 8),
                height=board_config.get('height', 5),
                square_length=board_config.get('square_length', 50.0),
                marker_length=board_config.get('marker_length', 37.0),
                dictionary=board_config.get('dictionary', 'DICT_6X6_100')
            )
        
        # Get list of captured images
        image_files = sorted(glob.glob(os.path.join(camera_dir, '*.jpg')) + \
                           glob.glob(os.path.join(camera_dir, '*.png')))
        
        # For quick loading, just return basic image info without detection
        # Pattern detection will only run when specifically requested
        images = []
        for i, img_path in enumerate(image_files, 1):
            images.append({
                'id': i,
                'path': img_path,
                'filename': os.path.basename(img_path)
            })
        
        # Check if calibration results exist
        calibration_results = None
        calibration_results_path = os.path.join(camera_dir, 'calibration_results.json')
        if os.path.exists(calibration_results_path):
            try:
                with open(calibration_results_path, 'r') as f:
                    calibration_results = json.load(f)
            except Exception as e:
                print(f"Error loading calibration results: {e}")
                import traceback
                traceback.print_exc()
        
        return jsonify({
            'success': True,
            'session': {
                'path': session_path,
                'camera_id': camera_id,
                'model': model,
                'board_config': board_config,
                'target_images': target_images,
                'images': images,
                'calibration_results': calibration_results
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


# ============================================================================
# Frontend Routes
# ============================================================================

@app.route('/')
def index():
    """Serve the main web interface."""
    frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
    print(f'Serving index.html from: {frontend_dir}')  # Debug
    return send_from_directory(frontend_dir, 'index.html')


@app.route('/api/calibration/run', methods=['POST'])
def run_calibration():
    """Run camera calibration with real-time progress updates via Server-Sent Events."""
    try:
        from datetime import datetime
        import json as json_module
        
        data = request.json
        session_name = data.get('session_name')
        camera_id = data.get('camera_id')
        board_config = data.get('board_config')
        
        if not session_name or not camera_id or not board_config:
            return jsonify({
                'success': False,
                'error': 'Missing required parameters: session_name, camera_id, and board_config'
            }), 400
        
        # Build path to session directory
        calibration_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'data', 'calibration'
        ))
        session_path = os.path.join(calibration_dir, session_name, str(camera_id))
        
        if not os.path.exists(session_path):
            return jsonify({
                'success': False,
                'error': f'Session directory not found: {session_path}'
            }), 404
        
        def generate():
            """Generator function for SSE stream with real-time updates."""
            import sys
            from pathlib import Path
            
            try:
                yield f"data: {json_module.dumps({'type': 'progress', 'message': 'Starting calibration...'})}\n\n"
                sys.stdout.flush()
                
                # Get calibration plugin
                calibration_plugin = feature_manager.get_plugin('Camera Calibration')
                if not calibration_plugin:
                    yield f"data: {json_module.dumps({'type': 'error', 'error': 'Calibration plugin not available'})}\n\n"
                    return
                
                # Configure board
                yield f"data: {json_module.dumps({'type': 'progress', 'message': 'Configuring ChArUco board...'})}\n\n"
                sys.stdout.flush()
                
                calibration_plugin.configure_board(
                    width=board_config['width'],
                    height=board_config['height'],
                    square_length=board_config['square_length'],
                    marker_length=board_config['marker_length'],
                    dictionary=board_config['dictionary']
                )
                
                # Get image paths
                yield f"data: {json_module.dumps({'type': 'progress', 'message': 'Loading calibration images...'})}\n\n"
                sys.stdout.flush()
                
                image_files = sorted(Path(session_path).glob("image_*.jpg"))
                image_paths = [str(img) for img in image_files]
                
                if len(image_paths) < 5:
                    yield f"data: {json_module.dumps({'type': 'error', 'error': f'Insufficient images for calibration. Found {len(image_paths)}, need at least 5.'})}\n\n"
                    return
                
                yield f"data: {json_module.dumps({'type': 'progress', 'message': f'Found {len(image_paths)} images to process'})}\n\n"
                sys.stdout.flush()
                
                # Process images one by one with progress updates
                import cv2
                import numpy as np
                
                all_charuco_corners = []
                all_charuco_ids = []
                valid_image_paths = []  # Track which images had valid detections
                image_size = None
                
                for i, img_path in enumerate(image_paths):
                    yield f"data: {json_module.dumps({'type': 'progress', 'message': f'Processing image {i+1}/{len(image_paths)}: {Path(img_path).name}'})}\n\n"
                    sys.stdout.flush()
                    
                    image = cv2.imread(img_path)
                    if image is None:
                        continue
                    
                    if image_size is None:
                        image_size = image.shape[:2][::-1]
                    
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    marker_corners, marker_ids, _ = calibration_plugin.detector.detectMarkers(gray)
                    
                    if marker_ids is None or len(marker_corners) < calibration_plugin.markers_required:
                        continue
                    
                    # Use CharucoDetector for OpenCV 4.7+ compatibility
                    charuco_detector = cv2.aruco.CharucoDetector(calibration_plugin.board)
                    charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)
                    retval = len(charuco_corners) if charuco_corners is not None else 0
                    
                    if retval > 0:
                        all_charuco_corners.append(charuco_corners)
                        all_charuco_ids.append(charuco_ids)
                        valid_image_paths.append(img_path)  # Track the path of this valid image
                
                if len(all_charuco_corners) < 5:
                    yield f"data: {json_module.dumps({'type': 'error', 'error': f'Not enough valid images. Found {len(all_charuco_corners)} images with patterns, need at least 5.'})}\n\n"
                    return
                
                yield f"data: {json_module.dumps({'type': 'progress', 'message': f'Found {len(all_charuco_corners)} valid images with detected patterns'})}\n\n"
                sys.stdout.flush()
                
                # Prepare object and image points
                yield f"data: {json_module.dumps({'type': 'progress', 'message': 'Preparing calibration data...'})}\n\n"
                sys.stdout.flush()
                
                all_obj_corners = calibration_plugin.board.getChessboardCorners()
                obj_points = []
                img_points = []
                
                for corners, ids in zip(all_charuco_corners, all_charuco_ids):
                    ids_flat = ids.flatten()
                    obj_pts = all_obj_corners[ids_flat]
                    obj_points.append(obj_pts)
                    img_points.append(corners)
                
                # Limit to reasonable number of images using spatial diversity
                max_images = 50
                selected_image_paths = valid_image_paths  # Default: use all valid images
                
                if len(obj_points) > max_images:
                    yield f"data: {json_module.dumps({'type': 'progress', 'message': f'Selecting {max_images} spatially diverse images from {len(obj_points)} total...'})}\n\n"
                    sys.stdout.flush()
                    
                    # Calculate mean corner position for each image (spatial diversity metric)
                    image_positions = []
                    for i, corners in enumerate(img_points):
                        mean_pos = np.mean(corners, axis=0).flatten()  # Average x,y position
                        image_positions.append((i, mean_pos))
                    
                    # K-means clustering to select spatially diverse images
                    from sklearn.cluster import KMeans
                    positions = np.array([pos for _, pos in image_positions])
                    kmeans = KMeans(n_clusters=max_images, random_state=42, n_init=10)
                    kmeans.fit(positions)
                    
                    # Select the image closest to each cluster center
                    selected_indices = []
                    for center in kmeans.cluster_centers_:
                        distances = np.linalg.norm(positions - center, axis=1)
                        closest_idx = np.argmin(distances)
                        if closest_idx not in selected_indices:
                            selected_indices.append(closest_idx)
                    
                    # Sort indices to maintain temporal order
                    selected_indices = sorted(selected_indices[:max_images])
                    
                    obj_points = [obj_points[i] for i in selected_indices]
                    img_points = [img_points[i] for i in selected_indices]
                    selected_image_paths = [valid_image_paths[i] for i in selected_indices]
                    yield f"data: {json_module.dumps({'type': 'progress', 'message': f'Selected {len(obj_points)} spatially diverse images'})}\n\n"
                    sys.stdout.flush()
                
                # Run calibration with optimized flags
                yield f"data: {json_module.dumps({'type': 'progress', 'message': f'Computing camera parameters from {len(obj_points)} images...'})}\n\n"
                sys.stdout.flush()
                
                # Use flags to speed up calibration
                flags = cv2.CALIB_RATIONAL_MODEL
                
                ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                    obj_points,
                    img_points,
                    image_size,
                    None,
                    None,
                    flags=flags
                )
                
                yield f"data: {json_module.dumps({'type': 'progress', 'message': 'Calibration computation complete!'})}\n\n"
                sys.stdout.flush()
                
                if not ret:
                    yield f"data: {json_module.dumps({'type': 'error', 'error': 'Calibration failed - unable to compute camera parameters'})}\n\n"
                    return
                
                # Calculate reprojection error (using only the calibrated images)
                yield f"data: {json_module.dumps({'type': 'progress', 'message': 'Calculating reprojection errors...'})}\n\n"
                sys.stdout.flush()
                
                total_error = 0
                num_points = 0
                image_errors = []
                
                for i in range(len(obj_points)):
                    img_pts, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
                    error = cv2.norm(img_points[i], img_pts, cv2.NORM_L2)
                    total_error += error
                    num_corners = len(obj_points[i])
                    num_points += num_corners
                    
                    # Calculate per-image error
                    per_image_error = error / num_corners if num_corners > 0 else 0
                    
                    # Get the image filename from the selected paths
                    image_filename = os.path.basename(selected_image_paths[i])
                    image_errors.append({
                        'filename': image_filename,
                        'error': float(per_image_error),
                        'corners': num_corners
                    })
                
                mean_error = total_error / num_points if num_points > 0 else 0
                
                # Build result
                result = {
                    'success': True,
                    'camera_matrix': camera_matrix.tolist(),
                    'distortion_coeffs': dist_coeffs.flatten().tolist(),
                    'reprojection_error': float(mean_error),
                    'images_used': len(obj_points),
                    'images_total': len(image_paths),
                    'image_size': list(image_size),
                    'board_config': board_config,
                    'timestamp': datetime.now().isoformat(),
                    'image_errors': image_errors
                }
                
                # Save calibration results
                yield f"data: {json_module.dumps({'type': 'progress', 'message': 'Saving calibration results...'})}\n\n"
                sys.stdout.flush()
                
                calibration_file = os.path.join(session_path, 'calibration_results.json')
                with open(calibration_file, 'w') as f:
                    json_module.dump(result, f, indent=2)
                
                # Send final result
                yield f"data: {json_module.dumps({'type': 'complete', 'result': result})}\n\n"
                sys.stdout.flush()
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                yield f"data: {json_module.dumps({'type': 'error', 'error': str(e)})}\n\n"
                sys.stdout.flush()
        
        return Response(generate(), mimetype='text/event-stream')
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/calibration/undistort-image', methods=['POST'])
def undistort_image_endpoint():
    """Apply calibration to undistort an image."""
    try:
        import numpy as np
        import cv2
        import base64
        
        data = request.json
        session_name = data.get('session_name')
        camera_id = data.get('camera_id')
        image_filename = data.get('image_filename')
        
        if not session_name or not camera_id or not image_filename:
            return jsonify({
                'success': False,
                'error': 'Missing required parameters'
            }), 400
        
        # Build paths
        calibration_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'data', 'calibration'
        ))
        session_path = os.path.join(calibration_dir, session_name, str(camera_id))
        image_path = os.path.join(session_path, image_filename)
        calibration_file = os.path.join(session_path, 'calibration_results.json')
        
        if not os.path.exists(image_path):
            return jsonify({
                'success': False,
                'error': f'Image not found: {image_filename}'
            }), 404
        
        if not os.path.exists(calibration_file):
            return jsonify({
                'success': False,
                'error': 'Calibration results not found'
            }), 404
        
        # Load calibration data
        import json
        with open(calibration_file, 'r') as f:
            calibration = json.load(f)
        
        camera_matrix = np.array(calibration['camera_matrix'])
        dist_coeffs = np.array(calibration['distortion_coeffs'])
        
        # Load and undistort image
        img = cv2.imread(image_path)
        if img is None:
            return jsonify({
                'success': False,
                'error': 'Failed to load image'
            }), 500
        
        # Undistort using the calibration parameters
        # Use alpha=0 to crop out black areas, or use simple undistort without optimal matrix
        h, w = img.shape[:2]
        
        # Method 1: Simple undistort (keeps all pixels, may have black borders)
        undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, camera_matrix)
        
        # Encode both images to JPEG and then base64
        _, original_buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        _, undistorted_buffer = cv2.imencode('.jpg', undistorted, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        # Convert to base64 and remove any newlines
        original_base64 = base64.b64encode(original_buffer.tobytes()).decode('utf-8').replace('\n', '')
        undistorted_base64 = base64.b64encode(undistorted_buffer.tobytes()).decode('utf-8').replace('\n', '')
        
        return jsonify({
            'success': True,
            'original': f'data:image/jpeg;base64,{original_base64}',
            'undistorted': f'data:image/jpeg;base64,{undistorted_base64}',
            'image_size': [w, h]
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/processing/<camera_id>/enable', methods=['POST'])
def enable_image_processing(camera_id):
    """Enable image processing for a camera with configuration."""
    try:
        config = request.json or {}
        
        # Add minimal debug logging
        print(f"Enabling {config.get('type', 'unknown')} processing for camera {camera_id}")
        
        image_processing_enabled[camera_id] = True
        image_processing_config[camera_id] = config
        
        return jsonify({
            'success': True,
            'message': f'Image processing enabled for camera {camera_id}',
            'config': config
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/processing/<camera_id>/disable', methods=['POST'])
def disable_image_processing(camera_id):
    """Disable image processing for a camera."""
    try:
        image_processing_enabled[camera_id] = False
        if camera_id in image_processing_config:
            del image_processing_config[camera_id]
        
        return jsonify({
            'success': True,
            'message': f'Image processing disabled for camera {camera_id}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/processing/<camera_id>/update', methods=['POST'])
def update_image_processing(camera_id):
    """Update image processing configuration."""
    try:
        data = request.json
        config = data.get('config', {})
        
        if camera_id in image_processing_config:
            image_processing_config[camera_id].update(config)
        else:
            image_processing_config[camera_id] = config
        
        return jsonify({
            'success': True,
            'config': image_processing_config[camera_id]
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/<path:path>')

def serve_static(path):
    """Serve static files (CSS, JS)."""
    # Don't intercept API routes
    if path.startswith('api/'):
        return jsonify({'error': 'Not found'}), 404
    
    frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
    print(f'Serving {path} from: {frontend_dir}')  # Debug
    try:
        return send_from_directory(frontend_dir, path)
    except Exception as e:
        print(f'Error serving {path}: {e}')  # Debug
        return jsonify({'error': 'File not found'}), 404


# ============================================================================
# Panorama Calibration Endpoints
# ============================================================================

# Store panorama session data in memory
panorama_sessions = {}

@app.route('/api/calibration/panorama/check-detection', methods=['POST'])
def panorama_check_detection():
    """
    Check if ChArUco board is detected in both cameras without capturing.
    Uses hardware-synchronized camera pair for accurate frame sync.
    """
    try:
        data = request.json
        camera0_id = data.get('camera0_id')
        camera1_id = data.get('camera1_id')
        board_config = data.get('board_config')
        
        if not camera0_id or not camera1_id or not board_config:
            return jsonify({'success': False, 'error': 'Missing required parameters'}), 400
        
        # Get or create synchronized pair
        sync_pair = sync_pair_manager.get_pair(camera0_id, camera1_id)
        if sync_pair is None:
            sync_pair = sync_pair_manager.create_pair(camera0_id, camera1_id)
            sync_pair.start()
        
        # Get hardware-synchronized frames (max 16ms difference = 1 frame at 60fps)
        sync_result = sync_pair.get_synchronized_frames(max_time_diff=0.016)
        
        if sync_result is None:
            # Frames not ready or not synchronized yet - this is normal during startup
            return jsonify({
                'success': True,
                'detected_both': False,
                'detected_cam0': False,
                'detected_cam1': False,
                'corners_cam0': 0,
                'corners_cam1': 0,
                'synchronized': False,
                'waiting_for_frames': True
            })
        
        frame1, frame2, avg_timestamp = sync_result
        
        # Quick detection check
        from backend.panorama_utils import detect_charuco_for_panorama
        detection1 = detect_charuco_for_panorama(frame1, board_config)
        detection2 = detect_charuco_for_panorama(frame2, board_config)
        
        detected_both = detection1 is not None and detection2 is not None
        
        # Extract corner positions with IDs for position tracking
        corner_data_cam0 = None
        corner_data_cam1 = None
        
        if detection1:
            # Create list of [id, x, y] for each corner
            ids = detection1['ids'].flatten().tolist()
            corners = detection1['corners'].reshape(-1, 2)
            corner_data_cam0 = [[int(ids[i]), float(corners[i, 0]), float(corners[i, 1])] 
                               for i in range(len(ids))]
        
        if detection2:
            ids = detection2['ids'].flatten().tolist()
            corners = detection2['corners'].reshape(-1, 2)
            corner_data_cam1 = [[int(ids[i]), float(corners[i, 0]), float(corners[i, 1])] 
                               for i in range(len(ids))]
        
        return jsonify({
            'success': True,
            'detected_both': detected_both,
            'detected_cam0': detection1 is not None,
            'detected_cam1': detection2 is not None,
            'corners_cam0': detection1['num_corners'] if detection1 else 0,
            'corners_cam1': detection2['num_corners'] if detection2 else 0,
            'corner_data_cam0': corner_data_cam0,
            'corner_data_cam1': corner_data_cam1,
            'synchronized': True,
            'waiting_for_frames': False
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/calibration/panorama/capture', methods=['POST'])
def panorama_capture():
    """
    Capture single image pair and add to panorama session.
    Uses hardware-synchronized camera pair for accurate frame sync.
    Images are saved to disk in session directory.
    """
    try:
        import cv2
        from datetime import datetime
        
        data = request.json
        camera0_id = data.get('camera0_id')
        camera1_id = data.get('camera1_id')
        board_config = data.get('board_config')
        session_id = data.get('session_id')
        
        if not camera0_id or not camera1_id:
            return jsonify({'success': False, 'error': 'Both camera IDs required'}), 400
        
        if not board_config:
            return jsonify({'success': False, 'error': 'Board configuration required'}), 400
        
        # Get or create synchronized pair
        sync_pair = sync_pair_manager.get_pair(camera0_id, camera1_id)
        if sync_pair is None:
            sync_pair = sync_pair_manager.create_pair(camera0_id, camera1_id)
            sync_pair.start()
        
        # Get hardware-synchronized frames (max 16ms difference = 1 frame at 60fps)
        sync_result = sync_pair.get_synchronized_frames(max_time_diff=0.016)
        
        if sync_result is None:
            return jsonify({'success': False, 'error': 'Failed to get synchronized frames'}), 500
        
        frame1, frame2, avg_timestamp = sync_result
        
        # Initialize session if needed
        if not session_id:
            session_id = f"panorama_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if session_id not in panorama_sessions:
            # Create session directory
            session_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'calibration', session_id)
            os.makedirs(session_dir, exist_ok=True)
            
            # Save session metadata
            metadata_file = os.path.join(session_dir, 'session_info.json')
            metadata = {
                'session_id': session_id,
                'camera0_id': camera0_id,
                'camera1_id': camera1_id,
                'board_config': board_config,
                'created_at': datetime.now().isoformat()
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            panorama_sessions[session_id] = {
                'camera0_id': camera0_id,
                'camera1_id': camera1_id,
                'board_config': board_config,
                'image_pairs': [],
                'capture_count': 0,
                'session_dir': session_dir
            }
        
        session = panorama_sessions[session_id]
        session['capture_count'] += 1
        capture_num = session['capture_count']
        
        # Save images to disk
        cam0_dir = os.path.join(session['session_dir'], camera0_id)
        cam1_dir = os.path.join(session['session_dir'], camera1_id)
        os.makedirs(cam0_dir, exist_ok=True)
        os.makedirs(cam1_dir, exist_ok=True)
        
        cam0_filename = f"image_{capture_num:04d}.jpg"
        cam1_filename = f"image_{capture_num:04d}.jpg"
        cam0_path = os.path.join(cam0_dir, cam0_filename)
        cam1_path = os.path.join(cam1_dir, cam1_filename)
        
        cv2.imwrite(cam0_path, frame1)
        cv2.imwrite(cam1_path, frame2)
        
        # Also keep in memory for immediate processing
        session['image_pairs'].append((frame1.copy(), frame2.copy()))
        
        # Quick validation - detect board in both images
        from backend.panorama_utils import detect_charuco_for_panorama, match_charuco_corners
        detection1 = detect_charuco_for_panorama(frame1, board_config)
        detection2 = detect_charuco_for_panorama(frame2, board_config)
        
        detected_both = detection1 is not None and detection2 is not None
        matches = 0
        
        if detected_both:
            try:
                points1, points2 = match_charuco_corners(detection1, detection2)
                matches = len(points1)
            except:
                pass
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'capture_count': capture_num,
            'detected_both': detected_both,
            'matches': matches,
            'corners_cam0': detection1['num_corners'] if detection1 else 0,
            'corners_cam1': detection2['num_corners'] if detection2 else 0,
            'cam0_path': cam0_path,
            'cam1_path': cam1_path
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/calibration/panorama/save-settings', methods=['POST'])
def panorama_save_settings():
    """
    Save panorama session settings (calibration files, flags) without running calibration.
    """
    import cv2
    
    try:
        data = request.json
        session_id = data.get('session_id')
        use_calibration = data.get('use_calibration', True)
        calibration_method = data.get('calibration_method', 'extrinsics_only')
        cam0_calib_name = data.get('camera0_calibration')
        cam1_calib_name = data.get('camera1_calibration')
        flag_names = data.get('flags', [])
        
        if not session_id or session_id not in panorama_sessions:
            return jsonify({'success': False, 'error': 'Invalid or expired session'}), 400
        
        session = panorama_sessions[session_id]
        
        # Convert flag names to OpenCV flags value
        flags_value = 0
        flag_mapping = {
            'CALIB_FIX_INTRINSIC': cv2.CALIB_FIX_INTRINSIC,
            'CALIB_FIX_FOCAL_LENGTH': cv2.CALIB_FIX_FOCAL_LENGTH,
            'CALIB_FIX_PRINCIPAL_POINT': cv2.CALIB_FIX_PRINCIPAL_POINT,
            'CALIB_FIX_ASPECT_RATIO': cv2.CALIB_FIX_ASPECT_RATIO,
            'CALIB_SAME_FOCAL_LENGTH': cv2.CALIB_SAME_FOCAL_LENGTH,
            'CALIB_ZERO_TANGENT_DIST': cv2.CALIB_ZERO_TANGENT_DIST
        }
        
        for flag_name in flag_names:
            if flag_name in flag_mapping:
                flags_value |= flag_mapping[flag_name]
        
        # Load calibrations if needed (to copy them)
        cam0_calib = None
        cam1_calib = None
        
        if use_calibration and cam0_calib_name and cam1_calib_name:
            calibration_plugin = feature_manager.get_plugin('Camera Calibration')
            if calibration_plugin:
                try:
                    cam0_result = calibration_plugin.load_calibration(cam0_calib_name)
                    if cam0_result.get('success'):
                        cam0_calib = {
                            'camera_matrix': np.array(cam0_result['camera_matrix']),
                            'distortion_coeffs': np.array(cam0_result['distortion_coeffs'])
                        }
                    
                    cam1_result = calibration_plugin.load_calibration(cam1_calib_name)
                    if cam1_result.get('success'):
                        cam1_calib = {
                            'camera_matrix': np.array(cam1_result['camera_matrix']),
                            'distortion_coeffs': np.array(cam1_result['distortion_coeffs'])
                        }
                except Exception as e:
                    print(f"Warning: Could not load calibrations: {e}")
        
        # Update session metadata
        if 'session_dir' in session:
            session_info_path = Path(session['session_dir']) / 'session_info.json'
            if session_info_path.exists():
                try:
                    with open(session_info_path, 'r') as f:
                        session_info = json.load(f)
                    
                    # Copy calibration files if available
                    calib_copies = {}
                    if cam0_calib and cam0_calib_name:
                        cam0_calib_copy_path = Path(session['session_dir']) / f"camera0_calibration_{cam0_calib_name.replace('/', '_')}.json"
                        with open(cam0_calib_copy_path, 'w') as f:
                            json.dump({
                                'camera_matrix': cam0_calib['camera_matrix'].tolist(),
                                'distortion_coeffs': cam0_calib['distortion_coeffs'].tolist(),
                                'source': cam0_calib_name
                            }, f, indent=2)
                        calib_copies['camera0_calibration_file'] = str(cam0_calib_copy_path.name)
                    
                    if cam1_calib and cam1_calib_name:
                        cam1_calib_copy_path = Path(session['session_dir']) / f"camera1_calibration_{cam1_calib_name.replace('/', '_')}.json"
                        with open(cam1_calib_copy_path, 'w') as f:
                            json.dump({
                                'camera_matrix': cam1_calib['camera_matrix'].tolist(),
                                'distortion_coeffs': cam1_calib['distortion_coeffs'].tolist(),
                                'source': cam1_calib_name
                            }, f, indent=2)
                        calib_copies['camera1_calibration_file'] = str(cam1_calib_copy_path.name)
                    
                    # Update metadata
                    session_info['camera0_calibration'] = cam0_calib_name
                    session_info['camera1_calibration'] = cam1_calib_name
                    session_info['calibration_method'] = calibration_method
                    session_info['calibration_flags'] = flag_names
                    session_info['calibration_flags_value'] = flags_value if flags_value > 0 else None
                    session_info['use_calibration'] = use_calibration
                    session_info.update(calib_copies)
                    
                    with open(session_info_path, 'w') as f:
                        json.dump(session_info, f, indent=2)
                    
                    return jsonify({'success': True, 'message': 'Session settings saved'})
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    return jsonify({'success': False, 'error': f'Failed to save settings: {str(e)}'}), 500
        
        return jsonify({'success': False, 'error': 'Session directory not found'}), 400
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/calibration/panorama/compute', methods=['POST'])
def panorama_compute():
    """
    Compute panorama homography from all captured image pairs in session.
    """
    import cv2
    import numpy as np
    
    try:
        data = request.json
        session_id = data.get('session_id')
        use_calibration = data.get('use_calibration', True)
        calibration_method = data.get('calibration_method', 'extrinsics_only')  # 'extrinsics_only' or 'full_stereo'
        flag_names = data.get('flags', [])
        cam0_calib_name = data.get('camera0_calibration')
        cam1_calib_name = data.get('camera1_calibration')
        
        if not session_id or session_id not in panorama_sessions:
            return jsonify({'success': False, 'error': 'Invalid or expired session'}), 400
        
        session = panorama_sessions[session_id]
        
        if len(session['image_pairs']) == 0:
            return jsonify({'success': False, 'error': 'No images captured in session'}), 400
        
        # Convert flag names to OpenCV flags
        flags_value = 0
        flag_mapping = {
            'CALIB_FIX_INTRINSIC': cv2.CALIB_FIX_INTRINSIC,
            'CALIB_FIX_FOCAL_LENGTH': cv2.CALIB_FIX_FOCAL_LENGTH,
            'CALIB_FIX_PRINCIPAL_POINT': cv2.CALIB_FIX_PRINCIPAL_POINT,
            'CALIB_FIX_ASPECT_RATIO': cv2.CALIB_FIX_ASPECT_RATIO,
            'CALIB_SAME_FOCAL_LENGTH': cv2.CALIB_SAME_FOCAL_LENGTH,
            'CALIB_ZERO_TANGENT_DIST': cv2.CALIB_ZERO_TANGENT_DIST
        }
        
        for flag_name in flag_names:
            if flag_name in flag_mapping:
                flags_value |= flag_mapping[flag_name]
        
        # Check if fix intrinsic flag is set - if so, calibrations are required
        fix_intrinsic_set = 'CALIB_FIX_INTRINSIC' in flag_names
        
        # Load individual calibrations if requested
        cam0_calib = None
        cam1_calib = None
        
        if use_calibration:
            calibration_plugin = feature_manager.get_plugin('Camera Calibration')
            if not calibration_plugin:
                return jsonify({
                    'success': False,
                    'error': 'Calibration plugin not available. Please uncheck "Use camera calibrations".'
                }), 500
            
            # Calibration files must be explicitly selected (no auto-selection)
            if not cam0_calib_name or not cam1_calib_name:
                return jsonify({
                    'success': False,
                    'error': 'Please select calibration files for both cameras, or uncheck "Use camera calibrations".'
                }), 400
            
            try:
                # Try to load calibration for each camera
                cam0_result = calibration_plugin.load_calibration(cam0_calib_name)
                if cam0_result.get('success'):
                    cam0_calib = {
                        'camera_matrix': np.array(cam0_result['camera_matrix']),
                        'distortion_coeffs': np.array(cam0_result['distortion_coeffs'])
                    }
                else:
                    return jsonify({
                        'success': False,
                        'error': f'Failed to load calibration for camera {session["camera0_id"]}: {cam0_result.get("error", "Unknown error")}'
                    }), 400
                
                cam1_result = calibration_plugin.load_calibration(cam1_calib_name)
                if cam1_result.get('success'):
                    cam1_calib = {
                        'camera_matrix': np.array(cam1_result['camera_matrix']),
                        'distortion_coeffs': np.array(cam1_result['distortion_coeffs'])
                    }
                else:
                    return jsonify({
                        'success': False,
                        'error': f'Failed to load calibration for camera {session["camera1_id"]}: {cam1_result.get("error", "Unknown error")}'
                    }), 400
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f'Error loading calibrations: {str(e)}'
                }), 400
        
        # If fix intrinsic flag is set, both calibrations must be available
        if fix_intrinsic_set and (cam0_calib is None or cam1_calib is None):
            missing = []
            if cam0_calib is None:
                missing.append(f"camera {session['camera0_id']}")
            if cam1_calib is None:
                missing.append(f"camera {session['camera1_id']}")
            
            return jsonify({
                'success': False,
                'error': f"Fix intrinsic parameters flag requires calibration files. Missing calibrations for: {', '.join(missing)}. Please select calibration files or uncheck 'Fix intrinsic parameters'."
            }), 400
        
        # Run panorama calibration with all captures
        from backend.panorama_utils import calibrate_panorama_multiple
        result = calibrate_panorama_multiple(
            session['image_pairs'],
            session['board_config'],
            cam0_calib,
            cam1_calib,
            flags_value if flags_value > 0 else None,
            calibration_method=calibration_method
        )
        
        if result['success']:
            # Save calibration to global settings directory
            save_path = save_panorama_calibration(
                result,
                session['camera0_id'],
                session['camera1_id']
            )
            result['save_path'] = save_path
            
            # Also save to session directory for archival
            session_calib_path = None
            if 'session_dir' in session:
                session_calib_file = Path(session['session_dir']) / 'panorama_calibration_results.json'
                with open(session_calib_file, 'w') as f:
                    json.dump(result, f, indent=2)
                session_calib_path = str(session_calib_file)
            
            # Update session metadata with calibration file references and copy calibration files
            if 'session_dir' in session:
                session_info_path = Path(session['session_dir']) / 'session_info.json'
                if session_info_path.exists():
                    try:
                        with open(session_info_path, 'r') as f:
                            session_info = json.load(f)
                        
                        # Copy individual calibration files to session directory for archival
                        calib_copies = {}
                        if cam0_calib and cam0_calib_name:
                            cam0_calib_copy_path = Path(session['session_dir']) / f"camera0_calibration_{cam0_calib_name.replace('/', '_')}.json"
                            with open(cam0_calib_copy_path, 'w') as f:
                                json.dump({
                                    'camera_matrix': cam0_calib['camera_matrix'].tolist(),
                                    'distortion_coeffs': cam0_calib['distortion_coeffs'].tolist(),
                                    'source': cam0_calib_name
                                }, f, indent=2)
                            calib_copies['camera0_calibration_file'] = str(cam0_calib_copy_path.name)
                        
                        if cam1_calib and cam1_calib_name:
                            cam1_calib_copy_path = Path(session['session_dir']) / f"camera1_calibration_{cam1_calib_name.replace('/', '_')}.json"
                            with open(cam1_calib_copy_path, 'w') as f:
                                json.dump({
                                    'camera_matrix': cam1_calib['camera_matrix'].tolist(),
                                    'distortion_coeffs': cam1_calib['distortion_coeffs'].tolist(),
                                    'source': cam1_calib_name
                                }, f, indent=2)
                            calib_copies['camera1_calibration_file'] = str(cam1_calib_copy_path.name)
                        
                        # Add calibration metadata
                        session_info['camera0_calibration'] = cam0_calib_name if cam0_calib_name else None
                        session_info['camera1_calibration'] = cam1_calib_name if cam1_calib_name else None
                        session_info['calibration_flags'] = flag_names  # Store flag names for readability
                        session_info['calibration_flags_value'] = flags_value if flags_value > 0 else None
                        session_info['calibration_result_path'] = save_path  # Global path
                        session_info['session_calibration_path'] = session_calib_path  # Session-local path
                        session_info.update(calib_copies)
                        
                        with open(session_info_path, 'w') as f:
                            json.dump(session_info, f, indent=2)
                    except Exception as e:
                        print(f"Warning: Failed to update session metadata: {e}")
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/calibration/panorama/stitch', methods=['POST'])
def panorama_stitch():
    """Generate a stitched panorama from current camera frames using calibration."""
    import cv2
    import numpy as np
    import base64
    
    try:
        data = request.json
        camera0_id = data.get('camera0_id') or data.get('camera1_id')  # Support both naming conventions
        camera1_id = data.get('camera1_id') or data.get('camera2_id')
        session_id = data.get('session_id')
        blend_width = data.get('blend_width', 50)
        
        # Load calibration from session if provided, otherwise use latest
        homography = None
        
        if session_id and session_id in panorama_sessions:
            session = panorama_sessions[session_id]
            session_dir = Path(session['session_dir'])
            
            # Try to load panorama calibration results
            results_file = session_dir / 'panorama_calibration_results.json'
            if results_file.exists():
                with open(results_file, 'r') as f:
                    calib_data = json.load(f)
                    if 'homography' in calib_data:
                        homography = np.array(calib_data['homography'])
        
        # If no homography from session, try to load from global settings
        if homography is None:
            # Use absolute path from backend directory
            import os
            backend_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            panorama_dir = backend_dir / 'camera' / 'settings' / 'panorama'
            panorama_file = panorama_dir / f'{camera0_id}_{camera1_id}.json'
            
            # Debug logging
            print(f"[Panorama Stitch] Attempting to load calibration file:")
            print(f"  Backend dir: {backend_dir}")
            print(f"  Panorama dir: {panorama_dir}")
            print(f"  Panorama file: {panorama_file}")
            print(f"  File exists: {panorama_file.exists()}")
            
            if not panorama_file.exists():
                # Try reverse order
                panorama_file = panorama_dir / f'{camera1_id}_{camera0_id}.json'
            
            error_message = None
            if panorama_file.exists():
                try:
                    with open(panorama_file, 'r') as f:
                        calib_data = json.load(f)
                        # Handle nested calibration structure
                        if 'calibration' in calib_data and 'homography' in calib_data['calibration']:
                            homography = np.array(calib_data['calibration']['homography'])
                        elif 'homography' in calib_data:
                            homography = np.array(calib_data['homography'])
                        else:
                            error_message = f"Calibration file found but missing 'homography' key. Available keys: {list(calib_data.keys())}"
                except json.JSONDecodeError as e:
                    error_message = f"Invalid JSON in calibration file: {e}"
                except Exception as e:
                    error_message = f"Error loading calibration file: {e}"
            else:
                error_message = f"Calibration file not found at {panorama_file}"
        
        if homography is None:
            # Provide specific error message
            return jsonify({
                'success': False,
                'error': error_message or f'No panorama calibration found for cameras {camera0_id} and {camera1_id}',
                'expected_file': str(panorama_file),
                'directory': str(panorama_dir),
                'file_exists': panorama_file.exists()
            }), 400
        
        # Get frames from both cameras
        frame0 = camera_backend.get_full_frame(camera0_id)
        frame1 = camera_backend.get_full_frame(camera1_id)
        
        if frame0 is None or frame1 is None:
            return jsonify({
                'success': False,
                'error': 'Failed to get frames from cameras. Make sure both cameras are streaming.'
            }), 400
        
        # Load rotation matrix for symmetric rectification (if available)
        rotation_matrix = None
        K_left = None
        K_right = None
        calibration_size = None
        
        try:
            with open(panorama_file, 'r') as f:
                calib_data_full = json.load(f)
                if 'calibration' in calib_data_full and 'extrinsics' in calib_data_full['calibration']:
                    extrinsics = calib_data_full['calibration']['extrinsics']
                    if 'rotation_matrix' in extrinsics:
                        rotation_matrix = np.array(extrinsics['rotation_matrix'])
            
            # Load camera intrinsics
            cam0_calib_path = Path('data/calibration/session_20251126_152723/0/calibration_results.json')
            if cam0_calib_path.exists():
                with open(cam0_calib_path, 'r') as f:
                    cam0_data = json.load(f)
                    if 'camera_matrix' in cam0_data:
                        K_left = np.array(cam0_data['camera_matrix'])
                    if 'image_size' in cam0_data:
                        calibration_size = tuple(cam0_data['image_size'])
            
            cam1_calib_path = Path('data/calibration/session_20251202_155515/1/calibration_results.json')
            if cam1_calib_path.exists():
                with open(cam1_calib_path, 'r') as f:
                    cam1_data = json.load(f)
                    if 'camera_matrix' in cam1_data:
                        K_right = np.array(cam1_data['camera_matrix'])
                        
        except Exception as e:
            print(f"Could not load calibration data: {e}")
        
        # Stitch the panorama using cylindrical projection
        if rotation_matrix is not None and K_left is not None and K_right is not None:
            from backend.panorama_utils import stitch_panorama_cylindrical
            stitched = stitch_panorama_cylindrical(
                frame0, frame1,
                rotation_matrix=rotation_matrix,
                K_left=K_left,
                K_right=K_right,
                blend_width=blend_width
            )
        else:
            from backend.panorama_utils import stitch_panorama
            stitched = stitch_panorama(frame0, frame1, homography, blend_width)
        
        # Encode as JPEG for display
        _, buffer = cv2.imencode('.jpg', stitched, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': f'data:image/jpeg;base64,{img_base64}',
            'width': stitched.shape[1],
            'height': stitched.shape[0]
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/calibration/panorama/files')
def list_panorama_calibration_files():
    """List available panorama calibration files."""
    import os
    backend_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    panorama_dir = backend_dir / 'camera' / 'settings' / 'panorama'
    
    files = []
    if panorama_dir.exists():
        for file in panorama_dir.glob('*.json'):
            files.append({
                'filename': file.name,
                'path': str(file),
                'relative_path': f'backend/camera/settings/panorama/{file.name}'
            })
    
    return jsonify({
        'success': True,
        'files': files,
        'directory': str(panorama_dir),
        'directory_exists': panorama_dir.exists()
    })


@app.route('/api/calibration/panorama/stream')
def panorama_stream():
    """Stream live stitched panorama."""
    import cv2
    import numpy as np
    
    camera0_id = request.args.get('camera0_id') or request.args.get('camera1_id', '0')
    camera1_id = request.args.get('camera1_id') or request.args.get('camera2_id', '1')
    blend_width = int(request.args.get('blend_width', 50))
    scale = float(request.args.get('scale', 1.0))  # No scaling by default for preview frames
    
    def generate():
        # Load homography
        homography = None
        # Use absolute path from backend directory
        import os
        backend_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        panorama_dir = backend_dir / 'camera' / 'settings' / 'panorama'
        panorama_file = panorama_dir / f'{camera0_id}_{camera1_id}.json'
        
        # Debug logging
        print(f"[Panorama Stream] Attempting to load calibration file:")
        print(f"  Backend dir: {backend_dir}")
        print(f"  Panorama dir: {panorama_dir}")
        print(f"  Panorama file: {panorama_file}")
        print(f"  File exists: {panorama_file.exists()}")
        
        if not panorama_file.exists():
            # Try reverse order
            panorama_file = panorama_dir / f'{camera1_id}_{camera0_id}.json'
            print(f"  Trying reverse order: {panorama_file}")
            print(f"  File exists: {panorama_file.exists()}")
        
        error_message = None
        if panorama_file.exists():
            try:
                with open(panorama_file, 'r') as f:
                    calib_data = json.load(f)
                    # Handle nested calibration structure
                    if 'calibration' in calib_data and 'homography' in calib_data['calibration']:
                        homography = np.array(calib_data['calibration']['homography'])
                        print(f"  Loaded homography from calibration.homography")
                    elif 'homography' in calib_data:
                        homography = np.array(calib_data['homography'])
                        print(f"  Loaded homography from root level")
                    else:
                        error_message = f"File found but missing 'homography' key"
                        print(f"  ERROR: {error_message}")
                        print(f"  Available keys: {list(calib_data.keys())}")
            except json.JSONDecodeError as e:
                error_message = f"Invalid JSON in calibration file: {e}"
                print(f"  ERROR: {error_message}")
            except Exception as e:
                error_message = f"Error loading calibration: {e}"
                print(f"  ERROR: {error_message}")
        else:
            error_message = f"Calibration file not found"
            print(f"  ERROR: {error_message}")
        
        if homography is None:
            # Return error frame with specific error message
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            if error_message:
                cv2.putText(error_frame, 'Panorama Calibration Error', (20, 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Split long error messages
                words = error_message.split()
                line = ""
                y = 220
                for word in words:
                    test_line = line + word + " "
                    if len(test_line) > 50:
                        cv2.putText(error_frame, line.strip(), (20, y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        line = word + " "
                        y += 30
                    else:
                        line = test_line
                if line:
                    cv2.putText(error_frame, line.strip(), (20, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y += 30
            else:
                cv2.putText(error_frame, f'No calibration for {camera0_id}_{camera1_id}', (20, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.putText(error_frame, f'File: {panorama_file.name}', (20, y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
            cv2.putText(error_frame, f'Path: {panorama_dir}', (20, y + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
            _, buffer = cv2.imencode('.jpg', error_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            return
        
        # Load rotation matrix for symmetric rectification (if available)
        rotation_matrix = None
        K_left = None
        K_right = None
        calibration_size = None
        
        try:
            with open(panorama_file, 'r') as f:
                calib_data = json.load(f)
                if 'calibration' in calib_data and 'extrinsics' in calib_data['calibration']:
                    extrinsics = calib_data['calibration']['extrinsics']
                    if 'rotation_matrix' in extrinsics:
                        rotation_matrix = np.array(extrinsics['rotation_matrix'])
                        print(f"  Loaded rotation matrix for symmetric rectification")
            
            # Load camera intrinsics from calibration sessions
            # Camera 0 intrinsics
            cam0_calib_path = Path('data/calibration/session_20251126_152723/0/calibration_results.json')
            if cam0_calib_path.exists():
                with open(cam0_calib_path, 'r') as f:
                    cam0_data = json.load(f)
                    if 'camera_matrix' in cam0_data:
                        K_left = np.array(cam0_data['camera_matrix'])
                        print(f"  Loaded camera 0 intrinsics")
                    if 'image_size' in cam0_data:
                        calibration_size = tuple(cam0_data['image_size'])  # (width, height)
                        print(f"  Calibration resolution: {calibration_size}")
            
            # Camera 1 intrinsics
            cam1_calib_path = Path('data/calibration/session_20251202_155515/1/calibration_results.json')
            if cam1_calib_path.exists():
                with open(cam1_calib_path, 'r') as f:
                    cam1_data = json.load(f)
                    if 'camera_matrix' in cam1_data:
                        K_right = np.array(cam1_data['camera_matrix'])
                        print(f"  Loaded camera 1 intrinsics")
                        
        except Exception as e:
            print(f"  Could not load calibration data: {e}")
        
        # Import stitching function - using cylindrical projection
        if rotation_matrix is not None and K_left is not None and K_right is not None:
            from backend.panorama_utils import stitch_panorama_cylindrical
            stitch_func = lambda img0, img1: stitch_panorama_cylindrical(
                img0, img1,
                rotation_matrix=rotation_matrix,
                K_left=K_left,
                K_right=K_right,
                blend_width=blend_width
            )
            print(f"  Using cylindrical projection stitching with camera intrinsics")
        else:
            from backend.panorama_utils import stitch_panorama
            stitch_func = lambda img0, img1: stitch_panorama(img0, img1, homography, blend_width)
            print(f"  Using standard homography stitching (fallback)")
        
        # Check if cameras are streaming, start them if not
        cameras_started = []
        
        if not camera_backend.is_streaming(camera0_id):
            print(f"  Camera {camera0_id} not streaming, starting it...")
            success = camera_backend.start_stream(camera0_id)
            if success:
                cameras_started.append(camera0_id)
                print(f"  Camera {camera0_id} started successfully")
                time.sleep(0.5)  # Give camera time to initialize
            else:
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, f'Failed to start camera {camera0_id}', (20, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                _, buffer = cv2.imencode('.jpg', error_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                return
            
        if not camera_backend.is_streaming(camera1_id):
            print(f"  Camera {camera1_id} not streaming, starting it...")
            success = camera_backend.start_stream(camera1_id)
            if success:
                cameras_started.append(camera1_id)
                print(f"  Camera {camera1_id} started successfully")
                time.sleep(0.5)  # Give camera time to initialize
            else:
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, f'Failed to start camera {camera1_id}', (20, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                _, buffer = cv2.imencode('.jpg', error_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                return
        
        print(f"  Starting panorama stream for cameras {camera0_id} and {camera1_id}")
        frame_count = 0
        last_time = time.time()
        
        try:
            while True:
                try:
                    # Get preview frames (already JPEG-encoded, lower resolution, more efficient)
                    frame0_jpeg = camera_backend.get_preview_frame(camera0_id)
                    frame1_jpeg = camera_backend.get_preview_frame(camera1_id)
                    
                    if frame0_jpeg is None or frame1_jpeg is None:
                        if frame_count < 3:
                            print(f"  Waiting for frames... (cam0: {frame0_jpeg is not None}, cam1: {frame1_jpeg is not None})")
                        time.sleep(0.033)  # ~30 FPS
                        continue
                    
                    # Decode JPEG to numpy arrays
                    frame0 = cv2.imdecode(np.frombuffer(frame0_jpeg, np.uint8), cv2.IMREAD_COLOR)
                    frame1 = cv2.imdecode(np.frombuffer(frame1_jpeg, np.uint8), cv2.IMREAD_COLOR)
                    
                    # Debug logging for first few frames
                    if frame_count < 3:
                        print(f"  Frame {frame_count}: cam0={frame0.shape if frame0 is not None else None}, cam1={frame1.shape if frame1 is not None else None}")
                    
                    if frame0 is None or frame1 is None:
                        if frame_count < 3:
                            print(f"  Decode failed - skipping frame")
                        time.sleep(0.033)
                        continue
                    
                    # Stitch using appropriate method
                    stitched = stitch_func(frame0, frame1)
                    
                    if frame_count < 3:
                        print(f"  Stitched shape: {stitched.shape}, min: {stitched.min()}, max: {stitched.max()}")
                    
                    # Scale down for streaming if requested
                    if scale != 1.0:
                        new_width = int(stitched.shape[1] * scale)
                        new_height = int(stitched.shape[0] * scale)
                        stitched = cv2.resize(stitched, (new_width, new_height), interpolation=cv2.INTER_AREA)
                        if frame_count == 3:
                            print(f"  Scaled to: {stitched.shape} (scale={scale})")
                    
                    # Encode to JPEG with good quality
                    _, buffer = cv2.imencode('.jpg', stitched, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    
                    # FPS logging (every 30 frames)
                    frame_count += 1
                    if frame_count % 30 == 0:
                        current_time = time.time()
                        fps = 30.0 / (current_time - last_time)
                        print(f"  Panorama stream FPS: {fps:.1f}")
                        last_time = current_time
                    
                except Exception as e:
                    print(f"Panorama stream error: {e}")
                    import traceback
                    traceback.print_exc()
                    break
        finally:
            # Clean up: stop cameras we started
            for cam_id in cameras_started:
                print(f"  Stopping camera {cam_id} (auto-started)")
                camera_backend.stop_stream(cam_id)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/calibration/list', methods=['GET'])
def list_calibrations():
    """List all available calibration files."""
    try:
        calibration_plugin = feature_manager.get_plugin('Camera Calibration')
        if not calibration_plugin:
            return jsonify({'success': False, 'error': 'Calibration plugin not available'}), 500
        
        data_dir = calibration_plugin.data_dir
        calibrations = []
        
        # Find all .npz and .json calibration files
        for file_path in data_dir.glob('*.npz'):
            calibrations.append({
                'name': file_path.stem,
                'type': 'npz',
                'path': str(file_path.relative_to(data_dir.parent))
            })
        
        for file_path in data_dir.glob('*.json'):
            # Skip session_info.json files
            if file_path.stem == 'session_info':
                continue
            calibrations.append({
                'name': file_path.stem,
                'type': 'json',
                'path': str(file_path.relative_to(data_dir.parent))
            })
        
        # Also check session subdirectories
        for session_dir in data_dir.glob('session_*'):
            for cam_dir in session_dir.glob('*'):
                if cam_dir.is_dir():
                    calib_file = cam_dir / 'calibration_results.json'
                    if calib_file.exists():
                        # Extract camera ID from directory structure
                        cam_id = cam_dir.name
                        session_id = session_dir.name
                        calibrations.append({
                            'name': f"{session_id}/{cam_id}",
                            'type': 'json',
                            'path': str(calib_file.relative_to(data_dir.parent))
                        })
        
        # Also check panorama session directories for local calibration copies
        for session_dir in data_dir.glob('panorama_*'):
            if session_dir.is_dir():
                # Look for copied calibration files
                for calib_file in session_dir.glob('camera*_calibration_*.json'):
                    # Use path relative to data_dir (calibration directory), not data_dir.parent
                    relative_path = str(calib_file.relative_to(data_dir))
                    calibrations.append({
                        'name': relative_path,
                        'type': 'json',
                        'path': str(calib_file.relative_to(data_dir.parent))
                    })
        
        return jsonify({'success': True, 'calibrations': calibrations})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/calibration/panorama/sessions', methods=['GET'])
def panorama_list_sessions():
    """List all available panorama calibration sessions."""
    try:
        data_dir = Path(os.path.dirname(__file__)) / '..' / 'data' / 'calibration'
        sessions = []
        
        # Find all panorama session directories
        for session_dir in data_dir.glob('panorama_*'):
            if session_dir.is_dir():
                metadata_file = session_dir / 'session_info.json'
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Count images in session
                    image_count = len(list(session_dir.glob('capture_*_cam*.jpg')))
                    
                    sessions.append({
                        'session_id': session_dir.name,
                        'camera0_id': metadata.get('camera0_id'),
                        'camera1_id': metadata.get('camera1_id'),
                        'created_at': metadata.get('created_at'),
                        'image_count': image_count // 2  # Divide by 2 since we have pairs
                    })
        
        # Sort by creation date, newest first
        sessions.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return jsonify({'success': True, 'sessions': sessions})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/calibration/panorama/load', methods=['POST'])
def panorama_load_session():
    """Load an existing panorama calibration session."""
    try:
        import cv2
        
        data = request.json
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'success': False, 'error': 'Session ID required'}), 400
        
        # Check if session directory exists
        session_dir = Path(os.path.dirname(__file__)) / '..' / 'data' / 'calibration' / session_id
        if not session_dir.exists():
            return jsonify({'success': False, 'error': f'Session not found: {session_id}'}), 404
        
        # Load session metadata
        metadata_file = session_dir / 'session_info.json'
        if not metadata_file.exists():
            return jsonify({'success': False, 'error': 'Session metadata not found'}), 404
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Load all image pairs from session
        # Images are stored in camera-specific subdirectories: 0/, 1/, etc.
        camera0_dir = session_dir / str(metadata['camera0_id'])
        camera1_dir = session_dir / str(metadata['camera1_id'])
        
        if not camera0_dir.exists() or not camera1_dir.exists():
            return jsonify({'success': False, 'error': 'Camera directories not found in session'}), 404
        
        # Get sorted list of images from each camera directory
        camera0_files = sorted(camera0_dir.glob('image_*.jpg'))
        camera1_files = sorted(camera1_dir.glob('image_*.jpg'))
        
        # Match images by index (they should be captured in pairs)
        image_pairs = []
        capture_info = []
        
        for img0_path, img1_path in zip(camera0_files, camera1_files):
            img0 = cv2.imread(str(img0_path))
            img1 = cv2.imread(str(img1_path))
            
            if img0 is not None and img1 is not None:
                image_pairs.append((img0, img1))
                
                # Detect markers in both images to get match count
                from backend.panorama_utils import detect_charuco_for_panorama, match_charuco_corners
                
                detected_both = False
                matches = 0
                corners_cam0 = 0
                corners_cam1 = 0
                
                try:
                    detection0 = detect_charuco_for_panorama(img0, metadata['board_config'])
                    detection1 = detect_charuco_for_panorama(img1, metadata['board_config'])
                    
                    if detection0 is not None and detection1 is not None:
                        corners_cam0 = detection0['num_corners']
                        corners_cam1 = detection1['num_corners']
                        
                        # Try to match corners
                        try:
                            points0, points1 = match_charuco_corners(detection0, detection1)
                            matches = len(points0)
                            detected_both = True
                        except ValueError:
                            pass
                except Exception as e:
                    print(f"Detection error for {img0_path.name}: {e}")
                
                capture_info.append({
                    'detected_both': detected_both,
                    'matches': matches,
                    'corners_cam0': corners_cam0,
                    'corners_cam1': corners_cam1
                })
        
        # Create session in memory
        panorama_sessions[session_id] = {
            'camera0_id': metadata['camera0_id'],
            'camera1_id': metadata['camera1_id'],
            'board_config': metadata['board_config'],
            'image_pairs': image_pairs,
            'capture_count': len(image_pairs),
            'session_dir': str(session_dir)
        }
        
        # Check if local calibration files exist in session directory
        local_cam0_calib = None
        local_cam1_calib = None
        
        # Look for calibration files saved in the session
        if metadata.get('camera0_calibration_file'):
            cam0_file = session_dir / metadata['camera0_calibration_file']
            if cam0_file.exists():
                local_cam0_calib = f"{session_id}/{metadata['camera0_calibration_file']}"
        
        if metadata.get('camera1_calibration_file'):
            cam1_file = session_dir / metadata['camera1_calibration_file']
            if cam1_file.exists():
                local_cam1_calib = f"{session_id}/{metadata['camera1_calibration_file']}"
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'camera0_id': metadata['camera0_id'],
            'camera1_id': metadata['camera1_id'],
            'board_config': metadata['board_config'],
            'image_count': len(image_pairs),
            'captures': capture_info,
            'camera0_calibration': local_cam0_calib or metadata.get('camera0_calibration'),
            'camera1_calibration': local_cam1_calib or metadata.get('camera1_calibration'),
            'calibration_flags': metadata.get('calibration_flags', []),
            'use_calibration': metadata.get('use_calibration', True)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/calibration/panorama/reset', methods=['POST'])
def panorama_reset():
    """Reset panorama calibration session."""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if session_id and session_id in panorama_sessions:
            del panorama_sessions[session_id]
        
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/calibration/panorama/init-pair', methods=['POST'])
def panorama_init_pair():
    """Initialize synchronized camera pair for panorama preview."""
    try:
        data = request.json
        camera1_id = data.get('camera0_id')
        camera2_id = data.get('camera1_id')
        
        if not camera1_id or not camera2_id:
            return jsonify({'success': False, 'error': 'Both camera IDs required'}), 400
        
        # Stop individual camera streams if they're running
        # (sync pair needs exclusive access to the cameras)
        if camera_backend.is_streaming(camera1_id):
            print(f"[Panorama] Stopping individual stream for camera {camera1_id}")
            camera_backend.stop_stream(camera1_id)
        if camera_backend.is_streaming(camera2_id):
            print(f"[Panorama] Stopping individual stream for camera {camera2_id}")
            camera_backend.stop_stream(camera2_id)
        
        # Get or create synchronized pair
        sync_pair = sync_pair_manager.get_pair(camera1_id, camera2_id)
        if sync_pair is None:
            sync_pair = sync_pair_manager.create_pair(camera1_id, camera2_id)
            started = sync_pair.start()
            
            if not started:
                return jsonify({'success': False, 'error': 'Failed to start synchronized pair'}), 500
        
        return jsonify({
            'success': True,
            'synchronized': True,
            'message': 'Synchronized camera pair initialized'
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/calibration/panorama/load', methods=['GET'])
def panorama_load():
    """Load existing panorama calibration between two cameras."""
    try:
        camera1_id = request.args.get('camera1_id')
        camera2_id = request.args.get('camera2_id')
        
        if not camera1_id or not camera2_id:
            return jsonify({'success': False, 'error': 'Both camera IDs required'}), 400
        
        calibration = load_panorama_calibration(camera1_id, camera2_id)
        
        if calibration is None:
            return jsonify({
                'success': False,
                'error': f'No panorama calibration found for {camera1_id} and {camera2_id}'
            }), 404
        
        return jsonify({
            'success': True,
            'calibration': calibration
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/calibration/panorama/capture-full-res', methods=['POST'])
def panorama_capture_full_res():
    """
    Capture full-resolution test image from synchronized camera pair.
    Returns the full composite (top-bottom) or individual frames.
    """
    import time
    import cv2
    import base64
    
    try:
        data = request.json
        camera1_id = data.get('camera0_id')
        camera2_id = data.get('camera1_id')
        format_type = data.get('format', 'composite')  # 'composite', 'separate', or 'both'
        
        if not camera1_id or not camera2_id:
            return jsonify({'success': False, 'error': 'Both camera IDs required'}), 400
        
        # Get or create synchronized pair
        sync_pair = sync_pair_manager.get_pair(camera1_id, camera2_id)
        if sync_pair is None:
            sync_pair = sync_pair_manager.create_pair(camera1_id, camera2_id)
            if not sync_pair.start():
                return jsonify({'success': False, 'error': 'Failed to start camera pair'}), 500
            # Wait for cameras to warm up
            time.sleep(0.5)
        
        # Get full-resolution composite
        composite = sync_pair.get_full_composite()
        
        if composite is None:
            return jsonify({'success': False, 'error': 'Failed to capture frames'}), 500
        
        response_data = {
            'success': True,
            'timestamp': time.time(),
            'resolution': {
                'width': composite.shape[1],
                'height': composite.shape[0]
            }
        }
        
        if format_type in ['composite', 'both']:
            # Encode full composite as JPEG
            success, buffer = cv2.imencode('.jpg', composite, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if success:
                composite_base64 = base64.b64encode(buffer).decode('utf-8')
                response_data['composite'] = f'data:image/jpeg;base64,{composite_base64}'
        
        if format_type in ['separate', 'both']:
            # Split and encode individual frames
            height = composite.shape[0] // 2
            frame1 = composite[:height, :]
            frame2 = composite[height:, :]
            
            success1, buffer1 = cv2.imencode('.jpg', frame1, [cv2.IMWRITE_JPEG_QUALITY, 95])
            success2, buffer2 = cv2.imencode('.jpg', frame2, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            if success1 and success2:
                frame1_base64 = base64.b64encode(buffer1).decode('utf-8')
                frame2_base64 = base64.b64encode(buffer2).decode('utf-8')
                response_data['frame1'] = f'data:image/jpeg;base64,{frame1_base64}'
                response_data['frame2'] = f'data:image/jpeg;base64,{frame2_base64}'
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# Stereo Calibration - Compute extrinsics between camera pairs
# ============================================================================

@app.route('/api/calibration/stereo/compute', methods=['POST'])
def stereo_calibrate():
    """
    Compute stereo calibration (extrinsics) between two cameras using synchronized captures.
    This calculates rotation and translation between cameras for panorama stitching.
    """
    import cv2
    
    try:
        data = request.json
        session_id = data.get('session_id')
        calibration_flags = data.get('flags', 'CALIB_FIX_INTRINSIC')
        cam1_calib_session = data.get('cam1_calibration_session')
        cam2_calib_session = data.get('cam2_calibration_session')
        
        if not session_id:
            return jsonify({'success': False, 'error': 'Session ID required'}), 400
        
        # Parse session ID to get camera directories
        # Format: panorama_YYYYMMDD_HHMMSS or 0_1_timestamp
        session_path = Path('data/calibration') / session_id
        
        if not session_path.exists():
            return jsonify({'success': False, 'error': f'Session not found: {session_id}'}), 404
        
        # Load session info to get camera IDs
        session_info_path = session_path / 'session_info.json'
        if not session_info_path.exists():
            return jsonify({'success': False, 'error': 'Session info not found'}), 404
        
        with open(session_info_path, 'r') as f:
            session_info = json.load(f)
        
        camera1_id = str(session_info.get('camera1_id') or session_info.get('camera_id', '0'))
        camera2_id = str(session_info.get('camera2_id', '1'))
        
        # Paths to synchronized images
        camera1_path = session_path / camera1_id
        camera2_path = session_path / camera2_id
        
        if not camera1_path.exists() or not camera2_path.exists():
            return jsonify({
                'success': False,
                'error': f'Camera image directories not found in session'
            }), 404
        
        # Load individual camera calibrations
        calibration_plugin = feature_manager.get_plugin('Camera Calibration')
        if not calibration_plugin:
            return jsonify({
                'success': False,
                'error': 'Calibration plugin not available'
            }), 500
        
        # Load camera 1 calibration from specified session or auto-detect
        if cam1_calib_session:
            cam1_calib_path = Path('data/calibration') / cam1_calib_session / camera1_id / 'calibration_results.json'
            if not cam1_calib_path.exists():
                return jsonify({
                    'success': False,
                    'error': f'Camera 0 calibration not found in session: {cam1_calib_session}',
                    'camera': camera1_id
                }), 400
            with open(cam1_calib_path, 'r') as f:
                cam1_result = json.load(f)
                cam1_result['success'] = True
        else:
            cam1_result = calibration_plugin.load_calibration(f"camera_{camera1_id}")
        
        if not cam1_result.get('success'):
            return jsonify({
                'success': False,
                'error': f'Camera {camera1_id} calibration not found. Please select a calibration session for Camera 0.',
                'camera': camera1_id
            }), 400
        
        # Load camera 2 calibration from specified session or auto-detect
        if cam2_calib_session:
            cam2_calib_path = Path('data/calibration') / cam2_calib_session / camera2_id / 'calibration_results.json'
            if not cam2_calib_path.exists():
                return jsonify({
                    'success': False,
                    'error': f'Camera 1 calibration not found in session: {cam2_calib_session}',
                    'camera': camera2_id
                }), 400
            with open(cam2_calib_path, 'r') as f:
                cam2_result = json.load(f)
                cam2_result['success'] = True
        else:
            cam2_result = calibration_plugin.load_calibration(f"camera_{camera2_id}")
        
        if not cam2_result.get('success'):
            return jsonify({
                'success': False,
                'error': f'Camera {camera2_id} calibration not found. Please select a calibration session for Camera 1.',
                'camera': camera2_id
            }), 400
        
        # Prepare calibration data
        camera1_calib = {
            'camera_matrix': cam1_result['camera_matrix'],
            'distortion_coeffs': cam1_result['distortion_coeffs']
        }
        camera2_calib = {
            'camera_matrix': cam2_result['camera_matrix'],
            'distortion_coeffs': cam2_result['distortion_coeffs']
        }
        
        # Get board config
        board_config = session_info.get('board_config', {
            'width': 8,
            'height': 5,
            'square_length': 0.050,
            'marker_length': 0.037,
            'dictionary': 'DICT_6X6_100'
        })
        
        # Parse calibration flags
        cv2_flags = 0
        if calibration_flags:
            if isinstance(calibration_flags, str):
                flags_list = [f.strip() for f in calibration_flags.split('|')]
            else:
                flags_list = calibration_flags if isinstance(calibration_flags, list) else [calibration_flags]
            
            for flag_name in flags_list:
                if hasattr(cv2, flag_name):
                    cv2_flags |= getattr(cv2, flag_name)
        
        # Get calibration plugin
        calibration_plugin = feature_manager.get_plugin('Camera Calibration')
        if not calibration_plugin:
            return jsonify({'success': False, 'error': 'Calibration plugin not available'}), 500
        
        # Progress tracking
        progress_messages = []
        def progress_callback(msg):
            progress_messages.append(msg)
            print(f"[Stereo Calibration] {msg}")
        
        # Run stereo calibration using plugin
        result = calibration_plugin.stereo_calibrate_from_sessions(
            str(camera1_path),
            str(camera2_path),
            board_config,
            camera1_calib,
            camera2_calib,
            flags=cv2_flags if cv2_flags > 0 else None,
            progress_callback=progress_callback
        )
        
        if result['success']:
            # Save stereo calibration
            from backend.panorama_utils import save_stereo_calibration
            save_path = save_stereo_calibration(result, camera1_id, camera2_id)
            result['save_path'] = save_path
            result['progress'] = progress_messages
            
            # Also compute optimal homography from stereo calibration
            H = calibration_plugin.compute_optimal_homography_from_stereo(result, tuple(result['image_size']))
            result['optimal_homography'] = H.tolist()
        else:
            result['progress'] = progress_messages
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/calibration/stereo/load', methods=['GET'])
def stereo_load():
    """Load existing stereo calibration between two cameras."""
    try:
        camera1_id = request.args.get('camera1_id')
        camera2_id = request.args.get('camera2_id')
        
        if not camera1_id or not camera2_id:
            return jsonify({'success': False, 'error': 'Both camera IDs required'}), 400
        
        from backend.panorama_utils import load_stereo_calibration
        calibration = load_stereo_calibration(camera1_id, camera2_id)
        
        if calibration is None:
            return jsonify({
                'success': False,
                'error': f'No stereo calibration found for cameras {camera1_id} and {camera2_id}'
            }), 404
        
        return jsonify({
            'success': True,
            'calibration': calibration
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/calibration/stereo/info', methods=['GET'])
def stereo_info():
    """Get information about stereo calibration capabilities."""
    return jsonify({
        'available': True,
        'message': 'Stereo calibration computes extrinsic relationship (rotation and translation) between camera pairs',
        'features': [
            'Rotation and translation matrix computation',
            'Essential and fundamental matrix calculation',
            'Epipolar geometry for geometric accuracy',
            'Rectification transforms for aligned images',
            'Optimal homography computation for stitching',
            'Support for divergent cameras with minimal overlap'
        ],
        'calibration_flags': {
            'CALIB_FIX_INTRINSIC': 'Use fixed camera matrices from individual calibrations (recommended)',
            'CALIB_USE_INTRINSIC_GUESS': 'Optimize camera matrices using individual calibrations as starting point',
            'CALIB_FIX_PRINCIPAL_POINT': 'Fix principal point during optimization',
            'CALIB_FIX_FOCAL_LENGTH': 'Fix focal lengths during optimization',
            'CALIB_FIX_ASPECT_RATIO': 'Fix aspect ratio (fx/fy) during optimization',
            'CALIB_SAME_FOCAL_LENGTH': 'Enforce same focal length for both cameras',
            'CALIB_RATIONAL_MODEL': 'Use rational distortion model (k4,k5,k6)',
            'CALIB_THIN_PRISM_MODEL': 'Use thin prism distortion model',
            'CALIB_FIX_S1_S2_S3_S4': 'Fix thin prism distortion coefficients'
        },
        'recommended_flags_divergent': [
            'CALIB_FIX_INTRINSIC',
            'For divergent cameras, use fixed intrinsics from individual calibrations'
        ]
    })


# ============================================================================
# Application Startup
# ============================================================================

if __name__ == '__main__':
    # Initialize backend systems
    initialize()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)

