"""
Main Flask application integrating camera backends, features, and workflows.
"""

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from pathlib import Path
import os
import threading
import time

from backend.camera.factory import get_camera_backend, detect_platform
from backend.camera.groups import CameraGroupManager
from backend.features.manager import FeatureManager
from backend.workflows.manager import WorkflowManager
from backend.calibration_utils import calibrate_camera_from_session, save_calibration_results


# Initialize Flask app
# Use absolute path for frontend folder
frontend_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
app = Flask(__name__, static_folder=frontend_folder, static_url_path='')
CORS(app)

# Initialize managers
camera_backend = None
group_manager = None
feature_manager = None
workflow_manager = None

# Calibration overlay state
calibration_overlay_enabled = {}
calibration_board_config = {}


def initialize():
    """Initialize all backend systems."""
    global camera_backend, group_manager, feature_manager, workflow_manager
    
    print("Detecting platform...")
    platform = detect_platform()
    print(f"Platform detected: {platform}")
    
    print("Initializing camera backend...")
    camera_backend = get_camera_backend()
    
    print("Initializing group manager...")
    group_manager = CameraGroupManager()
    
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


@app.route('/api/cameras/<camera_id>/stream')
def camera_stream(camera_id):
    """MJPEG stream endpoint for a camera."""
    import cv2
    import numpy as np
    
    def generate():
        """Generate MJPEG stream."""
        while camera_backend.is_streaming(camera_id):
            frame_jpeg = camera_backend.get_preview_frame(camera_id)
            
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
            
            if frame_jpeg:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_jpeg + b'\r\n')
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
        
        # Get all calibration session directories
        calibration_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'calibration')
        session_dirs = glob.glob(os.path.join(calibration_dir, 'session_*'))
        
        sessions = []
        for session_dir in sorted(session_dirs, reverse=True):  # Most recent first
            try:
                session_name = os.path.basename(session_dir)
                
                # Parse timestamp from session name (format: session_YYYYMMDD_HHMMSS)
                timestamp_str = session_name.replace('session_', '')
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
                    
                    if image_count > 0:  # Only include sessions with images
                        sessions.append({
                            'path': session_name,
                            'camera_id': camera_id,
                            'date': date_str,
                            'image_count': image_count
                        })
            except Exception as e:
                print(f"Error processing session {session_dir}: {e}")
                continue
        
        return jsonify({
            'success': True,
            'sessions': sessions
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
        
        images = []
        for i, img_path in enumerate(image_files, 1):
            image_data = {
                'id': i,
                'path': img_path,
                'filename': os.path.basename(img_path)
            }
            
            # Analyze image for pattern detection if calibration plugin is available
            if calibration_plugin and board_config:
                try:
                    # Load image
                    img = cv2.imread(img_path)
                    if img is not None:
                        # Detect pattern
                        detection = calibration_plugin.detect_charuco_pattern(img)
                        
                        if detection.get('detected'):
                            # Add detection data
                            image_data['detection'] = {
                                'detected': True,
                                'markers_detected': detection.get('markers_detected', 0),
                                'corners_detected': detection.get('corners_detected', 0),
                                'quality': detection.get('quality', 'Unknown'),
                                'image_width': img.shape[1],
                                'image_height': img.shape[0]
                            }
                            
                            # Add marker corners
                            if detection.get('marker_corners') is not None:
                                marker_corners = detection['marker_corners']
                                image_data['detection']['marker_corners'] = [
                                    corners.reshape(-1, 2).tolist() for corners in marker_corners
                                ]
                            
                            # Add marker IDs
                            if detection.get('marker_ids') is not None:
                                image_data['detection']['marker_ids'] = detection['marker_ids'].flatten().tolist()
                            
                            # Classify pose
                            if detection.get('marker_corners') is not None:
                                poses = classify_board_pose(
                                    detection['marker_corners'],
                                    img.shape[1],
                                    img.shape[0]
                                )
                                image_data['detection']['poses'] = poses
                except Exception as e:
                    print(f"Error analyzing image {img_path}: {e}")
                    # Continue without detection data for this image
            
            images.append(image_data)
        
        return jsonify({
            'success': True,
            'session': {
                'path': session_path,
                'camera_id': camera_id,
                'model': model,
                'board_config': board_config,
                'target_images': target_images,
                'images': images
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
    """Run camera calibration on captured images."""
    try:
        from datetime import datetime
        
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
        
        # Run calibration
        results = calibrate_camera_from_session(session_path, board_config)
        
        if not results.get('success', False):
            return jsonify(results), 400
        
        # Add timestamp
        results['timestamp'] = datetime.now().isoformat()
        
        # Save calibration results to session directory
        calibration_file = os.path.join(session_path, 'calibration_results.json')
        import json
        with open(calibration_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return jsonify(results)
        
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
        
        original_base64 = base64.b64encode(original_buffer).decode('utf-8')
        undistorted_base64 = base64.b64encode(undistorted_buffer).decode('utf-8')
        
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
# Application Startup
# ============================================================================

if __name__ == '__main__':
    # Initialize backend systems
    initialize()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
