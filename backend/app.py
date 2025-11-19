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
    def generate():
        """Generate MJPEG stream."""
        while camera_backend.is_streaming(camera_id):
            frame = camera_backend.get_preview_frame(camera_id)
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
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
        calibration_plugin = feature_manager.get_plugin('calibration')
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
            return jsonify({
                'success': False,
                'message': 'Failed to get frame from camera. Is the camera streaming?'
            }), 500
        
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
        if detection.get('charuco_corners') is not None:
            response_data['detection']['charuco_corners'] = detection['charuco_corners'].reshape(-1, 2).tolist()
        
        return jsonify(response_data)
        
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
