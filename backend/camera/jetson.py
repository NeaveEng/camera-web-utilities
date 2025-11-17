"""
Jetson Orin Nano camera backend using GStreamer and nvarguscamerasrc.

This backend provides access to CSI cameras on NVIDIA Jetson platforms using
the hardware-accelerated nvarguscamerasrc GStreamer element.
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import threading
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import time

from .base import CameraBackend

# Initialize GStreamer
Gst.init(None)


class JetsonCameraBackend(CameraBackend):
    """Camera backend for NVIDIA Jetson platforms with CSI cameras."""
    
    # Default sensor configurations
    SENSOR_CONFIGS = {
        'IMX219': {
            'fps_map': {
                (3840, 2160): 30,
                (1920, 1080): 60,
                (1280, 720): 60,
                (640, 480): 60,
            },
            'default_fps': 30
        },
        'IMX477': {
            'fps_map': {
                (4032, 3040): 10,
                (1920, 1080): 50,
                (1280, 720): 120,
            },
            'default_fps': 30
        }
    }
    
    # Active configuration (loaded on init or set via API)
    RESOLUTION_FPS_MAP = SENSOR_CONFIGS['IMX219']['fps_map'].copy()
    DEFAULT_FPS = SENSOR_CONFIGS['IMX219']['default_fps']
    current_sensor = 'IMX219'

    def __init__(self):
        """Initialize the Jetson camera backend."""
        self.platform_name = "jetson"
        self.active_cameras: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        
        # Profile directory
        self.profile_dir = Path(__file__).parent / "profiles" / self.platform_name
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        
        # Sensor config directory
        self.sensor_config_dir = Path(__file__).parent / "sensor_configs"
        self.sensor_config_dir.mkdir(parents=True, exist_ok=True)
        
        # Load saved sensor configuration
        self._load_sensor_config()
        
        # Settings directory for camera settings
        self.settings_dir = Path(__file__).parent / "settings" / self.platform_name
        self.settings_dir.mkdir(parents=True, exist_ok=True)

    def _load_sensor_config(self):
        """Load saved sensor configuration from disk."""
        config_file = self.sensor_config_dir / "active_sensor.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    data = json.load(f)
                    sensor_name = data.get('sensor', 'IMX219')
                    
                    # Load sensor-specific config if it exists
                    if sensor_name in self.SENSOR_CONFIGS:
                        self.current_sensor = sensor_name
                        self.RESOLUTION_FPS_MAP = self.SENSOR_CONFIGS[sensor_name]['fps_map'].copy()
                        self.DEFAULT_FPS = self.SENSOR_CONFIGS[sensor_name]['default_fps']
                    
                    # Override with custom values if provided
                    if 'custom_fps_map' in data:
                        custom_map = {}
                        for res_str, fps in data['custom_fps_map'].items():
                            w, h = map(int, res_str.split('x'))
                            custom_map[(w, h)] = fps
                        self.RESOLUTION_FPS_MAP = custom_map
                    
                    if 'custom_default_fps' in data:
                        self.DEFAULT_FPS = data['custom_default_fps']
            except Exception as e:
                print(f"Error loading sensor config: {e}")

    def _save_sensor_config(self, sensor_name: str, custom_fps_map=None, custom_default_fps=None):
        """Save sensor configuration to disk."""
        sensor_file = self.sensor_config_dir / f"{sensor_name}.json"
        active_file = self.sensor_config_dir / "active_sensor.json"
        
        try:
            data = {}
            
            if custom_fps_map:
                # Handle both dict and tuple-keyed dict formats
                if isinstance(next(iter(custom_fps_map.keys()), None), tuple):
                    # Convert tuples to strings for JSON
                    data['custom_fps_map'] = {
                        f"{w}x{h}": fps 
                        for (w, h), fps in custom_fps_map.items()
                    }
                else:
                    # Already in string format
                    data['custom_fps_map'] = custom_fps_map
            
            if custom_default_fps:
                data['custom_default_fps'] = custom_default_fps
            
            # Save sensor-specific config
            with open(sensor_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Save as active sensor
            active_data = {'sensor': sensor_name}
            active_data.update(data)
            with open(active_file, 'w') as f:
                json.dump(active_data, f, indent=2)
            
            # Update class variables
            self.current_sensor = sensor_name
            if custom_fps_map:
                if isinstance(next(iter(custom_fps_map.keys()), None), tuple):
                    self.RESOLUTION_FPS_MAP = custom_fps_map.copy()
                else:
                    # Convert string keys to tuples
                    new_map = {}
                    for res_str, fps in custom_fps_map.items():
                        w, h = map(int, res_str.split('x'))
                        new_map[(w, h)] = fps
                    self.RESOLUTION_FPS_MAP = new_map
            if custom_default_fps:
                self.DEFAULT_FPS = custom_default_fps
                
        except Exception as e:
            logger.error(f"Error saving sensor config: {e}")
            raise
    
    def _get_available_sensors(self):
        """Get list of all available sensors (predefined + custom)"""
        sensors = list(self.SENSOR_CONFIGS.keys())  # Start with predefined
        
        # Add any custom sensors from saved configs
        try:
            if self.sensor_config_dir.exists():
                for filepath in self.sensor_config_dir.glob('*.json'):
                    if filepath.name != 'active_sensor.json':
                        sensor_name = filepath.stem
                        if sensor_name not in sensors:
                            sensors.append(sensor_name)
        except Exception as e:
            logger.error(f"Error listing sensor configs: {e}")
        
        return sorted(sensors)

    def enumerate_cameras(self) -> List[Dict[str, Any]]:
        """Enumerate available CSI cameras by testing sensor IDs."""
        cameras = []
        
        # Jetson Orin Nano Dev Kit has 2 CSI camera connectors (sensor-id 0-1)
        # Skip availability testing to prevent Argus CaptureSession conflicts
        # Cameras that don't exist will fail gracefully on start attempt
        for sensor_id in range(2):
            camera_id = str(sensor_id)
            
            # Check if camera is already streaming
            is_streaming = camera_id in self.active_cameras
            
            cameras.append({
                'id': camera_id,
                'name': f'CSI Camera {sensor_id}',
                'type': 'CSI',
                'available': True,  # Assume available, will fail on start if not
                'streaming': is_streaming
            })
        
        return cameras

    def _get_fps_for_resolution(self, width: int, height: int) -> int:
        """Get the appropriate framerate for a given resolution based on sensor capabilities."""
        resolution = (width, height)
        return self.RESOLUTION_FPS_MAP.get(resolution, self.DEFAULT_FPS)

    def _test_camera(self, sensor_id: int) -> bool:
        """Test if a camera with given sensor_id is available."""
        import sys
        import io
        
        try:
            # Suppress stderr temporarily to hide nvarguscamerasrc errors
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            
            # Create a minimal pipeline to test camera availability  
            pipeline_str = (
                f'nvarguscamerasrc sensor-id={sensor_id} num-buffers=1 ! '
                f'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1 ! '
                f'fakesink'
            )
            
            pipeline = Gst.parse_launch(pipeline_str)
            
            # Try to set state to PLAYING
            ret = pipeline.set_state(Gst.State.PLAYING)
            
            # Restore stderr
            captured_stderr = sys.stderr.getvalue()
            sys.stderr = old_stderr
            
            # If state change failed immediately, camera doesn't exist
            if ret == Gst.StateChangeReturn.FAILURE:
                pipeline.set_state(Gst.State.NULL)
                return False
            
            # Check for error messages in stderr
            if "Invalid camera device" in captured_stderr:
                pipeline.set_state(Gst.State.NULL)
                return False
            
            # Wait for state change to complete or error
            bus = pipeline.get_bus()
            msg = bus.timed_pop_filtered(
                2 * Gst.SECOND,
                Gst.MessageType.ERROR | Gst.MessageType.ASYNC_DONE | Gst.MessageType.EOS
            )
            
            # Determine result before cleanup
            result = False
            if msg and msg.type == Gst.MessageType.ERROR:
                result = False
            elif msg and msg.type in (Gst.MessageType.ASYNC_DONE, Gst.MessageType.EOS):
                result = True
            else:
                # Timeout - likely means camera doesn't exist
                result = False
            
            # Properly clean up pipeline - set to NULL and wait for completion
            pipeline.set_state(Gst.State.NULL)
            pipeline.get_state(Gst.CLOCK_TIME_NONE)  # Wait for NULL state
            
            # Delay to allow Argus daemon to fully release camera resources
            import time
            time.sleep(0.3)
            
            return result
            
        except Exception as e:
            # Restore stderr if exception occurred
            try:
                sys.stderr = old_stderr
            except:
                pass
            print(f"Error testing camera {sensor_id}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return False

    def get_capabilities(self, camera_id: str) -> Dict[str, Any]:
        """Get capabilities for a Jetson CSI camera."""
        # Jetson CSI cameras typically support these resolutions
        return {
            'resolutions': [
                (640, 480),
                (1280, 720),
                (1920, 1080),
                (3264, 2464),  # Max resolution for IMX219
            ],
            'formats': ['NV12', 'I420', 'BGR'],
            'frame_rates': [15, 21, 30, 60],
            'controls': self.get_controls(camera_id)
        }

    def start_stream(self, camera_id: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """Start streaming from a Jetson CSI camera."""
        with self.lock:
            if camera_id in self.active_cameras:
                return True  # Already streaming
            
            # Parse configuration
            config = config or {}
            width = config.get('width', 1920)
            height = config.get('height', 1080)
            fps = config.get('fps', 30)
            # Extract configuration
            preview_width = config.get('preview_width', 640)
            preview_height = config.get('preview_height', 480)
            preview_quality = config.get('preview_quality', 70)
            
            # Load saved settings to get preferred resolutions before creating pipeline
            # Only apply saved settings if not already specified in config
            saved_settings = self.load_camera_settings(camera_id)
            if saved_settings:
                # Apply capture resolution if saved and not already in config
                if 'capture_resolution' in saved_settings and 'width' not in config:
                    try:
                        cap_w, cap_h = map(int, saved_settings['capture_resolution'].split('x'))
                        width = cap_w
                        height = cap_h
                    except ValueError:
                        pass  # Use defaults if invalid format
                
                # Apply stream resolution if saved
                if 'stream_resolution' in saved_settings:
                    try:
                        prev_w, prev_h = map(int, saved_settings['stream_resolution'].split('x'))
                        preview_width = prev_w
                        preview_height = prev_h
                    except ValueError:
                        pass  # Use defaults if invalid format
            
            # Set fps from config or lookup table (ignore saved fps)
            if 'fps' not in config:
                fps = self._get_fps_for_resolution(width, height)
            
            try:
                # Create pipeline with tee for dual-stream (full-res + preview)
                pipeline_str = (
                    f'nvarguscamerasrc sensor-id={camera_id} name=src ! '
                    f'video/x-raw(memory:NVMM),width={width},height={height},framerate={fps}/1 ! '
                    f'tee name=t '
                    
                    # Full-resolution branch for features (convert NVMM to system memory)
                    f't. ! queue ! nvvidconv name=nvvidconv_full ! video/x-raw ! videoconvert ! video/x-raw,format=BGR ! '
                    f'appsink name=full_sink emit-signals=true max-buffers=1 drop=true '
                    
                    # Preview branch for web streaming
                    f't. ! queue ! nvvidconv name=nvvidconv_preview ! '
                    f'video/x-raw,width={preview_width},height={preview_height} ! '
                    f'jpegenc quality={preview_quality} ! '
                    f'appsink name=preview_sink emit-signals=true max-buffers=1 drop=true'
                )
                
                pipeline = Gst.parse_launch(pipeline_str)
                
                # Get appsinks
                full_sink = pipeline.get_by_name('full_sink')
                preview_sink = pipeline.get_by_name('preview_sink')
                
                # Get source element for control
                src_element = pipeline.get_by_name('src')
                
                # Get nvvidconv elements for rotation control
                nvvidconv_full = pipeline.get_by_name('nvvidconv_full')
                nvvidconv_preview = pipeline.get_by_name('nvvidconv_preview')
                
                # Setup camera state
                camera_state = {
                    'pipeline': pipeline,
                    'full_sink': full_sink,
                    'preview_sink': preview_sink,
                    'src_element': src_element,
                    'nvvidconv_full': nvvidconv_full,
                    'nvvidconv_preview': nvvidconv_preview,
                    'full_frame': None,
                    'preview_frame': None,
                    'full_frame_lock': threading.Lock(),
                    'preview_frame_lock': threading.Lock(),
                    'config': {
                        'width': width,
                        'height': height,
                        'fps': fps,
                        'preview_width': preview_width,
                        'preview_height': preview_height,
                    },
                    'controls': self._get_default_controls()
                }
                
                # Load saved settings if they exist (already loaded above for resolution)
                if saved_settings:
                    camera_state['controls'].update(saved_settings)
                    # Update resolutions to match what was actually applied to pipeline
                    camera_state['controls']['stream_resolution'] = f'{preview_width}x{preview_height}'
                    camera_state['controls']['capture_resolution'] = f'{width}x{height}'
                
                # Connect callbacks for frame capture
                full_sink.connect('new-sample', self._on_full_frame, camera_id)
                preview_sink.connect('new-sample', self._on_preview_frame, camera_id)
                
                # Store state before starting
                self.active_cameras[camera_id] = camera_state
                
                # Start pipeline
                ret = pipeline.set_state(Gst.State.PLAYING)
                if ret == Gst.StateChangeReturn.FAILURE:
                    del self.active_cameras[camera_id]
                    return False
                
                # Apply saved settings to hardware after pipeline starts
                if saved_settings:
                    for control_name, value in saved_settings.items():
                        # Skip resolutions - they were already applied during pipeline creation
                        # Applying them here would trigger a restart loop
                        if control_name in ('stream_resolution', 'capture_resolution'):
                            continue
                        try:
                            # Apply setting without triggering auto-save (avoid recursion)
                            self._apply_control(camera_id, control_name, value)
                        except Exception as e:
                            print(f"Error applying saved setting {control_name}={value}: {e}")
                
                return True
                
            except Exception as e:
                import traceback
                print(f"Error starting camera {camera_id}: {e}", flush=True)
                print(traceback.format_exc(), flush=True)
                if camera_id in self.active_cameras:
                    del self.active_cameras[camera_id]
                return False

    def _on_full_frame(self, sink, camera_id):
        """Callback for new full-resolution frame."""
        sample = sink.emit('pull-sample')
        if sample:
            buffer = sample.get_buffer()
            caps = sample.get_caps()
            
            # Get frame dimensions
            struct = caps.get_structure(0)
            width = struct.get_value('width')
            height = struct.get_value('height')
            
            # Map buffer and convert to numpy array
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if success:
                # Convert to numpy array (BGR format)
                frame = np.ndarray(
                    shape=(height, width, 3),
                    dtype=np.uint8,
                    buffer=map_info.data
                )
                
                # Store frame
                with self.active_cameras[camera_id]['full_frame_lock']:
                    self.active_cameras[camera_id]['full_frame'] = frame.copy()
                
                buffer.unmap(map_info)
        
        return Gst.FlowReturn.OK

    def _on_preview_frame(self, sink, camera_id):
        """Callback for new preview frame (JPEG)."""
        sample = sink.emit('pull-sample')
        if sample:
            buffer = sample.get_buffer()
            
            # Map buffer and get JPEG bytes
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if success:
                jpeg_data = bytes(map_info.data)
                
                # Store JPEG frame
                with self.active_cameras[camera_id]['preview_frame_lock']:
                    self.active_cameras[camera_id]['preview_frame'] = jpeg_data
                
                buffer.unmap(map_info)
        
        return Gst.FlowReturn.OK

    def stop_stream(self, camera_id: str) -> bool:
        """Stop streaming from a camera."""
        with self.lock:
            if camera_id not in self.active_cameras:
                return False
            
            try:
                pipeline = self.active_cameras[camera_id]['pipeline']
                
                # Set pipeline to NULL and wait for it to complete
                pipeline.set_state(Gst.State.NULL)
                pipeline.get_state(Gst.CLOCK_TIME_NONE)  # Block until NULL state reached
                
                # Delay to allow Argus daemon to fully release camera resources
                import time
                time.sleep(0.3)
                
                del self.active_cameras[camera_id]
                return True
            except Exception as e:
                print(f"Error stopping camera {camera_id}: {e}", flush=True)
                return False

    def get_full_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """Get latest full-resolution frame."""
        if camera_id not in self.active_cameras:
            return None
        
        with self.active_cameras[camera_id]['full_frame_lock']:
            return self.active_cameras[camera_id]['full_frame']

    def get_preview_frame(self, camera_id: str) -> Optional[bytes]:
        """Get latest preview frame as JPEG."""
        if camera_id not in self.active_cameras:
            return None
        
        with self.active_cameras[camera_id]['preview_frame_lock']:
            return self.active_cameras[camera_id]['preview_frame']

    def get_controls(self, camera_id: str) -> Dict[str, Dict[str, Any]]:
        """Get available controls for nvarguscamerasrc."""
        return {
            'exposure': {
                'type': 'range',
                'min': -2,
                'max': 2,
                'step': 0.1,
                'default': 0,
                'current': self.get_control(camera_id, 'exposure') or 0,
                'unit': 'EV',
                'label': 'Brightness',
                'description': 'Exposure compensation (-2 to +2 EV)',
                'platform_name': 'exposurecompensation'
            },
            'auto_exposure': {
                'type': 'bool',
                'default': True,
                'current': self.get_control(camera_id, 'auto_exposure') or True,
                'label': 'Auto Exposure Lock',
                'description': 'Lock auto-exposure at current value'
            },
            'gain': {
                'type': 'range',
                'min': 0,
                'max': 100,
                'step': 1,
                'default': 0,
                'current': self.get_control(camera_id, 'gain') or 0,
                'unit': '%',
                'label': 'Digital Gain',
                'description': 'ISP digital gain amplification',
                'platform_name': 'ispdigitalgainrange'
            },
            'auto_gain': {
                'type': 'bool',
                'default': True,
                'current': self.get_control(camera_id, 'auto_gain') or True,
                'disables': ['gain']
            },
            'white_balance': {
                'type': 'menu',
                'options': ['off', 'auto', 'incandescent', 'fluorescent', 'warm-fluorescent', 
                           'daylight', 'cloudy-daylight', 'twilight', 'shade', 'manual'],
                'default': 'auto',
                'current': self.get_control(camera_id, 'white_balance') or 'auto',
                'platform_name': 'wbmode'
            },
            'saturation': {
                'type': 'range',
                'min': 0.0,
                'max': 2.0,
                'step': 0.1,
                'default': 1.0,
                'current': self.get_control(camera_id, 'saturation') or 1.0,
                'platform_name': 'saturation'
            },
            'edge_enhancement': {
                'type': 'range',
                'min': -1.0,
                'max': 1.0,
                'step': 0.1,
                'default': 0.0,
                'current': self.get_control(camera_id, 'edge_enhancement') or 0.0,
                'platform_name': 'ee-strength'
            },
            'rotation': {
                'type': 'menu',
                'options': ['none', 'rotate-90-cw', 'rotate-180', 'rotate-90-ccw', 
                           'flip-horizontal', 'flip-vertical'],
                'default': 'none',
                'current': self.get_control(camera_id, 'rotation') or 'none',
                'label': 'Rotation/Flip',
                'description': 'Rotate or flip the camera image',
                'platform_name': 'flip-method'
            },
            'stream_resolution': {
                'type': 'menu',
                'options': ['320x240', '640x480', '800x600', '1280x720', '1920x1080'],
                'default': '640x480',
                'current': self.get_control(camera_id, 'stream_resolution') or '640x480',
                'label': 'Stream Resolution',
                'description': 'Resolution for browser preview stream'
            },
            'capture_resolution': {
                'type': 'menu',
                'options': ['1920x1080', '3840x2160'],
                'default': '1920x1080',
                'current': self.get_control(camera_id, 'capture_resolution') or '1920x1080',
                'label': 'Capture Resolution',
                'description': 'Full resolution for frame capture and processing'
            },
            'camera_name': {
                'type': 'text',
                'default': '',
                'current': self.get_control(camera_id, 'camera_name') or '',
                'label': 'Camera Name',
                'description': 'Custom name for this camera'
            }
        }

    def _get_default_controls(self) -> Dict[str, Any]:
        """Get default control values."""
        return {
            'exposure': 0,  # 0 EV compensation
            'auto_exposure': True,
            'gain': 0,  # 0% digital gain
            'auto_gain': True,
            'white_balance': 'auto',
            'saturation': 1.0,
            'edge_enhancement': 0.0,
            'rotation': 'none',
            'stream_resolution': '640x480',
            'capture_resolution': '1920x1080',
            'camera_name': ''
        }

    def set_control(self, camera_id: str, control_name: str, value: Any) -> bool:
        """Set a camera control on active pipeline."""
        # Save settings automatically
        result = self._apply_control(camera_id, control_name, value, save=True)
        return result

    def _apply_control(self, camera_id: str, control_name: str, value: Any, save: bool = False) -> bool:
        """Internal method to apply a control value to hardware."""
        if camera_id not in self.active_cameras:
            return False
        
        try:
            src = self.active_cameras[camera_id]['src_element']
            controls = self.get_controls(camera_id)
            
            if control_name not in controls:
                return False
            
            platform_name = controls[control_name].get('platform_name', control_name)
            
            # Set property on nvarguscamerasrc or nvvidconv
            if control_name == 'exposure':
                # For nvarguscamerasrc, use exposurecompensation for dynamic brightness control
                # Maps our -100 to +100 range to -2.0 to 2.0 EV compensation
                value_int = int(value)
                ev_compensation = (value_int / 100.0) * 2.0  # -2 to +2 EV
                src.set_property('exposurecompensation', float(ev_compensation))
                
            elif control_name == 'gain':
                # For gain, use ispdigitalgainrange which works dynamically
                # Maps our 0-100% range to ISP digital gain 1.0-8.0
                value_int = int(value)
                normalized = value_int / 100.0  # 0 to 1
                isp_gain = 1.0 + (normalized * 7.0)  # 1 to 8
                src.set_property('ispdigitalgainrange', f'{isp_gain:.2f} {isp_gain:.2f}')
                
            elif control_name == 'auto_exposure':
                # Control auto-exposure lock
                if value:
                    # Enable auto exposure: unlock
                    src.set_property('aelock', False)
                else:
                    # Disable auto exposure: lock at current value
                    src.set_property('aelock', True)
                    src.set_property('aeantibanding', 0)  # Off mode
            elif control_name == 'auto_gain':
                # Auto gain is controlled by setting gainrange to a range vs fixed value
                # When enabling auto, set to full range
                if value:
                    src.set_property('gainrange', '1.0 22.25')
                    
            elif control_name == 'rotation':
                # Map friendly names to GStreamer flip-method enum values
                rotation_map = {
                    'none': 0,
                    'rotate-90-ccw': 1,  # counterclockwise
                    'rotate-180': 2,
                    'rotate-90-cw': 3,   # clockwise
                    'flip-horizontal': 4,
                    'flip-vertical': 6
                }
                flip_method = rotation_map.get(value, 0)
                
                # Apply to both nvvidconv elements for this specific camera
                print(f"Setting rotation for camera {camera_id} to {value} (flip-method={flip_method})")
                self.active_cameras[camera_id]['nvvidconv_full'].set_property('flip-method', flip_method)
                self.active_cameras[camera_id]['nvvidconv_preview'].set_property('flip-method', flip_method)
                
            elif control_name == 'stream_resolution':
                # Parse resolution string (e.g., "640x480")
                try:
                    width, height = map(int, value.split('x'))
                    
                    # Store current config for restart
                    config = self.active_cameras[camera_id]['config'].copy()
                    config['preview_width'] = width
                    config['preview_height'] = height
                    
                    # Store all current control values
                    current_controls = self.active_cameras[camera_id]['controls'].copy()
                    
                    # Restart the stream with new resolution
                    print(f"Restarting camera {camera_id} with new stream resolution {width}x{height}")
                    self.stop_stream(camera_id)
                    
                    # Delay to ensure Argus daemon fully releases camera
                    import time
                    time.sleep(0.5)
                    
                    # Restart with new config
                    if self.start_stream(camera_id, config):
                        # Reapply all controls except stream_resolution
                        for ctrl_name, ctrl_value in current_controls.items():
                            if ctrl_name != 'stream_resolution':
                                self._apply_control(camera_id, ctrl_name, ctrl_value, save=False)
                    else:
                        print(f"Failed to restart camera {camera_id}")
                        return False
                        
                except ValueError:
                    print(f"Invalid resolution format: {value}")
                    return False
            
            elif control_name == 'capture_resolution':
                # Parse resolution string (e.g., "1920x1080")
                try:
                    width, height = map(int, value.split('x'))
                    
                    # Get appropriate fps from lookup table
                    fps = self._get_fps_for_resolution(width, height)
                    
                    # Store current config for restart
                    config = self.active_cameras[camera_id]['config'].copy()
                    config['width'] = width
                    config['height'] = height
                    config['fps'] = fps
                    
                    # Store all current control values
                    current_controls = self.active_cameras[camera_id]['controls'].copy()
                    
                    # Restart the stream with new capture resolution
                    print(f"Restarting camera {camera_id} with new capture resolution {width}x{height}")
                    self.stop_stream(camera_id)
                    
                    # Delay to ensure Argus daemon fully releases camera
                    import time
                    time.sleep(0.5)
                    
                    # Restart with new config
                    if self.start_stream(camera_id, config):
                        # Reapply all controls except capture_resolution
                        for ctrl_name, ctrl_value in current_controls.items():
                            if ctrl_name not in ('capture_resolution', 'stream_resolution'):
                                self._apply_control(camera_id, ctrl_name, ctrl_value, save=False)
                    else:
                        print(f"Failed to restart camera {camera_id}")
                        return False
                        
                except ValueError:
                    print(f"Invalid resolution format: {value}")
                    return False
            
            elif control_name == 'camera_name':
                # Camera name is metadata only, no hardware property to set
                pass
                
            else:
                # Direct property mapping
                src.set_property(platform_name, value)
            
            # Update stored value
            self.active_cameras[camera_id]['controls'][control_name] = value
            
            # Auto-save settings if requested
            if save:
                self.save_camera_settings(camera_id)
            
            return True
            
        except Exception as e:
            print(f"Error setting control {control_name}={value}: {e}")
            return False

    def get_control(self, camera_id: str, control_name: str) -> Optional[Any]:
        """Get current control value."""
        if camera_id not in self.active_cameras:
            return None
        
        return self.active_cameras[camera_id]['controls'].get(control_name)

    def get_profiles(self) -> List[str]:
        """Get list of available profiles."""
        profiles = []
        if self.profile_dir.exists():
            for profile_file in self.profile_dir.glob('*.json'):
                profiles.append(profile_file.stem)
        return sorted(profiles)

    def get_profile(self, profile_name: str) -> Optional[Dict[str, Any]]:
        """Load a profile from disk."""
        profile_path = self.profile_dir / f'{profile_name}.json'
        if not profile_path.exists():
            return None
        
        try:
            with open(profile_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading profile {profile_name}: {e}")
            return None

    def save_profile(self, profile_name: str, settings: Dict[str, Any]) -> bool:
        """Save a profile to disk."""
        profile_path = self.profile_dir / f'{profile_name}.json'
        try:
            with open(profile_path, 'w') as f:
                json.dump(settings, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving profile {profile_name}: {e}")
            return False

    def delete_profile(self, profile_name: str) -> bool:
        """Delete a profile."""
        profile_path = self.profile_dir / f'{profile_name}.json'
        try:
            if profile_path.exists():
                profile_path.unlink()
            return True
        except Exception as e:
            print(f"Error deleting profile {profile_name}: {e}")
            return False

    def apply_profile(self, camera_id: str, profile_name: str) -> bool:
        """Apply a profile to a camera."""
        profile = self.get_profile(profile_name)
        if not profile:
            return False
        
        # Apply all settings from profile
        success = True
        for control_name, value in profile.items():
            if not self.set_control(camera_id, control_name, value):
                success = False
        
        return success

    def _get_settings_path(self, camera_id: str) -> Path:
        """Get settings file path for a camera."""
        return self.settings_dir / f'camera_{camera_id}.json'

    def load_camera_settings(self, camera_id: str) -> Dict[str, Any]:
        """Load saved settings for a camera."""
        settings_path = self._get_settings_path(camera_id)
        if not settings_path.exists():
            return {}
        
        try:
            with open(settings_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading settings for camera {camera_id}: {e}")
            return {}

    def save_camera_settings(self, camera_id: str) -> bool:
        """Save current settings for a camera."""
        if camera_id not in self.active_cameras:
            return False
        
        try:
            settings = self.active_cameras[camera_id]['controls'].copy()
            settings_path = self._get_settings_path(camera_id)
            
            with open(settings_path, 'w') as f:
                json.dump(settings, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving settings for camera {camera_id}: {e}")
            return False

    def is_streaming(self, camera_id: str) -> bool:
        """Check if camera is streaming."""
        return camera_id in self.active_cameras

    def get_platform_name(self) -> str:
        """Get platform name."""
        return self.platform_name
