# Camera Streaming Platform

An extensible, multi-hardware camera streaming platform with support for Jetson Orin Nano, Raspberry Pi, Luxonis DepthAI, and standard webcams.

## Features

- **Multi-Platform Support**: Abstracted camera backend supporting multiple hardware platforms
- **Multi-Camera Streaming**: Stream up to 3 cameras simultaneously
- **Camera Groups**: Create stereo pairs or multi-camera groups
- **Real-Time Controls**: Adjust exposure, white balance, gain, and other camera settings during streaming
- **Dual-Stream Architecture**: High-resolution for processing, compressed preview for web
- **Extensible Plugin System**: Add calibration, object detection, and other features
- **Workflow System**: Step-by-step guided processes for setup and calibration
- **Profile Management**: Save and apply camera settings profiles

## Architecture

```
backend/
├── camera/
│   ├── base.py              # Abstract camera backend interface
│   ├── jetson.py            # Jetson Orin Nano implementation (GStreamer)
│   ├── raspberry_pi.py      # Raspberry Pi stub (future)
│   ├── luxonis.py           # Luxonis DepthAI stub (future)
│   ├── webcam.py            # Generic webcam stub (future)
│   ├── factory.py           # Platform detection and backend factory
│   ├── groups.py            # Camera grouping system
│   └── profiles/
│       └── jetson/          # Platform-specific profiles
├── features/
│   ├── base.py              # Abstract feature plugin interface
│   ├── manager.py           # Plugin discovery and management
│   └── plugins/             # Feature plugin implementations
├── workflows/
│   ├── base.py              # Abstract workflow interface
│   ├── manager.py           # Workflow management
│   └── templates/           # Global workflow templates
└── app.py                   # Flask application with REST API

frontend/
├── index.html               # Web interface
├── style.css                # Responsive CSS
└── app.js                   # Frontend JavaScript

data/
├── camera_groups/           # Saved camera group configurations
└── workflows/
    └── state/               # Workflow instance states
```

## Installation

### Prerequisites

**For Jetson Orin Nano:**
- NVIDIA JetPack installed
- GStreamer with nvarguscamerasrc plugin
- Python 3.8+

```bash
# Install system dependencies (REQUIRED - do this first!)
sudo apt-get update
sudo apt-get install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    python3-gi \
    python3-gst-1.0 \
    python3-pip

# Create virtual environment with system packages access
python3 -m venv venv --system-site-packages
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

**Note:** The `--system-site-packages` flag is important to access the pre-installed GStreamer Python bindings.

## Usage

### Starting the Server

```bash
# From the project root
cd backend
python app.py
```

The server will start on `http://0.0.0.0:5000`

### Web Interface

Open your browser to:
- Local: `http://localhost:5000`
- Network: `http://<your-ip>:5000`

### Basic Workflow

1. **Select Cameras**: Check cameras in the sidebar to start streaming
2. **Adjust Controls**: Click on a video feed to load camera controls
3. **Create Groups**: Group cameras for stereo or multi-camera setups
4. **Save Profiles**: Save current camera settings as a named profile
5. **Run Features**: Select feature plugins for processing
6. **Execute Workflows**: Follow guided workflows for calibration

## API Documentation

### Camera Endpoints

- `GET /api/cameras` - List all cameras
- `GET /api/cameras/<id>/capabilities` - Get camera capabilities
- `POST /api/cameras/<id>/start` - Start camera stream
- `POST /api/cameras/<id>/stop` - Stop camera stream
- `GET /api/cameras/<id>/stream` - MJPEG stream endpoint
- `GET /api/cameras/<id>/controls` - Get available controls
- `PUT /api/cameras/<id>/control/<name>` - Set control value
- `GET /api/cameras/<id>/profiles` - List profiles
- `POST /api/cameras/<id>/profiles/<name>` - Save profile
- `POST /api/cameras/<id>/profile` - Apply profile

### Camera Group Endpoints

- `GET /api/camera-groups` - List all groups
- `POST /api/camera-groups` - Create group
- `GET /api/camera-groups/<id>` - Get group details
- `PUT /api/camera-groups/<id>` - Update group
- `DELETE /api/camera-groups/<id>` - Delete group
- `POST /api/camera-groups/<id>/start` - Start all cameras in group
- `GET /api/camera-groups/<id>/calibration` - Get calibration data
- `POST /api/camera-groups/<id>/calibration` - Save calibration data

### Feature Plugin Endpoints

- `GET /api/features` - List all features
- `POST /api/features/<name>/process` - Process with feature
- `POST /api/features/<name>/process-group` - Process camera group

### Workflow Endpoints

- `GET /api/workflows` - List all workflows
- `POST /api/workflows/<name>/start` - Start workflow instance
- `GET /api/workflows/<id>/state` - Get workflow state
- `POST /api/workflows/<id>/step/<step_id>` - Execute workflow step
- `POST /api/workflows/<id>/reset` - Reset workflow
- `DELETE /api/workflows/<id>` - Delete workflow instance

## Extending the Platform

### Adding a Feature Plugin

Create a new file in `backend/features/plugins/`:

```python
from backend.features.base import FeaturePlugin

class MyFeaturePlugin(FeaturePlugin):
    def get_metadata(self):
        return {
            'name': 'my_feature',
            'version': '1.0.0',
            'description': 'My custom feature',
            'requires_cameras': 1,
            'requires_high_res': True,
            'supports_groups': False
        }
    
    def get_ui_schema(self):
        return {
            'controls': [
                {
                    'name': 'threshold',
                    'type': 'range',
                    'label': 'Threshold',
                    'min': 0,
                    'max': 255,
                    'default': 128
                }
            ]
        }
    
    def process_frames(self, camera_backend, camera_ids, params=None):
        # Get frame
        frame = camera_backend.get_full_frame(camera_ids[0])
        
        # Process frame
        # ...
        
        return {
            'success': True,
            'result': 'Processing complete'
        }
```

### Adding a Workflow

Create a new file in `backend/workflows/templates/`:

```python
from backend.workflows.base import Workflow, WorkflowStep

class MyWorkflow(Workflow):
    def get_metadata(self):
        return {
            'name': 'my_workflow',
            'description': 'My custom workflow',
            'version': '1.0.0',
            'category': 'setup',
            'requires_cameras': 1
        }
    
    def get_steps(self):
        return [
            WorkflowStep(
                'step1',
                'Step 1',
                'Description of step 1',
                ui_schema={'controls': [...]}
            ),
            # More steps...
        ]
    
    def validate_step(self, step_id, data):
        # Validate step data
        return {'valid': True}
    
    def execute_step(self, step_id, data, camera_backend, group_manager):
        # Execute step logic
        return {'success': True}
```

### Adding Platform Support

1. Create backend file (e.g., `backend/camera/my_platform.py`)
2. Implement `CameraBackend` abstract class
3. Add detection logic to `backend/camera/factory.py`
4. Test with your hardware

## Camera Controls (Jetson)

The Jetson backend provides normalized controls:

- **exposure**: Exposure time (nanoseconds)
- **auto_exposure**: Auto exposure enable/disable
- **gain**: Sensor gain (1.0 - 16.0)
- **auto_gain**: Auto gain enable/disable
- **white_balance**: WB mode (auto/daylight/cloudy/etc.)
- **saturation**: Color saturation (0.0 - 2.0)
- **edge_enhancement**: Sharpness (-1.0 - 1.0)
- **noise_reduction**: TNR mode (off/fast/high-quality)

## Profiles

Pre-configured profiles are available in `backend/camera/profiles/jetson/`:

- **indoor**: Optimized for indoor lighting
- **outdoor**: Optimized for daylight
- **low_light**: Optimized for low-light conditions

Profiles are JSON files that can be created, edited, and shared.

## Troubleshooting

### Cameras not detected

```bash
# Test GStreamer pipeline manually
gst-launch-1.0 nvarguscamerasrc sensor-id=0 num-buffers=10 ! \
  'video/x-raw(memory:NVMM)' ! fakesink

# Check for CSI cameras
ls -l /dev/video*
```

### Permission issues

```bash
# Add user to video group
sudo usermod -a -G video $USER
# Log out and back in
```

### Module import errors

```bash
# Ensure you're in the virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## Future Enhancements

- WebRTC streaming for lower latency
- Hardware H.264 encoding (nvenc)
- Raspberry Pi libcamera support
- Luxonis DepthAI SDK integration
- Generic webcam V4L2 support
- Stereo calibration workflow
- Object detection feature
- Recording and playback
- Multi-user support
- Authentication and authorization

## License

MIT License (or your preferred license)

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues, questions, or contributions, please open an issue on the repository.
