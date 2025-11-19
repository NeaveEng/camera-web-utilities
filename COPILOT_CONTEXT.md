# GitHub Copilot Context - Camera Streaming Platform

**Last Updated:** 19 November 2025  
**Primary Platform:** NVIDIA Jetson Orin Nano (platform-agnostic design)  
**Cameras:** 2x IMX477 CSI cameras (supports multiple camera models)  
**Status:** Fully functional dual-camera streaming system with web interface

---

## Project Overview

A platform-agnostic Flask-based camera streaming platform designed for multiple hardware platforms and camera types. Currently developed and tested on NVIDIA Jetson with CSI cameras. Features real-time MJPEG streaming, hardware-accelerated processing via GStreamer, and a responsive web UI for camera control.

### Key Architecture
- **Backend:** Python/Flask with GStreamer (platform-specific backends)
- **Frontend:** Vanilla HTML/CSS/JavaScript
- **Streaming:** Dual-stream architecture (full-res + preview JPEG)
- **Platform Design:** Abstracted camera backends via factory pattern
- **Current Hardware:** NVIDIA Jetson Orin Nano with 2x IMX477 CSI cameras
- **Supported Cameras:** IMX477 (12MP), IMX219 (8MP), and expandable to other models

---

## Current Implementation Status

### ✅ Completed Features

#### Camera Backend (`backend/camera/`)
- **Platform abstraction:** Factory pattern with base class and platform-specific implementations
  - `base.py`: Abstract camera interface
  - `factory.py`: Platform detection and backend selection
  - `jetson.py`: NVIDIA Jetson implementation (nvarguscamerasrc)
  - `webcam.py`, `raspberry_pi.py`, `luxonis.py`: Additional platform support
- **Dual-stream GStreamer pipeline:**
  - Full-res branch: 1920x1080 BGR for processing
  - Preview branch: Configurable JPEG stream (default 640x480)
- **Dynamic controls:** Brightness (EV compensation), gain (ISP digital), rotation, white balance, saturation, edge enhancement, noise reduction
- **Settings persistence:** Per-camera JSON files in `backend/camera/settings/{platform}/`
- **Auto-load/save:** Settings automatically restored on startup and saved on change
- **Resolution switching:** Stream resolution can be changed dynamically (auto-restarts camera)
- **Camera compatibility:** IMX477 (12MP, 4056x3040), IMX219 (8MP, 3280x2464), and other CSI/USB cameras

#### Web Interface (`frontend/`)
- **Dual-camera slots:** Side-by-side video display
- **Per-camera controls:** Controls embedded under each camera feed
- **Real-time updates:** Slider changes apply immediately
- **Reset buttons:** Each control has a reset-to-default button
- **Profile system:** Save/load control presets
- **Resolution display:** Shows capture and stream resolutions
- **Page refresh handling:** Reconnects to already-streaming cameras automatically

#### Control System
- **Brightness:** -100 to +100 (maps to -2 to +2 EV compensation via `exposurecompensation`)
- **Digital Gain:** 0-100% (maps to ISP digital gain 1.0-8.0x via `ispdigitalgainrange`)
- **Rotation:** none, 90°CW, 90°CCW, 180°, flip-h, flip-v (via nvvidconv `flip-method`)
- **Stream Resolution:** 320x240, 640x480, 800x600, 1280x720, 1920x1080
- **White Balance:** off, auto, incandescent, fluorescent, warm-fluorescent, daylight, cloudy, twilight, shade
- **Auto-exposure lock:** Lock/unlock AE via `aelock` property

### 🔧 Technical Details

#### GStreamer Pipeline Structure
```
nvarguscamerasrc → tee → [full-res: nvvidconv→videoconvert→BGR→appsink]
                       → [preview: nvvidconv→resize→jpegenc→appsink]
```

#### Key Implementation Notes
1. **nvarguscamerasrc limitations:**
   - `exposuretimerange` and `gainrange` only constrain auto-exposure, don't set fixed values
   - Used `exposurecompensation` (-2 to +2 EV) for dynamic brightness control instead
   - Used `ispdigitalgainrange` for digital gain (works dynamically, unlike analog `gainrange`)

2. **Control scoping:**
   - Each camera has independent controls (fixed bug where controls affected both cameras)
   - Controls use `#controls-${cameraId}` container scoping
   - Value displays use `#value-${cameraId}-${controlName}` for uniqueness

3. **Settings files:**
   - Location: `backend/camera/settings/jetson/camera_0.json`, `camera_1.json`
   - Format: JSON with all control values
   - Auto-saved on every control change
   - Applied via `_apply_control()` with `save=False` to avoid recursion

4. **Page refresh behavior:**
   - Backend tracks streaming state in `enumerate_cameras()` response
   - Frontend calls `restoreStreamingCameras()` on load
   - Recreates video elements and loads controls without restarting cameras

#### File Structure
```
backend/
  ├── app.py                 # Flask API server
  ├── camera/
  │   ├── base.py            # Abstract base class for all camera backends
  │   ├── factory.py         # Platform detection and backend instantiation
  │   ├── jetson.py          # Jetson camera backend (nvarguscamerasrc)
  │   ├── webcam.py          # USB webcam backend
  │   ├── raspberry_pi.py    # Raspberry Pi camera backend
  │   ├── luxonis.py         # Luxonis OAK camera backend
  │   ├── groups.py          # Camera grouping and synchronization
  │   ├── profiles/{platform}/   # Platform-specific control presets
  │   ├── settings/{platform}/   # Per-camera persistent settings by platform
  │   └── sensor_configs/    # Camera sensor configurations
  ├── features/              # Plugin framework for image processing
  │   ├── base.py            # Base feature class
  │   ├── manager.py         # Feature plugin management
  │   └── plugins/           # Feature implementations (calibration, etc.)
  └── workflows/             # Multi-camera workflow system
      ├── base.py            # Base workflow class
      ├── manager.py         # Workflow management
      └── templates/         # Workflow templates

frontend/
  ├── index.html             # Main UI
  ├── app.js                 # Client logic
  └── style.css              # Styling
```

---

## Common Development Tasks

### Adding a New Camera Control

1. **Add to `get_controls()` in `jetson.py`:**
```python
'control_name': {
    'type': 'range|bool|menu',
    'min': 0, 'max': 100,  # for range
    'options': [...],       # for menu
    'default': value,
    'current': self.get_control(camera_id, 'control_name') or default,
    'label': 'Display Name',
    'description': 'Help text',
    'platform_name': 'gstreamer-property'  # if applicable
}
```

2. **Add to `_get_default_controls()`**

3. **Add handler in `_apply_control()`:**
```python
elif control_name == 'control_name':
    # Apply to GStreamer element
    src.set_property('property-name', value)
```

4. **Add to frontend category in `app.js`:**
```javascript
const categories = {
    'Category': ['control_name'],
    ...
}
```

### Debugging Camera Issues

**Check GStreamer pipeline:**
```bash
gst-inspect-1.0 nvarguscamerasrc
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! fakesink
```

**View backend logs:**
- Terminal running `python -m backend.app` shows all debug output
- Look for "Setting rotation for camera X" messages to verify control routing

**Check saved settings:**
```bash
cat backend/camera/settings/jetson/camera_0.json
```

### Known Issues & Workarounds

1. **Exposure control doesn't change actual exposure time:**
   - nvarguscamerasrc doesn't support true manual exposure on running pipeline
   - Workaround: Using `exposurecompensation` for relative brightness adjustment
   - Works with auto-exposure enabled, provides -2 to +2 EV range

2. **Analog gain control doesn't work:**
   - `gainrange` property constrains auto-gain but doesn't set fixed value
   - Workaround: Using `ispdigitalgainrange` for digital gain amplification
   - Maps 0-100% to ISP digital gain 1.0-8.0x

3. **Stream resolution requires restart:**
   - GStreamer caps can't be changed on running pipeline
   - Implementation auto-stops and restarts camera with new config
   - All other settings are reapplied after restart

---

## Environment Setup

### System Packages Required
```bash
python3-gi python3-gst-1.0 gstreamer1.0-tools 
gstreamer1.0-plugins-base gstreamer1.0-plugins-good
```

### Python Virtual Environment
```bash
python3 -m venv venv --system-site-packages  # Need system-site-packages for GStreamer
source venv/bin/activate
pip install flask numpy
```

### Running the Server
```bash
cd /home/dev/utils/web-preview
source venv/bin/activate
python -m backend.app
# Access at http://localhost:5000 or http://<jetson-ip>:5000
```

---

## API Endpoints Reference

### Camera Management
- `GET /api/cameras` - List cameras with streaming status
- `POST /api/cameras/<id>/start` - Start camera with config
- `POST /api/cameras/<id>/stop` - Stop camera
- `GET /api/cameras/<id>/stream` - MJPEG stream endpoint

### Controls
- `GET /api/cameras/<id>/controls` - Get all controls + resolution info
- `PUT /api/cameras/<id>/control/<name>` - Set control value
  - Body: `{"value": <value>}`

### Profiles
- `GET /api/cameras/<id>/profiles` - List available profiles
- `GET /api/cameras/<id>/profiles/<name>` - Get profile settings
- `POST /api/cameras/<id>/profiles/<name>` - Save profile
- `POST /api/cameras/<id>/profile` - Apply profile
  - Body: `{"profile_name": "name"}`

---

## Recent Bug Fixes

### 1. Controls Affecting Both Cameras (FIXED)
**Problem:** Changing rotation on camera 0 rotated both cameras  
**Cause:** `attachControlListeners()` used `document.querySelectorAll()` globally  
**Solution:** Scoped to container: `controlsContainer.querySelectorAll('[data-control]')`

### 2. Page Refresh Losing Cameras (FIXED)
**Problem:** Refreshing page showed no cameras even though backend still streaming  
**Cause:** Frontend didn't check streaming status on load  
**Solution:** 
- Backend returns `streaming: true/false` in camera list
- Frontend calls `restoreStreamingCameras()` to recreate UI

### 3. Settings Not Persisting (FIXED)
**Problem:** Control values reset on app restart  
**Cause:** No persistence layer  
**Solution:**
- Created `backend/camera/settings/jetson/` directory
- Auto-save on every control change
- Auto-load on camera start

---

## Future Enhancement Ideas

- [ ] Capture resolution control (currently fixed at 1920x1080)
- [ ] Frame rate control
- [ ] Snapshot/recording functionality
- [ ] Multi-camera synchronization
- [ ] Feature plugins (motion detection, object tracking)
- [ ] Workflow automation
- [ ] WebRTC streaming (lower latency than MJPEG)
- [ ] True manual exposure control (may require v4l2src instead of nvarguscamerasrc)

---

### Quick Start for New Copilot Instance

1. **Understand the architecture:** Platform-agnostic design with backend abstraction layer
   - Primary development: Jetson Orin Nano with IMX477 CSI cameras
   - Factory pattern selects appropriate backend (Jetson, Raspberry Pi, webcam, etc.)
2. **Check running state:** `ps aux | grep python` - is server running?
3. **View settings:** `ls backend/camera/settings/{platform}/` - see saved camera configs
4. **Test cameras:** Access web UI at http://localhost:5000
5. **Check logs:** Terminal output shows all control changes and debug info
6. **Key files:** 
   - `backend/camera/base.py` - Abstract camera interface
   - `backend/camera/factory.py` - Platform detection
   - `backend/camera/jetson.py` - Current implementation
   - `frontend/app.js` - UI logic

### Most Common User Requests
- "Add control for X" → See "Adding a New Camera Control" above
- "Control not working" → Check if it's in `_apply_control()`, verify GStreamer property name
- "Settings not saving" → Check `backend/camera/settings/jetson/camera_X.json` exists
- "Camera won't start" → Check GStreamer pipeline, verify camera hardware with `gst-launch-1.0`

