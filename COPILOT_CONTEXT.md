# GitHub Copilot Context - Camera Streaming Platform

**Last Updated:** 20 November 2025  
**Primary Platform:** NVIDIA Jetson Orin Nano (platform-agnostic design)  
**Cameras:** 2x IMX477 CSI cameras (supports multiple camera models)  
**Status:** Fully functional dual-camera streaming system with web interface, camera calibration wizard, real-time image processing, and panorama calibration with synchronized dual-camera capture

---

## 🚧 ACTIVE WORK - Stereo Calibration for Panorama (21 Nov 2025)

### Recently Implemented
✅ **Stereo calibration workflow** for computing extrinsics between camera pairs
✅ **Divergent camera support** with configurable calibration flags for minimal overlap scenarios
✅ **Optimal homography computation** from stereo rotation matrix for geometric accuracy
✅ **UI controls** for stereo calibration settings and flag selection
✅ **Calibration selection interface** - Users can now select which calibration session to use for each camera

### What's Working
✅ Stereo calibration API endpoints (`/api/calibration/stereo/compute`, `/load`, `/info`)
✅ **Calibration session dropdowns** for Camera 0 and Camera 1 in panorama modal
✅ Individual camera calibration loading from selected sessions
✅ Visual status indicators (✅/❌) showing calibration selection state
✅ Validation that both calibrations are selected before stereo calibration
✅ ChArUco-based correspondence matching across synchronized pairs
✅ Rotation (R) and Translation (T) matrix computation via cv2.stereoCalibrate()
✅ Essential and Fundamental matrix calculation
✅ Rectification transforms for epipolar alignment
✅ Optimal homography: H = K2 * R * K1^-1
✅ Configurable calibration flags via UI (CALIB_FIX_INTRINSIC, etc.)
✅ Comprehensive results display with Euler angles and translation vectors

### Calibration Selection Features
- **Dropdown population:** Automatically loads all calibration sessions from `/api/calibration/sessions`
- **Session filtering:** Shows calibration sessions with results, filtered by camera ID
- **Error display:** Shows reprojection error in dropdown (e.g., "session_xxx (0.523px)")
- **Status indicators:** Green checkmark when calibration selected, red X when missing
- **Validation:** Frontend checks both calibrations are selected before allowing stereo calibration
- **Backend support:** API accepts `cam1_calibration_session` and `cam2_calibration_session` parameters
- **Manual loading:** Loads calibration results directly from session's `calibration_results.json`

### UI Changes
**Added to Panorama Modal (frontend/index.html):**
- Orange-highlighted "Individual Camera Calibrations" panel
- Two dropdown selects: `panorama-cam1-calib-select` and `panorama-cam2-calib-select`
- Status divs showing selection state
- Help text: "Required for stereo calibration. Use single-camera calibration wizard if needed."

**JavaScript Functions Added (frontend/app.js):**
- `loadPanoramaCalibrationOptions()`: Fetches and populates calibration dropdowns
- `updatePanoramaCalibrationStatus(camera)`: Updates visual status indicators
- Updated `computePanoramaCalibration()`: Validates selections and sends session names to API

**Backend Changes (backend/app.py):**
- Added `cam1_calibration_session` and `cam2_calibration_session` parameters
- Manual JSON loading from specified sessions instead of auto-detection
- Better error messages indicating which camera calibration is missing

### Implementation Notes
1. **Calibration Loading:** Instead of using the calibration plugin's auto-load (which looks for `camera_{id}` in default location), the system now loads directly from the user-selected session's `data/calibration/{session}/{camera_id}/calibration_results.json`

2. **Session Validation:** The frontend validates that calibration results exist before adding sessions to dropdowns

3. **Flexible Camera IDs:** System handles camera IDs as strings ('0', '1') or integers (0, 1) for compatibility

4. **Error Handling:** Clear error messages guide users to select calibrations if missing

### Files Modified (Calibration Selection Fix)
- `frontend/index.html`: Added calibration selection dropdowns with status indicators
- `frontend/app.js`: Added `loadPanoramaCalibrationOptions()` and validation logic
- `backend/app.py`: Updated stereo calibration endpoint to accept session names

### Previous Issue (Session Loading)
❌ Session dropdown in panorama calibration wizard not populating with existing sessions
- Status: Not yet addressed, stereo calibration was prioritized

---

### Current Issue
Session dropdown in panorama calibration wizard not populating with existing sessions after implementing session persistence.

### What's Working
✅ Composite preview stream with side-by-side cameras (1280x360 total: 640x360 per camera)
✅ Hardware synchronization (<16ms frame sync via single GStreamer pipeline)
✅ Text overlays for camera identification ("Camera 0", "Camera 1" in green)
✅ 16:9 aspect ratio matching full-res (640x360 preview matches 1920x1080 ratio)
✅ Session persistence to disk with proper directory structure
✅ Image capture saves to data/calibration/panorama_YYYYMMDD_HHMMSS/{0,1}/
✅ Session metadata (session_info.json) stores camera pair and timestamps
✅ Updated /api/calibration/sessions endpoint to detect panorama sessions

### What's Being Debugged
❌ Session dropdown not showing existing sessions in UI
❌ JavaScript loadPanoramaSessions() may not be filtering correctly
❌ Potential mismatch between old session format (0_1_timestamp) and new format (panorama_*)

### Session Format Evolution
- **New format:** `panorama_YYYYMMDD_HHMMSS/` with subdirectories `0/` and `1/`
- **Old format:** `0_1_1763667668575/` (timestamp-based) - needs backward compatibility
- **Detection:** Backend now checks for `session_info.json` to identify old panorama sessions
- **API:** Returns all sessions, frontend filters for panorama_* (may need client-side fix)

### Files Modified for Panorama
1. `backend/camera/sync_pair.py`: Composite pipeline with compositor element
2. `backend/app.py`: Session persistence, calibration loading fixes, panorama endpoints
3. `frontend/index.html`: Two-column layout, session selector dropdown
4. `frontend/app.js`: loadPanoramaSessions(), session selection handlers
5. `frontend/style.css`: panorama-grid layout

### Next Steps for Tomorrow
1. Restart server and test /api/calibration/sessions endpoint with curl
2. Check if endpoint returns all 3 existing sessions (including 0_1_1763667668575)
3. Debug JavaScript console for errors in loadPanoramaSessions()
4. Fix client-side filtering to include old-format sessions (check for session_info.json)
5. Fix separate issue: calibration_utils import error in camera stream code

### Existing Sessions on Disk
- `session_20251119_190310` (unknown type - check session_info.json)
- `session_20251120_155650` (unknown type - check session_info.json)
- `0_1_1763667668575` (old panorama format with session_info.json confirmed)

### API Endpoints for Panorama
- `GET /api/calibration/sessions` - Lists all calibration sessions (panorama and single-camera)
- `POST /api/calibration/panorama/capture` - Capture synchronized image pair
- `POST /api/calibration/panorama/compute` - Compute stereo calibration from session
- `POST /api/calibration/panorama/stitch` - Generate panorama from calibrated pair

---

---

## Project Overview

A platform-agnostic Flask-based camera streaming platform designed for multiple hardware platforms and camera types. Currently developed and tested on NVIDIA Jetson with CSI cameras. Features real-time MJPEG streaming, hardware-accelerated processing via GStreamer, camera calibration using ChArUco boards, and a responsive web UI for camera control.

### Key Architecture
- **Backend:** Python/Flask with GStreamer (platform-specific backends) + OpenCV for calibration
- **Frontend:** Vanilla HTML/CSS/JavaScript with wizard-based calibration interface
- **Streaming:** Dual-stream architecture (full-res + preview JPEG)
- **Platform Design:** Abstracted camera backends via factory pattern
- **Current Hardware:** NVIDIA Jetson Orin Nano with 2x IMX477 CSI cameras
- **Supported Cameras:** IMX477 (12MP), IMX219 (8MP), and expandable to other models
- **Calibration:** ChArUco board detection with automatic image capture and OpenCV camera calibration

---

## Current Implementation Status

### ✅ Completed Features

#### Camera Calibration System (`backend/calibration_utils.py`, `frontend/`)
- **Full calibration wizard:** 4-step process (Setup → Capture → Calibration → Review)
- **ChArUco board support:** Configurable board size, marker dictionary, and physical dimensions
- **Automatic pattern detection:** Real-time detection on preview stream with marker count and quality feedback
- **Auto-capture mode:** Interval-based capture (1-10 seconds) with movement detection (50px threshold)
- **Pose diversity tracking:** 12 pose types (center, 4 corners, 4 edges, tilted, close, far) with visual feedback
- **Coverage map:** Visual representation of captured board positions with quality indicators
- **Session management:** 
  - Save/load calibration sessions
  - Auto-generated session names (session_YYYYMMDD_HHMMSS)
  - Image persistence to disk (data/calibration/session_*/camera_id/)
  - Session metadata (session_info.json)
- **OpenCV calibration:**
  - ArUco marker detection → ChArUco corner interpolation
  - Camera matrix and distortion coefficient calculation
  - Per-image reprojection error analysis
  - Quality assessment (Excellent/Very Good/Good/Fair/Poor based on error)
- **Distortion correction preview:** Side-by-side comparison of original vs undistorted images
- **Image navigation:** Previous/Next buttons to review all calibration images
- **Skip to calibration:** Button to jump directly to calibration when loading existing sessions
- **Calibration results:**
  - Reprojection error (pixels)
  - Camera matrix (3x3)
  - Distortion coefficients (5 values)
  - Per-image error breakdown with visual indicators
  - Image details table with corner counts and errors

#### Image Processing System (`backend/app.py`, `frontend/`)
- **Real-time transformation:** Apply image processing to live camera streams
- **Multiple processing types:**
  - Undistortion using calibration data
  - Perspective transformation (4-point)
  - Affine transformation (3-point)
  - Rotation with scale
  - Custom transformation matrices
- **Dynamic parameter control:** Real-time adjustment via web UI sliders
- **Calibration integration:** Load saved calibration files for undistortion
- **Processing pipeline:** Applied after calibration overlay in stream
- **UI controls:** Toggle enable/disable, type selector, parameter inputs

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
  ├── app.py                 # Flask API server with calibration endpoints
  ├── calibration_utils.py   # OpenCV calibration functions
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
  ├── index.html             # Main UI with calibration wizard
  ├── app.js                 # Client logic including calibration workflow
  └── style.css              # Styling with wizard and calibration UI

data/
  └── calibration/           # Calibration sessions (gitignored)
      └── session_*/         # Timestamped session folders
          └── {camera_id}/   # Per-camera images and results
              ├── image_*.jpg           # Captured calibration images
              ├── session_info.json     # Board config and metadata
              └── calibration_results.json  # Calibration output
```

---

## Calibration System Details

### Calibration Workflow

**Step 1: Setup**
- Select camera and calibration model (pinhole/fisheye)
- Configure ChArUco board (dictionary, size, physical dimensions)
- Set target image count (recommended 20-30)
- Load previous calibration sessions for continuation

**Step 2: Capture**
- Live camera preview with real-time ChArUco detection overlay
- Pattern detection status (markers detected, quality feedback)
- Auto-capture mode with configurable interval and movement detection
- Manual capture button
- Coverage map showing board positions across image
- Pose diversity tracking (12 pose types)
- Captured image count vs target

**Step 3: Calibration**
- Run OpenCV calibration on captured images
- Progress bar with detailed log messages
- Processing steps: load images → detect patterns → compute parameters → refine
- Real-time progress updates

**Step 4: Review**
- Side-by-side original vs undistorted image comparison
- Navigate through all captured images with Prev/Next buttons
- Calibration metrics (reprojection error, quality assessment)
- Camera matrix (3x3 intrinsic parameters)
- Distortion coefficients (k1, k2, p1, p2, k3)
- Per-image error breakdown with visual indicators
- Save/export options

### Calibration API Endpoints

- `GET /api/calibration/sessions` - List all calibration sessions with image counts
- `GET /api/calibration/session/<path>` - Load specific session with detection analysis
- `POST /api/calibration/capture-image` - Save calibration image to session
- `POST /api/calibration/run` - Execute calibration on session images
- `POST /api/calibration/undistort-image` - Generate original vs undistorted comparison
- `POST /api/calibration/overlay/<id>/enable` - Enable ChArUco detection overlay
- `POST /api/calibration/overlay/<id>/disable` - Disable detection overlay

### Image Processing API Endpoints

- `POST /api/processing/<id>/enable` - Enable image processing with configuration
  - Body: `{"type": "undistort|perspective|affine|rotation|custom_matrix", ...params}`
- `POST /api/processing/<id>/disable` - Disable image processing for camera
- `POST /api/processing/<id>/update` - Update processing parameters
  - Body: Same as enable endpoint

**Processing Types:**

1. **Undistort** (uses calibration data):
   - `calibration_file`: Path to calibration.json
   - `alpha`: Crop factor (0.0 = crop invalid pixels, 1.0 = keep all)
   
2. **Perspective Transform**:
   - `src_points`: 4 source points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
   - `dst_points`: 4 destination points in same format
   
3. **Affine Transform**:
   - `src_points`: 3 source points [[x1,y1], [x2,y2], [x3,y3]]
   - `dst_points`: 3 destination points in same format
   
4. **Rotation**:
   - `angle`: Rotation angle in degrees (-180 to 180)
   - `scale`: Scale factor (0.1 to 3.0)
   
5. **Custom Matrix**:
   - `matrix`: List of 9 values (3x3 perspective) or 6 values (2x3 affine)

**Implementation:**
- Processing applied in `camera_stream()` after calibration overlay
- Frame decoded → transformation applied → re-encoded to JPEG
- Frontend controls integrated in camera control panel
- Real-time parameter adjustment via sliders and inputs

### Calibration Configuration

**ChArUco Board Parameters:**
- Dictionary: 17 ArUco dictionaries (4x4_50 through ARUCO_ORIGINAL)
- Board dimensions: 4-12 markers wide, 3-10 markers tall
- Square length: Physical size in mm (10-200mm)
- Marker length: Physical size in mm (5-150mm)

**Capture Settings:**
- Auto-capture interval: 1-10 seconds
- Movement threshold: 50 pixels minimum
- Minimum markers: 10 for valid detection
- Minimum corners: 10 ChArUco corners for calibration

**Pose Classification:**
- Center: Board in central region
- Corners: 4 corner positions (top-left, top-right, bottom-left, bottom-right)
- Edges: 4 edge positions (top, bottom, left, right)
- Tilted: Significant rotation (>15°)
- Close: Large board size (close to camera)
- Far: Small board size (far from camera)

### Calibration Implementation Notes

1. **Detection Pipeline:**
   - Server-side detection on preview stream (not full-res for performance)
   - `cv2.aruco.detectMarkers()` finds ArUco markers
   - `cv2.aruco.interpolateCornersCharuco()` computes ChArUco corners
   - Pose classification analyzes marker positions for diversity

2. **Image Capture:**
   - Full-resolution frames saved to disk as JPEG
   - Sequential numbering (image_0001.jpg, image_0002.jpg, etc.)
   - Session metadata includes board config and timestamps
   - Auto-generated session names with timestamp

3. **Calibration Algorithm:**
   - OpenCV's `cv2.aruco.calibrateCameraCharuco()`
   - Requires minimum 5 images with sufficient corners
   - Computes camera matrix and distortion coefficients
   - Calculates per-image reprojection error
   - Returns comprehensive results with image-level details

4. **Session Management:**
   - Sessions stored in `data/calibration/` (gitignored)
   - Load previous sessions to continue capturing
   - Session analysis rebuilds coverage map and pose stats
   - "Skip to Calibration" button for re-running calibration

5. **Coverage Map Persistence:**
   - Coverage data stored during session loading
   - Map preserved when navigating between wizard steps
   - Canvas redrawn when entering capture step
   - Not cleared when camera starts (preserves loaded sessions)

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

### Camera Calibration System (COMPLETED)
**Features Added:**
- Complete 4-step calibration wizard (Setup → Capture → Calibration → Review)
- ChArUco board detection with configurable parameters
- Auto-capture with interval and movement detection
- Pose diversity tracking (12 pose types)
- Coverage map visualization
- Session persistence and loading
- OpenCV calibration with comprehensive results
- Side-by-side distortion correction preview
- Image navigation (Previous/Next buttons)
- Skip to calibration for existing sessions

**Bug Fixes:**
1. **Image capture not saving to disk** - Added `/api/calibration/capture-image` endpoint
2. **Calibration using wrong API** - Changed from CharucoDetector to classic detectMarkers + interpolateCornersCharuco
3. **Coverage map cleared on session load** - Fixed to preserve loaded coverage data
4. **Board config not available in calibration step** - Changed to use calibrationData.board instead of DOM elements
5. **Undistorted image severely warped** - Fixed to use simple undistort() without getOptimalNewCameraMatrix
6. **Coverage map disappears briefly** - Fixed to not clear coverageMapData when camera starts if data exists

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

- [x] Camera calibration wizard with ChArUco boards
- [x] Distortion correction preview
- [x] Calibration session management
- [ ] Apply calibration to live camera streams
- [ ] Save/export calibration data for external use
- [ ] Stereo calibration for dual-camera setup
- [ ] Capture resolution control (currently fixed at 1920x1080)
- [ ] Frame rate control
- [ ] Snapshot/recording functionality beyond calibration
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
4. **View calibration sessions:** `ls data/calibration/` - see saved calibration data
5. **Test cameras:** Access web UI at http://localhost:5000
6. **Test calibration:** Click "Calibrate Camera" button for any camera
7. **Check logs:** Terminal output shows all control changes and debug info
8. **Key files:** 
   - `backend/camera/base.py` - Abstract camera interface
   - `backend/camera/factory.py` - Platform detection
   - `backend/camera/jetson.py` - Current implementation
   - `backend/calibration_utils.py` - Calibration algorithms
   - `frontend/app.js` - UI logic including calibration wizard

### Most Common User Requests
- "Add control for X" → See "Adding a New Camera Control" above
- "Control not working" → Check if it's in `_apply_control()`, verify GStreamer property name
- "Settings not saving" → Check `backend/camera/settings/jetson/camera_X.json` exists
- "Camera won't start" → Check GStreamer pipeline, verify camera hardware with `gst-launch-1.0`
- "Calibration not detecting board" → Check ChArUco board parameters match physical board
- "Calibration images not saving" → Check `data/calibration/session_*/` directory permissions
- "Coverage map not showing" → Check browser console for errors, verify canvas initialization

