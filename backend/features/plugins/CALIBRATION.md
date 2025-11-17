# Camera Calibration Feature

The calibration feature provides comprehensive camera calibration capabilities using ChArUco boards.

## Features

- **Single Camera Calibration**: Calibrate individual cameras to correct lens distortion
- **Stereo Calibration**: Calibrate camera pairs for stereo vision applications
- **Standard & Fisheye Models**: Support for both standard pinhole and fisheye camera models
- **Guided Workflow**: Step-by-step calibration process
- **Real-time Marker Detection**: Visual feedback during image capture
- **Profile Management**: Save and load calibration profiles

## Architecture

### Plugin Structure

```
backend/features/plugins/calibration.py   # Calibration feature plugin
backend/workflows/templates/calibration.py # Calibration workflows
data/calibration/                          # Calibration data storage
  ├── session_YYYYMMDD_HHMMSS/            # Capture session
  │   ├── camera_0/                       # Per-camera images
  │   │   └── capture_*.png
  │   ├── camera_1/
  │   │   └── capture_*.png
  │   ├── camera_0_calibration.json       # Calibration results
  │   └── camera_1_calibration.json
```

### Integration with ne-calib-utils

The calibration plugin uses OpenCV's ArUco/ChArUco functionality directly and is designed to be platform-agnostic by integrating with the existing `CameraBackend` abstraction. The `ne-calib-utils` submodule serves as a reference implementation.

## Usage

### API Endpoints

All calibration operations are accessed through the feature plugin API:

#### Detect Markers
```bash
POST /api/features/calibration/process
{
  "camera_ids": ["0"],
  "params": {
    "action": "detect_markers"
  }
}
```

#### Start Calibration Session
```bash
POST /api/features/calibration/process
{
  "camera_ids": ["0"],
  "params": {
    "action": "start_session",
    "session_id": "my_session",
    "calibration_type": "single",
    "min_images": 20,
    "fisheye_model": false
  }
}
```

#### Capture Calibration Image
```bash
POST /api/features/calibration/process
{
  "camera_ids": ["0"],
  "params": {
    "action": "capture_image",
    "session_id": "my_session"
  }
}
```

#### Run Calibration
```bash
POST /api/features/calibration/process
{
  "camera_ids": ["0"],
  "params": {
    "action": "run_calibration",
    "session_id": "my_session",
    "calibration_type": "single",
    "fisheye_model": false
  }
}
```

### Workflow API

The guided calibration workflow provides a structured approach:

```bash
# Start workflow
POST /api/workflows/camera_calibration/start
{
  "camera_ids": ["0"]
}

# Execute step
POST /api/workflows/{instance_id}/step/{step_id}
{
  "data": {
    "camera_id": "0",
    "fisheye": false
  }
}
```

## Calibration Board

The calibration uses an 8x5 ChArUco board with:
- **Dictionary**: DICT_6X6_100
- **Square Size**: 50mm (0.05m)
- **Marker Size**: 37mm (0.037m)
- **Total Markers**: 20 (8×5 / 2)
- **Required Markers**: 18 (90% of total)

### Printing the Board

You can generate and print the calibration board using OpenCV:

```python
import cv2
import numpy as np

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)
board = cv2.aruco.CharucoBoard((8, 5), 0.05, 0.037, dictionary)
img = board.generateImage((800, 500))
cv2.imwrite('charuco_board.png', img)
```

Print the board on A4 paper ensuring:
- No scaling (actual size)
- High quality printer
- Flat, rigid surface mounting

## Calibration Process

### Single Camera Calibration

1. **Setup**
   - Print calibration board
   - Select camera
   - Configure fisheye model if needed

2. **Image Capture**
   - Capture 20-30 images
   - Vary board position and angle
   - Cover entire camera view
   - Ensure all markers visible

3. **Calibration**
   - Process images
   - Calculate camera matrix and distortion coefficients
   - Review reprojection error

4. **Save Results**
   - Save calibration profile
   - Apply to camera for undistortion

### Stereo Calibration

1. **Setup**
   - Select camera pair/group
   - Ensure rigid mounting

2. **Synchronized Capture**
   - Capture 25-40 synchronized pairs
   - Both cameras must see full board
   - Cover overlapping field of view

3. **Individual Calibration**
   - Calibrate each camera separately
   - Calculate intrinsic parameters

4. **Stereo Calibration**
   - Calculate extrinsic parameters
   - Compute rotation and translation
   - Generate rectification maps

5. **Review**
   - Check baseline distance
   - Verify rectification
   - Save stereo calibration

## Calibration Output

### Single Camera Calibration JSON

```json
{
  "camera_id": "0",
  "camera_matrix": [
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
  ],
  "distortion_coefficients": [k1, k2, p1, p2, k3],
  "image_size": [width, height],
  "reprojection_error": 0.23,
  "num_images": 25,
  "fisheye": false,
  "timestamp": "2025-11-17T..."
}
```

### Parameters

- **camera_matrix**: Intrinsic camera parameters (focal length, principal point)
- **distortion_coefficients**: Lens distortion parameters (radial and tangential)
- **reprojection_error**: RMS error in pixels (lower is better, <0.5 is excellent)
- **num_images**: Number of images used in calibration

## Integration with Camera Backend

The calibration plugin integrates with the existing camera backend architecture:

```python
# Get frames for calibration
frame = camera_backend.get_full_frame(camera_id)

# Apply calibration to undistort
undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)
```

## Future Enhancements

- [ ] Real-time undistortion preview
- [ ] Automatic image quality assessment
- [ ] Board pose estimation visualization
- [ ] Multi-camera calibration (>2 cameras)
- [ ] Calibration quality metrics dashboard
- [ ] Auto-capture with quality filtering
- [ ] Integration with camera profiles
- [ ] Depth map visualization for stereo
- [ ] Bundle adjustment optimization
- [ ] Support for different board types (checkerboard, April tags)

## Troubleshooting

### Markers Not Detected

- Ensure good lighting (even, no shadows)
- Print board at actual size
- Mount board on flat surface
- Keep board fully visible in frame

### High Reprojection Error

- Capture more images (30-40)
- Ensure varied board positions/angles
- Check for motion blur
- Verify board is flat and undamaged

### Calibration Fails

- Check minimum image count (15+)
- Verify marker detection in images
- Ensure sufficient marker visibility
- Try recapturing with better conditions

## References

- [OpenCV Camera Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [ChArUco Board Detection](https://docs.opencv.org/4.x/df/d4a/tutorial_charuco_detection.html)
- [Stereo Calibration](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
