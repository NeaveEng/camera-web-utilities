# Stereo Calibration for Panorama Stitching

## Overview

The stereo calibration feature computes the **extrinsic relationship** (rotation and translation) between two cameras using synchronized ChArUco board captures. This provides a geometrically accurate foundation for panorama stitching, especially for **divergent cameras with minimal overlap**.

## Why Stereo Calibration?

### Traditional Homography Approach
- Finds 2D planar transformation between overlapping regions
- Assumes cameras are viewing a planar scene
- Works for flat scenes but struggles with depth variation
- Doesn't account for 3D camera relationship

### Stereo Calibration Approach
- Computes actual 3D rotation (R) and translation (T) between cameras
- Provides Essential (E) and Fundamental (F) matrices for epipolar geometry
- Enables computation of **optimal homography** from rotation matrix
- Better handles parallax and non-planar scenes
- Reusable for fixed camera rigs

## Workflow

### 1. Prerequisites
- Individual calibration for both cameras (camera matrix and distortion coefficients)
- Synchronized image captures with ChArUco board visible in both cameras
- Minimum 3 valid pairs (recommended 10+ for accuracy)

### 2. Capture Process
1. Open Panorama Calibration modal
2. Select both cameras
3. Configure ChArUco board settings
4. Position board where both cameras can see it (even partially)
5. Capture multiple synchronized image pairs
6. Board should be in different positions/orientations

### 3. Stereo Calibration Settings

#### Compute Stereo Calibration
Enable to calculate rotation and translation between cameras.

#### Calibration Flags
Select one or more flags to control the calibration process:

- **CALIB_FIX_INTRINSIC** (Recommended for divergent cameras)
  - Uses fixed camera matrices from individual calibrations
  - Only computes extrinsics (R, T)
  - Best when individual calibrations are already accurate

- **CALIB_USE_INTRINSIC_GUESS**
  - Optimizes camera matrices using individual calibrations as starting point
  - Can improve results if individual calibrations are approximate

- **CALIB_FIX_PRINCIPAL_POINT**
  - Fixes principal point (cx, cy) during optimization
  - Use when principal point is well-calibrated

- **CALIB_FIX_FOCAL_LENGTH**
  - Fixes focal lengths (fx, fy) during optimization
  - Prevents focal length changes

- **CALIB_FIX_ASPECT_RATIO**
  - Maintains fixed aspect ratio (fx/fy)
  - Only optimizes one focal length dimension

- **CALIB_SAME_FOCAL_LENGTH**
  - Enforces identical focal lengths for both cameras
  - Only use if cameras are truly identical

#### Use Optimal Homography
When enabled, computes homography from stereo calibration:
```
H = K2 * R * K1^-1
```
where:
- K1, K2 = camera matrices
- R = rotation matrix between cameras

This is more geometrically accurate than feature-based homography estimation.

## Results

### Stereo Calibration Output
- **Reprojection Error**: Quality metric in pixels (lower is better)
  - < 1.0 px: Excellent
  - < 2.0 px: Good
  - > 2.0 px: Consider more captures

- **Rotation Matrix (3x3)**: 3D rotation from camera 1 to camera 2
  - Also shown as Euler angles (pitch, yaw, roll)

- **Translation Vector (3x1)**: 3D position offset
  - X, Y, Z components in same units as board dimensions
  - Distance magnitude calculated

- **Essential Matrix (E)**: Encodes rotation and translation
  - Used for epipolar geometry

- **Fundamental Matrix (F)**: Relates corresponding points
  - Accounts for camera intrinsics

- **Rectification Transforms**: R1, R2, P1, P2, Q
  - For aligning epipolar lines (optional use)

## For Divergent Cameras

Divergent cameras (pointing in different directions) have special considerations:

### Challenges
- Minimal overlap region
- Fewer matching features
- Greater geometric distortion

### Best Practices
1. **Use CALIB_FIX_INTRINSIC flag**
   - Trust individual camera calibrations
   - Only solve for rotation and translation

2. **Capture strategy**
   - Position board in the small overlap region
   - Use larger boards for better detection
   - Capture from multiple distances and angles
   - Aim for 10-20 valid pairs minimum

3. **Board placement tips**
   - Board doesn't need to fill both frames
   - Partial board visibility is acceptable (minimum 4 corners)
   - Ensure consistent lighting across both views
   - Vary board tilt to improve geometry

4. **Validation**
   - Check that rotation angles match physical camera setup
   - Translation vector should align with camera positions
   - Low reprojection error indicates good calibration

## API Endpoints

### Compute Stereo Calibration
```
POST /api/calibration/stereo/compute
Body: {
  "session_id": "panorama_20251121_120000",
  "flags": "CALIB_FIX_INTRINSIC|CALIB_FIX_PRINCIPAL_POINT"
}
```

### Load Stereo Calibration
```
GET /api/calibration/stereo/load?camera1_id=0&camera2_id=1
```

### Get Info
```
GET /api/calibration/stereo/info
```

## Storage

Stereo calibrations are saved to:
```
backend/camera/settings/stereo/{camera1_id}_{camera2_id}.json
```

Format:
```json
{
  "camera1_id": "0",
  "camera2_id": "1",
  "stereo_calibration": {
    "rotation_matrix": [[...], [...], [...]],
    "translation_vector": [x, y, z],
    "essential_matrix": [[...], [...], [...]],
    "fundamental_matrix": [[...], [...], [...]],
    "reprojection_error": 0.85,
    "optimal_homography": [[...], [...], [...]],
    ...
  },
  "version": "1.0"
}
```

## Implementation Details

### Backend Functions

**`stereo_calibrate_from_sessions()`** - `backend/calibration_utils.py`
- Loads synchronized image pairs
- Detects ChArUco in both cameras
- Matches common corners by ID
- Runs `cv2.stereoCalibrate()`
- Computes rectification transforms

**`compute_optimal_homography_from_stereo()`** - `backend/calibration_utils.py`
- Derives homography from rotation matrix
- Formula: H = K2 @ R @ inv(K1)
- Normalized to H[2,2] = 1

**`save_stereo_calibration()`** - `backend/panorama_utils.py`
- Saves results to JSON
- Creates stereo directory if needed

### Frontend Integration

**UI Controls** - `frontend/index.html`
- Stereo calibration checkbox
- Multi-select flag selector
- Optimal homography checkbox

**Calibration Flow** - `frontend/app.js`
- `computePanoramaCalibration()`: Orchestrates stereo + homography workflow
- `displayStereoResults()`: Shows rotation, translation, and metrics
- `displayPanoramaResults()`: Updated to show both methods

## Troubleshooting

### "Not enough valid stereo pairs"
- Need minimum 3 pairs with common corners
- Ensure board is visible in both cameras
- Check that board detection is working (green overlays)

### High reprojection error
- Add more image pairs
- Improve board visibility and lighting
- Ensure board is flat and printed accurately
- Check that individual camera calibrations are good

### "Camera calibration not found"
- Calibrate individual cameras first
- Use the single-camera calibration wizard
- Verify calibration files exist in `data/calibration/*/camera_{id}/`

### Rotation angles don't match camera setup
- Check camera selection (camera1 vs camera2)
- Verify cameras are streaming from correct sensors
- Ensure ChArUco board orientation is consistent

## Advanced Usage

### Cylindrical Projection
For wide panoramas, you can use the rectification transforms to project onto a cylinder:
```python
# Use P1, P2 from stereo calibration
# Apply cylindrical warp before stitching
```

### Depth Mapping
Essential matrix enables stereo matching:
```python
# Rectify images using R1, R2
# Compute disparity map
# Convert to depth using Q matrix
```

### Quality Metrics
Monitor these for calibration quality:
- Reprojection error < 1.0 px
- Inlier ratio > 70%
- Common corners per pair > 10
- RMSE < 2.0 px

## References

- OpenCV Stereo Calibration: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
- Multiple View Geometry in Computer Vision (Hartley & Zisserman)
- ChArUco Boards: https://docs.opencv.org/4.x/df/d4a/tutorial_charuco_detection.html
