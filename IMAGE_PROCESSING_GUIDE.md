# Image Processing Guide

This guide explains how to use the real-time image processing features to apply transformations to your camera streams.

## Overview

The image processing system allows you to apply various transformations to live camera feeds, including:
- **Undistortion** - Remove lens distortion using calibration data
- **Perspective Transform** - Apply perspective warping (4-point)
- **Affine Transform** - Apply affine transformation (3-point)
- **Rotation** - Rotate and scale the image
- **Custom Matrix** - Apply custom transformation matrices

## Quick Start

1. **Start a camera** in the web interface
2. **Scroll down** in the camera controls panel to find "Image Processing"
3. **Check "Enable"** to activate processing
4. **Select processing type** from the dropdown
5. **Configure parameters** based on the selected type
6. **Click "Apply Changes"** to see the effect on the live stream

## Processing Types

### 1. Undistort (Calibration)

Uses saved calibration data to remove lens distortion.

**Parameters:**
- **Calibration File**: Select from previously completed calibration sessions
- **Alpha (Crop Factor)**: 
  - `0.0` = Crop all invalid pixels (black borders removed)
  - `1.0` = Keep all pixels (may show black borders)
  - Intermediate values balance between field of view and border removal

**Usage:**
1. Complete a camera calibration first (see calibration wizard)
2. Enable image processing
3. Select "Undistort" as the type
4. Click "🔄 Refresh Files" to load available calibration sessions
5. Choose a calibration from the dropdown
6. Adjust alpha slider as desired
7. Click "Apply Changes"

**When to use:**
- Removing fisheye or barrel distortion from wide-angle lenses
- Achieving accurate measurements from camera images
- Preparing images for computer vision tasks requiring straight lines

### 2. Perspective Transform

Apply a perspective warp using 4 corresponding points.

**Parameters:**
- **Source Points**: 4 points on the original image (x1,y1 x2,y2 x3,y3 x4,y4)
- **Destination Points**: Where those points should map to

**Format:** Space-separated coordinate pairs, e.g., `0,0 100,0 100,100 0,100`

**Usage Example:**
```
Source Points:     50,80 250,60 280,200 20,220
Destination Points: 0,0 200,0 200,200 0,200
```

**When to use:**
- Creating a "bird's eye view" from an angled camera
- Straightening tilted images
- Correcting perspective distortion

### 3. Affine Transform

Apply an affine transformation using 3 corresponding points.

**Parameters:**
- **Source Points**: 3 points on the original image (x1,y1 x2,y2 x3,y3)
- **Destination Points**: Where those points should map to

**Format:** Same as perspective, but only 3 points

**When to use:**
- Simpler transformations (rotation, scaling, skewing)
- When you don't need full perspective correction
- Faster processing than perspective transform

### 4. Rotation

Rotate the image around its center.

**Parameters:**
- **Rotation Angle**: -180° to +180° (use slider)
- **Scale**: 0.1 to 3.0 (zoom in/out while rotating)

**When to use:**
- Correcting camera mounting angle
- Rotating image for different viewing orientations
- Creating zoom effects

### 5. Custom Matrix

Apply a custom transformation matrix (advanced).

**Parameters:**
- **Matrix Type**: 
  - `3x3 Perspective` - Full perspective transform
  - `2x3 Affine` - Affine transform
- **Matrix Values**: Comma-separated values

**3x3 Perspective Format (9 values):**
```
m11,m12,m13,m21,m22,m23,m31,m32,m33
```
Identity: `1,0,0,0,1,0,0,0,1`

**2x3 Affine Format (6 values):**
```
m11,m12,m13,m21,m22,m23
```
Identity: `1,0,0,0,1,0`

**When to use:**
- Implementing custom transformations
- Combining multiple effects
- Advanced computer vision applications
- Testing specific matrix values

## Tips and Best Practices

### Performance
- Processing is applied to the preview stream (not full resolution)
- Multiple processing types can be tested quickly
- Disable processing when not needed to reduce CPU usage

### Calibration Workflow
1. First, complete camera calibration with the wizard
2. Review calibration results in the review panel
3. If quality is good (reprojection error < 0.5), save the calibration
4. Use the saved calibration in image processing for undistortion

### Testing Transformations
1. Start with simple transformations (rotation, undistortion)
2. Use small parameter changes to see effects clearly
3. The "Apply Changes" button lets you test without continuous updates
4. Disable processing to compare original vs processed stream

### Point Selection
- Points are in image coordinates (pixels)
- Origin (0,0) is top-left corner
- For perspective/affine, ensure points form a valid transformation
- Point format: `x,y` separated by spaces

## Troubleshooting

### Processing Not Applied
- Check that "Enable" checkbox is checked
- Ensure you clicked "Apply Changes" button
- Verify parameters are valid (correct number of points, valid matrix values)
- Check browser console for error messages

### Calibration File Not Available
- Complete a camera calibration session first
- Click "🔄 Refresh Files" to reload the list
- Check that calibration was saved successfully (review panel shows results)

### Image Distorted or Black
- For undistortion: Try different alpha values (0.0 to 1.0)
- For perspective/affine: Check that points are correct
- For custom matrix: Verify matrix values are valid
- Try disabling and re-enabling processing

### Performance Issues
- Processing adds computational overhead
- Consider reducing stream resolution if frames lag
- Disable processing when not actively needed
- Some processing types (perspective) are more expensive than others

## API Integration

For programmatic control, use the REST API endpoints:

**Enable Processing:**
```bash
curl -X POST http://localhost:5000/api/processing/0/enable \
  -H "Content-Type: application/json" \
  -d '{"type": "undistort", "calibration_file": "data/calibration/session_20251120_123456/calibration.json", "alpha": 0.0}'
```

**Update Parameters:**
```bash
curl -X POST http://localhost:5000/api/processing/0/update \
  -H "Content-Type: application/json" \
  -d '{"type": "rotation", "angle": 45, "scale": 1.2}'
```

**Disable Processing:**
```bash
curl -X POST http://localhost:5000/api/processing/0/disable
```

## Technical Details

- Processing is applied in `camera_stream()` function
- Transformations use OpenCV functions:
  - `cv2.undistort()` for undistortion
  - `cv2.warpPerspective()` for perspective transforms
  - `cv2.warpAffine()` for affine transforms and rotation
- Frame pipeline: Decode JPEG → Apply transform → Re-encode JPEG
- Processing state stored per-camera in backend dictionaries
- Configuration persists until disabled or server restart

## Examples

### Example 1: Remove Lens Distortion
1. Calibrate camera with ChArUco board (20+ images)
2. Enable processing, select "Undistort"
3. Choose calibration file from dropdown
4. Set alpha to 0.0 for clean image
5. Apply changes

### Example 2: Bird's Eye View
1. Enable processing, select "Perspective Transform"
2. Identify 4 corners of rectangular object in image
3. Enter source points (measured from camera view)
4. Enter destination points forming a rectangle
5. Apply to see top-down view

### Example 3: Rotate Mounted Camera
1. Enable processing, select "Rotation"
2. Adjust angle slider to correct orientation
3. Adjust scale if needed to fit frame
4. Apply changes

## Future Enhancements

Planned features:
- Interactive point selection on camera preview (click to select)
- Preset transformations (common perspectives, rotations)
- Processing presets save/load
- Multi-step processing pipeline
- Processing applied to full-resolution capture
