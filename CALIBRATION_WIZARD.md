# Camera Calibration Wizard - UI Implementation

**Created:** 19 November 2025  
**Status:** UI Complete - Backend Integration Pending

## Overview

A comprehensive 4-step wizard interface for camera calibration without backend functionality. This provides the complete user experience for capturing calibration images, running calibration, and reviewing results.

---

## Wizard Steps

### Step 1: Setup
**Purpose:** Configure calibration parameters

**Features:**
- Camera selection dropdown
- Calibration pattern type selection (Checkerboard, Circles Grid, Asymmetric Circles)
- Pattern dimensions (width/height in corners)
- Physical square size (mm)
- Target number of images

**Validation:**
- Requires camera selection
- Captures all settings to `calibrationData` object

### Step 2: Capture
**Purpose:** Capture calibration images from different angles

**Features:**
- Live camera preview
- Start/Stop camera controls
- Capture image button
- Real-time statistics:
  - Images captured vs target
  - Pattern detection status
  - Coverage quality progress bar
- Thumbnail gallery with remove option
- Clear all images button
- Helpful capture tips

**Validation:**
- Minimum 10 images required to proceed

**Placeholder Functionality:**
- Camera preview uses actual stream endpoint
- Pattern detection simulated (random)
- Image thumbnails are SVG placeholders

### Step 3: Calibration
**Purpose:** Process images and compute calibration parameters

**Features:**
- Run Calibration button
- Status indicator (icon + messages)
- Progress bar with percentage
- Live calibration log
- Simulated processing steps

**Placeholder Functionality:**
- Generates fake calibration results:
  - Reprojection error: 0.2-0.7 pixels
  - Camera matrix (3x3)
  - Distortion coefficients (5 values)

**Validation:**
- Must run calibration before proceeding

### Step 4: Review
**Purpose:** Review results and save calibration

**Features:**
- **Calibration Metrics Card:**
  - Reprojection error
  - Images used
  - Quality assessment (Excellent/Good/Fair)

- **Camera Matrix Display:**
  - 3x3 matrix formatted display

- **Distortion Coefficients Display:**
  - 5 coefficient values

- **Save Options:**
  - Calibration name input
  - Save calibration images checkbox
  - Apply immediately checkbox

**Actions:**
- Save Calibration (placeholder)
- Export Data (downloads JSON file)
- Recalibrate (returns to Step 2)
- Finish (closes wizard)

---

## UI Components

### Progress Indicator
- 4-step horizontal progress bar
- Shows current step (highlighted)
- Completed steps show checkmark
- Connectors turn green when step completed
- Smooth animations

### Navigation
- Previous/Next buttons
- Previous disabled on Step 1
- Next becomes "Finish" on Step 4
- Cancel button with confirmation
- Step validation before advancing

### Styling
- Responsive design (mobile-friendly)
- Gradient progress bars
- Color-coded status indicators
- Card-based layout for results
- Smooth transitions and animations

---

## Data Structure

```javascript
calibrationData = {
    camera_id: null,              // Selected camera ID
    pattern: {
        type: 'checkerboard',     // Pattern type
        width: 9,                  // Corners width
        height: 6,                 // Corners height
        square_size: 25            // Physical size (mm)
    },
    target_images: 20,            // Target number of images
    captured_images: [],          // Array of captured image data
    results: {                    // Calibration results (null until run)
        reprojection_error: 0.345,
        images_used: 20,
        camera_matrix: [[...], [...], [...]],
        distortion_coeffs: [...]
    }
}
```

---

## File Changes

### `frontend/index.html`
- Added "Camera Calibration" button in Features section
- Added complete calibration wizard modal with all 4 steps
- Integrated with existing modal system

### `frontend/style.css`
- Added comprehensive calibration wizard styles (~450 lines)
- Progress indicator styling
- Form controls and layouts
- Capture preview and thumbnails
- Results cards and matrix displays
- Responsive breakpoints

### `frontend/app.js`
- Added calibration wizard initialization (~550 lines)
- Step navigation and validation
- Setup step handlers
- Capture step with camera controls
- Calibration step with progress simulation
- Review step with results display
- Export functionality

---

## Usage

1. Click "📐 Camera Calibration" button in sidebar
2. **Setup:** Select camera and configure pattern settings
3. **Capture:** Start camera, capture 10+ images from different angles
4. **Calibration:** Click "Run Calibration" and wait for processing
5. **Review:** Review results, enter name, and save

---

## Backend Integration Requirements

To make this functional, implement these API endpoints:

### Pattern Detection
```
GET /api/calibration/detect-pattern?camera_id=X
Returns: { detected: true/false, corners: [...] }
```

### Capture Image
```
POST /api/calibration/capture
Body: { camera_id, pattern_config }
Returns: { success: true, image_id, thumbnail }
```

### Run Calibration
```
POST /api/calibration/run
Body: { camera_id, pattern, image_ids }
Returns: { 
    success: true,
    reprojection_error,
    camera_matrix,
    distortion_coeffs,
    images_used
}
```

### Save Calibration
```
POST /api/calibration/save
Body: {
    name,
    camera_id,
    results,
    save_images,
    apply_immediately
}
Returns: { success: true, calibration_id }
```

---

## Known Placeholders

1. **Pattern Detection:** Currently random simulation
2. **Image Capture:** Creates SVG placeholders instead of actual images
3. **Calibration Processing:** Simulated with setTimeout
4. **Results:** Randomly generated values
5. **Save Operation:** Console log only

---

## Next Steps

1. Implement backend calibration plugin API endpoints
2. Integrate OpenCV calibration algorithms
3. Add real pattern detection visualization
4. Store captured images to filesystem
5. Persist calibration results to database
6. Add calibration history/management UI
7. Implement calibration application to camera pipeline

---

## Testing Checklist

- [x] Wizard opens and closes properly
- [x] Step navigation works forward/backward
- [x] Camera selection populates from available cameras
- [x] Pattern configuration saves to data object
- [x] Validation prevents advancing without requirements
- [x] Capture UI shows camera stream
- [x] Thumbnails can be added and removed
- [x] Progress animations work smoothly
- [x] Results display properly formatted
- [x] Export downloads JSON file
- [x] Responsive design works on mobile
- [ ] Backend integration (pending)
- [ ] Real pattern detection (pending)
- [ ] Actual calibration processing (pending)

---

## Design Decisions

1. **Wizard Pattern:** Provides guided experience for complex multi-step process
2. **Visual Progress:** Clear feedback on current position and completion
3. **Inline Validation:** Prevents user from advancing without meeting requirements
4. **Placeholder Data:** Allows UI testing without backend
5. **Export Option:** Enables data portability and debugging
6. **Tips & Guidance:** Helps users capture quality calibration images
7. **Quality Metrics:** Shows users if calibration is good enough
