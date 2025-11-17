# Camera Settings

This directory stores per-camera settings that persist between application runs.

## File Format

Each camera has its own settings file named `camera_<id>.json` containing the current control values:

```json
{
  "exposure": 0,
  "auto_exposure": true,
  "gain": 0,
  "auto_gain": true,
  "white_balance": "auto",
  "saturation": 1.0,
  "edge_enhancement": 0.0,
  "noise_reduction": "fast",
  "rotation": "none"
}
```

## Auto-Save

Settings are automatically saved whenever a control value is changed through the API.

## Auto-Load

When a camera is started, its saved settings are automatically loaded and applied to the hardware.

## Manual Editing

You can manually edit these files while the application is stopped. Invalid values will be ignored and replaced with defaults.
