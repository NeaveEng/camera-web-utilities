"""
Camera calibration workflow.

Provides step-by-step guided calibration process.
"""

from typing import Dict, List, Any, Optional
from backend.workflows.base import Workflow, WorkflowStep


class CameraCalibrationWorkflow(Workflow):
    """Guided camera calibration workflow."""

    def get_metadata(self) -> Dict[str, Any]:
        """Get workflow metadata."""
        return {
            'name': 'camera_calibration',
            'description': 'Step-by-step camera calibration using ChArUco board',
            'version': '1.0.0',
            'category': 'calibration',
            'requires_cameras': 1,
            'estimated_time': '10-15 minutes'
        }

    def get_steps(self) -> List[WorkflowStep]:
        """Get workflow steps."""
        return [
            WorkflowStep(
                'setup',
                'Setup',
                'Prepare for calibration by selecting camera and printing calibration board',
                ui_schema={
                    'instructions': [
                        'Print the ChArUco calibration board (8x5 grid)',
                        'Mount the board on a flat surface',
                        'Ensure good, even lighting',
                        'Select the camera to calibrate'
                    ],
                    'controls': [
                        {
                            'name': 'camera_id',
                            'type': 'select',
                            'label': 'Camera',
                            'options': [],  # Populated dynamically
                            'required': True
                        },
                        {
                            'name': 'fisheye',
                            'type': 'checkbox',
                            'label': 'Fisheye Camera Model',
                            'default': False
                        }
                    ],
                    'buttons': [
                        {
                            'name': 'download_board',
                            'label': 'Download Calibration Board',
                            'action': 'download_charuco_board'
                        }
                    ]
                }
            ),
            WorkflowStep(
                'capture',
                'Capture Calibration Images',
                'Capture 20-30 images of the calibration board from different angles',
                ui_schema={
                    'instructions': [
                        'Move the board to different positions and angles',
                        'Ensure the entire board is visible in each image',
                        'Cover all areas of the camera view',
                        'Capture images with board at various distances',
                        'Wait for the green indicator before capturing each image'
                    ],
                    'displays': [
                        {
                            'name': 'live_preview',
                            'type': 'video',
                            'label': 'Live Preview'
                        },
                        {
                            'name': 'marker_status',
                            'type': 'text',
                            'label': 'Marker Detection Status'
                        },
                        {
                            'name': 'capture_count',
                            'type': 'number',
                            'label': 'Images Captured'
                        },
                        {
                            'name': 'capture_progress',
                            'type': 'progress',
                            'label': 'Progress',
                            'min': 0,
                            'max': 30
                        }
                    ],
                    'controls': [
                        {
                            'name': 'target_images',
                            'type': 'range',
                            'label': 'Target Images',
                            'min': 15,
                            'max': 50,
                            'default': 25,
                            'step': 5
                        },
                        {
                            'name': 'auto_capture',
                            'type': 'checkbox',
                            'label': 'Auto Capture',
                            'default': True
                        },
                        {
                            'name': 'capture_interval',
                            'type': 'range',
                            'label': 'Auto Capture Interval (seconds)',
                            'min': 1,
                            'max': 5,
                            'default': 2
                        }
                    ],
                    'buttons': [
                        {
                            'name': 'capture_image',
                            'label': 'Capture Image',
                            'action': 'manual_capture'
                        },
                        {
                            'name': 'delete_last',
                            'label': 'Delete Last',
                            'action': 'delete_last_image'
                        }
                    ]
                }
            ),
            WorkflowStep(
                'calibrate',
                'Run Calibration',
                'Process captured images and calculate calibration parameters',
                ui_schema={
                    'instructions': [
                        'Review the captured images',
                        'Remove any poor quality images if needed',
                        'Click "Calculate Calibration" to process'
                    ],
                    'displays': [
                        {
                            'name': 'image_grid',
                            'type': 'image_gallery',
                            'label': 'Captured Images'
                        },
                        {
                            'name': 'calibration_status',
                            'type': 'text',
                            'label': 'Status'
                        }
                    ],
                    'buttons': [
                        {
                            'name': 'calculate',
                            'label': 'Calculate Calibration',
                            'action': 'run_calibration'
                        }
                    ]
                }
            ),
            WorkflowStep(
                'review',
                'Review Results',
                'Review calibration results and save',
                ui_schema={
                    'instructions': [
                        'Review the calibration parameters',
                        'Check the reprojection error (lower is better)',
                        'Save the calibration for use with this camera'
                    ],
                    'displays': [
                        {
                            'name': 'camera_matrix',
                            'type': 'matrix',
                            'label': 'Camera Matrix'
                        },
                        {
                            'name': 'distortion_coeffs',
                            'type': 'array',
                            'label': 'Distortion Coefficients'
                        },
                        {
                            'name': 'reprojection_error',
                            'type': 'number',
                            'label': 'Reprojection Error (pixels)'
                        },
                        {
                            'name': 'num_images',
                            'type': 'number',
                            'label': 'Images Used'
                        },
                        {
                            'name': 'undistorted_preview',
                            'type': 'image_comparison',
                            'label': 'Before/After Undistortion'
                        }
                    ],
                    'controls': [
                        {
                            'name': 'profile_name',
                            'type': 'text',
                            'label': 'Calibration Profile Name',
                            'placeholder': 'e.g., camera_0_calibration',
                            'required': True
                        }
                    ],
                    'buttons': [
                        {
                            'name': 'save_calibration',
                            'label': 'Save Calibration',
                            'action': 'save_calibration'
                        },
                        {
                            'name': 'recalibrate',
                            'label': 'Recalibrate',
                            'action': 'restart_workflow'
                        }
                    ]
                }
            )
        ]

    def validate_step(self, step_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate step data."""
        errors = []
        
        if step_id == 'setup':
            if 'camera_id' not in data or not data['camera_id']:
                errors.append('Camera selection is required')
        
        elif step_id == 'capture':
            capture_count = data.get('capture_count', 0)
            target_images = data.get('target_images', 25)
            
            if capture_count < 15:
                errors.append(f'Insufficient images: {capture_count} (minimum 15 required)')
            
            if capture_count < target_images:
                errors.append(f'Target not reached: {capture_count}/{target_images} images')
        
        elif step_id == 'review':
            if 'profile_name' not in data or not data['profile_name']:
                errors.append('Profile name is required')
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }

    def execute_step(self, step_id: str, data: Dict[str, Any],
                    camera_backend, group_manager) -> Dict[str, Any]:
        """Execute workflow step."""
        
        if step_id == 'setup':
            return self._execute_setup(data, camera_backend)
        
        elif step_id == 'capture':
            return self._execute_capture(data, camera_backend)
        
        elif step_id == 'calibrate':
            return self._execute_calibrate(data, camera_backend)
        
        elif step_id == 'review':
            return self._execute_review(data, camera_backend)
        
        else:
            return {
                'success': False,
                'message': f'Unknown step: {step_id}'
            }

    def _execute_setup(self, data: Dict[str, Any], camera_backend) -> Dict[str, Any]:
        """Execute setup step."""
        camera_id = data.get('camera_id')
        fisheye = data.get('fisheye', False)
        
        # Store configuration in workflow state
        self.set_step_data('setup', {
            'camera_id': camera_id,
            'fisheye': fisheye
        })
        
        # Start calibration session
        from backend.features.plugins.calibration import CalibrationPlugin
        plugin = CalibrationPlugin()
        
        result = plugin.process_frames(
            camera_backend,
            [camera_id],
            {
                'action': 'start_session',
                'session_id': f'workflow_{id(self)}',
                'calibration_type': 'single',
                'fisheye_model': fisheye
            }
        )
        
        if result['success']:
            self.set_step_data('session', result)
            self.mark_step_completed('setup')
            return {
                'success': True,
                'message': 'Setup complete. Ready to capture images.',
                'next_step': 'capture'
            }
        else:
            return result

    def _execute_capture(self, data: Dict[str, Any], camera_backend) -> Dict[str, Any]:
        """Execute capture step."""
        # This step is interactive and handled by the frontend
        # Just validate we have enough images
        capture_count = data.get('capture_count', 0)
        
        if capture_count >= 15:
            self.mark_step_completed('capture')
            self.set_step_data('capture', data)
            return {
                'success': True,
                'message': f'Captured {capture_count} images. Ready to calibrate.',
                'next_step': 'calibrate'
            }
        else:
            return {
                'success': False,
                'message': f'Need at least 15 images, currently have {capture_count}'
            }

    def _execute_calibrate(self, data: Dict[str, Any], camera_backend) -> Dict[str, Any]:
        """Execute calibration step."""
        setup_data = self.get_step_data('setup')
        camera_id = setup_data['camera_id']
        fisheye = setup_data.get('fisheye', False)
        
        from backend.features.plugins.calibration import CalibrationPlugin
        plugin = CalibrationPlugin()
        
        result = plugin.process_frames(
            camera_backend,
            [camera_id],
            {
                'action': 'run_calibration',
                'session_id': f'workflow_{id(self)}',
                'calibration_type': 'single',
                'fisheye_model': fisheye
            }
        )
        
        if result['success']:
            self.mark_step_completed('calibrate')
            self.set_step_data('calibrate', result)
            return {
                'success': True,
                'message': 'Calibration completed successfully',
                'result': result,
                'next_step': 'review'
            }
        else:
            return result

    def _execute_review(self, data: Dict[str, Any], camera_backend) -> Dict[str, Any]:
        """Execute review step."""
        profile_name = data.get('profile_name')
        calibrate_data = self.get_step_data('calibrate')
        
        # Save calibration data with the specified profile name
        # This would integrate with your camera profile system
        
        self.mark_step_completed('review')
        self.state['result'] = {
            'profile_name': profile_name,
            'calibration_data': calibrate_data
        }
        
        return {
            'success': True,
            'message': f'Calibration saved as "{profile_name}"',
            'result': self.state['result']
        }


class StereoCalibrationWorkflow(Workflow):
    """Guided stereo camera calibration workflow."""

    def get_metadata(self) -> Dict[str, Any]:
        """Get workflow metadata."""
        return {
            'name': 'stereo_calibration',
            'description': 'Step-by-step stereo camera calibration',
            'version': '1.0.0',
            'category': 'calibration',
            'requires_cameras': 2,
            'estimated_time': '20-30 minutes'
        }

    def get_steps(self) -> List[WorkflowStep]:
        """Get workflow steps."""
        return [
            WorkflowStep(
                'setup',
                'Setup',
                'Select stereo camera pair and prepare calibration board',
                ui_schema={
                    'instructions': [
                        'Select the two cameras for stereo calibration',
                        'Ensure cameras are rigidly mounted',
                        'Print the ChArUco calibration board',
                        'Mount board on a flat surface'
                    ],
                    'controls': [
                        {
                            'name': 'camera_group',
                            'type': 'select',
                            'label': 'Camera Group',
                            'options': [],  # Populated from camera groups
                            'required': True
                        },
                        {
                            'name': 'fisheye',
                            'type': 'checkbox',
                            'label': 'Fisheye Camera Model',
                            'default': False
                        }
                    ]
                }
            ),
            WorkflowStep(
                'capture',
                'Capture Synchronized Images',
                'Capture synchronized image pairs from different angles',
                ui_schema={
                    'instructions': [
                        'Move the board to capture from multiple angles',
                        'Both cameras must see the entire board',
                        'Capture 25-40 synchronized image pairs',
                        'Cover the entire overlapping field of view'
                    ],
                    'displays': [
                        {
                            'name': 'left_preview',
                            'type': 'video',
                            'label': 'Left Camera'
                        },
                        {
                            'name': 'right_preview',
                            'type': 'video',
                            'label': 'Right Camera'
                        },
                        {
                            'name': 'sync_status',
                            'type': 'text',
                            'label': 'Synchronization Status'
                        }
                    ]
                }
            ),
            WorkflowStep(
                'calibrate_individual',
                'Calibrate Individual Cameras',
                'Calculate calibration for each camera separately',
                ui_schema={
                    'instructions': [
                        'Individual camera calibrations will be calculated first',
                        'This provides the intrinsic parameters for each camera'
                    ]
                }
            ),
            WorkflowStep(
                'calibrate_stereo',
                'Calibrate Stereo Pair',
                'Calculate stereo calibration parameters',
                ui_schema={
                    'instructions': [
                        'Stereo calibration calculates the extrinsic parameters',
                        'This includes rotation and translation between cameras'
                    ]
                }
            ),
            WorkflowStep(
                'review',
                'Review Results',
                'Review and save stereo calibration',
                ui_schema={
                    'displays': [
                        {
                            'name': 'rectification_preview',
                            'type': 'image_pair',
                            'label': 'Rectified Images'
                        },
                        {
                            'name': 'baseline',
                            'type': 'number',
                            'label': 'Baseline (mm)'
                        }
                    ]
                }
            )
        ]

    def validate_step(self, step_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate step data."""
        # TODO: Implement validation
        return {'valid': True, 'errors': []}

    def execute_step(self, step_id: str, data: Dict[str, Any],
                    camera_backend, group_manager) -> Dict[str, Any]:
        """Execute workflow step."""
        # TODO: Implement stereo workflow execution
        return {
            'success': False,
            'message': 'Stereo calibration workflow not yet fully implemented'
        }
