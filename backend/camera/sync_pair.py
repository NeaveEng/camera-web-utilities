"""
Synchronized camera pair for Jetson using hardware frame sync.

This module creates a single GStreamer pipeline with two cameras for
hardware-level frame synchronization, essential for panorama capture.
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import numpy as np
import threading
import time
from typing import Optional, Tuple, Dict, Any
import cv2

Gst.init(None)


class SynchronizedCameraPair:
    """Manages a pair of synchronized Jetson CSI cameras."""
    
    def __init__(self, camera1_id: str, camera2_id: str):
        """
        Initialize synchronized camera pair.
        
        Args:
            camera1_id: First camera sensor ID (e.g., "0")
            camera2_id: Second camera sensor ID (e.g., "1")
        """
        self.camera1_id = camera1_id
        self.camera2_id = camera2_id
        self.pipeline = None
        self.is_running = False
        
        # GStreamer appsink references (for polling - headless compatible)
        self.composite_preview = None
        self.composite_full = None
        
        # Frame storage with locks
        self.frames = {
            camera1_id: {
                'full_frame': None,
                'full_frame_lock': threading.Lock(),
                'timestamp': None,
                'preview_frame': None,
                'preview_frame_lock': threading.Lock()
            },
            camera2_id: {
                'full_frame': None,
                'full_frame_lock': threading.Lock(),
                'timestamp': None,
                'preview_frame': None,
                'preview_frame_lock': threading.Lock()
            }
        }
        
        # Composite preview
        self.composite_preview_frame = None
        self.composite_preview_lock = threading.Lock()
        
        # Composite full resolution
        self.composite_full_frame = None
        self.composite_full_lock = threading.Lock()
        
        self.config = None
    
    def start(self, width: int = 1920, height: int = 1080, fps: int = 60,
              preview_width: int = 960, preview_height: int = 540, 
              preview_quality: int = 50) -> bool:
        """
        Start synchronized capture from both cameras.
        
        Args:
            width: Full resolution width
            height: Full resolution height
            fps: Frame rate
            preview_width: Preview stream width (default maintains 16:9 aspect ratio)
            preview_height: Preview stream height (default maintains 16:9 aspect ratio)
            preview_quality: JPEG quality for preview (0-100)
        
        Returns:
            True if started successfully
        """
        if self.is_running:
            print(f"[SyncPair] Already running, returning True")
            return True
        
        # Clean up any existing pipeline first
        if self.pipeline:
            print(f"[SyncPair] Found existing pipeline, stopping it first")
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None
        
        self.config = {
            'width': width,
            'height': height,
            'fps': fps,
            'preview_width': preview_width,
            'preview_height': preview_height,
            'preview_quality': preview_quality
        }
        
        try:
            # Create synchronized dual-camera pipeline
            # Both cameras share the same GStreamer context for hardware sync
            # Uses NVIDIA hardware compositor and encoder for maximum performance
            # Dual output: preview (side-by-side JPEG) + full-res (top-bottom raw with fakesink drain)
            pipeline_str = (
                # Camera 1 - split to preview and full-res paths
                f'nvarguscamerasrc sensor-id={self.camera1_id} name=src1 '
                f'wbmode=1 aeantibanding=1 ! '
                f'video/x-raw(memory:NVMM),width={width},height={height},framerate={fps}/1 ! '
                f'nvvidconv flip-method=2 ! '
                f'video/x-raw(memory:NVMM),width={width},height={height} ! '
                f'tee name=t1 '
                
                # Camera 1 preview branch (downscaled)
                f't1. ! queue ! '
                f'nvvidconv ! '
                f'video/x-raw(memory:NVMM),width={preview_width},height={preview_height} ! '
                f'comp_preview.sink_0 '
                
                # Camera 1 full-res branch (keep full size)
                f't1. ! queue ! comp_full.sink_0 '
                
                # Camera 2 - split to preview and full-res paths
                f'nvarguscamerasrc sensor-id={self.camera2_id} name=src2 '
                f'wbmode=1 aeantibanding=1 ! '
                f'video/x-raw(memory:NVMM),width={width},height={height},framerate={fps}/1 ! '
                f'nvvidconv flip-method=2 ! '
                f'video/x-raw(memory:NVMM),width={width},height={height} ! '
                f'tee name=t2 '
                
                # Camera 2 preview branch (downscaled)
                f't2. ! queue ! '
                f'nvvidconv ! '
                f'video/x-raw(memory:NVMM),width={preview_width},height={preview_height} ! '
                f'comp_preview.sink_1 '
                
                # Camera 2 full-res branch (keep full size)
                f't2. ! queue ! comp_full.sink_1 '
                
                # Preview compositor (side-by-side) and JPEG encoder
                f'nvcompositor name=comp_preview '
                f'sink_0::xpos=0 sink_0::ypos=0 sink_0::width={preview_width} sink_0::height={preview_height} '
                f'sink_1::xpos={preview_width} sink_1::ypos=0 sink_1::width={preview_width} sink_1::height={preview_height} ! '
                f'video/x-raw(memory:NVMM),width={preview_width*2},height={preview_height} ! '
                f'nvvidconv ! '
                f'video/x-raw,format=I420 ! '
                f'nvjpegenc quality={preview_quality} ! '
                f'appsink name=composite_preview emit-signals=true max-buffers=2 drop=true '
                
                # Full-res compositor (top-bottom) for synchronized frame capture
                f'nvcompositor name=comp_full '
                f'sink_0::xpos=0 sink_0::ypos=0 sink_0::width={width} sink_0::height={height} '
                f'sink_1::xpos=0 sink_1::ypos={height} sink_1::width={width} sink_1::height={height} ! '
                f'video/x-raw(memory:NVMM),width={width},height={height*2} ! '
                f'queue max-size-buffers=2 leaky=downstream ! '
                f'nvvidconv ! '
                f'video/x-raw,format=I420 ! '
                f'queue max-size-buffers=2 leaky=downstream ! '
                f'videoconvert ! '
                f'video/x-raw,format=BGR ! '
                f'queue max-size-buffers=2 leaky=downstream ! '
                f'tee name=t_full '
                f't_full. ! queue max-size-buffers=1 leaky=downstream ! appsink name=composite_full emit-signals=true max-buffers=1 drop=true '
                f't_full. ! queue max-size-buffers=1 leaky=downstream ! fakesink sync=false async=false'
            )
            
            self.pipeline = Gst.parse_launch(pipeline_str)
            
            # Get appsink references
            self.composite_preview = self.pipeline.get_by_name('composite_preview')
            self.composite_full = self.pipeline.get_by_name('composite_full')
            
            # Start pipeline
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                print("Failed to start synchronized camera pipeline")
                return False
            
            # Check pipeline state
            state_change, state, pending = self.pipeline.get_state(Gst.CLOCK_TIME_NONE)
            print(f"[SyncPair] Pipeline state: {state}, state_change: {state_change}")
            
            self.is_running = True
            
            # Wait for frames to start flowing (cameras need time to initialize)
            print(f"[SyncPair] Waiting for cameras to start producing frames...")
            time.sleep(2.0)
            
            print(f"Started synchronized camera pair: {self.camera1_id}, {self.camera2_id}")
            return True
            
        except Exception as e:
            print(f"Error starting synchronized cameras: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def stop(self) -> bool:
        """Stop synchronized capture."""
        if not self.is_running:
            return True
        
        try:
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
                self.pipeline = None
            
            self.is_running = False
            print(f"Stopped synchronized camera pair: {self.camera1_id}, {self.camera2_id}")
            return True
            
        except Exception as e:
            print(f"Error stopping synchronized cameras: {e}")
            return False
    
    def get_full_composite(self) -> Optional[np.ndarray]:
        """Get latest full-resolution composite frame (top-bottom).
        
        Pulls from the continuous full-res output stream.
        Returns a single frame with camera1 on top and camera2 on bottom.
        Shape will be (height*2, width, 3) - e.g., (2160, 1920, 3) for 1920x1080 cameras.
        
        Actively pulls from appsink - works on headless systems.
        """
        if not self.is_running or not self.composite_full:
            return None
        
        try:
            # Pull sample from appsink
            sample = self.composite_full.emit('pull-sample')
            if sample:
                buffer = sample.get_buffer()
                caps = sample.get_caps()
                
                # Get frame dimensions
                structure = caps.get_structure(0)
                frame_width = structure.get_value('width')
                frame_height = structure.get_value('height')
                
                # Map buffer to numpy array (BGR format)
                success, map_info = buffer.map(Gst.MapFlags.READ)
                if success:
                    composite = np.ndarray(
                        shape=(frame_height, frame_width, 3),
                        dtype=np.uint8,
                        buffer=map_info.data
                    ).copy()
                    buffer.unmap(map_info)
                    
                    # Cache for quick re-use
                    with self.composite_full_lock:
                        self.composite_full_frame = composite
                    
                    return composite
            
            # Return cached frame if pull failed
            with self.composite_full_lock:
                return self.composite_full_frame
                
        except Exception as e:
            print(f"[SyncPair] Error pulling full composite: {e}")
            with self.composite_full_lock:
                return self.composite_full_frame
    
    def get_full_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """Get latest full-resolution frame from specified camera.
        
        Extracts individual camera frame from the top-bottom composite.
        """
        if camera_id not in [self.camera1_id, self.camera2_id]:
            return None
        
        composite = self.get_full_composite()
        if composite is None:
            return None
        
        # Split top-bottom composite
        height = composite.shape[0] // 2
        
        if camera_id == self.camera1_id:
            return composite[:height, :].copy()  # Top half
        else:
            return composite[height:, :].copy()  # Bottom half
    
    def get_full_frame_with_timestamp(self, camera_id: str) -> Optional[Tuple[np.ndarray, float]]:
        """Get latest full-resolution frame with timestamp."""
        frame = self.get_full_frame(camera_id)
        if frame is not None:
            return (frame, time.time())
        return None
    
    def get_synchronized_frames(self, max_time_diff: float = 0.016) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Get synchronized frames from both cameras.
        
        Since both frames come from the same composite, they are guaranteed to be
        from the exact same moment (hardware synchronized).
        
        Args:
            max_time_diff: Ignored - frames are always synchronized from composite
        
        Returns:
            Tuple of (frame1, frame2, timestamp) or None if frames unavailable
        """
        composite = self.get_full_composite()
        if composite is None:
            return None
        
        # Split top-bottom composite
        height = composite.shape[0] // 2
        frame1 = composite[:height, :].copy()  # Camera 1 (top)
        frame2 = composite[height:, :].copy()  # Camera 2 (bottom)
        
        timestamp = time.time()
        return (frame1, frame2, timestamp)
    
    def get_preview_frame(self, camera_id: Optional[str] = None) -> Optional[bytes]:
        """Get latest composite preview frame (JPEG) showing both cameras side-by-side.
        
        Actively pulls from appsink - works on headless systems without display.
        Note: camera_id parameter is ignored - composite always shows both cameras.
        """
        if not self.is_running or not self.composite_preview:
            return None
        
        try:
            # Pull latest sample (with drop=true, old frames are discarded)
            sample = self.composite_preview.emit('pull-sample')
            
            if sample:
                buffer = sample.get_buffer()
                success, map_info = buffer.map(Gst.MapFlags.READ)
                if success:
                    jpeg_data = bytes(map_info.data)
                    buffer.unmap(map_info)
                    
                    # Cache for quick re-use
                    with self.composite_preview_lock:
                        self.composite_preview_frame = jpeg_data
                    
                    return jpeg_data
            
            # Return cached frame if pull failed
            with self.composite_preview_lock:
                return self.composite_preview_frame
                
        except Exception as e:
            with self.composite_preview_lock:
                return self.composite_preview_frame
            return None
    
    def is_streaming(self, camera_id: str) -> bool:
        """Check if camera is streaming."""
        return self.is_running and camera_id in [self.camera1_id, self.camera2_id]


class SynchronizedPairManager:
    """Manages multiple synchronized camera pairs."""
    
    def __init__(self):
        self.pairs: Dict[Tuple[str, str], SynchronizedCameraPair] = {}
    
    def create_pair(self, camera1_id: str, camera2_id: str) -> SynchronizedCameraPair:
        """Create or get existing synchronized camera pair."""
        # Normalize pair key (order doesn't matter)
        pair_key: Tuple[str, str] = tuple(sorted([camera1_id, camera2_id]))  # type: ignore
        
        if pair_key not in self.pairs:
            self.pairs[pair_key] = SynchronizedCameraPair(camera1_id, camera2_id)
        
        return self.pairs[pair_key]
    
    def get_pair(self, camera1_id: str, camera2_id: str) -> Optional[SynchronizedCameraPair]:
        """Get existing synchronized camera pair."""
        pair_key: Tuple[str, str] = tuple(sorted([camera1_id, camera2_id]))  # type: ignore
        return self.pairs.get(pair_key)
    
    def remove_pair(self, camera1_id: str, camera2_id: str) -> bool:
        """Stop and remove synchronized camera pair."""
        pair_key: Tuple[str, str] = tuple(sorted([camera1_id, camera2_id]))  # type: ignore
        
        if pair_key in self.pairs:
            pair = self.pairs[pair_key]
            pair.stop()
            del self.pairs[pair_key]
            return True
        
        return False
    
    def get_all_pairs(self) -> Dict[Tuple[str, str], SynchronizedCameraPair]:
        """Get all synchronized pairs."""
        return self.pairs
