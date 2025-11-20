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
        
        self.config = None
    
    def start(self, width: int = 1920, height: int = 1080, fps: int = 60,
              preview_width: int = 640, preview_height: int = 360, 
              preview_quality: int = 85) -> bool:
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
            return True
        
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
            pipeline_str = (
                # Camera 1
                f'nvarguscamerasrc sensor-id={self.camera1_id} name=src1 '
                f'wbmode=1 aeantibanding=1 ! '
                f'video/x-raw(memory:NVMM),width={width},height={height},framerate={fps}/1 ! '
                f'tee name=t1 '
                
                # Camera 1 full-res branch (rotate 180 degrees)
                f't1. ! queue max-size-buffers=1 leaky=downstream ! '
                f'nvvidconv flip-method=2 ! video/x-raw ! videoconvert ! video/x-raw,format=BGR ! '
                f'appsink name=full_sink1 emit-signals=true max-buffers=1 drop=true '
                
                # Camera 1 preview branch (rotate 180 degrees, with label)
                f't1. ! queue max-size-buffers=1 leaky=downstream ! '
                f'nvvidconv flip-method=2 ! video/x-raw,width={preview_width},height={preview_height} ! '
                f'videoconvert ! '
                f'textoverlay text="Camera {self.camera1_id}" valignment=top halignment=left '
                f'font-desc="Sans Bold 16" color=0xFF00FF00 ! '
                f'queue ! comp.sink_0 '
                
                # Camera 2
                f'nvarguscamerasrc sensor-id={self.camera2_id} name=src2 '
                f'wbmode=1 aeantibanding=1 ! '
                f'video/x-raw(memory:NVMM),width={width},height={height},framerate={fps}/1 ! '
                f'tee name=t2 '
                
                # Camera 2 full-res branch (rotate 180 degrees)
                f't2. ! queue max-size-buffers=1 leaky=downstream ! '
                f'nvvidconv flip-method=2 ! video/x-raw ! videoconvert ! video/x-raw,format=BGR ! '
                f'appsink name=full_sink2 emit-signals=true max-buffers=1 drop=true '
                
                # Camera 2 preview branch (rotate 180 degrees, with label)
                f't2. ! queue max-size-buffers=1 leaky=downstream ! '
                f'nvvidconv flip-method=2 ! video/x-raw,width={preview_width},height={preview_height} ! '
                f'videoconvert ! '
                f'textoverlay text="Camera {self.camera2_id}" valignment=top halignment=left '
                f'font-desc="Sans Bold 16" color=0xFF00FF00 ! '
                f'queue ! comp.sink_1 '
                
                # Compositor: side-by-side layout
                f'compositor name=comp '
                f'sink_0::xpos=0 sink_0::ypos=0 '
                f'sink_1::xpos={preview_width} sink_1::ypos=0 ! '
                f'video/x-raw,width={preview_width*2},height={preview_height} ! '
                f'jpegenc quality={preview_quality} ! '
                f'appsink name=composite_preview emit-signals=true max-buffers=1 drop=true'
            )
            
            self.pipeline = Gst.parse_launch(pipeline_str)
            
            # Connect callbacks for full-res frames
            full_sink1 = self.pipeline.get_by_name('full_sink1')
            full_sink1.connect('new-sample', self._on_full_frame, self.camera1_id)
            print(f"[SyncPair] Connected full_sink1 to camera {self.camera1_id}")
            
            full_sink2 = self.pipeline.get_by_name('full_sink2')
            full_sink2.connect('new-sample', self._on_full_frame, self.camera2_id)
            print(f"[SyncPair] Connected full_sink2 to camera {self.camera2_id}")
            
            # Connect callback for composite preview
            composite_preview = self.pipeline.get_by_name('composite_preview')
            composite_preview.connect('new-sample', self._on_composite_preview)
            print(f"[SyncPair] Connected composite preview sink")
            
            # Store composite preview
            self.composite_preview_frame = None
            self.composite_preview_lock = threading.Lock()
            
            # Start pipeline
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                print("Failed to start synchronized camera pipeline")
                return False
            
            self.is_running = True
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
    
    def _on_full_frame(self, sink, camera_id):
        """Callback for new full-resolution frame."""
        sample = sink.emit('pull-sample')
        if sample:
            buffer = sample.get_buffer()
            caps = sample.get_caps()
            
            # Get frame dimensions
            structure = caps.get_structure(0)
            width = structure.get_value('width')
            height = structure.get_value('height')
            
            # Map buffer to numpy array
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if success:
                frame = np.ndarray(
                    shape=(height, width, 3),
                    dtype=np.uint8,
                    buffer=map_info.data
                )
                
                # Store frame with timestamp
                with self.frames[camera_id]['full_frame_lock']:
                    self.frames[camera_id]['full_frame'] = frame.copy()
                    self.frames[camera_id]['timestamp'] = time.time()
                
                buffer.unmap(map_info)
        
        return Gst.FlowReturn.OK
    
    def _on_composite_preview(self, sink):
        """Callback for composite preview frame (JPEG)."""
        sample = sink.emit('pull-sample')
        if sample:
            buffer = sample.get_buffer()
            success, map_info = buffer.map(Gst.MapFlags.READ)
            
            if success:
                jpeg_data = bytes(map_info.data)
                
                with self.composite_preview_lock:
                    self.composite_preview_frame = jpeg_data
                
                buffer.unmap(map_info)
        
        return Gst.FlowReturn.OK
    
    def get_full_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """Get latest full-resolution frame from specified camera."""
        if camera_id not in self.frames:
            return None
        
        with self.frames[camera_id]['full_frame_lock']:
            return self.frames[camera_id]['full_frame']
    
    def get_full_frame_with_timestamp(self, camera_id: str) -> Optional[Tuple[np.ndarray, float]]:
        """Get latest full-resolution frame with timestamp."""
        if camera_id not in self.frames:
            return None
        
        with self.frames[camera_id]['full_frame_lock']:
            frame = self.frames[camera_id]['full_frame']
            timestamp = self.frames[camera_id]['timestamp']
            if frame is not None and timestamp is not None:
                return (frame.copy(), timestamp)
            return None
    
    def get_synchronized_frames(self, max_time_diff: float = 0.016) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Get synchronized frames from both cameras.
        
        Args:
            max_time_diff: Maximum allowed time difference in seconds (default 16ms = 1 frame at 60fps)
        
        Returns:
            Tuple of (frame1, frame2, avg_timestamp) or None if not synchronized
        """
        result1 = self.get_full_frame_with_timestamp(self.camera1_id)
        result2 = self.get_full_frame_with_timestamp(self.camera2_id)
        
        if result1 is None or result2 is None:
            return None
        
        frame1, ts1 = result1
        frame2, ts2 = result2
        
        # Check if timestamps are close enough
        time_diff = abs(ts1 - ts2)
        if time_diff > max_time_diff:
            return None
        
        avg_timestamp = (ts1 + ts2) / 2
        return (frame1, frame2, avg_timestamp)
    
    def get_preview_frame(self, camera_id: Optional[str] = None) -> Optional[bytes]:
        """Get latest composite preview frame (JPEG) showing both cameras side-by-side.
        
        Note: camera_id parameter is ignored - composite always shows both cameras.
        Kept for API compatibility.
        """
        with self.composite_preview_lock:
            return self.composite_preview_frame
    
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
