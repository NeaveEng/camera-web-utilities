"""
Camera grouping system for managing multi-camera setups.

This module provides functionality to group cameras together for stereo,
tri-camera, or custom multi-camera configurations.
"""

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


class CameraGroup:
    """Represents a group of cameras (e.g., stereo pair, tri-cam setup)."""

    def __init__(self, group_id: str, name: str, camera_ids: List[str], 
                 group_type: str = "custom", metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a camera group.

        Args:
            group_id: Unique identifier for the group
            name: Human-readable group name
            camera_ids: List of camera IDs in this group
            group_type: Type of group ("stereo", "tri-cam", "custom")
            metadata: Optional metadata (calibration data, etc.)
        """
        self.group_id = group_id
        self.name = name
        self.camera_ids = camera_ids
        self.group_type = group_type
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert group to dictionary."""
        return {
            'group_id': self.group_id,
            'name': self.name,
            'camera_ids': self.camera_ids,
            'group_type': self.group_type,
            'metadata': self.metadata,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CameraGroup':
        """Create group from dictionary."""
        group = cls(
            group_id=data['group_id'],
            name=data['name'],
            camera_ids=data['camera_ids'],
            group_type=data.get('group_type', 'custom'),
            metadata=data.get('metadata', {})
        )
        group.created_at = data.get('created_at', group.created_at)
        group.updated_at = data.get('updated_at', group.updated_at)
        return group

    def update_metadata(self, metadata: Dict[str, Any]):
        """Update group metadata."""
        self.metadata.update(metadata)
        self.updated_at = datetime.now().isoformat()


class CameraGroupManager:
    """Manages camera groups with persistence."""

    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize the group manager.

        Args:
            storage_dir: Directory for storing group configurations
        """
        self.storage_dir = storage_dir or Path('data/camera_groups')
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.groups: Dict[str, CameraGroup] = {}
        self.lock = threading.Lock()
        
        # Load existing groups
        self._load_groups()

    def _load_groups(self):
        """Load groups from storage."""
        for group_file in self.storage_dir.glob('*.json'):
            try:
                with open(group_file, 'r') as f:
                    data = json.load(f)
                    group = CameraGroup.from_dict(data)
                    self.groups[group.group_id] = group
            except Exception as e:
                print(f"Error loading group {group_file}: {e}")

    def _save_group(self, group: CameraGroup):
        """Save a group to storage."""
        group_file = self.storage_dir / f'{group.group_id}.json'
        try:
            with open(group_file, 'w') as f:
                json.dump(group.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Error saving group {group.group_id}: {e}")
            raise

    def create_group(self, name: str, camera_ids: List[str], 
                    group_type: str = "custom", metadata: Optional[Dict[str, Any]] = None) -> CameraGroup:
        """
        Create a new camera group.

        Args:
            name: Human-readable group name
            camera_ids: List of camera IDs
            group_type: Type of group
            metadata: Optional metadata

        Returns:
            Created CameraGroup instance
        """
        with self.lock:
            # Generate unique ID
            group_id = f"group_{len(self.groups)}_{int(datetime.now().timestamp())}"
            
            group = CameraGroup(group_id, name, camera_ids, group_type, metadata)
            self.groups[group_id] = group
            self._save_group(group)
            
            return group

    def get_group(self, group_id: str) -> Optional[CameraGroup]:
        """Get a group by ID."""
        return self.groups.get(group_id)

    def list_groups(self) -> List[CameraGroup]:
        """List all groups."""
        return list(self.groups.values())

    def update_group(self, group_id: str, name: Optional[str] = None, 
                    camera_ids: Optional[List[str]] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update a group's properties.

        Args:
            group_id: Group ID to update
            name: New name (optional)
            camera_ids: New camera IDs (optional)
            metadata: Metadata to merge (optional)

        Returns:
            True if updated successfully
        """
        with self.lock:
            group = self.groups.get(group_id)
            if not group:
                return False
            
            if name:
                group.name = name
            if camera_ids is not None:
                group.camera_ids = camera_ids
            if metadata:
                group.update_metadata(metadata)
            
            group.updated_at = datetime.now().isoformat()
            self._save_group(group)
            return True

    def delete_group(self, group_id: str) -> bool:
        """Delete a group."""
        with self.lock:
            if group_id not in self.groups:
                return False
            
            # Remove from memory
            del self.groups[group_id]
            
            # Remove from storage
            group_file = self.storage_dir / f'{group_id}.json'
            if group_file.exists():
                group_file.unlink()
            
            return True

    def get_groups_for_camera(self, camera_id: str) -> List[CameraGroup]:
        """Get all groups containing a specific camera."""
        return [
            group for group in self.groups.values()
            if camera_id in group.camera_ids
        ]

    def save_calibration(self, group_id: str, calibration_data: Dict[str, Any]) -> bool:
        """
        Save calibration data for a camera group.

        Args:
            group_id: Group ID
            calibration_data: Calibration data to save

        Returns:
            True if saved successfully
        """
        with self.lock:
            group = self.groups.get(group_id)
            if not group:
                return False
            
            group.metadata['calibration'] = calibration_data
            group.metadata['calibration_date'] = datetime.now().isoformat()
            group.updated_at = datetime.now().isoformat()
            
            self._save_group(group)
            return True

    def get_calibration(self, group_id: str) -> Optional[Dict[str, Any]]:
        """Get calibration data for a camera group."""
        group = self.groups.get(group_id)
        if group and 'calibration' in group.metadata:
            return group.metadata['calibration']
        return None
