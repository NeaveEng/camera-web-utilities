"""
Feature plugin manager.

Discovers and manages feature plugins, handles frame routing.
"""

import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Any

from .base import FeaturePlugin


class FeatureManager:
    """Manages feature plugin discovery, loading, and execution."""

    def __init__(self, plugin_dir: Optional[Path] = None):
        """
        Initialize the feature manager.

        Args:
            plugin_dir: Directory containing plugin modules
        """
        self.plugin_dir = plugin_dir or (Path(__file__).parent / 'plugins')
        self.plugin_dir.mkdir(parents=True, exist_ok=True)
        
        self.plugins: Dict[str, FeaturePlugin] = {}
        self._discover_plugins()

    def _discover_plugins(self):
        """Discover and load plugins from plugin directory."""
        # Create __init__.py if it doesn't exist
        init_file = self.plugin_dir / '__init__.py'
        if not init_file.exists():
            init_file.write_text('"""Feature plugins."""\n')
        
        # Scan for Python files
        for plugin_file in self.plugin_dir.glob('*.py'):
            if plugin_file.name.startswith('_'):
                continue
            
            try:
                # Import the module
                module_name = f'backend.features.plugins.{plugin_file.stem}'
                module = importlib.import_module(module_name)
                
                # Find FeaturePlugin subclasses
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, FeaturePlugin) and obj is not FeaturePlugin:
                        # Instantiate plugin
                        plugin = obj()
                        metadata = plugin.get_metadata()
                        plugin_name = metadata.get('name', name)
                        
                        # Initialize plugin
                        plugin.initialize()
                        
                        self.plugins[plugin_name] = plugin
                        print(f"Loaded plugin: {plugin_name}")
                        
            except Exception as e:
                import traceback
                print(f"Error loading plugin {plugin_file}: {e}")
                traceback.print_exc()

    def list_plugins(self) -> List[Dict[str, Any]]:
        """
        List all available plugins.

        Returns:
            List of plugin metadata dictionaries
        """
        return [
            {
                **plugin.get_metadata(),
                'ui_schema': plugin.get_ui_schema()
            }
            for plugin in self.plugins.values()
        ]

    def get_plugin(self, plugin_name: str) -> Optional[FeaturePlugin]:
        """Get a plugin by name."""
        return self.plugins.get(plugin_name)

    def process_with_plugin(self, plugin_name: str, camera_backend, 
                           camera_ids: List[str], params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process frames using a specific plugin.

        Args:
            plugin_name: Name of the plugin
            camera_backend: CameraBackend instance
            camera_ids: List of camera IDs
            params: Optional parameters

        Returns:
            Processing result dictionary
        """
        plugin = self.plugins.get(plugin_name)
        if not plugin:
            return {
                'success': False,
                'message': f'Plugin {plugin_name} not found'
            }
        
        # Validate camera count
        required = plugin.required_cameras()
        if required > 0 and len(camera_ids) != required:
            return {
                'success': False,
                'message': f'Plugin requires {required} cameras, got {len(camera_ids)}'
            }
        
        # Process frames
        try:
            return plugin.process_frames(camera_backend, camera_ids, params)
        except Exception as e:
            return {
                'success': False,
                'message': f'Plugin processing error: {str(e)}'
            }

    def process_group_with_plugin(self, plugin_name: str, camera_backend, 
                                 group, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a camera group using a specific plugin.

        Args:
            plugin_name: Name of the plugin
            camera_backend: CameraBackend instance
            group: CameraGroup instance
            params: Optional parameters

        Returns:
            Processing result dictionary
        """
        plugin = self.plugins.get(plugin_name)
        if not plugin:
            return {
                'success': False,
                'message': f'Plugin {plugin_name} not found'
            }
        
        if not plugin.supports_groups():
            return {
                'success': False,
                'message': f'Plugin {plugin_name} does not support camera groups'
            }
        
        try:
            return plugin.process_group(camera_backend, group, params)
        except Exception as e:
            return {
                'success': False,
                'message': f'Plugin processing error: {str(e)}'
            }

    def register_plugin_routes(self, app):
        """
        Register all plugin routes with the application.

        Args:
            app: Flask/FastAPI application instance
        """
        for plugin_name, plugin in self.plugins.items():
            try:
                plugin.register_routes(app)
            except Exception as e:
                print(f"Error registering routes for {plugin_name}: {e}")

    def get_workflows(self) -> Dict[str, List[Any]]:
        """
        Get all workflows from all plugins.

        Returns:
            Dictionary mapping plugin names to their workflows
        """
        workflows = {}
        for plugin_name, plugin in self.plugins.items():
            plugin_workflows = plugin.get_workflows()
            if plugin_workflows:
                workflows[plugin_name] = plugin_workflows
        return workflows

    def cleanup(self):
        """Cleanup all plugins."""
        for plugin in self.plugins.values():
            try:
                plugin.cleanup()
            except Exception as e:
                print(f"Error cleaning up plugin: {e}")
