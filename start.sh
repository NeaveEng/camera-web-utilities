#!/bin/bash
# Startup script for Camera Streaming Platform

echo "🎥 Camera Streaming Platform"
echo "=============================="
echo ""

# Check for required system packages
echo "Checking system dependencies..."
MISSING_DEPS=""

if ! python3 -c "import gi" 2>/dev/null; then
    MISSING_DEPS="$MISSING_DEPS python3-gi"
fi

if ! python3 -c "import gi; gi.require_version('Gst', '1.0'); from gi.repository import Gst" 2>/dev/null; then
    MISSING_DEPS="$MISSING_DEPS python3-gst-1.0 gstreamer1.0-tools"
fi

if [ -n "$MISSING_DEPS" ]; then
    echo "⚠️  Missing system dependencies. Please install them first:"
    echo ""
    echo "    sudo apt-get update"
    echo "    sudo apt-get install -y $MISSING_DEPS"
    echo ""
    read -p "Press Enter to continue anyway or Ctrl+C to exit..."
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv --system-site-packages
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt --quiet

# Create necessary directories
mkdir -p data/camera_groups
mkdir -p data/workflows/state
mkdir -p backend/camera/profiles/jetson
mkdir -p backend/features/plugins
mkdir -p backend/workflows/templates

echo ""
echo "Starting server..."
echo "Open your browser to:"
echo "  - Local:   http://localhost:5000"
echo "  - Network: http://$(hostname -I | awk '{print $1}'):5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the application
python -m backend.app
