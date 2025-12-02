// Camera Streaming Platform - Frontend JavaScript

const API_BASE = '';
let cameras = [];
let activeCameras = {};
let selectedCameraSlot = null;
let groups = [];
let features = [];
let workflows = [];

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    initialize();
    setupEventListeners();
});

async function initialize() {
    await loadPlatformInfo();
    await loadCameras();
    await loadGroups();
    await loadFeatures();
    await loadWorkflows();
}

function setupEventListeners() {
    // Camera refresh
    document.getElementById('refresh-cameras').addEventListener('click', loadCameras);

    // Create group
    document.getElementById('create-group').addEventListener('click', showCreateGroupModal);

    // Modal close buttons
    document.querySelectorAll('.close').forEach(el => {
        el.addEventListener('click', () => {
            el.closest('.modal').classList.remove('show');
        });
    });

    // Create group form
    document.getElementById('create-group-form').addEventListener('submit', handleCreateGroup);

    // Close modals when clicking outside
    window.addEventListener('click', (e) => {
        if (e.target.classList.contains('modal')) {
            e.target.classList.remove('show');
        }
    });
}

// ============================================================================
// Platform Info
// ============================================================================

async function loadPlatformInfo() {
    try {
        const response = await fetch(`${API_BASE}/api/cameras`);
        const data = await response.json();

        if (data.success) {
            document.getElementById('platform-name').textContent =
                `Platform: ${data.platform.charAt(0).toUpperCase() + data.platform.slice(1)}`;
        }
    } catch (error) {
        console.error('Error loading platform info:', error);
    }
}

// ============================================================================
// Camera Management
// ============================================================================

async function loadCameras() {
    try {
        const response = await fetch(`${API_BASE}/api/cameras`);
        const data = await response.json();

        if (data.success) {
            cameras = data.cameras;
            renderCameraList();
            populateCameraSelectors();

            // Restore UI state for already-streaming cameras
            await restoreStreamingCameras();
        }
    } catch (error) {
        console.error('Error loading cameras:', error);
        document.getElementById('camera-list').innerHTML =
            '<p style="color: red;">Failed to load cameras</p>';
    }
}

async function renderCameraList() {
    const list = document.getElementById('camera-list');

    if (cameras.length === 0) {
        list.innerHTML = '<p>No cameras found</p>';
        return;
    }

    // Fetch custom names for all cameras
    const customNames = {};
    for (const camera of cameras) {
        try {
            const response = await fetch(`${API_BASE}/api/cameras/${camera.id}/controls`);
            const data = await response.json();
            if (data.success && data.controls.camera_name?.current) {
                customNames[camera.id] = data.controls.camera_name.current;
            }
        } catch (e) {
            // Ignore errors, use default name
        }
    }

    list.innerHTML = cameras.map(camera => {
        const displayName = customNames[camera.id] || camera.name;
        return `
        <div class="camera-item">
            <input type="checkbox" 
                   id="camera-${camera.id}" 
                   data-camera-id="${camera.id}"
                   ${!camera.available ? 'disabled' : ''}
                   ${camera.streaming ? 'checked' : ''}>
            <label for="camera-${camera.id}">${displayName} (Port ${camera.id})</label>
        </div>
    `;
    }).join('');

    // Attach event listeners after rendering
    cameras.forEach(camera => {
        const checkbox = document.getElementById(`camera-${camera.id}`);
        if (checkbox) {
            checkbox.addEventListener('change', (e) => {
                handleCameraToggle(camera.id, e.target.checked);
            });
        }
    });
}

async function populateCameraSelectors() {
    // Get custom names for cameras by loading controls
    const customNames = {};
    for (const camera of cameras) {
        try {
            const response = await fetch(`${API_BASE}/api/cameras/${camera.id}/controls`);
            const data = await response.json();
            if (data.success && data.controls.camera_name?.current) {
                customNames[camera.id] = data.controls.camera_name.current;
            }
        } catch (e) {
            // Ignore errors, use default name
        }
    }

    // Populate dropdowns in each video slot
    for (let slot = 0; slot < 2; slot++) {
        const selector = document.getElementById(`selector-${slot}`);
        if (!selector) continue;

        // Clear existing options except the first one
        selector.innerHTML = '<option value="">Select camera...</option>';

        // Add camera options
        cameras.forEach(camera => {
            const option = document.createElement('option');
            option.value = camera.id;
            option.textContent = getCameraDisplayName(camera.id, customNames[camera.id]);
            selector.appendChild(option);
        });

        // Attach change handler
        selector.addEventListener('change', (e) => {
            handleSlotCameraChange(slot, e.target.value);
        });
    }
}

async function handleSlotCameraChange(slot, cameraId) {
    // Stop any camera currently in this slot
    const currentCamera = Object.entries(activeCameras).find(([id, info]) => info.slot === slot);
    if (currentCamera) {
        const [currentId] = currentCamera;
        await stopCamera(currentId);
        // Uncheck the checkbox
        const checkbox = document.getElementById(`camera-${currentId}`);
        if (checkbox) checkbox.checked = false;
    }

    // Start the new camera if one was selected
    if (cameraId) {
        await startCamera(cameraId, slot);
        // Check the checkbox
        const checkbox = document.getElementById(`camera-${cameraId}`);
        if (checkbox) checkbox.checked = true;

        // Update selector value
        const selector = document.getElementById(`selector-${slot}`);
        if (selector) selector.value = cameraId;
    }
}

async function restoreStreamingCameras() {
    // Restore UI for cameras that are already streaming
    const streamingCameras = cameras.filter(c => c.streaming);

    for (const camera of streamingCameras) {
        const slot = findAvailableSlot();
        if (slot !== null) {
            // Restore UI without calling start (camera is already streaming)
            const videoContainer = document.getElementById(`video-${slot}`);
            const placeholder = videoContainer.querySelector('.video-placeholder');
            const statusDot = videoContainer.querySelector('.camera-status');
            const nameSpan = videoContainer.querySelector('.camera-name');

            // Create image element for MJPEG stream
            const img = document.createElement('img');
            img.src = `${API_BASE}/api/cameras/${camera.id}/stream`;
            img.alt = `Camera ${camera.id}`;

            placeholder.replaceWith(img);
            statusDot.classList.add('active');
            nameSpan.textContent = camera.name;

            // Track active camera
            activeCameras[camera.id] = { slot, element: img };

            // Update selector dropdown
            const selector = document.getElementById(`selector-${slot}`);
            if (selector) selector.value = camera.id;

            // Load controls for this camera
            selectCameraForControls(camera.id);
        }
    }
}

async function handleCameraToggle(cameraId, checked) {
    if (checked) {
        // Find available slot
        const slot = findAvailableSlot();
        if (slot === null) {
            alert('All video slots are in use. Stop a camera first.');
            document.getElementById(`camera-${cameraId}`).checked = false;
            return;
        }

        await startCamera(cameraId, slot);
    } else {
        await stopCamera(cameraId);
    }
}

function findAvailableSlot() {
    for (let i = 0; i < 2; i++) {  // Changed from 3 to 2 slots
        if (!Object.values(activeCameras).some(c => c.slot === i)) {
            return i;
        }
    }
    return null;
}

async function startCamera(cameraId, slot) {
    try {
        // Start camera stream
        const response = await fetch(`${API_BASE}/api/cameras/${cameraId}/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                width: 1920,
                height: 1080,
                fps: 30,
                preview_width: 640,
                preview_height: 480,
                preview_quality: 70
            })
        });

        const data = await response.json();

        if (data.success) {
            // Update UI
            const videoContainer = document.getElementById(`video-${slot}`);
            const placeholder = videoContainer.querySelector('.video-placeholder');
            const statusDot = videoContainer.querySelector('.camera-status');
            const nameSpan = videoContainer.querySelector('.camera-name');

            // Create image element for MJPEG stream
            const img = document.createElement('img');
            // Add timestamp to prevent browser caching of stream
            img.src = `${API_BASE}/api/cameras/${cameraId}/stream?t=${Date.now()}`;
            img.alt = `Camera ${cameraId}`;

            placeholder.replaceWith(img);
            statusDot.classList.add('active');

            // Track active camera
            activeCameras[cameraId] = { slot, element: img };

            // Update selector dropdown
            const selector = document.getElementById(`selector-${slot}`);
            if (selector) selector.value = cameraId;

            // Load controls first to get custom name
            selectCameraForControls(cameraId);
        } else {
            alert(`Failed to start camera: ${data.message}`);
            document.getElementById(`camera-${cameraId}`).checked = false;
        }
    } catch (error) {
        console.error('Error starting camera:', error);
        alert('Error starting camera');
        document.getElementById(`camera-${cameraId}`).checked = false;
    }
}

async function stopCamera(cameraId) {
    try {
        const response = await fetch(`${API_BASE}/api/cameras/${cameraId}/stop`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            const cameraInfo = activeCameras[cameraId];
            if (cameraInfo) {
                const videoContainer = document.getElementById(`video-${cameraInfo.slot}`);
                const img = cameraInfo.element;
                const statusDot = videoContainer.querySelector('.camera-status');

                // Replace image with placeholder
                const placeholder = document.createElement('div');
                placeholder.className = 'video-placeholder';
                placeholder.innerHTML = '<p>No camera selected</p>';

                img.replaceWith(placeholder);
                statusDot.classList.remove('active');

                // Clear selector dropdown
                const selector = document.getElementById(`selector-${cameraInfo.slot}`);
                if (selector) selector.value = '';

                delete activeCameras[cameraId];

                // Clear controls for this camera
                const controlsContainer = document.getElementById(`controls-${cameraInfo.slot}`);
                if (controlsContainer) {
                    controlsContainer.innerHTML = `
                        <div class="no-camera-message">
                            <p>Select camera to view controls</p>
                        </div>
                    `;
                }
            }
        }
    } catch (error) {
        console.error('Error stopping camera:', error);
    }
}

async function selectCameraForControls(cameraId) {
    // Controls are now per-camera, so this function loads controls for the specific camera
    try {
        // Find which slot this camera is in
        const cameraInfo = activeCameras[cameraId];
        if (!cameraInfo) return;

        const slot = cameraInfo.slot;

        // Load camera controls
        const response = await fetch(`${API_BASE}/api/cameras/${cameraId}/controls`);
        const data = await response.json();

        if (data.success) {
            renderControls(slot, cameraId, data.controls, data.resolution_info);

            // Update camera name display
            updateCameraNameDisplay(slot, cameraId, data.controls.camera_name?.current);
        }
    } catch (error) {
        console.error('Error loading controls:', error);
    }
}

function updateCameraNameDisplay(slot, cameraId, customName) {
    const videoContainer = document.getElementById(`video-${slot}`);
    if (!videoContainer) return;

    const nameSpan = videoContainer.querySelector('.camera-name');
    if (!nameSpan) return;

    const camera = cameras.find(c => c.id === cameraId);
    const port = camera ? `(Port ${cameraId})` : '';
    nameSpan.textContent = customName ? `${customName} ${port}` : (camera?.name || `Camera ${cameraId}`);
}

function getCameraDisplayName(cameraId, customName) {
    const camera = cameras.find(c => c.id === cameraId);
    const port = `(Port ${cameraId})`;
    if (customName) {
        return `${customName} ${port}`;
    }
    return camera ? `${camera.name} ${port}` : `Camera ${cameraId}`;
}

function renderControls(slot, cameraId, controls, resolutionInfo) {
    const container = document.getElementById(`controls-${slot}`);
    if (!container) return;

    const cameraName = cameras.find(c => c.id === cameraId)?.name || `Camera ${cameraId}`;

    let html = `
        <h3 style="color: #667eea; margin-bottom: 15px; font-size: 1em;">Controls</h3>
    `;

    // Add resolution info if available
    if (resolutionInfo) {
        html += `
            <div style="background: #f8f9fa; padding: 12px; border-radius: 8px; margin-bottom: 15px; font-size: 0.85em;">
                <div style="margin-bottom: 6px;">
                    <strong style="color: #667eea;">Capture:</strong> 
                    <span>${resolutionInfo.capture_width}×${resolutionInfo.capture_height} @ ${resolutionInfo.capture_fps}fps</span>
                </div>
                <div>
                    <strong style="color: #667eea;">Stream:</strong> 
                    <span>${resolutionInfo.preview_width}×${resolutionInfo.preview_height}</span>
                </div>
            </div>
        `;
    }

    // Group controls by category
    const categories = {
        'Camera': ['camera_name'],
        'Exposure': ['auto_exposure', 'exposure'],
        'Gain': ['auto_gain', 'gain'],
        'White Balance': ['white_balance'],
        'Image Quality': ['saturation', 'edge_enhancement', 'noise_reduction'],
        'Transform': ['rotation'],
        'Stream': ['capture_resolution', 'stream_resolution']
    };

    for (const [category, controlNames] of Object.entries(categories)) {
        const categoryControls = controlNames
            .filter(name => controls[name])
            .map(name => ({ name, ...controls[name] }));

        if (categoryControls.length === 0) continue;

        html += `
            <div class="control-section">
                <h3 onclick="this.parentElement.classList.toggle('collapsed')">
                    ${category} ▼
                </h3>
                <div class="control-items">
                    ${categoryControls.map(control => renderControl(cameraId, control)).join('')}
                </div>
            </div>
        `;
    }

    // Add profile section
    html += renderProfileSection(cameraId);

    // Add calibration button
    html += `
        <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #ddd;">
            <button class="btn btn-primary" onclick="openCalibrationForCamera('${cameraId}')" style="width: 100%;">
                📐 Calibrate Camera
            </button>
        </div>
    `;

    // Add image processing section
    html += `
        <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #ddd;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <h3 style="color: #667eea; margin: 0; font-size: 1em;">Image Processing</h3>
                <label style="margin: 0;">
                    <input type="checkbox" id="processing-enable-${cameraId}" onchange="toggleProcessing('${cameraId}', this.checked)">
                    Enable
                </label>
            </div>
            <div id="processing-panel-${cameraId}" style="display: none;">
                <div class="control-item">
                    <label>Processing Type</label>
                    <select id="processing-type-${cameraId}" onchange="updateProcessingType('${cameraId}', this.value)">
                        <option value="undistort">Undistort (Calibration)</option>
                        <option value="perspective">Perspective Transform</option>
                        <option value="affine">Affine Transform</option>
                        <option value="rotation">Rotation</option>
                        <option value="custom_matrix">Custom Matrix</option>
                    </select>
                </div>
                
                <div id="processing-params-${cameraId}">
                    <!-- Parameters will be injected here based on type -->
                </div>
                
                <button class="btn btn-secondary" onclick="applyProcessing('${cameraId}')" style="width: 100%; margin-top: 10px;">
                    Apply Changes
                </button>
            </div>
        </div>
    `;

    container.innerHTML = html;

    // Attach event listeners
    attachControlListeners(cameraId);

    // Restore processing state from localStorage
    const processingState = JSON.parse(localStorage.getItem('processingState') || '{}');
    const savedState = processingState[cameraId];

    if (savedState) {
        // Restore enabled state
        const enableCheckbox = document.getElementById(`processing-enable-${cameraId}`);
        if (enableCheckbox && savedState.enabled) {
            enableCheckbox.checked = true;
            const panel = document.getElementById(`processing-panel-${cameraId}`);
            if (panel) {
                panel.style.display = 'block';
            }
        }

        // Restore type selection
        const typeSelector = document.getElementById(`processing-type-${cameraId}`);
        if (typeSelector && savedState.type) {
            typeSelector.value = savedState.type;
        }
    }

    // Initialize processing panel with saved or default type
    const initialType = savedState?.type || 'undistort';
    updateProcessingType(cameraId, initialType);
}

function renderControl(cameraId, control) {
    const { name, type, current, min, max, step, options, unit, disables, label, description } = control;
    const value = current !== undefined ? current : control.default;
    const defaultValue = control.default;

    // Use custom label if provided, otherwise format the name
    const controlLabel = label || formatControlName(name);

    if (type === 'bool') {
        return `
            <div class="control-item">
                <label>
                    <input type="checkbox" 
                           data-control="${name}"
                           data-default="${defaultValue}"
                           ${value ? 'checked' : ''}>
                    ${controlLabel}
                </label>
                <button class="btn-reset" data-reset="${name}" title="Reset to default">↺</button>
                ${description ? `<small style="color: #888; display: block; margin-top: 2px;">${description}</small>` : ''}
            </div>
        `;
    } else if (type === 'range') {
        const displayValue = unit ? `${value}${unit}` : value;
        return `
            <div class="control-item">
                <label>
                    ${controlLabel}
                    <span class="control-value" id="value-${cameraId}-${name}">${displayValue}</span>
                </label>
                <div style="display: flex; gap: 8px; align-items: center;">
                    <input type="range"
                           data-control="${name}"
                           data-unit="${unit || ''}"
                           data-default="${defaultValue}"
                           min="${min}"
                           max="${max}"
                           step="${step || 1}"
                           value="${value}"
                           style="flex: 1;">
                    <button class="btn-reset" data-reset="${name}" title="Reset to default">↺</button>
                </div>
                ${description ? `<small style="color: #888; display: block; margin-top: 2px;">${description}</small>` : ''}
            </div>
        `;
    } else if (type === 'menu') {
        return `
            <div class="control-item">
                <label>${controlLabel}</label>
                <div style="display: flex; gap: 8px; align-items: center;">
                    <select data-control="${name}" data-default="${defaultValue}" style="flex: 1;">
                        ${options.map(opt => `
                            <option value="${opt}" ${opt === value ? 'selected' : ''}>
                                ${opt}
                            </option>
                        `).join('')}
                    </select>
                    <button class="btn-reset" data-reset="${name}" title="Reset to default">↺</button>
                </div>
                ${description ? `<small style="color: #888; display: block; margin-top: 2px;">${description}</small>` : ''}
            </div>
        `;
    } else if (type === 'text') {
        return `
            <div class="control-item">
                <label>${controlLabel}</label>
                <div style="display: flex; gap: 8px; align-items: center;">
                    <input type="text"
                           data-control="${name}"
                           data-default="${defaultValue}"
                           value="${value || ''}"
                           placeholder="${description || ''}"
                           style="flex: 1; padding: 6px 10px; border: 1px solid #ddd; border-radius: 4px;">
                    <button class="btn-reset" data-reset="${name}" title="Reset to default">↺</button>
                </div>
            </div>
        `;
    }

    return '';
}

function formatControlName(name) {
    return name.split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

function renderProfileSection(cameraId) {
    return `
        <div class="profile-section">
            <h3>Profiles</h3>
            <select id="profile-select">
                <option value="">Select profile...</option>
            </select>
            <div class="profile-actions">
                <button class="btn btn-secondary" onclick="applyProfile('${cameraId}')">
                    Apply
                </button>
                <button class="btn btn-secondary" onclick="saveCurrentProfile('${cameraId}')">
                    Save
                </button>
            </div>
        </div>
    `;
}

function attachControlListeners(cameraId) {
    // Only select controls within this camera's control container
    const controlsContainer = document.getElementById(`controls-${cameraId}`);
    if (!controlsContainer) return;

    const controls = controlsContainer.querySelectorAll('[data-control]');

    controls.forEach(control => {
        const controlName = control.dataset.control;

        control.addEventListener('change', async (e) => {
            const value = control.type === 'checkbox' ? control.checked :
                control.type === 'range' ? parseFloat(control.value) :
                    control.value;

            await setControl(cameraId, controlName, value);

            // If camera name changed, update the display
            if (controlName === 'camera_name' && activeCameras[cameraId]) {
                const slot = activeCameras[cameraId].slot;
                updateCameraNameDisplay(slot, cameraId, value);
                // Also refresh the sidebar camera list
                renderCameraList();
            }

            // If stream or capture resolution changed, reload the stream image
            if ((controlName === 'stream_resolution' || controlName === 'capture_resolution') && activeCameras[cameraId]) {
                const img = activeCameras[cameraId].element;
                // Force reload by adding timestamp
                const baseUrl = `${API_BASE}/api/cameras/${cameraId}/stream`;
                img.src = `${baseUrl}?t=${Date.now()}`;

                // Reload controls to update resolution/fps info (wait for restart to complete)
                setTimeout(() => selectCameraForControls(cameraId), 1000);
            }

            // Update display value for range inputs
            if (control.type === 'range') {
                const valueDisplay = controlsContainer.querySelector(`#value-${cameraId}-${controlName}`);
                if (valueDisplay) {
                    // Get the unit from the control's dataset or the control definition
                    const unit = control.dataset?.unit || '';
                    valueDisplay.textContent = unit ? `${value}${unit}` : value;
                }
            }
        });
    });

    // Attach reset button listeners
    const resetButtons = controlsContainer.querySelectorAll('[data-reset]');
    resetButtons.forEach(button => {
        const controlName = button.dataset.reset;

        button.addEventListener('click', async () => {
            const control = controlsContainer.querySelector(`[data-control="${controlName}"]`);
            if (!control) return;

            const defaultValue = control.dataset.default;

            // Reset the control to default value
            if (control.type === 'checkbox') {
                control.checked = defaultValue === 'true';
            } else if (control.type === 'range') {
                control.value = defaultValue;
            } else {
                control.value = defaultValue;
            }

            // Trigger change event to update backend
            control.dispatchEvent(new Event('change'));
        });
    });

    // Load profiles
    loadProfilesForCamera(cameraId);
}

async function setControl(cameraId, controlName, value) {
    try {
        const response = await fetch(`${API_BASE}/api/cameras/${cameraId}/control/${controlName}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ value })
        });

        const data = await response.json();
        if (!data.success) {
            console.error('Failed to set control:', data.message);
        }
    } catch (error) {
        console.error('Error setting control:', error);
    }
}

async function loadProfilesForCamera(cameraId) {
    try {
        const response = await fetch(`${API_BASE}/api/cameras/${cameraId}/profiles`);
        const data = await response.json();

        if (data.success && data.profiles) {
            const select = document.getElementById('profile-select');
            data.profiles.forEach(profile => {
                const option = document.createElement('option');
                option.value = profile;
                option.textContent = profile;
                select.appendChild(option);
            });
        }
    } catch (error) {
        console.error('Error loading profiles:', error);
    }
}

async function applyProfile(cameraId) {
    const select = document.getElementById('profile-select');
    const profileName = select.value;

    if (!profileName) {
        alert('Please select a profile');
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/api/cameras/${cameraId}/profile`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ profile_name: profileName })
        });

        const data = await response.json();
        if (data.success) {
            alert('Profile applied successfully');
            // Reload controls to show updated values
            selectCameraForControls(cameraId);
        } else {
            alert(`Failed to apply profile: ${data.message}`);
        }
    } catch (error) {
        console.error('Error applying profile:', error);
        alert('Error applying profile');
    }
}

async function saveCurrentProfile(cameraId) {
    const profileName = prompt('Enter profile name:');
    if (!profileName) return;

    // Collect current control values
    const controls = {};
    document.querySelectorAll('[data-control]').forEach(control => {
        const name = control.dataset.control;
        const value = control.type === 'checkbox' ? control.checked :
            control.type === 'range' ? parseFloat(control.value) :
                control.value;
        controls[name] = value;
    });

    try {
        const response = await fetch(`${API_BASE}/api/cameras/${cameraId}/profiles/${profileName}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(controls)
        });

        const data = await response.json();
        if (data.success) {
            alert('Profile saved successfully');
            loadProfilesForCamera(cameraId);
        } else {
            alert(`Failed to save profile: ${data.message}`);
        }
    } catch (error) {
        console.error('Error saving profile:', error);
        alert('Error saving profile');
    }
}

// ============================================================================
// Camera Groups
// ============================================================================

async function loadGroups() {
    try {
        const response = await fetch(`${API_BASE}/api/camera-groups`);
        const data = await response.json();

        if (data.success) {
            groups = data.groups;
            renderGroupList();
        }
    } catch (error) {
        console.error('Error loading groups:', error);
    }
}

function renderGroupList() {
    const list = document.getElementById('group-list');

    if (groups.length === 0) {
        list.innerHTML = '<p>No groups</p>';
        return;
    }

    list.innerHTML = groups.map(group => `
        <div class="group-item" onclick="selectGroup('${group.group_id}')">
            <strong>${group.name}</strong> (${group.group_type})
            <br><small>${group.camera_ids.length} cameras</small>
        </div>
    `).join('');
}

function showCreateGroupModal() {
    const modal = document.getElementById('create-group-modal');
    const selection = document.getElementById('group-camera-selection');

    // Populate camera selection
    selection.innerHTML = cameras.map(camera => `
        <label style="display: block; margin: 5px 0;">
            <input type="checkbox" value="${camera.id}">
            ${camera.name}
        </label>
    `).join('');

    modal.classList.add('show');
}

async function handleCreateGroup(e) {
    e.preventDefault();

    const name = document.getElementById('group-name').value;
    const type = document.getElementById('group-type').value;
    const selectedCameras = Array.from(
        document.querySelectorAll('#group-camera-selection input:checked')
    ).map(cb => cb.value);

    if (selectedCameras.length === 0) {
        alert('Please select at least one camera');
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/api/camera-groups`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name,
                camera_ids: selectedCameras,
                group_type: type
            })
        });

        const data = await response.json();
        if (data.success) {
            alert('Group created successfully');
            document.getElementById('create-group-modal').classList.remove('show');
            document.getElementById('create-group-form').reset();
            loadGroups();
        } else {
            alert(`Failed to create group: ${data.message}`);
        }
    } catch (error) {
        console.error('Error creating group:', error);
        alert('Error creating group');
    }
}

function selectGroup(groupId) {
    // Future: Start all cameras in group, show group controls
    console.log('Selected group:', groupId);
}

// ============================================================================
// Features
// ============================================================================

async function loadFeatures() {
    try {
        const response = await fetch(`${API_BASE}/api/features`);
        const data = await response.json();

        if (data.success) {
            features = data.features;
            renderFeatureList();
        }
    } catch (error) {
        console.error('Error loading features:', error);
    }
}

function renderFeatureList() {
    const list = document.getElementById('feature-list');

    if (features.length === 0) {
        list.innerHTML = '<p>No features available</p>';
        return;
    }

    list.innerHTML = features.map(feature => `
        <button class="btn btn-primary" onclick="selectFeature('${feature.name}')" title="${feature.description}" style="width: 100%; margin-bottom: 8px;">
            ${feature.name}
        </button>
    `).join('');
}

function selectFeature(featureName) {
    // Open calibration wizard for Camera Calibration feature
    if (featureName === 'Camera Calibration') {
        openCalibrationWizard();
    } else {
        console.log('Selected feature:', featureName);
    }
}

// ============================================================================
// Workflows
// ============================================================================

async function loadWorkflows() {
    try {
        const response = await fetch(`${API_BASE}/api/workflows`);
        const data = await response.json();

        if (data.success) {
            workflows = data.workflows;
            renderWorkflowList();
        }
    } catch (error) {
        console.error('Error loading workflows:', error);
    }
}

function renderWorkflowList() {
    const list = document.getElementById('workflow-list');

    if (workflows.length === 0) {
        list.innerHTML = '<p>No workflows available</p>';
        return;
    }

    list.innerHTML = workflows.map(workflow => `
        <div class="workflow-item" onclick="startWorkflow('${workflow.workflow_name}')">
            <strong>${workflow.name}</strong>
            <br><small>${workflow.description}</small>
        </div>
    `).join('');
}

async function startWorkflow(workflowName) {
    // Future: Start workflow and show wizard
    console.log('Starting workflow:', workflowName);
}

// ============================================================================
// Sensor Configuration
// ============================================================================

let currentSensorConfig = null;
let availableSensors = [];

async function openSensorConfig() {
    try {
        const response = await fetch(`${API_BASE}/api/sensor-config`);
        const data = await response.json();

        if (data.success) {
            currentSensorConfig = data;
            availableSensors = data.available_sensors || ['IMX219', 'IMX477'];

            // Populate sensor dropdown
            const sensorSelect = document.getElementById('sensor-select');
            sensorSelect.innerHTML = availableSensors.map(sensor =>
                `<option value="${sensor}" ${sensor === data.current_sensor ? 'selected' : ''}>${sensor}</option>`
            ).join('');

            renderSensorConfig();
            document.getElementById('sensor-config-modal').style.display = 'flex';
        } else {
            alert('Failed to load sensor configuration: ' + data.message);
        }
    } catch (error) {
        console.error('Error loading sensor config:', error);
        alert('Error loading sensor configuration');
    }
}

function closeSensorConfig() {
    document.getElementById('sensor-config-modal').style.display = 'none';
    cancelAddSensor();
}

function showAddSensorForm() {
    document.getElementById('add-sensor-form').style.display = 'block';
    document.getElementById('new-sensor-name').focus();
}

function cancelAddSensor() {
    document.getElementById('add-sensor-form').style.display = 'none';
    document.getElementById('new-sensor-name').value = '';
}

function confirmAddSensor() {
    const sensorName = document.getElementById('new-sensor-name').value.trim().toUpperCase();

    if (!sensorName) {
        alert('Please enter a sensor name');
        return;
    }

    if (!/^[A-Z0-9]+$/.test(sensorName)) {
        alert('Sensor name must contain only uppercase letters and numbers');
        return;
    }

    if (availableSensors.includes(sensorName)) {
        alert(`Sensor ${sensorName} already exists`);
        return;
    }

    // Add to available sensors
    availableSensors.push(sensorName);

    // Create default config for new sensor
    currentSensorConfig.fps_map = { '1920x1080': 30 };
    currentSensorConfig.default_fps = 30;
    currentSensorConfig.current_sensor = sensorName;

    // Update dropdown
    const sensorSelect = document.getElementById('sensor-select');
    sensorSelect.innerHTML = availableSensors.map(sensor =>
        `<option value="${sensor}" ${sensor === sensorName ? 'selected' : ''}>${sensor}</option>`
    ).join('');

    renderSensorConfig();
    cancelAddSensor();
}

async function deleteSensor() {
    const selectedSensor = document.getElementById('sensor-select').value;

    if (!selectedSensor) {
        alert('No sensor selected');
        return;
    }

    // Don't allow deleting if it's the only sensor
    if (availableSensors.length <= 1) {
        alert('Cannot delete the only sensor. Create another sensor first.');
        return;
    }

    // Confirm deletion
    if (!confirm(`Are you sure you want to delete sensor ${selectedSensor}?\n\nThis will remove all its configuration data.`)) {
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/api/sensor-config/${selectedSensor}`, {
            method: 'DELETE'
        });

        const data = await response.json();

        if (data.success) {
            // Remove from available sensors
            availableSensors = availableSensors.filter(s => s !== selectedSensor);

            // Switch to first available sensor
            const newSensor = availableSensors[0];
            currentSensorConfig.current_sensor = newSensor;

            // Update dropdown
            const sensorSelect = document.getElementById('sensor-select');
            sensorSelect.innerHTML = availableSensors.map(sensor =>
                `<option value="${sensor}" ${sensor === newSensor ? 'selected' : ''}>${sensor}</option>`
            ).join('');

            // Load the new sensor's config
            await loadSensorPreset();

            alert(`Sensor ${selectedSensor} deleted successfully`);
        } else {
            alert('Failed to delete sensor: ' + data.message);
        }
    } catch (error) {
        console.error('Error deleting sensor:', error);
        alert('Error deleting sensor');
    }
}

async function loadSensorPreset() {
    const sensorSelect = document.getElementById('sensor-select');
    const selectedSensor = sensorSelect.value;

    try {
        // Try to load saved config for this sensor from backend
        const response = await fetch(`${API_BASE}/api/sensor-config/${selectedSensor}`);
        const data = await response.json();

        if (data.success && data.fps_map) {
            // Use saved config
            currentSensorConfig.fps_map = data.fps_map;
            currentSensorConfig.default_fps = data.default_fps;
            currentSensorConfig.current_sensor = selectedSensor;
        } else {
            // Use preset if no saved config
            const presets = {
                'IMX219': {
                    fps_map: { '3840x2160': 30, '1920x1080': 60, '1280x720': 60, '640x480': 60 },
                    default_fps: 30
                },
                'IMX477': {
                    fps_map: { '4032x3040': 10, '1920x1080': 50, '1280x720': 120 },
                    default_fps: 30
                }
            };

            const preset = presets[selectedSensor] || { fps_map: { '1920x1080': 30 }, default_fps: 30 };
            currentSensorConfig.fps_map = preset.fps_map;
            currentSensorConfig.default_fps = preset.default_fps;
            currentSensorConfig.current_sensor = selectedSensor;
        }

        renderSensorConfig();
    } catch (error) {
        console.error('Error loading sensor preset:', error);
    }
}

function renderSensorConfig() {
    const list = document.getElementById('fps-map-list');
    const fpsMap = currentSensorConfig.fps_map || {};
    const defaultFps = currentSensorConfig.default_fps || 30;

    let html = '<div style="margin-bottom: 15px;">';
    html += '<label style="display: block; margin-bottom: 5px;"><strong>Default FPS (for unknown resolutions):</strong></label>';
    html += `<input type="number" id="default-fps" value="${defaultFps}" min="1" max="120" style="width: 100px; padding: 8px; border: 1px solid #ddd; border-radius: 4px;">`;
    html += '</div>';

    html += '<div style="border-top: 1px solid #ddd; padding-top: 15px;">';
    html += '<strong>Resolution → FPS Mappings:</strong>';
    html += '<div style="margin-top: 10px;">';

    for (const [resolution, fps] of Object.entries(fpsMap)) {
        html += `
            <div style="display: flex; gap: 10px; margin-bottom: 10px; align-items: center;">
                <input type="text" value="${resolution}" readonly 
                    style="width: 150px; padding: 8px; border: 1px solid #ddd; border-radius: 4px; background: #f5f5f5;">
                <span>→</span>
                <input type="number" value="${fps}" min="1" max="120" data-resolution="${resolution}"
                    class="fps-value" style="width: 100px; padding: 8px; border: 1px solid #ddd; border-radius: 4px;">
                <span>fps</span>
                <button onclick="removeFpsMapping('${resolution}')" class="btn btn-secondary" style="padding: 6px 12px;">Remove</button>
            </div>
        `;
    }

    html += '</div></div>';

    html += '<div id="new-mapping" style="margin-top: 15px; display: none; padding: 15px; background: #f8f9fa; border-radius: 8px;">';
    html += '<strong>Add New Mapping:</strong>';
    html += '<div style="display: flex; gap: 10px; margin-top: 10px; align-items: center;">';
    html += '<input type="text" id="new-resolution" placeholder="1920x1080" style="width: 150px; padding: 8px; border: 1px solid #ddd; border-radius: 4px;">';
    html += '<span>→</span>';
    html += '<input type="number" id="new-fps" placeholder="60" min="1" max="120" style="width: 100px; padding: 8px; border: 1px solid #ddd; border-radius: 4px;">';
    html += '<span>fps</span>';
    html += '<button onclick="confirmAddFpsMapping()" class="btn btn-primary" style="padding: 6px 12px;">Add</button>';
    html += '<button onclick="cancelAddFpsMapping()" class="btn btn-secondary" style="padding: 6px 12px;">Cancel</button>';
    html += '</div></div>';

    list.innerHTML = html;
}

function addFpsMapping() {
    document.getElementById('new-mapping').style.display = 'block';
}

function cancelAddFpsMapping() {
    document.getElementById('new-mapping').style.display = 'none';
    document.getElementById('new-resolution').value = '';
    document.getElementById('new-fps').value = '';
}

function confirmAddFpsMapping() {
    const resolution = document.getElementById('new-resolution').value.trim();
    const fps = document.getElementById('new-fps').value;

    if (!resolution || !fps) {
        alert('Please enter both resolution and FPS');
        return;
    }

    // Validate resolution format
    if (!/^\d+x\d+$/.test(resolution)) {
        alert('Invalid resolution format. Use format like: 1920x1080');
        return;
    }

    currentSensorConfig.fps_map[resolution] = parseInt(fps);
    renderSensorConfig();
    cancelAddFpsMapping();
}

function removeFpsMapping(resolution) {
    if (confirm(`Remove mapping for ${resolution}?`)) {
        delete currentSensorConfig.fps_map[resolution];
        renderSensorConfig();
    }
}

async function saveSensorConfig() {
    try {
        // Collect all FPS values from inputs
        const fpsInputs = document.querySelectorAll('.fps-value');
        fpsInputs.forEach(input => {
            const resolution = input.getAttribute('data-resolution');
            currentSensorConfig.fps_map[resolution] = parseInt(input.value);
        });

        // Get default FPS
        const defaultFps = parseInt(document.getElementById('default-fps').value);
        currentSensorConfig.default_fps = defaultFps;

        // Get selected sensor
        const selectedSensor = document.getElementById('sensor-select').value;

        // Save to backend
        const response = await fetch(`${API_BASE}/api/sensor-config`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sensor: selectedSensor,
                fps_map: currentSensorConfig.fps_map,
                default_fps: currentSensorConfig.default_fps
            })
        });

        const data = await response.json();

        if (data.success) {
            alert('Sensor configuration saved successfully!');
            closeSensorConfig();
        } else {
            alert('Failed to save configuration: ' + data.message);
        }
    } catch (error) {
        console.error('Error saving sensor config:', error);
        alert('Error saving sensor configuration');
    }
}

// Add event listener for sensor config button
document.getElementById('sensor-config-btn').addEventListener('click', openSensorConfig);

// ============================================================================
// CUSTOM DIALOG UTILITIES
// ============================================================================

function showConfirm(title, message) {
    return new Promise((resolve) => {
        const modal = document.getElementById('confirm-dialog');
        const titleEl = document.getElementById('confirm-title');
        const messageEl = document.getElementById('confirm-message');
        const cancelBtn = document.getElementById('confirm-cancel');
        const okBtn = document.getElementById('confirm-ok');

        titleEl.textContent = title;
        messageEl.textContent = message;
        modal.classList.add('show');

        const cleanup = () => {
            modal.classList.remove('show');
            cancelBtn.onclick = null;
            okBtn.onclick = null;
        };

        cancelBtn.onclick = () => {
            cleanup();
            resolve(false);
        };

        okBtn.onclick = () => {
            cleanup();
            resolve(true);
        };

        // Close on click outside
        modal.onclick = (e) => {
            if (e.target === modal) {
                cleanup();
                resolve(false);
            }
        };
    });
}

function showAlert(title, message) {
    return new Promise((resolve) => {
        const modal = document.getElementById('alert-dialog');
        const titleEl = document.getElementById('alert-title');
        const messageEl = document.getElementById('alert-message');
        const okBtn = document.getElementById('alert-ok');

        titleEl.textContent = title;
        messageEl.textContent = message;
        modal.classList.add('show');

        const cleanup = () => {
            modal.classList.remove('show');
            okBtn.onclick = null;
        };

        okBtn.onclick = () => {
            cleanup();
            resolve(true);
        };

        // Close on click outside
        modal.onclick = (e) => {
            if (e.target === modal) {
                cleanup();
                resolve(true);
            }
        };
    });
}

// ============================================================================
// CALIBRATION WIZARD
// ============================================================================

let currentWizardStep = 1;
let calibrationData = {
    camera_id: null,
    pattern: {
        type: 'checkerboard',
        width: 9,
        height: 6,
        square_size: 25
    },
    target_images: 20,
    captured_images: [],
    results: null
};

// Coverage map tracking
let coverageMapCanvas = null;
let coverageMapContext = null;
let coverageMapData = [];  // Stores detected marker positions
let captureCount = 0;  // Track number of captures

// Auto-capture tracking
let autoCaptureEnabled = false;
let autoCaptureInterval = 3;  // Default 3 seconds
let autoCaptureTimer = null;
let lastCaptureMarkers = null;  // Store last capture's marker positions for movement detection
const MIN_MARKER_MOVEMENT = 50;  // Minimum pixel movement required (average across all markers)

// Pose diversity tracking for calibration quality
let poseCounts = {
    center: 0,
    topLeft: 0,
    topRight: 0,
    bottomLeft: 0,
    bottomRight: 0,
    topEdge: 0,
    bottomEdge: 0,
    leftEdge: 0,
    rightEdge: 0,
    tilted: 0,
    close: 0,
    far: 0
};

// Initialize calibration wizard
function initializeCalibrationWizard() {
    const calibrationModal = document.getElementById('calibration-modal');
    const closeBtn = calibrationModal.querySelector('.close');

    // Close modal
    closeBtn.addEventListener('click', () => {
        closeCalibrationWizard();
    });

    // Setup step event listeners
    setupSetupStepListeners();
    setupCaptureStepListeners();
    setupCalibrationStepListeners();
    setupReviewStepListeners();
}

function openCalibrationWizard() {
    const modal = document.getElementById('calibration-modal');
    modal.classList.add('show');

    // Reset wizard state
    currentWizardStep = 1;
    calibrationData = {
        camera_id: null,
        model: 'pinhole',
        board: {
            width: 8,
            height: 5,
            square_length: 50,
            marker_length: 37,
            dictionary: 'DICT_6X6_100'
        },
        target_images: 20,
        captured_images: [],
        results: null,
        session_name: null  // Will be auto-generated on first capture
    };

    // Hide skip to calibration button initially
    const skipBtn = document.getElementById('skip-to-calibration-btn');
    if (skipBtn) {
        skipBtn.style.display = 'none';
    }

    // Populate camera selector
    populateCalibrationCameraSelect();

    // Load previous calibration sessions
    loadPreviousCalibrationSessions();

    // Show first step
    showWizardStep(1);
}

function openCalibrationForCamera(cameraId) {
    // Open the wizard
    openCalibrationWizard();

    // Set the camera
    calibrationData.camera_id = cameraId;
    const cameraSelect = document.getElementById('calibration-camera-select');
    if (cameraSelect) {
        cameraSelect.value = cameraId;
        // Trigger the preview update
        updateSetupCameraPreview(cameraId);
    }
}

function closeCalibrationWizard() {
    const modal = document.getElementById('calibration-modal');
    modal.classList.remove('show');

    // Clean up any active camera streams
    stopCalibrationCamera();
}

function populateCalibrationCameraSelect() {
    const select = document.getElementById('calibration-camera-select');
    select.innerHTML = '<option value="">Choose a camera...</option>';

    cameras.forEach(camera => {
        const option = document.createElement('option');
        option.value = camera.id;
        option.textContent = `Camera ${camera.id} - ${camera.name}`;
        select.appendChild(option);
    });
}

function navigateWizard(direction) {
    const targetStep = currentWizardStep + direction;

    if (targetStep < 1 || targetStep > 4) return;

    // Validate current step before moving forward
    if (direction > 0) {
        validateWizardStep(currentWizardStep).then(isValid => {
            if (isValid) {
                showWizardStep(targetStep);
            }
        });
    } else {
        showWizardStep(targetStep);
    }
}

function showWizardStep(stepNumber) {
    currentWizardStep = stepNumber;

    // Update progress indicator
    document.querySelectorAll('.wizard-step').forEach((step, index) => {
        const stepNum = index + 1;
        step.classList.remove('active', 'completed');

        if (stepNum < currentWizardStep) {
            step.classList.add('completed');
        } else if (stepNum === currentWizardStep) {
            step.classList.add('active');
        }
    });

    // Update panels
    document.querySelectorAll('.wizard-panel').forEach((panel, index) => {
        panel.classList.remove('active');
        if (index + 1 === currentWizardStep) {
            panel.classList.add('active');
        }
    });

    // Step-specific actions
    if (stepNumber === 2) {
        // Auto-start camera preview when entering capture step
        initializeCapturePreview();

        // Initialize and draw coverage map if we have loaded data
        if (coverageMapData.length > 0 && !coverageMapCanvas) {
            coverageMapCanvas = document.getElementById('coverage-map-canvas');
            if (coverageMapCanvas) {
                coverageMapContext = coverageMapCanvas.getContext('2d');
                // Set canvas size based on first detection or default
                const imageWidth = coverageMapData[0]?.imageWidth || 1920;
                const imageHeight = coverageMapData[0]?.imageHeight || 1080;
                coverageMapCanvas.width = imageWidth;
                coverageMapCanvas.height = imageHeight;

                // Draw the coverage map
                drawCoverageMap();

                // Hide placeholder
                const placeholder = document.querySelector('.coverage-map-placeholder');
                if (placeholder) placeholder.style.display = 'none';
            }
        } else if (coverageMapData.length > 0 && coverageMapCanvas) {
            // Canvas exists, just redraw
            drawCoverageMap();
            const placeholder = document.querySelector('.coverage-map-placeholder');
            if (placeholder) placeholder.style.display = 'none';
        }
    } else if (stepNumber !== 2 && calibrationCameraActive) {
        // Stop camera when leaving capture step
        stopCalibrationCamera();
    }

    // Update navigation buttons in current panel
    const activePanel = document.querySelector(`.wizard-panel[data-panel="${currentWizardStep}"]`);
    if (!activePanel) return;

    const prevBtn = activePanel.querySelector('.wizard-prev');
    const nextBtn = activePanel.querySelector('.wizard-next');
    const cancelBtn = activePanel.querySelector('.wizard-cancel');

    if (prevBtn) {
        prevBtn.disabled = currentWizardStep === 1;
        prevBtn.onclick = () => navigateWizard(-1);
    }

    if (nextBtn) {
        if (currentWizardStep === 4) {
            nextBtn.textContent = 'Finish';
            nextBtn.onclick = finishCalibration;
            // Update review panel when entering step 4
            updateReviewPanel().catch(err => console.error('Error updating review panel:', err));
        } else {
            nextBtn.textContent = 'Next →';
            nextBtn.onclick = () => navigateWizard(1);
        }
    }

    if (cancelBtn) {
        cancelBtn.onclick = handleWizardCancel;
    }
}

async function handleWizardCancel() {
    const confirmed = await showConfirm(
        'Cancel Calibration',
        'Are you sure you want to cancel calibration? All progress will be lost.'
    );
    if (confirmed) {
        closeCalibrationWizard();
    }
}

async function validateWizardStep(stepNumber) {
    switch (stepNumber) {
        case 1: // Setup
            const cameraId = document.getElementById('calibration-camera-select').value;
            if (!cameraId) {
                await showAlert('Camera Required', 'Please select a camera to continue.');
                return false;
            }
            calibrationData.camera_id = cameraId;

            // Update calibration model
            calibrationData.model = document.getElementById('calibration-model').value;

            // Update ChArUco board configuration
            calibrationData.board.width = parseInt(document.getElementById('board-width').value);
            calibrationData.board.height = parseInt(document.getElementById('board-height').value);
            calibrationData.board.square_length = parseFloat(document.getElementById('square-length').value);
            calibrationData.board.marker_length = parseFloat(document.getElementById('marker-length').value);
            calibrationData.board.dictionary = document.getElementById('aruco-dictionary').value;
            calibrationData.target_images = parseInt(document.getElementById('target-images').value);            // Update target display in capture step
            document.getElementById('images-target').textContent = calibrationData.target_images;

            return true;

        case 2: // Capture
            // Allow proceeding to calibration step
            return true;

        case 3: // Calibration
            // Check if calibration has been run
            if (!calibrationData.results) {
                await showAlert('Calibration Required', 'Please run the calibration before proceeding to review.');
                return false;
            }
            return true;

        default:
            return true;
    }
}

// ============================================================================
// Step 1: Setup
// ============================================================================

function setupSetupStepListeners() {
    // Previous calibration session selector
    const previousCalibrationSelect = document.getElementById('previous-calibration-select');
    if (previousCalibrationSelect) {
        previousCalibrationSelect.addEventListener('change', async (e) => {
            if (e.target.value) {
                await loadPreviousCalibration(e.target.value);
            }
        });
    }

    // Camera selection change handler
    const cameraSelect = document.getElementById('calibration-camera-select');
    if (cameraSelect) {
        cameraSelect.addEventListener('change', async (e) => {
            await updateSetupCameraPreview(e.target.value);
        });
    }

    // Board preset selector
    const boardPreset = document.getElementById('board-preset');
    if (boardPreset) {
        boardPreset.addEventListener('change', (e) => {
            applyBoardPreset(e.target.value);
        });
    }

    // ArUco dictionary selector
    const dictionarySelect = document.getElementById('aruco-dictionary');
    if (dictionarySelect) {
        dictionarySelect.addEventListener('change', updateBoardInfo);
    }

    // Board configuration inputs
    const boardInputs = ['board-width', 'board-height', 'square-length', 'marker-length'];
    boardInputs.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.addEventListener('input', updateBoardInfo);
        }
    });

    // Skip to Calibration button
    const skipBtn = document.getElementById('skip-to-calibration-btn');
    if (skipBtn) {
        skipBtn.addEventListener('click', () => {
            // Jump directly to step 3 (calibration)
            showWizardStep(3);
        });
    }

    // Generate board button
    const generateBtn = document.getElementById('generate-board-btn');
    if (generateBtn) {
        generateBtn.addEventListener('click', generateCharucoBoard);
    }

    // Initialize board info
    updateBoardInfo();
}

async function loadPreviousCalibrationSessions() {
    try {
        const response = await fetch(`${API_BASE}/api/calibration/sessions`);
        const data = await response.json();

        const select = document.getElementById('previous-calibration-select');
        if (!select) return;

        // Clear existing options except the first one
        select.innerHTML = '<option value="">Start new calibration...</option>';

        if (data.success && data.session_details && data.session_details.length > 0) {
            data.session_details.forEach(session => {
                const option = document.createElement('option');
                option.value = session.path;
                option.textContent = `${session.camera_id} - ${session.date} (${session.image_count} images)`;
                select.appendChild(option);
            });
        }
    } catch (error) {
        console.error('Failed to load previous calibration sessions:', error);
    }
}

async function loadPreviousCalibration(sessionPath) {
    // Show loading indicator in the dropdown area
    const select = document.getElementById('previous-calibration-select');
    const progressContainer = document.getElementById('session-load-progress');
    const progressBar = document.getElementById('session-load-bar');
    const progressCount = document.getElementById('session-load-count');
    const originalHTML = select.innerHTML;

    select.disabled = true;
    if (progressContainer) {
        progressContainer.style.display = 'block';
    }

    try {
        const response = await fetch(`${API_BASE}/api/calibration/session/${encodeURIComponent(sessionPath)}`);
        const data = await response.json();

        if (!data.success) {
            select.disabled = false;
            select.innerHTML = originalHTML;
            if (progressContainer) progressContainer.style.display = 'none';
            await showAlert('Error', data.message || 'Failed to load calibration session');
            return;
        }

        const session = data.session;
        const totalImages = session.images.length;

        // Populate calibration data
        calibrationData.camera_id = session.camera_id;
        calibrationData.model = session.model || 'pinhole';
        calibrationData.board = session.board_config || calibrationData.board;
        calibrationData.target_images = session.target_images || 20;
        calibrationData.captured_images = session.images || [];
        calibrationData.session_name = session.path;  // Use the session path as session name

        // Reset and rebuild pose counts and coverage map
        Object.keys(poseCounts).forEach(key => {
            poseCounts[key] = 0;
        });
        coverageMapData = [];
        captureCount = 0;

        // Get camera resolution for coverage map
        let imageWidth = 1920;
        let imageHeight = 1080;

        // Process each image's detection data with async updates
        for (let index = 0; index < session.images.length; index++) {
            const image = session.images[index];

            // Update progress bar
            const percent = ((index + 1) / totalImages) * 100;
            if (progressBar) progressBar.style.width = `${percent}%`;
            if (progressCount) progressCount.textContent = `${index + 1}/${totalImages}`;

            // Force immediate UI update by yielding to browser every iteration
            await new Promise(resolve => setTimeout(resolve, 0));

            if (image.detection && image.detection.detected) {
                const detection = image.detection;

                // Update image dimensions
                if (detection.image_width && detection.image_height) {
                    imageWidth = detection.image_width;
                    imageHeight = detection.image_height;
                }

                // Increment capture count
                captureCount++;

                // Update pose counts
                if (detection.poses && Array.isArray(detection.poses)) {
                    detection.poses.forEach(pose => {
                        if (poseCounts.hasOwnProperty(pose)) {
                            poseCounts[pose]++;
                        }
                    });
                }

                // Add marker positions to coverage map
                if (detection.marker_corners && Array.isArray(detection.marker_corners)) {
                    detection.marker_corners.forEach((corners, markerIndex) => {
                        // Calculate center of this marker
                        let centerX = 0, centerY = 0;
                        corners.forEach(point => {
                            centerX += point[0];
                            centerY += point[1];
                        });
                        centerX /= corners.length;
                        centerY /= corners.length;

                        // Add marker position
                        coverageMapData.push({
                            x: centerX,
                            y: centerY,
                            captureNumber: captureCount,
                            markerIndex: markerIndex,
                            totalMarkers: detection.markers_detected,
                            corners: detection.corners_detected,
                            poses: detection.poses || []
                        });
                    });
                }
            }
        }

        // Initialize coverage map canvas with proper dimensions
        if (coverageMapCanvas) {
            coverageMapCanvas.width = imageWidth;
            coverageMapCanvas.height = imageHeight;
        }

        // Update UI
        document.getElementById('calibration-camera-select').value = session.camera_id;
        document.getElementById('calibration-model').value = session.model || 'pinhole';
        document.getElementById('target-images').value = session.target_images || 20;

        // Update board configuration
        if (session.board_config) {
            document.getElementById('board-width').value = session.board_config.width || 8;
            document.getElementById('board-height').value = session.board_config.height || 5;
            document.getElementById('square-length').value = session.board_config.square_length || 50;
            document.getElementById('marker-length').value = session.board_config.marker_length || 37;
            document.getElementById('aruco-dictionary').value = session.board_config.dictionary || 'DICT_6X6_100';
            updateBoardInfo();
        }

        // Update camera preview
        await updateSetupCameraPreview(session.camera_id);

        // Update capture stats
        updateCaptureStats();

        // Initialize coverage map canvas if not already done
        if (!coverageMapCanvas) {
            coverageMapCanvas = document.getElementById('coverage-map-canvas');
            if (coverageMapCanvas) {
                coverageMapContext = coverageMapCanvas.getContext('2d');
                coverageMapCanvas.width = imageWidth;
                coverageMapCanvas.height = imageHeight;

                // Draw coverage map now that canvas is initialized
                drawCoverageMap();
                // Hide placeholder
                const placeholder = document.querySelector('.coverage-map-placeholder');
                if (placeholder) placeholder.style.display = 'none';
            }
        } else {
            // Canvas already exists, just redraw
            coverageMapCanvas.width = imageWidth;
            coverageMapCanvas.height = imageHeight;
            drawCoverageMap();
            const placeholder = document.querySelector('.coverage-map-placeholder');
            if (placeholder) placeholder.style.display = 'none';
        }

        // Update pose diversity stats
        updatePoseDiversityStats();

        // Show "Skip to Calibration" button since we have captured images
        const skipBtn = document.getElementById('skip-to-calibration-btn');
        if (skipBtn && session.images.length > 0) {
            skipBtn.style.display = 'inline-block';
        }

        // Re-enable dropdown and hide progress
        select.disabled = false;
        select.innerHTML = originalHTML;
        select.value = sessionPath;
        if (progressContainer) progressContainer.style.display = 'none';

        await showAlert('Session Loaded', `Loaded calibration session with ${session.images.length} captured images.\n\nPose diversity: ${captureCount} captures analyzed.\n\nYou can continue capturing or proceed to calibration.`);

    } catch (error) {
        console.error('Failed to load calibration session:', error);
        select.disabled = false;
        select.innerHTML = originalHTML;
        if (progressContainer) progressContainer.style.display = 'none';
        await showAlert('Error', 'Failed to load calibration session: ' + error.message);
    }
}

function applyBoardPreset(preset) {
    const presets = {
        'default': { width: 8, height: 5, square_length: 50, marker_length: 37 },
        'large': { width: 10, height: 7, square_length: 75, marker_length: 56 },
        'small': { width: 6, height: 4, square_length: 30, marker_length: 22 }
    };

    if (preset && presets[preset]) {
        const config = presets[preset];
        document.getElementById('board-width').value = config.width;
        document.getElementById('board-height').value = config.height;
        document.getElementById('square-length').value = config.square_length;
        document.getElementById('marker-length').value = config.marker_length;
        updateBoardInfo();
    }
}

function updateBoardInfo() {
    const width = parseInt(document.getElementById('board-width').value) || 8;
    const height = parseInt(document.getElementById('board-height').value) || 5;
    const dictionary = document.getElementById('aruco-dictionary').value;

    const markersTotal = Math.floor((width * height) / 2);
    const markersRequired = Math.floor(markersTotal * 0.9);

    document.getElementById('board-markers-total').textContent = markersTotal;
    document.getElementById('board-markers-required').textContent = markersRequired;
    document.getElementById('board-dictionary-display').textContent = dictionary;
}

function generateCharucoBoard() {
    const width = parseInt(document.getElementById('board-width').value);
    const height = parseInt(document.getElementById('board-height').value);
    const squareLength = parseFloat(document.getElementById('square-length').value);
    const markerLength = parseFloat(document.getElementById('marker-length').value);
    const dictionary = document.getElementById('aruco-dictionary').value;

    // Placeholder: In real implementation, this would call backend to generate board image
    showAlert(
        'Generate Board',
        `Board configuration:\n• Size: ${width}×${height} markers\n• Square: ${squareLength}mm\n• Marker: ${markerLength}mm\n• Dictionary: ${dictionary}\n\nBoard generation will be implemented with backend integration.`
    );
}

async function updateSetupCameraPreview(cameraId) {
    const previewImg = document.getElementById('setup-camera-preview');
    const placeholder = document.getElementById('setup-preview-placeholder');
    const cameraName = document.getElementById('setup-camera-name');

    if (!cameraId) {
        // No camera selected
        previewImg.style.display = 'none';
        placeholder.style.display = 'flex';
        cameraName.textContent = 'No camera selected';
        return;
    }

    // Find camera info
    const camera = cameras.find(c => c.id == cameraId);
    if (!camera) {
        placeholder.style.display = 'flex';
        previewImg.style.display = 'none';
        cameraName.textContent = 'Camera not found';
        return;
    }

    // Update camera name
    cameraName.textContent = `Camera ${cameraId} - ${camera.name}`;

    // Check if camera is already streaming
    if (camera.streaming) {
        // Camera already streaming, show preview
        previewImg.src = `${API_BASE}/api/cameras/${cameraId}/stream?t=${Date.now()}`;
        previewImg.style.display = 'block';
        placeholder.style.display = 'none';
    } else {
        // Need to start camera
        try {
            const response = await fetch(`${API_BASE}/api/cameras/${cameraId}/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({})
            });

            const data = await response.json();
            if (data.success) {
                // Camera started, show preview
                previewImg.src = `${API_BASE}/api/cameras/${cameraId}/stream?t=${Date.now()}`;
                previewImg.style.display = 'block';
                placeholder.style.display = 'none';

                // Update camera list state
                camera.streaming = true;
            } else {
                throw new Error(data.message || 'Failed to start camera');
            }
        } catch (error) {
            console.error('Error starting camera preview:', error);
            placeholder.innerHTML = `
                <p style="color: #f44336;">⚠️ Camera Error</p>
                <p class="preview-hint">${error.message}</p>
            `;
            placeholder.style.display = 'flex';
            previewImg.style.display = 'none';
        }
    }
}

// ============================================================================
// Step 2: Capture
// ============================================================================

let calibrationCameraActive = false;
let calibrationStreamInterval = null;
let markerDetectionCanvas = null;
let markerDetectionContext = null;

function setupCaptureStepListeners() {
    document.getElementById('start-capture').addEventListener('click', toggleCalibrationCamera);
    document.getElementById('capture-image').addEventListener('click', captureCalibrationImage);
    document.getElementById('clear-captures').addEventListener('click', clearCapturedImages);
    document.getElementById('toggle-auto-capture').addEventListener('click', toggleAutoCapture);
    document.getElementById('auto-capture-interval').addEventListener('change', (e) => {
        autoCaptureInterval = parseInt(e.target.value);
        if (autoCaptureEnabled) {
            // Restart auto-capture with new interval
            stopAutoCapture();
            startAutoCapture();
        }
    });
}

async function initializeCapturePreview() {
    // Auto-start camera preview if not already active
    if (!calibrationCameraActive && calibrationData.camera_id) {
        const btn = document.getElementById('start-capture');
        btn.textContent = '⏹️ Stop Camera';
        btn.classList.remove('btn-primary');
        btn.classList.add('btn-danger');
        await startCalibrationCamera();
    }
}

async function toggleCalibrationCamera() {
    const btn = document.getElementById('start-capture');

    if (calibrationCameraActive) {
        stopCalibrationCamera();
        btn.textContent = '▶️ Start Camera';
        btn.classList.remove('btn-danger');
        btn.classList.add('btn-primary');
    } else {
        await startCalibrationCamera();
        btn.textContent = '⏹️ Stop Camera';
        btn.classList.remove('btn-primary');
        btn.classList.add('btn-danger');
    }
}

async function startCalibrationCamera() {
    try {
        // Start the camera if not already started
        const cameraId = calibrationData.camera_id;
        const response = await fetch(`${API_BASE}/api/cameras/${cameraId}/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });

        const data = await response.json();
        if (!data.success) {
            throw new Error(data.message || 'Failed to start camera');
        }

        // Enable calibration overlay on server side
        await fetch(`${API_BASE}/api/calibration/overlay/${cameraId}/enable`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                board_config: calibrationData.board
            })
        });

        // Create simple preview without canvas overlay
        const previewDiv = document.querySelector('.capture-preview');
        previewDiv.innerHTML = `
            <img id="capture-stream-img" 
                 src="${API_BASE}/api/cameras/${cameraId}/stream" 
                 style="max-width: 100%; max-height: 100%; width: auto; height: auto; object-fit: contain; display: block;">
        `;

        // Get camera resolution to initialize coverage map
        const resResponse = await fetch(`${API_BASE}/api/cameras/${cameraId}/controls`);
        const resData = await resResponse.json();

        let canvasWidth = 1920;
        let canvasHeight = 1080;

        if (resData.success && resData.resolution_info) {
            canvasWidth = resData.resolution_info.capture_width || 1920;
            canvasHeight = resData.resolution_info.capture_height || 1080;
        }

        // Initialize coverage map canvas with actual camera resolution
        coverageMapCanvas = document.getElementById('coverage-map-canvas');
        coverageMapContext = coverageMapCanvas.getContext('2d');

        // Only clear coverage data if we don't have loaded data from a previous session
        if (coverageMapData.length === 0) {
            coverageMapData = [];
            captureCount = 0;
        }

        // Set canvas size to match camera resolution
        coverageMapCanvas.width = canvasWidth;
        coverageMapCanvas.height = canvasHeight;

        // Draw coverage map (either empty or with loaded data)
        drawCoverageMap();

        // Hide placeholder
        const placeholder = document.querySelector('.coverage-map-placeholder');
        if (placeholder) placeholder.style.display = 'none';

        calibrationCameraActive = true;
        document.getElementById('capture-image').disabled = false;
        document.getElementById('toggle-auto-capture').disabled = false;

        // Update pattern status periodically (just for the status text)
        startPatternDetection();

    } catch (error) {
        console.error('Error starting calibration camera:', error);
        await showAlert('Camera Error', 'Failed to start camera: ' + error.message);
    }
}

function stopCalibrationCamera() {
    calibrationCameraActive = false;
    document.getElementById('capture-image').disabled = true;
    document.getElementById('toggle-auto-capture').disabled = true;

    // Stop auto-capture if running
    if (autoCaptureEnabled) {
        stopAutoCapture();
        const btn = document.getElementById('toggle-auto-capture');
        btn.textContent = '⏯️ Start Auto-Capture';
        btn.classList.remove('btn-danger');
        btn.classList.add('btn-secondary');
    }

    // Disable calibration overlay on server side
    if (calibrationData.camera_id) {
        fetch(`${API_BASE}/api/calibration/overlay/${calibrationData.camera_id}/disable`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        }).catch(err => console.error('Error disabling overlay:', err));
    }

    const previewDiv = document.querySelector('.capture-preview');
    previewDiv.innerHTML = `
        <div class="preview-placeholder">
            <p>📷 Camera preview will appear here</p>
            <p class="preview-hint">Start camera to begin capturing</p>
        </div>
    `;

    // Keep coverage map data and display intact - don't reset
    // Coverage map persists across camera start/stop cycles

    // Clean up canvas references
    markerDetectionCanvas = null;
    markerDetectionContext = null;

    stopPatternDetection();
}

function startPatternDetection() {
    // Poll backend API just to update the pattern status text
    calibrationStreamInterval = setInterval(async () => {
        if (!calibrationCameraActive || !calibrationData.camera_id) return;

        try {
            const response = await fetch(`${API_BASE}/api/calibration/detect-pattern`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    camera_id: calibrationData.camera_id,
                    board_config: calibrationData.board
                })
            });

            const data = await response.json();

            if (data.success && data.detection) {
                const detection = data.detection;
                document.getElementById('pattern-status').textContent = detection.detected ? '✅ Yes' : '❌ No';
            } else {
                document.getElementById('pattern-status').textContent = '❌ No';
            }
        } catch (error) {
            console.error('Pattern detection error:', error);
            // Silently fail - don't disrupt the UI
        }
    }, 1000); // Check every second (reduced frequency since we're not drawing)
}

function updateCoverageMap(detection) {
    if (!coverageMapContext || !coverageMapCanvas) return;

    // Increment capture count
    captureCount++;

    // Update pose counts from backend classification
    if (detection.poses && Array.isArray(detection.poses)) {
        detection.poses.forEach(pose => {
            if (poseCounts.hasOwnProperty(pose)) {
                poseCounts[pose]++;
            }
        });
    }

    // Add each marker corner to coverage data
    if (detection.marker_corners && detection.marker_corners.length > 0) {
        detection.marker_corners.forEach((corners, markerIndex) => {
            // Calculate center of this marker
            let centerX = 0, centerY = 0;
            corners.forEach(point => {
                centerX += point[0];
                centerY += point[1];
            });
            centerX /= corners.length;
            centerY /= corners.length;

            // Add marker position
            coverageMapData.push({
                x: centerX,
                y: centerY,
                captureNumber: captureCount,
                markerIndex: markerIndex,
                totalMarkers: detection.markers_detected,
                corners: detection.corners_detected,
                poses: detection.poses || []  // Store backend-provided poses
            });
        });
    }

    // Redraw coverage map
    drawCoverageMap();

    // Update pose diversity stats
    updatePoseDiversityStats();
}

function drawCoverageMap() {
    if (!coverageMapContext || !coverageMapCanvas) return;

    const ctx = coverageMapContext;
    const width = coverageMapCanvas.width;
    const height = coverageMapCanvas.height;

    // Clear canvas
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, width, height);

    // Draw grid
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    const gridSize = 50;

    for (let x = 0; x < width; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
    }

    for (let y = 0; y < height; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
    }

    // Count unique captures
    const uniqueCaptures = captureCount;

    // Draw coverage points (each marker)
    coverageMapData.forEach((point, index) => {
        // Color based on quality (more corners = better)
        const quality = Math.min(1, point.corners / 20);
        const r = Math.floor(255 * (1 - quality));
        const g = Math.floor(255 * quality);

        // Draw circle
        ctx.fillStyle = `rgb(${r}, ${g}, 0)`;
        ctx.beginPath();
        ctx.arc(point.x, point.y, 6, 0, 2 * Math.PI);
        ctx.fill();

        // Draw outline
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1;
        ctx.stroke();
    });

    // Draw text overlay with pose diversity
    const overlayHeight = 85;
    ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
    ctx.fillRect(5, 5, 210, overlayHeight);

    ctx.fillStyle = '#00ff00';
    ctx.font = '13px monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(`Captures: ${uniqueCaptures}`, 10, 22);
    ctx.fillText(`Markers: ${coverageMapData.length}`, 10, 38);

    // Calculate pose diversity score
    const poseTypes = ['center', 'topLeft', 'topRight', 'bottomLeft', 'bottomRight',
        'topEdge', 'bottomEdge', 'leftEdge', 'rightEdge', 'tilted', 'close', 'far'];
    const capturedPoses = poseTypes.filter(pose => poseCounts[pose] > 0).length;
    const diversityPercent = Math.round((capturedPoses / poseTypes.length) * 100);

    ctx.fillStyle = diversityPercent > 70 ? '#00ff00' : diversityPercent > 40 ? '#ffaa00' : '#ff4444';
    ctx.fillText(`Diversity: ${diversityPercent}%`, 10, 54);

    ctx.fillStyle = '#888';
    ctx.font = '11px monospace';
    ctx.fillText(`${capturedPoses}/${poseTypes.length} pose types`, 10, 70);
    ctx.fillText(`Green = High Quality`, 10, 85);
}

// Update pose diversity display in UI stats
function updatePoseDiversityStats() {
    const statsDiv = document.querySelector('.pose-diversity-stats');
    if (!statsDiv) return;

    const recommendedCounts = {
        center: 3,
        corners: 2,  // Combined count for all 4 corners
        edges: 2,    // Combined count for all 4 edges
        tilted: 4,
        close: 2,
        far: 2
    };

    const cornerCount = poseCounts.topLeft + poseCounts.topRight + poseCounts.bottomLeft + poseCounts.bottomRight;
    const edgeCount = poseCounts.topEdge + poseCounts.bottomEdge + poseCounts.leftEdge + poseCounts.rightEdge;

    const getStatus = (current, recommended) => {
        if (current >= recommended) return '✅';
        if (current > 0) return '⚠️';
        return '❌';
    };

    statsDiv.innerHTML = `
        <div class="pose-item">
            <span>${getStatus(poseCounts.center, recommendedCounts.center)}</span>
            <span class="pose-label">Center:</span>
            <span class="pose-count">${poseCounts.center}/${recommendedCounts.center}</span>
        </div>
        <div class="pose-item">
            <span>${getStatus(cornerCount, recommendedCounts.corners * 4)}</span>
            <span class="pose-label">Corners:</span>
            <span class="pose-count">${cornerCount}/${recommendedCounts.corners * 4}</span>
        </div>
        <div class="pose-item">
            <span>${getStatus(edgeCount, recommendedCounts.edges * 4)}</span>
            <span class="pose-label">Edges:</span>
            <span class="pose-count">${edgeCount}/${recommendedCounts.edges * 4}</span>
        </div>
        <div class="pose-item">
            <span>${getStatus(poseCounts.tilted, recommendedCounts.tilted)}</span>
            <span class="pose-label">Tilted:</span>
            <span class="pose-count">${poseCounts.tilted}/${recommendedCounts.tilted}</span>
        </div>
        <div class="pose-item">
            <span>${getStatus(poseCounts.close, recommendedCounts.close)}</span>
            <span class="pose-label">Close:</span>
            <span class="pose-count">${poseCounts.close}/${recommendedCounts.close}</span>
        </div>
        <div class="pose-item">
            <span>${getStatus(poseCounts.far, recommendedCounts.far)}</span>
            <span class="pose-label">Far:</span>
            <span class="pose-count">${poseCounts.far}/${recommendedCounts.far}</span>
        </div>
    `;
}

function stopPatternDetection() {
    if (calibrationStreamInterval) {
        clearInterval(calibrationStreamInterval);
        calibrationStreamInterval = null;
    }
    document.getElementById('pattern-status').textContent = '❌ No';
    clearMarkerOverlay();
}

function drawMarkerOverlay(detection) {
    if (!markerDetectionContext || !markerDetectionCanvas) return;

    // Clear previous overlay
    markerDetectionContext.clearRect(0, 0, markerDetectionCanvas.width, markerDetectionCanvas.height);

    const scaleX = markerDetectionCanvas.width / (detection.image_width || 1);
    const scaleY = markerDetectionCanvas.height / (detection.image_height || 1);

    // Draw detected ArUco markers
    if (detection.marker_corners && detection.marker_corners.length > 0) {
        markerDetectionContext.strokeStyle = '#00ff00';
        markerDetectionContext.lineWidth = 3;

        detection.marker_corners.forEach((corners, idx) => {
            markerDetectionContext.beginPath();
            corners.forEach((corner, i) => {
                const x = corner[0] * scaleX;
                const y = corner[1] * scaleY;
                if (i === 0) {
                    markerDetectionContext.moveTo(x, y);
                } else {
                    markerDetectionContext.lineTo(x, y);
                }
            });
            markerDetectionContext.closePath();
            markerDetectionContext.stroke();

            // Draw marker ID if available
            if (detection.marker_ids && detection.marker_ids[idx] !== undefined) {
                const centerX = corners.reduce((sum, c) => sum + c[0], 0) / 4 * scaleX;
                const centerY = corners.reduce((sum, c) => sum + c[1], 0) / 4 * scaleY;
                markerDetectionContext.fillStyle = '#00ff00';
                markerDetectionContext.font = '16px Arial';
                markerDetectionContext.fillText(detection.marker_ids[idx].toString(), centerX - 10, centerY + 5);
            }
        });
    }

    // Draw ChArUco corners
    if (detection.charuco_corners && detection.charuco_corners.length > 0) {
        markerDetectionContext.fillStyle = '#ff0000';
        detection.charuco_corners.forEach(corner => {
            const x = corner[0] * scaleX;
            const y = corner[1] * scaleY;
            markerDetectionContext.beginPath();
            markerDetectionContext.arc(x, y, 5, 0, 2 * Math.PI);
            markerDetectionContext.fill();
        });
    }

    // Draw detection quality indicator
    if (detection.corners_detected !== undefined) {
        markerDetectionContext.fillStyle = 'rgba(0, 0, 0, 0.7)';
        markerDetectionContext.fillRect(10, 10, 280, 80);
        markerDetectionContext.fillStyle = '#ffffff';
        markerDetectionContext.font = '14px Arial';
        markerDetectionContext.fillText(`Markers: ${detection.markers_detected || 0}`, 20, 30);
        markerDetectionContext.fillText(`Corners: ${detection.corners_detected || 0}`, 20, 50);
        markerDetectionContext.fillText(`Quality: ${detection.quality || 'Unknown'}`, 20, 70);
    }
}

function clearMarkerOverlay() {
    if (markerDetectionContext && markerDetectionCanvas) {
        markerDetectionContext.clearRect(0, 0, markerDetectionCanvas.width, markerDetectionCanvas.height);
    }
}

// Classify the pose of the calibration board based on its position and characteristics
// Calculate average movement of markers compared to last capture
function calculateMarkerMovement(currentMarkers, previousMarkers) {
    if (!previousMarkers || !currentMarkers) return Infinity;  // No previous capture, allow this one

    if (currentMarkers.length !== previousMarkers.length) return Infinity;  // Different marker count, allow

    let totalMovement = 0;
    let markerCount = 0;

    // Calculate average movement across all markers
    for (let i = 0; i < currentMarkers.length; i++) {
        const current = currentMarkers[i];
        const previous = previousMarkers[i];

        if (!current || !previous || current.length === 0 || previous.length === 0) continue;

        // Each marker has 4 corners, calculate centroid
        let currX = 0, currY = 0, prevX = 0, prevY = 0;
        for (let j = 0; j < current[0].length; j++) {
            currX += current[0][j][0];
            currY += current[0][j][1];
            prevX += previous[0][j][0];
            prevY += previous[0][j][1];
        }
        currX /= current[0].length;
        currY /= current[0].length;
        prevX /= previous[0].length;
        prevY /= previous[0].length;

        // Calculate Euclidean distance
        const dx = currX - prevX;
        const dy = currY - prevY;
        const distance = Math.sqrt(dx * dx + dy * dy);

        totalMovement += distance;
        markerCount++;
    }

    return markerCount > 0 ? totalMovement / markerCount : 0;
}

async function captureCalibrationImage(skipMovementCheck = false) {
    try {
        // Try to detect pattern multiple times before giving up
        const maxAttempts = 3;
        let detection = null;

        for (let attempt = 1; attempt <= maxAttempts; attempt++) {
            const response = await fetch(`${API_BASE}/api/calibration/detect-pattern`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    camera_id: calibrationData.camera_id,
                    board_config: calibrationData.board
                })
            });

            const data = await response.json();

            if (data.success && data.detection && data.detection.detected) {
                detection = data.detection;
                break;
            }

            // Wait a bit before retrying (except on last attempt)
            if (attempt < maxAttempts) {
                await new Promise(resolve => setTimeout(resolve, 200));
            }
        }

        if (!detection) {
            // No pattern detected after retries - silently skip
            console.log('No calibration pattern detected, skipping capture');
            return;
        }

        // Check minimum marker count (require at least 10 markers for good calibration)
        const minMarkers = 10;
        if (detection.markers_detected < minMarkers) {
            console.log(`Insufficient markers detected (${detection.markers_detected}/${minMarkers}), skipping capture`);
            return;
        }

        // Check if markers have moved significantly from last capture (unless explicitly skipped)
        if (!skipMovementCheck && lastCaptureMarkers) {
            const movement = calculateMarkerMovement(detection.marker_corners, lastCaptureMarkers);
            if (movement < MIN_MARKER_MOVEMENT) {
                console.log(`Markers haven't moved enough (${movement.toFixed(1)}px < ${MIN_MARKER_MOVEMENT}px), skipping capture`);
                return;
            }
        }

        // Store current markers for next movement check
        lastCaptureMarkers = detection.marker_corners ? JSON.parse(JSON.stringify(detection.marker_corners)) : null;

        // Update coverage map with this detection
        if (detection.marker_corners && coverageMapContext) {
            updateCoverageMap(detection);
        }

        // Actually capture and save the image
        const captureResponse = await fetch(`${API_BASE}/api/calibration/capture-image`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                camera_id: calibrationData.camera_id,
                board_config: calibrationData.board,
                model: calibrationData.model,
                target_images: calibrationData.target_images,
                session_name: calibrationData.session_name  // Will be auto-generated if not set
            })
        });

        const captureData = await captureResponse.json();

        if (!captureData.success) {
            console.error('Failed to save image:', captureData.message);
            await showAlert('Capture Error', 'Failed to save image: ' + captureData.message);
            return;
        }

        // Store session name for subsequent captures
        if (!calibrationData.session_name && captureData.session_name) {
            calibrationData.session_name = captureData.session_name;
        }

        // Create image data entry
        const imageCount = calibrationData.captured_images.length + 1;
        const imageData = {
            id: imageCount,
            timestamp: new Date().toISOString(),
            filename: captureData.image_filename,
            path: captureData.image_path,
            thumbnail: 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="75"><rect fill="%23ddd" width="100" height="75"/><text x="50%" y="50%" text-anchor="middle" dy=".3em" fill="%23999">' + imageCount + '</text></svg>',
            detection: detection  // Store detection data with image
        };

        calibrationData.captured_images.push(imageData);

        // Update UI
        updateCaptureStats();

    } catch (error) {
        console.error('Error capturing image:', error);
        // Don't show modal on error, just log it
    }
}

function updateCaptureStats() {
    const count = calibrationData.captured_images.length;
    const target = calibrationData.target_images;

    document.getElementById('images-captured').textContent = count;

    const coverage = Math.min((count / target) * 100, 100);
    document.getElementById('coverage-progress').style.width = coverage + '%';
    document.getElementById('coverage-percent').textContent = Math.round(coverage) + '%';
}

function addThumbnail(imageData) {
    const gallery = document.getElementById('capture-thumbnails');

    const thumb = document.createElement('div');
    thumb.className = 'thumbnail-item';
    thumb.dataset.id = imageData.id;
    thumb.innerHTML = `
        <img src="${imageData.thumbnail}" alt="Capture ${imageData.id}">
        <button class="thumbnail-remove" onclick="removeThumbnail(${imageData.id})">×</button>
    `;

    gallery.appendChild(thumb);
}

function removeThumbnail(imageId) {
    calibrationData.captured_images = calibrationData.captured_images.filter(img => img.id !== imageId);

    const thumb = document.querySelector(`.thumbnail-item[data-id="${imageId}"]`);
    if (thumb) thumb.remove();

    updateCaptureStats();
}

async function clearCapturedImages() {
    if (calibrationData.captured_images.length === 0) return;

    const confirmed = await showConfirm(
        'Clear Images',
        'Are you sure you want to clear all captured images?'
    );
    if (confirmed) {
        calibrationData.captured_images = [];
        document.getElementById('capture-thumbnails').innerHTML = '';
        updateCaptureStats();

        // Reset coverage map
        coverageMapData = [];
        captureCount = 0;
        lastCaptureMarkers = null;  // Reset movement tracking

        // Reset pose counts
        Object.keys(poseCounts).forEach(key => {
            poseCounts[key] = 0;
        });

        if (coverageMapContext) {
            drawCoverageMap();
        }
        updatePoseDiversityStats();
    }
}

function toggleAutoCapture() {
    const btn = document.getElementById('toggle-auto-capture');

    if (autoCaptureEnabled) {
        stopAutoCapture();
        btn.textContent = '⏯️ Start Auto-Capture';
        btn.classList.remove('btn-danger');
        btn.classList.add('btn-secondary');
    } else {
        startAutoCapture();
        btn.textContent = '⏹️ Stop Auto-Capture';
        btn.classList.remove('btn-secondary');
        btn.classList.add('btn-danger');
    }
}

function startAutoCapture() {
    autoCaptureEnabled = true;
    const interval = autoCaptureInterval * 1000;  // Convert to milliseconds

    console.log(`Starting auto-capture with ${autoCaptureInterval}s interval`);

    // Attempt capture at set interval
    autoCaptureTimer = setInterval(async () => {
        if (!calibrationCameraActive) {
            stopAutoCapture();
            return;
        }

        console.log('Auto-capture: attempting capture...');
        await captureCalibrationImage(false);  // Don't skip movement check
    }, interval);
}

function stopAutoCapture() {
    autoCaptureEnabled = false;
    if (autoCaptureTimer) {
        clearInterval(autoCaptureTimer);
        autoCaptureTimer = null;
    }
    console.log('Auto-capture stopped');
}

// ============================================================================
// Step 3: Calibration
// ============================================================================

function setupCalibrationStepListeners() {
    document.getElementById('run-calibration').addEventListener('click', runCalibration);
}

async function runCalibration() {
    const btn = document.getElementById('run-calibration');
    btn.disabled = true;
    btn.textContent = '⏳ Calibrating...';

    // Clear previous log entries
    document.getElementById('calibration-log').innerHTML = '';

    // Update status
    updateCalibrationStatus('⏳', 'Starting Calibration', 'Preparing to process images...');
    addCalibrationLog('Calibration process started');

    // Reset progress
    const progressBar = document.getElementById('calibration-progress');
    const progressLabel = document.getElementById('calibration-progress-label');
    progressBar.style.width = '0%';
    progressLabel.textContent = '0%';

    try {
        // Check if we have captured images
        if (!calibrationData.session_name) {
            throw new Error('No calibration session found. Please capture images first.');
        }

        if (calibrationData.captured_images.length === 0) {
            throw new Error('No images captured. Please capture calibration images first.');
        }

        addCalibrationLog(`Found ${calibrationData.captured_images.length} captured images`);

        // Use board configuration from calibrationData
        const boardConfig = {
            dictionary: calibrationData.board.dictionary,
            width: calibrationData.board.width,
            height: calibrationData.board.height,
            square_length: calibrationData.board.square_length,
            marker_length: calibrationData.board.marker_length
        };

        addCalibrationLog('Board configuration loaded');
        addCalibrationLog('Sending calibration request to server...');
        updateCalibrationStatus('⏳', 'Processing Images', 'Running calibration algorithm...');

        // Use fetch with streaming response
        const response = await fetch('/api/calibration/run', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_name: calibrationData.session_name,
                camera_id: calibrationData.camera_id,
                board_config: boardConfig
            })
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error('Calibration request failed:', errorText);
            throw new Error('Calibration request failed: ' + errorText);
        }

        console.log('Starting to read streaming response...');
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let progressStep = 10;
        let messageCount = 0;

        while (true) {
            const { done, value } = await reader.read();

            if (done) {
                console.log(`Stream complete. Received ${messageCount} messages total.`);
                // If we got messages but no 'complete' message, the stream may have ended prematurely
                if (messageCount > 0 && !calibrationData.results) {
                    console.warn('Stream ended without completion message');
                    throw new Error('Calibration stream ended unexpectedly. Please try again.');
                }
                break;
            }

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n\n');
            buffer = lines.pop(); // Keep incomplete line in buffer

            console.log(`Processing ${lines.length} SSE lines`);

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.substring(6));
                        messageCount++;
                        console.log(`Message ${messageCount}:`, data);

                        if (data.type === 'progress') {
                            // Update progress incrementally
                            progressStep = Math.min(progressStep + 3, 90);
                            progressBar.style.width = `${progressStep}%`;
                            progressLabel.textContent = `${progressStep}%`;

                            // Add to log
                            addCalibrationLog(data.message);

                        } else if (data.type === 'complete') {
                            const result = data.result;
                            console.log('Calibration complete:', result);

                            // Complete progress
                            progressBar.style.width = '100%';
                            progressLabel.textContent = '100%';

                            // Store calibration results
                            calibrationData.results = result;

                            addCalibrationLog(`Processed ${result.images_used} images successfully`);
                            addCalibrationLog(`Reprojection error: ${result.reprojection_error.toFixed(4)} pixels`);
                            addCalibrationLog('Calibration completed successfully!', 'success');

                            // Update status to complete
                            updateCalibrationStatus('✅', 'Calibration Complete',
                                `Successfully calibrated with error: ${result.reprojection_error.toFixed(4)} pixels`);
                            addCalibrationLog('Ready to review results!', 'success');

                            // Update button
                            btn.disabled = false;
                            btn.textContent = '✓ Calibration Complete';
                            btn.classList.remove('btn-primary');
                            btn.classList.add('btn-success');

                            // Enable next button
                            const nextBtn = document.querySelector('[data-panel="3"] .wizard-next');
                            if (nextBtn) {
                                nextBtn.disabled = false;
                            }

                        } else if (data.type === 'error') {
                            console.error('Calibration error:', data.error);
                            throw new Error(data.error || 'Calibration failed');
                        }
                    } catch (parseError) {
                        console.error('Error parsing SSE message:', parseError, 'Line:', line);
                    }
                }
            }
        }

    } catch (error) {
        console.error('Calibration error:', error);

        updateCalibrationStatus('❌', 'Calibration Failed', error.message);
        addCalibrationLog(`Error: ${error.message}`, 'error');

        btn.disabled = false;
        btn.textContent = '🔄 Retry Calibration';
        btn.classList.remove('btn-primary');
        btn.classList.add('btn-danger');

        await showAlert('Calibration Failed', error.message);
    }
}

function updateCalibrationStatus(icon, title, detail) {
    document.getElementById('calibration-status-icon').textContent = icon;
    document.getElementById('calibration-status-text').textContent = title;
    document.getElementById('calibration-status-detail').textContent = detail;
}

function addCalibrationLog(message, type = '') {
    const log = document.getElementById('calibration-log');
    const entry = document.createElement('div');
    entry.className = 'log-entry' + (type ? ' ' + type : '');
    entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    log.appendChild(entry);
    log.scrollTop = log.scrollHeight;
}

// ============================================================================
// Step 4: Review
// ============================================================================

function setupReviewStepListeners() {
    document.getElementById('save-calibration').addEventListener('click', saveCalibration);
    document.getElementById('export-calibration').addEventListener('click', exportCalibration);
    document.getElementById('recalibrate').addEventListener('click', recalibrate);
}

// Update review panel when entering step 4
async function updateReviewPanel() {
    if (!calibrationData.results) return;

    const results = calibrationData.results;

    // Update metrics
    document.getElementById('result-error').textContent =
        typeof results.reprojection_error === 'number'
            ? results.reprojection_error.toFixed(4)
            : results.reprojection_error;
    document.getElementById('result-images').textContent = results.images_used;

    // Determine quality based on reprojection error
    const error = parseFloat(results.reprojection_error);
    let quality, qualityColor;
    if (error < 0.3) {
        quality = 'Excellent';
        qualityColor = '#28a745';
    } else if (error < 0.5) {
        quality = 'Very Good';
        qualityColor = '#5cb85c';
    } else if (error < 0.8) {
        quality = 'Good';
        qualityColor = '#ffc107';
    } else if (error < 1.5) {
        quality = 'Fair';
        qualityColor = '#fd7e14';
    } else {
        quality = 'Poor';
        qualityColor = '#dc3545';
    }

    const qualityEl = document.getElementById('result-quality');
    qualityEl.textContent = quality;
    qualityEl.style.color = qualityColor;
    qualityEl.style.fontWeight = 'bold';

    // Update camera matrix
    const matrixHTML = results.camera_matrix
        .map(row => row.map(v => parseFloat(v).toFixed(2).padStart(10)).join('  '))
        .join('\n');
    document.getElementById('camera-matrix').innerHTML = `<code>${matrixHTML}</code>`;

    // Update distortion coefficients
    const distHTML = results.distortion_coeffs
        .map(v => parseFloat(v).toFixed(6).padStart(12))
        .join('\n');
    document.getElementById('distortion-coeffs').innerHTML = `<code>${distHTML}</code>`;

    // Set default calibration name using session name
    const timestamp = new Date().toISOString().slice(0, 10).replace(/-/g, '');
    const sessionPart = calibrationData.session_name ?
        calibrationData.session_name.replace('session_', '') :
        timestamp;
    document.getElementById('calibration-name').value =
        `camera_${calibrationData.camera_id}_${sessionPart}`;

    // Display per-image errors if available
    if (results.image_errors && results.image_errors.length > 0) {
        const errorContainer = document.getElementById('image-errors');
        const maxError = Math.max(...results.image_errors.map(img => img.error));
        const minError = Math.min(...results.image_errors.map(img => img.error));

        let html = `<div style="margin-bottom: 10px; padding: 8px; background: #f8f9fa; border-radius: 4px; font-size: 0.85em;">
            <strong>Error Range:</strong> ${minError.toFixed(4)} - ${maxError.toFixed(4)} pixels
        </div>`;

        html += '<div style="max-height: 200px; overflow-y: auto;">';
        html += '<table style="width: 100%; font-size: 0.85em; border-collapse: collapse;">';
        html += '<thead><tr style="background: #f0f4ff; border-bottom: 2px solid #ddd;">';
        html += '<th style="padding: 6px; text-align: left;">Image</th>';
        html += '<th style="padding: 6px; text-align: center;">Corners</th>';
        html += '<th style="padding: 6px; text-align: right;">Error (px)</th>';
        html += '</tr></thead><tbody>';

        results.image_errors.forEach((img, idx) => {
            const errorPercent = ((img.error - minError) / (maxError - minError)) * 100;
            let barColor = '#28a745'; // Green
            if (img.error > 1.0) barColor = '#dc3545'; // Red
            else if (img.error > 0.5) barColor = '#ffc107'; // Yellow

            html += `<tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 6px;">${img.filename}</td>
                <td style="padding: 6px; text-align: center;">${img.corners}</td>
                <td style="padding: 6px; text-align: right;">
                    <div style="display: flex; align-items: center; justify-content: flex-end; gap: 8px;">
                        <div style="width: 60px; height: 4px; background: #e9ecef; border-radius: 2px; overflow: hidden;">
                            <div style="width: ${errorPercent}%; height: 100%; background: ${barColor};"></div>
                        </div>
                        <span style="font-family: monospace; min-width: 50px;">${img.error.toFixed(4)}</span>
                    </div>
                </td>
            </tr>`;
        });

        html += '</tbody></table></div>';
        errorContainer.innerHTML = html;
    }

    // Populate image selector for comparison
    await populateImageSelector();
}

async function populateImageSelector() {
    const select = document.getElementById('preview-image-select');

    if (!calibrationData.results || !calibrationData.results.image_errors) {
        select.innerHTML = '<option value="">No images available</option>';
        return;
    }

    // Populate with available images
    select.innerHTML = '';
    calibrationData.results.image_errors.forEach((img, idx) => {
        const option = document.createElement('option');
        option.value = img.filename;
        option.textContent = `${img.filename} (${img.corners} corners, ${img.error.toFixed(4)}px error)`;
        select.appendChild(option);
    });

    // Load first image by default
    if (calibrationData.results.image_errors.length > 0) {
        select.value = calibrationData.results.image_errors[0].filename;
        await loadComparisonImages(calibrationData.results.image_errors[0].filename);
    }

    // Add event listener for image selection
    select.addEventListener('change', async (e) => {
        if (e.target.value) {
            await loadComparisonImages(e.target.value);
            updateImageNavigationButtons();
        }
    });

    // Setup previous/next buttons
    const prevBtn = document.getElementById('prev-image-btn');
    const nextBtn = document.getElementById('next-image-btn');

    if (prevBtn) {
        prevBtn.addEventListener('click', () => navigateImage(-1));
    }

    if (nextBtn) {
        nextBtn.addEventListener('click', () => navigateImage(1));
    }

    // Update button states
    updateImageNavigationButtons();
}

function navigateImage(direction) {
    const select = document.getElementById('preview-image-select');
    const currentIndex = select.selectedIndex;
    const newIndex = currentIndex + direction;

    if (newIndex >= 0 && newIndex < select.options.length) {
        select.selectedIndex = newIndex;
        select.dispatchEvent(new Event('change'));
    }
}

function updateImageNavigationButtons() {
    const select = document.getElementById('preview-image-select');
    const prevBtn = document.getElementById('prev-image-btn');
    const nextBtn = document.getElementById('next-image-btn');

    if (prevBtn) {
        prevBtn.disabled = select.selectedIndex === 0;
    }

    if (nextBtn) {
        nextBtn.disabled = select.selectedIndex === select.options.length - 1;
    }
}

async function loadComparisonImages(filename) {
    try {
        const response = await fetch('/api/calibration/undistort-image', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_name: calibrationData.session_name,
                camera_id: calibrationData.camera_id,
                image_filename: filename
            })
        });

        const result = await response.json();

        if (!response.ok || !result.success) {
            console.error('Failed to load comparison images:', result.error);
            return;
        }

        // Load images
        const originalImg = new Image();
        const undistortedImg = new Image();

        originalImg.onload = () => {
            const canvas = document.getElementById('original-image');
            const ctx = canvas.getContext('2d');

            // Set canvas size to match image aspect ratio
            const maxWidth = 500;
            const scale = maxWidth / originalImg.width;
            canvas.width = maxWidth;
            canvas.height = originalImg.height * scale;

            ctx.drawImage(originalImg, 0, 0, canvas.width, canvas.height);
        };

        undistortedImg.onload = () => {
            const canvas = document.getElementById('undistorted-image');
            const ctx = canvas.getContext('2d');

            // Set canvas size to match image aspect ratio
            const maxWidth = 500;
            const scale = maxWidth / undistortedImg.width;
            canvas.width = maxWidth;
            canvas.height = undistortedImg.height * scale;

            ctx.drawImage(undistortedImg, 0, 0, canvas.width, canvas.height);
        };

        originalImg.src = result.original;
        undistortedImg.src = result.undistorted;

    } catch (error) {
        console.error('Error loading comparison images:', error);
    }
}

async function saveCalibration() {
    const name = document.getElementById('calibration-name').value;
    if (!name) {
        await showAlert('Name Required', 'Please enter a calibration name.');
        return;
    }

    const saveImages = document.getElementById('save-images').checked;
    const applyImmediately = document.getElementById('apply-immediately').checked;

    // Placeholder: In real implementation, this would call backend API
    console.log('Saving calibration:', {
        name,
        camera_id: calibrationData.camera_id,
        results: calibrationData.results,
        save_images: saveImages,
        apply_immediately: applyImmediately
    });

    await showAlert('Success', 'Calibration saved successfully!');
    closeCalibrationWizard();
}

function exportCalibration() {
    // Placeholder: Export calibration data as JSON
    const exportData = {
        camera_id: calibrationData.camera_id,
        pattern: calibrationData.pattern,
        results: calibrationData.results,
        timestamp: new Date().toISOString()
    };

    const dataStr = JSON.stringify(exportData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);

    const link = document.createElement('a');
    link.href = url;
    link.download = `calibration_camera_${calibrationData.camera_id}_${Date.now()}.json`;
    link.click();

    URL.revokeObjectURL(url);
}

async function recalibrate() {
    const confirmed = await showConfirm(
        'Recalibrate',
        'Return to capture step to recalibrate? Current results will be discarded.'
    );
    if (confirmed) {
        showWizardStep(2);
    }
}

function finishCalibration() {
    closeCalibrationWizard();
}

// Initialize calibration wizard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Wait a bit to ensure other initialization is complete
    setTimeout(initializeCalibrationWizard, 100);
    setTimeout(initializePanoramaWizard, 100);
});


// ===== Image Processing Controls =====

async function toggleProcessing(cameraId, enabled) {
    const panel = document.getElementById(`processing-panel-${cameraId}`);
    if (panel) {
        panel.style.display = enabled ? 'block' : 'none';
    }

    // Save state to localStorage
    const processingState = JSON.parse(localStorage.getItem('processingState') || '{}');
    processingState[cameraId] = processingState[cameraId] || {};
    processingState[cameraId].enabled = enabled;
    localStorage.setItem('processingState', JSON.stringify(processingState));

    if (!enabled) {
        // Disable processing on backend
        try {
            const response = await fetch(`/api/processing/${cameraId}/disable`, {
                method: 'POST'
            });
            if (!response.ok) {
                console.error('Failed to disable processing');
            }
        } catch (error) {
            console.error('Error disabling processing:', error);
        }
    } else {
        // Re-apply saved processing when enabling
        const savedConfig = processingState[cameraId];
        if (savedConfig && savedConfig.type) {
            // Update UI to match saved type
            const typeSelector = document.getElementById(`processing-type-${cameraId}`);
            if (typeSelector && savedConfig.type) {
                typeSelector.value = savedConfig.type;
                await updateProcessingType(cameraId, savedConfig.type);
            }
            // Apply processing after a short delay to ensure UI is updated
            setTimeout(() => applyProcessing(cameraId, true), 100);
        }
    }
}

function updateProcessingType(cameraId, type) {
    const paramsContainer = document.getElementById(`processing-params-${cameraId}`);
    if (!paramsContainer) return;

    // Save type to localStorage
    const processingState = JSON.parse(localStorage.getItem('processingState') || '{}');
    processingState[cameraId] = processingState[cameraId] || {};
    processingState[cameraId].type = type;
    localStorage.setItem('processingState', JSON.stringify(processingState));

    let html = '';

    switch (type) {
        case 'undistort':
            // Get saved state
            const savedState = JSON.parse(localStorage.getItem('processingState') || '{}')[cameraId] || {};
            const savedAlpha = savedState.alpha || 0;
            const savedCalibrationFile = savedState.calibration_file || '';

            html = `
                <div class="control-item">
                    <label>Calibration File</label>
                    <select id="processing-calibration-file-${cameraId}" onchange="applyProcessing('${cameraId}', true)">
                        <option value="">Select calibration...</option>
                    </select>
                    <button class="btn btn-secondary" onclick="loadCalibrationFiles('${cameraId}')" style="margin-top: 5px; width: 100%;">
                        🔄 Refresh Files
                    </button>
                </div>
                <div class="control-item">
                    <label>
                        Alpha (Crop Factor)
                        <span class="control-value" id="processing-alpha-value-${cameraId}">${savedAlpha}</span>
                    </label>
                    <input type="range" id="processing-alpha-${cameraId}"
                           min="0" max="1" step="0.1" value="${savedAlpha}"
                           oninput="document.getElementById('processing-alpha-value-${cameraId}').textContent = this.value; applyProcessing('${cameraId}', true)">
                    <small style="color: #888;">0 = crop invalid pixels, 1 = keep all pixels</small>
                </div>
            `;
            // Load calibration files and restore selection
            setTimeout(() => {
                loadCalibrationFiles(cameraId).then(() => {
                    const select = document.getElementById(`processing-calibration-file-${cameraId}`);
                    if (select && savedCalibrationFile) {
                        select.value = savedCalibrationFile;
                    }
                });
            }, 100);
            break;

        case 'perspective':
            html = `
                <div class="control-item">
                    <label>Source Points (x1,y1 x2,y2 x3,y3 x4,y4)</label>
                    <input type="text" id="processing-src-points-${cameraId}"
                           placeholder="0,0 100,0 100,100 0,100"
                           onchange="applyProcessing('${cameraId}', true)">
                </div>
                <div class="control-item">
                    <label>Destination Points (x1,y1 x2,y2 x3,y3 x4,y4)</label>
                    <input type="text" id="processing-dst-points-${cameraId}"
                           placeholder="0,0 100,0 100,100 0,100"
                           onchange="applyProcessing('${cameraId}', true)">
                </div>
                <button class="btn btn-secondary" onclick="selectPointsInteractive('${cameraId}')" style="width: 100%; margin-top: 5px;">
                    🎯 Select Points on Image
                </button>
            `;
            break;

        case 'affine':
            html = `
                <div class="control-item">
                    <label>Source Points (x1,y1 x2,y2 x3,y3)</label>
                    <input type="text" id="processing-src-points-${cameraId}"
                           placeholder="0,0 100,0 0,100"
                           onchange="applyProcessing('${cameraId}', true)">
                </div>
                <div class="control-item">
                    <label>Destination Points (x1,y1 x2,y2 x3,y3)</label>
                    <input type="text" id="processing-dst-points-${cameraId}"
                           placeholder="0,0 100,0 0,100"
                           onchange="applyProcessing('${cameraId}', true)">
                </div>
            `;
            break;

        case 'rotation':
            const savedRotState = JSON.parse(localStorage.getItem('processingState') || '{}')[cameraId] || {};
            const savedAngle = savedRotState.angle || 0;
            const savedScale = savedRotState.scale || 1.0;

            html = `
                <div class="control-item">
                    <label>
                        Rotation Angle (degrees)
                        <span class="control-value" id="processing-angle-value-${cameraId}">${savedAngle}</span>
                    </label>
                    <input type="range" id="processing-angle-${cameraId}"
                           min="-180" max="180" step="1" value="${savedAngle}"
                           oninput="document.getElementById('processing-angle-value-${cameraId}').textContent = this.value; applyProcessing('${cameraId}', true)">
                </div>
                <div class="control-item">
                    <label>
                        Scale
                        <span class="control-value" id="processing-scale-value-${cameraId}">${savedScale}</span>
                    </label>
                    <input type="range" id="processing-scale-${cameraId}"
                           min="0.1" max="3.0" step="0.1" value="${savedScale}"
                           oninput="document.getElementById('processing-scale-value-${cameraId}').textContent = this.value; applyProcessing('${cameraId}', true)">
                </div>
            `;
            break;

        case 'custom_matrix':
            html = `
                <div class="control-item">
                    <label>Matrix Type</label>
                    <select id="processing-matrix-type-${cameraId}" onchange="applyProcessing('${cameraId}', true)">
                        <option value="perspective">3x3 Perspective</option>
                        <option value="affine">2x3 Affine</option>
                    </select>
                </div>
                <div class="control-item">
                    <label>Matrix Values (comma-separated)</label>
                    <textarea id="processing-matrix-${cameraId}"
                              rows="3"
                              placeholder="1,0,0,0,1,0,0,0,1"
                              style="width: 100%; font-family: monospace;"
                              onchange="applyProcessing('${cameraId}', true)"></textarea>
                    <small style="color: #888;">Perspective: 9 values (3x3), Affine: 6 values (2x3)</small>
                </div>
            `;
            break;
    }

    paramsContainer.innerHTML = html;
}

async function loadCalibrationFiles(cameraId) {
    try {
        const response = await fetch('/api/calibration/sessions');
        const data = await response.json();

        const select = document.getElementById(`processing-calibration-file-${cameraId}`);
        if (!select) return;

        // Clear existing options
        select.innerHTML = '<option value="">Select calibration...</option>';

        // Get session details from response
        const sessions = data.session_details || [];

        // Add sessions that have calibration results for this camera
        for (const session of sessions) {
            if (session.calibrated && session.camera_id === cameraId.toString()) {
                // Construct path: data/calibration/session_name/camera_id/calibration_results.json
                const calibrationFile = `data/calibration/${session.path}/${session.camera_id}/calibration_results.json`;
                const option = document.createElement('option');
                option.value = calibrationFile;
                option.textContent = `Camera ${session.camera_id} - ${session.timestamp} (${session.num_images_used || session.num_images} images, error: ${session.reprojection_error?.toFixed(4) || 'N/A'})`;
                select.appendChild(option);
            }
        }
    } catch (error) {
        console.error('Error loading calibration files:', error);
    }
}

async function applyProcessing(cameraId, silent = false) {
    const enabled = document.getElementById(`processing-enable-${cameraId}`).checked;
    if (!enabled) {
        if (!silent) {
            await showAlert('Processing Disabled', 'Enable processing first.');
        }
        return;
    }

    const type = document.getElementById(`processing-type-${cameraId}`).value;
    const config = { type };

    // Get processing state for saving
    const processingState = JSON.parse(localStorage.getItem('processingState') || '{}');
    processingState[cameraId] = processingState[cameraId] || {};
    processingState[cameraId].type = type;

    try {
        switch (type) {
            case 'undistort':
                const calibrationFile = document.getElementById(`processing-calibration-file-${cameraId}`).value;
                if (!calibrationFile) {
                    if (!silent) {
                        await showAlert('No Calibration Selected', 'Please select a calibration file.');
                    }
                    return;
                }
                config.calibration_file = calibrationFile;
                config.alpha = parseFloat(document.getElementById(`processing-alpha-${cameraId}`).value);

                // Save to localStorage
                processingState[cameraId].calibration_file = calibrationFile;
                processingState[cameraId].alpha = config.alpha;
                break;

            case 'perspective':
                const srcPoints = document.getElementById(`processing-src-points-${cameraId}`).value;
                const dstPoints = document.getElementById(`processing-dst-points-${cameraId}`).value;
                config.src_points = parsePoints(srcPoints, 4);
                config.dst_points = parsePoints(dstPoints, 4);
                if (config.src_points.length !== 4 || config.dst_points.length !== 4) {
                    if (!silent) {
                        await showAlert('Invalid Points', 'Please provide exactly 4 points for each set.');
                    }
                    return;
                }
                break;

            case 'affine':
                const srcPointsAffine = document.getElementById(`processing-src-points-${cameraId}`).value;
                const dstPointsAffine = document.getElementById(`processing-dst-points-${cameraId}`).value;
                config.src_points = parsePoints(srcPointsAffine, 3);
                config.dst_points = parsePoints(dstPointsAffine, 3);
                if (config.src_points.length !== 3 || config.dst_points.length !== 3) {
                    if (!silent) {
                        await showAlert('Invalid Points', 'Please provide exactly 3 points for each set.');
                    }
                    return;
                }
                break;

            case 'rotation':
                config.angle = parseFloat(document.getElementById(`processing-angle-${cameraId}`).value);
                config.scale = parseFloat(document.getElementById(`processing-scale-${cameraId}`).value);

                // Save to localStorage
                processingState[cameraId].angle = config.angle;
                processingState[cameraId].scale = config.scale;
                break;

            case 'custom_matrix':
                const matrixType = document.getElementById(`processing-matrix-type-${cameraId}`).value;
                const matrixStr = document.getElementById(`processing-matrix-${cameraId}`).value;
                config.matrix = matrixStr.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v));

                const expectedLength = matrixType === 'perspective' ? 9 : 6;
                if (config.matrix.length !== expectedLength) {
                    if (!silent) {
                        await showAlert('Invalid Matrix', `Please provide exactly ${expectedLength} values.`);
                    }
                    return;
                }
                break;
        }

        // Send to backend
        const response = await fetch(`/api/processing/${cameraId}/enable`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        });

        if (response.ok) {
            // Save state after successful apply
            localStorage.setItem('processingState', JSON.stringify(processingState));

            if (!silent) {
                await showAlert('Success', 'Processing applied successfully!');
            }
        } else {
            const error = await response.json();
            if (!silent) {
                await showAlert('Error', error.error || 'Failed to apply processing');
            } else {
                console.error('Processing error:', error);
            }
        }
    } catch (error) {
        console.error('Error applying processing:', error);
        if (!silent) {
            await showAlert('Error', 'Failed to apply processing: ' + error.message);
        }
    }
}

function parsePoints(str, expectedCount) {
    // Parse "x1,y1 x2,y2 ..." format
    const points = [];
    const parts = str.trim().split(/\s+/);

    for (const part of parts) {
        const coords = part.split(',').map(v => parseFloat(v.trim()));
        if (coords.length === 2 && !isNaN(coords[0]) && !isNaN(coords[1])) {
            points.push(coords);
        }
    }

    return points;
}

function selectPointsInteractive(cameraId) {
    // TODO: Implement interactive point selection on camera preview
    showAlert('Not Implemented', 'Interactive point selection coming soon. Please enter points manually.');
}


// ============================================================================
// Panorama Calibration
// ============================================================================

let panoramaPreviewInterval = null;
let panoramaCalibrationResult = null;
let panoramaSessionId = null;
let panoramaCaptureCount = 0;
let panoramaCaptureList = [];
let panoramaAutoCapture = false;
let panoramaAutoCaptureInterval = null;
let panoramaLastBoardDetection = { cam1: false, cam2: false };
let panoramaOverlayEnabled = false;

function initializePanoramaWizard() {
    const panoramaBtn = document.getElementById('panorama-btn');
    const panoramaModal = document.getElementById('panorama-modal');
    const closeBtn = panoramaModal.querySelector('.close');

    // Open panorama wizard
    panoramaBtn.addEventListener('click', openPanoramaCalibration);

    // Close modal
    closeBtn.addEventListener('click', closePanoramaCalibration);

    // Session selection listener
    document.getElementById('panorama-session-select').addEventListener('change', onPanoramaSessionChange);

    // Camera selection listeners - both trigger the same update function
    document.getElementById('panorama-camera1-select').addEventListener('change', updatePanoramaPreview);
    document.getElementById('panorama-camera2-select').addEventListener('change', updatePanoramaPreview);

    // Auto-capture toggle
    document.getElementById('toggle-panorama-auto-capture').addEventListener('click', togglePanoramaAutoCapture);
}

function openPanoramaCalibration() {
    const modal = document.getElementById('panorama-modal');
    modal.classList.add('show');

    // Populate camera selects
    populatePanoramaCameraSelects();

    // Load available sessions
    loadPanoramaSessions();

    // Reset state
    panoramaCalibrationResult = null;
    panoramaSessionId = null;
    panoramaCaptureCount = 0;
    panoramaCaptureList = [];
    panoramaAutoCapture = false;
    panoramaLastBoardDetection = { cam1: false, cam2: false };
    if (panoramaAutoCaptureInterval) {
        clearInterval(panoramaAutoCaptureInterval);
        panoramaAutoCaptureInterval = null;
    }
    document.getElementById('panorama-results').style.display = 'none';
    document.getElementById('panorama-test-btn').style.display = 'none';
    document.getElementById('panorama-compute-btn').style.display = 'none';
    document.getElementById('panorama-capture-list').innerHTML = '';
    document.getElementById('toggle-panorama-auto-capture').classList.remove('active');
    updatePanoramaCaptureStatus();
}

async function loadPanoramaSessions() {
    try {
        const response = await fetch(`${API_BASE}/api/calibration/sessions`);
        const data = await response.json();

        const select = document.getElementById('panorama-session-select');
        select.innerHTML = '<option value="">New Session</option>';

        if (data.success && data.sessions) {
            // Filter for panorama sessions
            const panoramaSessions = data.sessions.filter(s => s.startsWith('panorama_'));
            panoramaSessions.sort().reverse(); // Most recent first

            panoramaSessions.forEach(session => {
                const option = document.createElement('option');
                option.value = session;
                option.textContent = session;
                select.appendChild(option);
            });
        }
    } catch (error) {
        console.error('Error loading sessions:', error);
    }
}

async function onPanoramaSessionChange() {
    const sessionName = document.getElementById('panorama-session-select').value;
    const sessionNameInput = document.getElementById('panorama-session-name');

    if (!sessionName) {
        // New session
        sessionNameInput.value = '';
        sessionNameInput.placeholder = 'Auto-generated';
        sessionNameInput.disabled = false;
        panoramaSessionId = null;
        panoramaCaptureCount = 0;
        panoramaCaptureList = [];
        document.getElementById('panorama-capture-list').innerHTML = '';
        updatePanoramaCaptureStatus();
        return;
    }

    // Load existing session
    sessionNameInput.value = sessionName;
    sessionNameInput.disabled = true;
    panoramaSessionId = sessionName;

    try {
        // Load session metadata
        const response = await fetch(`${API_BASE}/api/calibration/session/${sessionName}`);
        const data = await response.json();

        if (data.success && data.session) {
            // Populate settings from session
            const metadata = data.session.metadata;
            if (metadata) {
                if (metadata.camera1_id || metadata.camera_id) {
                    document.getElementById('panorama-camera1-select').value = metadata.camera1_id || metadata.camera_id;
                }
                if (metadata.camera2_id) {
                    document.getElementById('panorama-camera2-select').value = metadata.camera2_id;
                }
                if (metadata.board_config) {
                    document.getElementById('panorama-board-width').value = metadata.board_config.width || 8;
                    document.getElementById('panorama-board-height').value = metadata.board_config.height || 5;
                    document.getElementById('panorama-square-length').value = metadata.board_config.square_length || 50;
                    document.getElementById('panorama-marker-length').value = metadata.board_config.marker_length || 37;
                    document.getElementById('panorama-aruco-dictionary').value = metadata.board_config.dictionary || 'DICT_6X6_100';
                }
            }

            // Count images
            const imageCount = data.session.images ? data.session.images.length : 0;
            panoramaCaptureCount = imageCount;
            updatePanoramaCaptureStatus();

            await showAlert('Session Loaded', `Loaded session "${sessionName}" with ${imageCount} captures`);
        }
    } catch (error) {
        console.error('Error loading session:', error);
        await showAlert('Error', 'Failed to load session');
    }
}

function closePanoramaCalibration() {
    const modal = document.getElementById('panorama-modal');
    modal.classList.remove('show');

    // Stop preview stream by clearing src
    const preview = document.getElementById('panorama-camera-preview-composite');
    if (preview) preview.src = '';

    // Stop auto-capture
    if (panoramaAutoCaptureInterval) {
        clearInterval(panoramaAutoCaptureInterval);
        panoramaAutoCaptureInterval = null;
    }
    panoramaAutoCapture = false;

    // Clear interval (though not needed for streams)
    if (panoramaPreviewInterval) {
        clearInterval(panoramaPreviewInterval);
        panoramaPreviewInterval = null;
    }
}

function populatePanoramaCameraSelects() {
    const select1 = document.getElementById('panorama-camera1-select');
    const select2 = document.getElementById('panorama-camera2-select');

    // Clear and populate
    select1.innerHTML = '<option value="">Choose camera...</option>';
    select2.innerHTML = '<option value="">Choose camera...</option>';

    cameras.forEach(camera => {
        const option1 = document.createElement('option');
        option1.value = camera.id;
        option1.textContent = camera.name || camera.id;
        select1.appendChild(option1);

        const option2 = document.createElement('option');
        option2.value = camera.id;
        option2.textContent = camera.name || camera.id;
        select2.appendChild(option2);
    });
}

async function updatePanoramaPreview() {
    const camera1Id = document.getElementById('panorama-camera1-select').value;
    const camera2Id = document.getElementById('panorama-camera2-select').value;
    const preview = document.getElementById('panorama-camera-preview-composite');
    const placeholder = document.getElementById('panorama-preview-placeholder');

    if (panoramaPreviewInterval) {
        clearInterval(panoramaPreviewInterval);
        panoramaPreviewInterval = null;
    }

    const testCaptureBtn = document.getElementById('panorama-test-capture-btn');
    const overlayBtn = document.getElementById('panorama-overlay-toggle-btn');

    if (!camera1Id || !camera2Id) {
        preview.style.display = 'none';
        placeholder.style.display = 'flex';
        preview.src = '';
        testCaptureBtn.style.display = 'none';
        overlayBtn.style.display = 'none';
        return;
    }

    // Both cameras selected - initialize synchronized pair (which starts the cameras)
    await initializePanoramaPair();

    // Wait a bit for streams to start
    await new Promise(resolve => setTimeout(resolve, 500));

    // Show composite stream with both cameras side-by-side
    updatePanoramaPreviewStream();
    preview.style.display = 'block';
    placeholder.style.display = 'none';
    testCaptureBtn.style.display = 'inline-block';
    overlayBtn.style.display = 'inline-block';
}

function updatePanoramaPreviewStream() {
    const camera1Id = document.getElementById('panorama-camera1-select').value;
    const camera2Id = document.getElementById('panorama-camera2-select').value;
    const preview = document.getElementById('panorama-camera-preview-composite');

    if (!camera1Id || !camera2Id) return;

    if (panoramaOverlayEnabled) {
        // Get board config for overlay
        const boardConfig = {
            board_width: document.getElementById('panorama-board-width').value,
            board_height: document.getElementById('panorama-board-height').value,
            square_length: document.getElementById('panorama-square-length').value,
            marker_length: document.getElementById('panorama-marker-length').value,
            dictionary: document.getElementById('panorama-aruco-dictionary').value
        };

        const params = new URLSearchParams(boardConfig);
        preview.src = `${API_BASE}/api/cameras/composite/${camera1Id}/${camera2Id}/stream-with-overlay?${params}&t=${Date.now()}`;
    } else {
        preview.src = `${API_BASE}/api/cameras/composite/${camera1Id}/${camera2Id}/stream?t=${Date.now()}`;
    }
}

function togglePanoramaOverlay() {
    panoramaOverlayEnabled = !panoramaOverlayEnabled;
    const btn = document.getElementById('panorama-overlay-toggle-btn');

    if (panoramaOverlayEnabled) {
        btn.classList.add('active');
        btn.textContent = '👁️ Hide Detection';
    } else {
        btn.classList.remove('active');
        btn.textContent = '👁️ Show Detection';
    }

    updatePanoramaPreviewStream();
}

async function initializePanoramaPair() {
    const camera1Id = document.getElementById('panorama-camera1-select').value;
    const camera2Id = document.getElementById('panorama-camera2-select').value;

    // Only initialize if both cameras are selected
    if (!camera1Id || !camera2Id) {
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/api/calibration/panorama/init-pair`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                camera1_id: camera1Id,
                camera2_id: camera2Id
            })
        });

        const result = await response.json();
        if (!result.success) {
            console.error('Failed to initialize synchronized pair:', result.error);
        }
    } catch (error) {
        console.error('Error initializing synchronized pair:', error);
    }
}

async function capturePanoramaCalibration() {
    const camera1Id = document.getElementById('panorama-camera1-select').value;
    const camera2Id = document.getElementById('panorama-camera2-select').value;

    if (!camera1Id || !camera2Id) {
        await showAlert('Error', 'Please select both cameras');
        return;
    }

    if (camera1Id === camera2Id) {
        await showAlert('Error', 'Please select different cameras');
        return;
    }

    // Initialize session ID if needed
    if (!panoramaSessionId) {
        const sessionName = document.getElementById('panorama-session-name').value.trim();
        panoramaSessionId = sessionName || null; // Will be auto-generated by backend if null
    }

    // Get board config
    const boardConfig = {
        width: parseInt(document.getElementById('panorama-board-width').value),
        height: parseInt(document.getElementById('panorama-board-height').value),
        square_length: parseFloat(document.getElementById('panorama-square-length').value),
        marker_length: parseFloat(document.getElementById('panorama-marker-length').value),
        dictionary: document.getElementById('panorama-aruco-dictionary').value
    };

    // Update status
    const statusEl = document.getElementById('panorama-detection-status');
    statusEl.innerHTML = '<span class="status-icon">⏳</span><span class="status-message">Capturing image pair...</span>';

    const captureBtn = document.getElementById('panorama-capture-btn');
    captureBtn.disabled = true;

    try {
        const response = await fetch(`${API_BASE}/api/calibration/panorama/capture`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                camera1_id: camera1Id,
                camera2_id: camera2Id,
                board_config: boardConfig,
                session_id: panoramaSessionId
            })
        });

        const result = await response.json();

        if (result.success) {
            // Update session ID from response (in case it was auto-generated)
            panoramaSessionId = result.session_id;
            document.getElementById('panorama-session-name').value = result.session_id;

            panoramaCaptureCount = result.capture_count;
            panoramaCaptureList.push(result);

            // Update UI
            updatePanoramaCaptureList();
            updatePanoramaCaptureStatus();

            if (result.detected_both) {
                statusEl.innerHTML = `<span class="status-icon">✅</span><span class="status-message">Capture ${result.capture_count}: ${result.matches} matches found</span>`;
            } else {
                statusEl.innerHTML = `<span class="status-icon">⚠️</span><span class="status-message">Capture ${result.capture_count}: Board not detected in both cameras</span>`;
            }

            // Show compute button after first successful capture
            if (panoramaCaptureCount >= 1) {
                document.getElementById('panorama-compute-btn').style.display = 'inline-block';
            }
        } else {
            statusEl.innerHTML = `<span class="status-icon">❌</span><span class="status-message">Error: ${result.error}</span>`;
            await showAlert('Capture Failed', result.error);
        }
    } catch (error) {
        console.error('Panorama capture error:', error);
        statusEl.innerHTML = `<span class="status-icon">❌</span><span class="status-message">Error: ${error.message}</span>`;
        await showAlert('Error', 'Failed to capture: ' + error.message);
    } finally {
        captureBtn.disabled = false;
    }
}

function updatePanoramaCaptureStatus() {
    const statusText = document.getElementById('panorama-capture-status-text');
    if (statusText) {
        const successCount = panoramaCaptureList.filter(c => c.detected_both).length;
        statusText.textContent = `${panoramaCaptureCount} captures (${successCount} with board detected in both cameras)`;
    }
}

function updatePanoramaCaptureList() {
    const listEl = document.getElementById('panorama-capture-list');
    if (!listEl) return;

    listEl.innerHTML = panoramaCaptureList.map((capture, idx) => {
        const icon = capture.detected_both ? '✅' : '⚠️';
        const status = capture.detected_both
            ? `${capture.matches} matches`
            : 'Board not detected';
        return `<div class="capture-item">
            <span>${icon} Capture ${idx + 1}</span>
            <span class="capture-detail">${status}</span>
        </div>`;
    }).join('');
}

async function computePanoramaCalibration() {
    const useCalibration = document.getElementById('panorama-use-calibration').checked;
    const statusEl = document.getElementById('panorama-detection-status');

    statusEl.innerHTML = '<span class="status-icon">⏳</span><span class="status-message">Computing homography from all captures...</span>';

    const computeBtn = document.getElementById('panorama-compute-btn');
    computeBtn.disabled = true;

    try {
        const response = await fetch(`${API_BASE}/api/calibration/panorama/compute`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: panoramaSessionId,
                use_calibration: useCalibration
            })
        });

        const result = await response.json();

        if (result.success) {
            panoramaCalibrationResult = result;
            displayPanoramaResults(result);
            statusEl.innerHTML = '<span class="status-icon">✅</span><span class="status-message">Calibration successful!</span>';
            document.getElementById('panorama-test-btn').style.display = 'inline-block';
        } else {
            statusEl.innerHTML = `<span class="status-icon">❌</span><span class="status-message">Error: ${result.error}</span>`;
            await showAlert('Calibration Failed', result.error);
        }
    } catch (error) {
        console.error('Panorama calibration error:', error);
        statusEl.innerHTML = `<span class="status-icon">❌</span><span class="status-message">Error: ${error.message}</span>`;
        await showAlert('Error', 'Failed to compute calibration: ' + error.message);
    } finally {
        computeBtn.disabled = false;
    }
}

async function resetPanoramaSession() {
    if (panoramaSessionId) {
        await fetch(`${API_BASE}/api/calibration/panorama/reset`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: panoramaSessionId })
        });
    }

    panoramaSessionId = null;
    panoramaCaptureCount = 0;
    panoramaCaptureList = [];
    panoramaCalibrationResult = null;

    document.getElementById('panorama-results').style.display = 'none';
    document.getElementById('panorama-test-btn').style.display = 'none';
    document.getElementById('panorama-compute-btn').style.display = 'none';
    document.getElementById('panorama-capture-list').innerHTML = '';
    updatePanoramaCaptureStatus();

    const statusEl = document.getElementById('panorama-detection-status');
    statusEl.innerHTML = '<span class="status-icon">ℹ️</span><span class="status-message">Position ChArUco board visible to both cameras in overlapping region</span>';
}

async function captureFullResTestImage() {
    const camera1Id = document.getElementById('panorama-camera1-select').value;
    const camera2Id = document.getElementById('panorama-camera2-select').value;

    if (!camera1Id || !camera2Id) {
        await showAlert('Error', 'Please select both cameras');
        return;
    }

    const btn = document.getElementById('panorama-test-capture-btn');
    btn.disabled = true;
    const originalText = btn.textContent;
    btn.textContent = '⏳ Capturing...';

    try {
        const response = await fetch(`${API_BASE}/api/calibration/panorama/capture-full-res`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                camera1_id: camera1Id,
                camera2_id: camera2Id,
                format: 'composite'  // Get the full top-bottom composite
            })
        });

        const result = await response.json();

        if (result.success) {
            // Create a new window/tab to display the full-resolution image
            const newWindow = window.open('', '_blank');
            if (newWindow) {
                newWindow.document.write(`
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Full-Res Test Capture - ${camera1Id} & ${camera2Id}</title>
                        <style>
                            body {
                                margin: 0;
                                padding: 20px;
                                background: #1e1e1e;
                                color: #fff;
                                font-family: Arial, sans-serif;
                                text-align: center;
                            }
                            .info {
                                margin-bottom: 20px;
                                padding: 15px;
                                background: #2d2d2d;
                                border-radius: 5px;
                            }
                            .info h2 {
                                margin: 0 0 10px 0;
                            }
                            .info p {
                                margin: 5px 0;
                                color: #aaa;
                            }
                            img {
                                max-width: 100%;
                                height: auto;
                                border: 2px solid #444;
                                border-radius: 5px;
                            }
                            .download-btn {
                                display: inline-block;
                                margin: 10px;
                                padding: 10px 20px;
                                background: #007acc;
                                color: white;
                                text-decoration: none;
                                border-radius: 5px;
                                cursor: pointer;
                            }
                            .download-btn:hover {
                                background: #005a9e;
                            }
                        </style>
                    </head>
                    <body>
                        <div class="info">
                            <h2>🎯 Full-Resolution Synchronized Test Capture</h2>
                            <p><strong>Cameras:</strong> ${camera1Id} (top) & ${camera2Id} (bottom)</p>
                            <p><strong>Resolution:</strong> ${result.resolution.width} × ${result.resolution.height}</p>
                            <p><strong>Format:</strong> Top-Bottom Composite (Hardware Synchronized)</p>
                            <a class="download-btn" id="download-btn">📥 Download Image</a>
                        </div>
                        <img src="${result.composite}" id="test-image" alt="Full-Resolution Test Capture">
                        <script>
                            document.getElementById('download-btn').addEventListener('click', function() {
                                const link = document.createElement('a');
                                link.href = document.getElementById('test-image').src;
                                link.download = 'full_res_test_${camera1Id}_${camera2Id}_' + Date.now() + '.jpg';
                                link.click();
                            });
                        </script>
                    </body>
                    </html>
                `);
                newWindow.document.close();
            } else {
                // Fallback: download directly if popup was blocked
                const link = document.createElement('a');
                link.href = result.composite;
                link.download = `full_res_test_${camera1Id}_${camera2Id}_${Date.now()}.jpg`;
                link.click();
                await showAlert('Image Captured', 'Full-resolution image downloaded (popup blocked)');
            }
        } else {
            await showAlert('Capture Failed', result.error || 'Unknown error');
        }
    } catch (error) {
        console.error('Full-res capture error:', error);
        await showAlert('Error', 'Failed to capture: ' + error.message);
    } finally {
        btn.disabled = false;
        btn.textContent = originalText;
    }
}

function togglePanoramaAutoCapture() {
    panoramaAutoCapture = !panoramaAutoCapture;
    const btn = document.getElementById('toggle-panorama-auto-capture');

    if (panoramaAutoCapture) {
        btn.classList.add('active');
        startPanoramaAutoCapture();
    } else {
        btn.classList.remove('active');
        stopPanoramaAutoCapture();
    }
}

function startPanoramaAutoCapture() {
    const interval = parseInt(document.getElementById('panorama-auto-capture-interval').value) || 2;

    // Check for board detection every 500ms
    panoramaAutoCaptureInterval = setInterval(async () => {
        if (!panoramaAutoCapture) return;

        const camera1Id = document.getElementById('panorama-camera1-select').value;
        const camera2Id = document.getElementById('panorama-camera2-select').value;

        if (!camera1Id || !camera2Id) return;

        // Check if board is detected in both cameras
        try {
            const boardConfig = {
                width: parseInt(document.getElementById('panorama-board-width').value),
                height: parseInt(document.getElementById('panorama-board-height').value),
                square_length: parseFloat(document.getElementById('panorama-square-length').value),
                marker_length: parseFloat(document.getElementById('panorama-marker-length').value),
                dictionary: document.getElementById('panorama-aruco-dictionary').value
            };

            const response = await fetch(`${API_BASE}/api/calibration/panorama/check-detection`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    camera1_id: camera1Id,
                    camera2_id: camera2Id,
                    board_config: boardConfig
                })
            });

            const result = await response.json();

            if (result.success && result.detected_both) {
                // Auto-capture if board detected in both
                await capturePanoramaCalibration();

                // Wait for the interval before next capture
                await new Promise(resolve => setTimeout(resolve, interval * 1000));
            }
        } catch (error) {
            console.error('Auto-capture detection error:', error);
        }
    }, 500);
}

function stopPanoramaAutoCapture() {
    if (panoramaAutoCaptureInterval) {
        clearInterval(panoramaAutoCaptureInterval);
        panoramaAutoCaptureInterval = null;
    }
}

function displayPanoramaResults(result) {
    const resultsDiv = document.getElementById('panorama-results');
    const contentDiv = document.getElementById('panorama-results-content');

    const metrics = result.metrics;

    contentDiv.innerHTML = `
        <div class="calibration-metrics">
            <div class="metric-item">
                <span class="metric-label">Total Captures:</span>
                <span class="metric-value">${result.total_captures}</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">Successful Captures:</span>
                <span class="metric-value">${result.successful_captures}</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">Total Matches:</span>
                <span class="metric-value">${result.total_matches}</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">Inliers:</span>
                <span class="metric-value">${metrics.inliers} (${(metrics.inlier_ratio * 100).toFixed(1)}%)</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">Mean Error:</span>
                <span class="metric-value">${metrics.mean_error.toFixed(3)} px</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">RMSE:</span>
                <span class="metric-value">${metrics.rmse.toFixed(3)} px</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">Max Error:</span>
                <span class="metric-value">${metrics.max_error.toFixed(3)} px</span>
            </div>
        </div>
        <p style="margin-top: 15px; color: #666;">
            ${metrics.inlier_ratio > 0.7 && metrics.rmse < 2.0
            ? '✅ Good calibration quality! Ready for panorama stitching.'
            : '⚠️ Calibration quality could be improved. Try capturing more images with the board in different positions.'}
        </p>
    `;

    resultsDiv.style.display = 'block';
}

async function testPanoramaStitch() {
    const camera1Id = document.getElementById('panorama-camera1-select').value;
    const camera2Id = document.getElementById('panorama-camera2-select').value;

    const statusEl = document.getElementById('panorama-detection-status');
    statusEl.innerHTML = '<span class="status-icon">⏳</span><span class="status-message">Generating stitched preview...</span>';

    try {
        const response = await fetch(`${API_BASE}/api/calibration/panorama/stitch`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                camera1_id: camera1Id,
                camera2_id: camera2Id,
                blend_width: 50
            })
        });

        const result = await response.json();

        if (result.success) {
            const stitchImage = document.getElementById('panorama-stitch-image');
            stitchImage.src = result.image;
            statusEl.innerHTML = '<span class="status-icon">✅</span><span class="status-message">Stitch preview generated</span>';
        } else {
            statusEl.innerHTML = `<span class="status-icon">❌</span><span class="status-message">Error: ${result.error}</span>`;
            await showAlert('Stitch Failed', result.error);
        }
    } catch (error) {
        console.error('Stitch error:', error);
        statusEl.innerHTML = `<span class="status-icon">❌</span><span class="status-message">Error: ${error.message}</span>`;
        await showAlert('Error', 'Failed to stitch panorama: ' + error.message);
    }
}


