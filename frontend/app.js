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

    container.innerHTML = html;

    // Attach event listeners
    attachControlListeners(cameraId);
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
        <div class="feature-item" onclick="selectFeature('${feature.name}')">
            <strong>${feature.name}</strong>
            <br><small>${feature.description}</small>
        </div>
    `).join('');
}

function selectFeature(featureName) {
    // Future: Show feature UI
    console.log('Selected feature:', featureName);
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

