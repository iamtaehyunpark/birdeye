/**
 * app.js — Main application orchestrator.
 *
 * Wires together:
 *   WSManager         (ws.js)
 *   CameraManager     (camera.js)
 *   CamRenderer       (renderer.js)
 *   RefRenderer       (renderer.js)
 *   CalibrationManager (calibration.js)
 *
 * Flow:
 *   1. User uploads reference image → /api/upload/reference
 *   2. User selects & connects camera → getUserMedia
 *   3. User clicks Auto-Init or Manual Calibration → H computed
 *   4. WS streams frames → backend → JSON results → canvas overlay
 */

const API = `http://${location.hostname}:8000`;
const WS  = `ws://${location.hostname}:8000/ws/frames`;

// ── Module instances ──────────────────────────────────────────────────────────
const ws         = new WSManager(WS);
const camManager = new CameraManager(
  document.getElementById('camera-video'),
  document.getElementById('camera-canvas'),
  document.getElementById('cam-overlay-label'),
);
const camRenderer = new CamRenderer(document.getElementById('camera-canvas'));
const refRenderer = new RefRenderer(document.getElementById('ref-canvas'));

camManager.onDrawFrame = () => {
  camRenderer.draw();
};

let calibManager = null;

// ── App state ─────────────────────────────────────────────────────────────────
const APP = {
  hasRef:        false,
  cameraActive:  false,
  initStatus:    'idle',
  tracks:        [],
  refImageSrc:   null,
};

// ── Toast ─────────────────────────────────────────────────────────────────────
function toast(msg, type = 'info', duration = 4000) {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  document.getElementById('toast-container').appendChild(el);
  setTimeout(() => el.remove(), duration);
}

// ── Status badge ──────────────────────────────────────────────────────────────
function setInitStatus(status, info = null, error = null) {
  APP.initStatus = status;

  const badge = document.getElementById('status-badge');
  const label = document.getElementById('status-label');

  badge.dataset.status = status;
  label.textContent = {
    idle:    'IDLE',
    running: 'INITIALIZING…',
    done:    'READY',
    error:   'ERROR',
  }[status] ?? status.toUpperCase();

  // Init details box
  const box        = document.getElementById('init-status-box');
  const methodEl   = document.getElementById('init-method-label');
  const inliersEl  = document.getElementById('init-inliers-label');
  const errorEl    = document.getElementById('init-error-msg');

  box.style.display = (status !== 'idle') ? '' : 'none';

  if (info) {
    methodEl.textContent  = info.method ?? '—';
    inliersEl.textContent = info.n_inliers != null ? `${info.n_inliers} inliers` : '';
    const methodInfoEl = document.getElementById('ref-info-method');
    if (methodInfoEl) methodInfoEl.textContent = info.method ?? '';
  }

  if (error) {
    errorEl.style.display = '';
    errorEl.textContent   = error;
  } else {
    errorEl.style.display = 'none';
  }

  // Toggle step buttons
  _updateButtonStates();
}

// ── WS connection indicator ───────────────────────────────────────────────────
ws.on('open', () => {
  const ind = document.getElementById('ws-indicator');
  ind.classList.add('connected');
  ind.querySelector('.conn-label').textContent = 'Connected';
  toast('Backend connected', 'success', 2000);
});

ws.on('close', () => {
  const ind = document.getElementById('ws-indicator');
  ind.classList.remove('connected');
  ind.querySelector('.conn-label').textContent = 'Reconnecting…';
});

// ── WS messages ───────────────────────────────────────────────────────────────
ws.on('message', (data) => {
  if (!data) return;

  // Update FPS
  if (data.fps != null)
    document.getElementById('fps-display').textContent = data.fps.toFixed(1);

  // Update init status from server
  if (data.init_status && data.init_status !== APP.initStatus) {
    setInitStatus(data.init_status);
  }

  // Cache detections for continuous rendering inside camera's animation frame loop
  camRenderer.setDetections(data.detections, data.static_detections);

  // Count only moving objects in the metric display
  document.getElementById('det-count').textContent = data.detections?.length ?? 0;

  // Update reference canvas tracks
  if (data.tracks) {
    APP.tracks = data.tracks;
    refRenderer.setTracks(data.tracks);
    document.getElementById('track-count').textContent = data.tracks.length;
    _renderTrackList(data.tracks);
  }
});

// ── Reference image upload ────────────────────────────────────────────────────
const refInput   = document.getElementById('ref-file-input');
const uploadZone = document.getElementById('ref-upload-zone');

uploadZone.addEventListener('click', () => refInput.click());

uploadZone.addEventListener('dragover', e => {
  e.preventDefault();
  uploadZone.classList.add('drag-over');
});
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) _uploadReference(file);
});

refInput.addEventListener('change', () => {
  if (refInput.files[0]) _uploadReference(refInput.files[0]);
});

document.getElementById('btn-load-mock').addEventListener('click', () => {
  _loadMockScenario();
});

document.getElementById('btn-remove-ref').addEventListener('click', () => {
  _clearReferenceUI();
  fetch(`${API}/api/reset`, { method: 'POST' });
});

async function _loadMockScenario() {
  try {
    const res = await fetch(`${API}/api/init/mock-scenario`, { method: 'POST' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail ?? 'Load mock scenario failed');

    APP.hasRef      = true;
    APP.refImageSrc = `${API}/api/reference-image?t=${Date.now()}`;

    // Show thumbnail
    const thumb = document.getElementById('ref-preview-thumb');
    thumb.src   = APP.refImageSrc;
    document.getElementById('ref-preview-wrap').style.display = '';
    uploadZone.style.display = 'none';
    const mockWrap = document.getElementById('mock-scenario-wrap');
    if (mockWrap) mockWrap.style.display = 'none';

    // Load into ref renderer
    refRenderer.loadImage(APP.refImageSrc);
    document.getElementById('ref-empty-state').style.display = 'none';

    // Show ref info bar
    const bar = document.getElementById('ref-info-bar');
    bar.style.display = '';
    document.getElementById('ref-info-size').textContent = `${data.width} × ${data.height}`;

    if (data.homography_loaded) {
      // Pre-computed H was found — system is immediately ready to track
      setInitStatus('done', { method: 'mock_precomputed', n_inliers: 4 });
      toast(`Mock scenario ready — H pre-loaded (${data.width}×${data.height})`, 'success', 5000);
    } else {
      // No pre-computed H — user must calibrate before tracking
      setInitStatus('idle');
      _updateButtonStates();
      toast(`Mock scenario loaded (${data.width}×${data.height}) — calibrate to begin tracking`, 'info');
    }
  } catch (err) {
    toast(`Failed to load mock scenario: ${err.message}`, 'error');
  }
}

async function _uploadReference(file) {
  const fd = new FormData();
  fd.append('file', file);

  uploadZone.querySelector('.upload-hint').textContent = 'Uploading…';
  try {
    const res  = await fetch(`${API}/api/upload/reference`, { method: 'POST', body: fd });
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail ?? 'Upload failed');

    APP.hasRef       = true;
    APP.refImageSrc  = `${API}/api/reference-image?t=${Date.now()}`;

    // Show thumbnail
    const thumb = document.getElementById('ref-preview-thumb');
    thumb.src   = URL.createObjectURL(file);
    document.getElementById('ref-preview-wrap').style.display = '';
    uploadZone.style.display = 'none';
    const mockWrap = document.getElementById('mock-scenario-wrap');
    if (mockWrap) mockWrap.style.display = 'none';

    // Load into ref renderer
    refRenderer.loadImage(APP.refImageSrc);
    document.getElementById('ref-empty-state').style.display = 'none';

    // Show ref info bar
    const bar = document.getElementById('ref-info-bar');
    bar.style.display = '';
    document.getElementById('ref-info-size').textContent = `${data.width} × ${data.height}`;

    setInitStatus('idle');
    _updateButtonStates();
    toast(`Reference image loaded (${data.width}×${data.height})`, 'success');

  } catch (err) {
    toast(`Upload failed: ${err.message}`, 'error');
    uploadZone.querySelector('.upload-hint').textContent = 'Upload failed — try again';
  }
}

function _clearReferenceUI() {
  APP.hasRef      = false;
  APP.refImageSrc = null;
  document.getElementById('ref-preview-wrap').style.display = 'none';
  uploadZone.style.display = '';
  uploadZone.querySelector('.upload-hint').textContent = 'No image loaded';
  const mockWrap = document.getElementById('mock-scenario-wrap');
  if (mockWrap) mockWrap.style.display = '';
  document.getElementById('ref-empty-state').style.display = '';
  document.getElementById('ref-info-bar').style.display = 'none';
  setInitStatus('idle');
}

// ── Camera setup ──────────────────────────────────────────────────────────────
const camSelect  = document.getElementById('camera-select');
const btnConnect = document.getElementById('btn-camera-connect');
const btnStop    = document.getElementById('btn-camera-stop');
const btnRefresh = document.getElementById('btn-camera-refresh');

async function _populateCameras() {
  try {
    const devices = await camManager.enumerateCameras();
    camSelect.innerHTML = '<option value="">— Select camera —</option>';
    
    // Add mock option
    const mockOpt = document.createElement('option');
    mockOpt.value = '__mock__';
    mockOpt.text = '🎥 Mock Video Feed (Simulated Runway)';
    camSelect.appendChild(mockOpt);

    devices.forEach(d => {
      const opt   = document.createElement('option');
      opt.value   = d.deviceId;
      opt.text    = d.label || `Camera ${camSelect.options.length}`;
      camSelect.appendChild(opt);
    });
    
    camSelect.disabled = false;
    camSelect.value = '__mock__';
  } catch (err) {
    camSelect.innerHTML = '<option value="">— Select camera —</option>';
    const mockOpt = document.createElement('option');
    mockOpt.value = '__mock__';
    mockOpt.text = '🎥 Mock Video Feed (Simulated Runway)';
    camSelect.appendChild(mockOpt);
    camSelect.disabled = false;
    camSelect.value = '__mock__';
    toast(`Camera enumeration failed: ${err.message}. Mock feed is still available.`, 'warning');
  }
}

btnRefresh.addEventListener('click', _populateCameras);

btnConnect.addEventListener('click', async () => {
  try {
    await camManager.start(camSelect.value || null);
    APP.cameraActive = true;
    btnConnect.disabled = true;
    btnStop.disabled    = false;
    camSelect.disabled  = true;

    // Start streaming frames to WS
    if (ws.connected) camManager.startSending(ws);
    _updateButtonStates();
    toast('Camera connected', 'success', 2000);
  } catch (err) {
    toast(`Camera error: ${err.message}`, 'error');
  }
});

// Also start sending when WS comes up while camera is already running
ws.on('open', () => {
  if (APP.cameraActive) camManager.startSending(ws);
});

btnStop.addEventListener('click', async () => {
  await camManager.stop();
  APP.cameraActive    = false;
  btnConnect.disabled = false;
  btnStop.disabled    = true;
  camSelect.disabled  = false;
  _updateButtonStates();
});

// FPS slider
const fpsSlider = document.getElementById('fps-slider');
fpsSlider.addEventListener('input', () => {
  const val = parseInt(fpsSlider.value);
  document.getElementById('fps-slider-val').textContent = `${val} fps`;
  camManager.setFPS(val);
});

// ── Auto-initialization ───────────────────────────────────────────────────────
document.getElementById('btn-auto-init').addEventListener('click', async () => {
  setInitStatus('running');
  try {
    // Capture current camera frame
    const b64 = await camManager.captureBase64(0.85);

    const res  = await fetch(`${API}/api/init/auto`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ frame_b64: b64 }),
    });
    const data = await res.json();

    if (res.ok && data.status === 'done') {
      setInitStatus('done', data.info);
      toast(`✦ Homography solved (${data.info.method}, ${data.info.n_inliers} inliers)`, 'success', 5000);
    } else {
      const errMsg = data.info?.error ?? 'Unknown error';
      setInitStatus('error', null, errMsg);
      toast(`Auto-init failed: ${errMsg}. Try manual calibration.`, 'error', 6000);
    }
  } catch (err) {
    setInitStatus('error', null, err.message);
    toast(`Auto-init error: ${err.message}`, 'error');
  }
});

// ── Manual calibration modal ──────────────────────────────────────────────────
const modal     = document.getElementById('calibration-modal');
const btnManual = document.getElementById('btn-manual-init');

btnManual.addEventListener('click', async () => {
  // Ensure we have a snapshot
  await camManager.cacheSnapshot();
  const snapshot = camManager.lastSnapshot;
  if (!snapshot) { toast('Camera not active', 'error'); return; }

  // Create CalibrationManager fresh each time
  calibManager = new CalibrationManager(
    document.getElementById('calib-cam-canvas'),
    document.getElementById('calib-ref-canvas'),
    (info) => {
      modal.style.display = 'none';
      setInitStatus('done', info);
      toast(`✦ Manual H computed (${info.n_inliers} inliers)`, 'success', 5000);
    },
  );

  // Load images into calibration canvases
  await calibManager.loadImages(snapshot, APP.refImageSrc);

  modal.style.display = 'flex';
});

document.getElementById('btn-calib-close').addEventListener('click', () => {
  modal.style.display = 'none';
});
document.getElementById('btn-calib-undo').addEventListener('click', () => {
  calibManager?.undo();
});
document.getElementById('btn-calib-clear').addEventListener('click', () => {
  calibManager?.clear();
});
document.getElementById('btn-calib-compute').addEventListener('click', async () => {
  if (!calibManager) return;
  try {
    await calibManager.computeH(API);
    modal.style.display = 'none';
  } catch (err) {
    toast(`Calibration failed: ${err.message}`, 'error');
  }
});

// Close modal on backdrop click
modal.addEventListener('click', e => {
  if (e.target === modal) modal.style.display = 'none';
});

// ── Reset ─────────────────────────────────────────────────────────────────────
document.getElementById('btn-reset').addEventListener('click', async () => {
  await fetch(`${API}/api/reset`, { method: 'POST' });
  setInitStatus('idle');
  refRenderer.setTracks([]);
  _renderTrackList([]);
  document.getElementById('track-count').textContent = '0';
  document.getElementById('det-count').textContent   = '0';
  toast('System reset', 'info', 2000);
});

// ── Ref canvas controls ───────────────────────────────────────────────────────
document.getElementById('btn-ref-zoom-in').addEventListener('click',  () => refRenderer.zoomIn());
document.getElementById('btn-ref-zoom-out').addEventListener('click', () => refRenderer.zoomOut());
document.getElementById('btn-ref-zoom-fit').addEventListener('click', () => refRenderer.fitToView());

// ── Ref canvas sizing ─────────────────────────────────────────────────────────
function _resizeRefCanvas() {
  const wrap = document.getElementById('ref-canvas-wrap');
  const rect = wrap.getBoundingClientRect();
  const canvas = document.getElementById('ref-canvas');
  canvas.width  = rect.width;
  canvas.height = rect.height;
  refRenderer.fitToView();
}
window.addEventListener('resize', _resizeRefCanvas);

// ── Track list sidebar ────────────────────────────────────────────────────────
function _renderTrackList(tracks) {
  const list = document.getElementById('track-list');
  if (!tracks.length) {
    list.innerHTML = '<div class="track-empty">No active tracks</div>';
    return;
  }
  list.innerHTML = '';
  tracks.forEach(t => {
    const hue   = (t.id * 137.508) % 360;
    const color = `hsl(${hue}, 72%, 62%)`;
    const speed = Math.sqrt((t.vx ?? 0) ** 2 + (t.vy ?? 0) ** 2).toFixed(1);

    const card = document.createElement('div');
    card.className = 'track-card';
    card.innerHTML = `
      <div class="track-color-dot" style="background:${color}"></div>
      <div class="track-info">
        <div class="track-id">#${t.id} · ${Math.round(t.confidence * 100)}%</div>
        <div class="track-class">${t.class}</div>
        <div class="track-pos">(${Math.round(t.ref_x)}, ${Math.round(t.ref_y)}) · ${speed}px/f</div>
      </div>
      <div class="track-conf">${Math.round(t.confidence * 100)}%</div>
    `;
    list.appendChild(card);
  });
}

// ── Button guard ──────────────────────────────────────────────────────────────
function _updateButtonStates() {
  const canInit = APP.hasRef && APP.cameraActive;
  document.getElementById('btn-auto-init').disabled   = !canInit;
  document.getElementById('btn-manual-init').disabled = !canInit;
}

// ── Boot sequence ─────────────────────────────────────────────────────────────
(async function init() {
  // Size ref canvas
  _resizeRefCanvas();

  // Populate cameras
  await _populateCameras();

  // Connect WS
  ws.connect();

  // Poll status to sync if page reloads mid-session
  try {
    const res  = await fetch(`${API}/api/status`);
    const data = await res.json();
    if (data.has_homography) {
      setInitStatus('done', data.init_info ?? { method: 'loaded' });
    }
    if (data.has_reference) {
      APP.hasRef      = true;
      APP.refImageSrc = `${API}/api/reference-image?t=${Date.now()}`;
      refRenderer.loadImage(APP.refImageSrc);
      document.getElementById('ref-empty-state').style.display  = 'none';
      document.getElementById('ref-preview-wrap').style.display = 'none';
      document.getElementById('ref-info-bar').style.display     = '';
      document.getElementById('ref-info-size').textContent =
        `${data.ref_image_size[0]} × ${data.ref_image_size[1]}`;
      _updateButtonStates();
    }
  } catch {
    // backend not up yet — WS reconnect will handle it
  }
})();
