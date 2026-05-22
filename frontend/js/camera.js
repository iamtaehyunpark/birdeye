/**
 * camera.js — Webcam capture and frame streaming.
 *
 * Enumerates cameras, starts getUserMedia stream,
 * draws video to a canvas (for detection overlays),
 * and sends JPEG-encoded frames to the WebSocket at a configurable rate.
 */
class CameraManager {
  constructor(videoEl, canvasEl, overlayLabelEl) {
    this.video        = videoEl;
    this.canvas       = canvasEl;
    this.overlayLabel = overlayLabelEl;
    this.ctx          = canvasEl.getContext('2d');
    this.stream       = null;
    this.sendTimer    = null;
    this.fps          = 15;
    this.active       = false;
    this._wsManager   = null;

    // Snapshot for calibration
    this._lastSnapshot = null;
  }

  /** Enumerate available video input devices */
  async enumerateCameras() {
    try {
      // Request permissions first so deviceIds are populated
      const tmpStream = await navigator.mediaDevices.getUserMedia({ video: true });
      tmpStream.getTracks().forEach(t => t.stop());
    } catch { /* ignore — may already have permission */ }

    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter(d => d.kind === 'videoinput');
  }

  /** Start streaming from the given device ID (or first available) */
  async start(deviceId = null) {
    await this.stop();

    const constraints = {
      video: deviceId
        ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
        : { width: { ideal: 1280 }, height: { ideal: 720 } }
    };

    this.stream = await navigator.mediaDevices.getUserMedia(constraints);
    this.video.srcObject = this.stream;
    await this.video.play();

    this.video.style.opacity = '0'; // video element hidden — we draw to canvas
    if (this.overlayLabel) this.overlayLabel.style.display = 'none';

    // Sync canvas size to video
    this.video.addEventListener('loadedmetadata', () => {
      this.canvas.width  = this.video.videoWidth;
      this.canvas.height = this.video.videoHeight;
    }, { once: true });

    this.active = true;
    this._drawLoop();
    console.log('[Camera] Started', deviceId ?? 'default');
  }

  /** Stop the camera stream */
  async stop() {
    this.active = false;
    this.stopSending();

    if (this.stream) {
      this.stream.getTracks().forEach(t => t.stop());
      this.stream = null;
    }
    this.video.srcObject = null;
    if (this.overlayLabel) this.overlayLabel.style.display = '';
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  /** Continuously render video → canvas (so overlays can be drawn on top) */
  _drawLoop() {
    if (!this.active) return;
    if (this.video.readyState >= 2) {
      this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
    }
    requestAnimationFrame(() => this._drawLoop());
  }

  /** Begin sending frames to the backend via WebSocket */
  startSending(wsManager) {
    this._wsManager = wsManager;
    this.stopSending();
    const intervalMs = Math.round(1000 / this.fps);
    this.sendTimer = setInterval(() => this._sendFrame(), intervalMs);
  }

  stopSending() {
    if (this.sendTimer) { clearInterval(this.sendTimer); this.sendTimer = null; }
  }

  setFPS(fps) {
    this.fps = Math.max(1, Math.min(30, fps));
    if (this.sendTimer && this._wsManager) {
      this.startSending(this._wsManager);
    }
  }

  /** Capture a single JPEG frame as a Blob */
  captureBlob(quality = 0.80) {
    return new Promise(resolve => {
      const off = document.createElement('canvas');
      off.width  = this.canvas.width  || this.video.videoWidth  || 640;
      off.height = this.canvas.height || this.video.videoHeight || 480;
      off.getContext('2d').drawImage(this.video, 0, 0, off.width, off.height);
      off.toBlob(resolve, 'image/jpeg', quality);
    });
  }

  /** Capture a frame and return as base64 string */
  async captureBase64(quality = 0.80) {
    const blob  = await this.captureBlob(quality);
    const buf   = await blob.arrayBuffer();
    const bytes = new Uint8Array(buf);
    let bin = '';
    bytes.forEach(b => bin += String.fromCharCode(b));
    return btoa(bin);
  }

  /** Capture and cache a snapshot (used for calibration modal) */
  async cacheSnapshot() {
    this._lastSnapshot = await this.captureBlob(0.92);
    return this._lastSnapshot;
  }

  get lastSnapshot() { return this._lastSnapshot; }

  // ── Private ────────────────────────────────────────────────────────────────
  _sendFrame() {
    if (!this._wsManager || !this._wsManager.connected) return;
    if (!this.active || !this.video.readyState >= 2) return;

    const off = document.createElement('canvas');
    off.width  = this.canvas.width;
    off.height = this.canvas.height;
    off.getContext('2d').drawImage(this.video, 0, 0, off.width, off.height);

    off.toBlob(blob => {
      if (!blob) return;
      blob.arrayBuffer().then(buf => {
        this._wsManager.sendBinary(buf);
      });
    }, 'image/jpeg', 0.75);
  }
}
