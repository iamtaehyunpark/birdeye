/**
 * calibration.js — Manual 4-point homography calibration UI.
 *
 * The user alternately clicks on the camera snapshot canvas (left)
 * and the reference image canvas (right) to define N ≥ 4 correspondences.
 * Once ≥ 4 pairs are selected, "Compute H" calls the backend /api/init/manual.
 *
 * Point interaction state machine:
 *   'cam'  → waiting for a click on the camera canvas
 *   'ref'  → waiting for a click on the reference canvas
 */
class CalibrationManager {
  /**
   * @param {HTMLCanvasElement} camCanvas
   * @param {HTMLCanvasElement} refCanvas
   * @param {Function} onComputed  — callback(info) when H is successfully computed
   */
  constructor(camCanvas, refCanvas, onComputed) {
    this.camCanvas = camCanvas;
    this.refCanvas = refCanvas;
    this.onComputed = onComputed;

    this.camCtx = camCanvas.getContext('2d');
    this.refCtx = refCanvas.getContext('2d');

    // Matched point pairs
    this.camPts = [];  // [{x, y}] — pixel coords in snapshot
    this.refPts = [];  // [{x, y}] — pixel coords in reference image

    // Pending cam point waiting for its ref match
    this._pendingCam = null;

    // Current mode: 'cam' | 'ref'
    this._mode = 'cam';

    // Cached images
    this._camImg = null;
    this._refImg = null;

    this._setupListeners();
  }

  /** Load images into both calibration canvases */
  loadImages(camBlob, refSrc) {
    return new Promise(resolve => {
      const camUrl = URL.createObjectURL(camBlob);
      const camImg = new Image();
      camImg.onload = () => {
        this._camImg = camImg;
        this._fitAndDraw(this.camCanvas, this.camCtx, camImg);
        URL.revokeObjectURL(camUrl);
        if (this._refImg) resolve();
      };
      camImg.src = camUrl;

      const refImg = new Image();
      refImg.onload = () => {
        this._refImg = refImg;
        this._fitAndDraw(this.refCanvas, this.refCtx, refImg);
        if (this._camImg) resolve();
      };
      refImg.src = refSrc;
    });
  }

  /** Remove the last pair */
  undo() {
    if (this._pendingCam) {
      this._pendingCam = null;
      this._mode = 'cam';
    } else {
      this.camPts.pop();
      this.refPts.pop();
    }
    this._redraw();
    this._updateUI();
  }

  /** Remove all pairs */
  clear() {
    this.camPts      = [];
    this.refPts      = [];
    this._pendingCam = null;
    this._mode       = 'cam';
    this._redraw();
    this._updateUI();
  }

  /** POST pairs to backend and fire callback */
  async computeH(apiBase) {
    const cam_pts = this.camPts.map(p => [p.xImg, p.yImg]);
    const ref_pts = this.refPts.map(p => [p.xImg, p.yImg]);

    const res = await fetch(`${apiBase}/api/init/manual`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ cam_pts, ref_pts }),
    });
    const data = await res.json();
    if (res.ok && data.status === 'done') {
      this.onComputed(data.info);
    } else {
      throw new Error(data.info?.error ?? 'Manual calibration failed');
    }
  }

  get pairCount() { return this.camPts.length; }
  get canCompute() { return this.camPts.length >= 4; }

  // ── Private ────────────────────────────────────────────────────────────────

  _setupListeners() {
    this.camCanvas.addEventListener('click', e => {
      if (this._mode !== 'cam') return;
      const pt = this._canvasPt(this.camCanvas, this._camImg, e);
      if (!pt) return;
      this._pendingCam = pt;
      this._mode = 'ref';
      this._redraw();
      this._updateUI();
    });

    this.refCanvas.addEventListener('click', e => {
      if (this._mode !== 'ref') return;
      const pt = this._canvasPt(this.refCanvas, this._refImg, e);
      if (!pt) return;
      this.camPts.push(this._pendingCam);
      this.refPts.push(pt);
      this._pendingCam = null;
      this._mode = 'cam';
      this._redraw();
      this._updateUI();
    });
  }

  /** Compute click position in BOTH canvas-display and image-actual coords */
  _canvasPt(canvas, img, event) {
    if (!img) return null;
    const rect    = canvas.getBoundingClientRect();
    const cx      = event.clientX - rect.left;
    const cy      = event.clientY - rect.top;
    const scaleX  = img.naturalWidth  / rect.width;
    const scaleY  = img.naturalHeight / rect.height;
    return {
      xCanvas: cx,
      yCanvas: cy,
      xImg:    cx * scaleX,
      yImg:    cy * scaleY,
    };
  }

  _fitAndDraw(canvas, ctx, img) {
    const rect = canvas.getBoundingClientRect();
    canvas.width  = rect.width  || 400;
    canvas.height = rect.height || 300;

    const scale = Math.min(canvas.width / img.naturalWidth, canvas.height / img.naturalHeight);
    const drawW = img.naturalWidth  * scale;
    const drawH = img.naturalHeight * scale;
    const offX  = (canvas.width  - drawW) / 2;
    const offY  = (canvas.height - drawH) / 2;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, offX, offY, drawW, drawH);

    // Store scale for point conversion
    canvas._imgScale = scale;
    canvas._imgOffX  = offX;
    canvas._imgOffY  = offY;
  }

  _redraw() {
    this._fitAndDraw(this.camCanvas, this.camCtx, this._camImg);
    this._fitAndDraw(this.refCanvas, this.refCtx, this._refImg);

    // Draw confirmed pairs
    const colors = ['#ef4444','#f59e0b','#10b981','#3b82f6','#a855f7','#ec4899','#14b8a6','#f97316'];
    this.camPts.forEach((pt, i) => this._drawDot(this.camCtx, pt, i, colors[i % colors.length], this.camCanvas));
    this.refPts.forEach((pt, i) => this._drawDot(this.refCtx, pt, i, colors[i % colors.length], this.refCanvas));

    // Draw pending cam point
    if (this._pendingCam) {
      const color = colors[this.camPts.length % colors.length];
      this._drawDot(this.camCtx, this._pendingCam, this.camPts.length, color, this.camCanvas, true);

      // Pulse on ref canvas hint
      const ctx = this.refCtx;
      ctx.save();
      ctx.font      = '13px Inter, sans-serif';
      ctx.fillStyle = 'rgba(255,255,255,0.4)';
      ctx.textAlign = 'center';
      ctx.fillText('Click the matching point here →', this.refCanvas.width / 2, 20);
      ctx.restore();
    }
  }

  _drawDot(ctx, pt, idx, color, canvas, pending = false) {
    const { xCanvas: cx, yCanvas: cy } = pt;

    ctx.save();

    if (pending) {
      // Animated ring
      ctx.strokeStyle = color;
      ctx.lineWidth   = 2;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.arc(cx, cy, 10, 0, Math.PI * 2);
      ctx.stroke();
    } else {
      // Solid dot with number
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(cx, cy, 8, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle    = '#fff';
      ctx.font         = 'bold 9px Inter, sans-serif';
      ctx.textAlign    = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(String(idx + 1), cx, cy + 0.5);
    }

    ctx.restore();
  }

  _updateUI() {
    const n        = this.camPts.length;
    const mode     = this._mode;
    const pending  = !!this._pendingCam;

    // Pair count label
    const label = document.getElementById('calib-pair-count');
    if (label) label.textContent = `${n} pair${n !== 1 ? 's' : ''}`;

    // Pair dots (visual progress)
    const dotsWrap = document.getElementById('calib-pair-dots');
    if (dotsWrap) {
      dotsWrap.innerHTML = '';
      for (let i = 0; i < Math.max(n + (pending ? 1 : 0), 4); i++) {
        const d = document.createElement('div');
        d.className = 'pair-dot' + (i < n ? ' filled' : '');
        dotsWrap.appendChild(d);
      }
    }

    // State hint
    const hint = document.getElementById('calib-state-hint');
    if (hint) {
      if (pending)         hint.textContent = 'Now click the matching point on the reference image →';
      else if (mode === 'cam') hint.textContent = n >= 4 ? 'Add more points or click Compute H' : 'Click a point on the camera snapshot';
    }

    // Button states
    const btnUndo    = document.getElementById('btn-calib-undo');
    const btnCompute = document.getElementById('btn-calib-compute');
    if (btnUndo)    btnUndo.disabled    = (n === 0 && !pending);
    if (btnCompute) btnCompute.disabled = !this.canCompute;
  }
}
