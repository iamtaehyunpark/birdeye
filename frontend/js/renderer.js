/**
 * renderer.js — Canvas rendering for both the camera feed and reference image.
 *
 * CamRenderer:  draws detection bounding boxes + ground-contact crosshairs
 *               on top of the camera canvas.
 *
 * RefRenderer:  draws the reference image + Kalman-smoothed track icons,
 *               trail polylines, velocity arrows, and uncertainty circles.
 */

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Deterministic HSL colour from integer track ID */
function trackColor(id) {
  const hue = (id * 137.508) % 360; // golden-angle step → well-distributed
  return `hsl(${hue}, 72%, 62%)`;
}

function trackColorAlpha(id, alpha) {
  const hue = (id * 137.508) % 360;
  return `hsla(${hue}, 72%, 62%, ${alpha})`;
}

/** Draw a rounded rectangle path (no fill/stroke — caller does that) */
function roundRect(ctx, x, y, w, h, r) {
  r = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.arcTo(x + w, y,     x + w, y + h, r);
  ctx.lineTo(x + w, y + h - r);
  ctx.arcTo(x + w, y + h, x,     y + h, r);
  ctx.lineTo(x + r, y + h);
  ctx.arcTo(x,     y + h, x, y,     r);
  ctx.lineTo(x, y + r);
  ctx.arcTo(x, y,     x + w, y, r);
  ctx.closePath();
}

/** Draw an arrowhead at the tip of a velocity vector */
function drawArrow(ctx, x, y, dx, dy, color) {
  const len = Math.sqrt(dx * dx + dy * dy);
  if (len < 2) return;
  const scale   = Math.min(len, 40);
  const nx      = dx / len;
  const ny      = dy / len;
  const ex      = x + nx * scale;
  const ey      = y + ny * scale;
  const headLen = 8;
  const angle   = Math.atan2(ny, nx);

  ctx.strokeStyle = color;
  ctx.lineWidth   = 1.5;
  ctx.globalAlpha = 0.75;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(ex, ey);
  ctx.stroke();

  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(ex, ey);
  ctx.lineTo(ex - headLen * Math.cos(angle - 0.45), ey - headLen * Math.sin(angle - 0.45));
  ctx.lineTo(ex - headLen * Math.cos(angle + 0.45), ey - headLen * Math.sin(angle + 0.45));
  ctx.closePath();
  ctx.fill();
  ctx.globalAlpha = 1;
}

// ══════════════════════════════════════════════════════════ CamRenderer ══════

class CamRenderer {
  /**
   * @param {HTMLCanvasElement} canvas — the camera overlay canvas
   */
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx    = canvas.getContext('2d');
    this.detections = [];
    this.staticDetections = [];
  }

  /** Cache detections to draw them on every animation frame */
  setDetections(detections, staticDetections) {
    this.detections = detections || [];
    this.staticDetections = staticDetections || [];
  }

  /** Render cached overlays */
  draw() {
    this.drawOverlays(this.detections);
    this.drawStaticOverlays(this.staticDetections);
  }

  /**
   * Draw detection overlays on top of whatever the camera module already
   * rendered (video → canvas).
   *
   * @param {Array} detections  — from WS payload .detections
   */
  drawOverlays(detections) {
    const ctx = this.ctx;
    const cw  = this.canvas.width;
    const ch  = this.canvas.height;

    // The camera module draws the raw video; we ONLY add overlays, not clear
    // (clearing would flash).  We paint overlays on each rAF separately.

    for (const det of detections) {
      const [x1, y1, x2, y2] = det.bbox;
      const [u, v]            = det.ground_px;
      const color             = this._classColor(det.class);

      const bw = x2 - x1;
      const bh = y2 - y1;

      // ── Bounding box ─────────────────────────────────────────────────────
      ctx.save();
      // Semi-transparent fill
      roundRect(ctx, x1, y1, bw, bh, 4);
      ctx.fillStyle = color.replace('hsl', 'hsla').replace(')', ', 0.08)');
      ctx.fill();
      // Stroke
      ctx.strokeStyle = color;
      ctx.lineWidth   = 1.5;
      roundRect(ctx, x1, y1, bw, bh, 4);
      ctx.stroke();
      ctx.restore();

      // ── Label pill ───────────────────────────────────────────────────────
      const label = `${det.class}  ${Math.round(det.confidence * 100)}%`;
      ctx.save();
      ctx.font         = '500 11px Inter, sans-serif';
      ctx.textBaseline = 'middle';
      const tw = ctx.measureText(label).width;
      const lx = x1;
      const ly = Math.max(y1 - 20, 2);
      roundRect(ctx, lx, ly, tw + 10, 17, 4);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.fillStyle = '#fff';
      ctx.fillText(label, lx + 5, ly + 8.5);
      ctx.restore();

      // ── Ground-contact crosshair ─────────────────────────────────────────
      const r = 5;
      ctx.save();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth   = 1.5;
      ctx.globalAlpha = 0.9;
      // Horizontal bar
      ctx.beginPath();
      ctx.moveTo(u - r, v);
      ctx.lineTo(u + r, v);
      ctx.stroke();
      // Vertical bar
      ctx.beginPath();
      ctx.moveTo(u, v - r);
      ctx.lineTo(u, v + r);
      ctx.stroke();
      // Centre dot
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(u, v, 2.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
  }

  /**
   * Draw background (static/suppressed) detections — dim dashed outline only.
   * These are YOLO detections rejected because they match the reference image
   * background (i.e., the object was already there before the camera was set up).
   */
  drawStaticOverlays(detections) {
    if (!detections || !detections.length) return;
    const ctx = this.ctx;
    for (const det of detections) {
      const [x1, y1, x2, y2] = det.bbox;
      const bw = x2 - x1;
      const bh = y2 - y1;

      ctx.save();
      ctx.globalAlpha = 0.3;
      ctx.strokeStyle = '#64748b';
      ctx.lineWidth   = 1;
      ctx.setLineDash([4, 4]);
      roundRect(ctx, x1, y1, bw, bh, 4);
      ctx.stroke();

      // Tiny "(bg)" label
      ctx.setLineDash([]);
      ctx.font         = '10px Inter, sans-serif';
      ctx.textBaseline = 'top';
      ctx.fillStyle    = '#94a3b8';
      ctx.fillText(`${det.class} (bg)`, x1 + 3, y1 + 3);
      ctx.restore();
    }
  }

  _classColor(cls) {
    // Deterministic but class-specific rather than ID-specific
    const palette = {
      person:      'hsl(210, 80%, 65%)',
      car:         'hsl(45,  85%, 60%)',
      truck:       'hsl(30,  80%, 60%)',
      airplane:    'hsl(280, 70%, 65%)',
      bus:         'hsl(170, 70%, 55%)',
      bicycle:     'hsl(330, 70%, 65%)',
      motorcycle:  'hsl(350, 75%, 60%)',
    };
    return palette[cls] ?? 'hsl(195, 65%, 60%)';
  }
}

// ══════════════════════════════════════════════════════════ RefRenderer ═══════

class RefRenderer {
  /**
   * @param {HTMLCanvasElement} canvas — the large reference image canvas
   */
  constructor(canvas) {
    this.canvas      = canvas;
    this.ctx         = canvas.getContext('2d');
    this._img        = null;       // HTMLImageElement
    this._imgLoaded  = false;
    this._zoom       = 1.0;
    this._panX       = 0;
    this._panY       = 0;
    this._rafId      = null;
    this._tracks     = [];

    this._setupPanZoom();
    this._renderLoop();
  }

  /** Load (or replace) the reference image */
  loadImage(src) {
    const img    = new Image();
    img.onload   = () => {
      this._img       = img;
      this._imgLoaded = true;
      this.fitToView();
    };
    img.onerror  = () => console.error('[RefRenderer] Failed to load image:', src);
    img.src      = src;
  }

  /** Update tracks (called on each WS message) */
  setTracks(tracks) { this._tracks = tracks; }

  /** Fit the image to fill the canvas while preserving aspect ratio */
  fitToView() {
    if (!this._img) return;
    const cw   = this.canvas.width;
    const ch   = this.canvas.height;
    const scaleX = cw / this._img.naturalWidth;
    const scaleY = ch / this._img.naturalHeight;
    this._zoom = Math.min(scaleX, scaleY) * 0.95;
    this._panX = (cw - this._img.naturalWidth  * this._zoom) / 2;
    this._panY = (ch - this._img.naturalHeight * this._zoom) / 2;
  }

  zoomIn()  { this._setZoom(this._zoom * 1.25); }
  zoomOut() { this._setZoom(this._zoom / 1.25); }

  _setZoom(z) {
    const cw = this.canvas.width;
    const ch = this.canvas.height;
    const prevZ = this._zoom;
    this._zoom  = Math.max(0.1, Math.min(8, z));
    // Zoom around centre
    this._panX = cw / 2 - (cw / 2 - this._panX) * (this._zoom / prevZ);
    this._panY = ch / 2 - (ch / 2 - this._panY) * (this._zoom / prevZ);
  }

  // ── Private rendering ──────────────────────────────────────────────────────

  _renderLoop() {
    this._rafId = requestAnimationFrame(() => this._renderLoop());
    this._draw();
  }

  _draw() {
    const ctx = this.ctx;
    const cw  = this.canvas.width;
    const ch  = this.canvas.height;

    ctx.clearRect(0, 0, cw, ch);

    if (!this._imgLoaded || !this._img) return;

    // ── Reference image ────────────────────────────────────────────────────
    ctx.save();
    ctx.translate(this._panX, this._panY);
    ctx.scale(this._zoom, this._zoom);
    ctx.drawImage(this._img, 0, 0);
    ctx.restore();

    // ── Tracks overlay ─────────────────────────────────────────────────────
    for (const track of this._tracks) {
      this._drawTrack(track);
    }
  }

  _drawTrack(track) {
    const ctx   = this.ctx;
    const color = trackColor(track.id);

    const toCanvas = (rx, ry) => ({
      x: rx * this._zoom + this._panX,
      y: ry * this._zoom + this._panY,
    });

    // Current position in canvas space
    const { x: cx, y: cy } = toCanvas(track.ref_x, track.ref_y);

    // ── Trail polyline ────────────────────────────────────────────────────
    if (track.trail && track.trail.length > 1) {
      const n = track.trail.length;
      for (let i = 1; i < n; i++) {
        const a = toCanvas(track.trail[i - 1][0], track.trail[i - 1][1]);
        const b = toCanvas(track.trail[i][0],     track.trail[i][1]);
        const alpha = i / n; // fade from transparent (old) → opaque (new)

        ctx.save();
        ctx.strokeStyle = trackColorAlpha(track.id, alpha * 0.8);
        ctx.lineWidth   = 1.5 + alpha;
        ctx.lineCap     = 'round';
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
        ctx.restore();
      }
    }

    // ── Uncertainty circle (Kalman covariance) ────────────────────────────
    const covR = (track.cov_radius ?? 10) * this._zoom;
    if (covR > 3) {
      ctx.save();
      ctx.beginPath();
      ctx.arc(cx, cy, covR, 0, Math.PI * 2);
      ctx.fillStyle   = trackColorAlpha(track.id, 0.06);
      ctx.strokeStyle = trackColorAlpha(track.id, 0.2);
      ctx.lineWidth   = 1;
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }

    // ── Velocity arrow ────────────────────────────────────────────────────
    const vx = (track.vx ?? 0) * this._zoom * 3;
    const vy = (track.vy ?? 0) * this._zoom * 3;
    drawArrow(ctx, cx, cy, vx, vy, color);

    // ── Track icon (filled circle + class initial) ─────────────────────────
    const r = 10;
    ctx.save();
    // Outer glow
    ctx.shadowColor  = color;
    ctx.shadowBlur   = 10;
    ctx.beginPath();
    ctx.arc(cx, cy, r + 2, 0, Math.PI * 2);
    ctx.strokeStyle = trackColorAlpha(track.id, 0.35);
    ctx.lineWidth   = 1.5;
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Filled circle
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();

    // Class initial letter
    ctx.fillStyle    = '#000';
    ctx.font         = `bold ${r}px Inter, sans-serif`;
    ctx.textAlign    = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText((track.class ?? '?')[0].toUpperCase(), cx, cy + 0.5);
    ctx.restore();

    // ── ID label ──────────────────────────────────────────────────────────
    const label = `#${track.id} ${track.class}`;
    ctx.save();
    ctx.font         = '500 10px Inter, sans-serif';
    ctx.textAlign    = 'left';
    ctx.textBaseline = 'bottom';
    const tw = ctx.measureText(label).width;

    // Background pill
    roundRect(ctx, cx + r + 3, cy - 8, tw + 8, 15, 4);
    ctx.fillStyle = 'rgba(7,10,18,0.8)';
    ctx.fill();
    roundRect(ctx, cx + r + 3, cy - 8, tw + 8, 15, 4);
    ctx.strokeStyle = trackColorAlpha(track.id, 0.4);
    ctx.lineWidth   = 0.75;
    ctx.stroke();

    ctx.fillStyle = color;
    ctx.fillText(label, cx + r + 7, cy + 6.5);
    ctx.restore();
  }

  // ── Pan & Zoom interaction ─────────────────────────────────────────────────
  _setupPanZoom() {
    let dragging = false;
    let lastX = 0, lastY = 0;

    this.canvas.addEventListener('mousedown', e => {
      dragging = true;
      lastX = e.clientX;
      lastY = e.clientY;
    });
    window.addEventListener('mousemove', e => {
      if (!dragging) return;
      this._panX += e.clientX - lastX;
      this._panY += e.clientY - lastY;
      lastX = e.clientX;
      lastY = e.clientY;
    });
    window.addEventListener('mouseup', () => { dragging = false; });

    this.canvas.addEventListener('wheel', e => {
      e.preventDefault();
      const rect  = this.canvas.getBoundingClientRect();
      const mx    = e.clientX - rect.left;
      const my    = e.clientY - rect.top;
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      const prevZ = this._zoom;
      this._zoom  = Math.max(0.1, Math.min(8, this._zoom * delta));
      this._panX  = mx - (mx - this._panX) * (this._zoom / prevZ);
      this._panY  = my - (my - this._panY) * (this._zoom / prevZ);
    }, { passive: false });
  }

  /** Convert a canvas-space click to reference-image pixel coords */
  canvasToRef(cx, cy) {
    return {
      x: (cx - this._panX) / this._zoom,
      y: (cy - this._panY) / this._zoom,
    };
  }
}
