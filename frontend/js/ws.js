/**
 * ws.js — WebSocket client manager.
 *
 * Maintains a single persistent connection to /ws/frames.
 * Handles reconnection with exponential backoff.
 * Provides simple send/onMessage API.
 */
class WSManager {
  constructor(url) {
    this.url        = url;
    this.ws         = null;
    this.connected  = false;
    this._handlers  = [];
    this._retryDelay = 1000;
    this._maxDelay   = 16000;
    this._retryTimer = null;
    this._pingTimer  = null;
  }

  connect() {
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) return;

    this.ws = new WebSocket(this.url);
    this.ws.binaryType = 'arraybuffer';

    this.ws.onopen = () => {
      this.connected   = true;
      this._retryDelay = 1000;
      clearTimeout(this._retryTimer);
      this._startPing();
      this._emit('open');
      console.log('[WS] Connected');
    };

    this.ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data);
        this._emit('message', data);
      } catch {
        // binary or non-JSON — ignore
      }
    };

    this.ws.onclose = () => {
      this.connected = false;
      clearInterval(this._pingTimer);
      this._emit('close');
      this._scheduleReconnect();
    };

    this.ws.onerror = (err) => {
      console.warn('[WS] Error:', err);
      this._emit('error', err);
    };
  }

  /** Send binary data (JPEG bytes) */
  sendBinary(data) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(data);
      return true;
    }
    return false;
  }

  /** Register an event handler. event: 'open' | 'close' | 'error' | 'message' */
  on(event, handler) {
    this._handlers.push({ event, handler });
    return this; // chainable
  }

  disconnect() {
    clearTimeout(this._retryTimer);
    clearInterval(this._pingTimer);
    if (this.ws) this.ws.close();
  }

  // ── Private ────────────────────────────────────────────────────────────────
  _emit(event, data) {
    this._handlers
      .filter(h => h.event === event)
      .forEach(h => h.handler(data));
  }

  _startPing() {
    clearInterval(this._pingTimer);
    this._pingTimer = setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'ping' }));
      }
    }, 5000);
  }

  _scheduleReconnect() {
    this._retryTimer = setTimeout(() => {
      console.log(`[WS] Reconnecting in ${this._retryDelay}ms…`);
      this.connect();
      this._retryDelay = Math.min(this._retryDelay * 2, this._maxDelay);
    }, this._retryDelay);
  }
}
