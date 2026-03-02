"""iPhone IMU bridge via WebSocket.

Serves a small HTML page over HTTP.  Open the printed URL in iPhone Safari,
tap 'Allow Motion & Start', and the phone will stream DeviceMotionEvent data
(accelerometer + all three rotation-rate channels) to this process.

The HTML page displays live alpha/beta/gamma values so you can see which
channel responds to your physical yaw motion.  Set ``yaw_axis`` (config key
``phone_yaw_axis``) to whichever channel changes when you turn:

    portrait  (phone upright)  : yaw_axis = "gamma"
    landscape (phone sideways) : yaw_axis = "beta"
    flat on a surface          : yaw_axis = "alpha"
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import ssl
import subprocess
import tempfile
import threading
import time
from collections import deque
from typing import Optional

from src.core.types import IMUReading

logger = logging.getLogger(__name__)

try:
    import websockets  # type: ignore
    _WEBSOCKETS_OK = True
except ImportError:
    _WEBSOCKETS_OK = False

_WS_PORT_DEFAULT = 8765
_HTTP_PORT_DEFAULT = 8766


def _local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _all_local_ips() -> list[str]:
    """Return all non-loopback local IPv4 addresses."""
    ips: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                ips.add(ip)
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    return sorted(ips)


def _mdns_hostname() -> str:
    """Return the .local mDNS hostname (reliable for Apple-to-Apple)."""
    import subprocess
    try:
        r = subprocess.run(
            ["scutil", "--get", "LocalHostName"],
            capture_output=True, text=True, timeout=2,
        )
        name = r.stdout.strip()
        if name:
            return name + ".local"
    except Exception:
        pass
    name = socket.gethostname()
    return name if name.endswith(".local") else name + ".local"


def _generate_cert(mdns: str, ips: list[str]) -> tuple[str, str, ssl.SSLContext]:
    """Generate a temporary self-signed cert valid for this machine.

    Returns (cert_path, key_path, ssl_context).  Caller owns the files.
    iOS 16+ requires HTTPS for DeviceMotionEvent.requestPermission().
    """
    tmpdir = tempfile.mkdtemp(prefix="horizon_hud_cert_")
    cert = os.path.join(tmpdir, "cert.pem")
    key = os.path.join(tmpdir, "key.pem")
    san_parts = [f"DNS:{mdns}", "DNS:localhost", "IP:127.0.0.1"]
    for ip in ips:
        san_parts.append(f"IP:{ip}")
    san = ",".join(san_parts)
    subprocess.run(
        [
            "openssl", "req", "-x509", "-nodes",
            "-newkey", "rsa:2048",
            "-keyout", key, "-out", cert,
            "-days", "1",
            "-subj", f"/CN={mdns}",
            "-addext", f"subjectAltName={san}",
        ],
        check=True, capture_output=True,
    )
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)
    return cert, key, ctx


def _make_html(use_wss: bool = True) -> str:
    ws_scheme = "wss" if use_wss else "ws"
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Horizon-HUD IMU</title>
  <style>
    body {{font-family:sans-serif;text-align:center;padding:30px;
           background:#111;color:#eee;}}
    button {{font-size:1.3em;padding:16px 36px;margin-top:16px;
             background:#2a7;color:#fff;border:none;border-radius:8px;}}
    #status {{font-size:1.05em;margin:14px 0;}}
    table   {{margin:16px auto;border-collapse:collapse;font-family:monospace;font-size:1em;}}
    td      {{padding:4px 14px;color:#8ef;}}
    th      {{padding:4px 14px;color:#aaa;font-weight:normal;}}
    #note   {{font-size:0.82em;color:#888;margin-top:16px;}}
  </style>
</head>
<body>
  <h2>Horizon-HUD IMU Bridge</h2>
  <p id="status">Tap the button to start streaming.</p>
  <table>
    <tr><th>channel</th><th>value (°/s)</th><th>maps to</th></tr>
    <tr><td>alpha</td><td id="va">—</td><td>gz (flat yaw)</td></tr>
    <tr><td>beta</td> <td id="vb">—</td><td>gx / landscape yaw</td></tr>
    <tr><td>gamma</td><td id="vg">—</td><td>gy / portrait yaw</td></tr>
  </table>
  <p id="note">Rotate the phone and watch which row changes.<br>
    Set <b>phone_yaw_axis</b> in config.yaml to that channel.</p>
  <button onclick="startIMU()">Allow Motion &amp; Start</button>
  <script>
    const WS_URL = '{ws_scheme}://' + location.host;
    let ws = null;
    const DEG = Math.PI / 180;

    // ── Step 1: request motion permission and start listening immediately ──────
    function startIMU() {{
      if (typeof DeviceMotionEvent !== 'undefined' &&
          typeof DeviceMotionEvent.requestPermission === 'function') {{
        DeviceMotionEvent.requestPermission()
          .then(r => {{
            if (r === 'granted') {{ startMotion(); openWS(); }}
            else setStatus('Motion permission denied.');
          }})
          .catch(e => setStatus('Permission error: ' + e));
      }} else {{
        startMotion();
        openWS();
      }}
    }}

    function startMotion() {{
      window.addEventListener('devicemotion', onMotion);
      setStatus('Motion active \u2014 connecting to Mac\u2026');
    }}

    // ── Step 2: open WS independently ─────────────────────────────────────────
    function openWS() {{
      ws = new WebSocket(WS_URL);
      ws.onopen  = () => setStatus('Motion active + WS connected \u2014 streaming.');
      ws.onclose = () => {{ setStatus('Motion active, WS disconnected. Retrying\u2026');
                             setTimeout(openWS, 2000); }};
      ws.onerror = () => setStatus('WS error \u2014 check Mac IP/port ' + WS_URL);
    }}

    // ── Step 3: always update display; send via WS when available ─────────────
    let lastSend = 0;
    function onMotion(e) {{
      const a = e.accelerationIncludingGravity || {{}};
      const r = e.rotationRate || {{}};
      const ra = r.alpha || 0, rb = r.beta || 0, rg = r.gamma || 0;

      // Always update display regardless of WS state
      document.getElementById('va').textContent = ra.toFixed(1);
      document.getElementById('vb').textContent = rb.toFixed(1);
      document.getElementById('vg').textContent = rg.toFixed(1);

      const now = Date.now();
      if (now - lastSend < 20) return;   // ~50 Hz
      lastSend = now;
      if (ws && ws.readyState === WebSocket.OPEN) {{
        ws.send(JSON.stringify({{
          ax:  (a.x || 0),
          ay: -(a.y || 0),
          az: -(a.z || 0),
          ra: ra * DEG,
          rb: rb * DEG,
          rg: rg * DEG
        }}));
      }}
    }}

    function setStatus(msg) {{
      document.getElementById('status').textContent = msg;
    }}
  </script>
</body>
</html>"""


class iPhoneIMUReader:
    """Streams IMU data from an iPhone via a single HTTPS/WSS server.

    One server on ``port`` handles both:
      - GET /  → serves the HTML bridge page
      - WebSocket upgrade → receives IMU JSON from the phone

    Using one port means one TLS cert acceptance in Safari covers both.
    Call ``close()`` to stop the server.
    """

    def __init__(
        self,
        port: int = _HTTP_PORT_DEFAULT,
        ws_port: int = _WS_PORT_DEFAULT,   # kept for back-compat, ignored
        http_port: int = _HTTP_PORT_DEFAULT,  # kept for back-compat, ignored
        yaw_axis: str = "gamma",
        buf_size: int = 5,
    ) -> None:
        if not _WEBSOCKETS_OK:
            raise RuntimeError(
                "websockets is not installed. Run: pip install websockets"
            )
        if yaw_axis not in ("alpha", "beta", "gamma"):
            raise ValueError(f"yaw_axis must be 'alpha', 'beta', or 'gamma', got {yaw_axis!r}")
        self._port = port
        self._yaw_axis = yaw_axis
        self._buf: deque[IMUReading] = deque(maxlen=buf_size)
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._ssl_ctx: Optional[ssl.SSLContext] = None

        mdns = _mdns_hostname()
        ips = _all_local_ips()

        try:
            _, _, self._ssl_ctx = _generate_cert(mdns, ips)
            scheme = "https"
        except Exception as exc:
            logger.warning("Could not generate TLS cert (%s); falling back to HTTP.", exc)
            scheme = "http"

        self._html_bytes = _make_html(use_wss=(self._ssl_ctx is not None)).encode("utf-8")
        self._start_server()

        logger.info("━━━ iPhone IMU bridge ready (yaw_axis=%s) ━━━", yaw_axis)
        logger.info("  Open in iPhone Safari (try mDNS first):")
        logger.info("    %s://%s:%d", scheme, mdns, port)
        for ip in ips:
            logger.info("    %s://%s:%d", scheme, ip, port)
        if scheme == "https":
            logger.info("  Safari will warn 'Not Secure' — tap Advanced → Visit Website.")
            logger.info("  After accepting, tap 'Allow Motion & Start'.")
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # ── combined HTTPS + WSS server on a single port ──────────────────────────

    def _start_server(self) -> None:
        self._ws_thread = threading.Thread(
            target=self._run_ws_loop, daemon=True, name="imu-server"
        )
        self._ws_thread.start()

    def _run_ws_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._ws_serve())
        except Exception:
            pass
        finally:
            self._loop.close()

    async def _ws_serve(self) -> None:
        from websockets.http11 import Response as _WsResponse
        from websockets.datastructures import Headers as _WsHeaders

        self._stop_fut: asyncio.Future = self._loop.create_future()  # type: ignore[assignment]
        html_bytes = self._html_bytes

        async def process_request(connection, request):
            if request.headers.get("upgrade", "").lower() == "websocket":
                return None  # proceed with WebSocket upgrade
            headers = _WsHeaders([
                ("Content-Type", "text/html; charset=utf-8"),
                ("Content-Length", str(len(html_bytes))),
                ("Connection", "close"),
            ])
            return _WsResponse(200, "OK", headers, html_bytes)

        async with websockets.serve(
            self._ws_handler,
            "0.0.0.0",
            self._port,
            ssl=self._ssl_ctx,
            process_request=process_request,
        ):
            await self._stop_fut

    async def _ws_handler(self, websocket) -> None:
        _axis_key = {"alpha": "ra", "beta": "rb", "gamma": "rg"}
        yaw_key = _axis_key[self._yaw_axis]
        async for message in websocket:
            try:
                data = json.loads(message)
                ra = float(data.get("ra", 0.0))
                rb = float(data.get("rb", 0.0))
                rg = float(data.get("rg", 0.0))
                gy = float(data.get(yaw_key, 0.0))
                # gx = pitch axis (beta), gz = flat-yaw axis (alpha)
                reading = IMUReading(
                    accel=(float(data["ax"]), float(data["ay"]), float(data["az"])),
                    gyro=(rb, gy, ra),
                    mag=(0.0, 0.0, 0.0),
                    timestamp=time.monotonic(),
                )
                with self._lock:
                    self._buf.append(reading)
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

    # ── public interface (matches IMUSimulator) ────────────────────────────────

    def read(self, timestamp: float) -> IMUReading:
        with self._lock:
            if self._buf:
                return self._buf[-1]
        return IMUReading(accel=(0.0, 0.0, 0.0), gyro=(0.0, 0.0, 0.0),
                          mag=(0.0, 0.0, 0.0), timestamp=timestamp)

    @property
    def source_name(self) -> str:
        return "phone"

    def cycle_scenario(self) -> None:
        pass

    def close(self) -> None:
        if self._loop and not self._loop.is_closed():
            def _stop():
                if not self._stop_fut.done():
                    self._stop_fut.set_result(None)
            self._loop.call_soon_threadsafe(_stop)
        if self._ws_thread:
            self._ws_thread.join(timeout=3.0)
