"""
HTTP server for serving the Web UI.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import logging
import os
import socketserver
import threading
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from agentdbg.core import get_debugger

logger = logging.getLogger(__name__)


class DebuggerHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler for the debugger UI and API."""

    def __init__(self, *args: Any, directory: str | None = None, **kwargs: Any) -> None:
        self.ui_directory = directory or self._get_ui_directory()
        super().__init__(*args, directory=self.ui_directory, **kwargs)

    def _get_ui_directory(self) -> str:
        """Get the UI directory path."""
        # Look for UI files relative to package
        package_dir = Path(__file__).parent.parent
        ui_dir = package_dir / "ui" / "dist"
        if ui_dir.exists():
            return str(ui_dir)

        # Fallback to current directory
        return str(Path.cwd())

    def do_GET(self) -> None:
        """Handle GET requests."""
        parsed = urlparse(self.path)

        # API endpoints
        if parsed.path.startswith("/api/"):
            self._handle_api_get(parsed.path, parsed.query)
            return

        # Serve static files
        super().do_GET()

    def do_POST(self) -> None:
        """Handle POST requests."""
        parsed = urlparse(self.path)

        if parsed.path.startswith("/api/"):
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            self._handle_api_post(parsed.path, body)
            return

        self.send_error(405, "Method Not Allowed")

    def _handle_api_get(self, path: str, query: str) -> None:
        """Handle API GET requests."""
        debugger = get_debugger()
        params = parse_qs(query)

        try:
            if path == "/api/traces":
                traces = [t.to_dict() for t in debugger.get_all_traces()]
                self._send_json({"traces": traces})

            elif path == "/api/trace":
                trace_id = params.get("id", [""])[0]
                trace = debugger.get_trace(trace_id)
                if trace:
                    self._send_json({"trace": trace.to_dict()})
                else:
                    self._send_json({"error": "Trace not found"}, 404)

            elif path == "/api/state":
                self._send_json({
                    "paused": debugger.is_paused,
                    "current_trace": debugger.get_current_trace().to_dict()
                    if debugger.get_current_trace()
                    else None,
                    "current_span": debugger.get_current_span().to_dict()
                    if debugger.get_current_span()
                    else None,
                })

            elif path == "/api/config":
                self._send_json(debugger.config.to_dict())

            else:
                self._send_json({"error": "Not found"}, 404)

        except Exception as e:
            logger.error(f"API error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _handle_api_post(self, path: str, body: str) -> None:
        """Handle API POST requests."""
        debugger = get_debugger()

        try:
            data = json.loads(body) if body else {}

            if path == "/api/pause":
                debugger.pause()
                self._send_json({"success": True, "paused": True})

            elif path == "/api/resume":
                debugger.resume()
                self._send_json({"success": True, "paused": False})

            elif path == "/api/step":
                debugger.step()
                self._send_json({"success": True})

            elif path == "/api/clear":
                debugger.clear_traces()
                self._send_json({"success": True})

            elif path == "/api/breakpoint":
                condition = data.get("condition", {})
                bp_type = condition.get("type")

                if bp_type == "cost":
                    threshold = condition.get("threshold", 1.0)
                    debugger.state.add_breakpoint(
                        lambda s, t=threshold: s.cost.total_cost >= t
                    )
                elif bp_type == "error":
                    debugger.state.add_breakpoint(lambda s: s.error is not None)

                self._send_json({"success": True})

            elif path == "/api/clear_breakpoints":
                debugger.state.clear_breakpoints()
                self._send_json({"success": True})

            else:
                self._send_json({"error": "Not found"}, 404)

        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON"}, 400)
        except Exception as e:
            logger.error(f"API error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _send_json(self, data: dict[str, Any], status: int = 200) -> None:
        """Send a JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def log_message(self, format: str, *args: Any) -> None:
        """Override to use logger instead of stderr."""
        logger.debug(f"{self.address_string()} - {format % args}")


class HTTPServer:
    """HTTP server for the debugger UI."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8766) -> None:
        self.host = host
        self.port = port
        self._server: socketserver.TCPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self, background: bool = True) -> None:
        """Start the HTTP server."""

        def create_handler(*args: Any, **kwargs: Any) -> DebuggerHTTPRequestHandler:
            return DebuggerHTTPRequestHandler(*args, **kwargs)

        self._server = socketserver.TCPServer((self.host, self.port), create_handler)
        logger.info(f"HTTP server started on http://{self.host}:{self.port}")

        if background:
            self._thread = threading.Thread(
                target=self._server.serve_forever, daemon=True
            )
            self._thread.start()
        else:
            self._server.serve_forever()

    def stop(self) -> None:
        """Stop the HTTP server."""
        if self._server:
            self._server.shutdown()
