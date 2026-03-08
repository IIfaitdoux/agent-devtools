"""
WebSocket server for real-time trace streaming and debugger control.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass
from typing import Any, Set

import websockets
from websockets.server import WebSocketServerProtocol

from agentdbg.core import get_debugger
from agentdbg.models import Span, Trace

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """WebSocket server configuration."""

    host: str = "127.0.0.1"
    port: int = 8765


class WebSocketServer:
    """WebSocket server for real-time debugging communication."""

    def __init__(self, config: ServerConfig | None = None) -> None:
        self.config = config or ServerConfig()
        self._clients: Set[WebSocketServerProtocol] = set()
        self._server: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    async def _handler(self, websocket: WebSocketServerProtocol) -> None:
        """Handle incoming WebSocket connections."""
        self._clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self._clients)}")

        try:
            # Send current state on connect
            await self._send_initial_state(websocket)

            async for message in websocket:
                await self._handle_message(websocket, message)

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self._clients.discard(websocket)
            logger.info(f"Client disconnected. Total clients: {len(self._clients)}")

    async def _send_initial_state(self, websocket: WebSocketServerProtocol) -> None:
        """Send current debugger state to newly connected client."""
        debugger = get_debugger()

        state = {
            "type": "initial_state",
            "data": {
                "paused": debugger.is_paused,
                "traces": [t.to_dict() for t in debugger.get_all_traces()],
                "current_trace": debugger.get_current_trace().to_dict()
                if debugger.get_current_trace()
                else None,
                "current_span": debugger.get_current_span().to_dict()
                if debugger.get_current_span()
                else None,
            },
        }

        await websocket.send(json.dumps(state))

    async def _handle_message(
        self, websocket: WebSocketServerProtocol, message: str
    ) -> None:
        """Handle incoming messages from clients."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            debugger = get_debugger()

            if msg_type == "pause":
                debugger.pause()
                await self._broadcast({"type": "paused"})

            elif msg_type == "resume":
                debugger.resume()
                await self._broadcast({"type": "resumed"})

            elif msg_type == "step":
                debugger.step()
                await self._broadcast({"type": "stepped"})

            elif msg_type == "get_traces":
                traces = [t.to_dict() for t in debugger.get_all_traces()]
                await websocket.send(json.dumps({"type": "traces", "data": traces}))

            elif msg_type == "get_trace":
                trace_id = data.get("trace_id")
                trace = debugger.get_trace(trace_id)
                if trace:
                    await websocket.send(
                        json.dumps({"type": "trace", "data": trace.to_dict()})
                    )

            elif msg_type == "clear_traces":
                debugger.clear_traces()
                await self._broadcast({"type": "traces_cleared"})

            elif msg_type == "add_breakpoint":
                condition = data.get("condition", {})
                bp = self._create_breakpoint(condition)
                if bp:
                    debugger.state.add_breakpoint(bp)
                    await websocket.send(
                        json.dumps({"type": "breakpoint_added", "success": True})
                    )

            elif msg_type == "clear_breakpoints":
                debugger.state.clear_breakpoints()
                await websocket.send(json.dumps({"type": "breakpoints_cleared"}))

            elif msg_type == "ping":
                await websocket.send(json.dumps({"type": "pong"}))

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON message: {message[:100]}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await websocket.send(
                json.dumps({"type": "error", "message": str(e)})
            )

    def _create_breakpoint(self, condition: dict[str, Any]) -> Any:
        """Create a breakpoint function from condition dict."""
        bp_type = condition.get("type")

        if bp_type == "cost":
            threshold = condition.get("threshold", 1.0)
            return lambda span: span.cost.total_cost >= threshold

        elif bp_type == "tokens":
            threshold = condition.get("threshold", 10000)
            return lambda span: span.cost.total_tokens >= threshold

        elif bp_type == "error":
            return lambda span: span.error is not None

        elif bp_type == "name":
            name = condition.get("name", "")
            return lambda span: name in span.name

        elif bp_type == "kind":
            kind = condition.get("kind", "")
            return lambda span: span.kind.value == kind

        return None

    async def _broadcast(self, message: dict[str, Any]) -> None:
        """Broadcast a message to all connected clients."""
        if not self._clients:
            return

        msg_str = json.dumps(message)
        await asyncio.gather(
            *[client.send(msg_str) for client in self._clients],
            return_exceptions=True,
        )

    def broadcast_span(self, span: Span) -> None:
        """Broadcast a span update to all clients."""
        if self._loop and self._running:
            asyncio.run_coroutine_threadsafe(
                self._broadcast({"type": "span_update", "data": span.to_dict()}),
                self._loop,
            )

    def broadcast_trace(self, trace: Trace) -> None:
        """Broadcast a trace update to all clients."""
        if self._loop and self._running:
            asyncio.run_coroutine_threadsafe(
                self._broadcast({"type": "trace_update", "data": trace.to_dict()}),
                self._loop,
            )

    async def _start_server(self) -> None:
        """Start the WebSocket server."""
        self._server = await websockets.serve(
            self._handler,
            self.config.host,
            self.config.port,
        )
        logger.info(f"WebSocket server started on ws://{self.config.host}:{self.config.port}")

        # Register callbacks with debugger
        debugger = get_debugger()
        debugger.on_span(self.broadcast_span)
        debugger.on_trace(self.broadcast_trace)

        await self._server.wait_closed()

    def _run_loop(self) -> None:
        """Run the event loop in a separate thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._running = True
        self._loop.run_until_complete(self._start_server())

    def start(self, background: bool = True) -> None:
        """Start the server."""
        if background:
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
        else:
            asyncio.run(self._start_server())

    def stop(self) -> None:
        """Stop the server."""
        self._running = False
        if self._server:
            self._server.close()
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)


def start_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    background: bool = True,
) -> WebSocketServer:
    """Convenience function to start the WebSocket server."""
    config = ServerConfig(host=host, port=port)
    server = WebSocketServer(config)
    server.start(background=background)
    return server
