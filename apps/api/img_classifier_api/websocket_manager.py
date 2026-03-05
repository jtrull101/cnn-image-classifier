"""
WebSocket Manager for handling real-time connections and broadcasting.

Manages WebSocket connections with support for:
- Connection pooling
- Broadcasting to all clients
- Targeted messaging to specific clients
- Connection health monitoring
"""

import asyncio
import logging
from typing import Any
from fastapi import WebSocket, WebSocketDisconnect


logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections and message broadcasting."""

    def __init__(self):
        # Active connections: Set of WebSocket objects
        self.active_connections: set[WebSocket] = set()

        # Connection metadata: WebSocket -> connection info
        self.connection_metadata: dict[WebSocket, dict[str, Any]] = {}

        # Room-based connections (for future use)
        self.rooms: dict[str, set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, client_id: str | None = None) -> None:
        """
        Accept and register a new WebSocket connection.

        Args:
            websocket: The WebSocket connection to register
            client_id: Optional unique identifier for the client
        """
        await websocket.accept()
        self.active_connections.add(websocket)

        # Store metadata
        self.connection_metadata[websocket] = {
            "client_id": client_id,
            "connected_at": asyncio.get_event_loop().time(),
        }

        logger.info("WebSocket connected. Total connections: %d", len(self.active_connections))

        # Send welcome message
        await self.send_personal_message(
            {
                "type": "connected",
                "message": "WebSocket connection established",
                "client_id": client_id,
            },
            websocket,
        )

    def disconnect(self, websocket: WebSocket):
        """
        Unregister a WebSocket connection.

        Args:
            websocket: The WebSocket connection to remove
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

        if websocket in self.connection_metadata:
            del self.connection_metadata[websocket]

        # Remove from all rooms
        for room_connections in self.rooms.values():
            room_connections.discard(websocket)

        logger.info("WebSocket disconnected. Total connections: %d", len(self.active_connections))

    async def send_personal_message(self, message: dict[str, Any], websocket: WebSocket) -> None:
        """
        Send a message to a specific WebSocket connection.

        Args:
            message: Dictionary to send as JSON
            websocket: Target WebSocket connection
        """
        try:
            await websocket.send_json(message)
        except WebSocketDisconnect:
            logger.warning("WebSocket disconnected during send")
            self.disconnect(websocket)
        except Exception as e:
            logger.error("Error sending personal message: %s", e)
            self.disconnect(websocket)

    async def broadcast(
        self, message: dict[str, Any], exclude: set[WebSocket] | None = None
    ) -> None:
        """
        Broadcast a message to all active connections.

        Args:
            message: Dictionary to send as JSON
            exclude: Optional set of WebSocket connections to exclude
        """
        exclude = exclude or set()
        disconnected = set()

        for connection in self.active_connections:
            if connection in exclude:
                continue

            try:
                await connection.send_json(message)
            except WebSocketDisconnect:
                disconnected.add(connection)
            except Exception as e:
                logger.error("Error broadcasting to connection: %s", e)
                disconnected.add(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    async def broadcast_to_room(self, room: str, message: dict[str, Any]) -> None:
        """
        Broadcast a message to all connections in a specific room.

        Args:
            room: Room name
            message: Dictionary to send as JSON
        """
        if room not in self.rooms:
            logger.warning("Room %r does not exist", room)
            return

        disconnected = set()

        for connection in self.rooms[room]:
            try:
                await connection.send_json(message)
            except WebSocketDisconnect:
                logger.debug("Client disconnected from room '%s' during broadcast", room)
                disconnected.add(connection)
            except Exception as e:
                logger.error("Error broadcasting to room '%s': %s", room, e)
                disconnected.add(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    def join_room(self, websocket: WebSocket, room: str) -> None:
        """
        Add a WebSocket connection to a room.

        Args:
            websocket: WebSocket connection
            room: Room name
        """
        if room not in self.rooms:
            self.rooms[room] = set()

        self.rooms[room].add(websocket)
        logger.info("WebSocket joined room %r", room)

    def leave_room(self, websocket: WebSocket, room: str) -> None:
        """
        Remove a WebSocket connection from a room.

        Args:
            websocket: WebSocket connection
            room: Room name
        """
        if room in self.rooms:
            self.rooms[room].discard(websocket)
            if not self.rooms[room]:
                del self.rooms[room]
            logger.info("WebSocket left room %r", room)

    async def send_model_loading_update(
        self,
        model_name: str,
        status: str,
        progress: float | None = None,
        error: str | None = None,
    ) -> None:
        """
        Broadcast model loading status to all connections.

        Args:
            model_name: Name of the model being loaded
            status: Loading status (started, progress, completed, error)
            progress: Optional progress percentage (0-100)
            error: Optional error message
        """
        message: dict[str, Any] = {
            "type": "model_loading",
            "model_name": model_name,
            "status": status,
        }

        if progress is not None:
            message["progress"] = progress

        if error is not None:
            message["error"] = error

        await self.broadcast(message)

    async def send_prediction_progress(
        self, image_name: str, progress: float, status: str = "processing"
    ) -> None:
        """
        Broadcast prediction progress to all connections.

        Args:
            image_name: Name of the image being processed
            progress: Progress percentage (0-100)
            status: Processing status
        """
        message = {
            "type": "prediction_progress",
            "image_name": image_name,
            "progress": progress,
            "status": status,
        }

        await self.broadcast(message)

    async def send_batch_progress(
        self, batch_id: str, completed: int, total: int, current_image: str | None = None
    ) -> None:
        """
        Broadcast batch processing progress to all connections.

        Args:
            batch_id: Unique identifier for the batch
            completed: Number of completed images
            total: Total number of images in batch
            current_image: Optional name of current image being processed
        """
        message = {
            "type": "batch_progress",
            "batch_id": batch_id,
            "completed": completed,
            "total": total,
            "progress": (completed / total * 100) if total > 0 else 0,
        }

        if current_image:
            message["current_image"] = current_image

        await self.broadcast(message)

    async def send_error(self, error_message: str, details: dict[str, Any] | None = None) -> None:
        """
        Broadcast an error message to all connections.

        Args:
            error_message: Error message to broadcast
            details: Optional additional error details
        """
        message: dict[str, Any] = {"type": "error", "message": error_message}

        if details:
            message["details"] = details

        await self.broadcast(message)

    def get_connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self.active_connections)

    def get_rooms(self) -> list[str]:
        """Get list of active room names."""
        return list(self.rooms.keys())


# Global WebSocket manager instance
manager = WebSocketManager()
