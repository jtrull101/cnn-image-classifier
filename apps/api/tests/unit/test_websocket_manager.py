"""Unit tests for WebSocketManager."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import WebSocketDisconnect

from img_classifier_api.websocket_manager import WebSocketManager


pytestmark = pytest.mark.unit


def _make_ws() -> MagicMock:
    """Create a mock WebSocket with async methods."""
    ws = MagicMock()
    ws.accept = AsyncMock()
    ws.send_json = AsyncMock()
    return ws


class TestConnect:
    async def test_accepts_websocket(self):
        manager = WebSocketManager()
        ws = _make_ws()
        await manager.connect(ws)
        ws.accept.assert_awaited_once()

    async def test_adds_to_active_connections(self):
        manager = WebSocketManager()
        ws = _make_ws()
        await manager.connect(ws)
        assert ws in manager.active_connections

    async def test_stores_client_id_in_metadata(self):
        manager = WebSocketManager()
        ws = _make_ws()
        await manager.connect(ws, client_id="test-client")
        assert manager.connection_metadata[ws]["client_id"] == "test-client"

    async def test_stores_none_client_id_when_omitted(self):
        manager = WebSocketManager()
        ws = _make_ws()
        await manager.connect(ws)
        assert manager.connection_metadata[ws]["client_id"] is None

    async def test_sends_welcome_message(self):
        manager = WebSocketManager()
        ws = _make_ws()
        await manager.connect(ws)
        ws.send_json.assert_awaited_once()
        payload = ws.send_json.call_args[0][0]
        assert payload["type"] == "connected"

    async def test_multiple_connects_increment_count(self):
        manager = WebSocketManager()
        ws1, ws2 = _make_ws(), _make_ws()
        await manager.connect(ws1)
        await manager.connect(ws2)
        assert manager.get_connection_count() == 2


class TestDisconnect:
    async def test_removes_from_active_connections(self):
        manager = WebSocketManager()
        ws = _make_ws()
        await manager.connect(ws)
        manager.disconnect(ws)
        assert ws not in manager.active_connections

    async def test_removes_metadata(self):
        manager = WebSocketManager()
        ws = _make_ws()
        await manager.connect(ws)
        manager.disconnect(ws)
        assert ws not in manager.connection_metadata

    def test_disconnect_unknown_ws_is_noop(self):
        manager = WebSocketManager()
        ws = _make_ws()
        manager.disconnect(ws)  # Should not raise
        assert ws not in manager.active_connections

    async def test_removes_from_rooms(self):
        manager = WebSocketManager()
        ws = _make_ws()
        await manager.connect(ws)
        manager.join_room(ws, "room1")
        manager.disconnect(ws)
        assert ws not in manager.rooms.get("room1", set())


class TestSendPersonalMessage:
    async def test_delivers_json_to_websocket(self):
        manager = WebSocketManager()
        ws = _make_ws()
        await manager.connect(ws)
        ws.send_json.reset_mock()

        msg = {"type": "test", "data": "hello"}
        await manager.send_personal_message(msg, ws)
        ws.send_json.assert_awaited_once_with(msg)

    async def test_disconnects_on_websocket_disconnect(self):
        manager = WebSocketManager()
        ws = _make_ws()
        await manager.connect(ws)
        ws.send_json.side_effect = WebSocketDisconnect()

        await manager.send_personal_message({"type": "test"}, ws)
        assert ws not in manager.active_connections

    async def test_disconnects_on_generic_exception(self):
        manager = WebSocketManager()
        ws = _make_ws()
        await manager.connect(ws)
        ws.send_json.side_effect = RuntimeError("broken pipe")

        await manager.send_personal_message({"type": "test"}, ws)
        assert ws not in manager.active_connections


class TestBroadcast:
    async def test_sends_to_all_connections(self):
        manager = WebSocketManager()
        ws1, ws2 = _make_ws(), _make_ws()
        await manager.connect(ws1)
        await manager.connect(ws2)
        ws1.send_json.reset_mock()
        ws2.send_json.reset_mock()

        msg = {"type": "broadcast"}
        await manager.broadcast(msg)
        ws1.send_json.assert_awaited_once_with(msg)
        ws2.send_json.assert_awaited_once_with(msg)

    async def test_excludes_specified_connections(self):
        manager = WebSocketManager()
        ws1, ws2 = _make_ws(), _make_ws()
        await manager.connect(ws1)
        await manager.connect(ws2)
        ws1.send_json.reset_mock()
        ws2.send_json.reset_mock()

        await manager.broadcast({"type": "test"}, exclude={ws1})
        ws1.send_json.assert_not_awaited()
        ws2.send_json.assert_awaited_once()

    async def test_cleans_up_disconnected_on_error(self):
        manager = WebSocketManager()
        ws1, ws2 = _make_ws(), _make_ws()
        await manager.connect(ws1)
        await manager.connect(ws2)
        ws1.send_json.reset_mock()
        ws1.send_json.side_effect = WebSocketDisconnect()

        await manager.broadcast({"type": "test"})
        assert ws1 not in manager.active_connections
        assert ws2 in manager.active_connections

    async def test_empty_connections_is_noop(self):
        manager = WebSocketManager()
        await manager.broadcast({"type": "test"})  # Should not raise


class TestBroadcastToRoom:
    async def test_sends_only_to_room_members(self):
        manager = WebSocketManager()
        ws1, ws2 = _make_ws(), _make_ws()
        await manager.connect(ws1)
        await manager.connect(ws2)
        manager.join_room(ws1, "room1")
        ws1.send_json.reset_mock()
        ws2.send_json.reset_mock()

        msg = {"type": "room_msg"}
        await manager.broadcast_to_room("room1", msg)
        ws1.send_json.assert_awaited_once_with(msg)
        ws2.send_json.assert_not_awaited()

    async def test_nonexistent_room_is_noop(self):
        manager = WebSocketManager()
        await manager.broadcast_to_room("nonexistent", {"type": "test"})  # Should not raise

    async def test_cleans_up_disconnected_on_error(self):
        manager = WebSocketManager()
        ws = _make_ws()
        await manager.connect(ws)
        manager.join_room(ws, "room1")
        ws.send_json.reset_mock()
        ws.send_json.side_effect = WebSocketDisconnect()

        await manager.broadcast_to_room("room1", {"type": "test"})
        assert ws not in manager.active_connections


class TestJoinRoom:
    def test_creates_room_and_adds_connection(self):
        manager = WebSocketManager()
        ws = _make_ws()
        manager.join_room(ws, "room1")
        assert "room1" in manager.rooms
        assert ws in manager.rooms["room1"]

    def test_multiple_connections_in_same_room(self):
        manager = WebSocketManager()
        ws1, ws2 = _make_ws(), _make_ws()
        manager.join_room(ws1, "room1")
        manager.join_room(ws2, "room1")
        assert len(manager.rooms["room1"]) == 2

    def test_connection_in_multiple_rooms(self):
        manager = WebSocketManager()
        ws = _make_ws()
        manager.join_room(ws, "room_a")
        manager.join_room(ws, "room_b")
        assert ws in manager.rooms["room_a"]
        assert ws in manager.rooms["room_b"]


class TestLeaveRoom:
    def test_removes_connection_from_room(self):
        manager = WebSocketManager()
        ws1, ws2 = _make_ws(), _make_ws()
        manager.join_room(ws1, "room1")
        manager.join_room(ws2, "room1")
        manager.leave_room(ws1, "room1")
        assert ws1 not in manager.rooms["room1"]
        assert ws2 in manager.rooms["room1"]

    def test_deletes_room_when_last_member_leaves(self):
        manager = WebSocketManager()
        ws = _make_ws()
        manager.join_room(ws, "room1")
        manager.leave_room(ws, "room1")
        assert "room1" not in manager.rooms

    def test_nonexistent_room_is_noop(self):
        manager = WebSocketManager()
        ws = _make_ws()
        manager.leave_room(ws, "nonexistent")  # Should not raise


class TestSendModelLoadingUpdate:
    async def test_broadcasts_correct_fields(self):
        manager = WebSocketManager()
        ws = _make_ws()
        await manager.connect(ws)
        ws.send_json.reset_mock()

        await manager.send_model_loading_update("my_model", "started")
        payload = ws.send_json.call_args[0][0]
        assert payload["type"] == "model_loading"
        assert payload["model_name"] == "my_model"
        assert payload["status"] == "started"

    async def test_includes_progress_when_provided(self):
        manager = WebSocketManager()
        ws = _make_ws()
        await manager.connect(ws)
        ws.send_json.reset_mock()

        await manager.send_model_loading_update("my_model", "progress", progress=50.0)
        payload = ws.send_json.call_args[0][0]
        assert payload["progress"] == 50.0

    async def test_includes_error_when_provided(self):
        manager = WebSocketManager()
        ws = _make_ws()
        await manager.connect(ws)
        ws.send_json.reset_mock()

        await manager.send_model_loading_update("my_model", "error", error="File not found")
        payload = ws.send_json.call_args[0][0]
        assert payload["error"] == "File not found"

    async def test_omits_optional_fields_when_not_provided(self):
        manager = WebSocketManager()
        ws = _make_ws()
        await manager.connect(ws)
        ws.send_json.reset_mock()

        await manager.send_model_loading_update("my_model", "completed")
        payload = ws.send_json.call_args[0][0]
        assert "progress" not in payload
        assert "error" not in payload


class TestSendPredictionProgress:
    async def test_broadcasts_correct_fields(self):
        manager = WebSocketManager()
        ws = _make_ws()
        await manager.connect(ws)
        ws.send_json.reset_mock()

        await manager.send_prediction_progress("image.jpg", 75.0)
        payload = ws.send_json.call_args[0][0]
        assert payload["type"] == "prediction_progress"
        assert payload["image_name"] == "image.jpg"
        assert payload["progress"] == 75.0
        assert payload["status"] == "processing"

    async def test_custom_status_is_used(self):
        manager = WebSocketManager()
        ws = _make_ws()
        await manager.connect(ws)
        ws.send_json.reset_mock()

        await manager.send_prediction_progress("image.jpg", 100.0, status="completed")
        payload = ws.send_json.call_args[0][0]
        assert payload["status"] == "completed"


class TestSendBatchProgress:
    async def test_broadcasts_correct_fields(self):
        manager = WebSocketManager()
        ws = _make_ws()
        await manager.connect(ws)
        ws.send_json.reset_mock()

        await manager.send_batch_progress("batch-123", 5, 10)
        payload = ws.send_json.call_args[0][0]
        assert payload["type"] == "batch_progress"
        assert payload["batch_id"] == "batch-123"
        assert payload["completed"] == 5
        assert payload["total"] == 10
        assert payload["progress"] == 50.0

    async def test_progress_is_zero_when_total_is_zero(self):
        manager = WebSocketManager()
        ws = _make_ws()
        await manager.connect(ws)
        ws.send_json.reset_mock()

        await manager.send_batch_progress("batch-123", 0, 0)
        payload = ws.send_json.call_args[0][0]
        assert payload["progress"] == 0

    async def test_includes_current_image_when_provided(self):
        manager = WebSocketManager()
        ws = _make_ws()
        await manager.connect(ws)
        ws.send_json.reset_mock()

        await manager.send_batch_progress("batch-123", 3, 10, current_image="img3.jpg")
        payload = ws.send_json.call_args[0][0]
        assert payload["current_image"] == "img3.jpg"

    async def test_omits_current_image_when_not_provided(self):
        manager = WebSocketManager()
        ws = _make_ws()
        await manager.connect(ws)
        ws.send_json.reset_mock()

        await manager.send_batch_progress("batch-123", 3, 10)
        payload = ws.send_json.call_args[0][0]
        assert "current_image" not in payload


class TestSendError:
    async def test_broadcasts_error_message(self):
        manager = WebSocketManager()
        ws = _make_ws()
        await manager.connect(ws)
        ws.send_json.reset_mock()

        await manager.send_error("Something went wrong")
        payload = ws.send_json.call_args[0][0]
        assert payload["type"] == "error"
        assert payload["message"] == "Something went wrong"

    async def test_includes_details_when_provided(self):
        manager = WebSocketManager()
        ws = _make_ws()
        await manager.connect(ws)
        ws.send_json.reset_mock()

        details = {"code": 500, "context": "prediction failed"}
        await manager.send_error("Error", details=details)
        payload = ws.send_json.call_args[0][0]
        assert payload["details"] == details

    async def test_omits_details_when_none(self):
        manager = WebSocketManager()
        ws = _make_ws()
        await manager.connect(ws)
        ws.send_json.reset_mock()

        await manager.send_error("Error", details=None)
        payload = ws.send_json.call_args[0][0]
        assert "details" not in payload


class TestGetConnectionCount:
    def test_zero_initially(self):
        manager = WebSocketManager()
        assert manager.get_connection_count() == 0

    async def test_increases_on_connect(self):
        manager = WebSocketManager()
        ws1, ws2 = _make_ws(), _make_ws()
        await manager.connect(ws1)
        await manager.connect(ws2)
        assert manager.get_connection_count() == 2

    async def test_decreases_on_disconnect(self):
        manager = WebSocketManager()
        ws = _make_ws()
        await manager.connect(ws)
        manager.disconnect(ws)
        assert manager.get_connection_count() == 0


class TestGetRooms:
    def test_empty_initially(self):
        manager = WebSocketManager()
        assert manager.get_rooms() == []

    def test_returns_room_names(self):
        manager = WebSocketManager()
        ws = _make_ws()
        manager.join_room(ws, "room_a")
        manager.join_room(ws, "room_b")
        assert set(manager.get_rooms()) == {"room_a", "room_b"}

    def test_room_removed_after_last_member_leaves(self):
        manager = WebSocketManager()
        ws = _make_ws()
        manager.join_room(ws, "room1")
        manager.leave_room(ws, "room1")
        assert manager.get_rooms() == []
