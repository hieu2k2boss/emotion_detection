"""
batch/speed_layer.py
───────────────────
Speed Layer — in-memory realtime tracking.
Giữ counters, feed, alert queue cho Serving Layer merge.
"""
import logging
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class SpeedLayer:
    """
    In-memory state cho realtime analytics.
    Mất khi restart — nhưng Batch Layer đã có historical data.
    """

    def __init__(self, feed_size: int = 100, alert_size: int = 200):
        # ── Counters ─────────────────────────────────────
        self._emotion_counts: defaultdict[str, int] = defaultdict(int)
        self._message_count: int = 0
        self._alert_count: int = 0
        self._ticket_count: int = 0

        # ── Feed & Alert Queue ───────────────────────────
        self._recent_feed: deque[Dict] = deque(maxlen=feed_size)
        self._alert_queue: deque[Dict] = deque(maxlen=alert_size)

        # ── Per-ticket latest emotion ────────────────────
        self._ticket_emotions: Dict[int, Dict] = {}

        # ── Start time ────────────────────────────────────
        self._started_at: datetime = datetime.now(timezone.utc)

        logger.info("[SpeedLayer] Initialized (feed_size=%d, alert_size=%d)",
                    feed_size, alert_size)

    # ── Ingest methods ──────────────────────────────────

    def ingest_message(
        self,
        ticket_id: int,
        message_id: int,
        role: str,
        content: str,
        session_id: str = ""
    ):
        """
        Gọi khi có message mới (customer hoặc bot).
        """
        self._message_count += 1

        # Chỉ track customer messages trong feed
        if role == "customer":
            self._recent_feed.append({
                "ticket_id": ticket_id,
                "message_id": message_id,
                "content": content[:200],  # truncate để tiết kiệm memory
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": session_id,
            })

    def ingest_emotion(
        self,
        ticket_id: int,
        message_id: int,
        emotion: str,
        confidence: float,
        reason: str,
        alert: bool,
    ):
        """
        Gọi sau khi Orchestrator phân tích xong emotion.
        """
        # Update counters
        self._emotion_counts[emotion] += 1

        if alert:
            self._alert_count += 1
            self._alert_queue.append({
                "ticket_id": ticket_id,
                "message_id": message_id,
                "emotion": emotion,
                "confidence": confidence,
                "reason": reason[:150],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        # Lưu emotion mới nhất cho ticket này
        self._ticket_emotions[ticket_id] = {
            "emotion": emotion,
            "confidence": confidence,
            "reason": reason,
            "alert": alert,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def ingest_new_ticket(self, ticket_id: int):
        """Gọi khi tạo ticket mới."""
        self._ticket_count += 1

    # ── Query methods (dùng cho Serving Layer) ────────

    def get_emotion_counts(self) -> Dict[str, int]:
        """Return dict emotion -> count."""
        return dict(self._emotion_counts)

    def get_stats(self) -> Dict:
        """
        Thống kê tổng quan từ lúc start.
        Merge với batch để có full history.
        """
        return {
            "started_at": self._started_at.isoformat(),
            "message_count": self._message_count,
            "ticket_count": self._ticket_count,
            "alert_count": self._alert_count,
            "emotion_counts": dict(self._emotion_counts),
            "uptime_seconds": (datetime.now(timezone.utc) - self._started_at).total_seconds(),
        }

    def get_recent_feed(self, limit: int = 50) -> List[Dict]:
        """
        N tin nhắn gần nhất (customer only).
        """
        feed_list = list(self._recent_feed)
        return feed_list[-limit:] if limit < len(feed_list) else feed_list

    def get_alert_queue(self, limit: int = 50) -> List[Dict]:
        """
        N alerts gần nhất.
        """
        alerts = list(self._alert_queue)
        return alerts[-limit:] if limit < len(alerts) else alerts

    def get_ticket_emotion(self, ticket_id: int) -> Optional[Dict]:
        """
        Lấy emotion analysis mới nhất cho ticket.
        Return None nếu chưa có.
        """
        return self._ticket_emotions.get(ticket_id)

    def get_all_ticket_emotions(self) -> Dict[int, Dict]:
        """
        Return toàn bộ ticket -> emotion mapping.
        Dùng để sync với batch view.
        """
        return dict(self._ticket_emotions)


# ── Global singleton ───────────────────────────────────

_speed_layer: Optional[SpeedLayer] = None


def init_speed_layer(feed_size: int = 100, alert_size: int = 200) -> SpeedLayer:
    """
    Init singleton instance. Gọi từ main.py lifespan.
    """
    global _speed_layer
    if _speed_layer is None:
        _speed_layer = SpeedLayer(feed_size=feed_size, alert_size=alert_size)
    return _speed_layer


def get_speed_layer() -> Optional[SpeedLayer]:
    """
    Get instance. Return None nếu chưa init.
    """
    return _speed_layer


# ── Helper để ingest từ main.py ───────────────────────

def track_message(ticket_id: int, message_id: int, role: str, content: str, session_id: str = ""):
    """
    Helper để track message từ Speed Layer endpoints.
    """
    layer = get_speed_layer()
    if layer:
        layer.ingest_message(ticket_id, message_id, role, content, session_id)


def track_emotion(ticket_id: int, message_id: int, emotion: str,
                  confidence: float, reason: str, alert: bool):
    """
    Helper để track emotion từ Speed Layer endpoints.
    """
    layer = get_speed_layer()
    if layer:
        layer.ingest_emotion(ticket_id, message_id, emotion, confidence, reason, alert)


def track_new_ticket(ticket_id: int):
    """
    Helper để track new ticket.
    """
    layer = get_speed_layer()
    if layer:
        layer.ingest_new_ticket(ticket_id)