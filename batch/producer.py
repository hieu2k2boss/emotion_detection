"""
batch/producer.py
─────────────────
Kafka Producer cho Speed Layer.
Được gọi từ main.py sau mỗi lần save_message() và save_emotion().

Thiết kế: fire-and-forget có retry, KHÔNG block API response.
Nếu Kafka down → log warning, tiếp tục bình thường (graceful degrade).
"""
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Optional

from kafka import KafkaProducer
from kafka.errors import KafkaError, NoBrokersAvailable

from batch.kafka_config import PRODUCER_CONFIG, TOPICS

logger = logging.getLogger(__name__)


def _make_serializer():
    return lambda v: json.dumps(v, ensure_ascii=False, default=str).encode("utf-8")

def _make_key_serializer():
    return lambda k: str(k).encode("utf-8") if k is not None else None


class CSKHProducer:
    """
    Thread-safe Kafka Producer wrapper.
    Dùng singleton pattern — chỉ khởi tạo 1 lần khi app start.
    """

    _instance: Optional["CSKHProducer"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._producer: Optional[KafkaProducer] = None
        self._available = False
        self._connect()

    # ── Singleton ────────────────────────────────

    @classmethod
    def get_instance(cls) -> "CSKHProducer":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── Connection ───────────────────────────────

    def _connect(self):
        try:
            cfg = dict(PRODUCER_CONFIG)
            cfg["value_serializer"] = _make_serializer()
            cfg["key_serializer"]   = _make_key_serializer()
            self._producer  = KafkaProducer(**cfg)
            self._available = True
            logger.info("[Producer] Kafka connected: %s", PRODUCER_CONFIG["bootstrap_servers"])
        except NoBrokersAvailable:
            logger.warning("[Producer] Kafka không khả dụng — chạy ở chế độ degraded")
            self._available = False
        except Exception as e:
            logger.warning("[Producer] Lỗi kết nối Kafka: %s", e)
            self._available = False

    def _send(self, topic: str, key: str, value: dict):
        """Internal send với error handling."""
        if not self._available or self._producer is None:
            return

        def on_error(exc):
            logger.error("[Producer] Gửi thất bại topic=%s key=%s: %s", topic, key, exc)

        try:
            self._producer.send(topic, key=key, value=value).add_errback(on_error)
        except KafkaError as e:
            logger.error("[Producer] KafkaError: %s", e)
        except Exception as e:
            logger.error("[Producer] Unexpected error: %s", e)

    # ── Public publish methods ────────────────────

    def publish_message(
        self,
        ticket_id:  int,
        message_id: int,
        role:       str,
        content:    str,
        session_id: str = "",
    ):
        """Publish khi có tin nhắn mới (customer hoặc bot)."""
        self._send(
            topic=TOPICS["messages"],
            key=str(ticket_id),
            value={
                "event_type": "message",
                "ticket_id":  ticket_id,
                "message_id": message_id,
                "role":       role,
                "content":    content,
                "session_id": session_id,
                "ts":         datetime.now(timezone.utc).isoformat(),
            },
        )

    def publish_emotion(
        self,
        ticket_id:  int,
        message_id: int,
        emotion:    str,
        confidence: float,
        reason:     str,
        alert:      bool,
    ):
        """Publish kết quả phân tích cảm xúc."""
        payload = {
            "event_type": "emotion",
            "ticket_id":  ticket_id,
            "message_id": message_id,
            "emotion":    emotion,
            "confidence": confidence,
            "reason":     reason,
            "alert":      alert,
            "ts":         datetime.now(timezone.utc).isoformat(),
        }
        self._send(TOPICS["emotions"], key=str(ticket_id), value=payload)

        # Nếu alert → publish thêm vào topic alerts để monitor real-time
        if alert:
            self._send(TOPICS["alerts"], key=str(ticket_id), value=payload)

    def flush(self, timeout: float = 5.0):
        """Flush pending messages trước khi shutdown."""
        if self._producer and self._available:
            self._producer.flush(timeout=timeout)

    def close(self):
        self.flush()
        if self._producer:
            self._producer.close()
        self._available = False
        logger.info("[Producer] Kafka producer closed")


# ── Module-level convenience functions ─────────
# Dùng trong main.py:
#   from batch.producer import publish_message, publish_emotion

def publish_message(ticket_id, message_id, role, content, session_id=""):
    CSKHProducer.get_instance().publish_message(
        ticket_id, message_id, role, content, session_id
    )

def publish_emotion(ticket_id, message_id, emotion, confidence, reason, alert):
    CSKHProducer.get_instance().publish_emotion(
        ticket_id, message_id, emotion, confidence, reason, alert
    )

def close_producer():
    CSKHProducer.get_instance().close()
