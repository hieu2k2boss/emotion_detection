"""
batch/consumer.py
─────────────────
Kafka Consumer — chạy như một tiến trình riêng biệt (không trong FastAPI).
Nhiệm vụ: consume events từ Kafka → ghi vào batch_raw_events.
Batch jobs sẽ đọc từ bảng này sau.

Chạy:
    python -m batch.consumer

Hoặc dùng Docker/supervisor để chạy song song với FastAPI.
"""
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Set

# Thêm project root vào path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kafka import KafkaConsumer
from kafka.errors import KafkaError, NoBrokersAvailable

from batch.kafka_config import (
    CONSUMER_CONFIG,
    GROUP_IDS,
    TOPICS,
)
from batch.batch_views_db import insert_raw_event, init_batch_db
from batch.jobs import run_all_jobs_now

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("consumer")

# ── Graceful shutdown ─────────────────────────
_running = True

def _handle_signal(sig, frame):
    global _running
    logger.info("[Consumer] Nhận signal %s → dừng sau poll hiện tại...", sig)
    _running = False

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ── Consumer class ────────────────────────────

class CSKHConsumer:
    """
    Consume từ tất cả cskh.* topics → lưu raw events.
    Dùng manual commit để đảm bảo exactly-once ghi vào DB.
    """

    ALL_TOPICS: list = list(TOPICS.values())

    def __init__(self, group_id: str = GROUP_IDS["raw_store"]):
        self.group_id  = group_id
        self._consumer = None
        self._connect()

    def _connect(self):
        cfg = dict(CONSUMER_CONFIG)
        cfg["value_deserializer"] = lambda v: json.loads(v.decode("utf-8"))
        cfg["group_id"]           = self.group_id

        while True:
            try:
                self._consumer = KafkaConsumer(*self.ALL_TOPICS, **cfg)
                logger.info("[Consumer] Subscribed to: %s", self.ALL_TOPICS)
                break
            except NoBrokersAvailable:
                logger.warning("[Consumer] Kafka chưa sẵn sàng, retry sau 5s...")
                time.sleep(5)

    def _process_message(self, msg) -> bool:
        """
        Xử lý 1 message. Trả về True nếu thành công.
        """
        try:
            payload   = msg.value
            topic     = msg.topic
            offset    = msg.offset
            event_type = payload.get("event_type", "unknown")
            ticket_id  = payload.get("ticket_id")
            message_id = payload.get("message_id")

            insert_raw_event(
                topic        = topic,
                event_type   = event_type,
                ticket_id    = ticket_id,
                message_id   = message_id,
                payload      = json.dumps(payload, ensure_ascii=False),
                kafka_offset = offset,
            )
            return True

        except Exception as e:
            logger.error("[Consumer] Lỗi xử lý message offset=%s: %s", msg.offset, e)
            return False

    def run(self, poll_timeout_ms: int = 1000):
        """Main loop — poll → process → commit."""
        logger.info("[Consumer] Starting main loop (group=%s)...", self.group_id)

        while _running:
            try:
                records = self._consumer.poll(timeout_ms=poll_timeout_ms)
                if not records:
                    continue

                total      = sum(len(msgs) for msgs in records.values())
                processed  = 0

                for tp, messages in records.items():
                    for msg in messages:
                        if self._process_message(msg):
                            processed += 1

                # Commit sau khi xử lý xong cả batch
                self._consumer.commit()
                logger.debug("[Consumer] Committed %d/%d messages", processed, total)

            except KafkaError as e:
                logger.error("[Consumer] KafkaError: %s", e)
                time.sleep(2)
            except Exception as e:
                logger.error("[Consumer] Unexpected error: %s", e, exc_info=True)
                time.sleep(2)

        self._consumer.close()
        logger.info("[Consumer] Đã dừng cleanly.")


# ── Entry point ───────────────────────────────

def main():
    logger.info("[Consumer] Initializing batch DB...")
    init_batch_db()

    # Chạy batch jobs ngay khi start để backfill nếu cần
    if os.getenv("RUN_JOBS_ON_START", "false").lower() == "true":
        logger.info("[Consumer] RUN_JOBS_ON_START=true → chạy jobs ngay...")
        run_all_jobs_now()

    consumer = CSKHConsumer()
    consumer.run()


if __name__ == "__main__":
    main()
