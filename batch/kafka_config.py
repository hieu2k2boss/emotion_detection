import os

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

# ── Topic definitions ─────────────────────────
TOPICS = {
    "messages":   "cskh.messages",    # Mỗi tin nhắn chat mới
    "emotions":   "cskh.emotions",    # Kết quả phân tích cảm xúc
    "alerts":     "cskh.alerts",      # Cảnh báo escalation
    "batch_done": "cskh.batch.done",  # Signal hoàn thành batch job
}

# ── Consumer group IDs ────────────────────────
GROUP_IDS = {
    "raw_store":        "cskh-raw-store",        # Lưu raw events xuống disk
    "emotion_agg":      "cskh-emotion-agg",      # Tổng hợp cảm xúc
    "alert_monitor":    "cskh-alert-monitor",    # Giám sát alert real-time
}

# ── Batch job schedule (APScheduler cron) ─────
BATCH_SCHEDULE = {
    "emotion_hourly":   {"hour": "*",  "minute": "0"},   # Mỗi giờ
    "customer_segment": {"hour": "*/4","minute": "30"},  # Mỗi 4 tiếng
    "alert_report":     {"hour": "8",  "minute": "0"},   # 8h sáng hàng ngày
    "accuracy_eval":    {"hour": "2",  "minute": "0"},   # 2h sáng hàng ngày
}

# ── Kafka producer config ─────────────────────
PRODUCER_CONFIG = {
    "bootstrap_servers": KAFKA_BOOTSTRAP_SERVERS,
    "value_serializer":  None,   # Set trong CSKHProducer
    "key_serializer":    None,
    "acks":              "all",  # Đảm bảo không mất message
    "retries":           3,
    "max_block_ms":      5000,   # Timeout nếu Kafka down → không block API
}

# ── Kafka consumer config ─────────────────────
CONSUMER_CONFIG = {
    "bootstrap_servers":  KAFKA_BOOTSTRAP_SERVERS,
    "auto_offset_reset":  "earliest",
    "enable_auto_commit": False,   # Manual commit để không mất data
    "max_poll_records":   100,
    "session_timeout_ms": 30000,
}
