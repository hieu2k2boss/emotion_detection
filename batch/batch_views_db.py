"""
batch/batch_views_db.py
───────────────────────
SQLite tables lưu kết quả Batch Layer.
Tách riêng khỏi db.py gốc để không làm hỏng Speed Layer.

Schema thiết kế theo nguyên tắc Lambda:
  batch_* tables = immutable batch views (ghi đè theo window)
  Serving Layer merge với speed views khi query.
"""
import sqlite3
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent
BATCH_DB_PATH = PROJECT_ROOT / ".cache" / "batch_views.db"


def get_conn() -> sqlite3.Connection:
    BATCH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(BATCH_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # concurrent reads
    return conn


def init_batch_db():
    conn = get_conn()
    conn.executescript("""
    -- ── Emotion aggregation theo giờ ──────────────────────────
    CREATE TABLE IF NOT EXISTS batch_emotion_hourly (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        window_start TEXT NOT NULL,          -- "2024-05-10T08:00:00"
        window_end   TEXT NOT NULL,          -- "2024-05-10T09:00:00"
        emotion      TEXT NOT NULL,
        count        INTEGER NOT NULL DEFAULT 0,
        avg_confidence REAL NOT NULL DEFAULT 0.0,
        alert_count  INTEGER NOT NULL DEFAULT 0,
        updated_at   TEXT DEFAULT (datetime('now')),
        UNIQUE(window_start, emotion)        -- upsert-safe
    );

    -- ── Customer risk scores (cập nhật mỗi 4 tiếng) ──────────
    CREATE TABLE IF NOT EXISTS batch_customer_risk (
        customer_id     INTEGER PRIMARY KEY,
        phone           TEXT,
        total_tickets   INTEGER DEFAULT 0,
        angry_count     INTEGER DEFAULT 0,
        disappointed_count INTEGER DEFAULT 0,
        alert_count     INTEGER DEFAULT 0,
        avg_confidence  REAL    DEFAULT 0.0,
        risk_score      REAL    DEFAULT 0.0,   -- 0.0 → 1.0
        risk_tier       TEXT    DEFAULT 'low', -- low | medium | high | critical
        last_emotion    TEXT,
        computed_at     TEXT DEFAULT (datetime('now'))
    );

    -- ── Alert summary theo ngày (báo cáo sáng) ───────────────
    CREATE TABLE IF NOT EXISTS batch_alert_daily (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        report_date  TEXT NOT NULL UNIQUE,     -- "2024-05-10"
        total_alerts INTEGER DEFAULT 0,
        angry_alerts INTEGER DEFAULT 0,
        disappointed_alerts INTEGER DEFAULT 0,
        top_emotion  TEXT,
        avg_response_confidence REAL DEFAULT 0.0,
        generated_at TEXT DEFAULT (datetime('now'))
    );

    -- ── Raw event store (Kafka consumer ghi vào) ─────────────
    CREATE TABLE IF NOT EXISTS batch_raw_events (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        topic      TEXT NOT NULL,
        event_type TEXT NOT NULL,
        ticket_id  INTEGER,
        message_id INTEGER,
        payload    TEXT NOT NULL,              -- JSON string
        kafka_offset INTEGER,
        received_at TEXT DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_raw_events_topic    ON batch_raw_events(topic);
    CREATE INDEX IF NOT EXISTS idx_raw_events_ticket   ON batch_raw_events(ticket_id);
    CREATE INDEX IF NOT EXISTS idx_raw_events_received ON batch_raw_events(received_at);

    -- ── Batch job log ────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS batch_job_log (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        job_name    TEXT NOT NULL,
        status      TEXT NOT NULL,   -- running | success | failed
        rows_processed INTEGER DEFAULT 0,
        duration_ms INTEGER,
        error_msg   TEXT,
        started_at  TEXT DEFAULT (datetime('now')),
        finished_at TEXT
    );
    """)
    conn.commit()
    conn.close()


# ── Write helpers ─────────────────────────────

def upsert_emotion_hourly(window_start: str, window_end: str,
                           emotion: str, count: int,
                           avg_conf: float, alert_count: int):
    conn = get_conn()
    conn.execute("""
        INSERT INTO batch_emotion_hourly
            (window_start, window_end, emotion, count, avg_confidence, alert_count, updated_at)
        VALUES (?,?,?,?,?,?, datetime('now'))
        ON CONFLICT(window_start, emotion) DO UPDATE SET
            window_end      = excluded.window_end,
            count           = excluded.count,
            avg_confidence  = excluded.avg_confidence,
            alert_count     = excluded.alert_count,
            updated_at      = datetime('now')
    """, (window_start, window_end, emotion, count, avg_conf, alert_count))
    conn.commit()
    conn.close()


def upsert_customer_risk(customer_id: int, phone: str, stats: dict):
    conn = get_conn()
    conn.execute("""
        INSERT INTO batch_customer_risk
            (customer_id, phone, total_tickets, angry_count, disappointed_count,
             alert_count, avg_confidence, risk_score, risk_tier, last_emotion, computed_at)
        VALUES (?,?,?,?,?,?,?,?,?,?, datetime('now'))
        ON CONFLICT(customer_id) DO UPDATE SET
            phone               = excluded.phone,
            total_tickets       = excluded.total_tickets,
            angry_count         = excluded.angry_count,
            disappointed_count  = excluded.disappointed_count,
            alert_count         = excluded.alert_count,
            avg_confidence      = excluded.avg_confidence,
            risk_score          = excluded.risk_score,
            risk_tier           = excluded.risk_tier,
            last_emotion        = excluded.last_emotion,
            computed_at         = datetime('now')
    """, (
        customer_id, phone,
        stats["total_tickets"], stats["angry_count"], stats["disappointed_count"],
        stats["alert_count"],   stats["avg_confidence"],
        stats["risk_score"],    stats["risk_tier"], stats["last_emotion"],
    ))
    conn.commit()
    conn.close()


def insert_alert_daily(report_date: str, stats: dict):
    conn = get_conn()
    conn.execute("""
        INSERT INTO batch_alert_daily
            (report_date, total_alerts, angry_alerts, disappointed_alerts,
             top_emotion, avg_response_confidence, generated_at)
        VALUES (?,?,?,?,?,?, datetime('now'))
        ON CONFLICT(report_date) DO UPDATE SET
            total_alerts               = excluded.total_alerts,
            angry_alerts               = excluded.angry_alerts,
            disappointed_alerts        = excluded.disappointed_alerts,
            top_emotion                = excluded.top_emotion,
            avg_response_confidence    = excluded.avg_response_confidence,
            generated_at               = datetime('now')
    """, (
        report_date,
        stats["total_alerts"], stats["angry_alerts"], stats["disappointed_alerts"],
        stats["top_emotion"],  stats["avg_response_confidence"],
    ))
    conn.commit()
    conn.close()


def insert_raw_event(topic: str, event_type: str, ticket_id: Optional[int],
                      message_id: Optional[int], payload: str, kafka_offset: Optional[int] = None):
    conn = get_conn()
    conn.execute("""
        INSERT INTO batch_raw_events (topic, event_type, ticket_id, message_id, payload, kafka_offset)
        VALUES (?,?,?,?,?,?)
    """, (topic, event_type, ticket_id, message_id, payload, kafka_offset))
    conn.commit()
    conn.close()


def log_job(job_name: str, status: str, rows: int = 0,
            duration_ms: int = 0, error: str = ""):
    conn = get_conn()
    conn.execute("""
        INSERT INTO batch_job_log (job_name, status, rows_processed, duration_ms, error_msg, finished_at)
        VALUES (?,?,?,?,?, datetime('now'))
    """, (job_name, status, rows, duration_ms, error or None))
    conn.commit()
    conn.close()


# ── Read helpers (Serving Layer dùng) ────────

def get_emotion_trend(hours: int = 24) -> list:
    """Lấy emotion distribution trong N giờ qua — cho dashboard."""
    conn = get_conn()
    rows = conn.execute("""
        SELECT emotion, SUM(count) as total, AVG(avg_confidence) as avg_conf,
               SUM(alert_count) as alerts
        FROM batch_emotion_hourly
        WHERE window_start >= datetime('now', ? || ' hours')
        GROUP BY emotion
        ORDER BY total DESC
    """, (f"-{hours}",)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_high_risk_customers(limit: int = 20) -> list:
    conn = get_conn()
    rows = conn.execute("""
        SELECT * FROM batch_customer_risk
        WHERE risk_tier IN ('high', 'critical')
        ORDER BY risk_score DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_alert_report(report_date: str) -> Optional[dict]:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM batch_alert_daily WHERE report_date = ?", (report_date,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_customer_risk(customer_id: int) -> Optional[dict]:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM batch_customer_risk WHERE customer_id = ?", (customer_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None
