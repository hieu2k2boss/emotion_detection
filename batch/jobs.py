"""
batch/jobs.py
─────────────
Batch processing jobs chạy theo lịch (APScheduler).
Đọc từ batch_raw_events (đã được consumer ghi) hoặc query thẳng cskh.db.

Jobs:
  1. emotion_hourly_job     — tổng hợp emotion per hour window
  2. customer_segment_job   — tính risk score từng khách
  3. alert_report_job       — báo cáo alert hàng ngày
  4. accuracy_eval_job      — đánh giá độ chính xác (optional)
"""
import json
import logging
import time
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from batch.kafka_config import BATCH_SCHEDULE, TOPICS
from batch.batch_views_db import (
    upsert_emotion_hourly,
    upsert_customer_risk,
    insert_alert_daily,
    log_job,
    get_conn as get_batch_conn,
)

# Import db gốc của Speed Layer
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import chatbot_api.db as speed_db

logger = logging.getLogger(__name__)

# ── Emotion labels ────────────────────────────
EMOTIONS       = ["neutral", "happy", "confused", "anxious", "frustrated", "disappointed", "angry"]
DANGER_EMOTIONS = {"angry", "disappointed"}


# ── Job 1: Emotion hourly aggregation ─────────

def emotion_hourly_job():
    """
    Chạy mỗi giờ. Đọc emotion_logs từ cskh.db trong 1 giờ qua,
    group by emotion → ghi vào batch_emotion_hourly.
    """
    job_name   = "emotion_hourly"
    started_at = time.perf_counter()
    rows_done  = 0

    try:
        now       = datetime.now(timezone.utc)
        win_end   = now.replace(minute=0, second=0, microsecond=0)
        win_start = win_end - timedelta(hours=1)

        win_start_str = win_start.strftime("%Y-%m-%dT%H:%M:%S")
        win_end_str   = win_end.strftime("%Y-%m-%dT%H:%M:%S")

        # Query cskh.db (Speed Layer database)
        conn  = speed_db.get_conn()
        rows  = conn.execute("""
            SELECT emotion,
                   COUNT(*)         as cnt,
                   AVG(confidence)  as avg_conf,
                   SUM(alert)       as alert_cnt
            FROM emotion_logs
            WHERE created_at >= ? AND created_at < ?
            GROUP BY emotion
        """, (win_start_str, win_end_str)).fetchall()
        conn.close()

        for row in rows:
            upsert_emotion_hourly(
                window_start = win_start_str,
                window_end   = win_end_str,
                emotion      = row["emotion"],
                count        = row["cnt"],
                avg_conf     = row["avg_conf"] or 0.0,
                alert_count  = row["alert_cnt"] or 0,
            )
            rows_done += 1

        duration = int((time.perf_counter() - started_at) * 1000)
        log_job(job_name, "success", rows_done, duration)
        logger.info("[Job] %s: %d emotions aggregated (window: %s → %s, %dms)",
                    job_name, rows_done, win_start_str, win_end_str, duration)

    except Exception as e:
        duration = int((time.perf_counter() - started_at) * 1000)
        log_job(job_name, "failed", rows_done, duration, str(e))
        logger.error("[Job] %s failed: %s", job_name, e, exc_info=True)


# ── Job 2: Customer risk segmentation ─────────

def _compute_risk_score(angry: int, disappointed: int, alerts: int, total: int) -> float:
    """
    Risk score 0.0 → 1.0.
    Công thức: weighted sum chuẩn hoá theo total tickets.
    """
    if total == 0:
        return 0.0
    raw = (angry * 1.0 + disappointed * 0.7 + alerts * 0.5) / total
    return min(round(raw, 3), 1.0)

def _risk_tier(score: float) -> str:
    if score >= 0.7: return "critical"
    if score >= 0.4: return "high"
    if score >= 0.2: return "medium"
    return "low"

def customer_segment_job():
    """
    Chạy mỗi 4 tiếng. Tính risk score cho tất cả khách hàng
    đã từng có ticket.
    """
    job_name   = "customer_segment"
    started_at = time.perf_counter()
    rows_done  = 0

    try:
        conn = speed_db.get_conn()

        # Lấy stats per customer từ emotion_logs + tickets
        rows = conn.execute("""
            SELECT
                c.id               as customer_id,
                c.phone,
                COUNT(DISTINCT t.id)                              as total_tickets,
                SUM(CASE WHEN el.emotion='angry' THEN 1 ELSE 0 END) as angry_count,
                SUM(CASE WHEN el.emotion='disappointed' THEN 1 ELSE 0 END) as disappointed_count,
                SUM(el.alert)                                     as alert_count,
                AVG(el.confidence)                                as avg_confidence,
                MAX(el.emotion)                                   as last_emotion
            FROM customers c
            JOIN tickets t ON t.customer_id = c.id
            LEFT JOIN emotion_logs el ON el.ticket_id = t.id
            GROUP BY c.id, c.phone
        """).fetchall()
        conn.close()

        for row in rows:
            score = _compute_risk_score(
                angry       = row["angry_count"] or 0,
                disappointed= row["disappointed_count"] or 0,
                alerts      = row["alert_count"] or 0,
                total       = row["total_tickets"] or 1,
            )
            upsert_customer_risk(
                customer_id = row["customer_id"],
                phone       = row["phone"],
                stats       = {
                    "total_tickets":       row["total_tickets"] or 0,
                    "angry_count":         row["angry_count"] or 0,
                    "disappointed_count":  row["disappointed_count"] or 0,
                    "alert_count":         row["alert_count"] or 0,
                    "avg_confidence":      round(row["avg_confidence"] or 0.0, 3),
                    "risk_score":          score,
                    "risk_tier":           _risk_tier(score),
                    "last_emotion":        row["last_emotion"] or "neutral",
                },
            )
            rows_done += 1

        duration = int((time.perf_counter() - started_at) * 1000)
        log_job(job_name, "success", rows_done, duration)
        logger.info("[Job] %s: %d customers scored (%dms)", job_name, rows_done, duration)

    except Exception as e:
        duration = int((time.perf_counter() - started_at) * 1000)
        log_job(job_name, "failed", rows_done, duration, str(e))
        logger.error("[Job] %s failed: %s", job_name, e, exc_info=True)


# ── Job 3: Daily alert report ─────────────────

def alert_report_job():
    """
    Chạy lúc 8h sáng. Tổng hợp alert của ngày hôm qua.
    """
    job_name   = "alert_report"
    started_at = time.perf_counter()

    try:
        yesterday    = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        date_start   = f"{yesterday}T00:00:00"
        date_end     = f"{yesterday}T23:59:59"

        conn = speed_db.get_conn()
        row  = conn.execute("""
            SELECT
                COUNT(*)                                              as total_alerts,
                SUM(CASE WHEN emotion='angry' THEN 1 ELSE 0 END)     as angry_alerts,
                SUM(CASE WHEN emotion='disappointed' THEN 1 ELSE 0 END) as disappointed_alerts,
                AVG(confidence)                                       as avg_conf
            FROM emotion_logs
            WHERE alert = 1 AND created_at BETWEEN ? AND ?
        """, (date_start, date_end)).fetchone()

        top_row = conn.execute("""
            SELECT emotion, COUNT(*) as cnt
            FROM emotion_logs
            WHERE created_at BETWEEN ? AND ?
            GROUP BY emotion ORDER BY cnt DESC LIMIT 1
        """, (date_start, date_end)).fetchone()
        conn.close()

        insert_alert_daily(
            report_date = yesterday,
            stats       = {
                "total_alerts":             row["total_alerts"] or 0,
                "angry_alerts":             row["angry_alerts"] or 0,
                "disappointed_alerts":      row["disappointed_alerts"] or 0,
                "top_emotion":              top_row["emotion"] if top_row else "neutral",
                "avg_response_confidence":  round(row["avg_conf"] or 0.0, 3),
            },
        )

        duration = int((time.perf_counter() - started_at) * 1000)
        log_job(job_name, "success", 1, duration)
        logger.info("[Job] %s: report for %s generated (%dms)", job_name, yesterday, duration)

    except Exception as e:
        duration = int((time.perf_counter() - started_at) * 1000)
        log_job(job_name, "failed", 0, duration, str(e))
        logger.error("[Job] %s failed: %s", job_name, e, exc_info=True)


# ── Scheduler setup ───────────────────────────

def create_scheduler() -> BackgroundScheduler:
    """
    Tạo và cấu hình APScheduler với tất cả batch jobs.
    Gọi từ main.py trong lifespan context.
    """
    scheduler = BackgroundScheduler(timezone="Asia/Ho_Chi_Minh")

    # Job 1: Emotion hourly — mỗi đầu giờ
    scheduler.add_job(
        emotion_hourly_job,
        CronTrigger(**BATCH_SCHEDULE["emotion_hourly"]),
        id="emotion_hourly",
        name="Emotion hourly aggregation",
        misfire_grace_time=300,   # Nếu trễ < 5 phút thì vẫn chạy
        coalesce=True,            # Không chạy bù nếu miss nhiều lần
    )

    # Job 2: Customer segmentation — mỗi 4 tiếng
    scheduler.add_job(
        customer_segment_job,
        CronTrigger(**BATCH_SCHEDULE["customer_segment"]),
        id="customer_segment",
        name="Customer risk segmentation",
        misfire_grace_time=600,
        coalesce=True,
    )

    # Job 3: Daily alert report — 8h sáng
    scheduler.add_job(
        alert_report_job,
        CronTrigger(**BATCH_SCHEDULE["alert_report"]),
        id="alert_report",
        name="Daily alert report",
        misfire_grace_time=1800,
        coalesce=True,
    )

    logger.info("[Scheduler] %d jobs registered", len(scheduler.get_jobs()))
    return scheduler


def run_all_jobs_now():
    """
    Chạy tất cả jobs ngay lập tức — dùng để test hoặc backfill.
    """
    logger.info("[Jobs] Running all batch jobs manually...")
    emotion_hourly_job()
    customer_segment_job()
    alert_report_job()
    logger.info("[Jobs] All done.")
