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

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from batch.kafka_config import BATCH_SCHEDULE, TOPICS
from batch.batch_views_db import (
    upsert_emotion_hourly,
    upsert_customer_risk,
    insert_alert_daily,
    log_job,
    init_batch_db
)
from chatbot_api.database import db

logger = logging.getLogger(__name__)

# ── Job 1: Emotion hourly aggregation ─────────

async def emotion_hourly_job():
    """
    Sử dụng MongoDB Aggregation Pipeline để tổng hợp cảm xúc theo giờ.
    Chạy mỗi giờ, đọc từ collection 'messages'.
    """
    job_name   = "emotion_hourly"
    started_at = time.perf_counter()
    rows_done  = 0

    try:
        now       = datetime.utcnow()
        win_end   = now.replace(minute=0, second=0, microsecond=0)
        win_start = win_end - timedelta(hours=1)

        # Pipeline tính toán thống kê
        pipeline = [
            {
                "$match": {
                    "timestamp": {"$gte": win_start, "$lt": win_end},
                    "ai_analysis": {"$ne": None}
                }
            },
            {
                "$group": {
                    "_id": "$ai_analysis.emotion_label",
                    "cnt": {"$sum": 1},
                    "avg_conf": {"$avg": "$ai_analysis.confidence"},
                    "alert_cnt": {
                        "$sum": {"$cond": [{"$eq": ["$ai_analysis.has_alert", True]}, 1, 0]}
                    }
                }
            },
            {
                "$project": {
                    "emotion": "$_id",
                    "cnt": 1,
                    "avg_conf": 1,
                    "alert_cnt": 1,
                    "_id": 0
                }
            }
        ]

        cursor = db.db["messages"].aggregate(pipeline)
        results = await cursor.to_list(length=100)

        for res in results:
            await upsert_emotion_hourly(
                window_start = win_start.isoformat(),
                window_end   = win_end.isoformat(),
                emotion      = res["emotion"],
                count        = res["cnt"],
                avg_conf     = res["avg_conf"] or 0.0,
                alert_count  = res["alert_cnt"] or 0,
            )
            rows_done += 1

        duration = int((time.perf_counter() - started_at) * 1000)
        await log_job(job_name, "success", rows_done, duration)
        logger.info(f"[Job] {job_name}: {rows_done} emotions aggregated")

    except Exception as e:
        duration = int((time.perf_counter() - started_at) * 1000)
        await log_job(job_name, "failed", rows_done, duration, str(e))
        logger.error(f"[Job] {job_name} failed: {e}")

# ── Job 2: Customer risk segmentation ─────────

async def customer_segment_job():
    """
    Tính risk score cho khách hàng sử dụng Aggregation Pipeline.
    """
    job_name   = "customer_segment"
    started_at = time.perf_counter()
    rows_done  = 0

    try:
        # Pipeline phức tạp để tính stats từ messages nhúng ai_analysis
        pipeline = [
            {"$match": {"ai_analysis": {"$ne": None}}},
            {
                "$group": {
                    "_id": "$ticket_id",
                    "customer_id": {"$first": "$customer_id"}, # Giả định ticket có customer_id
                    "angry_count": {"$sum": {"$cond": [{"$eq": ["$ai_analysis.emotion_label", "angry"]}, 1, 0]}},
                    "disappointed_count": {"$sum": {"$cond": [{"$eq": ["$ai_analysis.emotion_label", "disappointed"]}, 1, 0]}},
                    "alert_count": {"$sum": {"$cond": [{"$eq": ["$ai_analysis.has_alert", True]}, 1, 0]}},
                    "avg_conf": {"$avg": "$ai_analysis.confidence"},
                    "last_emotion": {"$last": "$ai_analysis.emotion_label"}
                }
            },
            {
                "$group": {
                    "_id": "$customer_id",
                    "total_tickets": {"$sum": 1},
                    "angry_count": {"$sum": "$angry_count"},
                    "disappointed_count": {"$sum": "$disappointed_count"},
                    "alert_count": {"$sum": "$alert_count"},
                    "avg_confidence": {"$avg": "$avg_conf"},
                    "last_emotion": {"$last": "$last_emotion"}
                }
            }
        ]

        cursor = db.db["messages"].aggregate(pipeline)
        customers = await cursor.to_list(length=1000)

        def _compute_risk_score(angry, disappointed, alerts, total):
            if total == 0: return 0.0
            raw = (angry * 1.0 + disappointed * 0.7 + alerts * 0.5) / total
            return min(round(raw, 3), 1.0)

        for c in customers:
            if not c["_id"]: continue
            score = _compute_risk_score(c["angry_count"], c["disappointed_count"], c["alert_count"], c["total_tickets"])
            
            stats = {
                "total_tickets": c["total_tickets"],
                "angry_count": c["angry_count"],
                "disappointed_count": c["disappointed_count"],
                "alert_count": c["alert_count"],
                "avg_confidence": round(c["avg_confidence"] or 0.0, 3),
                "risk_score": score,
                "risk_tier": "critical" if score >= 0.7 else "high" if score >= 0.4 else "medium" if score >= 0.2 else "low",
                "last_emotion": c["last_emotion"] or "neutral"
            }
            # Lấy phone từ collection khách hàng
            cust_doc = await db.db["customers"].find_one({"id": c["_id"]})
            phone = cust_doc["phone"] if cust_doc else "unknown"

            await upsert_customer_risk(c["_id"], phone, stats)
            rows_done += 1

        duration = int((time.perf_counter() - started_at) * 1000)
        await log_job(job_name, "success", rows_done, duration)
        logger.info(f"[Job] {job_name}: {rows_done} customers scored")

    except Exception as e:
        duration = int((time.perf_counter() - started_at) * 1000)
        await log_job(job_name, "failed", rows_done, duration, str(e))
        logger.error(f"[Job] {job_name} failed: {e}")

# ── Scheduler setup ───────────────────────────

def create_scheduler() -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone="Asia/Ho_Chi_Minh")

    scheduler.add_job(
        emotion_hourly_job,
        CronTrigger(**BATCH_SCHEDULE["emotion_hourly"]),
        id="emotion_hourly"
    )

    scheduler.add_job(
        customer_segment_job,
        CronTrigger(**BATCH_SCHEDULE["customer_segment"]),
        id="customer_segment"
    )

    logger.info("[Scheduler] Async jobs registered")
    return scheduler

async def run_all_jobs_now():
    logger.info("[Jobs] Running all batch jobs manually...")
    await emotion_hourly_job()
    await customer_segment_job()
    logger.info("[Jobs] All done.")
