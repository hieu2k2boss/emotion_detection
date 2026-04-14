import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from chatbot_api.database import db

logger = logging.getLogger(__name__)

# ── Write helpers ─────────────────────────────

async def upsert_emotion_hourly(window_start: str, window_end: str,
                               emotion: str, count: int,
                               avg_conf: float, alert_count: int):
    """Lưu kết quả tổng hợp emotion theo giờ vào collection batch_emotion_hourly."""
    try:
        await db.db["batch_emotion_hourly"].update_one(
            {"window_start": window_start, "emotion": emotion},
            {
                "$set": {
                    "window_end": window_end,
                    "count": count,
                    "avg_confidence": avg_conf,
                    "alert_count": alert_count,
                    "updated_at": datetime.utcnow()
                }
            },
            upsert=True
        )
    except Exception as e:
        logger.error(f"Lỗi upsert_emotion_hourly: {e}")

async def upsert_customer_risk(customer_id: int, phone: str, stats: dict):
    """Lưu risk score của khách hàng vào collection batch_customer_risk."""
    try:
        await db.db["batch_customer_risk"].update_one(
            {"customer_id": customer_id},
            {
                "$set": {
                    "phone": phone,
                    **stats,
                    "computed_at": datetime.utcnow()
                }
            },
            upsert=True
        )
    except Exception as e:
        logger.error(f"Lỗi upsert_customer_risk: {e}")

async def insert_alert_daily(report_date: str, stats: dict):
    """Lưu báo cáo alert hàng ngày."""
    try:
        await db.db["batch_alert_daily"].update_one(
            {"report_date": report_date},
            {
                "$set": {
                    **stats,
                    "generated_at": datetime.utcnow()
                }
            },
            upsert=True
        )
    except Exception as e:
        logger.error(f"Lỗi insert_alert_daily: {e}")

async def log_job(job_name: str, status: str, rows: int = 0,
                  duration_ms: int = 0, error: str = ""):
    """Log trạng thái chạy của Batch Job."""
    try:
        await db.db["batch_job_log"].insert_one({
            "job_name": job_name,
            "status": status,
            "rows_processed": rows,
            "duration_ms": duration_ms,
            "error_msg": error,
            "finished_at": datetime.utcnow()
        })
    except Exception as e:
        logger.error(f"Lỗi log_job: {e}")

# ── Read helpers (Serving Layer) ───────────

async def get_emotion_trend(hours: int = 24) -> List[Dict]:
    """Lấy xu hướng cảm xúc trong N giờ qua từ Batch Views."""
    try:
        since = datetime.utcnow() - timedelta(hours=hours)
        # Vì window_start lưu dạng chuỗi, ta cần cẩn thận hoặc chuyển sang date
        cursor = db.db["batch_emotion_hourly"].aggregate([
            {
                "$group": {
                    "_id": "$emotion",
                    "total": {"$sum": "$count"},
                    "avg_conf": {"$avg": "$avg_confidence"},
                    "alerts": {"$sum": "$alert_count"}
                }
            },
            {"$project": {"emotion": "$_id", "total": 1, "avg_conf": 1, "alerts": 1, "_id": 0}},
            {"$sort": {"total": -1}}
        ])
        return await cursor.to_list(length=100)
    except Exception as e:
        logger.error(f"Lỗi get_emotion_trend: {e}")
        return []

async def get_high_risk_customers(limit: int = 20) -> List[Dict]:
    """Lấy danh sách khách hàng có nguy cơ cao."""
    try:
        cursor = db.db["batch_customer_risk"].find(
            {"risk_tier": {"$in": ["high", "critical"]}}
        ).sort("risk_score", -1).limit(limit)
        return await cursor.to_list(length=limit)
    except Exception as e:
        logger.error(f"Lỗi get_high_risk_customers: {e}")
        return []

async def get_alert_report(report_date: str) -> Optional[Dict]:
    """Lấy báo cáo alert theo ngày."""
    try:
        return await db.db["batch_alert_daily"].find_one({"report_date": report_date})
    except Exception as e:
        logger.error(f"Lỗi get_alert_report: {e}")
        return None

async def init_batch_db():
    """Tạo index cho các collection batch views."""
    try:
        await db.db["batch_emotion_hourly"].create_index([("window_start", 1), ("emotion", 1)], unique=True)
        await db.db["batch_customer_risk"].create_index([("customer_id", 1)], unique=True)
        await db.db["batch_alert_daily"].create_index([("report_date", 1)], unique=True)
        logger.info("Hoàn tất khởi tạo Batch Views Index.")
    except Exception as e:
        logger.error(f"Lỗi khởi tạo Batch DB: {e}")
