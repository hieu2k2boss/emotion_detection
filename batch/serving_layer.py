"""
batch/serving_layer.py
─────────────────────
Serving Layer — merge Batch View + Speed Layer.
Trả về kết quả accurate nhất cho Admin Dashboard.
"""
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any

from batch.batch_views_db import (
    get_emotion_trend,
    get_high_risk_customers,
    get_alert_report,
)
from batch.speed_layer import get_speed_layer

logger = logging.getLogger(__name__)


# ── Merge: Batch + Speed ─────────────────────────────

def merge_emotion_stats(hours: int = 24) -> Dict:
    """
    Merge emotion statistics:
      - Batch: pre-computed views (đến batch gần nhất)
      - Speed: realtime counters (từ batch đến hiện tại)

    Return:
        {
          "hours": hours,
          "batch_emotion_trend": {...},  # từ batch_emotion_hourly
          "speed_delta": {...},           #增量từ Speed Layer
          "merged_emotion_dist": {...},   # tổng hợp
        }
    """
    # 1. Batch view — emotion trend theo giờ
    batch_data = get_emotion_trend(hours=hours) or []

    # 2. Speed Layer — counters từ lúc start
    layer = get_speed_layer()
    speed_counts: Dict[str, int] = {}
    speed_alerts = 0

    if layer:
        stats = layer.get_stats()
        speed_counts = stats.get("emotion_counts", {})
        speed_alerts = stats.get("alert_count", 0)

    # 3. Merge — cộng batch + speed
    merged_counts: Dict[str, int] = {}

    # Start với batch totals
    for row in batch_data:
        emotion = row.get("emotion")
        count = row.get("count", 0)
        if emotion:
            merged_counts[emotion] = merged_counts.get(emotion, 0) + count

    # Add speed delta
    for emotion, count in speed_counts.items():
        merged_counts[emotion] = merged_counts.get(emotion, 0) + count

    return {
        "hours": hours,
        "batch_view_count": len(batch_data),
        "speed_counts": speed_counts,
        "merged_emotion_distribution": merged_counts,
        "total_messages": sum(merged_counts.values()),
        "total_alerts": speed_alerts,  # Speed Layer only (batch chưa track tổng alerts)
        "source": "serving_layer_merged",
    }


def merge_high_risk_customers(limit: int = 20) -> Dict:
    """
    Merge high risk customers:
      - Batch: computed risk score (chạy 4 tiếng/lần)
      - Speed: realtime emotion gần nhất

    Return:
        {
          "customers": [
            {
              "customer_id": ...,
              "phone": ...,
              "batch_risk_score": ...,
              "batch_risk_tier": ...,
              "speed_recent_emotion": ...,
              "speed_alerted_recently": bool,
              "merged_risk": "critical" | "high" | "medium" | "low",
            },
            ...
          ]
        }
    """
    # 1. Batch view
    batch_customers = get_high_risk_customers(limit=limit) or []

    # 2. Speed Layer — recent emotions
    layer = get_speed_layer()
    ticket_emotions: Dict[int, Dict] = {}
    if layer:
        ticket_emotions = layer.get_all_ticket_emotions()

    # 3. Merge each customer
    merged_customers = []

    for cust in batch_customers:
        customer_id = cust.get("customer_id")

        # Lấy emotion data từ Speed Layer (ticket → customer mapping tạm thời)
        # TODO: thêm ticket_id trong batch view để mapping chính xác hơn
        recent_emotion = None
        recent_alert = False

        if ticket_emotions:
            # Tìm emotion mới nhất (vịc tạm thời — cần cải thiện)
            for ticket_id, emotion_data in ticket_emotions.items():
                if emotion_data.get("alert"):
                    recent_alert = True
                    recent_emotion = emotion_data.get("emotion")
                    break

            if not recent_emotion and ticket_emotions:
                # Lấy emotion đầu tiên tìm được
                recent_emotion = next(iter(ticket_emotions.values())).get("emotion")

        # Re-compute risk tier có tính speed data
        batch_risk = cust.get("risk_score", 0.0)
        batch_tier = cust.get("risk_tier", "low")

        if recent_alert or recent_emotion in ["angry", "disappointed"]:
            # Boost risk tier nếu có realtime alert
            merged_tier = "critical" if batch_tier in ["critical", "high"] else "high"
        else:
            merged_tier = batch_tier

        merged_customers.append({
            "customer_id": customer_id,
            "phone": cust.get("phone"),
            "batch_risk_score": batch_risk,
            "batch_risk_tier": batch_tier,
            "speed_recent_emotion": recent_emotion,
            "speed_alerted_recently": recent_alert,
            "merged_risk_tier": merged_tier,
            "stats": cust.get("stats", {}),
        })

    # Sort theo merged risk tier
    tier_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    merged_customers.sort(key=lambda x: tier_order.get(x["merged_risk_tier"], 4))

    return {
        "customers": merged_customers[:limit],
        "source": "serving_layer_merged",
    }


def merge_alert_report(date: str) -> Dict:
    """
    Merge alert report:
      - Batch: daily summary (chạy 8h sáng)
      - Speed: alerts từ hôm nay đến hiện tại

    Return:
        {
          "date": date,
          "batch_report": {...},
          "speed_alerts_today": [...],
          "merged_total_alerts": int,
        }
    """
    # 1. Batch daily report
    batch_report = get_alert_report(date)

    if not batch_report:
        # Nếu chưa có batch report, return speed data only
        batch_report = {
            "total_alerts": 0,
            "angry_alerts": 0,
            "disappointed_alerts": 0,
            "top_emotion": "neutral",
            "avg_response_confidence": 0.0,
        }

    # 2. Speed Layer — alerts hôm nay
    layer = get_speed_layer()
    speed_alerts_today = []

    if layer:
        all_alerts = layer.get_alert_queue(limit=200)

        # Filter alerts hôm nay (trong timezone Việt Nam)
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

        for alert in all_alerts:
            alert_time = datetime.fromisoformat(alert.get("timestamp", ""))
            if alert_time >= today_start:
                speed_alerts_today.append(alert)

    # 3. Merge totals
    batch_total = batch_report.get("total_alerts", 0)
    speed_total = len(speed_alerts_today)

    return {
        "date": date,
        "batch_report": batch_report,
        "speed_alerts_today": speed_alerts_today,
        "merged_total_alerts": batch_total + speed_total,
        "speed_delta_count": speed_total,
        "source": "serving_layer_merged",
    }


# ── Live Feed Merge ──────────────────────────────────

def merge_live_feed(limit: int = 50) -> Dict:
    """
    Merge live feed:
    └─ Speed Layer: 50 messages gần nhất (customer only)

    Return:
        {
          "feed": [...],
          "source": "speed_layer_only",
        }
    """
    layer = get_speed_layer()

    if not layer:
        return {
            "feed": [],
            "source": "speed_layer_not_initialized",
        }

    feed = layer.get_recent_feed(limit=limit)

    # Enrich với emotion data
    enriched = []
    for item in feed:
        ticket_id = item["ticket_id"]
        emotion_data = layer.get_ticket_emotion(ticket_id)

        enriched.append({
            **item,
            "emotion": emotion_data.get("emotion") if emotion_data else "neutral",
            "confidence": emotion_data.get("confidence", 0.0) if emotion_data else 0.0,
            "alert": emotion_data.get("alert", False) if emotion_data else False,
            "reason": emotion_data.get("reason", "") if emotion_data else "",
        })

    return {
        "feed": enriched,
        "source": "speed_layer_enriched",
    }


# ── Dashboard Overview (merge all) ───────────────────

def get_dashboard_overview() -> Dict:
    """
    Tổng hợp dashboard: stats + top alerts + live feed.
    Merge Batch + Speed dữliệu.
    """
    # 1. Emotion stats (24h)
    emotion_stats = merge_emotion_stats(hours=24)

    # 2. High risk customers
    high_risk = merge_high_risk_customers(limit=10)

    # 3. Live feed
    live_feed = merge_live_feed(limit=20)

    # 4. Speed Layer realtime stats
    layer = get_speed_layer()
    speed_stats = layer.get_stats() if layer else {}

    return {
        "emotion_stats_24h": emotion_stats,
        "high_risk_customers": high_risk,
        "live_feed": live_feed,
        "speed_realtime": speed_stats,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "serving_layer_dashboard",
    }