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

async def merge_emotion_stats(hours: int = 24) -> Dict:
    """Merge emotion statistics (Async)."""
    # 1. Batch view
    batch_data = await get_emotion_trend(hours=hours) or []

    # 2. Speed Layer - counters from memory
    layer = get_speed_layer()
    speed_counts: Dict[str, int] = {}
    speed_alerts = 0

    if layer:
        stats = layer.get_stats()
        speed_counts = stats.get("emotion_counts", {})
        speed_alerts = stats.get("alert_count", 0)

    # 3. Merge
    merged_counts: Dict[str, int] = {}
    for row in batch_data:
        emotion = row.get("emotion")
        count = row.get("total", 0) # Tên field đã đổi trong batch_views_db.py
        if emotion:
            merged_counts[emotion] = merged_counts.get(emotion, 0) + count

    for emotion, count in speed_counts.items():
        merged_counts[emotion] = merged_counts.get(emotion, 0) + count

    return {
        "hours": hours,
        "batch_view_count": len(batch_data),
        "speed_counts": speed_counts,
        "merged_emotion_distribution": merged_counts,
        "total_messages": sum(merged_counts.values()),
        "total_alerts": speed_alerts,
        "source": "serving_layer_merged_async",
    }

async def merge_high_risk_customers(limit: int = 20) -> Dict:
    """Merge high risk customers (Async)."""
    batch_customers = await get_high_risk_customers(limit=limit) or []
    layer = get_speed_layer()
    ticket_emotions: Dict[int, Dict] = {}
    if layer:
        ticket_emotions = layer.get_all_ticket_emotions()

    merged_customers = []
    for cust in batch_customers:
        customer_id = cust.get("customer_id")
        recent_emotion = None
        recent_alert = False

        if ticket_emotions:
            # Simple mapping logic
            for ticket_id, emotion_data in ticket_emotions.items():
                if emotion_data.get("alert"):
                    recent_alert = True
                    recent_emotion = emotion_data.get("emotion")
                    break
        
        batch_tier = cust.get("risk_tier", "low")
        merged_tier = "critical" if (recent_alert and batch_tier != "low") else batch_tier

        merged_customers.append({
            **cust,
            "speed_recent_emotion": recent_emotion,
            "speed_alerted_recently": recent_alert,
            "merged_risk_tier": merged_tier,
        })

    return {
        "customers": merged_customers[:limit],
        "source": "serving_layer_merged_async",
    }

async def merge_alert_report(date: str) -> Dict:
    """Merge alert report (Async)."""
    batch_report = await get_alert_report(date)
    layer = get_speed_layer()
    speed_alerts_today = []

    if layer:
        all_alerts = layer.get_alert_queue(limit=200)
        today_str = datetime.now().strftime("%Y-%m-%d")
        for alert in all_alerts:
            if alert.get("timestamp", "").startswith(today_str):
                speed_alerts_today.append(alert)

    return {
        "date": date,
        "batch_report": batch_report,
        "speed_alerts_today": speed_alerts_today,
        "source": "serving_layer_merged_async",
    }

async def merge_live_feed(limit: int = 50) -> Dict:
    """Live feed Enrichment (Async)."""
    layer = get_speed_layer()
    if not layer: return {"feed": []}

    feed = layer.get_recent_feed(limit=limit)
    enriched = []
    for item in feed:
        ticket_id = item["ticket_id"]
        emotion_data = layer.get_ticket_emotion(ticket_id)
        enriched.append({
            **item,
            "emotion": emotion_data.get("emotion") if emotion_data else "neutral",
            "alert": emotion_data.get("alert", False) if emotion_data else False,
        })
    return {"feed": enriched}

async def get_dashboard_overview() -> Dict:
    """Tổng hợp dashboard (Async)."""
    emotion_stats = await merge_emotion_stats(hours=24)
    high_risk = await merge_high_risk_customers(limit=10)
    live_feed = await merge_live_feed(limit=20)

    return {
        "emotion_stats_24h": emotion_stats,
        "high_risk_customers": high_risk,
        "live_feed": live_feed,
        "generated_at": datetime.utcnow().isoformat(),
        "source": "serving_layer_dashboard_async"
    }