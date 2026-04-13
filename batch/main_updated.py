"""
main.py (updated) — tích hợp Batch Layer
─────────────────────────────────────────
Thay đổi so với phiên bản gốc:
  1. Lifespan: khởi tạo Batch DB + APScheduler
  2. /chat và /chat/stream: publish event lên Kafka sau khi save
  3. Thêm /admin/batch/* endpoints (Serving Layer)
"""
from chatbot_api.orchestrator import orchestrator
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional
import asyncio
import chatbot_api.db as db
from chatbot_api.rag import load_kb
from chatbot_api.chatbot import chat
from chatbot_api.tools import is_order_query, order_lookup

# ── Batch Layer imports ────────────────────────
from batch.batch_views_db import (
    init_batch_db,
    get_emotion_trend,
    get_high_risk_customers,
    get_alert_report,
    get_customer_risk,
)
from batch.producer import publish_message, publish_emotion, close_producer
from batch.jobs import create_scheduler, run_all_jobs_now

import logging
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⏳ Initializing Speed Layer...")
    db.init_db()
    db.seed_db()
    load_kb("data")

    print("⏳ Initializing Batch Layer...")
    init_batch_db()

    # Khởi APScheduler (non-blocking background thread)
    scheduler = create_scheduler()
    scheduler.start()
    print(f"✅ Scheduler started — {len(scheduler.get_jobs())} jobs active")

    print("✅ Ready!")
    yield

    # Shutdown
    scheduler.shutdown(wait=False)
    close_producer()
    print("👋 Shutdown complete")


app = FastAPI(title="CSKH Chatbot API", version="2.1-lambda", lifespan=lifespan)
app.add_middleware(CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Ticket-ID", "X-Accel-Buffering"],
)


# ── Request/Response models ───────────────────

class ChatRequest(BaseModel):
    message:   str
    ticket_id: Optional[int] = None

class ChatResponse(BaseModel):
    ticket_id:  int
    reply:      str
    emotion:    str
    confidence: float
    alert:      bool
    reason:     str


# ── Helpers ───────────────────────────────────

def _safe_publish_message(ticket_id, message_id, role, content, session_id=""):
    """Publish lên Kafka — không raise exception nếu Kafka down."""
    try:
        publish_message(ticket_id, message_id, role, content, session_id)
    except Exception as e:
        logger.warning("[Kafka] publish_message failed (degraded mode): %s", e)

def _safe_publish_emotion(ticket_id, message_id, emotion_data):
    try:
        publish_emotion(
            ticket_id,  message_id,
            emotion_data.get("emotion", "neutral"),
            emotion_data.get("confidence", 0.0),
            emotion_data.get("reason", ""),
            emotion_data.get("alert", False),
        )
    except Exception as e:
        logger.warning("[Kafka] publish_emotion failed (degraded mode): %s", e)


# ── Speed Layer endpoints ─────────────────────

@app.get("/")
def root():
    return {"status": "ok", "version": "2.1-lambda", "architecture": "Lambda (Speed + Batch)"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    ticket_id = req.ticket_id or db.create_ticket(None, None, "other")

    history = db.get_messages(ticket_id)
    history_turns = [
        {"role": "customer" if m["role"] == "customer" else "agent",
         "text": m["content"]}
        for m in history
    ]

    # Save + publish customer message
    msg_id = db.save_message(ticket_id, "customer", req.message)
    _safe_publish_message(ticket_id, msg_id, "customer", req.message, str(ticket_id))

    # Process
    reply, emotion = chat(req.message, history_turns, session_id=str(ticket_id))

    # Save + publish bot reply
    bot_msg_id = db.save_message(ticket_id, "bot", reply)
    _safe_publish_message(ticket_id, bot_msg_id, "bot", reply, str(ticket_id))

    # Save + publish emotion
    db.save_emotion(
        ticket_id  = ticket_id,
        message_id = msg_id,
        emotion    = emotion.get("emotion", "neutral"),
        confidence = emotion.get("confidence", 0.0),
        reason     = emotion.get("reason", ""),
        alert      = emotion.get("alert", False),
    )
    _safe_publish_emotion(ticket_id, msg_id, emotion)

    return ChatResponse(
        ticket_id  = ticket_id,
        reply      = reply,
        emotion    = emotion.get("emotion", "neutral"),
        confidence = emotion.get("confidence", 0.0),
        alert      = emotion.get("alert", False),
        reason     = emotion.get("reason", ""),
    )


@app.post("/chat/stream")
async def chat_stream_endpoint(req: ChatRequest):
    ticket_id = req.ticket_id or db.create_ticket(None, None, "other")

    history = db.get_messages(ticket_id)
    history_turns = [
        {"role": "customer" if m["role"] == "customer" else "agent",
         "text": m["content"]}
        for m in history
    ]
    msg_id = db.save_message(ticket_id, "customer", req.message)
    _safe_publish_message(ticket_id, msg_id, "customer", req.message, str(ticket_id))

    async def generate():
        full_reply = []

        from chatbot_api.orchestrator import strategy
        async for token in strategy.process_chat_stream(req.message, history_turns, session_id=str(ticket_id)):
            full_reply.append(token)
            yield token.encode("utf-8")

        bot_reply  = "".join(full_reply)
        bot_msg_id = db.save_message(ticket_id, "bot", bot_reply)
        _safe_publish_message(ticket_id, bot_msg_id, "bot", bot_reply, str(ticket_id))

        turns   = history_turns[-8:] + [{"role": "customer", "text": req.message}]
        emotion = strategy.analyze_emotion(turns)
        if emotion:
            db.save_emotion(
                ticket_id  = ticket_id, message_id = msg_id,
                emotion    = emotion.get("emotion", "neutral"),
                confidence = emotion.get("confidence", 0.0),
                reason     = emotion.get("reason", ""),
                alert      = emotion.get("alert", False),
            )
            _safe_publish_emotion(ticket_id, msg_id, emotion)

    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8",
        headers={
            "X-Ticket-ID":       str(ticket_id),
            "X-Accel-Buffering": "no",
            "Cache-Control":     "no-cache",
        },
    )


# ── Existing endpoints ────────────────────────

@app.get("/history/{ticket_id}")
def get_history(ticket_id: int):
    msgs = db.get_messages(ticket_id)
    if not msgs: raise HTTPException(404, "Ticket not found")
    return {"ticket_id": ticket_id, "messages": msgs}

@app.get("/orders/{order_id}")
def get_order(order_id: str):
    result = db.get_order(order_id)
    if not result: raise HTTPException(404, "Order not found")
    return result

@app.get("/admin/tickets")
def admin_tickets():
    tickets = db.get_all_tickets_with_emotions()
    return {"tickets": tickets}


# ── Batch Layer (Serving Layer) endpoints ─────

@app.get("/admin/batch/emotion-trend")
def batch_emotion_trend(hours: int = 24):
    """
    Serving Layer: emotion distribution trong N giờ qua.
    Đọc từ batch_views (pre-computed) — rất nhanh.
    """
    trend = get_emotion_trend(hours=hours)
    return {"hours": hours, "data": trend, "source": "batch_view"}


@app.get("/admin/batch/high-risk-customers")
def batch_high_risk(limit: int = 20):
    """
    Serving Layer: danh sách khách hàng risk cao.
    Kết quả từ batch customer_segment_job.
    """
    customers = get_high_risk_customers(limit=limit)
    return {"customers": customers, "source": "batch_view"}


@app.get("/admin/batch/alert-report/{date}")
def batch_alert_report(date: str):
    """
    Serving Layer: báo cáo alert ngày cụ thể (YYYY-MM-DD).
    """
    report = get_alert_report(date)
    if not report:
        raise HTTPException(404, f"Chưa có report cho ngày {date}")
    return {**report, "source": "batch_view"}


@app.get("/admin/batch/customer-risk/{customer_id}")
def batch_customer_risk(customer_id: int):
    """
    Serving Layer: risk profile của 1 khách hàng.
    Merge: batch risk score + speed layer recent emotions.
    """
    batch_risk = get_customer_risk(customer_id)

    # Speed view: lấy emotion gần nhất từ Speed Layer
    speed_conn = db.get_conn()
    recent = speed_conn.execute("""
        SELECT el.emotion, el.confidence, el.alert, el.created_at
        FROM emotion_logs el
        JOIN tickets t ON t.id = el.ticket_id
        WHERE t.customer_id = ?
        ORDER BY el.created_at DESC LIMIT 5
    """, (customer_id,)).fetchall()
    speed_conn.close()

    return {
        "customer_id":    customer_id,
        "batch_risk":     batch_risk,                        # computed giờ trước
        "recent_emotions": [dict(r) for r in recent],       # real-time
        "source":         "merged (batch + speed)",
    }


@app.post("/admin/batch/run-now")
def trigger_batch_now(job: Optional[str] = None):
    """
    Manual trigger — chạy batch jobs ngay (debug / backfill).
    job: "emotion_hourly" | "customer_segment" | "alert_report" | None (all)
    """
    import threading
    from batch.jobs import emotion_hourly_job, customer_segment_job, alert_report_job

    job_map = {
        "emotion_hourly":   emotion_hourly_job,
        "customer_segment": customer_segment_job,
        "alert_report":     alert_report_job,
    }

    if job and job not in job_map:
        raise HTTPException(400, f"Job không hợp lệ. Chọn: {list(job_map.keys())}")

    fn = job_map.get(job, run_all_jobs_now)
    threading.Thread(target=fn, daemon=True).start()

    return {"status": "triggered", "job": job or "all"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
