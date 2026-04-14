from chatbot_api.orchestrator import orchestrator
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional
import asyncio
import chatbot_api.db as db_ops
from chatbot_api.database import db
from chatbot_api.rag import load_kb
from chatbot_api.chatbot import chat

# ── Batch Layer imports ────────────────────────
from batch.batch_views_db import (
    init_batch_db,
    get_emotion_trend,
    get_high_risk_customers,
    get_alert_report,
)
from batch.producer import publish_message, publish_emotion, close_producer
from batch.jobs import create_scheduler, run_all_jobs_now
from batch.speed_layer import init_speed_layer, get_speed_layer, track_message, track_emotion, track_new_ticket

import logging
logger = logging.getLogger(__name__)

# ── Lifespan ──────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⏳ Connecting to MongoDB...")
    await db.connect()
    
    print("⏳ Initializing Databases...")
    await db_ops.init_db()
    await db_ops.seed_db()  # Seed demo data
    await init_batch_db()
    
    load_kb("data")

    # Init Speed Layer in-memory tracking
    init_speed_layer(feed_size=100, alert_size=200)
    
    # Backfill dữ liệu từ DB vào Speed Layer (nếu có)
    from batch.speed_layer import sync_speed_layer_from_db
    await sync_speed_layer_from_db()

    # Khởi APScheduler (AsyncIOScheduler)
    scheduler = create_scheduler()
    scheduler.start()
    print(f"✅ Startup complete - MongoDB Connected - Scheduler active")

    yield

    # Shutdown
    scheduler.shutdown(wait=False)
    await db.close()
    close_producer()
    print("👋 Shutdown complete")


app = FastAPI(title="CSKH EmotionAI MongoDB", version="3.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Ticket-ID", "X-Accel-Buffering"],
)

# ── Models ───────────────────

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

# ── Endpoints ─────────────────────

@app.get("/")
async def root():
    return {"status": "ok", "database": "mongodb", "engine": "motor"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    if not req.ticket_id:
        ticket_id = await db_ops.create_ticket(None, None, "other")
        track_new_ticket(ticket_id)
    else:
        ticket_id = req.ticket_id

    history = await db_ops.get_messages(ticket_id)
    history_turns = [
        {"role": "customer" if m["role"] == "customer" else "agent", "text": m["content"]}
        for m in history
    ]

    # Save customer message
    msg_id = await db_ops.save_message(ticket_id, "customer", req.message)
    track_message(ticket_id, msg_id, "customer", req.message)
    
    # Process with LLM
    reply, emotion = await chat(req.message, history_turns, session_id=str(ticket_id))

    # Save bot reply
    bot_msg_id = await db_ops.save_message(ticket_id, "bot", reply)
    track_message(ticket_id, bot_msg_id, "bot", reply)

    # Save emotion analysis
    await db_ops.save_emotion(
        ticket_id  = ticket_id,
        message_id = msg_id,
        emotion    = emotion.get("emotion", "neutral"),
        confidence = emotion.get("confidence", 0.0),
        reason     = emotion.get("reason", ""),
        alert      = emotion.get("alert", False),
    )
    track_emotion(
        ticket_id  = ticket_id,
        message_id = msg_id,
        emotion    = emotion.get("emotion", "neutral"),
        confidence = emotion.get("confidence", 0.0),
        reason     = emotion.get("reason", ""),
        alert      = emotion.get("alert", False),
    )

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
    if not req.ticket_id:
        ticket_id = await db_ops.create_ticket(None, None, "other")
        track_new_ticket(ticket_id)
    else:
        ticket_id = req.ticket_id

    history = await db_ops.get_messages(ticket_id)
    history_turns = [
        {"role": "customer" if m["role"] == "customer" else "agent", "text": m["content"]}
        for m in history
    ]
    msg_id = await db_ops.save_message(ticket_id, "customer", req.message)
    track_message(ticket_id, msg_id, "customer", req.message)

    async def generate():
        full_reply = []
        from chatbot_api.orchestrator import strategy
        async for token in strategy.process_chat_stream(req.message, history_turns, session_id=str(ticket_id)):
            full_reply.append(token)
            yield token.encode("utf-8")

        bot_reply = "".join(full_reply)
        bot_msg_id = await db_ops.save_message(ticket_id, "bot", bot_reply)
        track_message(ticket_id, bot_msg_id, "bot", bot_reply)

        turns = history_turns[-8:] + [{"role": "customer", "text": req.message}]
        emotion = strategy.analyze_emotion(turns)
        if emotion:
            await db_ops.save_emotion(
                ticket_id=ticket_id, message_id=msg_id,
                emotion=emotion.get("emotion", "neutral"),
                confidence=emotion.get("confidence", 0.0),
                reason=emotion.get("reason", ""),
                alert=emotion.get("alert", False),
            )
            track_emotion(
                ticket_id=ticket_id, message_id=msg_id,
                emotion=emotion.get("emotion", "neutral"),
                confidence=emotion.get("confidence", 0.0),
                reason=emotion.get("reason", ""),
                alert=emotion.get("alert", False),
            )

    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8",
        headers={
            "X-Ticket-ID": str(ticket_id),
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
        },
    )

@app.get("/history/{ticket_id}")
async def get_history(ticket_id: int):
    msgs = await db_ops.get_messages(ticket_id)
    if not msgs: raise HTTPException(404, "Ticket not found")
    return {"ticket_id": ticket_id, "messages": msgs}

@app.get("/orders/{order_id}")
async def get_order(order_id: str):
    result = await db_ops.get_order(order_id)
    if not result: raise HTTPException(404, "Order not found")
    return result

@app.get("/admin/tickets")
async def admin_tickets():
    tickets = await db_ops.get_all_tickets_with_emotions()
    return {"tickets": tickets}

@app.get("/admin/serving/dashboard")
async def serving_dashboard():
    """Serving Layer: Dashboard overview merged from Batch + Speed."""
    from batch.serving_layer import get_dashboard_overview
    return await get_dashboard_overview()

@app.get("/admin/serving/emotion-stats")
async def serving_emotion_stats(hours: int = 24):
    from batch.serving_layer import merge_emotion_stats
    return await merge_emotion_stats(hours=hours)

@app.get("/admin/serving/live-feed")
async def serving_live_feed(limit: int = 50):
    from batch.serving_layer import merge_live_feed
    return await merge_live_feed(limit=limit)

@app.get("/admin/serving/high-risk")
async def serving_high_risk(limit: int = 20):
    from batch.serving_layer import merge_high_risk_customers
    return await merge_high_risk_customers(limit=limit)

@app.post("/admin/batch/run-now")
async def trigger_batch_now():
    """Manual trigger for batch jobs."""
    asyncio.create_task(run_all_jobs_now())
    return {"status": "triggered"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
