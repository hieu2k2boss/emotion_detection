from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional
import chatbot_api.db as db
from chatbot_api.rag import load_kb
from chatbot_api.chatbot import chat

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("⏳ Initializing...")
    db.init_db()
    db.seed_db()
    load_kb("data")     # load data/*.json → embed → ChromaDB + BM25
    print("✅ Ready!")
    yield
    # Shutdown (nếu cần cleanup)

app = FastAPI(title="CSKH Chatbot API", version="2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

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

@app.get("/")
def root():
    return {"status": "ok", "version": "2.0 — Agentic RAG"}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    # Tạo hoặc lấy ticket
    ticket_id = req.ticket_id or db.create_ticket(None, None, "other")

    # Lịch sử hội thoại
    history = db.get_messages(ticket_id)   # [{"role","content","created_at"}]
    history_turns = [
        {"role": "customer" if m["role"]=="customer" else "agent",
         "text": m["content"]}
        for m in history
    ]

    # Lưu tin nhắn khách
    msg_id = db.save_message(ticket_id, "customer", req.message)

    # Xử lý
    reply, emotion = chat(req.message, history_turns)

    # Lưu reply + emotion
    db.save_message(ticket_id, "bot", reply)
    db.save_emotion(
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

