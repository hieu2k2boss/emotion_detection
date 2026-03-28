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
from chatbot_api.api_client import call_api_stream
from chatbot_api.tools import is_order_query, order_lookup

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⏳ Initializing...")
    db.init_db()
    db.seed_db()
    load_kb("data")
    print(" Ready!")
    yield

app = FastAPI(title="CSKH Chatbot API", version="2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Ticket-ID", "X-Accel-Buffering"], 
)

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

# ── Chat thường ───────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    ticket_id = req.ticket_id or db.create_ticket(None, None, "other")

    history = db.get_messages(ticket_id)
    history_turns = [
        {"role": "customer" if m["role"] == "customer" else "agent",
         "text": m["content"]}
        for m in history
    ]

    msg_id = db.save_message(ticket_id, "customer", req.message)
    reply, emotion = chat(req.message, history_turns, session_id=str(ticket_id))

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

# ── Chat streaming ────────────────────────────────────────────────────────────
@app.post("/chat/stream")
async def chat_stream_endpoint(req: ChatRequest):
    """
    Trả về token theo từng chunk (plain text stream).
    Header X-Ticket-ID chứa ticket_id để client lưu lại dùng lần sau.
    """
    ticket_id = req.ticket_id or db.create_ticket(None, None, "other")

    history = db.get_messages(ticket_id)
    history_turns = [
        {"role": "customer" if m["role"] == "customer" else "agent",
         "text": m["content"]}
        for m in history
    ]
    db.save_message(ticket_id, "customer", req.message)

    # Order query → không cần stream, trả một lần
    if is_order_query(req.message):
        reply = order_lookup(req.message)
        db.save_message(ticket_id, "bot", reply)
        async def one_shot():
            yield reply.encode("utf-8")
        return StreamingResponse(
            one_shot(),
            media_type="text/plain; charset=utf-8",
            headers={"X-Ticket-ID": str(ticket_id)},
        )

    # Ghép prompt
    SYSTEM_PROMPT = """Bạn là trợ lý CSKH tiếng Việt của shop.

    QUY TẮC BẮT BUỘC:
    1. TUYỆT ĐỐI không bịa đặt: số tài khoản, giá tiền, tên sản phẩm, quy trình, chính sách.
    2. Nếu không có thông tin → trả lời: "Em chưa có thông tin về vấn đề này, anh/chị vui lòng liên hệ nhân viên hỗ trợ nhé!"
    3. Chỉ tra cứu đơn hàng khi khách cung cấp mã đơn (VD: DH001) hoặc số điện thoại.
    4. Không tự ý hướng dẫn đặt hàng, thanh toán nếu không được cung cấp thông tin chính xác.
    5. Trả lời ngắn gọn, lịch sự, xưng "em"."""
    lines = [SYSTEM_PROMPT, ""]
    for m in history_turns[-10:]:
        prefix = "Khách" if m["role"] == "customer" else "Agent"
        lines.append(f"{prefix}: {m['text']}")
    lines.append(f"Khách: {req.message}")
    full_message = "\n".join(lines)

    async def generate():
        full_reply = []
        loop = asyncio.get_event_loop()

        # Generator đồng bộ → chạy trong thread pool, không block event loop
        tokens = await loop.run_in_executor(
            None,
            lambda: list(call_api_stream(full_message, session_id=str(ticket_id)))
        )
        for token in tokens:
            full_reply.append(token)
            yield token.encode("utf-8")

        db.save_message(ticket_id, "bot", "".join(full_reply))

    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8",
        headers={
            "X-Ticket-ID":       str(ticket_id),
            "X-Accel-Buffering": "no",
            "Cache-Control":     "no-cache",
        },
    )

# ── Các endpoint khác ─────────────────────────────────────────────────────────
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)