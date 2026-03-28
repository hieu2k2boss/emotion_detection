from chatbot_api.tools import is_order_query, order_lookup
from chatbot_api.orchestrator import orchestrator
from chatbot_api.api_client import call_api

SYSTEM_PROMPT = """Bạn là trợ lý CSKH tiếng Việt thân thiện.
Trả lời ngắn gọn, lịch sự. Không bịa đặt thông tin đơn hàng."""

def chat(user_message: str, history: list, session_id: str = "default") -> tuple[str, dict]:
    """
    history:    [{"role":"customer/agent","text":"..."}]
    session_id: dùng ticket_id để API giữ đúng ngữ cảnh mỗi cuộc hội thoại
    """
    # ── OrderLookup (không cần RAG) ─────────────
    if is_order_query(user_message):
        reply   = order_lookup(user_message)
        emotion = {"emotion": "neutral", "confidence": 0.9,
                   "reason": "Hỏi thông tin đơn hàng", "alert": False}
        return reply, emotion

    # ── Ghép system prompt + lịch sử + tin mới ──
    lines = [SYSTEM_PROMPT, ""]
    for m in history[-10:]:
        prefix = "Khách" if m["role"] == "customer" else "Agent"
        lines.append(f"{prefix}: {m['text']}")
    lines.append(f"Khách: {user_message}")
    full_message = "\n".join(lines)

    try:
        reply = call_api(full_message, session_id=session_id)
        if not reply:
            reply = "Xin lỗi, em chưa nhận được phản hồi. Anh/chị thử lại sau nhé!"
    except Exception as e:
        reply = "Xin lỗi, em đang gặp sự cố. Anh/chị thử lại sau nhé!"

    # ── Orchestrator RAG phân tích cảm xúc ───────
    turns  = history[-8:] + [{"role": "customer", "text": user_message}]
    emotion = orchestrator(turns)

    return reply, emotion