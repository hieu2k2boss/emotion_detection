from chatbot_api.tools import is_order_query, order_lookup
from chatbot_api.orchestrator import orchestrator
from chatbot_api.api_client import call_api

SYSTEM_PROMPT = """Bạn là trợ lý CSKH tiếng Việt thân thiện.
Trả lời ngắn gọn, lịch sự. Không bịa đặt thông tin đơn hàng."""

def chat(user_message: str, history: list, session_id: str = "default") -> tuple[str, dict]:
    if is_order_query(user_message):
        reply = order_lookup(user_message)
        # Vẫn phân tích cảm xúc thật thay vì hardcode neutral
        turns  = history[-8:] + [{"role": "customer", "text": user_message}]
        emotion = orchestrator(turns)
        # Chỉ fallback neutral nếu orchestrator fail
        if not emotion:
            emotion = {"emotion": "neutral", "confidence": 0.8,
                       "reason": "Hỏi thông tin đơn hàng", "alert": False}
        return reply, emotion 

    lines = [SYSTEM_PROMPT, ""]
    for m in history[-10:]:
        prefix = "Khách" if m["role"] == "customer" else "Agent"
        # Chỉ lấy 1 dòng đầu, tránh reply bị nhiễm từ lần trước
        clean_text = m["text"].split("\n")[0].strip()

        lines.append(f"{prefix}: {clean_text}")
    lines.append(f"Khách: {user_message}")
    lines.append("Agent:")  # ← FIX 1: bắt API điền tiếp, không bịa kịch bản mới

    full_message = "\n".join(lines)

    try:
        raw_reply = call_api(full_message, session_id=session_id)
        # FIX 2: cắt tại dòng đầu tiên có "Khách:" để loại bỏ phần API tự bịa tiếp
        reply = raw_reply.split("\nKhách:")[0].split("\nAgent:")[0].strip()
        if not reply:
            reply = "Xin lỗi, em chưa nhận được phản hồi. Anh/chị thử lại sau nhé!"
    except Exception as e:
        reply = "Xin lỗi, em đang gặp sự cố. Anh/chị thử lại sau nhé!"

    turns  = history[-8:] + [{"role": "customer", "text": user_message}]
    emotion = orchestrator(turns)

    return reply, emotion