from openai import OpenAI
from chatbot_api.tools import is_order_query, order_lookup
from chatbot_api.orchestrator import orchestrator

client = OpenAI(
    api_key  = "sk-abc",
    base_url = "https://api.abc.com"
)

SYSTEM_PROMPT = """Bạn là trợ lý CSKH tiếng Việt thân thiện.
Trả lời ngắn gọn, lịch sự. Không bịa đặt thông tin đơn hàng."""

def chat(user_message: str, history: list) -> tuple[str, dict]:
    """
    history: [{"role":"customer/agent","text":"..."}]
    """
    # ── OrderLookup (không cần RAG) ─────────────
    if is_order_query(user_message):
        reply   = order_lookup(user_message)
        emotion = {"emotion":"neutral","confidence":0.9,
                   "reason":"Hỏi thông tin đơn hàng","alert":False}
        return reply, emotion

    # ── General reply (DeepSeek) ─────────────────
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history[-10:]:
        role = "user" if m["role"] == "customer" else "assistant"
        messages.append({"role": role, "content": m["text"]})
    messages.append({"role": "user", "content": user_message})

    try:
        res   = client.chat.completions.create(
            model="deepseek-chat", messages=messages,
            max_tokens=300, temperature=0.7,
        )
        reply = res.choices[0].message.content.strip()
    except Exception as e:
        reply = f"Xin lỗi, em đang gặp sự cố. Anh/chị thử lại sau nhé!"

    # ── Orchestrator RAG phân tích cảm xúc ───────
    turns = history[-8:] + [{"role": "customer", "text": user_message}]
    emotion = orchestrator(turns)

    return reply, emotion