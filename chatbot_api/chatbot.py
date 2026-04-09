from chatbot_api.orchestrator import strategy

def chat(user_message: str, history: list, session_id: str = "default") -> tuple[str, dict]:
    """
    Entry point cho chat, điều phối qua strategy đã được khởi tạo (Real hoặc Mock).
    Loại bỏ logic nếu/thì rườm rà, tuân thủ nguyên tắc Delegation.
    """
    return strategy.process_chat(user_message, history, session_id)