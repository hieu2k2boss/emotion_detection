from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, AsyncGenerator
import json
import uuid
import asyncio

from chatbot_api.rag import hybrid_search, load_kb
from chatbot_api.api_client import call_api, call_api_stream
from chatbot_api.tools import is_order_query, order_lookup

# Dictionary chứa các câu trả lời Template cho Mock Mode
MOCK_TEMPLATES = {
    "happy": "Dạ, cảm ơn anh/chị đã tin tưởng và ủng hộ shop ạ! Chúc anh/chị một ngày tốt lành.",
    "neutral": "Dạ em đã nhận được thông tin ạ. Anh/chị cần em hỗ trợ gì thêm không ạ?",
    "confused": "Dạ, vấn đề này em chưa nắm rõ lắm, anh/chị có thể vui lòng giải thích chi tiết hơn giúp em được không ạ?",
    "anxious": "Dạ anh/chị yên tâm nhé, em đang kiểm tra ngay đây ạ. Mọi việc sẽ sớm được giải quyết thôi ạ.",
    "frustrated": "Dạ em thực sự xin lỗi vì sự bất tiện này ạ. Em hiểu cảm giác của anh/chị, em sẽ cố gắng xử lý nhanh nhất có thể.",
    "disappointed": "Dạ em rất tiếc vì sản phẩm/dịch vụ chưa làm hài lòng mình ạ. Shop ghi nhận và sẽ phản hồi sớm nhất để bù đắp cho mình ạ.",
    "angry": "Dạ em vô cùng xin lỗi anh/chị vì trải nghiệm không tốt này ạ. Mong anh/chị bớt giận, em đã chuyển thông tin này cho quản lý để xử lý gấp cho mình rồi ạ."
}

class ChatStrategy(ABC):
    """Interface cho các chiến lược xử lý hội thoại và phân tích cảm xúc."""
    
    @abstractmethod
    async def process_chat(self, user_message: str, history: List[Dict[str, str]], session_id: str = "default") -> Tuple[str, Dict[str, Any]]:
        """Xử lý tin nhắn (Non-streaming)."""
        pass

    @abstractmethod
    def process_chat_stream(self, user_message: str, history: List[Dict[str, str]], session_id: str = "default") -> AsyncGenerator[str, None]:
        """Xử lý tin nhắn (Streaming tokens)."""
        pass

    @abstractmethod
    def analyze_emotion(self, turns: List[Dict[str, str]]) -> Dict[str, Any]:
        """Chỉ phân tích cảm xúc."""
        pass

class LLMChatStrategy(ChatStrategy):
    """Chiến lược sử dụng LLM thật (Real Mode)."""

    def _build_full_prompt(self, user_message: str, history: List[Dict[str, str]]) -> str:
        SYSTEM_PROMPT = """Bạn là trợ lý CSKH tiếng Việt thân thiện. Trả lời ngắn gọn, lịch sự. Không bịa đặt thông tin đơn hàng."""
        lines = [SYSTEM_PROMPT, ""]
        for m in history[-10:]:
            prefix = "Khách" if m["role"] == "customer" else "Agent"
            # Giữ clean text để tránh bị loop logic cũ
            clean_text = m["text"].split("\n")[0].strip()
            lines.append(f"{prefix}: {clean_text}")
        lines.append(f"Khách: {user_message}")
        lines.append("Agent:")
        return "\n".join(lines)

    async def process_chat(self, user_message: str, history: List[Dict[str, str]], session_id: str = "default") -> Tuple[str, Dict[str, Any]]:
        if is_order_query(user_message):
            reply = await order_lookup(user_message)
            emotion = self.analyze_emotion(history[-8:] + [{"role": "customer", "text": user_message}])
            return reply, emotion

        full_message = self._build_full_prompt(user_message, history)
        try:
            raw_reply = call_api(full_message, session_id=session_id)
            reply = raw_reply.split("\nKhách:")[0].split("\nAgent:")[0].strip()
            if not reply:
                reply = "Xin lỗi, em chưa nhận được phản hồi."
        except Exception:
            reply = "Xin lỗi, em đang gặp sự cố."

        emotion = self.analyze_emotion(history[-8:] + [{"role": "customer", "text": user_message}])
        return reply, emotion

    async def process_chat_stream(self, user_message: str, history: List[Dict[str, str]], session_id: str = "default") -> AsyncGenerator[str, None]:
        full_message = self._build_full_prompt(user_message, history)
        
        loop = asyncio.get_event_loop()
        queue = asyncio.Queue()

        def producer():
            try:
                for token in call_api_stream(full_message, session_id=session_id):
                    loop.call_soon_threadsafe(queue.put_nowait, token)
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, f"[ERROR: {str(e)}]")
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        loop.run_in_executor(None, producer)

        buffer = ""
        while True:
            token = await queue.get()
            if token is None: break
            
            buffer += token
            # Cắt bỏ phần bịa nếu LLM tự sinh tiếp vai Khách
            if "Khách:" in buffer:
                clean = buffer.split("Khách:")[0].strip()
                if clean: yield clean
                break

            if len(buffer) > 20 or token.endswith((".", "!", "?", "\n")):
                yield buffer
                buffer = ""
        
        if buffer and "Khách:" not in buffer:
            yield buffer

    def analyze_emotion(self, turns: List[Dict[str, str]]) -> Dict[str, Any]:
        from chatbot_api.orchestrator import real_llm_orchestrator
        return real_llm_orchestrator(turns)

class MockChatStrategy(ChatStrategy):
    """Chiến lược Mock (Semantic Router) phục vụ demo offline."""

    async def process_chat(self, user_message: str, history: List[Dict[str, str]], session_id: str = "default") -> Tuple[str, Dict[str, Any]]:
        emotion_data = self.analyze_emotion(history[-8:] + [{"role": "customer", "text": user_message}])
        
        if is_order_query(user_message):
            reply = await order_lookup(user_message)
            return reply, emotion_data

        label = emotion_data.get("emotion", "neutral")
        reply = MOCK_TEMPLATES.get(label, MOCK_TEMPLATES["neutral"])
        return reply, emotion_data

    async def process_chat_stream(self, user_message: str, history: List[Dict[str, str]], session_id: str = "default") -> AsyncGenerator[str, None]:
        # Giả lập streaming bằng cách trả về từng từ hoặc đoạn từ template
        reply, _ = await self.process_chat(user_message, history, session_id)
        
        # Chia nhỏ reply thành các "token" giả để tạo hiệu ứng gõ
        words = reply.split(" ")
        for i in range(len(words)):
            yield words[i] + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0.05) # Delay nhỏ cho mượt

    def analyze_emotion(self, turns: List[Dict[str, str]]) -> Dict[str, Any]:
        last_cust = next((t["text"] for t in reversed(turns) if t["role"] == "customer"), "")
        search_results = hybrid_search(last_cust, top_k=1)
        
        if search_results:
            label = search_results[0]["label"]
            is_alert = label in ["angry", "disappointed"]
            return {
                "emotion": label, "confidence": 0.85, 
                "reason": "Mocked by Semantic Router", "alert": is_alert
            }
        
        return {"emotion": "neutral", "confidence": 0.5, "reason": "Default label", "alert": False}

class StrategyFactory:
    @staticmethod
    def get_strategy(use_mock: bool = False) -> ChatStrategy:
        if use_mock:
            return MockChatStrategy()
        return LLMChatStrategy()
