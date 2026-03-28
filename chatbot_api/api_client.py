import requests

API_URL = "https://4723fb21400be5.lhr.life"
API_KEY = "4a7ab110b2dd6c3c0b5902accc2d91d3"

HEADERS = {
    "x-api-key":    API_KEY,
    "Content-Type": "application/json",
}

def call_api(message: str, session_id: str = "default") -> str:
    """Gọi /chat — trả về toàn bộ response một lần."""
    try:
        resp = requests.post(
            f"{API_URL}/chat",
            headers=HEADERS,
            json={"message": message, "session_id": session_id},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
    except requests.exceptions.Timeout:
        raise RuntimeError("API timeout sau 60 giây")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Lỗi gọi API: {e}")


def call_api_stream(message: str, session_id: str = "default"):
    """
    Gọi /chat/stream — yield từng token khi nhận được.

    Ví dụ dùng trong FastAPI:
        for token in call_api_stream("Xin chào", session_id="123"):
            print(token, end="", flush=True)
    """
    with requests.post(
        f"{API_URL}/chat/stream",
        headers=HEADERS,
        json={"message": message, "session_id": session_id},
        stream=True,   # ← không download hết một lần
        timeout=120,
    ) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_content(chunk_size=None):
            if chunk:
                yield chunk.decode("utf-8")