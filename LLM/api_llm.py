"""
LLM API Server với Queue System
- FastAPI server
- Request queue (tránh chạy song song làm chậm nhau)
- Streaming support
- Chạy: uvicorn llm_api_server:app --host 0.0.0.0 --port 8000
- Cài: pip install fastapi uvicorn llama-cpp-python huggingface_hub
"""

import asyncio
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from pydantic import BaseModel

# ──────────────────────────────────────────────────
# CẤU HÌNH
# ──────────────────────────────────────────────────
MODEL_REPO = "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF"
MODEL_FILE = "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"
N_THREADS   = 8
N_CTX       = 4096
MAX_QUEUE   = 20   # Tối đa bao nhiêu request xếp hàng

DEFAULT_SYSTEM = "Bạn là trợ lý AI thông minh. Hãy trả lời bằng tiếng Việt, rõ ràng và đầy đủ."

# ──────────────────────────────────────────────────
# GLOBAL STATE
# ──────────────────────────────────────────────────
llm: Optional[Llama] = None
request_queue: asyncio.Queue = asyncio.Queue(maxsize=MAX_QUEUE)
queue_positions: dict = {}   # request_id → position info

# ──────────────────────────────────────────────────
# MODELS
# ──────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    system: str = DEFAULT_SYSTEM
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = False

class ChatResponse(BaseModel):
    request_id: str
    answer: str
    time_sec: float
    tokens_per_sec: float
    queue_wait_sec: float

class QueueStatus(BaseModel):
    request_id: str
    position: int
    estimated_wait_sec: float

# ──────────────────────────────────────────────────
# STARTUP / SHUTDOWN
# ──────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm
    print("⬇️  Loading model...")
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    llm = Llama(model_path=model_path, n_ctx=N_CTX, n_threads=N_THREADS, verbose=False)
    print("✅ Model loaded! Server sẵn sàng.\n")

    # Khởi động worker xử lý queue
    worker_task = asyncio.create_task(queue_worker())
    yield
    worker_task.cancel()

app = FastAPI(
    title="LLM API Server",
    description="Local LLM API với Request Queue",
    version="1.0.0",
    lifespan=lifespan
)

# ──────────────────────────────────────────────────
# QUEUE WORKER — chỉ xử lý 1 request tại 1 thời điểm
# ──────────────────────────────────────────────────
async def queue_worker():
    """Worker chạy background, lấy request từ queue và xử lý tuần tự."""
    while True:
        request_id, request, future = await request_queue.get()
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, process_request, request
            )
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        finally:
            queue_positions.pop(request_id, None)
            request_queue.task_done()

def process_request(request: ChatRequest) -> dict:
    """Xử lý inference (chạy trong thread pool để không block event loop)."""
    prompt = (
        f"<bos><start_of_turn>user\n{request.message}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    t = time.time()
    out = llm(
        prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        stop=["<end_of_turn>"],
        echo=False,
    )
    elapsed = time.time() - t
    tokens = out["usage"]["completion_tokens"]

    return {
        "answer": out["choices"][0]["text"].strip(),
        "time_sec": round(elapsed, 2),
        "tokens_per_sec": round(tokens / elapsed, 1) if elapsed > 0 else 0,
    }

def build_prompt(request: ChatRequest) -> str:
    return (
        f"<bos><start_of_turn>user\n{request.message}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

# ──────────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "status": "running",
        "model": MODEL_FILE,
        "queue_size": request_queue.qsize(),
        "endpoints": ["/chat", "/chat/stream", "/queue/status", "/health"]
    }

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": llm is not None, "queue_size": request_queue.qsize()}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Gửi message và chờ response (blocking).
    Request sẽ được xếp hàng nếu đang có request khác chạy.
    """
    if request_queue.full():
        raise HTTPException(status_code=503, detail=f"Queue đầy ({MAX_QUEUE} requests). Thử lại sau.")

    request_id = str(uuid.uuid4())[:8]
    future = asyncio.get_event_loop().create_future()
    queue_wait_start = time.time()

    # Lưu vị trí queue
    queue_positions[request_id] = {"position": request_queue.qsize() + 1, "enqueued_at": time.time()}

    await request_queue.put((request_id, request, future))

    # Chờ kết quả
    result = await future
    queue_wait = time.time() - queue_wait_start - result["time_sec"]

    return ChatResponse(
        request_id=request_id,
        answer=result["answer"],
        time_sec=result["time_sec"],
        tokens_per_sec=result["tokens_per_sec"],
        queue_wait_sec=round(max(0, queue_wait), 2),
    )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming response — trả về từng token ngay khi sinh ra.
    """
    if request_queue.full():
        raise HTTPException(status_code=503, detail="Queue đầy. Thử lại sau.")

    async def generate() -> AsyncGenerator[str, None]:
        prompt = build_prompt(request)
        # Streaming chạy trực tiếp (không qua queue worker)
        # vì stream cần giữ connection liên tục
        loop = asyncio.get_event_loop()

        def stream_tokens():
            for chunk in llm(
                prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stop=["<end_of_turn>"],
                echo=False,
                stream=True,
            ):
                yield chunk["choices"][0]["text"]

        for token in await loop.run_in_executor(None, lambda: list(stream_tokens())):
            yield f"data: {token}\n\n"
            await asyncio.sleep(0)  # yield control to event loop

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/queue/status")
async def queue_status():
    """Xem trạng thái queue hiện tại."""
    return {
        "queue_size": request_queue.qsize(),
        "max_queue": MAX_QUEUE,
        "pending_requests": list(queue_positions.keys()),
        "estimated_wait_sec": request_queue.qsize() * 30,  # ước tính ~30s/request
    }