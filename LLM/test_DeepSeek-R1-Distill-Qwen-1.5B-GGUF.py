from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import time

# Đổi thành 7B để so sánh, hoặc giữ 1.5B
# MODEL_REPO = "bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF"
# MODEL_FILE = "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"

MODEL_REPO = "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF"
MODEL_FILE = "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"

print("⬇️ Loading model...")
model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)

llm = Llama(model_path=model_path, n_ctx=4096, n_threads=12, verbose=False, n_batch=512)
print("✅ Done!\n")

SYSTEM = "Bạn là trợ lý AI. Hãy trả lời bằng tiếng Việt, đầy đủ và chính xác."

def chat(question: str, max_tokens=512):
    # ✅ Đúng format Qwen/DeepSeek R1 Distill
    prompt = (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    t = time.time()
    out = llm(prompt, max_tokens=max_tokens, stop=["<|im_end|>"], echo=False)
    elapsed = time.time() - t

    answer = out["choices"][0]["text"].strip()
    tokens = out["usage"]["completion_tokens"]

    print(f"💬 {answer}")
    print(f"\n⏱ {elapsed:.1f}s | ⚡ {tokens/elapsed:.1f} tok/s\n")

questions = [
    "Tại sao bầu trời có màu xanh?",
    "Nếu 5 máy làm 5 sản phẩm trong 5 phút, 100 máy cần bao nhiêu phút để làm 100 sản phẩm?",
    "Viết hàm Python kiểm tra số nguyên tố.",
]

for i, q in enumerate(questions, 1):
    print(f"{'─'*50}\n❓ [{i}] {q}\n")
    chat(q)