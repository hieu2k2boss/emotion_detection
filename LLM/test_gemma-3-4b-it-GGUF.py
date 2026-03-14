from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import time

model_path = hf_hub_download(
    repo_id="bartowski/google_gemma-3-4b-it-GGUF",       # ✅ thêm google_
    filename="google_gemma-3-4b-it-Q4_K_M.gguf"          # ✅ thêm google_
)

llm = Llama(model_path=model_path, n_ctx=4096, n_threads=8, verbose=False)
print("✅ Model loaded!\n")

SYSTEM = "Bạn là trợ lý AI thông minh. Hãy trả lời bằng tiếng Việt, rõ ràng và đầy đủ."

def chat(question, max_tokens=512):
    # Gemma dùng format riêng, khác Qwen
    prompt = (
        f"<start_of_turn>user\n{question}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    t = time.time()
    out = llm(prompt, max_tokens=max_tokens, stop=["<end_of_turn>"], echo=False)
    elapsed = time.time() - t
    tokens = out["usage"]["completion_tokens"]
    print(f"💬 {out['choices'][0]['text'].strip()}")
    print(f"\n⏱ {elapsed:.1f}s | ⚡ {tokens/elapsed:.1f} tok/s\n")

questions = [
    "Tại sao bầu trời có màu xanh?",
    "Nếu 5 máy làm 5 sản phẩm trong 5 phút, 100 máy cần mấy phút để làm 100 sản phẩm?",
    "Viết hàm Python kiểm tra số nguyên tố.",
    "Giải phương trình: 2x² - 5x + 3 = 0",
]

for i, q in enumerate(questions, 1):
    print(f"{'─'*50}\n❓ [{i}] {q}\n")
    chat(q)