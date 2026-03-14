from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, time

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

print("⬇️ Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,  # CPU dùng float32
    device_map="cpu"
)

prompt = "Nếu có 5 máy sản xuất 5 sản phẩm trong 5 phút, 100 máy cần bao nhiêu phút?"
inputs = tokenizer(prompt, return_tensors="pt")

start = time.time()
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.6, do_sample=True)
elapsed = time.time() - start

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print(f"\n⏱ {elapsed:.1f}s")