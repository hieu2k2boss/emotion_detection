from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, time

model_id = "google/gemma-3-4b-it"  # full model, không cần GGUF

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,  #  GPU dùng bfloat16, nhẹ hơn float32 2x
    device_map="cuda"  #  tự động lên GPU
)


def chat(question, max_tokens=512):
    prompt = f"<bos><start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    t = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
        )
    elapsed = time.time() - t

    result = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    print(f"💬 {result}")
    print(f"\n⏱ {elapsed:.1f}s | ⚡ {tokens / elapsed:.1f} tok/s\n")


chat("Tại sao bầu trời có màu xanh?")