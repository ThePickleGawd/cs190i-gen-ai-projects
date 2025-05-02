from unsloth import FastLanguageModel
from transformers import AutoTokenizer
import torch

model_name = "Llama-3.2-3B-bnb-4bit"

# === Load base model ===
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = f"unsloth/{model_name}",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# === Load LoRA adapter ===
model.load_adapter(f"outputs/{model_name}/checkpoint-543")

# === Generate text ===
prompt = "[Verse 1]\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.95,
    top_k=75,
    top_p=0.90,
    eos_token_id=tokenizer.eos_token_id
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
