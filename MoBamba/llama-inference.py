from unsloth import FastLanguageModel
from transformers import AutoTokenizer
import torch
from datetime import datetime
import os

full_pretrain = False
model_name = "Llama-3.2-3B-bnb-4bit" + ("-full" if full_pretrain else "")

# === Load base model ===
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = f"outputs/{model_name}" if full_pretrain else f"unsloth/{model_name}",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# === Load LoRA adapter ===
if not full_pretrain:
    model.load_adapter(f"outputs/{model_name}/checkpoint-543")

# === Generate text ===
prompt = """[Verse 1]\n"""
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=2048,
    do_sample=True,
    temperature=0.95,
    top_k=50,
    top_p=0.90,
    eos_token_id=tokenizer.eos_token_id
)

decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded_output)

# === Save to file ===
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
os.makedirs("model_outputs", exist_ok=True)
filepath = f"model_outputs/{model_name}_{timestamp}.txt"

with open(filepath, "w") as f:
    f.write(decoded_output)
