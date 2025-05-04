from unsloth import FastLanguageModel
from transformers import AutoTokenizer
import torch
from datetime import datetime
import os

is_qlora = False
is_lora = False
is_full_finetune = True
model_name = "Llama-3.2-1B-bnb-4bit"

# === Load base model ===
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=f"outputs/{model_name}-full" if is_full_finetune else f"unsloth/{model_name}",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# === Load LoRA adapter ===
if is_lora:
    model.load_adapter(f"outputs/{model_name}-lora/checkpoint-543")
    model_name = model_name + "-lora"
elif is_qlora:
    model.load_adapter(f"outputs/{model_name}-qlora/checkpoint-543")
    model_name = model_name + "-lora"
elif is_full_finetune:
    model_name = model_name + "-full"

# === Generate text ===
prompt = """[Verse 1]\n"""
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.95,
    top_k=50,
    top_p=0.90,
    eos_token_id=tokenizer.eos_token_id
)

decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded_output)

# === Save to file ===
timestamp = datetime.now().strftime("%m-%d-%H-%M")
os.makedirs("model_outputs", exist_ok=True)
filepath = f"model_outputs/{model_name}_{timestamp}.txt"

with open(filepath, "w") as f:
    f.write(decoded_output)
