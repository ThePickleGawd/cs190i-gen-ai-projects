from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from datetime import datetime
import os

# === Config ===
base_model_path = "AntonV/mamba2-1.3b-hf"
lora_adapter_path = "outputs/mamba-rap-lora/checkpoint-1089"  # or latest checkpoint
prompt = "[Verse 1]\n"

# === Load tokenizer and base model ===
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16).cuda()

# === Load LoRA adapter ===
model = PeftModel.from_pretrained(model, lora_adapter_path)
model.eval()

# === Tokenize prompt ===
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# === Generate text ===
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.95,
        top_k=50,
        top_p=0.90,
        eos_token_id=tokenizer.eos_token_id,
    )

decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded)

# === Save to file ===
timestamp = datetime.now().strftime("%m-%d-%H-%M")
output_dir = f"model_outputs/mamba-rap-lora"
os.makedirs(output_dir, exist_ok=True)
with open(f"{output_dir}/{timestamp}.txt", "w") as f:
    f.write(decoded)
