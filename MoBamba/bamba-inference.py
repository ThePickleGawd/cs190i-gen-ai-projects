from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from datetime import datetime
import os

is_qlora = False
is_lora = True  # Change based on the adapter used
is_full_finetune = False
model_base = "ibm-ai-platform/Bamba-9B-v2"
model_name = "mo-bamba-9B"

# === Load base model and tokenizer ===
model = AutoModelForCausalLM.from_pretrained(
    model_base,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_base)

# === Load LoRA adapter ===
if is_lora:
    adapter_path = f"outputs/{model_name}-lora/checkpoint-543"
    model = PeftModel.from_pretrained(model, adapter_path)
    model_name += "-lora"
elif is_qlora:
    adapter_path = f"outputs/{model_name}-qlora/checkpoint-543"
    model = PeftModel.from_pretrained(model, adapter_path)
    model_name += "-qlora"
elif is_full_finetune:
    model_path = f"outputs/{model_name}-full"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    model_name += "-full"

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
