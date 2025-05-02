from unsloth import FastLanguageModel
from transformers import AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "ibm-ai-platform/Bamba-9B-v2",
    max_seq_length = 24,
    load_in_4bit = True,
    max_lora_rank=16,
    gpu_memory_utilization=0.1
)

message = ["Mamba is a snake with following properties  "]
inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

outputs = model.generate(**inputs, max_new_tokens=64)
# outputs = model._old_generate(**inputs, max_new_tokens=64)

print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])