from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
import math
import json

# Load dataset (just has 'text' field)
dataset = load_dataset("JunhaoYu/processed_rap_lyrics", split="train")

# Load model and tokenizer
model_path = "ibm-ai-platform/Bamba-9B-v2" # Or ibm-ai-platform/Bamba-9B-fp8
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Add PAD token if missing
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    model.resize_token_embeddings(len(tokenizer))

# Formatting: treat each rap lyric as a single text sample
def formatting_prompts_func(example):
    return [text for text in example["text"]]

# No need for a response template in rap generation
collator = DataCollatorForCompletionOnlyLM("", tokenizer=tokenizer)

# Training configuration
train_args = SFTConfig(
    per_device_train_batch_size=2,
    output_dir="outputs/mo-bamba-9B-lora",
    gradient_checkpointing=True,
    num_train_epochs=3,
    save_strategy="epoch",
    learning_rate = 2e-4,
)

# add peft config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    args=train_args,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

# Train the model
trainer_stats = trainer.train()

# Save basic training metrics
metrics = trainer_stats.metrics
with open(f"outputs/mo-bamba-9B-lora/training_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)