from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
import torch
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from datasets import load_dataset

# Dataset "JunhaoYu/processed_rap_lyrics"
tokenizer = AutoTokenizer.from_pretrained("AntonV/mamba2-130m-hf")
model = AutoModelForCausalLM.from_pretrained(
    "AntonV/mamba2-130m-hf",
    load_in_8bit=True
)

# Model "AntonV/mamba2-130m-hf"
dataset = load_dataset("JunhaoYu/processed_rap_lyrics", split="train")

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=2e-3
)

lora_config = LoraConfig(
    r=8,
    target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
    task_type="CAUSAL_LM",
    bias="none"
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=lora_config,
    train_dataset=dataset,
    # dataset_text_field="text",
)
trainer.train()