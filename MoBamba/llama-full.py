from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer
from datasets import load_dataset
import torch
import json

output_dir = "outputs/Llama-3.2-3B-bnb-4bit-full"
max_seq_length = 2048
dtype = None  # Auto-detect (float16 or bfloat16)
load_in_4bit = False  # Full fine-tuning needs full-precision

# Load model and tokenizer (no LoRA)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Enable gradient checkpointing (optional)
model.gradient_checkpointing_enable()

# Load dataset
dataset = load_dataset("JunhaoYu/processed_rap_lyrics", split="train")

# Set up trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=DataCollatorForSeq2Seq(tokenizer),
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
        save_strategy="epoch",
        output_dir=output_dir
    ),
)

train_output = trainer.train()

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Save metrics
with open(f"{output_dir}/training_metrics.json", "w") as f:
    json.dump(train_output.metrics, f, indent=2)

# Save model + tokenizer
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
