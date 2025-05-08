from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from datasets import load_dataset
import json

# === Load dataset ===
dataset = load_dataset("JunhaoYu/processed_rap_lyrics", split="train")

# === Load model and tokenizer ===
model_path = "AntonV/mamba2-2.7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# === Add PAD token if missing ===
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    model.resize_token_embeddings(len(tokenizer))

# === Data collator for causal LM ===
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# === LoRA config (ensure these target modules exist in Mamba) ===
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["in_proj", "out_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# === Training config ===
train_config = SFTConfig(
    output_dir="./outputs/mamba-rap-lora",
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_dir="./logs",
    save_strategy="epoch",
    gradient_checkpointing=True,
)

# === Trainer ===
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=train_config,
    peft_config=lora_config,
    train_dataset=dataset,
    data_collator=collator,
)

# === Train ===
trainer_stats = trainer.train()

# === Save training metrics ===
with open("outputs/mamba-rap-lora/training_metrics.json", "w") as f:
    json.dump(trainer_stats.metrics, f, indent=2)
