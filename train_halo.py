import json
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset

model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    max_seq_length=1024,
    load_in_4bit=True,
    device_map="cuda:0",
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

with open("halo_training_final.json", "r", encoding="utf-8") as f:
    raw = json.load(f)

def format_sample(example):
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

dataset = Dataset.from_list(raw)
dataset = dataset.map(lambda x: {"text": format_sample(x)})

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=1024,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=2e-4,
        bf16=True,
        fp16=False,
        logging_steps=1,
        output_dir="./halo_output",
        save_strategy="epoch",
        gradient_checkpointing=True,
    ),
)
trainer.train()

model.save_pretrained("./halo_lora")
tokenizer.save_pretrained("./halo_lora")
print("Done.")
