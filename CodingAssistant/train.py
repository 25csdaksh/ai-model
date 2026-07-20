import os
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from datasets import load_dataset, Dataset
# pyrefly: ignore [missing-import]
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
# pyrefly: ignore [missing-import]
from peft import LoraConfig, get_peft_model, TaskType
# pyrefly: ignore [missing-import]
from trl import SFTTrainer, SFTConfig

# ==========================================
# 1. Fine-Tuning Configurations
# ==========================================
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # or "Qwen/Qwen2.5-Coder-7B-Instruct"
OUTPUT_DIR = "./finetuned_qwen_lora"
DATASET_PATH = "sample"  # Set to your local .json / .csv file or Hugging Face dataset name

def main():
    print(f"[+] Starting LoRA Fine-Tuning Pipeline for model: {MODEL_NAME}")

    # ==========================================
    # 2. Load Tokenizer & Model
    # ==========================================
    print("[+] Step 1: Loading Tokenizer & Model...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine Device & Precision
    if torch.cuda.is_available():
        device_map = {"": 0}
        torch_dtype = torch.float16
        use_cpu = False
    else:
        device_map = {"": "cpu"}
        torch_dtype = torch.float32
        use_cpu = True

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True
    )

    print("[+] Model & Tokenizer loaded successfully!")

    # ==========================================
    # 3. Configure LoRA (PEFT)
    # ==========================================
    print("[+] Step 2: Applying LoRA Configuration...")
    peft_config = LoraConfig(
        r=16,                          # Rank of LoRA updates
        lora_alpha=32,                 # Scaling factor
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Target attention layers
        lora_dropout=0.05,             # Dropout probability
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ==========================================
    # 4. Load Dataset
    # ==========================================
    print("[+] Step 3: Loading Training Dataset...")
    if os.path.exists(DATASET_PATH) and DATASET_PATH.endswith(".json"):
        dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    elif os.path.exists(DATASET_PATH) and DATASET_PATH.endswith(".csv"):
        dataset = load_dataset("csv", data_files=DATASET_PATH, split="train")
    else:
        print("[*] Using Sample Demonstration Instruction Dataset...")
        sample_data = {
            "instruction": [
                "Write a Python function for Fibonacci series.",
                "Create a REST API endpoint using FastAPI.",
                "Write a function to check if a string is a palindrome."
            ],
            "response": [
                "```python\ndef fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        yield a\n        a, b = b, a + b\n```",
                "```python\nfrom fastapi import FastAPI\napp = FastAPI()\n@app.get('/')\ndef read_root():\n    return {'status': 'success'}\n```",
                "```python\ndef is_palindrome(s):\n    s = ''.join(c.lower() for c in s if c.isalnum())\n    return s == s[::-1]\n```"
            ]
        }
        dataset = Dataset.from_dict(sample_data)

    # Format dataset into single text column for SFT
    if "text" not in dataset.column_names:
        dataset = dataset.map(lambda x: {"text": f"### Instruction:\n{x['instruction']}\n\n### Response:\n{x['response']}"})

    print(f"[+] Dataset ready with {len(dataset)} examples!")

    # ==========================================
    # 5. Training Arguments Setup
    # ==========================================
    print("[+] Step 4: Configuring SFT Parameters...")
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        num_train_epochs=3,
        fp16=torch.cuda.is_available(),
        use_cpu=use_cpu,
        save_strategy="epoch",
        report_to="none",
        max_length=512,
        dataset_text_field="text",
    )


    # ==========================================
    # 6. Initialize & Run Trainer
    # ==========================================
    print("[+] Step 5: Initializing SFT Trainer & Starting Fine-Tuning...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
    )

    trainer.train()

    # Save fine-tuned LoRA weights & tokenizer
    print(f"[+] Step 6: Saving Fine-Tuned LoRA Adapter Weights to {OUTPUT_DIR}...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("[+] LoRA Fine-Tuning Completed Successfully!")

if __name__ == "__main__":
    main()
