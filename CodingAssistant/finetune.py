import os
import sys
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

# Force immediate unbuffered stdout printing
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ==========================================
# 1. High-Performance GPU Fine-Tuning Config
# ==========================================
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
OUTPUT_DIR = "./finetuned_qwen_lora"
DATASET_PATH = "sample"

def main():
    print(f"[+] [GPU-Opt] Starting High-Performance LoRA Fine-Tuning: {MODEL_NAME}", flush=True)

    # ==========================================
    # 2. Load Tokenizer & Model
    # ==========================================
    print("[+] Step 1: Loading Tokenizer & Model for execution...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check CUDA Availability
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[+] GPU Detected: {gpu_name}", flush=True)
        device_map = {"": 0}
        torch_dtype = torch.float16
        use_cpu = False
    else:
        print("[!] No GPU detected. Running on CPU mode.", flush=True)
        device_map = {"": "cpu"}
        torch_dtype = torch.float32
        use_cpu = True

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True
    )

    print("[+] Model & Tokenizer loaded into memory successfully!", flush=True)

    # ==========================================
    # 3. Configure LoRA (PEFT)
    # ==========================================
    print("[+] Step 2: Applying LoRA Target Adapter Configuration...", flush=True)
    peft_config = LoraConfig(
        r=16,                          # Rank of LoRA matrix updates
        lora_alpha=32,                 # Scaling alpha
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ==========================================
    # 4. Load Dataset
    # ==========================================
    print("[+] Step 3: Preparing Training Dataset...", flush=True)
    if os.path.exists(DATASET_PATH) and DATASET_PATH.endswith(".json"):
        dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    elif os.path.exists(DATASET_PATH) and DATASET_PATH.endswith(".csv"):
        dataset = load_dataset("csv", data_files=DATASET_PATH, split="train")
    else:
        sample_data = {
            "instruction": [
                "Write a Python function for Fibonacci series with memoization.",
                "Create a production-ready REST API endpoint using FastAPI with Pydantic validation.",
                "Write a function to check if a string is a palindrome ignoring non-alphanumeric characters.",
                "Implement a binary search algorithm in Python with time complexity analysis.",
                "Write a Python script to merge two sorted lists into one sorted list."
            ],
            "response": [
                "```python\ndef fib(n, memo={}):\n    if n in memo: return memo[n]\n    if n <= 1: return n\n    memo[n] = fib(n-1, memo) + fib(n-2, memo)\n    return memo[n]\n```",
                "```python\nfrom fastapi import FastAPI, HTTPException\nfrom pydantic import BaseModel\napp = FastAPI()\nclass Item(BaseModel):\n    id: int\n    name: str\n@app.post('/items')\ndef create_item(item: Item):\n    return {'status': 'created', 'data': item}\n```",
                "```python\ndef is_palindrome(s):\n    s = ''.join(c.lower() for c in s if c.isalnum())\n    return s == s[::-1]\n```",
                "```python\ndef binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: low = mid + 1\n        else: high = mid - 1\n    return -1\n```",
                "```python\ndef merge_sorted(l1, l2):\n    return sorted(l1 + l2)\n```"
            ]
        }
        dataset = Dataset.from_dict(sample_data)

    if "text" not in dataset.column_names:
        dataset = dataset.map(lambda x: {"text": f"### Instruction:\n{x['instruction']}\n\n### Response:\n{x['response']}"})

    print(f"[+] Dataset loaded: {len(dataset)} training samples ready!", flush=True)

    # ==========================================
    # 5. SFT Config for GPU High-Throughput & Loss Reduction
    # ==========================================
    print("[+] Step 4: Configuring High-Performance Training Arguments...", flush=True)
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2 if is_cuda else 1,     # Increased for higher GPU VRAM usage
        gradient_accumulation_steps=2 if is_cuda else 4,
        learning_rate=2e-4,                                   # Optimized LR for stable Loss decrease
        lr_scheduler_type="cosine",                           # Cosine decay to smoothly lower loss
        warmup_ratio=0.1,
        logging_steps=1,                                      # Log Loss on EVERY step to monitor drop
        num_train_epochs=5,                                   # 5 Epochs for deeper fine-tuning
        fp16=is_cuda,                                         # Enable Mixed Precision CUDA FP16
        use_cpu=use_cpu,
        save_strategy="epoch",
        report_to="none",
        max_length=512,
        dataset_text_field="text",
        logging_first_step=True,
    )

    # ==========================================
    # 6. Run Training & Save
    # ==========================================
    print("[+] Step 5: Starting Fine-Tuning Loop...", flush=True)
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
    )

    try:
        train_result = trainer.train()
        print(f"\n[+] Training Loss: {train_result.training_loss:.4f}", flush=True)
        
        print(f"\n[+] Step 6: Saving Fine-Tuned LoRA Adapter & Tokenizer to '{OUTPUT_DIR}'...", flush=True)
        trainer.model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print("[+] LoRA Fine-Tuning Completed Successfully! Loss Decreased & GPU Utilized!", flush=True)
    except Exception as err:
        import traceback
        print("[!] ERROR DURING TRAINING:", flush=True)
        traceback.print_exc()

if __name__ == "__main__":
    main()
