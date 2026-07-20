import os
import sys
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# Force immediate unbuffered stdout printing
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Optimize CPU Threads
num_cores = os.cpu_count() or 4
torch.set_num_threads(num_cores)

# ==========================================
# 1. Fast CPU/GPU Fine-Tuning Config
# ==========================================
# Using 0.5B Model: Ultra lightweight, 5x faster on CPU, no memory errors
MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
OUTPUT_DIR = "./finetuned_qwen_lora"

# Custom Progress Callback to print step progress immediately
class LiveProgressCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        max_steps = state.max_steps
        loss_str = ""
        if state.log_history:
            latest = state.log_history[-1]
            if "loss" in latest:
                loss_str = f" | Loss: {latest['loss']:.4f}"
        print(f"👉 [Progress] Step {step}/{max_steps} completed!{loss_str}", flush=True)

def main():
    print(f"[+] Starting Fast LoRA Fine-Tuning Pipeline: {MODEL_NAME}", flush=True)
    print(f"[+] CPU Threads Configured: {num_cores} cores", flush=True)

    # Check CUDA Availability
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[+] GPU Detected: {gpu_name}", flush=True)
        device_map = {"": 0}
        torch_dtype = torch.float16
        use_cpu = False
    else:
        print("[!] Running in Fast CPU Mode...", flush=True)
        device_map = {"": "cpu"}
        torch_dtype = torch.float32
        use_cpu = True

    # ==========================================
    # 2. Load Tokenizer & Model
    # ==========================================
    print("[+] Step 1: Loading Tokenizer & Model into memory...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True
    )

    print("[+] Model & Tokenizer loaded successfully!", flush=True)

    # ==========================================
    # 3. Configure LoRA (PEFT)
    # ==========================================
    print("[+] Step 2: Applying LoRA Target Adapter Configuration...", flush=True)
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
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
    dataset = dataset.map(lambda x: {"text": f"### Instruction:\n{x['instruction']}\n\n### Response:\n{x['response']}"})

    print(f"[+] Dataset ready with {len(dataset)} examples!", flush=True)

    # ==========================================
    # 5. SFT Config for Ultra-Fast Completion
    # ==========================================
    print("[+] Step 4: Configuring Fast Training Parameters...", flush=True)
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        max_steps=3,                                          # Fast 3 steps completion!
        logging_steps=1,                                      # Log Loss on EVERY step
        fp16=is_cuda,
        use_cpu=use_cpu,
        save_strategy="no",
        report_to="none",
        max_length=128,
        dataset_text_field="text",
        logging_first_step=True,
        disable_tqdm=False
    )

    # ==========================================
    # 6. Run Training & Save
    # ==========================================
    print("[+] Step 5: Executing Fine-Tuning Steps...", flush=True)
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
        callbacks=[LiveProgressCallback()]
    )

    try:
        train_result = trainer.train()
        print(f"\n[+] Final Training Loss: {train_result.training_loss:.4f}", flush=True)
        
        print(f"\n[+] Step 6: Saving Fine-Tuned LoRA Adapter Weights to '{OUTPUT_DIR}'...", flush=True)
        trainer.model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print("[+] SUCCESS: LoRA Fine-Tuning & Model Saving Completed!", flush=True)
    except Exception as err:
        import traceback
        print("[!] ERROR DURING TRAINING:", flush=True)
        traceback.print_exc()

if __name__ == "__main__":
    main()
