"""
Use Fine-Tuned LoRA Model for Code Generation
તમારા Fine-Tuned AI Model ને ટેસ્ટ કરવાનો Python સ્ક્રિપ્ટ
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
LORA_PATH = "models/coding-assistant"

def main():
    print("=" * 60)
    print("🤖 Loading Fine-Tuned LoRA Model on GPU...")
    print("=" * 60)

    # Check CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    # Load Base Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )

    # Load LoRA Weights
    print(f"[+] Loading LoRA Fine-Tuned Weights from '{LORA_PATH}'...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    print("\n✅ Fine-Tuned Model ready! Type 'exit' to quit.\n")

    while True:
        prompt = input("👤 Ask your coding question: ").strip()
        if not prompt or prompt.lower() in ["exit", "quit"]:
            print("👋 Bye!")
            break

        formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

        print("\n🤖 AI Output:")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(response)
        print("-" * 60 + "\n")

if __name__ == "__main__":
    main()
