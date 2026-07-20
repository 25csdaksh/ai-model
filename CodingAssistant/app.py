import os
import sys
import torch
import gradio as gr

# Force immediate unbuffered output
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# GPU Model Config
device = "cuda" if torch.cuda.is_available() else "cpu"
gpu_model = None
gpu_tokenizer = None

def load_local_fine_tuned_model():
    global gpu_model, gpu_tokenizer, device
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        base_model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
        lora_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "coding-assistant"))

        if os.path.exists(lora_dir):
            print(f"[+] Web App loading Fine-Tuned GPU model on {device.upper()}...", flush=True)
            gpu_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            gpu_model = PeftModel.from_pretrained(base_model, lora_dir)
            gpu_model.eval()
            print("[+] SUCCESS: Fine-Tuned GPU Model loaded into Web UI!", flush=True)
    except Exception as e:
        print(f"[!] Warning: Could not pre-load GPU model: {e}", flush=True)

load_local_fine_tuned_model()

DEFAULT_SYSTEM_PROMPT = """You are Qwen2.5-Coder, an expert AI coding assistant fine-tuned for software development.
Your job is to help developers write clean, efficient, bug-free, and well-documented code.
Always format code in standard Markdown code blocks with language identifiers. Provide clear explanations."""

def chat_fn(message, history):
    if not message or not message.strip():
        return "Please enter a coding question."

    if gpu_model is not None and gpu_tokenizer is not None:
        try:
            prompt = f"### System:\n{DEFAULT_SYSTEM_PROMPT}\n\n### Instruction:\n{message}\n\n### Response:\n"
            inputs = gpu_tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = gpu_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=gpu_tokenizer.eos_token_id
                )
            
            reply = gpu_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return reply
        except Exception as err:
            return f"⚠️ GPU Generation Error: {str(err)}"

    return f"🤖 AI Response: Your prompt '{message}' was received!"

# Gradio ChatInterface
demo = gr.ChatInterface(
    fn=chat_fn,
    title="🚀 Daksh AI - Fine-Tuned GPU Coding Assistant",
    description=f"**Device:** `{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}` | **Model:** `./finetuned_qwen_lora`",
    textbox=gr.Textbox(placeholder="Ask your coding question (e.g. Write a Python script for Fibonacci)...", container=False, scale=7),
    examples=[
        "Write a Python script for Fibonacci sequence with memoization.",
        "Create a production-ready FastAPI REST API with Pydantic validation.",
        "Write a function to check if a string is a palindrome."
    ]
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True, inbrowser=True)
