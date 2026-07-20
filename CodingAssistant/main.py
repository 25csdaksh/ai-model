# pyrefly: ignore [missing-import]
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # pyrefly: ignore [missing-import]

model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

print(f"Loading tokenizer and model for {model_name}...")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="auto"
)

# Example prompt for coding assistant
prompt = "Write a Python script to calculate Fibonacci sequence up to N numbers with memoization."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful expert coding assistant."},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

print("Generating code response...")
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("\n--- AI Response ---")
print(response)
