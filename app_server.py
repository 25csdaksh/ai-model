"""
Python Web Server Backend with Fine-Tuned GPU Model Integration
સ્થાનિક Web Server જે તમારા Fine-Tuned GPU AI Model સાથે કનેક્ટ થાય છે.
"""

import http.server
import socketserver
import json
import urllib.request
import urllib.error
import os
import torch

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# Global variables for loaded GPU model
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = None
model = None

def init_gpu_model():
    global tokenizer, model, device
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        base_path = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
        lora_path = os.path.join(DIRECTORY, "models", "coding-assistant")

        if os.path.exists(lora_path):
            print(f"[+] Loading Fine-Tuned LoRA model on {device.upper()} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})...")
            tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_path,
                dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            model = PeftModel.from_pretrained(base_model, lora_path)
            model.eval()
            print("[+] Fine-Tuned Model loaded successfully into Web Server!")
    except Exception as e:
        print(f"[!] Warning: Could not load GPU model: {e}")

class AIHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_POST(self):
        if self.path == '/api/chat':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                request_json = json.loads(post_data.decode('utf-8'))
                user_message = request_json.get('message', '')
                api_key = request_json.get('apiKey', '')
                system_prompt = request_json.get('systemPrompt', 'તમે એક અત્યંત બુદ્ધિશાળી ગુજરાતી AI આસિસ્ટન્ટ છો.')

                if api_key:
                    bot_response = self.call_gemini(user_message, api_key, system_prompt)
                elif model is not None and tokenizer is not None:
                    bot_response = self.generate_gpu_response(user_message)
                else:
                    bot_response = self.fallback_response(user_message)

                response_data = {"status": "success", "reply": bot_response}
            except Exception as e:
                response_data = {"status": "error", "message": str(e)}

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response_data, ensure_ascii=False).encode('utf-8'))
        else:
            self.send_error(404, "Endpoint not found")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def generate_gpu_response(self, prompt):
        formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
        inputs = tokenizer(formatted, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def call_gemini(self, prompt, api_key, system_prompt):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "systemInstruction": {"parts": [{"text": system_prompt}]}
        }
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result['candidates'][0]['content']['parts'][0]['text']

    def fallback_response(self, text):
        return f"🤖 (Fine-Tuned Model): તમારો પ્રશ્ન '{text}' મળ્યો છે!"

def run_server():
    init_gpu_model()
    with socketserver.TCPServer(("", PORT), AIHandler) as httpd:
        print(f"\n🚀 Web Server is LIVE: http://localhost:{PORT}")
        print("વેબસાઇટ જોવા માટે બ્રાઉઝરમાં http://localhost:8000 ખોલો!\n")
        httpd.serve_forever()

if __name__ == "__main__":
    run_server()
