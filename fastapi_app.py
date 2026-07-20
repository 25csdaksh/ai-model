"""
Daksh AI - FastAPI Ultra-Fast GPU AI Chatbot Server
Features: Real-Time Live Token Streaming, KV Cache Optimization, CUDA Inference Mode
Architecture: User -> FastAPI -> Fine-Tuned LoRA Model (NVIDIA RTX 3050 GPU) -> Instant Answer
"""

import os
import sys
import time
import torch
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel

# Force UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
LORA_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "coding-assistant")

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = None
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model, device
    print("=" * 60)
    print("⚡ Starting Ultra-Fast FastAPI GPU Server...")
    print(f"[+] Device: {device.upper()} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print(f"[+] Model Path: {LORA_MODEL_PATH}")
    print("=" * 60)

    try:
        if os.path.exists(LORA_MODEL_PATH):
            print("[+] Loading Tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
            
            print(f"[+] Loading Base Model into {device.upper()} memory...")
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME,
                dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            
            print(f"[+] Applying Fine-Tuned LoRA Weights...")
            model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
            model.eval()
            
            # Enable PyTorch CUDA optimizations
            if device == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            print("⚡ SUCCESS: Ultra-Fast GPU Model ready for instant streaming!")
        else:
            print(f"[!] Warning: LoRA path '{LORA_MODEL_PATH}' not found.")
    except Exception as e:
        print(f"[!] Error loading model: {e}")

    yield
    print("👋 Shutting down FastAPI server...")

app = FastAPI(
    title="Daksh AI - FastAPI GPU Stream Chatbot",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    prompt: str = Field(..., example="Write a Python function for Fibonacci series.")
    max_tokens: int = Field(default=256, ge=16, le=1024)
    temperature: float = Field(default=0.7, ge=0.1, le=1.0)
    system_prompt: str = Field(
        default="You are Qwen2.5-Coder, an expert AI coding assistant fine-tuned for software development."
    )

@app.get("/api/status")
def get_status():
    return {
        "status": "online",
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "model_loaded": model is not None
    }

# Ultra-Fast Live Streaming Endpoint (Server-Sent Events)
@app.post("/api/stream")
async def stream_chat(req: ChatRequest):
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="GPU Model not loaded")

    prompt = f"### System:\n{req.system_prompt}\n\n### Instruction:\n{req.prompt}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        do_sample=True if req.temperature > 0.1 else False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True
    )

    # Run generation in a separate thread so tokens stream out instantly
    thread = threading.Thread(target=_generate_thread, args=(generation_kwargs,))
    thread.start()

    def token_generator():
        for token_text in streamer:
            yield f"data: {token_text}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")

@torch.inference_mode()
def _generate_thread(kwargs):
    model.generate(**kwargs)

# Fast Non-Streaming REST Endpoint
@app.post("/api/chat")
@torch.inference_mode()
def generate_chat(req: ChatRequest):
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="GPU Model not loaded")

    start_time = time.time()
    formatted_prompt = f"### System:\n{req.system_prompt}\n\n### Instruction:\n{req.prompt}\n\n### Response:\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        do_sample=True if req.temperature > 0.1 else False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True
    )

    reply = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    elapsed = round(time.time() - start_time, 3)

    return {
        "prompt": req.prompt,
        "reply": reply,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
        "elapsed_seconds": elapsed,
        "status": "success"
    }

@app.get("/", response_class=HTMLResponse)
def serve_chat_ui():
    gpu_label = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    return f"""<!DOCTYPE html>
<html lang="gu">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Daksh AI - Live Streaming GPU Chatbot</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/remixicon@4.2.0/fonts/remixicon.css" rel="stylesheet"/>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css">
    <style>
        :root {{
            --bg-main: #0f172a;
            --bg-card: #1e293b;
            --bg-input: #334155;
            --primary-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border-color: rgba(255, 255, 255, 0.1);
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--bg-main);
            color: var(--text-main);
            height: 100vh; display: flex; flex-direction: column;
        }}
        header {{
            padding: 16px 30px; background: var(--bg-card);
            border-bottom: 1px solid var(--border-color);
            display: flex; align-items: center; justify-content: space-between;
        }}
        .logo {{ display: flex; align-items: center; gap: 12px; }}
        .logo-icon {{
            width: 40px; height: 40px; border-radius: 10px;
            background: var(--primary-gradient);
            display: flex; align-items: center; justify-content: center;
            font-size: 22px; color: white;
            box-shadow: 0 0 16px rgba(99, 102, 241, 0.4);
        }}
        .gpu-badge {{
            padding: 6px 14px; background: rgba(99, 102, 241, 0.15);
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 20px; color: #818cf8; font-size: 13px; font-weight: 600;
            display: flex; align-items: center; gap: 6px;
        }}
        main {{
            flex: 1; display: flex; flex-direction: column;
            max-width: 900px; width: 100%; margin: 0 auto; padding: 20px;
            overflow: hidden;
        }}
        #chatMessages {{
            flex: 1; overflow-y: auto; display: flex; flex-direction: column;
            gap: 16px; padding: 10px; scroll-behavior: smooth;
        }}
        .msg-row {{ display: flex; gap: 14px; width: 100%; animation: fadeIn 0.3s ease; }}
        .msg-row.user {{ flex-direction: row-reverse; }}
        .avatar {{
            width: 36px; height: 36px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-size: 18px; flex-shrink: 0; color: white;
        }}
        .msg-row.user .avatar {{ background: #3b82f6; }}
        .msg-row.bot .avatar {{ background: var(--primary-gradient); }}
        .bubble {{
            max-width: 80%; padding: 14px 18px; border-radius: 14px;
            font-size: 14.5px; line-height: 1.6; position: relative;
        }}
        .msg-row.user .bubble {{ background: #4f46e5; color: white; border-bottom-right-radius: 4px; }}
        .msg-row.bot .bubble {{ background: var(--bg-card); border: 1px solid var(--border-color); border-top-left-radius: 4px; }}
        pre {{ background: #1e1e2e; padding: 12px; border-radius: 8px; overflow-x: auto; margin: 10px 0; border: 1px solid var(--border-color); }}
        code {{ font-family: 'Fira Code', monospace; font-size: 13px; }}
        .input-box {{
            display: flex; gap: 10px; background: var(--bg-card);
            padding: 10px 14px; border-radius: 16px; border: 1px solid var(--border-color);
            margin-top: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        input[type="text"] {{
            flex: 1; background: transparent; border: none; outline: none;
            color: white; font-family: inherit; font-size: 15px; padding: 8px;
        }}
        button {{
            background: var(--primary-gradient); border: none; color: white;
            padding: 10px 22px; border-radius: 12px; font-weight: 600;
            cursor: pointer; display: flex; align-items: center; gap: 6px;
            transition: transform 0.2s ease;
        }}
        button:hover {{ transform: scale(1.04); }}
        .stats-tag {{ font-size: 11px; color: var(--text-muted); margin-top: 6px; }}
        @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(8px); }} to {{ opacity: 1; transform: translateY(0); }} }}
    </style>
</head>
<body>
    <header>
        <div class="logo">
            <div class="logo-icon"><i class="ri-flashlight-line"></i></div>
            <div>
                <h2>Daksh AI Live Stream</h2>
                <small style="color: var(--text-muted);">Real-Time Token Streaming on GPU</small>
            </div>
        </div>
        <div class="gpu-badge">
            <i class="ri-flashlight-fill"></i> {gpu_label}
        </div>
    </header>
    <main>
        <div id="chatMessages">
            <div class="msg-row bot">
                <div class="avatar"><i class="ri-robot-fill"></i></div>
                <div class="bubble">
                    ⚡ <strong>નમસ્તે! Ultra-Fast Streaming Mode ચાલુ થઈ ગયું છે.</strong><br>
                    હવે તમે પ્રશ્ન પૂછશો એટલે જવાબ સેકન્ડના હજારમા ભાગમાં લાઈવ ટાઇપ થતો દેખાશે! 🚀
                </div>
            </div>
        </div>
        <div class="input-box">
            <input type="text" id="promptInput" placeholder="તમારો કોડિંગ પ્રશ્ન પૂછો..." autocomplete="off">
            <button id="sendBtn"><i class="ri-send-plane-fill"></i> Send</button>
        </div>
    </main>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.1/marked.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <script>
        const chatMessages = document.getElementById('chatMessages');
        const promptInput = document.getElementById('promptInput');
        const sendBtn = document.getElementById('sendBtn');

        promptInput.addEventListener('keypress', (e) => {{ if(e.key === 'Enter') handleSend(); }});
        sendBtn.addEventListener('click', handleSend);

        async function handleSend() {{
            const text = promptInput.value.trim();
            if(!text) return;

            appendMsg('user', text);
            promptInput.value = '';

            const botBubble = appendMsg('bot', '⚡ ...');
            let fullText = '';
            const startTime = performance.now();

            try {{
                const response = await fetch('/api/stream', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ prompt: text, max_tokens: 256, temperature: 0.7 }})
                }});

                const reader = response.body.getReader();
                const decoder = new TextDecoder();

                while (true) {{
                    const {{ done, value }} = await reader.read();
                    if (done) break;

                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\\n');

                    for (const line of lines) {{
                        if (line.startsWith('data: ')) {{
                            const token = line.replace('data: ', '');
                            if (token === '[DONE]') break;
                            fullText += token;
                            botBubble.innerHTML = marked.parse(fullText);
                            document.querySelectorAll('pre code').forEach((b) => hljs.highlightElement(b));
                            chatMessages.scrollTop = chatMessages.scrollHeight;
                        }}
                    }}
                }}

                const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
                botBubble.innerHTML += `<div class="stats-tag">⚡ Live Stream Speed: ${{elapsed}}s | NVIDIA RTX 3050 GPU</div>`;
            }} catch(err) {{
                botBubble.innerHTML = `⚠️ Error: ${{err.message}}`;
            }}
        }}

        function appendMsg(role, content) {{
            const row = document.createElement('div');
            row.className = `msg-row ${{role}}`;
            row.innerHTML = `
                <div class="avatar"><i class="ri-${{role==='user'?'user-3-fill':'robot-fill'}}"></i></div>
                <div class="bubble">${{marked.parse(content)}}</div>
            `;
            chatMessages.appendChild(row);
            chatMessages.scrollTop = chatMessages.scrollHeight;
            return row.querySelector('.bubble');
        }}
    </script>
</body>
</html>"""
