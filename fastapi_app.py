"""
Daksh AI - FastAPI Ultra-Fast GPU AI Chatbot Server
Redesigned UI/UX: Pastel Gradient Mesh, Floating Mac-Window Chat Widget, Translucent Glassmorphism
Architecture: User -> FastAPI -> Fine-Tuned LoRA Model (NVIDIA RTX 3050 GPU) -> Instant Streaming Answer
"""

import os
import sys
import time
import json
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

DEFAULT_SYSTEM_PROMPT = "You are Qwen2.5-Coder, an expert AI programming assistant. Always format code blocks cleanly with proper newlines and indentation inside Markdown blocks (e.g. ```c\n#include <stdio.h>\n...\n```). Never collapse code into a single line."

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
    title="Daksh AI - FastAPI Premium GPU Chatbot",
    version="3.0.0",
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
    max_tokens: int = Field(default=384, ge=16, le=1024)
    temperature: float = Field(default=0.7, ge=0.1, le=1.0)
    system_prompt: str = Field(default=DEFAULT_SYSTEM_PROMPT)

def build_chatml_prompt(user_prompt: str, sys_prompt: str) -> str:
    return f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

@app.get("/api/status")
def get_status():
    return {
        "status": "online",
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "model_loaded": model is not None
    }

@app.post("/api/stream")
async def stream_chat(req: ChatRequest):
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="GPU Model not loaded")

    prompt = build_chatml_prompt(req.prompt, req.system_prompt or DEFAULT_SYSTEM_PROMPT)
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

    thread = threading.Thread(target=_generate_thread, args=(generation_kwargs,))
    thread.start()

    def token_generator():
        for token_text in streamer:
            json_token = json.dumps(token_text)
            yield f"data: {json_token}\n\n"
        yield 'data: "[DONE]"\n\n'

    return StreamingResponse(token_generator(), media_type="text/event-stream")

@torch.inference_mode()
def _generate_thread(kwargs):
    model.generate(**kwargs)

@app.post("/api/chat")
@torch.inference_mode()
def generate_chat(req: ChatRequest):
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="GPU Model not loaded")

    start_time = time.time()
    prompt = build_chatml_prompt(req.prompt, req.system_prompt or DEFAULT_SYSTEM_PROMPT)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

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
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Custom AI Chatbot Trained On Your Data</title>
    
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
    
    <!-- Remix Icons & Highlight.js CSS -->
    <link href="https://cdn.jsdelivr.net/npm/remixicon@4.2.0/fonts/remixicon.css" rel="stylesheet"/>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css">
    
    <style>
        :root {{
            --bg-gradient: linear-gradient(135deg, #fce7f3 0%, #e0e7ff 45%, #fae8ff 75%, #f1f5f9 100%);
            --glass-bg: rgba(255, 255, 255, 0.55);
            --glass-border: rgba(255, 255, 255, 0.7);
            --glass-shadow: 0 20px 50px rgba(0, 0, 0, 0.06);
            --text-dark: #0f172a;
            --text-muted: #64748b;
            --accent-black: #09090b;
            --accent-pink: #ec4899;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Plus Jakarta Sans', sans-serif;
            background: var(--bg-gradient);
            color: var(--text-dark);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
        }}

        /* Header Navigation Bar */
        nav {{
            padding: 24px 5%;
            display: flex;
            align-items: center;
            justify-content: space-between;
            width: 100%;
            max-width: 1400px;
            margin: 0 auto;
        }}

        .nav-left {{
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            padding: 8px 20px 8px 12px;
            border-radius: 40px;
            display: flex;
            align-items: center;
            gap: 24px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.03);
        }}

        .logo-box {{
            width: 36px;
            height: 36px;
            background: var(--accent-black);
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
        }}

        .nav-links {{
            display: flex;
            gap: 20px;
            list-style: none;
        }}

        .nav-links a {{
            text-decoration: none;
            color: var(--text-dark);
            font-size: 14px;
            font-weight: 600;
            transition: color 0.2s;
        }}

        .nav-links a:hover {{
            color: #6366f1;
        }}

        .get-started-btn {{
            background: var(--accent-black);
            color: white;
            padding: 12px 24px;
            border-radius: 40px;
            font-weight: 600;
            font-size: 14px;
            text-decoration: none;
            display: flex;
            align-items: center;
            gap: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: all 0.2s ease;
        }}

        .get-started-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        }}

        /* Main Container */
        .container {{
            flex: 1;
            max-width: 1350px;
            width: 100%;
            margin: 0 auto;
            padding: 20px 5% 40px;
            display: grid;
            grid-template-columns: 1fr 1.1fr;
            gap: 40px;
            align-items: center;
        }}

        /* Left Hero Content */
        .hero-section {{
            display: flex;
            flex-direction: column;
            gap: 24px;
        }}

        .hero-title {{
            font-size: 54px;
            font-weight: 800;
            line-height: 1.12;
            letter-spacing: -1.5px;
            color: var(--accent-black);
        }}

        .hero-subtitle {{
            font-size: 16px;
            color: var(--text-muted);
            font-weight: 500;
        }}

        .cta-pill {{
            align-self: flex-start;
            background: var(--glass-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--glass-border);
            padding: 12px 24px;
            border-radius: 40px;
            font-weight: 600;
            font-size: 14.5px;
            color: var(--text-dark);
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.03);
            transition: all 0.2s ease;
            margin-top: 10px;
        }}

        .cta-pill:hover {{
            background: white;
            transform: translateY(-2px);
        }}

        .trusted-section {{
            margin-top: 40px;
        }}

        .trusted-title {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted);
            font-weight: 700;
            margin-bottom: 16px;
        }}

        .brand-logos {{
            display: flex;
            align-items: center;
            gap: 24px;
            flex-wrap: wrap;
            opacity: 0.75;
        }}

        .brand-badge {{
            font-size: 13px;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 6px;
            color: var(--text-dark);
        }}

        /* Right Floating Mac Window Chat Widget */
        .chat-window {{
            background: var(--glass-bg);
            backdrop-filter: blur(24px);
            border: 1px solid var(--glass-border);
            border-radius: 28px;
            box-shadow: var(--glass-shadow);
            height: 600px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            position: relative;
        }}

        .mac-header {{
            padding: 16px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid rgba(255, 255, 255, 0.4);
            background: rgba(255, 255, 255, 0.2);
        }}

        .mac-dots {{
            display: flex;
            gap: 8px;
        }}

        .dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }}
        .dot.red {{ background: #ff5f56; }}
        .dot.yellow {{ background: #ffbd2e; }}
        .dot.green {{ background: #27c93f; }}

        .mac-title {{
            font-size: 12px;
            font-weight: 700;
            color: var(--text-muted);
            letter-spacing: 0.5px;
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        /* Chat Messages Container */
        .chat-body {{
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 16px;
            scroll-behavior: smooth;
        }}

        .msg-group {{
            display: flex;
            flex-direction: column;
            gap: 6px;
            max-width: 88%;
        }}

        .msg-group.user {{
            align-self: flex-end;
            align-items: flex-end;
        }}

        .msg-group.bot {{
            align-self: flex-start;
            align-items: flex-start;
        }}

        .msg-bubble {{
            padding: 14px 18px;
            border-radius: 20px;
            font-size: 14.5px;
            line-height: 1.6;
        }}

        .msg-group.user .msg-bubble {{
            background: var(--accent-black);
            color: white;
            border-bottom-right-radius: 4px;
        }}

        .msg-group.bot .msg-bubble {{
            background: rgba(255, 255, 255, 0.85);
            color: var(--text-dark);
            border-top-left-radius: 4px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.02);
            border: 1px solid rgba(255, 255, 255, 0.9);
        }}

        pre {{
            background: #181825;
            color: #cdd6f4;
            padding: 14px;
            border-radius: 12px;
            overflow-x: auto;
            margin: 10px 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}

        code {{
            font-family: 'Fira Code', monospace;
            font-size: 13px;
        }}

        /* Quick Suggestions Chips */
        .suggestions-row {{
            display: flex;
            gap: 10px;
            padding: 0 20px 10px;
            overflow-x: auto;
        }}

        .chip-btn {{
            background: var(--accent-black);
            color: white;
            border: none;
            padding: 10px 18px;
            border-radius: 30px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            white-space: nowrap;
            transition: all 0.2s ease;
            font-family: inherit;
        }}

        .chip-btn:hover {{
            transform: translateY(-1px);
            opacity: 0.9;
        }}

        /* Bottom Floating Input Bar */
        .chat-footer {{
            padding: 16px 20px 20px;
            background: rgba(255, 255, 255, 0.2);
            border-top: 1px solid rgba(255, 255, 255, 0.4);
        }}

        .input-pill {{
            background: rgba(255, 255, 255, 0.75);
            border: 1px solid rgba(255, 255, 255, 0.9);
            border-radius: 40px;
            padding: 6px 8px 6px 20px;
            display: flex;
            align-items: center;
            gap: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.03);
            transition: all 0.2s ease;
        }}

        .input-pill:focus-within {{
            background: white;
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.15);
            border-color: #818cf8;
        }}

        .input-pill input {{
            flex: 1;
            background: transparent;
            border: none;
            outline: none;
            color: var(--text-dark);
            font-size: 14.5px;
            font-family: inherit;
        }}

        .send-arrow-btn {{
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: var(--accent-black);
            color: white;
            border: none;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            cursor: pointer;
            transition: all 0.2s ease;
        }}

        .send-arrow-btn:hover {{
            transform: scale(1.06);
        }}

        .speed-badge {{
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 4px;
            font-weight: 500;
        }}

        /* Responsive Breakpoints */
        @media (max-width: 968px) {{
            .container {{
                grid-template-columns: 1fr;
                gap: 30px;
            }}
            .hero-title {{
                font-size: 38px;
            }}
            .chat-window {{
                height: 520px;
            }}
        }}
    </style>
</head>
<body>

    <!-- Header Navigation -->
    <nav>
        <div class="nav-left">
            <div class="logo-box">✨</div>
            <ul class="nav-links">
                <li><a href="#">Home</a></li>
                <li><a href="#">Product</a></li>
                <li><a href="#">Resource</a></li>
                <li><a href="#">Contact</a></li>
            </ul>
        </div>
        <a href="#" class="get-started-btn">
            <span>Get Started</span>
            <i class="ri-arrow-up-right-line"></i>
        </a>
    </nav>

    <!-- Main Section -->
    <div class="container">
        <!-- Left Hero Column -->
        <div class="hero-section">
            <h1 class="hero-title">Custom AI Chatbot trained on your data</h1>
            <p class="hero-subtitle">FastAPI Local GPU Inference Engine powered by NVIDIA GeForce RTX 3050 & Qwen2.5-Coder.</p>
            
            <a href="#" class="cta-pill">
                <span>Try Chat-GPT bot</span>
                <i class="ri-arrow-up-right-line"></i>
            </a>

            <div class="trusted-section">
                <div class="trusted-title">Trusted by & Tech Stack</div>
                <div class="brand-logos">
                    <div class="brand-badge"><i class="ri-flashlight-line"></i> FastAPI</div>
                    <div class="brand-badge"><i class="ri-cpu-line"></i> RTX 3050 GPU</div>
                    <div class="brand-badge"><i class="ri-code-s-slash-line"></i> Qwen2.5</div>
                    <div class="brand-badge"><i class="ri-fire-line"></i> PyTorch CUDA</div>
                </div>
            </div>
        </div>

        <!-- Right Floating Chat Widget Container -->
        <div class="chat-window">
            <!-- Mac Top Window Header -->
            <div class="mac-header">
                <div class="mac-dots">
                    <div class="dot red"></div>
                    <div class="dot yellow"></div>
                    <div class="dot green"></div>
                </div>
                <div class="mac-title">
                    <i class="ri-sparkling-fill" style="color: #6366f1;"></i>
                    <span>{gpu_label}</span>
                </div>
            </div>

            <!-- Chat History -->
            <div class="chat-body" id="chatBody">
                <div class="msg-group bot">
                    <div class="msg-bubble">
                        Hi there, ask me anything! 👋<br>
                        This site is powered by your fine-tuned local GPU model. Ask any coding question or syntax prompt below.
                    </div>
                </div>
            </div>

            <!-- Quick Suggestions Pills -->
            <div class="suggestions-row">
                <button class="chip-btn" onclick="sendPrompt('print hello world in c language like syntex form')">Print C Hello World</button>
                <button class="chip-btn" onclick="sendPrompt('Write a Python function for Fibonacci series.')">Fibonacci Python</button>
                <button class="chip-btn" onclick="sendPrompt('Create a REST API endpoint using FastAPI.')">FastAPI Example</button>
            </div>

            <!-- Bottom Floating Input Bar -->
            <div class="chat-footer">
                <div class="input-pill">
                    <input type="text" id="userInput" placeholder="Ask a question..." autocomplete="off">
                    <button class="send-arrow-btn" id="sendBtn">
                        <i class="ri-arrow-right-line"></i>
                    </button>
                </div>
            </div>
        </div>
    </div>

    <!-- JS Libraries -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.1/marked.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>

    <script>
        marked.setOptions({{
            breaks: true,
            gfm: true
        }});

        const chatBody = document.getElementById('chatBody');
        const userInput = document.getElementById('userInput');
        const sendBtn = document.getElementById('sendBtn');

        userInput.addEventListener('keypress', (e) => {{ if (e.key === 'Enter') handleSend(); }});
        sendBtn.addEventListener('click', handleSend);

        function sendPrompt(promptText) {{
            userInput.value = promptText;
            handleSend();
        }}

        async function handleSend() {{
            const text = userInput.value.trim();
            if (!text) return;

            appendMessage('user', text);
            userInput.value = '';

            const botBubble = appendMessage('bot', '⚡ ...');
            let fullText = '';
            const startTime = performance.now();

            try {{
                const response = await fetch('/api/stream', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ prompt: text, max_tokens: 384, temperature: 0.7 }})
                }});

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';

                while (true) {{
                    const {{ done, value }} = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, {{ stream: true }});
                    const lines = buffer.split('\\n\\n');
                    buffer = lines.pop();

                    for (const line of lines) {{
                        const trimmed = line.trim();
                        if (trimmed.startsWith('data: ')) {{
                            const jsonStr = trimmed.slice(6);
                            if (jsonStr === '"[DONE]"') break;
                            try {{
                                const token = JSON.parse(jsonStr);
                                fullText += token;
                                botBubble.innerHTML = marked.parse(fullText);
                                document.querySelectorAll('pre code').forEach((b) => hljs.highlightElement(b));
                                chatBody.scrollTop = chatBody.scrollHeight;
                            }} catch (err) {{
                                console.error(err);
                            }}
                        }}
                    }}
                }}

                const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
                botBubble.innerHTML += `<div class="speed-badge">⚡ Live Stream: ${{elapsed}}s | NVIDIA RTX 3050 GPU</div>`;
            }} catch (err) {{
                botBubble.innerHTML = `⚠️ Error: ${{err.message}}`;
            }}
        }}

        function appendMessage(role, content) {{
            const group = document.createElement('div');
            group.className = `msg-group ${{role}}`;
            group.innerHTML = `<div class="msg-bubble">${{marked.parse(content)}}</div>`;
            chatBody.appendChild(group);
            chatBody.scrollTop = chatBody.scrollHeight;
            return group.querySelector('.msg-bubble');
        }}
    </script>
</body>
</html>"""
