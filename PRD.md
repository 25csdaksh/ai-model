# Product Requirement Document (PRD)
## Daksh AI - Fine-Tuned GPU Coding Assistant & FastAPI Web Platform

---

## 1. Executive Summary

**Daksh AI** is an end-to-end, high-performance local AI Coding Assistant designed to fine-tune lightweight Large Language Models (LLMs) on domain-specific programming data and deliver real-time token streaming responses via a modern glassmorphic web interface.

The system is optimized specifically for **NVIDIA GeForce RTX 3050 (6GB VRAM)** laptop GPUs using PyTorch CUDA 12.1 acceleration, PEFT/LoRA adapter fine-tuning, FastAPI REST backends, and Server-Sent Events (SSE) streaming.

---

## 2. Product Objectives & Vision

- **Ultra-Fast Local Inference**: Deliver sub-second code generation using hardware acceleration (NVIDIA RTX 3050 GPU + PyTorch CUDA 12.1).
- **Line-by-Line Formatted Code Output**: Eliminate text collapsing bugs by implementing Qwen2.5 ChatML prompt templates and Markdown code block syntax highlighting.
- **Multi-Device Accessibility**: Provide local PC access (`127.0.0.1:8000`), local Wi-Fi network access (`0.0.0.0:8000`), and encrypted global public access via ngrok HTTPS tunnels.
- **State-of-the-Art UI/UX**: Feature a soft pastel gradient mesh design, floating glassmorphic Mac window chat widget, and quick suggestion chips.

---

## 3. Technology Stack & Dependencies

| Layer | Component | Technology / Library |
| :--- | :--- | :--- |
| **Hardware** | GPU | NVIDIA GeForce RTX 3050 (6GB VRAM Laptop GPU) |
| **Compute Core** | Deep Learning Core | PyTorch 2.5.1 + CUDA 12.1 (`torch-2.5.1+cu121`) |
| **Base Model** | Foundation LLM | `Qwen/Qwen2.5-Coder-0.5B-Instruct` |
| **Fine-Tuning** | Parameter Efficient Tuning | PEFT LoRA (`r=8`, `alpha=16`, target: `q_proj`, `v_proj`), Hugging Face `TRL / SFTTrainer` |
| **Model Storage** | Weight Persistence | `d:\daksh\models\coding-assistant\` |
| **Backend Server** | Web API | FastAPI 0.139, Uvicorn 0.51 (ASGI), Pydantic v2 |
| **Streaming** | Token Streaming | Hugging Face `TextIteratorStreamer`, EventSource / SSE (JSON-encoded tokens) |
| **Tunneling** | Public Access | `pyngrok 8.1.2` (HTTPS Tunnel) |
| **Frontend UI** | Client Interface | Vanilla HTML5/CSS3 (Pastel Glassmorphism), `marked.js` v12, `highlight.js` v11 |

---

## 4. System Architecture & Data Flow

```
[ User Device (PC / Wi-Fi Mobile / Public Internet) ]
                        │
                        ▼
[ Public ngrok HTTPS / Local Uvicorn Server (0.0.0.0:8000) ]
                        │
                        ▼
[ FastAPI App (fastapi_app.py) - /api/stream ]
                        │
                        ▼
[ Qwen ChatML Prompt Builder (<|im_start|>system...<|im_end|>) ]
                        │
                        ▼
[ PyTorch CUDA Inference Engine (@torch.inference_mode() + KV Cache) ]
                        │
                        ▼
[ Fine-Tuned Model Weights (models/coding-assistant/ on RTX 3050 GPU) ]
                        │
                        ▼
[ Real-Time JSON SSE Token Stream -> Line-by-Line Markdown Web UI ]
```

---

## 5. Key Functional Requirements

### 5.1. Model Fine-Tuning & Storage
- **Requirement 5.1.1**: The system shall fine-tune `Qwen2.5-Coder-0.5B-Instruct` using LoRA adapter targets on CUDA.
- **Requirement 5.1.2**: All adapter configuration and model weights shall be persisted cleanly in `d:\daksh\models\coding-assistant\`.
- **Requirement 5.1.3**: Fine-tuning step progress and training loss shall log in real-time.

### 5.2. FastAPI Backend & Streaming API
- **Requirement 5.2.1**: FastAPI server shall auto-detect CUDA and load base model + LoRA adapter weights on startup.
- **Requirement 5.2.2**: The `/api/stream` POST endpoint shall stream tokens in real-time using `TextIteratorStreamer` and Server-Sent Events (SSE).
- **Requirement 5.2.3**: Every token yielded via SSE shall be JSON-encoded (`json.dumps(token)`) to preserve newlines (`\n`), tabs (`\t`), spaces, and indentation.
- **Requirement 5.2.4**: Non-streaming endpoint `/api/chat` and status endpoint `/api/status` shall be exposed for REST integrations.

### 5.3. Frontend UI/UX Design System
- **Requirement 5.3.1**: The web UI shall feature a soft pastel aura gradient background, glassmorphism containers (`backdrop-filter: blur(24px)`), and pill navigation.
- **Requirement 5.3.2**: The chat widget shall mimic a floating Mac window with red, yellow, and green window controls.
- **Requirement 5.3.3**: Generated code snippets shall render in dedicated Markdown code blocks with line-by-line formatting and `highlight.js` syntax highlighting.
- **Requirement 5.3.4**: Interactive suggestion chips (`Print C Hello World`, `Fibonacci Python`, `FastAPI Example`) shall execute automated one-click prompts.

### 5.4. Multi-Device & Global Networking
- **Requirement 5.4.1**: FastAPI server shall bind to `0.0.0.0:8000`, allowing any device on the same Wi-Fi network to connect via `http://<LOCAL_IP>:8000`.
- **Requirement 5.4.2**: The launcher script `start_fastapi.py` shall automatically initialize a `pyngrok` HTTPS tunnel, generating a public URL accessible from any mobile or PC globally.

---

## 6. Non-Functional Requirements (NFRs)

- **Performance & Speed**: First-token latency shall be < 0.1 seconds, with average token generation speeds > 50 tokens/sec on NVIDIA RTX 3050 GPU.
- **Reliability & Port Management**: Server startup scripts shall detect and clean up stale processes bound to port 8000 prior to launching Uvicorn.
- **Code Integrity**: Markdown line breaks (`marked.setOptions({ breaks: true, gfm: true })`) shall ensure zero single-line collapsing errors in generated source code.
- **Security**: Public ngrok tunnels shall enforce HTTPS encryption.

---

## 7. Verification & Testing Plan

1. **Automated GPU Detection Test**: Verify `torch.cuda.is_available() == True` and `torch.cuda.get_device_name(0)` returns `NVIDIA GeForce RTX 3050 6GB Laptop GPU`.
2. **Model Persistence Verification**: Confirm `adapter_config.json` and `adapter_model.safetensors` exist in `d:\daksh\models\coding-assistant\`.
3. **Line-by-Line Formatting Test**: Execute C language prompt `print hello world in c language like syntex form` and verify multi-line code block rendering.
4. **Multi-Device Test**: Open `https://wisdom-embassy-spookily.ngrok-free.dev` on an external mobile device and confirm live token streaming.

---

## 8. Release & Maintenance History

- **v1.0.0**: Initial PyTorch CUDA 12.1 setup and CLI LoRA fine-tuning pipeline.
- **v2.0.0**: Model weights organized under `models/coding-assistant/` and FastAPI backend implemented.
- **v3.0.0**: JSON SSE token streaming added, ChatML prompt template integrated, Pastel Glassmorphism UI deployed, and authenticated ngrok public tunnel enabled.
