"""
FastAPI Server Launcher for Daksh AI
User -> FastAPI -> Model -> Answer Architecture (Live Token Streaming)
"""

import os
import sys
import time
import subprocess
import webbrowser

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

print("=" * 60)
print("⚡ Launching Ultra-Fast Live Streaming FastAPI GPU Server...")
print("Architecture: User -> FastAPI -> Fine-Tuned GPU Model -> Answer")
print("=" * 60)

python_exe = os.path.join(os.path.dirname(__file__), "CodingAssistant", "venv", "Scripts", "python.exe")

# Kill any existing process on port 8000 on Windows
try:
    subprocess.run(["cmd", "/c", "for /f \"tokens=5\" %a in ('netstat -aon ^| findstr :8000') do taskkill /f /pid %a"], capture_output=True)
except Exception:
    pass

time.sleep(1)

# Launch Uvicorn FastAPI Server on port 8000
cmd = [python_exe, "-m", "uvicorn", "fastapi_app:app", "--host", "127.0.0.1", "--port", "8000"]

proc = subprocess.Popen(cmd)

# Wait 4 seconds for GPU model to load into VRAM
time.sleep(4)

url = "http://127.0.0.1:8000"
print(f"\n🌐 Opening Ultra-Fast Live Streaming Chat UI at {url} in your Web Browser...")
webbrowser.open(url)

try:
    proc.wait()
except KeyboardInterrupt:
    print("\n👋 FastAPI server stopped.")
    proc.terminate()
