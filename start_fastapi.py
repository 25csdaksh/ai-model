"""
FastAPI Server & Public Tunnel Launcher for Daksh AI
Multi-Device Global Public Access (Option 2)
Architecture: Any Device Anywhere on Earth (HTTPS) -> ngrok Tunnel -> FastAPI -> GPU Model -> Answer
"""

import os
import sys
import time
import socket
import subprocess
import webbrowser

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

local_ip = get_local_ip()
port = 8000

print("=" * 65)
print("🌐 Launching Multi-Device Global Public FastAPI Server...")
print("Architecture: Any Device Anywhere on Earth -> Public HTTPS Tunnel -> GPU Model")
print("=" * 65)

python_exe = os.path.join(os.path.dirname(__file__), "CodingAssistant", "venv", "Scripts", "python.exe")

# Kill any existing process on port 8000
try:
    subprocess.run(["cmd", "/c", "for /f \"tokens=5\" %a in ('netstat -aon ^| findstr :8000') do taskkill /f /pid %a"], capture_output=True)
except Exception:
    pass

time.sleep(1)

# Launch Uvicorn FastAPI Server binding to 0.0.0.0 for network access
cmd = [python_exe, "-m", "uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", str(port)]
proc = subprocess.Popen(cmd)

time.sleep(4)

# Create Public HTTPS Tunnel using pyngrok
public_url = None
try:
    from pyngrok import ngrok
    public_url = ngrok.connect(port).public_url
    print("\n" + "=" * 65)
    print("🌍 PUBLIC WORLDWIDE ACCESS LINK (Share with any mobile/device):")
    print(f"👉 {public_url}")
    print("=" * 65 + "\n")
except Exception as e:
    print(f"[!] ngrok status notice: {e}")

url = public_url or f"http://127.0.0.1:{port}"
print(f"🌐 Local Access: http://127.0.0.1:{port}")
print(f"📱 Local Wi-Fi Access: http://{local_ip}:{port}")
print(f"🚀 Opening Web Chat UI in Browser...")

webbrowser.open(url)

try:
    proc.wait()
except KeyboardInterrupt:
    print("\n👋 FastAPI server stopped.")
    proc.terminate()
