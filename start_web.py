"""
One-Click Web App Launcher for Daksh AI
ઓટોમેટિક બ્રાઉઝર ઓપન કરનાર સ્ક્રિપ્ટ
"""

import os
import sys
import subprocess
import time
import webbrowser

# Force UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

print("=" * 60)
print("🚀 Starting Daksh AI Web Application & Opening Browser...")
print("=" * 60)

app_script = os.path.join(os.path.dirname(__file__), "CodingAssistant", "app.py")
python_exe = os.path.join(os.path.dirname(__file__), "CodingAssistant", "venv", "Scripts", "python.exe")

proc = subprocess.Popen([python_exe, app_script])

# Wait 4 seconds for GPU model to load and server to bind
time.sleep(4)

web_url = "http://127.0.0.1:7860"
print(f"🌐 Opening {web_url} in your Web Browser...")
webbrowser.open(web_url)

try:
    proc.wait()
except KeyboardInterrupt:
    print("\n👋 Server stopped.")
    proc.terminate()
