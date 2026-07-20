"""
Python Web Server Backend for Daksh AI Chatbot
આ Python સ્ક્રિપ્ટ સ્થાનિક Web Server ચલાવે છે.
"""

import http.server
import socketserver
import json
import urllib.request
import urllib.error
import os

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

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
                system_prompt = request_json.get('systemPrompt', 'તમે એક મદદગાર ગુજરાતી AI આસિસ્ટન્ટ છો.')

                if api_key:
                    bot_response = self.call_gemini(user_message, api_key, system_prompt)
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
        return f"🤖 (Python Server): તમારો પ્રશ્ન '{text}' મળ્યો છે. Python બૅકએન્ડ સફળતાપૂર્વક કામ કરી રહ્યું છે!"

def run_server():
    with socketserver.TCPServer(("", PORT), AIHandler) as httpd:
        print(f"🚀 Python Web Server ચાલુ થઈ ગયું છે: http://localhost:{PORT}")
        print("બ્રાઉઝરમાં ખોલીને તમારું AI Chatbot ચલાવો. બંધ કરવા Ctrl+C દબાવો.")
        httpd.serve_forever()

if __name__ == "__main__":
    run_server()
