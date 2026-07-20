"""
Daksh AI Chatbot - Python Terminal Edition
તમારું પોતાનું AI Chatbot (Python માં)
"""

import sys
import json
import urllib.request
import urllib.error
import os

class AIChatbot:
    def __init__(self, api_key=None, system_prompt=None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.system_prompt = system_prompt or "તમે એક અત્યંત બુદ્ધિશાળી અને મદદગાર ગુજરાતી AI આસિસ્ટન્ટ છો."
        self.history = []

    def set_api_key(self, key):
        self.api_key = key.strip()

    def generate_response(self, user_text):
        # Add user message to history
        self.history.append({"role": "user", "parts": [{"text": user_text}]})

        if self.api_key:
            try:
                response_text = self._call_gemini_api()
                self.history.append({"role": "model", "parts": [{"text": response_text}]})
                return response_text
            except Exception as e:
                return f"⚠️ API Error: {str(e)}"
        else:
            # Smart Built-in Fallback for testing without API Key
            response_text = self._smart_fallback(user_text)
            self.history.append({"role": "model", "parts": [{"text": response_text}]})
            return response_text

    def _call_gemini_api(self):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"
        
        payload = {
            "contents": self.history,
            "systemInstruction": {
                "parts": [{"text": self.system_prompt}]
            },
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 2048
            }
        }

        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result['candidates'][0]['content']['parts'][0]['text']

    def _smart_fallback(self, text):
        text_lower = text.lower()
        if "કેમ છો" in text_lower or "hello" in text_lower or "hi" in text_lower:
            return "નમસ્તે! હું તમારો Python AI Chatbot છું. આજે હું તમને કેવી રીતે મદદ કરી શકું?"
        elif "python" in text_lower:
            return "Python એ એક ખૂબ જ શક્તિશાળી અને શીખવામાં સરળ પ્રોગ્રામિંગ ભાષા છે. AI અને Machine Learning માટે Python શ્રેષ્ઠ છે!"
        elif "api key" in text_lower or "key" in text_lower:
            return "લાઈવ AI રિસ્પોન્સ માટે તમારી Gemini API Key સેટ કરો. (Google AI Studio માંથી ફ્રીમાં મળી શકે છે)"
        else:
            return f"તમારો પ્રશ્ન: '{text}' મળ્યો છે. (💡 નોંધ: લાઈવ Gemini AI માટે તમારી API Key દાખલ કરો)"

def main():
    print("=" * 60)
    print("🤖 Python AI Chatbot (Daksh AI)")
    print("=" * 60)
    
    api_key = input("🔑 તમારી Gemini API Key લખો (ઓપ્શનલ - ખાલી રાખીને Enter દબાવી શકો છો): ").strip()
    chatbot = AIChatbot(api_key=api_key)
    
    print("\n✨ Chatbot તૈયાર છે! બહાર નીકળવા માટે 'exit' અથવા 'quit' લખો.\n")

    while True:
        try:
            user_input = input("👤 તમે: ").strip()
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'બહાર']:
                print("👋 આવજો! આભાર.")
                break

            print("🤖 AI વિચાર કરી રહ્યું છે...", end="\r")
            bot_reply = chatbot.generate_response(user_input)
            print(" " * 40, end="\r") # clear line
            print(f"🤖 AI: {bot_reply}\n")

        except KeyboardInterrupt:
            print("\n👋 ચેટ સમાપ્ત કરી.")
            break

if __name__ == "__main__":
    main()
