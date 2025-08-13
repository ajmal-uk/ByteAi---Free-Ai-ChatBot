import requests


API_KEY = "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"


URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost",   # optional but recommended
    "X-Title": "DeepSeek Chatbot Demo"    # optional
}

MODEL = "deepseek/deepseek-chat-v3-0324:free"

def deepseek_chat():
    """Interactive DeepSeek chatbot with conversation memory."""
    messages = [
        {"role": "system", "content": "You are a helpful, friendly AI assistant."}
    ]
    
    print("🤖 DeepSeek Chatbot (type 'exit' to quit)")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Bot: Goodbye! 👋")
            break

        messages.append({"role": "user", "content": user_input})

        try:
            response = requests.post(
                URL,
                headers=HEADERS,
                json={
                    "model": MODEL,
                    "messages": messages
                }
            )
            response.raise_for_status()
            data = response.json()
            reply = data["choices"][0]["message"]["content"]
            print(f"Bot: {reply}")
            messages.append({"role": "assistant", "content": reply})

        except requests.exceptions.RequestException as e:
            print(f"❌ API Error: {e}")

# Run the chatbot
if __name__ == "__main__":
    deepseek_chat()
