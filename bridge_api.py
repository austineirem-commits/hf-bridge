"""
Bridge API - Deploy this on Render.com (FREE)
This receives requests from PythonAnywhere and forwards them to Hugging Face
"""
from flask import Flask, request, jsonify
from openai import OpenAI
import os

app = Flask(__name__)

# Hugging Face client
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ.get("HF_TOKEN")
)

@app.route("/chat", methods=["POST"])
def chat():
    """Forward chat requests to Hugging Face API"""
    try:
        data = request.json
        user_message = data.get("message", "")
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        # Call Hugging Face API
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3.2:novita",
            messages=[{"role": "user", "content": user_message}],
            max_tokens=500,
            stream=False
        )
        
        reply = response.choices[0].message.content
        
        return jsonify({
            "reply": reply,
            "success": True
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
