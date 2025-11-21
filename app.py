from flask import Flask, render_template, request, jsonify
import os
import re
import requests
import json
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

HF_API_URL_TEMPLATE = "https://router.huggingface.co/{model_name}"
DEFAULT_MODEL = os.getenv("HF_MODEL", "gpt2")


def preprocess_question(text: str) -> dict:
    original = text.strip()
    lowered = original.lower()
    processed = re.sub(r"[^\w\s]", "", lowered)
    processed = re.sub(r"\s+", " ", processed).strip()
    tokens = processed.split() if processed else []
    return {"original": original, "processed": processed, "tokens": tokens}


def build_prompt(processed_question: str) -> str:
    return (
        f"You are an assistant that answers student questions concisely and clearly.\n"
        f"Question (preprocessed): {processed_question}\n"
        f"Provide a short, clear answer and, when relevant, one-line example or suggestion.\n"
        f"Answer:"
    )


def call_hf_inference(prompt: str, model_name: str, hf_token: str, max_new_tokens: int = 200, timeout: int = 60):
    url = HF_API_URL_TEMPLATE.format(model_name=model_name)
    headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_new_tokens, "temperature": 0.2}}

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    rjson = resp.json()
    if isinstance(rjson, list) and rjson and "generated_text" in rjson[0]:
        return rjson[0]["generated_text"].strip()
    if isinstance(rjson, dict) and "generated_text" in rjson:
        return rjson["generated_text"].strip()
    # otherwise, just return stringified JSON
    return json.dumps(rjson, indent=2)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.json or {}
    question = data.get("question", "")
    model = data.get("model", DEFAULT_MODEL)
    max_tokens = int(data.get("max_tokens", 200))

    if not question:
        return jsonify({"error": "No question provided"}), 400

    pre = preprocess_question(question)

    hf_token = os.getenv("HF_API_TOKEN")
    if not hf_token:
        return jsonify({"error": "HF_API_TOKEN not configured on server"}), 500

    prompt = build_prompt(pre["processed"])
    try:
        answer = call_hf_inference(prompt, model, hf_token, max_new_tokens=max_tokens)
    except requests.HTTPError as e:
        return jsonify({"error": f"Hugging Face API HTTP error: {e}", "status_code": e.response.status_code, "response_text": e.response.text}), 502
    except Exception as e:
        return jsonify({"error": f"Unexpected error calling HF API: {e}"}), 500

    # Return processed question, raw LLM response and final answer
    return jsonify({
        "original_question": pre["original"],
        "processed_question": pre["processed"],
        "tokens": pre["tokens"],
        "prompt": prompt,
        "answer": answer
    })


if __name__ == "__main__":
    # Default host/port suitable for local development
    app.run(host="0.0.0.0", port=5000, debug=True)
