#!/usr/bin/env python3
"""
LLM_QA_CLI.py
Python 3.10-compatible CLI to:
 - accept a natural-language question
 - preprocess it (lowercase, tokenize, remove punctuation)
 - construct a prompt and call Hugging Face Inference API
 - display the final answer

Usage:
  # interactive mode
  python LLM_QA_CLI.py

  # one-shot mode
  python LLM_QA_CLI.py --question "What is polymorphism in OOP?"

Environment:
  Set HF_API_TOKEN (your Hugging Face Inference API token).
"""

import os
import re
import json
import argparse
import requests
from typing import Tuple, Dict
from dotenv import load_dotenv
load_dotenv()

HF_API_URL_TEMPLATE = "https://router.huggingface.co/{model_name}"
DEFAULT_MODEL = "gpt2"  # lightweight fallback; replace with a better model if you have quota (e.g. 'bigscience/bloom' or 'OpenAssistant/release')


def preprocess_question(text: str) -> Dict[str, str]:
    """
    Basic preprocessing:
      - strip leading/trailing whitespace
      - lowercase
      - remove punctuation (keeps letters, numbers and whitespace)
      - tokenization by whitespace (returns a token list)
    Returns a dict with 'original', 'processed', 'tokens'
    """
    original = text.strip()
    lowered = original.lower()
    # remove punctuation (keep underscores if you want)
    processed = re.sub(r"[^\w\s]", "", lowered)
    # normalize whitespace
    processed = re.sub(r"\s+", " ", processed).strip()
    tokens = processed.split() if processed else []
    return {"original": original, "processed": processed, "tokens": tokens}


def build_prompt(processed_question: str, instructions: str = "") -> str:
    """
    Build a prompt to send to the LLM. Keep it explicit so the model knows the expected output.
    """
    instruction_block = instructions.strip()
    if instruction_block:
        instruction_block = instruction_block + "\n\n"
    prompt = (
        f"{instruction_block}"
        f"You are an assistant that answers student questions concisely and clearly.\n"
        f"Question (preprocessed): {processed_question}\n"
        f"Provide a short, clear answer and, when relevant, one-line example or suggestion.\n"
        f"Answer:"
    )
    return prompt


def call_hf_inference(prompt: str, model_name: str, hf_token: str, max_new_tokens: int = 200, timeout: int = 60) -> Tuple[bool, str]:
    """
    Call Hugging Face Inference API (text generation). Returns (success, answer_or_error).
    """
    url = HF_API_URL_TEMPLATE.format(model_name=model_name)
    headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_new_tokens, "temperature": 0.2},
        # "options": {"use_cache": False}  # optional
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    except requests.RequestException as e:
        return False, f"Request failed: {e}"

    if resp.status_code != 200:
        # Many HF models return helpful error json
        try:
            err = resp.json()
            return False, f"Hugging Face API error {resp.status_code}: {json.dumps(err)}"
        except Exception:
            return False, f"Hugging Face API returned status {resp.status_code}: {resp.text}"

    # The HF text-generation endpoints often return either:
    #  - [{"generated_text": "..." }] or
    #  - {"error":"..."} for models with special endpoints
    try:
        rjson = resp.json()
        if isinstance(rjson, list) and rjson and "generated_text" in rjson[0]:
            return True, rjson[0]["generated_text"].strip()
        # Some models return string directly or dict with 'generated_text'
        if isinstance(rjson, dict) and "generated_text" in rjson:
            return True, rjson["generated_text"].strip()
        # If it's something else, return the whole JSON
        return True, json.dumps(rjson, indent=2)
    except Exception:
        # Fallback: raw text
        return True, resp.text.strip()


def main():
    parser = argparse.ArgumentParser(description="LLM QA CLI (Hugging Face Inference)")
    parser.add_argument("--question", "-q", type=str, help="Question to ask (if omitted, runs interactive prompt)")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL, help="Hugging Face model id (e.g. 'gpt2' or 'bigscience/bloom')")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max tokens the model should generate")
    args = parser.parse_args()

    hf_token = os.getenv("HF_API_TOKEN")
    if not hf_token:
        print("ERROR: HF_API_TOKEN not found in environment. Set it before running.")
        print("Example (Linux/macOS): export HF_API_TOKEN='your_token_here'")
        return

    if args.question:
        question = args.question
    else:
        try:
            question = input("Enter your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            return

    if not question:
        print("No question provided. Exiting.")
        return

    pre = preprocess_question(question)
    print("\n--- Preprocessing ---")
    print(f"Original: {pre['original']}")
    print(f"Processed: {pre['processed']}")
    print(f"Tokens: {pre['tokens']}\n")

    prompt = build_prompt(pre["processed"])
    print("--- Prompt sent to model ---")
    print(prompt + "\n")

    print("Calling Hugging Face Inference API... (this may take a few seconds)")
    ok, resp_text = call_hf_inference(prompt, args.model, hf_token, max_new_tokens=args.max_tokens)

    print("\n--- LLM Response ---")
    if not ok:
        print("ERROR:", resp_text)
        return

    # Post-processing: trim the response to the first reasonable chunk / paragraph
    # Keep as-is for this assignment; show the full generated text.
    print(resp_text)
    print("\n--- End ---")


if __name__ == "__main__":
    main()
