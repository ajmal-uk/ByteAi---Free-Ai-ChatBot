from time import sleep, time
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import random
from groq import Groq
from knowledge_base import get_ai_response
import re
from flask_cors import CORS
from googlesearch import search
import os
import base64
import threading
from google import genai as genai_image
from google.genai import types
from dotenv import load_dotenv
import difflib
import logging
import html
from typing import List

# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from dotenv import load_dotenv
load_dotenv()

# Ensure static/images exists
STATIC_IMAGE_DIR = os.path.join("static", "images")
os.makedirs(STATIC_IMAGE_DIR, exist_ok=True)

GEMINI_API_LIST = [k.strip() for k in os.getenv("GEMINI_API_LIST", "").split(",") if k.strip()]

if not GEMINI_API_LIST:
    raise ValueError("No Gemini API keys found in GEMINI_API_LIST environment variable.")

# Track current key index
_gemini_key_index = 0

def rotate_gemini_image_key():
    """Rotate through the available Gemini API keys and return the next one."""
    global _gemini_key_index
    if not GEMINI_API_LIST:
        raise ValueError("No Gemini API keys available to rotate.")
    api_key = GEMINI_API_LIST[_gemini_key_index]
    _gemini_key_index = (_gemini_key_index + 1) % len(GEMINI_API_LIST)
    return api_key

def Gemini_generate_image(prompt: str):
    """Generate an image using Gemini API with key rotation, save it to static/images, and return paths."""
    system_prompt = (
        "You are an expert image generation AI. Always produce high-quality, detailed, "
        "and visually coherent images based on the prompt. Keep characters consistent, "
        "use correct proportions, and match the style requested by the user."
    )

    max_attempts = max(1, len(GEMINI_API_LIST))
    last_exc = None

    for _ in range(max_attempts):
        try:
            api_key = rotate_gemini_image_key()
            client = genai_image.Client(api_key=api_key)

            # Combine system prompt and user prompt
            contents = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]

            response = client.models.generate_content(
                model="gemini-2.0-flash-exp-image-generation",
                contents=contents,
                config=types.GenerateContentConfig(response_modalities=['TEXT', 'IMAGE'])
            )

            for cand in response.candidates:
                for part in cand.content.parts:
                    if part.inline_data is not None:
                        image_data = part.inline_data.data
                        
                        # Save as file
                        filename = f"{int(time.time())}.png"
                        file_path = os.path.join(STATIC_IMAGE_DIR, filename)
                        with open(file_path, "wb") as f:
                            f.write(image_data)

                        # Convert to Base64 for inline usage
                        image_b64 = base64.b64encode(image_data).decode('utf-8')
                        
                        return {
                            "file_path": file_path,
                            "base64": f"data:image/png;base64,{image_b64}"
                        }

        except Exception as e:
            logger.exception("Gemini image generation failed on a key, rotating.")
            last_exc = e
            continue

    raise RuntimeError(f"Image generation failed with all Gemini keys. Last error: {last_exc}")


# Example usage
if __name__ == "__main__":
    result = Gemini_generate_image("create an image of cat")
    print("Saved image path:", result["file_path"])
