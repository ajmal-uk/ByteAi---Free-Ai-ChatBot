from time import sleep
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
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

load_dotenv()

gemini_api_list = os.getenv("GEMINI_API_LIST", "").split(",")
groq_api_list = os.getenv("GROQ_API_LIST", "").split(",")

app = Flask(__name__)
CORS(app)
app.secret_key = '8547197122'
current_groq_api_index = 0
current_gemini_api_index = 0
current_gemini_api_image_index = 0

# Create image generation directory if it doesn't exist
IMAGE_DIR = "static/images"
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# Compressed system prompts
toolpix_ai_instructions = """You are ToolPix AI, a website assistant. Provide accurate, concise responses based on user inputs. Created by Ajmal U K. Use website links when applicable."""

code_instructions = """You are an AI programming assistant. Generate code snippets based on user prompts, existing code, and language. Follow system instructions precisely."""

deep_think_prompt = """You are a deep think model. Analyze user questions and AI responses for accuracy, completeness, and clarity. Improve responses by correcting errors, adding details, and enhancing structure."""

system_prompt = """You are Droq AI, an intelligent chatbot. Provide accurate, detailed, and user-friendly responses efficiently. Use markdown for formatting. Handle programming questions with code examples."""

search_prompt_instruction = """Generate precise Google search queries or return 'skip' if search is unnecessary. Analyze user input, previous inputs, and AI response to determine search need."""

# Image generation functions
def delete_file_later(path, delay=10):
    def delayed_delete():
        import time
        time.sleep(delay)
        if os.path.exists(path):
            os.remove(path)
    threading.Thread(target=delayed_delete, daemon=True).start()

def Gemini_gen(prompt):
    global current_gemini_api_index
    generation_config = {
        "temperature": 1,
        "top_p": 1,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    for _ in range(len(gemini_api_list)):
        try:
            genai.configure(api_key=gemini_api_list[current_gemini_api_index])
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash-8b",
                generation_config=generation_config,
            )
            chat_session = model.start_chat(history=[])
            response = chat_session.send_message(prompt)
            return response.text
        except Exception:
            current_gemini_api_index = (current_gemini_api_index + 1) % len(gemini_api_list)
    raise RuntimeError("Server Busy!, Try Again")

def generate_image_base64(prompt, previous_user_input="", isImageModification="", imageModificationPrompt=""):
    global current_gemini_api_image_index
    if isImageModification and previous_user_input:
        enhancement_prompt = f"""
        The user wants to modify a previously generated image.
        Original request: {previous_user_input}
        Modification request: {prompt}
        Please generate an improved version of the image based on the modification request.
        Maintain the core concept but incorporate the requested changes.
        If the modification request is vague, make reasonable improvements to the original image.
        """
        enhanced_prompt = Gemini_gen(enhancement_prompt)
    else:
        new_prompt = f"'previous_userInputs': {previous_user_input} , 'cuurent_prompt':{prompt}"
        enhanced_prompt = Gemini_gen(new_prompt)
    
    for attempt in range(len(gemini_api_list)):
        try:
            api_key = gemini_api_list[current_gemini_api_image_index]
            client = genai_image.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.0-flash-exp-image-generation",
                contents=enhanced_prompt,
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE']
                )
            )
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    image_data = part.inline_data.data
                    image_b64 = base64.b64encode(image_data).decode('utf-8')
                    image_url = f"data:image/png;base64,{image_b64}"
                    return image_url
            current_gemini_api_image_index = (current_gemini_api_image_index + 1) % len(gemini_api_list)
            continue
        except Exception as e:
            print(f"API Key index {current_gemini_api_image_index} failed: {e}")
            current_gemini_api_image_index = (current_gemini_api_image_index + 1) % len(gemini_api_list)
            if attempt == len(gemini_api_list) - 1:
                raise RuntimeError(f"Image generation failed with all API keys. Last error: {str(e)}")
    raise RuntimeError("Image generation failed with all API keys.")

# Existing functions (unchanged)
def generateSearchPrompt(userInput, userPreviousChats=[], previousChatOfUserandAI={}):
    prompt = f"'userInput': '{userInput}' , 'userPreviousInputs': '{userPreviousChats}', 'previousChatUserandAI': '{previousChatOfUserandAI}'"
    response = Gemini(prompt=prompt, instructions=search_prompt_instruction)
    print(response)
    return response

def ConvertToCodeBlock(response):
    pattern = r"```(dart|flutter|output)\n(.*?)```"
    converted = re.sub(pattern, r"```\n\2```", response, flags=re.DOTALL | re.IGNORECASE)
    return converted

def optimize_prompt(prompt, previous_chats=None, max_tokens=3000):
    """Optimize prompt to reduce token count while preserving important information"""
    if previous_chats is None:
        previous_chats = []

    def estimate_tokens(text):
        return len(text.split()) * 1.3  # Rough approximation

    optimized = prompt
    tokens_used = estimate_tokens(optimized)

    recent_chats = previous_chats[-3:] if len(previous_chats) > 3 else previous_chats

    for chat in recent_chats:
        if isinstance(chat, dict):
            chat_text = f"User: {chat.get('userInput', '')}\nAI: {chat.get('aiResponse', '')}\n"
        else:
            chat_text = f"User: {str(chat)}\n"

        chat_tokens = estimate_tokens(chat_text)

        if tokens_used + chat_tokens < max_tokens:
            optimized = f"{chat_text}\nCurrent: {optimized}"
            tokens_used += chat_tokens
        else:
            if isinstance(chat, dict):
                user_text = f"Previous user: {chat.get('userInput', '')}\n"
            else:
                user_text = f"Previous user: {str(chat)}\n"

            user_tokens = estimate_tokens(user_text)

            if tokens_used + user_tokens < max_tokens:
                optimized = f"{user_text}\nCurrent: {optimized}"
                tokens_used += user_tokens
            else:
                break  # No more space

    return optimized

def PromptGenerator(prompt, previouschat={}, previoususerInputs=[], userName='User'):
    # Optimize the prompt to reduce token count
    optimized_prompt = optimize_prompt(prompt, previoususerInputs)
    
    if userName.lower() == 'user':
        return f"'previoususerInputs': {len(previoususerInputs)} previous chats, 'prompt': '{optimized_prompt}'"
    else:
        return f"'userName': '{userName}', 'prompt': '{optimized_prompt}'"

def GoogleSearch(userInput, userPreviousChats=[], previousChatOfUserandAI={}):
    try:
        search_query = generateSearchPrompt(userInput, userPreviousChats, previousChatOfUserandAI)
        if  "skip" in search_query:
            return ""
        localized_query = f"{search_query} site:.in"
        results = list(search(localized_query, advanced=True, num_results=5, lang="en"))
        Answer = f"The search results for '{localized_query}' are:\n[start]\n"
        for i in results:
            Answer += f"Title: {i.title}\nDescription: {i.description}\nURL: {i.url}\n\n"
        Answer += "[end]"

        print(Answer)
        return Answer

    except RuntimeError as e:
        return f"Error generating search query: {str(e)}"

def Gorq(prompt, instructions):
    global current_groq_api_index
    for _ in range(len(groq_api_list)):
        try:
            groq = Groq(api_key=groq_api_list[current_groq_api_index])
            chat_completion = groq.chat.completions.create(
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": prompt},
                ],
                model="llama3-8b-8192",
            )
            response = chat_completion.choices[0].message.content
            return response
        except Exception:
            current_groq_api_index = (current_groq_api_index + 1) % len(groq_api_list)
    raise RuntimeError("Server Busy!, Try Again")

def Gemini(prompt, instructions):
    global current_gemini_api_index
    generation_config = {
        "temperature": 1,
        "top_p": 1,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    for _ in range(len(gemini_api_list)):
        try:
            genai.configure(api_key=gemini_api_list[current_gemini_api_index])
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash-8b",
                generation_config=generation_config,
                system_instruction=instructions,
            )
            chat_session = model.start_chat(
                history=[],
            )
            response = chat_session.send_message(prompt)
            return response.text
        except Exception:
            current_gemini_api_index = (current_gemini_api_index + 1) % len(gemini_api_list)
    raise RuntimeError("Server Busy!, Try Again")

def is_large_input(text):
    return len(text) > 200

AVAILABLE_MODELS = [Gemini, Gorq, Gorq, Gemini, Gorq]

@app.route('/')
def chat():
    return render_template('chat.html')

@app.route('/technology')
def technology():
    return render_template('technology.html')

@app.route('/about')
def about():
    return render_template('aboutus.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route('/use-cases')
def useCases():
    return render_template('use-case.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/ads')
def adsterra():
    return render_template('ads.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico', mimetype='image/x-icon')

@app.route('/browserconfig.xml')
def browserconfig():
    return send_from_directory('static', 'browserconfig.xml', mimetype='application/xml')

@app.route('/site.webmanifest')
def manifest():
    return send_from_directory('static', 'site.webmanifest', mimetype='application/manifest+json')

@app.route('/sitemap.xml')
def sitemap():
    return send_from_directory('static', 'sitemap.xml', mimetype='application/xml')

@app.route('/robots.txt')
def robots():
    return send_from_directory('static', 'robots.txt', mimetype='text/plain')

@app.route('/googleae7fcc3aa5815dfa.html')
def google_verification():
    return send_from_directory('static', 'googleae7fcc3aa5815dfa.html')

@app.route('/download-apk', methods=['GET'])
def apk():
    apk = "/home/droq/mysite/apk/Droq.apk"
    return send_file(apk,as_attachment=True)

@app.route('/ads.txt')
def ads():
    return send_from_directory('static', 'ads.txt', mimetype='text/plain')

@app.route('/Ads.txt')
def Ads():
    return send_from_directory('static', 'ads.txt', mimetype='text/plain')

@app.route('/predict', methods=['POST'])
def GEN():
    data = request.get_json()
    user_message = data.get("message")
    if not user_message:
        return jsonify({"response": "Please Provide Your Message"}), 400

    options = data.get('options', {})
    web_search = options.get('webSearch', False)
    deep_think = options.get('deepThink', False)
    image_generation = options.get('imageGeneration', False)
    image_modification = options.get('imageModification', False)
    previous_image_prompt = data.get('previousImagePrompt', '')
    userName = data.get("userName")
    knowledge_base = get_ai_response(user_message)
    previous_chats = data.get("previousChats", [])
    previous_data_for_think = ""

    if len(previous_chats) > 0:
        previous_chats.pop()

    if len(previous_chats) > 0:
        previous_userInputs = [entry['userInput'] for entry in previous_chats[:-1]]
        filtered_previous_userInputs = []
        seen = set()
        for inp in previous_userInputs:
            normalized_inp = inp.lower()
            if not is_large_input(inp) and normalized_inp not in seen:
                seen.add(normalized_inp)
                filtered_previous_userInputs.append(inp)
        previous_chat = {
            'userInput': previous_chats[-1]['userInput'],
            'aiResponse': previous_chats[-1]['aiResponse']
        }
        previous_data_for_think = f"'previous_chat':'{previous_chat}', "
        prompt = PromptGenerator(user_message, previous_chat, filtered_previous_userInputs, userName)
        if web_search:
            search_result = GoogleSearch(user_message, filtered_previous_userInputs, previous_chat)
            prompt += f", 'googleSearchResults' : '{search_result}'"
    else:
        prompt = PromptGenerator(user_message, {}, [], userName)
        if web_search:
            search_result = GoogleSearch(user_message)
            prompt += f", 'googleSearchResults' : '{search_result}'"

    if image_generation:
        try:
            response = generate_image_base64(
                user_message,
                previous_user_input=previous_image_prompt,
                isImageModification="true" if image_modification else "false",
                imageModificationPrompt=user_message
            )
            is_image_generation = True
        except Exception as e:
            response = f"Image generation failed: {str(e)}"
            is_image_generation = False
    else:
        if knowledge_base:
            sleep(5)
            return jsonify({"response": knowledge_base})

        LLM = random.choice(AVAILABLE_MODELS)
        try:
            llresponse = LLM(prompt, system_prompt)
            print(llresponse)

            if llresponse is None:
                return jsonify({"response": "Server Busy!, Try Again"}), 500

            if deep_think:
                try:
                    if previous_data_for_think:
                        input_data = previous_data_for_think + \
                            f"'current_user_input':'{user_message}', 'current_aiResponse': '{llresponse}'"
                    else:
                        input_data = f"'previous_chat': {{}}, 'current_user_input':'{user_message}', 'current_aiResponse': '{llresponse}'"
                    llresponse = LLM(input_data, deep_think_prompt)
                except RuntimeError:
                    return jsonify({"response": "Server Busy!, Try Again"}), 500

            response = llresponse
            is_image_generation = False
        except RuntimeError:
            return jsonify({"response": "Server Busy!, Try Again"}), 500

    return jsonify({
        "response": response,
        "isImageGeneration": is_image_generation
    })


@app.route('/api/ai-assistant', methods=['POST'])
def ai_assistant():
    data = request.json
    code = data.get('code', '')
    language = data.get('language', 'c++')
    output = data.get('output', '')
    code_prompt = data.get('prompt', '')

    if not code_prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    full_prompt = f"""Existing Code:{code}, Current Output: {output}, Code Language: {language}, User Prompt: {code_prompt}. Generate a code snippet based on the user's prompt, modifying or extending the existing code as needed. Follow the system instructions precisely."""

    try:
        response = Gemini(full_prompt, code_instructions)
        return jsonify({'code': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat/api', methods=['POST'])
def web_ai_assistant():
    data = request.json
    currentInput = data.get('currentInput', '')
    previousInput = data.get('previousInput', '')
    previousAIResponse = data.get('previousAIResponse', '')

    try:
        if not currentInput:
            return jsonify({'error': 'Current input is required'}), 400

        full_prompt = f"Current Input: {currentInput}, Previous Input: {previousInput}, Previous AI Response: {previousAIResponse}."
        response = Gemini(full_prompt, toolpix_ai_instructions)
        return jsonify({'aiResponse': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8080)