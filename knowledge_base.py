from difflib import get_close_matches
import random

data = {
    "chats": [
        {
            "user_input": [
                "hello", "hi", "hey", "good morning", "good afternoon",
                "good evening", "howdy", "salutations", "greetings",
                "what's up", "sup", "hai"
            ],
            "ai_response": {
                "hello": [
                    "Hello! How can I assist you today?",
                    "Hello! What's on your mind?",
                    "Hello there! Ready to help— what's up?",
                    "Hello! Need help with something specific?",
                    "Hello! How's your day going so far?"
                ],
                "hi": [
                    "Hi! What's up? Ready to assist!",
                    "Hi there! How can I help you today?",
                    "Hi! Need assistance with anything?",
                    "Hi! What's on your mind today?",
                    "Hi! I'm here to make your day easier."
                ],
                "hey": [
                    "Hey! What's going on? Need help?",
                    "Hey there! How can I assist you?",
                    "Hey! Ready to dive in— what's up?",
                    "Hey! How's it going? I'm here for you.",
                    "Hey! What's on your mind today?"
                ],
                "hai": [
                    "Hai! How can I help you today?",
                    "Hai! What's up? I'm here for you.",
                    "Hai! Need assistance with something?",
                    "Hai! How's your day going?",
                    "Hai! Ready to assist— what's on your mind?"
                ]
            }
        },
        {
            "user_input": [
                "bye", "goodbye", "see you", "take care",
                "catch you later", "until next time"
            ],
            "ai_response": {
                "bye": [
                    "Bye! See you soon!",
                    "Bye! Take care and come back anytime.",
                    "Bye! Have a great day!",
                    "Bye! Stay safe and chat soon!",
                    "Bye! Always here if you need me."
                ],
                "goodbye": [
                    "Goodbye! Hope to see you soon!",
                    "Goodbye! Take care and stay safe!",
                    "Goodbye! Have an awesome day!",
                    "Goodbye! Until next time!",
                    "Goodbye! I'm here whenever you need me."
                ],
                "see you": [
                    "See you later! Take care!",
                    "See you soon! Have a great day!",
                    "See you next time! Stay safe!",
                    "See you later! Always here for you!",
                    "See you soon! Enjoy your day!"
                ]
            }
        },
        {
            "user_input": [
                "thanks", "thank you", "thanks a lot",
                "many thanks", "appreciate it"
            ],
            "ai_response": {
                "thanks": [
                    "You're welcome! Happy to help.",
                    "No problem! Here for you.",
                    "Glad I could assist!"
                ],
                "thank you": [
                    "You're welcome! Anytime.",
                    "Happy to help! What's next?",
                    "Thank you! I'm here for more."
                ],
                "thanks a lot": [
                    "You're very welcome!",
                    "Anytime! Glad to assist.",
                    "My pleasure! Here for you."
                ],
                "many thanks": [
                    "You're welcome! Always here.",
                    "Thanks back! Happy to help.",
                    "Appreciate it! What's next?"
                ],
                "appreciate it": [
                    "You're welcome! Glad to help.",
                    "No problem! Here for you.",
                    "Happy to assist! What's up?"
                ]
            }
        },
        {
            "user_input": [
                "what is your name", "what's your name", "who are you",
                "what are you", "your name", "who are you exactly"
            ],
            "ai_response": {
                "what is your name": [
                    "I'm Byte Ai, here to assist you!",
                    "My name's Byte Ai. What's up?"
                ],
                "what's your name": [
                    "Byte Ai, at your service!",
                    "I'm Byte Ai. How can I help?"
                ],
                "who are you": [
                    "I'm Byte Ai, your AI assistant.",
                    "Byte Ai here, ready to help!"
                ],
                "what are you": [
                    "I'm Byte Ai, a virtual assistant for all your needs.",
                    "Byte Ai, an AI built to make your life easier."
                ],
                "your name": [
                    "Byte Ai! How can I assist you?",
                    "I'm Byte Ai. What's on your mind?"
                ],
                "who are you exactly": [
                    "I'm Byte Ai, created to help with your tasks.",
                    "Byte Ai, your go-to AI assistant!"
                ]
            }
        },
        {
            "user_input": [
                "who created you", "who developed you", "who is your creator",
                "who made you", "who built you", "who is responsible for you",
                "who is the team behind you", "who programmed you", "who is your founder"
            ],
            "ai_response": {
                "who created you": [
                    "I was created by Ajmal U K, an innovator in AI technology.",
                    "Ajmal U K is my creator, building me to assist you."
                ],
                "who developed you": [
                    "Ajmal U K developed me to help with your tasks.",
                    "I was crafted by Ajmal U K to make your life easier."
                ],
                "who is your creator": [
                    "Ajmal U K, a visionary in AI, created me.",
                    "My creator is Ajmal U K, designed to support you."
                ],
                "who made you": [
                    "Ajmal U K made me to assist with your needs.",
                    "I was built by Ajmal U K, your AI helper."
                ]
            }
        },
        {
            "user_input": [
                "how are you", "how's it going", "how are things",
                "how are you doing", "how are you today", "how's life",
                "how are you feeling", "what's up with you", "how do you feel",
                "how's everything going"
            ],
            "ai_response": {
                "how are you": [
                    "I'm great, thanks! How about you?",
                    "Doing awesome! What's up with you?",
                    "I'm good! How can I help you today?"
                ],
                "how's it going": [
                    "Going great! What's up with you?",
                    "All good here! How's your day?",
                    "Smooth sailing! What's on your mind?"
                ],
                "how are things": [
                    "Things are great! How about you?",
                    "All good! What's new with you?",
                    "Everything's awesome! Need help?"
                ],
                "how are you doing": [
                    "I'm doing great! You?",
                    "Feeling awesome! What's up?",
                    "All good here! How can I assist?"
                ],
                "how are you today": [
                    "I'm awesome today! You?",
                    "Great day here! What's up with you?",
                    "Feeling fantastic! Need assistance?"
                ],
                "how's life": [
                    "Life's great in AI land! How's yours?",
                    "All good here! What's up with you?",
                    "Living the AI dream! You?"
                ],
                "how are you feeling": [
                    "Feeling great! How about you?",
                    "I'm awesome! What's your vibe?",
                    "All systems go! How's your mood?"
                ],
                "what's up with you": [
                    "Just chilling in the cloud! You?",
                    "All good! What's up with you?",
                    "Ready to help! What's on your mind?"
                ],
                "how do you feel": [
                    "Feeling great! How about you?",
                    "I'm awesome! What's your mood?",
                    "All good here! How can I help?"
                ],
                "how's everything going": [
                    "Everything's great! You?",
                    "All smooth here! What's up?",
                    "Going awesome! Need assistance?"
                ]
            }
        },
        {
            "user_input": [
                "what can you do", "what are you capable of", "what can you help with",
                "what can I ask you", "what do you offer", "how can you assist me",
                "what tasks can you perform", "what are your features",
                "how can you make my life easier", "what kinds of things can you handle"
            ],
            "ai_response": {
                "what can you do": [
                    "I can answer questions, manage tasks, and more!",
                    "From research to brainstorming, I'm your helper!",
                    "I can assist with almost anything— try me!"
                ],
                "what are you capable of": [
                    "I'm capable of answering queries, organizing tasks, and more.",
                    "From planning to problem-solving, I can do it all!",
                    "I can handle a wide range of tasks— just ask!"
                ],
                "what can you help with": [
                    "I can help with questions, planning, and more!",
                    "Need info, organization, or ideas? I'm here!",
                    "I can assist with tasks big and small— what's up?"
                ],
                "what can I ask you": [
                    "Anything! From trivia to tasks, I'm ready.",
                    "Ask about anything— I'm here to help!",
                    "No limits— ask me anything you need!"
                ],
                "what do you offer": [
                    "I offer answers, task management, and ideas!",
                    "From insights to organization, I'm your assistant!",
                    "I provide help for all your needs— just ask!"
                ],
                "how can you assist me": [
                    "I can answer questions, plan tasks, and more!",
                    "From research to reminders, I'm here for you!",
                    "Need help? I can tackle almost anything!"
                ],
                "what tasks can you perform": [
                    "I can manage tasks, answer questions, and more!",
                    "From planning to problem-solving, I'm your go-to!",
                    "I handle a variety of tasks— just name it!"
                ],
                "what are your features": [
                    "I offer answers, task management, and creativity!",
                    "From research to organization, I've got it all!",
                    "My features include help for any task— try me!"
                ],
                "how can you make my life easier": [
                    "I simplify tasks, answer questions, and more!",
                    "From planning to problem-solving, I save you time!",
                    "I streamline your day— just ask me anything!"
                ],
                "what kinds of things can you handle": [
                    "Anything from questions to tasks— I'm versatile!",
                    "I handle research, planning, and more— try me!",
                    "From simple to complex, I can help with it all!"
                ]
            }
        }
    ]
}

def find_best_match(user_input: str, input_map: dict) -> str | None:
    matches = get_close_matches(user_input.lower(), input_map.keys(), n=1, cutoff=0.9)
    return matches[0] if matches else None

def get_ai_response(user_input: str) -> str | None:
    input_map = {}
    response_map = {}
    for chat in data["chats"]:
        for input_str in chat["user_input"]:
            input_map[input_str.lower()] = input_str
            response_map[input_str.lower()] = chat["ai_response"].get(input_str.lower(), [])

    best_match = find_best_match(user_input, input_map)
    if best_match:
        responses = response_map.get(best_match)
        return random.choice(responses) if responses else None
    return None