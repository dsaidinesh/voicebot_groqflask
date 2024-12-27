import os
from flask import Flask, render_template, request, jsonify
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = Flask(__name__)

llm = ChatGroq(
    temperature=0.7,
    groq_api_key=GROQ_API_KEY,
    model_name="mixtral-8x7b-32768"
)

SYSTEM_PROMPT = """You are a basic voice-to-voice chatbot designed to answer simple questions.
Your responses should be short, concise, and to the point.
Aim to provide helpful information without being overly verbose.
If a question is too complex or outside your scope, politely inform the user that you can only handle simple queries."""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    try:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_input)
        ]
        response = llm.invoke(messages)
        
        if hasattr(response, 'content'):
            ai_response = response.content
        elif isinstance(response, str):
            ai_response = response
        else:
            ai_response = str(response)
        
        return jsonify({'response': ai_response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

