from flask import Flask, render_template, request, jsonify
from rag_chatbot import chat, initialize_chatbot, test_embeddings
import os

app = Flask(__name__)

chat_history = []
chatbot_initialized = False

def initialize():
    global chatbot_initialized
    if not chatbot_initialized:
        print("Starting RAG chatbot initialization...")
        initialize_chatbot()
        chatbot_initialized = True
        print("\n" + "="*50)
        print("RAG Chatbot initialization complete!")
        print("Flask app is ready to serve requests.")
        print("You can now access the chatbot at http://localhost:5001")
        print("="*50 + "\n")

@app.route('/')
def home():
    global chatbot_initialized
    if not chatbot_initialized:
        initialize()
    print("Home route accessed")
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def handle_chat():
    query = request.json['query']
    global chat_history
    
    answer, retrieved_chunks, evaluation = chat(query, chat_history)
    
    chat_history.append((query, answer))
    
    return jsonify({
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "evaluation": evaluation
    })

@app.route('/test_embeddings', methods=['POST'])
def handle_test_embeddings():
    query = request.json['query']
    k = request.json.get('k', 5)
    
    test_embeddings(query, k)
    
    return jsonify({"status": "Embeddings test complete. Check console for results."})

@app.route('/feedback', methods=['POST'])
def feedback():
    feedback_data = request.json
    print(f"Feedback received: {feedback_data}")
    return jsonify({"status": "success"})

@app.route('/test')
def test():
    return "Hello, this is a test route!"

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True, port=5001)