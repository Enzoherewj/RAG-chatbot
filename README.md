# Brothers Karamazov RAG-Chatbot

This project implements a Retrieval-Augmented Generation (RAG) chatbot that answers questions about "The Brothers Karamazov" by Fyodor Dostoevsky.

## Features

- Uses RAG to provide accurate answers based on the novel's content
- Web interface for easy interaction
- Evaluation metrics for response quality
- User feedback collection
- Retrieval performance evaluation using multiple methods (MinSearch, Chroma Semantic Search, and Hybrid Search)

## Setup

1. Clone the repository
2. Create and activate a virtual environment:
   ```
   python -m venv rag-chatbot
   source rag-chatbot/bin/activate  # On Windows use `rag-chatbot\Scripts\activate`
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
5. Ensure you have the "Brothers_Karamazov.txt" file in the project root

## Running the Chatbot

1. Activate the virtual environment (if not already activated):
   ```
   source rag-chatbot/bin/activate  # On Windows use `rag-chatbot\Scripts\activate`
   ```
2. Start the Flask app:
   ```
   python app.py
   ```
3. Open a web browser and go to `http://localhost:5001`
4. Start chatting with the AI about "The Brothers Karamazov"!

## Project Structure

- `app.py`: Flask application for the web interface
- `rag_chatbot.py`: Core RAG chatbot logic
- `evaluation.py`: Functions for evaluating response quality
- `templates/index.html`: HTML template for the web interface
- `experiments/test_retreival.py`: Script for evaluating retrieval performance

## Retrieval Performance Evaluation

To evaluate the retrieval performance:

1. Ensure you're in the `experiments` directory
2. Run the evaluation script:
   ```
   python test_retreival.py
   ```
3. The script will evaluate retrieval performance using:
   - MinSearch
   - Chroma Semantic Search
   - Hybrid Search (combining embedding-based and TF-IDF)
4. Results will be saved in `../vectorstore/retrieval_results.json`

## Notes

- The chatbot uses OpenAI's GPT model and embeddings
- Responses are generated based on relevant passages from the novel
- User feedback is collected to help improve the chatbot
- The retrieval evaluation compares different methods to find the most effective approach for this specific use case

Enjoy exploring "The Brothers Karamazov" with your new AI assistant!
