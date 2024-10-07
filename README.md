# Brothers Karamazov RAG-Chatbot

This project implements a Retrieval-Augmented Generation (RAG) chatbot that answers questions about "The Brothers Karamazov" by Fyodor Dostoevsky.

## Features

- Uses RAG to provide accurate answers based on the novel's content
- Web interface for easy interaction
- Evaluation metrics for response quality
- User feedback collection
- Retrieval performance evaluation using multiple methods (MinSearch, Chroma Semantic Search, and Hybrid Search)
- Monitoring dashboard with interactive visualizations

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

## Data Ingestion and Embeddings

The project comes with pre-generated embeddings for your convenience. However, if you wish to regenerate the embeddings or ingest new data:

1. Run the ingestion script:
   ```
   python ingest.py
   ```
   This will create or update the `vectorstore` directory with the processed data.

Note: Using the pre-generated embeddings will significantly reduce the initial setup time.

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
- `ingest.py`: Script for ingesting the document and creating the vectorstore (optional)
- `evaluation.py`: Functions for evaluating response quality
- `templates/index.html`: HTML template for the web interface
- `experiments/`: Directory containing retrieval performance evaluation scripts and data
  - `test_retreival.py`: Script for evaluating retrieval performance
  - `generated_questions.csv`: Pre-generated test questions
  - `retrieval_results.json`: Results of retrieval performance evaluation
- `dashboard.py`: Streamlit dashboard for monitoring chatbot performance
- `vectorstore/`: Directory containing pre-generated embeddings and Chroma database

## Retrieval Performance Evaluation

To evaluate the retrieval performance:

1. Navigate to the `experiments` directory:
   ```
   cd experiments
   ```
2. Run the evaluation script:
   ```
   python test_retreival.py
   ```
3. The script will evaluate retrieval performance using:
   - MinSearch
   - Chroma Semantic Search
   - Hybrid Search (combining embedding-based and TF-IDF)
4. Results will be saved in `retrieval_results.json` within the `experiments` directory.

### How the Retrieval Evaluation Works

The `test_retreival.py` script follows this workflow:

1. **Data Loading**: It loads the preprocessed chunks and their embeddings from the Chroma vectorstore.

2. **Question Loading**: It loads pre-generated test questions from `generated_questions.csv`. If this file doesn't exist, it will generate new questions based on the content of each chunk.

3. **Sampling**: A subset of questions is sampled to reduce evaluation time while maintaining statistical significance.

4. **Index Creation**: It creates indexes for each retrieval method (MinSearch, Chroma, and Hybrid).

5. **Evaluation**: For each retrieval method:
   - It runs each test question through the retrieval function.
   - It checks if the expected chunk (the one the question was generated from) is in the top retrieved results.
   - It calculates two metrics:
     - Hit Rate: The proportion of questions where the correct chunk was retrieved.
     - Mean Reciprocal Rank (MRR): A measure of how high the correct chunk appears in the results, on average.

6. **Results Compilation**: The script compiles the results for each method and saves them to `retrieval_results.json`.

We evaluated the retrieval performance using three different methods:

1. MinSearch
2. Chroma Semantic Search
3. Hybrid Search (combining embedding-based and TF-IDF)

The results of our evaluation are as follows:

| Method                  | Hit Rate | Mean Reciprocal Rank (MRR) |
|-------------------------|----------|-----------------------------|
| MinSearch               | 0.62     | 0.3715                      |
| Chroma Semantic Search  | 0.64     | 0.4839                      |
| Hybrid Search           | 0.61     | 0.4420                      |

Based on these results, we have chosen **Chroma Semantic Search** as our primary retrieval method for the chatbot. It demonstrated the highest hit rate (0.64) and mean reciprocal rank (0.4839), indicating better overall performance in retrieving relevant passages from "The Brothers Karamazov."

## Monitoring Dashboard

The project includes a Streamlit dashboard for monitoring the chatbot's performance and user interactions.

To view the monitoring dashboard:

1. Ensure you have collected some interaction and feedback data by using the chatbot.
2. Run the Streamlit dashboard:
   ```
   streamlit run dashboard.py
   ```
3. Open your web browser and go to the URL provided by Streamlit (usually `http://localhost:8501`).

The dashboard includes the following interactive visualizations:

1. **Interactions Over Time**: A line chart showing the cumulative number of interactions with the chatbot over time.
2. **User Feedback Distribution**: A pie chart displaying the distribution of helpful vs. not helpful feedback from users.
3. **Average Evaluation Scores Over Time**: A line chart showing how the chatbot's evaluation scores have changed over time.
4. **Query Length Distribution**: A histogram showing the distribution of query lengths, helping to understand the complexity of user questions.
5. **Top 10 Most Common Words in Queries**: A bar chart displaying the most frequently used words in user queries, providing insights into common topics or themes.

These visualizations provide valuable insights into the chatbot's performance, user behavior, and areas for potential improvement.

## Notes

- The chatbot uses OpenAI's GPT model and embeddings
- Responses are generated based on relevant passages from the novel
- User feedback is collected to help improve the chatbot
- The retrieval evaluation compares different methods to find the most effective approach for this specific use case
- The monitoring dashboard helps in tracking the chatbot's performance and user engagement over time

Enjoy exploring "The Brothers Karamazov" with your new AI assistant!

## Docker Deployment

To run the entire application (including the Flask app and Streamlit dashboard) using Docker Compose:

1. Ensure you have Docker and Docker Compose installed on your system.

2. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

3. Build and start the containers:
   ```
   docker-compose up --build
   ```

4. Access the applications:
   - Flask app: `http://localhost:5001`
   - Streamlit dashboard: `http://localhost:8501`

5. To stop the containers, use:
   ```
   docker-compose down
   ```

Note: The first time you run the containers, it may take some time as the system generates embeddings for the novel. Subsequent runs will be faster as the vectorstore is persisted.

## Project Structure

- `app.py`: Flask application for the web interface
- `dashboard.py`: Streamlit dashboard for monitoring chatbot performance
- `rag_chatbot.py`: Core RAG chatbot logic
- `ingest.py`: Script for ingesting the document and creating the vectorstore
- `evaluation.py`: Functions for evaluating response quality
- `init_setup.py`: Script to initialize the vectorstore (used in Docker setup)
- `Dockerfile`: Defines the Docker image for both Flask app and Streamlit dashboard
- `docker-compose.yml`: Defines the multi-container Docker application
- `requirements.txt`: List of Python dependencies
- `.dockerignore`: Specifies files and directories to be excluded from the Docker build context

## Notes

- The chatbot uses OpenAI's GPT model and embeddings
- Responses are generated based on relevant passages from the novel
- User feedback is collected to help improve the chatbot
- The retrieval evaluation compares different methods to find the most effective approach for this specific use case
- The monitoring dashboard helps in tracking the chatbot's performance and user engagement over time

Enjoy exploring "The Brothers Karamazov" with your new AI assistant!
