import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from minsearch import Index
import json
import os
import sys
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from multiprocessing import Pool, cpu_count
from functools import partial
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')

# Initialize stemmer
stemmer = PorterStemmer()

# At the top of the file, configure logging for debug level
logging.basicConfig(
    filename='test_retreival_debug.log',
    filemode='w',  # Overwrite the log file each run
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)

# Initialize OpenAI client
client = OpenAI()

def load_data(vectorstore_path):
    """
    Load chunks and their embeddings from the Chroma vectorstore.

    Args:
        vectorstore_path (str): Path to the Chroma vectorstore.

    Returns:
        tuple: 
            - df_chunks (pd.DataFrame): DataFrame containing chunk_id, text, and vector.
            - vectorstore (Chroma): Chroma vectorstore instance.
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings)

    chunks = vectorstore.get()
    df_chunks = pd.DataFrame({
        'chunk_id': chunks['ids'],
        'text': chunks['documents'],
        'vector': chunks['embeddings']
    })

    # **Add Debugging Statements**
    logging.debug(f"chunk_id types: {df_chunks['chunk_id'].dtype}")
    logging.debug("Sample chunk_ids:\n" + df_chunks['chunk_id'].head().to_string())

    return df_chunks, vectorstore

def generate_questions_for_chunk(chunk, num_questions_per_chunk=5, max_retries=3):
    """Generate questions for a single chunk using OpenAI API with retries."""
    prompt_template = """
    You are a user of our novel assistant application for "The Brothers Karamazov".
    Formulate {num_questions} questions based on the following excerpt from the novel.
    Make the questions specific to this excerpt.
    The excerpt should contain the answer to the questions, and the questions should
    be complete and not too short. Use as few words as possible from the excerpt.

    Excerpt:
    {text}

    Provide the output in JSON format without using code blocks:
    {{"questions": ["question1", "question2", ..., "question{num_questions}"]}}
    """
    
    prompt = prompt_template.format(num_questions=num_questions_per_chunk, text=chunk['text'])
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )

            logging.info(f"Chunk ID {chunk['chunk_id']} - API Response: {repr(response.choices[0].message.content)}")

            response_content = response.choices[0].message.content.replace('\x00', '').strip()
            
            # Replace curly quotes with standard quotes
            response_content = response_content.replace('“', '"').replace('”', '"')
            
            # Remove Markdown code block delimiters if present
            if response_content.startswith('```'):
                lines = response_content.split('\n')
                # Assuming the code block starts with ```json and ends with ```
                if len(lines) >= 3 and lines[0].startswith('```') and lines[-1].startswith('```'):
                    response_content = '\n'.join(lines[1:-1])

            questions_json = json.loads(response_content)

            questions = []
            for question in questions_json['questions']:
                questions.append({
                    'chunk_id': chunk['chunk_id'],
                    'question': question
                })
            
            logging.info(f"Chunk ID {chunk['chunk_id']} - Successfully generated questions.")
            return questions

        except json.decoder.JSONDecodeError as e:
            logging.error(f"Attempt {attempt + 1}: Error parsing JSON for chunk {chunk['chunk_id']}: {str(e)}")
            logging.error(f"Response Content: {response.choices[0].message.content}")
            if attempt < max_retries - 1:
                logging.info("Retrying...")
                continue
            else:
                logging.error(f"Failed to parse JSON after {max_retries} attempts for chunk {chunk['chunk_id']}. Skipping.")
                return []
        except Exception as e:
            logging.error(f"Attempt {attempt + 1}: Unexpected error for chunk {chunk['chunk_id']}: {str(e)}")
            if attempt < max_retries - 1:
                logging.info("Retrying...")
                continue
            else:
                logging.error(f"Failed due to unexpected error after {max_retries} attempts for chunk {chunk['chunk_id']}. Skipping.")
                return []

def generate_questions(df_chunks, questions_path, num_questions_per_chunk=5):
    """Generate questions for each chunk in parallel, resuming from where it left off."""
    processed_chunk_ids = set()
    questions = []
    
    # Load existing questions if the CSV exists
    if os.path.exists(questions_path):
        existing_questions = pd.read_csv(questions_path)
        processed_chunk_ids = set(existing_questions['chunk_id'].tolist())
        questions = existing_questions.to_dict(orient='records')
        print(f"Resuming from {len(processed_chunk_ids)} already processed chunks.")
    else:
        # Initialize the CSV with headers if it doesn't exist
        pd.DataFrame(columns=['chunk_id', 'question']).to_csv(questions_path, index=False)
    
    # Filter out already processed chunks
    df_chunks = df_chunks[~df_chunks['chunk_id'].isin(processed_chunk_ids)]
    
    if df_chunks.empty:
        print("All chunks have already been processed.")
        return pd.DataFrame(questions)
    
    # **Add Validation Step**
    print("Validating chunk_id integrity in questions...")
    invalid_chunk_ids = set(df_questions['chunk_id']) - set(df_chunks['chunk_id'])
    if invalid_chunk_ids:
        logging.warning(f"Found {len(invalid_chunk_ids)} invalid chunk_ids in questions.")
    
    # Define the number of worker processes (use all available CPUs)
    num_processes = cpu_count()
    
    with Pool(processes=num_processes) as pool:
        # Create a partial function with fixed num_questions_per_chunk
        worker = partial(generate_questions_for_chunk, num_questions_per_chunk=num_questions_per_chunk)
        
        # Use imap_unordered for efficient parallel processing
        for chunk_questions in tqdm(pool.imap_unordered(worker, df_chunks.to_dict(orient='records')), 
                                    total=len(df_chunks), desc="Generating questions"):
            if chunk_questions:
                questions.extend(chunk_questions)
                # Append new questions to the CSV incrementally
                pd.DataFrame(chunk_questions).to_csv(questions_path, mode='a', header=False, index=False)
    
    return pd.DataFrame(questions)

def create_minsearch_index(documents):
    """Create and fit a MinSearch index with given documents."""
    index = Index(
        text_fields=['text'],
        keyword_fields=['chunk_id']
    )
    index.fit(documents)
    return index

def create_chroma_index(documents):
    """
    Create a Chroma vectorstore index from the given documents.

    Args:
        documents (list of dict): List of documents, each containing 'chunk_id' and 'text'.

    Returns:
        Chroma: Chroma vectorstore instance.
    """
    texts = [doc['text'] for doc in documents]
    metadatas = [{'chunk_id': doc['chunk_id']} for doc in documents]

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )
    return vectorstore

def minsearch_retrieval(index, query, num_results=10):
    """Retrieve documents using MinSearch index."""
    results = index.search(
        query=query,
        filter_dict={},
        boost_dict={},
        num_results=num_results
    )
    return results

def chroma_retrieval(vectorstore, query, num_results=10):
    """
    Retrieve documents using Chroma vectorstore's similarity search.

    Args:
        vectorstore (Chroma): Chroma vectorstore instance.
        query (str): The query string to search for.
        num_results (int): Number of top results to retrieve.

    Returns:
        list of dict: List containing retrieved documents with 'chunk_id' and 'text'.
    """
    results = vectorstore.similarity_search_with_relevance_scores(query, k=num_results)

    retrieved_documents = []
    for doc, score in results:
        retrieved_document = {
            'chunk_id': doc.metadata['chunk_id'],
            'text': doc.page_content,
            'score': score
        }
        retrieved_documents.append(retrieved_document)
    
    # Debugging output
    retrieved_ids = [doc['chunk_id'] for doc in retrieved_documents]
    logging.debug(f"Retrieved chunk_ids for query '{query}': {retrieved_ids}")

    return retrieved_documents

def hit_rate(relevance_total):
    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)

def mrr(relevance_total):
    total_score = 0.0

    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score = total_score + 1 / (rank + 1)

    return total_score / len(relevance_total)

def evaluate_retrieval(df_questions, retrieval_func, method_name="Generic Retrieval"):
    """
    Evaluate retrieval performance using hit rate and MRR.

    Args:
        df_questions (pd.DataFrame): DataFrame containing questions and their corresponding chunk_ids.
        retrieval_func (callable): The retrieval function to use.
        method_name (str): Name of the retrieval method for logging purposes.

    Returns:
        dict: Dictionary containing 'method', 'hit_rate', and 'mrr'.
    """
    relevance_total = []

    for idx, row in tqdm(df_questions.iterrows(), total=len(df_questions), desc=f"Evaluating questions using {method_name}"):
        expected_chunk_id = row['chunk_id']
        question = row['question']

        results = retrieval_func(question)
        retrieved_chunk_ids = [str(doc['chunk_id']) for doc in results]

        # **Add Debugging Statements**
        if not retrieved_chunk_ids:
            logging.debug(f"No documents retrieved for question ID {idx}: {question}")
        
        relevance = [doc_id == str(expected_chunk_id) for doc_id in retrieved_chunk_ids]
        relevance_total.append(relevance)
    
    hr = hit_rate(relevance_total)
    mrr_score = mrr(relevance_total)

    return {
        'method': method_name,
        'hit_rate': hr,
        'mrr': mrr_score
    }

def preprocess_text(text):
    # Tokenize and stem the text
    tokens = word_tokenize(text.lower())
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

def hybrid_retrieval(vectorstore, tfidf_vectorizer, tfidf_matrix, query, num_results=10, alpha=0.5):
    # Preprocess the query
    processed_query = preprocess_text(query)
    
    # Get embedding-based results
    embedding_results = chroma_retrieval(vectorstore, processed_query, num_results)
    
    # Get TF-IDF based results
    query_vector = tfidf_vectorizer.transform([processed_query])
    tfidf_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    tfidf_top_indices = tfidf_similarities.argsort()[-num_results:][::-1]
    
    # Combine results
    combined_results = []
    for i, (emb_result, tfidf_index) in enumerate(zip(embedding_results, tfidf_top_indices)):
        combined_score = alpha * emb_result['score'] + (1 - alpha) * tfidf_similarities[tfidf_index]
        combined_results.append({
            'chunk_id': emb_result['chunk_id'],
            'text': emb_result['text'],
            'score': combined_score
        })
    
    # Sort by combined score
    combined_results.sort(key=lambda x: x['score'], reverse=True)
    
    return combined_results[:num_results]

def main():
    vectorstore_path = "../vectorstore"
    output_path = os.path.join('../vectorstore', 'retrieval_results.json')
    questions_path = os.path.join('../vectorstore', 'generated_questions.csv')

    print("Loading data from Chroma vectorstore...")
    df_chunks, _ = load_data(vectorstore_path)
    print(f"Loaded {len(df_chunks)} chunks.")

    print("Recreating Chroma vectorstore with chunk_id metadata...")
    documents = df_chunks.to_dict(orient='records')
    vectorstore = create_chroma_index(documents)
    print("Chroma vectorstore recreated.")

    if os.path.exists(questions_path):
        print("Loading existing questions...")
        df_questions = pd.read_csv(questions_path)
    else:
        print("Generating questions...")
        df_questions = generate_questions(df_chunks, questions_path)  # Pass questions_path here
        df_questions.to_csv(questions_path, index=False)
    print(f"Total questions: {len(df_questions)}")

    # **Add Sampling Step**
    sample_size = 100  # You can adjust the sample size as needed
    df_sampled = df_questions.sample(n=sample_size, random_state=42)
    logging.debug(f"Sampled {len(df_sampled)} questions for testing.")

    print("Creating MinSearch index...")
    documents = df_chunks.to_dict(orient='records')
    minsearch_index = create_minsearch_index(documents)
    print("MinSearch index created and fitted.")

    print("Evaluating retrieval performance using MinSearch...")
    # Define a wrapper for minsearch_retrieval to match the expected signature
    def minsearch_retrieval_wrapper(query, num_results=10):
        return minsearch_retrieval(minsearch_index, query, num_results)
    
    results_minsearch = evaluate_retrieval(df_sampled, minsearch_retrieval_wrapper, method_name="MinSearch")
    print("MinSearch Evaluation Results:")
    print(results_minsearch)

    print("Evaluating retrieval performance using Chroma Semantic Search...")
    # Define a wrapper for chroma_retrieval to match the expected signature
    def chroma_retrieval_wrapper(query, num_results=10):
        return chroma_retrieval(vectorstore, query, num_results)
    
    results_chroma = evaluate_retrieval(df_sampled, chroma_retrieval_wrapper, method_name="Chroma Semantic Search")
    print("Chroma Semantic Search Evaluation Results:")
    print(results_chroma)

    # Create TF-IDF vectorizer and matrix
    tfidf_vectorizer = TfidfVectorizer(preprocessor=preprocess_text)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_chunks['text'])

    print("Evaluating retrieval performance using Hybrid Search...")
    def hybrid_retrieval_wrapper(query, num_results=10):
        return hybrid_retrieval(vectorstore, tfidf_vectorizer, tfidf_matrix, query, num_results)
    
    results_hybrid = evaluate_retrieval(df_sampled, hybrid_retrieval_wrapper, method_name="Hybrid Search")
    print("Hybrid Search Evaluation Results:")
    print(results_hybrid)

    # Combine results for comparison
    combined_results = [results_minsearch, results_chroma, results_hybrid]
    print("Combined Evaluation Results:")
    for res in combined_results:
        print(f"Method: {res['method']}, Hit Rate: {res['hit_rate']:.4f}, MRR: {res['mrr']:.4f}")

    # Save combined results to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(combined_results, f, indent=2)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()