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

# Initialize OpenAI client
client = OpenAI()

def load_data(vectorstore_path):
    """
    Load chunks and their embeddings from the Chroma vectorstore.
    
    Args:
        vectorstore_path (str): Path to the Chroma vectorstore.
    
    Returns:
        df_chunks (pd.DataFrame): DataFrame containing chunk_id, text, and vector.
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings)
    
    chunks = vectorstore.get()
    df_chunks = pd.DataFrame({
        'chunk_id': chunks['ids'],
        'text': chunks['documents'],
        'vector': chunks['embeddings']
    })
    
    return df_chunks

def generate_questions(df_chunks, num_questions_per_chunk=5):
    """Generate questions for each chunk using OpenAI API."""
    questions = []
    
    prompt_template = """
    You are a user of our novel assistant application for "The Brothers Karamazov".
    Formulate {num_questions} questions based on the following excerpt from the novel.
    Make the questions specific to this excerpt.
    The excerpt should contain the answer to the questions, and the questions should
    be complete and not too short. Use as few words as possible from the excerpt.

    Excerpt:
    {text}

    Provide the output in JSON format:
    {{"questions": ["question1", "question2", ..., "question{num_questions}"]}}
    """

    for _, chunk in tqdm(df_chunks.iterrows(), total=len(df_chunks), desc="Generating questions"):
        prompt = prompt_template.format(num_questions=num_questions_per_chunk, text=chunk['text'])
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        
        questions_json = json.loads(response.choices[0].message.content)
        for question in questions_json['questions']:
            questions.append({
                'chunk_id': chunk['chunk_id'],
                'question': question
            })
    
    return pd.DataFrame(questions)

def create_minsearch_index(documents):
    """Create and fit a MinSearch index with given documents."""
    index = Index(
        text_fields=['text'],
        keyword_fields=['chunk_id']
    )
    index.fit(documents)
    return index

def minsearch_retrieval(index, query, num_results=10):
    """Retrieve documents using MinSearch index."""
    results = index.search(
        query=query,
        filter_dict={},
        boost_dict={},
        num_results=num_results
    )
    return results

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

def evaluate_retrieval(df_questions, index, retrieval_func):
    """Evaluate retrieval performance using hit rate and MRR."""
    relevance_total = []

    for _, row in tqdm(df_questions.iterrows(), total=len(df_questions), desc="Evaluating questions"):
        expected_chunk_id = row['chunk_id']
        question = row['question']

        results = retrieval_func(index, question)
        retrieved_chunk_ids = [str(doc['chunk_id']) for doc in results]
        relevance = [doc_id == str(expected_chunk_id) for doc_id in retrieved_chunk_ids]
        relevance_total.append(relevance)
    
    hr = hit_rate(relevance_total)
    mrr_score = mrr(relevance_total)

    return {
        'hit_rate': hr,
        'mrr': mrr_score
    }

def main():
    vectorstore_path = "vectorstore"
    output_path = os.path.join('vectorstores', 'retrieval_results.json')
    questions_path = os.path.join('vectorstores', 'generated_questions.csv')

    print("Loading data from Chroma vectorstore...")
    df_chunks = load_data(vectorstore_path)
    print(f"Loaded {len(df_chunks)} chunks.")

    if os.path.exists(questions_path):
        print("Loading existing questions...")
        df_questions = pd.read_csv(questions_path)
    else:
        print("Generating questions...")
        df_questions = generate_questions(df_chunks)
        df_questions.to_csv(questions_path, index=False)
    print(f"Total questions: {len(df_questions)}")

    print("Creating MinSearch index...")
    documents = df_chunks.to_dict(orient='records')
    index = create_minsearch_index(documents)
    print("MinSearch index created and fitted.")

    print("Evaluating retrieval performance...")
    results = evaluate_retrieval(df_questions, index, minsearch_retrieval)
    print("Evaluation Results:")
    print(results)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()