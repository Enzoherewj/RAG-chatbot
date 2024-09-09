from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

def calculate_relevance(query, answer, context):
    query_embedding = embeddings.embed_query(query)
    answer_embedding = embeddings.embed_query(answer)
    context_embedding = embeddings.embed_query(context)
    
    query_answer_similarity = cosine_similarity([query_embedding], [answer_embedding])[0][0]
    answer_context_similarity = cosine_similarity([answer_embedding], [context_embedding])[0][0]
    
    return (query_answer_similarity + answer_context_similarity) / 2

def evaluate_response(query, answer, context):
    relevance = calculate_relevance(query, answer, context)
    return {
        "relevance": relevance,
        # Add more metrics here as needed
    }