import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import logging
from evaluation import evaluate_response
from tqdm import tqdm

# Global variables
qa_chain = None

def initialize_chatbot():
    global qa_chain
    
    print("Starting RAG chatbot initialization...")

    # Load environment variables from .env file
    load_dotenv()

    # Get OpenAI API key from environment variable
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Set up logging
    logging.basicConfig(filename='chatbot.log', level=logging.INFO)

    # Define path for saving vectorstore
    VECTORSTORE_PATH = "vectorstore"

    embeddings, vectorstore = load_or_create_embeddings(VECTORSTORE_PATH)

    print("Setting up conversational chain...")
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o-mini")

    # Define the prompt template
    prompt_template = """You are a novel assistant, specifically knowledgeable about "The Brothers Karamazov" by Fyodor Dostoevsky. Your task is to answer questions based on the following context from the novel:

    {context}

    Human: {question}
    AI Assistant: Let me answer that based on the information from "The Brothers Karamazov":"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )

    print("RAG chatbot initialization complete!")

def load_or_create_embeddings(VECTORSTORE_PATH):
    embeddings = OpenAIEmbeddings()
    
    if os.path.exists(VECTORSTORE_PATH):
        print("Loading existing vectorstore...")
        vectorstore = Chroma(persist_directory=VECTORSTORE_PATH, embedding_function=embeddings)
        return embeddings, vectorstore

    print("Loading document...")
    loader = TextLoader("Brothers_Karamazov.txt")
    documents = loader.load()

    print("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks. Now creating embeddings and storing in vector database...")
    
    # Use tqdm to show progress
    vectorstore = Chroma.from_documents(tqdm(chunks), embeddings, persist_directory=VECTORSTORE_PATH)

    return embeddings, vectorstore

def chat(query, chat_history):
    global qa_chain
    try:
        result = qa_chain({"question": query, "chat_history": chat_history})
        answer = result["answer"]
        source_docs = result["source_documents"]
        
        # Evaluate the response
        context = " ".join([doc.page_content for doc in source_docs])
        evaluation = evaluate_response(query, answer, context)
        
        # Log the interaction and evaluation
        logging.info(f"Query: {query}")
        logging.info(f"Answer: {answer}")
        logging.info(f"Evaluation: {evaluation}")
        
        return answer, source_docs, evaluation
    except Exception as e:
        logging.error(f"Error in chat function: {str(e)}")
        raise