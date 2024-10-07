import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from tqdm import tqdm

def ingest_document(file_path, vectorstore_path):
    # Load environment variables
    load_dotenv()

    # Ensure OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    print(f"Loading document from {file_path}...")
    loader = TextLoader(file_path)
    documents = loader.load()

    print("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks. Now creating embeddings and storing in vector database...")
    
    embeddings = OpenAIEmbeddings()
    
    # Use tqdm to show progress
    vectorstore = Chroma.from_documents(
        tqdm(chunks),
        embeddings,
        persist_directory=vectorstore_path
    )

    print(f"Ingestion complete. Vectorstore saved to {vectorstore_path}")

if __name__ == "__main__":
    FILE_PATH = "Brothers_Karamazov.txt"
    VECTORSTORE_PATH = "vectorstore"
    
    ingest_document(FILE_PATH, VECTORSTORE_PATH)