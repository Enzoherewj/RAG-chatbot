import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import logging
from evaluation import evaluate_response

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

    print("Loading existing vectorstore...")
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory=VECTORSTORE_PATH, embedding_function=embeddings)

    print("Setting up conversational chain...")
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-4-1106-preview")  # Update to the latest available model

    # Define the prompt template
    prompt_template = """You are a novel assistant, specifically knowledgeable about "The Brothers Karamazov" by Fyodor Dostoevsky. Your task is to answer questions based on the following context from the novel:

    {context}
    
    Try to be specific to the novel. Can give references to specific passages, characters, events, or themes from the novel. Don't need to mention about the context given since the user won't be able to see it.
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

def chat(query, chat_history):
    global qa_chain
    try:
        # Ensure chat_history is in the correct format
        formatted_history = [(q, a) for q, a in chat_history]
        
        # Use the invoke method instead of __call__
        result = qa_chain.invoke({"question": query, "chat_history": formatted_history})
        
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
        logging.error(f"Error in chat function: {str(e)}", exc_info=True)
        raise