import os
from ingest import ingest_document

def main():
    FILE_PATH = "/app/Brothers_Karamazov.txt"
    VECTORSTORE_PATH = "/app/vectorstore"

    if not os.path.exists(VECTORSTORE_PATH) or not os.listdir(VECTORSTORE_PATH):
        print("Vectorstore not found. Generating embeddings...")
        ingest_document(FILE_PATH, VECTORSTORE_PATH)
    else:
        print("Vectorstore already exists. Skipping embedding generation.")

if __name__ == "__main__":
    main()