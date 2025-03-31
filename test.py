from langchain.vectorstores import Chroma
from src.helper import download_hugging_face_embeddings

# Initialize embeddings
embeddings = download_hugging_face_embeddings()

# Define the directory where Chroma will persist data
persist_directory = './chroma_db'
# Initialize Chroma vector store from the persisted directory
docsearch = Chroma(
    collection_name="medical_bot",  # must match exactly
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)
query = "what is diabetes"
retrieved_docs = docsearch.similarity_search(query, k=5)
print(f"Retrieved {len(retrieved_docs)} documents")
for doc in retrieved_docs:
    print(f"Doc snippet: {doc.page_content[:200]}... Metadata: {doc.metadata}")

