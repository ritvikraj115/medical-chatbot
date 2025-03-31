from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import os
from langchain.schema import Document


load_dotenv()

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()
print(f"Number of text chunks: {len(text_chunks)}")
print("Sample chunk:", text_chunks[4000])


persist_directory = "./chroma_db"  # Directory to store ChromaDB files
vector_store = Chroma(
    collection_name="medical_bot",
    embedding_function=embeddings,
    persist_directory=persist_directory
)
# Define a default metadata dictionary
default_metadata = {"source": "unknown"}

# Create Document instances with non-empty metadata
documents = [
    Document(page_content=str(chunk), metadata=default_metadata) for chunk in text_chunks
]
print(f"Number of documents created: {len(documents)}")
store=vector_store.add_documents(documents)
print(len(store))
vector_store.persist()
