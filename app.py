from flask import Flask, render_template, request
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.helper import download_hugging_face_embeddings
from src.prompt import prompt_template
from langchain.memory import ConversationBufferMemory

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Initialize embeddings
embeddings = download_hugging_face_embeddings()

# Initialize Chroma vector store from the persisted directory
docsearch = Chroma(
    collection_name="medical_bot",  # must match exactly
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Define the prompt template
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Initialize conversational memory with specified input and output keys
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    input_key='query',          # Input key expected by the chain
    output_key='result'         # Output key produced by the chain
)

# Define chain type arguments
chain_type_kwargs = {"prompt": PROMPT}

# Initialize the language model
llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={'max_new_tokens': 512, 'temperature': 0.8}
)

# Create the RetrievalQA chain with memory
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs,
    memory=memory
)



@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    print(f"User Input: {msg}")
    result = qa({"query": msg})
    print(result)
    response = result["result"]
    print(f"Response: {response}")
    return response

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
