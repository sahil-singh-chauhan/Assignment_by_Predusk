import os
import uuid
from typing import List

from flask import Flask, jsonify, render_template, request, session

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_pinecone import PineconeRerank
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI


# ----------------------------
# Flask setup
# ----------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

# Ensure upload directory exists
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ----------------------------
# Environment variables (as provided)
# ----------------------------
os.environ.setdefault('OPENAI_API_KEY', "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")
os.environ.setdefault('OPENAI_API_BASE', "https://openrouter.ai/api/v1")
os.environ.setdefault('PINECONE_API_KEY', "pcsk_74vokm_KUuxFinVGUYHVxjtpyxwMixzQVV9whb7TCyUpUs1JjPkxupr95GShq4Fx8e6USz")


# ----------------------------
# Embedding model wrapper (as provided)
# ----------------------------
class SentenceTransformerEmbeddingsWrapper(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()


model_name = "latterworks/ollama-embeddings"
embeddings_model = SentenceTransformerEmbeddingsWrapper(model_name)


# ----------------------------
# Pinecone / Vector store helpers
# ----------------------------
pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
INDEX_NAME = "apirag"


def get_vector_store_existing() -> PineconeVectorStore:
    return PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings_model)


def upsert_pdf_to_vectorstore(pdf_path: str) -> int:
    splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=100)
    loader = UnstructuredPDFLoader(pdf_path)
    data = loader.load()
    chunks = splitter.split_documents(data)
    for chunk in chunks:
        chunk.metadata["source_file"] = pdf_path
    PineconeVectorStore.from_documents(chunks, index_name=INDEX_NAME, embedding=embeddings_model)
    return len(chunks)


# ----------------------------
# RAG pipeline pieces (as provided, completed with an LLM)
# ----------------------------
# Use a valid OpenRouter model id; can be overridden via OPENAI_MODEL env var
llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "openai/gpt-4o-mini"))

query_prompt = PromptTemplate(
    input_variables=["question"],
    template=(
        """You are an AI language model assistant. Your task is to generate
     five different versions of the given user question to retrieve relevant documents
    from a vector database. By generating multiple perspectives on the user question,
    your goal is to help the user overcome some of the limitations of the distance
    based similarity search. Provide these alternative questions separated by new lines.
    Original question: {question}"""
    ),
)


def build_retriever():
    vector_store = get_vector_store_existing()
    base_retriever = vector_store.as_retriever()
    return MultiQueryRetriever.from_llm(base_retriever, llm, prompt=query_prompt)


reranker = PineconeRerank(api_key=os.environ.get('PINECONE_API_KEY'), top_n=5)

template = (
    """Answer the question based only on the following context and also give source snippets from the pdf below the answer:
{context}

Question: {question}
"""
)
prompt = ChatPromptTemplate.from_template(template)


def build_chains():
    retriever = build_retriever()

    def retrieve_and_rerank(input_dict):
        question = input_dict["question"]
        chat_history = input_dict.get("chat_history", [])
        retrieved_docs = retriever.get_relevant_documents(question)
        reranked_docs = reranker.rerank(retrieved_docs, question)
        return {"question": question, "context": reranked_docs, "chat_history": chat_history}

    chain_with_memory_reranked = (
        RunnableLambda(retrieve_and_rerank)
        | prompt
        | llm
        | StrOutputParser()
    )

    chain = (
        {"context": RunnableLambda(lambda x: reranker.rerank(retriever.invoke(x['question']), x['question'])),
         "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

    return chain_with_memory_reranked, chain


def clean_output(output_string: str) -> str:
    formatted_output = output_string.replace('\n\n', '\n')
    lines = formatted_output.split('\n')
    processed_lines = []
    for line in lines:
        processed_line = line.lstrip('* ').strip()
        processed_lines.append(processed_line)
    final_output = '\n'.join(processed_lines)
    return final_output


# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def index():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
        session["chat_history"] = []
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"}), 400
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"success": False, "error": "Only PDF files are supported"}), 400

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(save_path)

    try:
        num_chunks = upsert_pdf_to_vectorstore(save_path)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

    return jsonify({"success": True, "message": "PDF uploaded and processed", "chunks": num_chunks})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"success": False, "error": "Question is required"}), 400

    chain_with_memory_reranked, _ = build_chains()

    chat_history = session.get("chat_history", [])
    try:
        result = chain_with_memory_reranked.invoke({
            "question": question,
            "chat_history": chat_history
        })
        answer = clean_output(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

    # Append to session history (simple string list for client chat rendering)
    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})
    session["chat_history"] = chat_history

    return jsonify({"success": True, "answer": answer})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)


