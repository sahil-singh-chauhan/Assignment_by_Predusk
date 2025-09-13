import os
import uuid
import requests
from typing import List

from flask import Flask, jsonify, render_template, request, session

from langchain_core.embeddings import Embeddings
# Removed sentence_transformers import - using Jina API instead
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
import re
import time
import tiktoken
from dotenv import load_dotenv
import google.generativeai as genai


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
# Environment variables loaded from .env (no hardcoded secrets)
# ----------------------------
load_dotenv(os.path.join(BASE_DIR, ".env"))
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_API_BASE = os.getenv('OPENAI_API_BASE', 'https://openrouter.ai/api/v1')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'apirag-1024')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'jina-embeddings-v3')
JINA_API_KEY = os.getenv('JINA_API_KEY')
JINA_EMBEDDING_DIMENSIONS = int(os.getenv('JINA_EMBEDDING_DIMENSIONS', '1024'))
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'openrouter').lower()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')

# Export for downstream SDKs that read from env
if OPENAI_API_KEY:
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['OPENAI_API_BASE'] = OPENAI_API_BASE
if PINECONE_API_KEY:
    os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

# Configure Gemini if selected
if LLM_PROVIDER == 'gemini' and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


# ----------------------------
# Jina Embeddings v3 wrapper using API
# ----------------------------
class JinaEmbeddingsWrapper(Embeddings):
    def __init__(self, model_name: str, api_key: str, dimensions: int = 1024):
        self.model_name = model_name
        self.api_key = api_key
        self.dimensions = dimensions
        if not api_key:
            raise ValueError("JINA_API_KEY is required for Jina embeddings")

    def _call_jina_api(self, texts: List[str]) -> List[List[float]]:
        """Call Jina embeddings API v3"""
        payload = {
            "model": self.model_name,
            "task": "retrieval.passage",
            "dimensions": self.dimensions,
            "late_chunking": False,
            "embedding_type": "float",
            "input": texts
        }
        
        try:
            print(f"DEBUG: Calling Jina embeddings API for {len(texts)} texts")
            resp = requests.post(
                "https://api.jina.ai/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=30
            )
            
            if resp.status_code != 200:
                print(f"DEBUG: Jina API error {resp.status_code}: {resp.text}")
                raise Exception(f"Jina API error: {resp.status_code} - {resp.text}")
            
            data = resp.json()
            embeddings = [item["embedding"] for item in data["data"]]
            print(f"DEBUG: Successfully got {len(embeddings)} embeddings from Jina API")
            return embeddings
            
        except Exception as e:
            print(f"DEBUG: Jina embeddings API failed: {e}")
            raise e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Process in batches to avoid API limits
        batch_size = 50  # Reasonable batch size for Jina API
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"DEBUG: Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size} with {len(batch)} texts")
            batch_embeddings = self._call_jina_api(batch)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        embeddings = self._call_jina_api([text])
        return embeddings[0]

embeddings_model = JinaEmbeddingsWrapper(EMBEDDING_MODEL, JINA_API_KEY, JINA_EMBEDDING_DIMENSIONS)


# ----------------------------
# Jina Reranker
# ----------------------------
def jina_rerank(question: str, docs: list, top_n: int = 3):
    """Rerank documents using Jina API"""
    api_key = os.getenv("JINA_API_KEY")
    model = os.getenv("JINA_RERANK_MODEL", "jina-reranker-v1-base-en")
    if not api_key or not docs:
        print(f"DEBUG: Jina rerank skipped - API key: {bool(api_key)}, docs: {len(docs)}")
        return docs  # fallback: no rerank

    # Extract plain text from docs or dicts (supports your current formats)
    def get_text(d):
        if hasattr(d, "page_content"):
            return d.page_content
        if isinstance(d, dict) and "document" in d:
            nd = d["document"]
            if hasattr(nd, "page_content"):
                return nd.page_content
            return nd.get("page_content", nd.get("text", ""))
        return d.get("page_content", d.get("text", ""))

    # Prepare documents for Jina API
    documents_for_api = [{"text": get_text(d)} for d in docs]
    
    payload = {
        "model": model,
        "query": question,
        "documents": documents_for_api,
        "top_n": int(os.getenv("JINA_TOP_N", str(top_n)))
    }
    
    try:
        print(f"DEBUG: Calling Jina rerank API with {len(documents_for_api)} documents")
        resp = requests.post(
            "https://api.jina.ai/v1/rerank",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=20,
        )
        
        if resp.status_code != 200:
            print(f"DEBUG: Jina API error {resp.status_code}: {resp.text}")
            return docs  # fallback on API error
            
        data = resp.json()
        print(f"DEBUG: Jina API response: {data}")
        
        # data["results"] contains ranked docs with indices
        results = data.get("results", [])
        if not results:
            print("DEBUG: No results from Jina API")
            return docs
            
        # Map back to original docs by indices
        out = []
        for r in results:
            idx = r.get("index")
            if idx is not None and 0 <= idx < len(docs):
                out.append(docs[idx])
        
        print(f"DEBUG: Jina rerank successful, returning {len(out)} documents")
        return out or docs
    except Exception as e:
        print(f"DEBUG: Jina rerank failed: {e}")
        return docs  # fallback on any API issue


# ----------------------------
# Pinecone / Vector store helpers
# ----------------------------
pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
INDEX_NAME = PINECONE_INDEX_NAME


def get_embedding_dimension() -> int:
    try:
        # For Jina embeddings v3, we know the dimension from configuration
        return JINA_EMBEDDING_DIMENSIONS
    except Exception:
        return 1024


def ensure_index_exists():
    dim = get_embedding_dimension()
    existing = [idx["name"] for idx in pc.list_indexes()]  # type: ignore
    if INDEX_NAME not in existing:
        pc.create_index(name=INDEX_NAME, dimension=dim, metric="cosine")


## Removed VectorStore wrapper to reduce startup memory


def upsert_pdf_to_vectorstore(pdf_path: str, namespace: str) -> int:
    print(f"DEBUG: Loading PDF from {pdf_path}")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,  # tokens - Jina v3 handles larger chunks better
        chunk_overlap=150,  # 15% overlap
    )
    loader = PyPDFLoader(pdf_path)
    data = loader.load()
    print(f"DEBUG: Loaded {len(data)} pages from PDF")
    chunks = splitter.split_documents(data)
    print(f"DEBUG: Split into {len(chunks)} chunks")

    # enrich metadata for citation
    title = os.path.basename(pdf_path)
    for idx, chunk in enumerate(chunks):
        chunk.metadata["source"] = pdf_path
        chunk.metadata["title"] = title
        # Unstructured may provide page number
        if "page" in chunk.metadata:
            section = f"page {chunk.metadata['page']}"
        else:
            section = "unknown"
        chunk.metadata["section"] = section
        chunk.metadata["position"] = idx

    print(f"DEBUG: Adding {len(chunks)} chunks to Pinecone namespace: {namespace}")
    
    try:
        # Try using Pinecone client directly instead of vector store
        index = pc.Index(INDEX_NAME)
        
        # Prepare vectors for direct upsert - batch embed all chunks at once
        chunk_texts = [chunk.page_content for chunk in chunks]
        print(f"DEBUG: Generating embeddings for {len(chunk_texts)} chunks in batch")
        embeddings = embeddings_model.embed_documents(chunk_texts)
        print(f"DEBUG: Generated {len(embeddings)} embeddings")
        
        vectors_to_upsert = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Prepare vector data
            vector_data = {
                "id": f"chunk_{i}_{namespace}",
                "values": embedding,
                "metadata": {
                    "text": chunk.page_content,
                    "source": chunk.metadata.get("source", ""),
                    "title": chunk.metadata.get("title", ""),
                    "section": chunk.metadata.get("section", ""),
                    "position": chunk.metadata.get("position", i)
                }
            }
            vectors_to_upsert.append(vector_data)
        
        print(f"DEBUG: Upserting {len(vectors_to_upsert)} vectors to Pinecone")
        
        upsert_response = index.upsert(vectors=vectors_to_upsert, namespace=namespace)
        print(f"DEBUG: Successfully upserted {upsert_response['upserted_count']} vectors")

        # Poll Pinecone until the namespace reports vectors (readiness)
        import time
        ready = False
        target_count = upsert_response.get('upserted_count', len(vectors_to_upsert))
        for _ in range(6):  # up to ~3s
            try:
                stats = index.describe_index_stats()
                ns = stats.get('namespaces', {}).get(namespace, {})
                count = ns.get('vector_count', 0)
                if count >= max(1, min(target_count, 1)):
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(0.5)
        print(f"DEBUG: Namespace ready: {ready}")
        
    except Exception as e:
        print(f"DEBUG: Failed to add chunks to Pinecone: {e}")
        import traceback
        traceback.print_exc()
        raise e
    
    # Verify the chunks were actually added
    try:
        index = pc.Index(INDEX_NAME)
        query_embedding = embeddings_model.embed_query("test")
        results = index.query(
            vector=query_embedding,
            top_k=1,
            namespace=namespace,
            include_metadata=True
        )
        print(f"DEBUG: Verification successful - {len(results['matches'])} matches found")
    except Exception as e:
        print(f"DEBUG: Verification failed: {e}")
    return len(chunks)


# ----------------------------
# RAG pipeline pieces (as provided, completed with an LLM)
# ----------------------------
# LLM initialization: OpenRouter (default) or Gemini (via env)
if LLM_PROVIDER == 'gemini':
    llm = None
else:
    llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "openai/gpt-oss-120b:free"))

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


def build_retriever(namespace: str | None):
    print(f"DEBUG: Building retriever for namespace: {namespace}")
    
    # Test direct Pinecone query
    try:
        index = pc.Index(INDEX_NAME)
        query_embedding = embeddings_model.embed_query("test")
        results = index.query(
            vector=query_embedding,
            top_k=5,  # increased for better retrieval
            namespace=namespace,
            include_metadata=True
        )
        print(f"DEBUG: Found {len(results['matches'])} documents in namespace")
    except Exception as e:
        print(f"DEBUG: Query failed: {e}")
    
    # Create a custom retriever that uses direct Pinecone queries
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
    from typing import List
    
    class DirectPineconeRetriever(BaseRetriever):
        index: object
        embeddings_model: object
        namespace: str
        
        def __init__(self, index, embeddings_model, namespace):
            super().__init__(
                index=index,
                embeddings_model=embeddings_model,
                namespace=namespace
            )
        
        def _get_relevant_documents(self, query: str) -> List[Document]:
            query_embedding = self.embeddings_model.embed_query(query)
            results = self.index.query(
                vector=query_embedding,
                top_k=5,  # increased for better retrieval
                namespace=self.namespace,
                include_metadata=True
            )
            
            # Convert Pinecone results to Document objects
            docs = []
            for match in results['matches']:
                # Handle different metadata structures
                metadata = match.get('metadata', {})
                text_content = metadata.get('text', '')
                if not text_content:
                    # Fallback to other possible text fields
                    text_content = metadata.get('page_content', '')
                
                doc = Document(
                    page_content=text_content,
                    metadata=metadata
                )
                docs.append(doc)
            return docs
    
    index = pc.Index(INDEX_NAME)
    base_retriever = DirectPineconeRetriever(index, embeddings_model, namespace)
    
    # If no LLM available (e.g., Gemini answering), skip MultiQuery and return base retriever
    if llm is None:
        return base_retriever
    
    # Test the custom retriever
    try:
        test_docs = base_retriever._get_relevant_documents("test")
        print(f"DEBUG: Retriever ready with {len(test_docs)} test documents")
    except Exception as e:
        print(f"DEBUG: Retriever test failed: {e}")
    
    return MultiQueryRetriever.from_llm(base_retriever, llm, prompt=query_prompt)



template = """Answer the question based only on the following context. Use inline citations [1], [2], etc. to reference specific sources from the context.

Context:
{context}

Question: {question}

Instructions:
1. Answer the question using only information from the provided context and in under 50 words.
2. Use inline citations [1], [2], [3], etc. to reference specific parts of the context that also under 80 words.
3. If you cannot find relevant information in the context, say "I cannot find relevant information in the provided context"
4. Do NOT provide source snippets - they will be added automatically

Answer:"""
prompt = ChatPromptTemplate.from_template(template)


def build_chains(namespace: str | None):
    # Backward-compat placeholder (not used). Keeping to avoid large refactor.
    retriever = build_retriever(namespace)

    def clean_snippet_text(text: str) -> str:
        # Remove common boilerplate and marketing lines
        patterns = [
            r"(?im)^\s*Scan to Download.*$",
            r"(?im)^\s*Written by .*$",
            r"(?im)^\s*Listen .*Audiobook.*$",
            r"(?im)^\s*About the book.*$",
            r"(?im)^\s*Check more about .*$",
            r"(?im)^\s*Key Point:.*$",
            r"(?im)^\s*inspiration\s*$",
        ]
        for pat in patterns:
            text = re.sub(pat, "", text)
        # Strip bullets and excessive whitespace
        lines = [re.sub(r"^[\-•\*\s]+", "", ln).strip() for ln in text.splitlines()]
        # Drop empty lines and very short headings
        lines = [ln for ln in lines if len(ln) > 2]
        cleaned = "\n".join(lines)
        # Collapse multiple newlines
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned

    def retrieve_and_rerank(input_dict):
        question = input_dict["question"]
        chat_history = input_dict.get("chat_history", [])
        retrieved_docs = retriever.invoke(question)
        print(f"DEBUG: Retrieved {len(retrieved_docs)} documents")
        # Use Jina reranker
        reranked_docs = jina_rerank(question, retrieved_docs, top_n=5)
        print(f"DEBUG: Reranked to {len(reranked_docs)} documents")
        
        # Format context with source information for citations
        formatted_context = []
        for i, doc in enumerate(reranked_docs, 1):
            source_info = f"[{i}] "
            
            # Handle both Document objects and dictionaries
            if hasattr(doc, 'metadata'):
                metadata = doc.metadata
                page_content = doc.page_content
            else:
                # Handle dictionary format from reranker
                if 'document' in doc:
                    # Reranker returns {id, index, score, document}
                    nested_doc = doc['document']
                    if hasattr(nested_doc, 'metadata'):
                        metadata = nested_doc.metadata
                        page_content = nested_doc.page_content
                    else:
                        metadata = nested_doc.get('metadata', {})
                        page_content = nested_doc.get('page_content', nested_doc.get('text', ''))
                else:
                    # Direct dictionary format
                    metadata = doc.get('metadata', {})
                    page_content = doc.get('page_content', doc.get('text', ''))
            
            if metadata.get('title'):
                source_info += f"Source: {metadata['title']}"
            if metadata.get('section') and metadata['section'] != 'unknown':
                source_info += f", {metadata['section']}"
            if metadata.get('position') is not None:
                source_info += f", Chunk {metadata['position'] + 1}"
            
            formatted_doc = f"{source_info}\n{page_content}"
            formatted_context.append(formatted_doc)
        
        context_text = "\n\n".join(formatted_context)
        print(f"DEBUG: Context length: {len(context_text)} characters")
        print(f"DEBUG: Context preview: {context_text[:200]}...")
        
        # Store source snippets for later appending
        source_snippets = []
        for i, doc in enumerate(reranked_docs, 1):
            if hasattr(doc, 'metadata'):
                metadata = doc.metadata
                page_content = doc.page_content
            else:
                if 'document' in doc:
                    nested_doc = doc['document']
                    if hasattr(nested_doc, 'metadata'):
                        metadata = nested_doc.metadata
                        page_content = nested_doc.page_content
                    else:
                        metadata = nested_doc.get('metadata', {})
                        page_content = nested_doc.get('page_content', nested_doc.get('text', ''))
                else:
                    metadata = doc.get('metadata', {})
                    page_content = doc.get('page_content', doc.get('text', ''))
            
            source_info = f"[{i}] "
            if metadata.get('title'):
                source_info += f"Source: {metadata['title']}"
            if metadata.get('section') and metadata['section'] != 'unknown':
                source_info += f", {metadata['section']}"
            if metadata.get('position') is not None:
                source_info += f", Chunk {metadata['position'] + 1}"
            
            # Clean and truncate snippet text
            snippet_raw = page_content
            snippet = clean_snippet_text(snippet_raw)
            if snippet:
                source_snippets.append(f"{source_info}\n{snippet}")
        
        return {"question": question, "context": context_text, "chat_history": chat_history, "source_snippets": source_snippets}

    # Not using continuous runnable chains anymore
    return None, None


def generate_answer_simple(namespace: str, question: str, chat_history: list[str] | list[dict]):
    retriever = build_retriever(namespace)

    # Retrieve and rerank
    retrieved_docs = retriever.invoke(question)
    print(f"DEBUG: Retrieved {len(retrieved_docs)} documents")
    # Use Jina reranker
    reranked_docs = jina_rerank(question, retrieved_docs, top_n=5)
    print(f"DEBUG: Reranked to {len(reranked_docs)} documents")

    # Build formatted context and source snippets (replicates previous logic)
    formatted_context = []
    source_snippets = []

    def extract_doc_fields(d):
        if hasattr(d, 'metadata'):
            return d.metadata, d.page_content
        if isinstance(d, dict) and 'document' in d:
            nd = d['document']
            if hasattr(nd, 'metadata'):
                return nd.metadata, nd.page_content
            return nd.get('metadata', {}), nd.get('page_content', nd.get('text', ''))
        return d.get('metadata', {}), d.get('page_content', d.get('text', ''))

    def clean_snippet_text(text: str) -> str:
        patterns = [
            r"(?im)^\s*Scan to Download.*$",
            r"(?im)^\s*Written by .*$",
            r"(?im)^\s*Listen .*Audiobook.*$",
            r"(?im)^\s*About the book.*$",
            r"(?im)^\s*Check more about .*$",
            r"(?im)^\s*Key Point:.*$",
            r"(?im)^\s*inspiration\s*$",
        ]
        for pat in patterns:
            text = re.sub(pat, "", text)
        lines = [re.sub(r"^[\-•\*\s]+", "", ln).strip() for ln in text.splitlines()]
        lines = [ln for ln in lines if len(ln) > 2]
        cleaned = "\n".join(lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned

    for i, d in enumerate(reranked_docs, 1):
        metadata, page_content = extract_doc_fields(d)
        source_info = f"[{i}] "
        if metadata.get('title'):
            source_info += f"Source: {metadata['title']}"
        if metadata.get('section') and metadata['section'] != 'unknown':
            source_info += f", {metadata['section']}"
        if metadata.get('position') is not None:
            try:
                source_info += f", Chunk {int(metadata['position']) + 1}"
            except Exception:
                pass
        formatted_context.append(f"{source_info}\n{page_content}")
        snippet = clean_snippet_text(page_content)
        if snippet:
            source_snippets.append(f"{source_info}\n{snippet}")

    context_text = "\n\n".join(formatted_context)
    print(f"DEBUG: Context length: {len(context_text)} characters")

    # Compose prompt and call LLM
    prompt_text = (
        "Answer the question based only on the following context. Use inline citations [1], [2], etc. to reference specific sources from the context.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {question}\n\n"
        "Instructions:\n"
        "1. Answer the question using only information from the provided context\n"
        "2. Use inline citations [1], [2], [3], etc. to reference specific parts of the context\n"
        "3. If you cannot find relevant information in the context, say \"I cannot find relevant information in the provided context\"\n"
        "4. Do NOT provide source snippets - they will be added automatically\n\n"
        "Answer:"
    )

    if LLM_PROVIDER == 'gemini':
        model = genai.GenerativeModel(GEMINI_MODEL)
        resp = model.generate_content(prompt_text)
        answer_text = (getattr(resp, 'text', '') or '').strip()
        if not answer_text:
            answer_text = "I cannot find relevant information in the provided context"
    else:
        llm_resp = llm.invoke(prompt_text)
        answer_text = getattr(llm_resp, 'content', str(llm_resp))
    answer = clean_output(answer_text)

    # Filter source snippets to only those cited
    cited_numbers = set(int(n) for n in re.findall(r"\[(\d+)\]", answer))
    filtered = []
    for snippet in source_snippets:
        m = re.match(r"\[(\d+)\]", snippet.strip())
        if m and int(m.group(1)) in cited_numbers:
            filtered.append(snippet)

    if filtered:
        answer += "\n\nSources:\n" + "\n\n".join(filtered)

    return answer


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
    # On refresh/new visit: clear only the current session's namespace
    old_namespace = session.get("namespace")
    if old_namespace:
        try:
            index = pc.Index(INDEX_NAME)
            index.delete(delete_all=True, namespace=old_namespace)
            print(f"DEBUG: Cleaned up old namespace: {old_namespace}")
        except Exception:
            pass
    
    # Clear session and upload directory completely
    session.clear()
    import shutil
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
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

    # Clear chat history for new PDF
    session["chat_history"] = []

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(save_path)

    try:
        # Use session-scoped namespace to avoid cross-file contamination
        if "session_id" not in session:
            session["session_id"] = str(uuid.uuid4())
        # Use a simpler namespace format
        namespace = f"pdf_{session['session_id'][:8]}"
        session["namespace"] = namespace
        print(f"DEBUG: Uploading PDF to namespace: {namespace}")
        num_chunks = upsert_pdf_to_vectorstore(save_path, namespace)
        print(f"DEBUG: Created {num_chunks} chunks from PDF")
    except Exception as e:
        print(f"DEBUG: Upload error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

    return jsonify({"success": True, "message": "PDF uploaded and processed", "chunks": num_chunks})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"success": False, "error": "Question is required"}), 400

    namespace = session.get("namespace")
    if not namespace:
        return jsonify({"success": False, "error": "No PDF uploaded in this session. Please upload a PDF first."}), 400

    print(f"DEBUG: Processing chat request for namespace: {namespace}")
    
    chat_history = session.get("chat_history", [])
    try:
        start = time.perf_counter()
        answer = generate_answer_simple(namespace, question, chat_history)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

    # Append to session history (simple string list for client chat rendering)
    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})
    session["chat_history"] = chat_history

    # rough token and cost estimates
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        encoding = None

    tokens_in = 0
    if encoding is not None:
        ctx = "\n".join([m.get("content", "") for m in chat_history[-6:]]) + "\n" + question
        tokens_in = len(encoding.encode(ctx))
    else:
        tokens_in = max(1, len(question) // 4)

    tokens_out = len(answer) // 4

    # very rough pricing (USD per 1K tokens)
    INPUT_PER_K = float(os.environ.get("MODEL_PRICE_IN", "0.0003"))
    OUTPUT_PER_K = float(os.environ.get("MODEL_PRICE_OUT", "0.0015"))
    cost_est = (tokens_in / 1000.0) * INPUT_PER_K + (tokens_out / 1000.0) * OUTPUT_PER_K

    return jsonify({
        "success": True,
        "answer": answer,
        "metrics": {
            "elapsed_ms": elapsed_ms,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd_est": round(cost_est, 6)
        }
    })


@app.route("/cleanup", methods=["POST"])
def cleanup_all():
    """Clean up all namespaces - for debugging"""
    try:
        index = pc.Index(INDEX_NAME)
        index.delete(delete_all=True)
        return jsonify({"success": True, "message": "All namespaces cleaned up"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/debug_chunks", methods=["GET"])
def debug_chunks():
    """Debug: Show sample chunks from current namespace"""
    namespace = session.get("namespace")
    if not namespace:
        return jsonify({"error": "No namespace found"}), 400
    
    try:
        index = pc.Index(INDEX_NAME)
        # Get some random vectors to see what's stored
        dummy_vector = [0.0] * JINA_EMBEDDING_DIMENSIONS
        results = index.query(
            vector=dummy_vector,
            top_k=5,
            namespace=namespace,
            include_metadata=True
        )
        
        chunks = []
        for match in results['matches']:
            metadata = match.get('metadata', {})
            chunks.append({
                'text': metadata.get('text', '')[:200] + '...',
                'source': metadata.get('source', ''),
                'section': metadata.get('section', ''),
                'position': metadata.get('position', '')
            })
        
        return jsonify({"namespace": namespace, "chunks": chunks})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)


