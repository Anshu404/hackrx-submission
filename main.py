import os
import requests
import fitz  # PyMuPDF
from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field
from groq import Groq
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from typing import List
import hashlib
import time
from dotenv import load_dotenv

# --- Configuration and API Clients ---
load_dotenv() # Load variables from .env file

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EXPECTED_BEARER_TOKEN = "047f7135ffac517d60dd147e9d9618873b5bd69f07215538dcc30c2352b4bc0b"

# --- Model and API Client Initialization ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Check for API keys
if not PINECONE_API_KEY or not GROQ_API_KEY:
    raise ValueError("API keys for Pinecone or Groq not found. Please check your .env file.")

try:
    # Initialize Pinecone and Groq clients
    pc = Pinecone(api_key=PINECONE_API_KEY)
    groq_client = Groq(api_key=GROQ_API_KEY)

    # Load a free, high-quality embedding model from Hugging Face
    print("Loading SentenceTransformer model... (This may take a moment on first run)")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded successfully.")
    
    EMBEDDING_DIMENSION = 384

except Exception as e:
    print(f"Error initializing API clients or loading model: {e}")
    exit()


# --- Pydantic Models for API Data Structure ---
class HackRxRequest(BaseModel):
    documents: str = Field(..., description="URL to the PDF document.")
    questions: List[str] = Field(..., description="A list of questions to ask about the document.")

class HackRxResponse(BaseModel):
    answers: List[str]

# --- FastAPI Application Setup ---
app = FastAPI(
    title="HackRx 6.0 Intelligent Retrieval System (Groq Edition)",
    description="Processes documents to answer contextual questions using Groq and SentenceTransformers.",
    version="FINAL"
)

# --- Helper Functions for Core Logic ---

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 150):
    """Splits text into overlapping chunks."""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

def process_and_index_document(doc_url: str) -> str:
    """
    Downloads, processes, chunks, and indexes a PDF document into Pinecone.
    Returns the name of the Pinecone index created for this document.
    """
    print("Starting document processing...")
    response = requests.get(doc_url)
    response.raise_for_status()
    
    doc_text = ""
    with fitz.open(stream=response.content, filetype="pdf") as doc:
        for page in doc:
            doc_text += page.get_text()

    text_chunks = chunk_text(doc_text)
    if not text_chunks:
        raise ValueError("Document is empty or could not be processed.")

    index_name = "hackrx-final-" + hashlib.sha256(doc_url.encode()).hexdigest()[:12]

    if index_name not in pc.list_indexes().names():
        print(f"Creating new Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        time.sleep(1)

    index = pc.Index(index_name)

    print("Embedding and upserting document chunks...")
    batch_size = 100
    for i in range(0, len(text_chunks), batch_size):
        batch_chunks = text_chunks[i:i + batch_size]
        embeddings = embedding_model.encode(batch_chunks).tolist()
        vectors_to_upsert = []
        for j, chunk in enumerate(batch_chunks):
            vector_id = f"doc-chunk-{i+j}"
            vectors_to_upsert.append({
                "id": vector_id,
                "values": embeddings[j],
                "metadata": {"text": chunk}
            })
        index.upsert(vectors=vectors_to_upsert)
    print("Document processing complete.")
    return index_name

def execute_rag(index_name: str, question: str) -> str:
    """
    Executes the Retrieval-Augmented Generation for a single question.
    """
    index = pc.Index(index_name)

    # 1. Retrieve: Create embedding for the question and query Pinecone
    query_embedding = embedding_model.encode(question).tolist()
    
    # We use a higher top_k to ensure the right context is found
    retrieval_results = index.query(vector=query_embedding, top_k=7, include_metadata=True)
    
    # Check if any results were found
    if not retrieval_results['matches']:
        return "No relevant information was found in the document to answer this question."

    context = "\n---\n".join([match['metadata']['text'] for match in retrieval_results['matches']])

    # 2. Generate: Create a prompt and get answer from Groq
    # THIS IS THE KEY CHANGE! The new prompt asks for a summary.
    # prompt = f"""
    # You are an expert insurance analyst. Your task is to answer the user's question based *only* on the provided context.
    # - Synthesize the information from the context into a clear and concise answer.
    # - Do not just copy the text. Provide a helpful summary.
    # - If the context does not contain the information to answer the question, state that the information is not available in the document.

    # CONTEXT:
    # {context}

    # QUESTION:
    # {question}

    # ANSWER:
    # """
    # The new, better prompt
    prompt = f"""
    You are an expert insurance analyst. Your task is to answer the user's question based *only* on the provided context.
    - Synthesize the information from the context into a clear and concise answer.
    - Do not just copy the text. Provide a helpful summary.
    - If the context does not contain the information to answer the question, state that the information is not available in the document.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
    





    completion_response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful assistant specialized in document analysis."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )

    return completion_response.choices[0].message.content.strip()

# --- API Endpoint ---
@app.post("/hackrx/run", response_model=HackRxResponse, tags=["Submission"])
async def run_submission(
    payload: HackRxRequest,
    Authorization: str = Header(None)
):
    """
    Processes a document to answer a list of questions using a RAG pipeline.
    """
    if not Authorization or Authorization.split(" ")[1] != EXPECTED_BEARER_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )
        
    try:
        index_name = process_and_index_document(payload.documents)
        
        answers = []
        for question in payload.questions:
            answer = execute_rag(index_name, question)
            answers.append(answer)
            
        # Optional: Clean up the created index for hygiene
        # pc.delete_index(index_name)

        return HackRxResponse(answers=answers)
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to download document: {e}")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {e}")
