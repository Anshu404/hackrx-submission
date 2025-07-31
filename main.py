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
    # This model runs locally on your CPU.
    print("Loading SentenceTransformer model... (This may take a moment on first run)")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded successfully.")
    
    # IMPORTANT: The dimension of this model is 384
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
    version="1.1.0"
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
    # 1. Download and Parse PDF
    response = requests.get(doc_url)
    response.raise_for_status()
    
    doc_text = ""
    with fitz.open(stream=response.content, filetype="pdf") as doc:
        for page in doc:
            doc_text += page.get_text()

    # 2. Chunk Text
    text_chunks = chunk_text(doc_text)
    if not text_chunks:
        raise ValueError("Document is empty or could not be processed.")

    # 3. Create a unique index name
    index_name = "hackrx-groq-" + hashlib.sha256(doc_url.encode()).hexdigest()[:12]

    # 4. Create Pinecone Index if it doesn't exist
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,  # Use the correct dimension: 384
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        time.sleep(1)

    index = pc.Index(index_name)

    # 5. Generate Embeddings and Upsert to Pinecone
    batch_size = 100
    for i in range(0, len(text_chunks), batch_size):
        batch_chunks = text_chunks[i:i + batch_size]
        
        # Create embeddings using SentenceTransformer model
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

    return index_name

def execute_rag(index_name: str, question: str) -> str:
    """
    Executes the Retrieval-Augmented Generation for a single question.
    """
    index = pc.Index(index_name)

    # 1. Retrieve: Create embedding for the question and query Pinecone
    query_embedding = embedding_model.encode(question).tolist()
    
    retrieval_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    
    context = "\n---\n".join([match['metadata']['text'] for match in retrieval_results['matches']])

    # 2. Generate: Create a prompt and get answer from Groq
    prompt = f"""
    You are an expert at reading and interpreting policy documents. Answer the following question based *only* on the provided context.
    If the context does not contain the answer, state that the information is not available in the provided text.
    Your answer should be direct, concise, and extracted from the text.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    
    completion_response = groq_client.chat.completions.create(
        model="llama3-8b-8192", # A great, fast model on Groq
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
            
        # Optional: Clean up the created index
        # pc.delete_index(index_name)

        return HackRxResponse(answers=answers)
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to download document: {e}")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {e}")

# To run: uvicorn main:app --reload