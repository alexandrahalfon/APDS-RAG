from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import tempfile
import shutil
import os
from pathlib import Path
from dotenv import load_dotenv
from mistralai import Mistral

# Import your pipeline modules
from doc_processing import process_pdf_complete
from chunking_step import semantic_chunk_document
from embedding_step import generate_embeddings, save_embeddings
from similarity_search import load_embeddings_from_files
from generation_step import ask_document

load_dotenv()

app = FastAPI(
    title="Document RAG API",
    description="Upload PDFs and ask questions using RAG",
    version="1.0.0"
)

# Request/Response models
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]

class UploadResponse(BaseModel):
    message: str
    files_processed: List[str]
    total_chunks: int

# Global state
class DocumentStore:
    def __init__(self):
        self.client = None
        self.embeddings = None
        self.metadata = None
        self.is_ready = False
    
    def initialize(self):
        api_key = os.getenv('MISTRAL_API_KEY')
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment")
        
        self.client = Mistral(api_key=api_key)
        
        # Try to load existing embeddings
        try:
            self.embeddings, self.metadata = load_embeddings_from_files()
            self.is_ready = True
            print(f"Loaded existing embeddings: {len(self.metadata)} chunks")
        except Exception as e:
            print(f"No existing embeddings found: {e}")
            self.is_ready = False

store = DocumentStore()

@app.on_event("startup")
async def startup_event():
    store.initialize()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Document RAG API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /upload - Upload PDF documents",
            "query": "POST /query - Ask questions about documents", 
            "status": "GET /status - Check system status",
            "clear": "DELETE /clear - Clear all documents"
        }
    }

@app.post("/upload", response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process PDF documents"""
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Validate file types
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type: {file.filename}. Only PDF files are supported."
            )
    
    processed_files = []
    all_chunks = []
    
    for file in files:
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                shutil.copyfileobj(file.file, tmp_file)
                temp_path = tmp_file.name
            
        
            structured_json_path = process_pdf_complete(temp_path, store.client)
            chunks = semantic_chunk_document(structured_json_path, save_to_disk=False)
            
            # Add source filename to chunks
            for chunk in chunks:
                chunk['source_file'] = file.filename
            
            all_chunks.extend(chunks)
            processed_files.append(file.filename)
            
            # Clean up temp file
            os.unlink(temp_path)
            
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Error processing {file.filename}: {str(e)}"
            )
    
    if all_chunks:
        # Generate embeddings
        chunks_with_embeddings = generate_embeddings(all_chunks, store.client, save_npy=False)
        
        # Save embeddings
        save_embeddings(chunks_with_embeddings)
        
        # Reload embeddings into store
        store.embeddings, store.metadata = load_embeddings_from_files()
        store.is_ready = True
    
    return UploadResponse(
        message="Documents processed successfully",
        files_processed=processed_files,
        total_chunks=len(all_chunks)
    )

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Ask questions about uploaded documents"""
    
    if not store.is_ready:
        raise HTTPException(
            status_code=400, 
            detail="No documents available. Please upload PDF files first."
        )
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        result = ask_document(
            query=request.question,
            embeddings=store.embeddings,
            metadata=store.metadata,
            mistral_client=store.client,
            top_k=request.top_k,
            show_sources=True
        )
        
        return QueryResponse(
            answer=result['answer'],
            sources=result['sources']
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing query: {str(e)}"
        )

@app.get("/status")
async def get_status():
    """Get system status"""
    if store.is_ready:
        return {
            "status": "ready",
            "total_chunks": len(store.metadata),
            "documents_count": len(set(chunk.get('source_file', 'unknown') for chunk in store.metadata))
        }
    else:
        return {
            "status": "not_ready",
            "total_chunks": 0,
            "documents_count": 0
        }

@app.delete("/clear")
async def clear_documents():
    """Clear all documents and embeddings"""
    try:
        # Remove embeddings directory
        embeddings_dir = Path("./embeddings")
        if embeddings_dir.exists():
            shutil.rmtree(embeddings_dir)
        
        # Reset store
        store.embeddings = None
        store.metadata = None
        store.is_ready = False
        
        return {"message": "All documents cleared successfully"}
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error clearing documents: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)