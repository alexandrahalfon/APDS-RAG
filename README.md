# pdfRAG

A RAG system for PDF document processing and natural language querying.
* Built using FastAPI and Mistral AI
* Combines OCR processing, semantic search, and LLM generation to deliver accurate answers with proper source attribution from uploaded documents.

### Implementation Overview
```
pdfRAG-system/
│
├── Core Application
│   ├── app.py                    FastAPI REST API with upload/query endpoints
│   ├── main.py                   CLI interface and server launcher
│   └── ui.py                     Streamlit web interface
│
├── Document Processing Pipeline
│   ├── doc_processing.py         PDF OCR and content extraction
│   ├── chunking_step.py          Semantic document segmentation
│   └── embedding_step.py         Vector embedding with batching
│
├── Search & Generation
│   ├── similarity_search.py      Hybrid semantic + keyword search
│   ├── query_processing.py       Query transformation and caching
│   └── generation_step.py        LLM generation with verification
│
├── Orchestration
│   └── pipeline_main.py          End-to-end pipeline coordinator
│
├── Configuration & Dependencies
│   ├── requirements.txt          Python package dependencies
│   └── .env                      API keys and secrets
│
└── Runtime Directories (auto-generated)
    ├── embeddings/               Vector storage
    ├── structured_docs/          OCR outputs
    └── uploads/                  Temporary files
```

### Key Components

**Document Processing**
- OCR-powered text extraction from any PDF format using Mistral AI
- Intelligent chunking that respects sentence boundaries and context
- Automated metadata extraction and document structuring

**Advanced Search**
- Hybrid retrieval combining vector similarity with keyword precision
- Adaptive query processing based on question type classification
- Fast response times with caching
- **Diversity Re-ranking**: Uses MMR (Maximal Marginal Relevance) to balance:
  - Relevance score (similarity to query)
  - Novelty score (dissimilarity to already-selected chunks)

**Answer Generation**
- Context-aware responses with LLM-powered synthesis
- Comprehensive source citations with page references
- Friendly conversational handling for greetings and system queries
- **Entailment Verification**:
  - Checks if generated answer is logically supported by retrieved context
  - Reduces hallucination rate
  - Falls back to "insufficient information" when verification fails

**Safety & Reliability**
- PII detection and appropriate handling
- Medical/legal disclaimers for sensitive queries
- Checkpoint recovery for large document processing

### Tech Stack

<table>
<tr><th>Component</th><th>Technology</th><th>Purpose & Usage</th></tr>

<tr><td><b>API Framework</b></td><td><a href="https://fastapi.tiangolo.com/">FastAPI</a></td><td>Python web framework for building REST APIs with automatic OpenAPI documentation, async support, and type validation</td></tr>

<tr><td><b>ASGI Server</b></td><td><a href="https://www.uvicorn.org/">Uvicorn</a></td><td>ASGI server implementation for running FastAPI applications in production with async capabilities</td></tr>

<tr><td><b>AI/ML Platform</b></td><td><a href="https://docs.mistral.ai/">Mistral AI</a></td><td>Unified AI platform providing OCR processing (mistral-ocr-latest), text embeddings (mistral-embed), and large language model generation (mistral-large-latest)</td></tr>

<tr><td><b>Numerical Computing</b></td><td><a href="https://numpy.org/">NumPy</a></td><td>Fundamental library for scientific computing, used for efficient vector operations, similarity calculations, and embedding storage</td></tr>

<tr><td><b>Natural Language Processing</b></td><td><a href="https://www.nltk.org/">NLTK</a></td><td>Natural Language Toolkit for text processing, specifically used for sentence tokenization and text segmentation</td></tr>

<tr><td><b>Text Processing</b></td><td><a href="https://docs.python.org/3/library/collections.html#collections.Counter">collections.Counter</a></td><td>Python built-in for efficient counting operations, used in custom BM25 implementation for term frequency calculations</td></tr>

<tr><td><b>Web Interface</b></td><td><a href="https://streamlit.io/">Streamlit</a></td><td>Python framework for building interactive web applications, used for the document upload and query interface</td></tr>

<tr><td><b>File Handling</b></td><td><a href="https://github.com/andrew-d/python-multipart">python-multipart</a></td><td>Library for parsing multipart form data, essential for handling PDF file uploads in FastAPI</td></tr>

<tr><td><b>Configuration</b></td><td><a href="https://github.com/theskumar/python-dotenv">python-dotenv</a></td><td>Loads environment variables from .env files for secure API key management and configuration</td></tr>

<tr><td><b>Search Algorithm</b></td><td>Custom BM25 + Cosine Similarity</td><td>Hybrid search implementation combining keyword-based BM25 scoring with semantic vector similarity for optimal retrieval</td></tr>
</table>

### Dependencies Overview

**Core Dependencies:**
- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern, fast web framework for building APIs
- **[Uvicorn](https://www.uvicorn.org/)** - ASGI server for running FastAPI applications  
- **[Mistral AI](https://docs.mistral.ai/)** - AI platform for OCR, embeddings, and text generation
- **[NumPy](https://numpy.org/)** - Numerical computing for vector operations
- **[NLTK](https://www.nltk.org/)** - Natural language processing toolkit
- **[Streamlit](https://streamlit.io/)** - Web app framework for user interface

**Supporting Libraries:**
- **[collections.Counter](https://docs.python.org/3/library/collections.html#collections.Counter)** - Built-in Python utility for term frequency counting in BM25
- **[python-multipart](https://github.com/andrew-d/python-multipart)** - Multipart form parsing
- **[python-dotenv](https://github.com/theskumar/python-dotenv)** - Environment variable management


---

## Architecture & Design

### Workflow

The system operates in three distinct phases:

#### Phase 1: Document Ingestion

```
User Upload
    ↓
┌───────────────────────────────────────┐
│ PDF Upload Handler                    │
│ • Validates file format and size     │
│ • Saves to temporary storage         │
│ • Returns upload confirmation        │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│ OCR Processing (Mistral AI)          │
│ • Extracts text from pages           │
│ • Processes embedded images          │
│ • Generates structured markdown      │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│ Metadata Extraction (LLM)            │
│ • Document type classification       │
│ • Title and topic extraction         │
│ • Confidence scoring                 │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│ Content Structuring                   │
│ • Organizes by page numbers          │
│ • Splits into paragraphs             │
│ • Saves as JSON with metadata        │
└───────────────────────────────────────┘
```

#### Phase 2: Embedding Generation

```
Structured Document
    ↓
┌───────────────────────────────────────┐
│ Semantic Chunking                     │
│ • NLTK sentence tokenization         │
│ • 400-word chunks with overlap       │
│ • Context boundary preservation      │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│ Vector Embedding (Batched)            │
│ • Mistral embedding API calls        │
│ • Processes in batches of 20         │
│ • Checkpoint-based recovery          │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│ Storage & Indexing                    │
│ • NumPy arrays for vectors           │
│ • JSON metadata persistence          │
│ • BM25 index construction            │
└───────────────────────────────────────┘
```

#### Phase 3: Query & Response

```
User Question
    ↓
┌───────────────────────────────────────┐
│ Conversational Query Check            │
│ • Greeting detection (hello, hi, etc)│
│ • System questions (what can you do) │
│ • Skip knowledge base if conversational│
└───────────────────────────────────────┘
    ↓ (if document-related)
┌───────────────────────────────────────┐
│ Query Preprocessing                   │
│ • Intent detection & classification  │
│ • LLM-based query enhancement        │
│ • Cache lookup (MD5-based)           │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│ Hybrid Search Execution               │
│                                       │
│  ┌─────────────────────────────────┐ │
│  │ Semantic Branch                 │ │
│  │ → Cosine similarity             │ │
│  │ → Top-K vector matches          │ │
│  └─────────────────────────────────┘ │
│                                       │
│  ┌─────────────────────────────────┐ │
│  │ Keyword Branch                  │ │
│  │ → BM25 scoring                  │ │
│  │ → Exact term matches            │ │
│  └─────────────────────────────────┘ │
│                                       │
│  Weighted Fusion: 0.7×semantic + 0.3×keyword
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│ Result Post-Processing                │
│ • Merge overlapping chunks (0.8 threshold)│
│ • Diversity-based re-ranking         │
│ • Final top-K selection              │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│ Context Retrieval & Assembly          │
│ • Retrieve top-K relevant chunks     │
│ • Assemble context from results      │
│ • Prepare structured context         │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│ Answer Generation                     │
│ • LLM prompt construction            │
│ • Source-grounded generation         │
│ • Response formatting                │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│ Verification & Safety                 │
│ • Entailment checking                │
│ • PII detection                      │
│ • Disclaimer injection (if needed)   │
└───────────────────────────────────────┘
    ↓
Formatted Response with Citations
```

### Component Details

**API Layer**
- `app.py` → REST endpoints, request validation, document state management, error handling

**Document Pipeline**
- `doc_processing.py` → OCR via Mistral API, markdown generation, metadata extraction
- `chunking_step.py` → Sentence-aware segmentation, overlap handling, context preservation
- `embedding_step.py` → Batch embedding generation, checkpoint system, rate limit compliance

**Retrieval System**
- `similarity_search.py` → Cosine similarity computation, BM25 implementation, hybrid fusion, overlap merging, diversity re-ranking
- `query_processing.py` → Context retrieval orchestration, query classification, enhancement strategies, MD5-based caching, result post-processing

**Generation & Output**
- `generation_step.py` → Conversational query handling, LLM prompting, entailment verification, safety filters, citation formatting

**Interfaces**
- `ui.py` → Streamlit web UI for document upload and interactive querying
- `main.py` → CLI for server launching and batch processing modes

**Orchestration**
- `pipeline_main.py` → End-to-end coordination of ingestion, processing, and storage steps

### Key Considerations

**Why NumPy-based Vector Storage?** Uses NumPy-based vector database, rather than an external database:
- Simplified deployment with no external dependencies
- Full control over search algorithms and scoring
- Sufficient performance for small-to-medium collections (<100K chunks)

**Why Hybrid Search?** Pure semantic search misses exact terms; pure keyword search misses paraphrases. The implemented weighted combination:
- Captures conceptual similarity through embeddings (70% weight)
- Ensures exact term matching through BM25 (30% weight)
- Delivers more relevant results across diverse query types

**Why Sentence-Based Chunking?** Fixed-size chunks (e.g., 512 tokens) often split mid-sentence, breaking semantic coherence. This approach:
- Preserves natural language boundaries
- Maintains context through configurable overlap
- Enables precise citation to specific sentences

**Why Checkpoint Recovery?** Large documents can take minutes to process. Without checkpoints:
- Network failures waste all progress
- Rate limit errors require full restart
- User experience degrades significantly

The checkpoint system saves progress after each batch, enabling graceful recovery.

**Why Result Post-Processing?** Raw search results often contain redundant or overly similar chunks. The post-processing pipeline:
- **Merges overlapping chunks:** Combines chunks with >80% text overlap to reduce redundancy
- **Diversity re-ranking:** Balances relevance with diversity to avoid repetitive context
- **Quality optimization:** Ensures the final context provides comprehensive, non-redundant information

This approach delivers higher-quality context for answer generation compared to raw similarity scores.

**Why Structured Context Retrieval?** The `retrieve_context` function orchestrates the entire search pipeline:
- **Adaptive Processing:** Adjusts search strategy based on query type classification
- **Multi-step Pipeline:** Coordinates query enhancement, hybrid search, and post-processing
- **Quality Control:** Ensures retrieved chunks meet relevance thresholds before generation
- **Efficient Caching:** Leverages MD5-based caching to avoid redundant processing

This centralized approach ensures consistent, high-quality context retrieval across all queries.

### API Interface

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/upload` | POST | Accepts PDF files and triggers full processing pipeline |
| `/query` | POST | Receives questions, returns answers with source attribution |
| `/status` | GET | Reports system health and document/chunk statistics |
| `/clear` | DELETE | Resets system state by removing all documents |
| `/docs` | GET | Serves interactive Swagger UI documentation |

Interactive documentation available at: `http://localhost:8000/docs`

---

## Setup

### Prerequisites Checklist

Before installation, ensure you have:
- Python 3.8 or higher
- pip package manager
- Mistral AI API key ([Sign up here](https://docs.mistral.ai/))

**Step 1: Clone Repository**
```bash
git clone <repository-url>
cd pdfRAG
```

**Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

### Configuration Setup

**Create your environment file:**
```bash
cp .env.example .env
```

**Add your Mistral API key:**
```env
MISTRAL_API_KEY=your_mistral_api_key_here
```

**Optional: Customize Processing Parameters**

Edit these files to adjust behavior:

| File | Parameter | Default | Purpose |
|------|-----------|---------|---------| 
| `chunking_step.py` | `chunk_size` | 400 | Words per chunk |
| `chunking_step.py` | `overlap_size` | 50 | Overlap between chunks |
| `similarity_search.py` | `alpha` | 0.7 | Semantic vs. keyword weight |
| `embedding_step.py` | `batch_size` | 3 (20 in fast mode) | Embeddings per API call |

### Launch Options

**Option 1: Web Interface**
```bash
python main.py ui
```
Then navigate to `http://localhost:8501` in your browser.

**Option 2: API Server (Development)**
```bash
python main.py serve
```
Server runs at `http://localhost:8000` with auto-reload on code changes.

**Option 3: API Server (Production)**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

**Option 4: Clear All Data**
```bash
python main.py clear
```
Removes all processed documents, embeddings, and temporary files to start fresh.

### Using the System

#### Via Web Interface

1. Launch the Streamlit UI: `python main.py ui`
2. Navigate to the "Upload Documents" section
3. Click "Browse Files" and select one or more PDFs
4. Monitor the processing progress bar
5. Once complete, switch to the "Ask Questions" tab
6. Type your question and press 'Ask'
7. Review the answer with source page references

#### Via API (cURL)

**Upload one or more documents:**
```bash
curl -X POST http://localhost:8000/upload \
  -H "Content-Type: multipart/form-data" \
  -F "files=@/path-to-your-doc.pdf" 
```

**Ask a question:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main conclusions?",
    "top_k": 3
  }'
```

**Check system status:**
```bash
curl http://localhost:8000/status
```

Response includes document count, chunk count, and processing statistics.

---

### Constraints & Limitations

**Current Limitations**

**Language Support**
- English-only processing (can be extended with language-specific tokenizers and multilingual models)

**File Format Support**
- PDF-only (can be extended to support Word, PowerPoint, Excel, HTML, and Markdown files)

**External Dependencies**
- Complete reliance on Mistral AI 
- Rate limiting constraints for free-tier users

**Scalability Constraints**
- In-memory vector storage limits corpus size to available RAM
- No horizontal scaling or load distribution

**Real-Time Features**
- No streaming responses or live processing status updates
- Single-turn conversation without multi-turn memory

---
