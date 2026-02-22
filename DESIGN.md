# Gemini Production RAG - Architecture & Design

## Project Overview

This is a production-ready Retrieval-Augmented Generation (RAG) application that combines document processing, intelligent retrieval, and generative AI to create a conversational interface over custom documents.

**Key Features:**
- Hybrid search combining BM25 (keyword) and semantic (vector) retrieval
- Contextual compression with re-ranking for improved accuracy
- Real-time document embedding with Google Generative AI
- Interactive visualization of semantic clusters
- Streamlit-based UI for easy interaction

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI (app.py)                │
│         User Interface for Chat & Visualization         │
└────────────┬────────────────────────────────────────────┘
             │
    ┌────────┴────────┐
    │                 │
┌───▼──────────┐  ┌──▼──────────┐
│ Ingestion    │  │ Retrieval   │
│ (ingest)     │  │ (retrieval) │
└───┬──────────┘  └──┬──────────┘
    │                │
    │        ┌───────┴────────┐
    │        │                │
┌───▼────────▼──┐       ┌─────▼─────────┐
│ Vector DB     │       │ Generation    │
│ (Chroma)      │◄──────┤ (generation)  │
└───────────────┘       └────────┬──────┘
                                 │
                                 ▼
                        ┌────────────────┐
                        │ Gemini LLM     │
                        │ Output Parser  │
                        └────────────────┘
```

### Module Breakdown

#### 1. **ingestion.py** - Document Processing
- **Purpose:** Load, chunk, and embed documents
- **Workflow:**
  - Load TXT and PDF files from `./data/`
  - Split documents using `RecursiveCharacterTextSplitter` (chunk_size=1000, overlap=100)
  - Generate embeddings using Google Generative AI (text-embedding-004)
  - Store in Chroma vector database (persistent at `./database/`)

#### 2. **retrieval.py** - Hybrid Information Retrieval
- **Purpose:** Implement multi-strategy document retrieval with re-ranking
- **Components:**
  - **BM25 Retriever:** Keyword-based search (40% weight)
  - **Vector Retriever:** Semantic similarity search (60% weight)
  - **Ensemble Retriever:** Combines both strategies
  - **FlashrankRerank:** Re-ranks results for improved relevance
  - **Contextual Compression:** Filters to most relevant content

#### 3. **generation.py** - RAG Chain
- **Purpose:** Build the retrieval-augmented generation pipeline
- **Features:**
  - Uses Google Gemini 2.5 Flash LLM
  - Implements LCEL (LangChain Expression Language) chain
  - Custom prompt template for context-aware responses
  - String output parser for clean responses

#### 4. **visualization.py** - Semantic Analysis
- **Purpose:** Visualize document embeddings in 2D space
- **Method:**
  - Extract embeddings (768-dimensional)
  - Apply PCA for dimensionality reduction
  - Scatter plot visualization of semantic clusters

#### 5. **metrics.py** - Evaluation (Future)
- **Purpose:** Track RAG performance metrics
- **Planned features:**
  - RAGAS evaluation framework integration
  - Retrieval accuracy metrics
  - Generation quality scoring
  - Latency and cost tracking

---

## Data Flow

### Ingestion Pipeline
```
Raw Documents (TXT/PDF)
    ▼
Load & Concatenate
    ▼
Recursive Chunking (1000 tokens, 100 overlap)
    ▼
Google Generative AI Embeddings
    ▼
Chroma Vector Store (Persistent)
```

### Query Pipeline
```
User Query
    ▼
BM25 Retrieval (5 results)  +  Vector Retrieval (5 results)
    ▼
Ensemble Combination (40% BM25 + 60% Vector)
    ▼
FlashrankRerank (Re-order by relevance)
    ▼
Contextual Compression (Filter noise)
    ▼
Prompt Template Injection
    ▼
Gemini LLM Processing
    ▼
Response to User
```

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Streamlit | Interactive web UI |
| **Document Processing** | LangChain, RecursiveCharacterTextSplitter | Chunking & formatting |
| **Embeddings** | Google Generative AI | Text-to-vector conversion |
| **Vector Database** | ChromaDB | Persistent vector storage |
| **Retrieval** | LangChain Classic, BM25, FlashrankRerank | Hybrid search & ranking |
| **LLM** | Google Gemini 2.5 Flash | Text generation |
| **Visualization** | Scikit-learn (PCA), Matplotlib | Embedding visualization |

---

## Configuration & Parameters

### Chunking Strategy
```python
chunk_size = 1000          # Tokens per chunk
chunk_overlap = 100        # Sliding window overlap
```

### Retrieval Weights
```python
bm25_weight = 0.4          # Keyword search weight
vector_weight = 0.6        # Semantic search weight
top_k = 5                  # Results per retriever
```

### LLM Parameters
```python
model = "gemini-2.5-flash"
temperature = 0            # Deterministic output
max_output_tokens = 2048
```

---

## Future Improvements & Scope

### 1. **Enhanced Retrieval**
- [ ] Implement re-ranking with cross-encoders
- [ ] Add parent-child document retrieval (context awareness)
- [ ] Support for recursive retrieval (multi-hop reasoning)
- [ ] Query expansion and decomposition
- [ ] Adaptive chunk size based on document type

### 2. **Metrics & Evaluation** (metrics.py)
- [ ] Integrate RAGAS framework for RAG evaluation
- [ ] Implement retrieval-focused metrics:
  - Mean Reciprocal Rank (MRR)
  - Normalized Discounted Cumulative Gain (NDCG)
  - Hit Rate, Mean Average Precision (MAP)
- [ ] Generation quality metrics:
  - Answer relevance
  - Factual consistency
  - Faithfulness to retrieved context
- [ ] Performance monitoring:
  - Query latency tracking
  - Cost analysis per query
  - Cache hit rates

### 3. **Advanced Features**
- [ ] Multi-document summarization before retrieval
- [ ] Conversation memory & context tracking
- [ ] Source attribution (show retrieved documents to user)
- [ ] Streaming responses for better UX
- [ ] Support for multiple LLMs (Claude, OpenAI, etc.)

### 4. **Data Pipeline Improvements**
- [ ] Batch processing for large document collections
- [ ] Incremental indexing (add new docs without reindexing all)
- [ ] Metadata extraction and filtering
- [ ] Document similarity deduplication
- [ ] Support for web scraping and API sources

### 5. **Production Readiness**
- [ ] Error handling & graceful degradation
- [ ] Input validation and sanitization
- [ ] Rate limiting and quota management
- [ ] Logging and monitoring (structured logs)
- [ ] Docker containerization
- [ ] Database backup and recovery mechanisms
- [ ] API endpoint exposure (REST/GraphQL)

### 6. **Performance Optimization**
- [ ] Vector database sharding for scale
- [ ] Query caching layer
- [ ] Batch embedding processing
- [ ] Async/concurrent retrieval
- [ ] GPU acceleration for embeddings

### 7. **UI/UX Enhancements**
- [ ] Dark mode support
- [ ] Advanced query filters
- [ ] Document upload UI
- [ ] Real-time retrieval confidence scores
- [ ] Response feedback mechanism

### 8. **Security**
- [ ] API key management (vault/secrets manager)
- [ ] Role-based access control
- [ ] Data encryption at rest and in transit
- [ ] Audit logging
- [ ] Input/output sanitization

---

## Dependencies

See [requirements.txt](requirements.txt) for full list. Key packages:
- `langchain` - RAG framework
- `langchain-community` - Community retrievers
- `langchain-classic` - Classic LangChain APIs
- `langchain-google-genai` - Google AI integration
- `chromadb` - Vector database
- `streamlit` - UI framework
- `rank_bm25` - BM25 implementation
- `flashrank` - Re-ranking
- `ragas` - RAG evaluation (installed, not yet used)

---

## Performance Characteristics

| Operation | Typical Latency | Notes |
|-----------|-----------------|-------|
| Document Ingestion | 5-10s per 100KB | Parallel processing for PDFs |
| Embedding Generation | ~2s per 1000 tokens | Google API dependent |
| Retrieval | ~200-500ms | BM25 + Vector search |
| Re-ranking | ~100-200ms | FlashrankRerank overhead |
| LLM Generation | 2-5s | Gemini API dependent |
| **End-to-end Query** | **~5-10s** | Network latency dominant |

---

## Security Considerations

⚠️ **Current State:** API key stored in `.env` (development only)

**Production Improvements Needed:**
- [ ] Use environment secrets in cloud deployment
- [ ] Implement API rate limiting
- [ ] Validate user input to prevent prompt injection
- [ ] Add authentication/authorization layer
- [ ] Encrypt sensitive data

---

## Development Notes

- **Python Version:** 3.10+
- **Virtual Environment:** `.venv/` (required)
- **Code Style:** PEP 8 compliant
- **Package Management:** pip (see requirements.txt)

---

## References

- [LangChain Documentation](https://docs.langchain.com)
- [ChromaDB Docs](https://docs.trychroma.com)
- [Google Generative AI API](https://ai.google.dev)
- [Streamlit Documentation](https://docs.streamlit.io)
- [RAGAS Evaluation Framework](https://github.com/explodinggradients/ragas)
