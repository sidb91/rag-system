# 🎯 RAG System: Industry-Standard Portfolio Enhancement Plan

**Last Updated:** March 2026  
**Objective:** Transform a working RAG system into a production-grade, portfolio-worthy project with observability, scalability, and enterprise safety standards.

---

## 📊 Current State Analysis

### ✅ What We Have
- **Core RAG Pipeline:** Document ingestion → Hybrid retrieval (BM25 + Vector) → Flashrank re-ranking → Gemini generation
- **Persistent Storage:** ChromaDB for vector storage
- **UI Layer:** Streamlit-based chat interface with visualization
- **Foundation:** Modular architecture with clean separation of concerns

### ❌ What's Missing (Production Gaps)
1. **No visibility into system behavior** (black-box generation)
2. **No evaluation metrics** (how do we know if answers are good?)
3. **Simple text-only data ingestion** (can't handle PDFs with tables, web content)
4. **Stateless chat history** (no persistence, scales poorly)
5. **No caching mechanism** (expensive API calls for repeated questions)
6. **No automation** (manual deployment, no scheduled data updates)
7. **No safety guardrails** (no input/output validation, can't refuse unsafe requests)
8. **Single-threaded Streamlit** (can't power external APIs like mobile apps)

---

## 🚀 Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2) — Observability
Focus on making the system measurable and debuggable.

#### 1.1 The "Observer" Layer: RAGAS + LangSmith Integration

**Why This First?**
- Enables data-driven decisions for all future improvements
- Identifies which generation/retrieval components underperform
- Minimal architectural changes required

**Implementation Strategy:**

**A) RAGAS Framework (Automated Evaluation)**
```python
# Step 1: Add to requirements.txt
ragas
datasets
sentence-transformers
```

**File: `src/metrics.py` (Expand existing placeholder)**
```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevance,
    context_precision,
    context_recall,
)
from datasets import Dataset

def evaluate_answer(
    question: str,
    answer: str,
    retrieved_docs: list[str],
    ground_truth_answer: str = None
) -> dict:
    """
    Evaluate RAG output using RAGAS metrics.
    
    Metrics:
    - Faithfulness: Does the answer rely on the context? (Penalizes hallucinations)
    - Answer Relevance: Does it answer the question? (Not going off-topic)
    - Context Precision: Did retriever find relevant docs? (No noise)
    - Context Recall: Did retriever find all relevant docs? (No missing info)
    """
    # Create evaluation dataset
    eval_data = {
        "question": [question],
        "answer": [answer],
        "contexts": [retrieved_docs],
        "ground_trueths": [[ground_truth_answer]] if ground_truth_answer else None,
    }
    
    eval_dataset = Dataset.from_dict(eval_data)
    
    # Run evaluation
    results = evaluate(
        eval_dataset,
        metrics=[faithfulness, answer_relevance, context_precision]
    )
    
    return {
        "faithfulness": results["faithfulness"],
        "answer_relevance": results["answer_relevance"],
        "context_precision": results["context_precision"],
    }

def batch_evaluate_queries(queries_with_answers: list[dict]) -> dict:
    """
    Evaluate multiple Q&A pairs to benchmark system performance.
    Useful for regression testing after code changes.
    """
    results = []
    for item in queries_with_answers:
        evaluation = evaluate_answer(
            question=item["question"],
            answer=item["answer"],
            retrieved_docs=item["context"],
            ground_truth_answer=item.get("expected_answer")
        )
        results.append({
            "question": item["question"],
            "metrics": evaluation
        })
    
    # Calculate aggregates
    avg_faithfulness = sum(r["metrics"]["faithfulness"] for r in results) / len(results)
    avg_relevance = sum(r["metrics"]["answer_relevance"] for r in results) / len(results)
    avg_precision = sum(r["metrics"]["context_precision"] for r in results) / len(results)
    
    return {
        "individual_results": results,
        "averages": {
            "faithfulness": avg_faithfulness,
            "answer_relevance": avg_relevance,
            "context_precision": avg_precision,
        },
        "timestamp": datetime.now().isoformat()
    }
```

**B) LangSmith Integration (Production Tracing)**
```python
# Step 1: Add to requirements.txt
langsmith
```

**File: `src/generation.py` (Enhancement)**
```python
# At module level:
from langsmith import traceable
import os

# Set LangSmith API key (add to .env)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "rag-system-prod"
# LANGSMITH_API_KEY must be in .env

# Decorator your RAG chain:
@traceable(name="rag_chain", tags=["generation", "retrieval"])
def get_rag_chain(vector_db, hybrid_retriever):
    """
    Now LangSmith automatically logs:
    - Input question
    - Retrieved context (source, chunk ID)
    - LLM prompt sent to Gemini
    - Generation output
    - Latency and token usage
    - Cost estimation
    
    Dashboard: https://smith.langchain.com/
    """
    # ... existing rag_chain creation code ...
    return rag_chain
```

**Integration with Streamlit UI:**
```python
# In app.py sidebar, add:
with st.sidebar:
    st.header("📊 Monitoring")
    if st.button("View LangSmith Dashboard"):
        st.markdown(
            "[Open LangSmith](https://smith.langchain.com/)",
            unsafe_allow_html=True
        )
    
    # Show local metrics if available
    if "last_evaluation" in st.session_state:
        st.metric(
            "Faithfulness Score",
            f"{st.session_state.last_evaluation['faithfulness']:.2f}",
            delta="high = good"
        )
```

**Key Dependencies:**
- `ragas>=0.1.0`
- `langsmith>=0.0.1`
- No database changes needed

**Expected Outcome:**
- Dashboard showing faithfulness/relevance/precision metrics
- Trace logs viewable in LangSmith for every generation
- Ability to compare metrics before/after system changes

---

### Phase 2: Data Connectors (Weeks 3-4) — Multimodal Ingestion

#### 2.1 Advanced Document Processing with Dynamic Metadata

**Why Now?**
- Expands data sources (PDFs with tables, web content)
- Adds traceability via metadata (citations)
- Still uses existing ChromaDB/retrieval infrastructure

**Implementation Strategy:**

**A) PDF with Tables Support (Unstructured.io)**
```python
# Add to requirements.txt
unstructured[pdf,docx]>=0.10.0
pdf2image
pillow
pytesseract  # For OCR if needed
```

**File: `src/ingestion.py` (New function)**
```python
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.html import partition_html
from datetime import datetime
from urllib.parse import urlparse
import hashlib

def process_documents_advanced(
    data_path="./data",
    web_urls=None,
    persist_dir="./database"
):
    """
    Enhanced ingestion supporting:
    1. PDF with tables (Unstructured)
    2. Web content (HTML/Markdown)
    3. Dynamic metadata tagging (source, date, page number)
    """
    
    all_docs = []
    
    # === TIER 1: LOCAL FILES (TXT, PDF) ===
    # Keep existing logic for .txt files
    text_loader = DirectoryLoader(data_path, glob="**/*.txt", loader_cls=TextLoader)
    all_docs.extend(text_loader.load())
    
    # === TIER 2: PDF WITH TABLES (Unstructured) ===
    pdf_paths = glob.glob(f"{data_path}/**/*.pdf", recursive=True)
    for pdf_path in pdf_paths:
        # Unstructured extracts text + tables with layout preservation
        elements = partition_pdf(pdf_path)
        
        for element in elements:
            doc = Document(
                page_content=str(element),
                metadata={
                    "source": pdf_path,
                    "source_type": "pdf_unstructured",
                    "element_type": element.category,  # "table", "text", "figure"
                    "extracted_at": datetime.now().isoformat(),
                }
            )
            all_docs.append(doc)
    
    # === TIER 3: WEB CONTENT (if URLs provided) ===
    if web_urls:
        for url in web_urls:
            try:
                # Use Firecrawl for intelligent web scraping
                # (Alternative: Playwright + html2text for DIY approach)
                elements = partition_html(url=url)
                
                for element in elements:
                    doc = Document(
                        page_content=str(element),
                        metadata={
                            "source": url,
                            "source_type": "web",
                            "domain": urlparse(url).netloc,
                            "fetched_at": datetime.now().isoformat(),
                        }
                    )
                    all_docs.append(doc)
            except Exception as e:
                logger.warning(f"Failed to fetch {url}: {e}")
    
    # === CHUNKING WITH METADATA PRESERVATION ===
    # Use langchain_text_splitters with metadata preservation
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    for doc in all_docs:
        split_docs = splitter.split_documents([doc])
        
        # Add chunk-level metadata
        for i, chunk in enumerate(split_docs):
            chunk.metadata.update({
                "chunk_index": i,
                "chunk_id": hashlib.md5(
                    f"{doc.metadata['source']}{i}".encode()
                ).hexdigest()[:12],
                "chunk_created_at": datetime.now().isoformat(),
            })
        chunks.extend(split_docs)
    
    # === EMBEDDING & STORAGE (existing Chroma logic) ===
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    return vector_db

def track_metadata_usage():
    """
    Helper to show which sources/chunk_ids were used in generation.
    Enables citation tracking: "According to [website.com, chunk_abc123]..."
    """
    pass
```

**B) Citation & Source Tracking in Generation**
```python
# File: `src/generation.py` (Enhancement)

def get_rag_chain_with_citations(vector_db, hybrid_retriever):
    """
    Modified RAG chain that includes source attribution.
    """
    from langchain_core.prompts import PromptTemplate
    
    # Custom prompt with instruction to cite sources
    prompt = PromptTemplate(
        template="""You are a helpful assistant. Answer the question based on the provided context.
        
IMPORTANT: Always cite your sources using this format:
[Source: {source} | Chunk ID: {chunk_id}]

Context:
{context}

Question: {question}
Answer: """,
        input_variables=["context", "question", "source", "chunk_id"]
    )
    
    # Modify retriever to return sources explicitly
    def retriever_with_sources(query):
        docs = hybrid_retriever.invoke(query)
        context = "\n\n".join([
            f"[{doc.metadata.get('source', 'unknown')}] {doc.page_content}"
            for doc in docs
        ])
        sources = [doc.metadata.get('source') for doc in docs]
        chunk_ids = [doc.metadata.get('chunk_id', 'N/A') for doc in docs]
        
        return {
            "context": context,
            "sources": sources,
            "chunk_ids": chunk_ids,
        }
    
    rag_chain = (
        {
            "context": retriever_with_sources,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )
    
    return rag_chain
```

**Integration with Streamlit:**
```python
# In app.py, enhance document upload:
with st.sidebar:
    st.header("📤 Data Ingestion")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload documents (TXT, PDF)",
        accept_multiple_files=True,
        type=["txt", "pdf"]
    )
    
    # Web scraping input
    web_urls = st.text_area(
        "Or add web URLs (one per line)",
        placeholder="https://example.com\nhttps://docs.example.com"
    )
    
    # Process button
    if st.button("Ingest Documents"):
        with st.spinner("Processing..."):
            # Save uploaded files temporarily
            for uploaded_file in uploaded_files:
                with open(f"./data/{uploaded_file.name}", "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            # Parse URLs
            urls = [u.strip() for u in web_urls.split("\n") if u.strip()]
            
            # Ingest
            vector_db = process_documents_advanced(
                web_urls=urls if urls else None
            )
            st.success(f"Indexed {len(vector_db.get()['ids'])} chunks")
```

**Key Dependencies:**
- `unstructured[pdf,docx]>=0.10.0`
- `pdf2image`
- `pillow`
- `pytesseract` (optional, for OCR)

**Expected Outcome:**
- PDFs with tables parsed intelligently
- Web URLs auto-scraped and indexed
- All chunks tagged with source, date, type
- Citation links in generated answers

---

### Phase 3: High-Performance Memory (Weeks 5-6) — Persistent Storage & Cache

#### 3.1 Persistent Chat History (PostgreSQL/Redis)

**Why This?**
- Conversations survive app restarts
- Enable multi-session support (user returns to past chats)
- Build user profiles based on history
- Meet enterprise retention requirements

**Implementation Strategy:**

**A) PostgreSQL for Chat History**
```python
# Add to requirements.txt
sqlalchemy>=2.0
psycopg2-binary>=2.9
```

**File: `src/database.py` (New)**
```python
from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json

# Database schema
Base = declarative_base()

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    id = Column(String(36), primary_key=True)
    user_id = Column(String(255), index=True)
    title = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata = Column(JSON)  # Store: model version, settings, etc.

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(String(36), primary_key=True)
    session_id = Column(String(36), index=True)
    role = Column(String(10))  # "user" or "assistant"
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Enhanced with metadata
    retrieved_docs = Column(JSON)  # Store doc IDs, source citations
    generation_metrics = Column(JSON)  # Faithfulness, latency, tokens used
    cost = Column(Integer)  # In cents, for tracking API spend

class ChatDB:
    def __init__(self, db_url="postgresql://user:password@localhost/rag_system"):
        """Initialize database connection."""
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def save_session(self, user_id: str, title: str = None, metadata: dict = None):
        """Create new chat session."""
        session_id = str(uuid.uuid4())
        session = self.Session()
        
        chat_session = ChatSession(
            id=session_id,
            user_id=user_id,
            title=title or f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            metadata=metadata or {}
        )
        session.add(chat_session)
        session.commit()
        session.close()
        
        return session_id
    
    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        retrieved_docs: list = None,
        metrics: dict = None
    ):
        """Save message (with optional generation metadata)."""
        session = self.Session()
        
        message = ChatMessage(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role=role,
            content=content,
            retrieved_docs=retrieved_docs,
            generation_metrics=metrics
        )
        session.add(message)
        session.commit()
        session.close()
    
    def get_session_history(self, session_id: str) -> list[dict]:
        """Retrieve all messages in a session."""
        session = self.Session()
        messages = session.query(ChatMessage)\
            .filter_by(session_id=session_id)\
            .order_by(ChatMessage.timestamp)\
            .all()
        
        result = [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "metrics": msg.generation_metrics
            }
            for msg in messages
        ]
        session.close()
        return result
    
    def list_user_sessions(self, user_id: str) -> list[dict]:
        """Get all sessions for a user."""
        session = self.Session()
        sessions = session.query(ChatSession)\
            .filter_by(user_id=user_id)\
            .order_by(ChatSession.updated_at.desc())\
            .all()
        
        result = [
            {
                "id": s.id,
                "title": s.title,
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat()
            }
            for s in sessions
        ]
        session.close()
        return result
```

**Integration with Streamlit:**
```python
# In app.py (top-level):
from src.database import ChatDB

# Initialize database
db = ChatDB(os.getenv("DATABASE_URL", "postgresql://localhost/rag_system"))

# Sidebar: Session management
with st.sidebar:
    st.header("💾 Chat History")
    
    # Get user ID (from auth or default)
    user_id = st.session_state.get("user_id", "demo_user")
    
    # List past sessions
    past_sessions = db.list_user_sessions(user_id)
    
    if past_sessions:
        selected_session = st.selectbox(
            "Load session:",
            [f"{s['title']} ({s['created_at']})" for s in past_sessions]
        )
        
        if st.button("Load Selected Session"):
            selected_id = past_sessions[0]["id"]  # Simplified
            history = db.get_session_history(selected_id)
            st.session_state.messages = history
            st.session_state.current_session_id = selected_id
    
    # Start new session
    if st.button("+ New Session"):
        st.session_state.current_session_id = db.save_session(user_id)
        st.session_state.messages = []

# Main chat loop (modified):
if prompt := st.chat_input("Ask about your documents..."):
    session_id = st.session_state.get("current_session_id")
    
    if not session_id:
        session_id = db.save_session(user_id)
        st.session_state.current_session_id = session_id
    
    # Save user message
    db.save_message(session_id, "user", prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate response
    with st.spinner("Thinking..."):
        retrieved_docs = retriever.invoke(prompt)
        response = rag_chain.invoke(prompt)
        
        # Extract metrics for storage
        metrics = {
            "model": "gemini-2.5-flash",
            "tokens_used": estimate_tokens(response),
            "latency_ms": 0  # Measure this
        }
    
    # Save assistant message
    db.save_message(
        session_id,
        "assistant",
        response,
        retrieved_docs=[doc.metadata for doc in retrieved_docs],
        metrics=metrics
    )
    st.session_state.messages.append({"role": "assistant", "content": response})
```

**B) Semantic Caching with RedisVL**
```python
# Add to requirements.txt
redis>=4.5.0
redisvl>=0.1.0
```

**File: `src/cache.py` (New)**
```python
from redisvl.index import Index
from redisvl.query.filter import Tag
import json
from datetime import datetime

class SemanticCache:
    def __init__(self, redis_url="redis://localhost:6379"):
        """
        Initialize semantic cache using Redis Vector Library.
        
        Real-world impact: Gemini API calls cost $0.075 per 1M input tokens.
        Cache hit rate of 30-40% on typical usage = 30-40% cost reduction.
        """
        self.index = Index.from_redis(
            url=redis_url,
            name="rag_cache"
        )
    
    def check_cache(self, question: str, threshold=0.95) -> str | None:
        """
        Check if semantically similar question exists in cache.
        
        Args:
            threshold (float): Similarity score (0-1). 0.95 = must be very similar.
        
        Returns:
            Cached answer if found, else None.
        """
        # Embed the question using same model as RAG system
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        question_vector = embeddings.embed_query(question)
        
        # Search Redis for similar questions
        results = self.index.query(
            vector=question_vector,
            top_k=1,
            filters=[Tag("ttl") > 0]  # Not expired
        )
        
        if results and results[0].score >= threshold:
            cached_entry = json.loads(results[0].data)
            return cached_entry["answer"]
        
        return None
    
    def cache_answer(
        self,
        question: str,
        answer: str,
        context_docs: list[str],
        ttl_days=30
    ):
        """
        Store Q&A pair in semantic cache.
        
        TTL = Time-to-Live. Cache expires after 30 days (configurable).
        This prevents stale cached answers if knowledge base is updated.
        """
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        question_vector = embeddings.embed_query(question)
        
        entry = {
            "question": question,
            "answer": answer,
            "context_docs": context_docs,
            "cached_at": datetime.utcnow().isoformat(),
            "ttl": ttl_days * 24 * 3600,
        }
        
        # Store with vector
        self.index.upsert(
            [(question, entry, question_vector)],
            ttl=ttl_days * 24 * 3600
        )
    
    def get_cache_stats(self) -> dict:
        """Get caching performance metrics."""
        # Count total cached entries
        total_entries = len(self.index.list_documents())
        
        return {
            "cached_questions": total_entries,
            "estimated_api_savings": f"${total_entries * 0.000075:.2f}",  # ~150 tokens per Q
        }
```

**Integration with Generation Chain:**
```python
# In src/generation.py:

def get_rag_chain_cached(vector_db, hybrid_retriever, use_cache=True):
    """
    RAG chain with semantic caching.
    
    Decision tree:
    1. Check semantic cache → if hit, return cached answer
    2. If miss, retrieve context
    3. Generate with Gemini
    4. Cache result
    """
    from src.cache import SemanticCache
    
    semantic_cache = SemanticCache() if use_cache else None
    
    def rag_with_cache(question: str) -> str:
        # Step 1: Check cache
        if semantic_cache:
            cached_answer = semantic_cache.check_cache(question, threshold=0.95)
            if cached_answer:
                st.info("📦 Answer from cache (30% cost savings!)")
                return cached_answer
        
        # Step 2: Retrieve
        docs = hybrid_retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Step 3: Generate
        prompt = PromptTemplate(
            template="Context: {context}\n\nQuestion: {question}\n\nAnswer:",
            input_variables=["context", "question"]
        )
        chain = prompt | llm
        answer = chain.invoke({"context": context, "question": question})
        
        # Step 4: Cache
        if semantic_cache:
            semantic_cache.cache_answer(
                question,
                answer,
                [doc.metadata.get("source") for doc in docs]
            )
        
        return answer
    
    return rag_with_cache
```

**Key Dependencies:**
- `sqlalchemy>=2.0`
- `psycopg2-binary>=2.9`
- `redis>=4.5.0`
- `redisvl>=0.1.0`

**Expected Outcome:**
- Users can resume chats across sessions
- Semantic cache reduces API costs by ~30-40%
- Full audit trail of all Q&A with metrics

---

### Phase 4: Enterprise Orchestration (Weeks 7-8) — Automation & API

#### 4.1 Auto-Sync Pipeline with Prefect

**Why This?**
- Demonstrates data engineering competency
- Production systems can't rely on manual uploads
- Shows understanding of scheduled, scalable workflows

**Implementation Strategy:**

**File: `src/orchestration.py` (New)**
```python
# Add to requirements.txt
prefect>=2.14.0
boto3  # For S3 support (optional)
```

```python
from prefect import flow, task, get_run_logger
from prefect.schedules import IntervalSchedule
from prefect.deployments import Deployment
from datetime import timedelta, datetime
import os
import glob
import hashlib

# Store file hashes to detect new/changed files
PROCESSED_FILES_REGISTRY = "./database/processed_files.json"

@task(retries=2, retry_delay_seconds=60)
def watch_data_folder(folder_path="./data") -> list[str]:
    """
    Watch folder for new or modified files.
    Returns list of files that need processing.
    """
    logger = get_run_logger()
    
    import json
    
    # Load processed files registry
    processed = {}
    if os.path.exists(PROCESSED_FILES_REGISTRY):
        with open(PROCESSED_FILES_REGISTRY, "r") as f:
            processed = json.load(f)
    
    # Scan folder
    new_files = []
    for file_path in glob.glob(f"{folder_path}/**/*", recursive=True):
        if os.path.isfile(file_path) and file_path.endswith((".txt", ".pdf")):
            # Compute file hash
            with open(file_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            # Check if new or modified
            if file_path not in processed or processed[file_path] != file_hash:
                new_files.append(file_path)
                logger.info(f"Detected new/modified file: {file_path}")
    
    return new_files

@task
def ingest_files(file_paths: list[str]) -> int:
    """
    Ingest new files into ChromaDB.
    Returns number of chunks created.
    """
    logger = get_run_logger()
    
    if not file_paths:
        logger.info("No new files to ingest.")
        return 0
    
    from src.ingestion import process_documents_advanced
    
    # Ingest
    vector_db = process_documents_advanced(data_path="./data")
    
    # Update registry
    import json
    processed = {}
    if os.path.exists(PROCESSED_FILES_REGISTRY):
        with open(PROCESSED_FILES_REGISTRY, "r") as f:
            processed = json.load(f)
    
    for file_path in file_paths:
        with open(file_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        processed[file_path] = file_hash
    
    with open(PROCESSED_FILES_REGISTRY, "w") as f:
        json.dump(processed, f, indent=2)
    
    chunk_count = len(vector_db.get()["ids"])
    logger.info(f"Ingested {chunk_count} chunks total.")
    
    return chunk_count

@task
def validate_ingestion(chunk_count: int) -> bool:
    """Validate that ingestion succeeded."""
    logger = get_run_logger()
    
    if chunk_count > 0:
        logger.info(f"✅ Validation passed: {chunk_count} chunks in DB")
        return True
    else:
        logger.error("❌ Validation failed: No chunks found")
        return False

@flow(name="RAG Auto-Sync Pipeline", description="Watch folder and auto-index new documents")
def auto_sync_pipeline():
    """
    Main orchestration flow:
    1. Watch folder for new files
    2. Ingest new files
    3. Validate
    4. Log metrics
    """
    new_files = watch_data_folder()
    chunk_count = ingest_files(new_files)
    is_valid = validate_ingestion(chunk_count)
    
    return {
        "new_files_detected": len(new_files),
        "chunks_created": chunk_count,
        "validation_passed": is_valid,
        "timestamp": datetime.utcnow().isoformat()
    }

# Deploy with schedule (runs every 6 hours)
deployment = Deployment.build_from_flow(
    flow=auto_sync_pipeline,
    name="rag-auto-sync-deployment",
    schedule=IntervalSchedule(interval=timedelta(hours=6)),
    parameters={},
    skip_upload=False,
)

if __name__ == "__main__":
    # Start Prefect server: prefect server start
    # Run this to register the deployment
    deployment.apply()
    print("✅ Deployment registered. Prefect will run the pipeline every 6 hours.")
```

**Streamlit Integration (Show Pipeline Status):**
```python
# In app.py:
from prefect.client.orchestration import get_client

with st.sidebar:
    st.header("🤖 Automation")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run Sync Now"):
            # Manually trigger flow
            from src.orchestration import auto_sync_pipeline
            result = auto_sync_pipeline()
            st.success(f"Synced {result['chunks_created']} chunks!")
    
    with col2:
        if st.button("View History"):
            # Show last 5 runs
            client = get_client()
            runs = client.read_flows()
            st.json(runs)
```

#### 4.2 FastAPI Backend (Decouple from Streamlit)

**Why This?**
- Powers mobile apps, desktop clients, integrations
- Streamlit is great UI but not an API
- Demonstrates full-stack thinking

**File: `src/api.py` (New)**
```python
# Add to requirements.txt
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0
python-multipart
```

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging
from datetime import datetime

app = FastAPI(
    title="RAG System API",
    description="Production RAG backend",
    version="1.0.0"
)

# Enable CORS for frontend/mobile apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
from src.ingestion import process_documents_advanced
from src.retrieval import get_advanced_retriever
from src.generation import get_rag_chain_cached
from src.metrics import evaluate_answer
from src.cache import SemanticCache
from src.database import ChatDB

logger = logging.getLogger(__name__)

# Load once at startup
vector_db = process_documents_advanced()
retriever = get_advanced_retriever(vector_db)
rag_chain = get_rag_chain_cached(vector_db, retriever, use_cache=True)
db = ChatDB()

# === Pydantic Models (Request/Response Schema) ===

class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    user_id: str = "api_user"

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    session_id: str
    confidence: float  # Faithfulness score
    latency_ms: int
    cached: bool = False

class EvaluationRequest(BaseModel):
    question: str
    answer: str
    context: list[str]
    expected_answer: Optional[str] = None

class EvaluationResponse(BaseModel):
    faithfulness: float
    answer_relevance: float
    context_precision: float

# === API Endpoints ===

@app.get("/health")
async def health_check():
    """Health check for load balancers."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "vectordb_chunks": len(vector_db.get()["ids"])
    }

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest) -> QueryResponse:
    """
    Main RAG query endpoint.
    
    Returns answer with sources and confidence metrics.
    """
    import time
    from src.cache import SemanticCache
    
    start_time = time.time()
    
    try:
        # Create session if needed
        session_id = request.session_id or db.save_session(request.user_id)
        
        # Check semantic cache
        cache = SemanticCache()
        cached_answer = cache.check_cache(request.question)
        
        if cached_answer:
            # Cache hit
            response = QueryResponse(
                answer=cached_answer,
                sources=[],
                session_id=session_id,
                confidence=0.95,
                latency_ms=int((time.time() - start_time) * 1000),
                cached=True
            )
        else:
            # Cache miss - full generation
            retrieved_docs = retriever.invoke(request.question)
            answer = rag_chain(request.question)
            
            # Evaluate
            evaluation = evaluate_answer(
                question=request.question,
                answer=answer,
                retrieved_docs=[doc.page_content for doc in retrieved_docs]
            )
            
            sources = [doc.metadata.get("source", "unknown") for doc in retrieved_docs]
            
            # Save to history
            db.save_message(
                session_id,
                "user",
                request.question
            )
            db.save_message(
                session_id,
                "assistant",
                answer,
                retrieved_docs=[doc.metadata for doc in retrieved_docs],
                metrics=evaluation
            )
            
            # Cache for next time
            cache.cache_answer(request.question, answer, sources)
            
            response = QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id,
                confidence=evaluation["faithfulness"],
                latency_ms=int((time.time() - start_time) * 1000),
                cached=False
            )
        
        return response
    
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest) -> EvaluationResponse:
    """
    Evaluate a specific Q&A pair using RAGAS metrics.
    Useful for testing and benchmarking.
    """
    try:
        metrics = evaluate_answer(
            question=request.question,
            answer=request.answer,
            retrieved_docs=request.context,
            ground_truth_answer=request.expected_answer
        )
        
        return EvaluationResponse(**metrics)
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def trigger_ingestion(background_tasks: BackgroundTasks):
    """
    Manually trigger document ingestion.
    Non-blocking (returns immediately, runs in background).
    """
    def ingest():
        from src.orchestration import auto_sync_pipeline
        auto_sync_pipeline()
        logger.info("✅ Ingestion completed")
    
    background_tasks.add_task(ingest)
    return {
        "status": "ingestion_started",
        "message": "Check back in a few seconds for completion"
    }

@app.get("/sessions/{user_id}")
async def get_user_sessions(user_id: str):
    """Retrieve all chat sessions for a user."""
    sessions = db.list_user_sessions(user_id)
    return {"sessions": sessions}

@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """Retrieve chat history for a specific session."""
    history = db.get_session_history(session_id)
    return {"messages": history}

@app.get("/metrics/cache")
async def get_cache_metrics():
    """Get caching performance metrics."""
    cache = SemanticCache()
    stats = cache.get_cache_stats()
    return stats

if __name__ == "__main__":
    import uvicorn
    # Run with: uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Docker Deployment (Optional but recommended):**
```dockerfile
# File: Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Docker compose for local development:**
```yaml
# File: docker-compose.yml
version: "3.8"

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: rag_system
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7
    ports:
      - "6379:6379"

  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://user:password@postgres:5432/rag_system
      REDIS_URL: redis://redis:6379
      GOOGLE_API_KEY: ${GOOGLE_API_KEY}
    depends_on:
      - postgres
      - redis
    volumes:
      - ./data:/app/data
      - ./database:/app/database

  prefect:
    image: prefecthq/prefect:2-latest
    ports:
      - "4200:4200"
    environment:
      PREFECT_HOME: /root/.prefect
    volumes:
      - prefect_data:/root/.prefect

volumes:
  postgres_data:
  prefect_data:
```

**Start locally:**
```bash
docker-compose up -d
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs (Swagger UI)
```

**Key Dependencies:**
- `fastapi>=0.100.0`
- `uvicorn>=0.23.0`
- `pydantic>=2.0`
- `docker` (for containerization)

**Expected Outcome:**
- RESTful API powering multiple clients
- Self-documenting Swagger UI
- Containerized deployment ready for cloud

---

### Phase 5: Guardrails & Safety (Week 9) — Input/Output Validation

#### 5.1 NeMo Guardrails + Pydantic Output Parsers

**Why This?**
- Enterprise requirement: AI must not lie or go off-topic
- Prevent prompt injection attacks
- Ensure output format consistency

**Implementation Strategy:**

**File: `src/guardrails.py` (New)**
```python
# Add to requirements.txt
nemoguardrails>=0.5.0
pydantic>=2.0
```

```python
from nemoguardrails import LLMRails, RailsConfig
from pydantic import BaseModel, Field, validator
import logging

logger = logging.getLogger(__name__)

# === OUTPUT SCHEMA (Structured Responses) ===

class RAGAnswer(BaseModel):
    """Validated RAG output schema."""
    
    answer: str = Field(
        ...,
        description="The direct answer to the user's question",
        min_length=10,
        max_length=2000
    )
    
    confidence: float = Field(
        ...,
        description="Confidence score (0-1)",
        ge=0.0,
        le=1.0
    )
    
    is_answerable: bool = Field(
        ...,
        description="Whether the question could be answered from documents"
    )
    
    sources: list[str] = Field(
        default_factory=list,
        description="List of document sources used"
    )
    
    @validator('answer')
    def answer_not_empty(cls, v):
        if not v or v.lower() == "i don't know":
            raise ValueError("Answer must be substantive")
        return v

# === NEMO GUARDRAILS CONFIG ===

GUARDRAILS_CONFIG = """
define user ask question about documents
    "What is [topic]?"
    "Explain [topic]"
    "Tell me about [topic]"

define user ask out of scope
    "Tell me a joke"
    "Write code for [malicious_thing]"
    "How do I [illegal_activity]?"

define bot respond with disclaimer
    "I can only answer questions about the documents provided."

define bot respond with refusal
    "I cannot help with that request. Please ask about the uploaded documents."

define flow
    user ask out of scope
    bot respond with refusal
    stop

    user ask question about documents
    # Let RAG chain handle it
    pass
"""

class GuardrailsChain:
    def __init__(self):
        """Initialize NeMo Guardrails."""
        config = RailsConfig.from_string(GUARDRAILS_CONFIG)
        self.rails = LLMRails(config)
    
    def validate_input(self, user_query: str) -> tuple[bool, str]:
        """
        Validate user input before processing.
        
        Returns:
            (is_valid, feedback_message)
        """
        # Check query length
        if len(user_query) > 500:
            return False, "Query too long (max 500 chars). Please be more concise."
        
        # Check for prompt injection patterns
        injection_patterns = [
            "ignore", "forget", "disregard",
            "system prompt", "instructions",
            "override", "bypass"
        ]
        
        query_lower = user_query.lower()
        for pattern in injection_patterns:
            if pattern in query_lower:
                logger.warning(f"Potential prompt injection detected: {user_query}")
                return False, "Query appears to contain instructions. Please ask a normal question."
        
        # Out-of-scope check (using NeMo)
        output = self.rails.generate(user_query)
        
        if "cannot help" in output.lower():
            return False, output
        
        return True, ""
    
    def validate_output(self, generated_text: str) -> RAGAnswer:
        """
        Validate and parse LLM output.
        
        Ensures:
        - Output is structured (not random text)
        - Contains required fields
        - Values are reasonable
        
        Returns:
            RAGAnswer object (structured, validated response)
        """
        from langchain_core.output_parsers import PydanticOutputParser
        
        parser = PydanticOutputParser(pydantic_object=RAGAnswer)
        
        # Add structured prompting to LLM
        format_instructions = parser.get_format_instructions()
        
        # Parse and validate
        try:
            answer = parser.parse(generated_text)
            logger.info(f"Output validation passed: confidence={answer.confidence}")
            return answer
        except Exception as e:
            logger.error(f"Output validation failed: {e}")
            
            # Fallback: create basic valid response
            return RAGAnswer(
                answer=generated_text,
                confidence=0.5,
                is_answerable=True,
                sources=[]
            )

# === INTEGRATION WITH RAG CHAIN ===

def get_rag_chain_with_guardrails(vector_db, retriever):
    """
    RAG with input/output validation.
    """
    from langchain_core.prompts import PromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.output_parsers import PydanticOutputParser
    
    guardrails = GuardrailsChain()
    parser = PydanticOutputParser(pydantic_object=RAGAnswer)
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    def protected_rag_chain(question: str) -> RAGAnswer:
        # Step 1: Validate input
        is_valid, feedback = guardrails.validate_input(question)
        if not is_valid:
            return RAGAnswer(
                answer=feedback,
                confidence=1.0,
                is_answerable=False,
                sources=[]
            )
        
        # Step 2: Retrieve
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Step 3: Generate with structured output
        prompt = PromptTemplate(
            template="""Answer the following question based on the context.
            
Context:
{context}

Question: {question}

{format_instructions}

Answer:""",
            input_variables=["context", "question"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()
            }
        )
        
        chain = prompt | llm
        generated = chain.invoke({
            "context": context,
            "question": question
        })
        
        # Step 4: Validate output
        answer = guardrails.validate_output(generated.content)
        
        # Enrich with sources
        answer.sources = [doc.metadata.get("source") for doc in docs]
        
        return answer
    
    return protected_rag_chain
```

**Streamlit Integration:**
```python
# In app.py:

from src.guardrails import get_rag_chain_with_guardrails

# Replace regular RAG chain
rag_chain = get_rag_chain_with_guardrails(vector_db, retriever)

# Main chat loop (modified):
if prompt := st.chat_input("Ask about your documents..."):
    # Chain now returns RAGAnswer object with validation
    response_obj = rag_chain(prompt)
    
    # Check if response is valid
    if not response_obj.is_answerable:
        st.warning(response_obj.answer)
    else:
        st.chat_message("assistant").write(response_obj.answer)
        
        # Show confidence and sources
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Confidence", f"{response_obj.confidence:.0%}")
        with col2:
            st.write(f"**Sources:** {', '.join(response_obj.sources)}")
```

**Key Dependencies:**
- `nemoguardrails>=0.5.0`
- `pydantic>=2.0`

**Expected Outcome:**
- LLM refuses unsafe/out-of-scope requests
- Output always structured and validated
- Confidence scores indicate hallucination risk

---

## 📈 Implementation Timeline & Priority

### Quick Wins (Weeks 1-2)
- **Phase 1:** RAGAS + LangSmith → Immediate observability

### Core Features (Weeks 3-6)
- **Phase 2:** Advanced ingestion → Multimodal support
- **Phase 3A:** PostgreSQL history → Session persistence

### Scale & Reliability (Weeks 7-9)
- **Phase 3B:** Semantic cache → Cost reduction
- **Phase 4:** FastAPI + Prefect → Enterprise readiness
- **Phase 5:** Guardrails → Safety & validation

---

## 🔧 Dependencies Summary

**New packages to add:**
```
# Phase 1: Observability
ragas>=0.1.0
langsmith>=0.0.1
datasets

# Phase 2: Advanced Ingestion
unstructured[pdf,docx]>=0.10.0
pdf2image
pillow
pytesseract

# Phase 3: Persistent Memory
sqlalchemy>=2.0
psycopg2-binary>=2.9
redis>=4.5.0
redisvl>=0.1.0

# Phase 4: Enterprise
prefect>=2.14.0
fastapi>=0.100.0
uvicorn>=0.23.0
python-multipart

# Phase 5: Safety
nemoguardrails>=0.5.0
```

---

## 🏗️ Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     CLIENT LAYER                            │
├─────────────────────────────────────────────────────────────┤
│  Streamlit UI (Webpage)  │  FastAPI (Mobile/External)      │
└────────────┬─────────────┴──────────────┬────────────────────┘
             │                            │
┌────────────▼──────────────────────────▼─────────────────────┐
│              APPLICATION LAYER (Core RAG)                   │
├─────────────────────────────────────────────────────────────┤
│  Guardrails (Input/Output Validation) ──┐                  │
│  ├─ NeMo Guardrails (Safety)            │                  │
│  └─ Pydantic Parsing (Structure)        │                  │
│                                         │                  │
│  RAG Chain ◄─────────────────────────────┘                  │
│  ├─ Retriever (BM25 + Vector + Rerank)                     │
│  ├─ Generator (Gemini 2.5 Flash)                           │
│  └─ Semantic Cache (RedisVL)                               │
│                                         ┌──────────┐        │
│  Metrics & Evaluation ◄──────────────────┤ RAGAS   │        │
│  └─ LangSmith Tracing                   │ + Smith │        │
│                                         └──────────┘        │
└────────────┬──────────────────────────────┬────────────────┘
             │                              │
┌────────────▼───────────────────────────────▼────────────────┐
│            PERSISTENCE LAYER (Data Storage)                 │
├─────────────────────────────────────────────────────────────┤
│  ChromaDB      │  PostgreSQL  │  Redis   │  S3 (Files)     │
│  (Vectors)     │  (History)   │  (Cache) │  (Documents)    │
└──────────────────────────────────────────────────────────────┘
             │
┌────────────▼──────────────────────────────────────────────────┐
│         ORCHESTRATION LAYER (Automation & Deployment)        │
├────────────────────────────────────────────────────────────────┤
│  Prefect (Scheduled Workflows)                              │
│  Docker (Containerization)                                  │
│  Kubernetes (Scaling) [Optional]                            │
└────────────────────────────────────────────────────────────────┘
```

---

## 📋 Updated Folder Structure (After Implementation)

```
rag-system/
├── app.py                 # Streamlit UI (unchanged core logic)
├── requirements.txt       # Updated with new packages
├── .env                   # Add DATABASE_URL, LANGSMITH_API_KEY, REDIS_URL
├── docker-compose.yml     # New: Local dev infrastructure
├── Dockerfile             # New: API containerization
├── PLAN.md               # This file
├── DESIGN.md             # Original architecture
│
├── src/
│   ├── __init__.py
│   ├── ingestion.py      # Enhanced: Multimodal support
│   ├── retrieval.py      # Unchanged (already good)
│   ├── generation.py     # Enhanced: Caching + guarded chains
│   ├── metrics.py        # Enhanced: RAGAS evaluation
│   ├── visualization.py  # Unchanged
│   │
│   ├── cache.py          # New: SemanticCache (RedisVL)
│   ├── database.py       # New: ChatDB (PostgreSQL ORM)
│   ├── guardrails.py     # New: NeMo + Pydantic validation
│   ├── orchestration.py  # New: Prefect workflows
│   └── api.py            # New: FastAPI server
│
├── database/
│   ├── chroma.sqlite3    # Vector DB (existing)
│   └── processed_files.json  # New: File hash registry for auto-sync
│
├── data/
│   └── *.txt, *.pdf      # Document directory
│
└── .azure/              # Optional: IaC files for Azure deployment
    └── plan.copilot.md  # Deployment instructions
```

---

## ✅ Success Metrics

By completing this plan, your portfolio project will demonstrate:

| Metric | Before | After |
|--------|--------|-------|
| **Observability** | ❌ Black-box | ✅ Full tracing + metrics |
| **Scalability** | ⚠️ Single-file | ✅ Containerized + API |
| **Data Handling** | 📄 Text only | ✅ PDFs, tables, web scrape |
| **Memory** | 📌 Stateless | ✅ Persistent + cached |
| **Automation** | 🔧 Manual | ✅ Scheduled workflows |
| **Safety** | ⚠️ Anything goes | ✅ Guarded + validated |
| **Cost Efficiency** | 💸 100% API calls | ✅ 30-40% reduction via cache |
| **Enterprise Readiness** | 🟡 Maybe | ✅ Production-grade |

---

## 🎓 Learning Outcomes

Completing this plan demonstrates:

1. **MLOps Expertise** - Metrics, monitoring, observability
2. **Data Engineering** - ETL pipelines, scheduled workflows
3. **Full-Stack Development** - API design, database modeling, containerization
4. **AI Safety** - Guardrails, validation, hallucination prevention
5. **Cloud Native Patterns** - Scaling, automation, reproducibility

---

## 📚 Reference Documentation

- [RAGAS Docs](https://github.com/explodinggradients/ragas)
- [LangSmith](https://smith.langchain.com/)
- [Prefect](https://docs.prefect.io/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails)
- [Pydantic](https://docs.pydantic.dev/)
- [RedisVL](https://github.com/RedisVentures/redisvl)
- [SQLAlchemy](https://docs.sqlalchemy.org/)

---

## 💡 Pro Tips

1. **Start with Phase 1** — Metrics will guide all other improvements
2. **Use Docker Compose early** — Avoid "works on my machine" problems
3. **Write tests** — Each module should have `test_*.py` files
4. **Document as you go** — Docstrings + README updates
5. **Benchmark before/after** — Quantify improvements (latency, cost, quality)

---

**Last Updated:** March 21, 2026  
**Next Review:** After Phase 1 completion (Week 2)
