"""Advanced Retrieval Module - Hybrid search with re-ranking for RAG.

This module implements a sophisticated multi-stage retrieval system:
1. HYBRID SEARCH: Combines keyword-based (BM25) and semantic (vector) search
2. ENSEMBLE: Weights both approaches to get diverse, relevant results
3. RE-RANKING: Uses Flashrank to reorder results by relevance

This approach gets better results than semantic search alone because:
- BM25 (TF-IDF) excels at finding exact keyword matches
- Vector search excels at semantic similarity
- Re-ranking ensures the most relevant results rank highest
"""

from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
)
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_core.documents import Document


def get_advanced_retriever(vector_db, chunks: None):
    """Build a hybrid retriever combining keyword and semantic search with re-ranking.
    
    This function creates an advanced retrieval pipeline:
    
    STAGE 1 - DUAL RETRIEVAL:
      - BM25: Fast keyword search (TF-IDF scoring) → Top 5 results
      - Vector: Semantic search (embedding similarity) → Top 5 results
      - Ensemble: Combines both with weights (40% keyword, 60% semantic)
      
    STAGE 2 - RE-RANKING:
      - Flashrank: ML-based re-ranker scores combined results
      - Returns final ranked list with most relevant at top
    
    WHY THIS APPROACH?
    - BM25 finds exact keyword matches (good for factual questions)
    - Vector search finds semantic equivalents (good for paraphrased questions)
    - Together they cover more use cases than either alone
    - Re-ranking ensures relevance is optimized
    
    Args:
        vector_db: A Chroma vector store object (from ingestion.py).
        chunks (list | None): Pre-extracted Document chunks. If None, extracts from vector_db.
                             Optional optimization to avoid re-extracting chunks.
    
    Returns:
        ContextualCompressionRetriever: A LangChain retriever that takes a query and
                                        returns re-ranked Document objects.
    
    Example:
        >>> from src.ingestion import process_documents
        >>> from src.retrieval import get_advanced_retriever
        >>> vector_db = process_documents()
        >>> retriever = get_advanced_retriever(vector_db)
        >>> results = retriever.invoke("What is RAG?")
        >>> for doc in results:
        ...     print(doc.page_content)
    
    Notes:
        - k=5: Returns top 5 results from each retriever before re-ranking
        - weights=[0.4, 0.6]: 40% BM25, 60% semantic (experiments suggest this balance)
        - Flashrank uses a pre-trained cross-encoder model for accurate re-ranking
    """

    # === STEP 1: PREPARE DOCUMENT CHUNKS ===
    # If chunks aren't provided, extract them from the vector database
    # This is necessary because Chroma.get() returns raw dictionaries,
    # but BM25Retriever needs LangChain Document objects
    if chunks is None:
        # Get raw data from Chroma
        # include=['documents', 'metadatas']: Only fetch text and metadata, not embeddings
        # (Embeddings are large and not needed for BM25)
        stored_data = vector_db.get(include=['documents', 'metadatas'])
        
        # Reconstruct Document objects from raw data
        # BM25Retriever.from_documents() requires Document objects with:
        # - page_content: The actual text to search
        # - metadata: Additional info (source, page number, etc.)
        chunks = [
            Document(page_content=doc, metadata=meta) 
            for doc, meta in zip(stored_data['documents'], stored_data['metadatas'])
        ]
    
    # === STEP 2: KEYWORD-BASED RETRIEVAL (BM25) ===
    # BM25 (Best Matching 25) is a probabilistic ranking function
    # - Used in traditional search engines before neural models
    # - Ranks documents by term frequency and inverse document frequency
    # - ADVANTAGES: Fast, doesn't require embeddings, great for rare keywords
    # - DISADVANTAGES: Misses semantic similarity
    # 
    # Example: Query "What is machine learning?"
    # - Strong match: "Machine learning is a subset of AI" (contains exact terms)
    # - Weak match: "Neural networks enable AI systems" (semantic but no keywords)
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 5  # Retrieve top 5 results
    
    # === STEP 3: SEMANTIC RETRIEVAL (VECTOR SEARCH) ===
    # Vector/embedding-based search finds semantically similar documents
    # - Converts query to vector, finds nearest neighbors in embedding space
    # - ADVANTAGES: Catches paraphrased content, semantic relationships
    # - DISADVANTAGES: Slower than BM25, requires embeddings
    # 
    # Example: Query "What is machine learning?"
    # - Strong match: "Automated pattern recognition and prediction" (semantic)
    # - Weak match: "Machine learning is XYZ" (too literal, if phrased differently)
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    
    # === STEP 4: HYBRID ENSEMBLE ===
    # EnsembleRetriever combines multiple retrievers
    # - Merges results from both BM25 and vector search
    # - weights=[0.4, 0.6]: Gives 40% importance to keyword, 60% to semantic
    # - This weighted combination works better than equal weights:
    #   * Semantic search is generally more effective for RAG
    #   * But keyword search catches important edge cases
    #   * Balance empirically chosen for most domains
    ensemble = EnsembleRetriever(
        retrievers=[bm25, vector_retriever],  # Both retrieval methods
        weights=[0.4, 0.6]                      # BM25: 40%, Vector: 60%
    )
    
    # === STEP 5: RE-RANKING WITH FLASHRANK ===
    # Flashrank uses a pre-trained cross-encoder (BERT-based) to re-rank
    # - Treats ranking as a classification problem: how relevant is this doc to query?
    # - More accurate than similarity scores but slower
    # - Typically improves precision by 10-20%
    # 
    # How it works:
    # 1. Gets candidate documents from ensemble (e.g., 10 results)
    # 2. Scores each document relative to query using cross-encoder
    # 3. Returns re-ordered list with most relevant on top
    # 
    # Example ranking change:
    # BEFORE re-rank: [Medium relevance, High relevance, Medium relevance]
    # AFTER re-rank:  [High relevance, Medium relevance, Medium relevance]
    reranker = FlashrankRerank()
    
    # === FINAL PIPELINE ===
    # ContextualCompressionRetriever applies re-ranking to ensemble results
    # Flow: Query → Ensemble (BM25 + Vector) → Flashrank → Final Results
    return ContextualCompressionRetriever(
        base_compressor=reranker,      # Re-ranking model
        base_retriever=ensemble        # Hybrid retriever
    )