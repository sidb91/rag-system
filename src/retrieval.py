from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
)
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_core.documents import Document

def get_advanced_retriever(vector_db, chunks: None):

    # 1. If chunks aren't passed, extract them from the Vector DB
    if chunks is None:
        # Get raw data from Chroma
        stored_data = vector_db.get(include=['documents', 'metadatas'])
        
        # Convert raw data back into LangChain Document objects
        # This fixes the 'str' object has no attribute 'page_content' error
        chunks = [
            Document(page_content=doc, metadata=meta) 
            for doc, meta in zip(stored_data['documents'], stored_data['metadatas'])
        ]
    
    # Keyword search (BM25)
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 5
    
    # Semantic search (Vector)
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    
    # Hybrid Combination
    ensemble = EnsembleRetriever(retrievers=[bm25, vector_retriever], weights=[0.4, 0.6])
    
    # Re-ranking (Accuracy Boost)
    reranker = FlashrankRerank()
    return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=ensemble)