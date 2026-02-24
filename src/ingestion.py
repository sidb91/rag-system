"""Document Ingestion Module - Loads, chunks, embeds, and stores documents.

This module handles the 'Data Preparation' step in the RAG pipeline. It:
1. Loads documents from multiple formats (TXT, PDF)
2. Splits them into manageable chunks for embedding and retrieval
3. Creates vector embeddings using Google's embedding model
4. Stores them in a persistent vector database (Chroma)
"""

from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma


def process_documents(data_path="./data", persist_dir="./database"):
    """Load documents, chunk them, embed them, and store in a vector database.
    
    This function performs the complete data ingestion pipeline:
    
    1. DOCUMENT LOADING: Reads .txt and .pdf files from the data directory
    2. CHUNKING: Splits documents into overlapping 1000-char chunks to:
       - Keep semantic meaning (context window)
       - Enable better retrieval (relevant chunks instead of whole docs)
       - Improve embedding quality (smaller units embed better)
    3. EMBEDDING: Converts text to 768-dimensional vectors using Google's model
    4. STORAGE: Saves to persistent Chroma database for later retrieval
    
    Args:
        data_path (str): Directory path containing documents. Supports recursive
                        search (glob="**/*") for nested folders.
                        Default: "./data"
        persist_dir (str): Directory where the Chroma vector database will be saved.
                          Default: "./database"
    
    Returns:
        Chroma: A LangChain vector store object that can be used for retrieval.
                Persist to disk so you can reload it without re-processing.
    
    Example:
        >>> vector_db = process_documents("./my_docs", "./my_vectors")
        >>> retriever = vector_db.as_retriever()
    
    Note:
        - Files are loaded with glob patterns: "**/*.txt" and "**/*.pdf"
        - Chunk size of 1000 chars with 100 char overlap balances semantic coherence
          with granularity for RAG
        - Requires GOOGLE_API_KEY environment variable set
    """
    
    # === STEP 1: LOAD DOCUMENTS ===
    # Use DirectoryLoader for recursive file discovery with glob patterns
    # This is more efficient than manually iterating through file systems
    
    # TextLoader for plain text files (.txt)
    # Recursively finds all .txt files in data_path and subdirectories
    text_loader = DirectoryLoader(data_path, glob="**/*.txt", loader_cls=TextLoader)
    
    # PyPDFLoader for PDF files (.pdf)
    # PDFs are more complex but PyPDFLoader extracts text automatically
    pdf_loader = DirectoryLoader(data_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    
    # Load documents from both sources and combine them
    # Each document is a LangChain Document object with:
    # - page_content: The actual text
    # - metadata: File source, page number (for PDFs), etc.
    docs = text_loader.load() + pdf_loader.load()
    
    # === STEP 2: SMART CHUNKING (Recursive Character Splitter) ===
    # Why recursive chunking?
    # - Tries to split on sentences first (better semantic coherence)
    # - If sentences too long, splits on paragraphs
    # - If paragraphs still too long, splits on words/chars
    # - This preserves meaning better than naive char splitting
    
    # chunk_size=1000: Each chunk ~250-300 words (typical document chunk size)
    # chunk_overlap=100: Consecutive chunks overlap by 100 chars to maintain
    #                    context continuity at chunk boundaries
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    
    # === STEP 3: EMBED DOCUMENTS ===
    # Google's embedding model converts text → vector (768 dimensions)
    # These vectors capture semantic meaning:
    # - Similar text → similar vectors (proximity in embedding space)
    # - Used for semantic search (find chunks by meaning, not keywords)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # === STEP 4: CREATE AND PERSIST VECTOR DATABASE ===
    # Chroma is an open-source vector database that:
    # - Stores embeddings with original text and metadata
    # - Allows fast similarity search (cosine distance in vector space)
    # - Persists to disk for future retrieval without re-embedding
    vector_db = Chroma.from_documents(
        documents=chunks,              # Original text chunks
        embedding=embeddings,          # Embedding model
        persist_directory=persist_dir  # Save location for persistence
    )
    
    return vector_db