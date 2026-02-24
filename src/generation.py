"""RAG Generation Module - Combines retrieved context with LLM generation.

This module handles the 'Generation' step in the RAG (Retrieval-Augmented Generation) pipeline.
It takes a retriever (which fetches relevant documents) and chains it with an LLM to generate
contextually-aware answers based on the retrieved documents.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def get_rag_chain(retriever):
    """Build a complete RAG (Retrieval-Augmented Generation) chain.
    
    This function constructs a pipeline that:
    1. Takes a user question
    2. Retrieves relevant document chunks using the passed retriever
    3. Formats them into a prompt with the question
    4. Sends it to Google's Gemini LLM for answer generation
    5. Parses the LLM output into clean text
    
    The chain uses LangChain's pipe (|) operator to connect components in a sequence,
    making it composable and easy to extend.
    
    Args:
        retriever: A LangChain retriever object (e.g., from retrieval.py) that
                   accepts a question and returns relevant Document objects.
    
    Returns:
        A LangChain Runnable chain that takes a question string and returns
        an answer string based on retrieved context.
    
    Example:
        >>> from src.retrieval import get_advanced_retriever
        >>> retriever = get_advanced_retriever(vector_db, chunks)
        >>> chain = get_rag_chain(retriever)
        >>> answer = chain.invoke("What is machine learning?")
    """
    
    # Initialize the LLM (Large Language Model) with specific configuration
    # - model="gemini-2.5-flash": Uses Google's latest fast model optimized for speed
    # - temperature=0: Deterministic output (no randomness) for consistent, factual answers
    # - max_output_tokens=2048: Limits response length (~1000 words)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0,
        max_output_tokens=2048
    )
    
    # Define the prompt template that guides the LLM's behavior
    # - {context}: Will be filled with retrieved document chunks
    # - {question}: Will be filled with the user's actual question
    # - Instructions: Prevent hallucination by telling LLM to say "I don't know"
    template = """Use the following pieces of context to answer the question. 
    If you don't know the answer, just say you don't know, don't make it up.
    Context: {context}
    Question: {question}
    Answer:"""
    
    # Convert the template string into a PromptTemplate object
    prompt = PromptTemplate.from_template(template)
    
    # Build the RAG chain using LangChain's pipe syntax
    # This creates a composable pipeline:
    # Input (question) → Retrieve context → Format with prompt → LLM → Parse output
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        # ^ Creates a dict: "context" comes from retriever, "question" is passed through as-is
        | prompt
        # ^ Formats the dict into the prompt template
        | llm
        # ^ Sends the formatted prompt to the LLM for generation
        | StrOutputParser()
        # ^ Extracts the text from the LLM response (removes metadata)
    )
    return chain