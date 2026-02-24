"""Visualization Module - Visualize document embeddings in 2D space.

This module provides tools to visualize the semantic structure of your document
collection. Since embeddings are 768-dimensional vectors, we use PCA to reduce
to 2D for visualization while preserving as much variance as possible.

You can see:
- How documents cluster semantically
- Which topics are similar/distinct
- Coverage of your knowledge base
"""

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def visualize_embeddings(vector_db):
    """Visualize document embeddings in 2D using PCA dimensionality reduction.
    
    This function:
    1. Extracts all embeddingsfrom the vector database
    2. Uses PCA (Principal Component Analysis) to reduce 768D → 2D
    3. Creates a scatter plot showing documents as points
    4. Points close together = semantically similar documents
    5. Points far apart = semantically distinct documents
    
    WHY PCA?
    - Preserves global structure (important distances stay preserved)
    - Fast and interpretable
    - Reduces 768 dimensions to 2 while keeping maximum variance
    - Alternative: t-SNE (better local structure, slower), UMAP (good balance)
    
    Args:
        vector_db: A Chroma vector store object (from ingestion.py) containing
                   embeddings and document text.
    
    Returns:
        matplotlib.figure.Figure: A matplotlib figure object that can be:
        - Displayed in Streamlit with st.pyplot(fig)
        - Saved with fig.savefig("path.png")
        - Further customized with matplotlib API
    
    Example:
        >>> from src.ingestion import process_documents
        >>> vector_db = process_documents()
        >>> fig = visualize_embeddings(vector_db)
        >>> fig.show()  # or st.pyplot(fig) in Streamlit
    
    Interpretation:
    - Tight clusters: Documents about the same topic
    - Scattered points: Diverse, unrelated documents
    - Large empty areas: Gaps in your knowledge base coverage
    - Outliers: Documents very different from main corpus
    """
    
    # === STEP 1: EXTRACT EMBEDDINGS FROM VECTOR DATABASE ===
    # Chroma.get() retrieves stored documents
    # include=['embeddings', 'documents']: Only fetch what we need
    # (exclude metadata to reduce memory usage for visualization)
    data = vector_db.get(include=['embeddings', 'documents'])
    
    # Convert embeddings list to numpy array for PCA
    # Shape: (num_documents, 768)
    # Each row is one document's embedding (768 dimensions from Gemini)
    vectors = np.array(data['embeddings'])
    
    # === STEP 2: DIMENSIONALITY REDUCTION WITH PCA ===
    # PCA analysis:
    # - Input: 768-dimensional vectors (very sparse representation)
    # - Output: 2-dimensional vectors (can be plotted)
    # - How it works:
    #   1. Computes principal components (directions of max variance)
    #   2. Projects high-dimensional points onto first 2 components
    #   3. Preserves relative distances (similar docs stay similar)
    #
    # Why not just pick 2 random dimensions?
    # - PCA selects the 2 dimensions with MOST variance
    # - Gives best visual representation of document relationships
    # - Maximum information retention with minimum dimensions
    pca = PCA(n_components=2)
    
    # Fit PCA to data and transform all vectors
    # fit_transform learns the components and applies them in one step
    # Result shape: (num_documents, 2)
    reduced = pca.fit_transform(vectors)
    
    # === STEP 3: CREATE VISUALIZATION ===
    # Create matplotlib figure for plotting
    fig, ax = plt.subplots()
    
    # Scatter plot of reduced embeddings
    # - reduced[:, 0]: First PCA component (x-axis)
    # - reduced[:, 1]: Second PCA component (y-axis)
    # - c='blue': All points blue (could color by topic/metadata)
    # - alpha=0.5: Semi-transparent to see overlapping points
    ax.scatter(reduced[:, 0], reduced[:, 1], c='blue', alpha=0.5)
    
    # Add title to help interpret the visualization
    ax.set_title("Semantic Clusters of Your Documents")
    
    # Return figure for display in Streamlit or other UI
    return fig