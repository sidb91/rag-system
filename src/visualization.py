import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def visualize_embeddings(vector_db):
    data = vector_db.get(include=['embeddings', 'documents'])
    vectors = np.array(data['embeddings'])
    
    # Reduce dimensions from 768 to 2
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vectors)
    
    fig, ax = plt.subplots()
    ax.scatter(reduced[:, 0], reduced[:, 1], c='blue', alpha=0.5)
    ax.set_title("Semantic Clusters of Your Documents")
    return fig