import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import valohai
import pandas as pd
import argparse
import os
import shutil
import glob


def parse_args():
    parser = argparse.ArgumentParser(description='Protein Similarity Visualization using PCA and Cosine Similarity.')
    parser.add_argument('--query_idx', type=int, default=0, help='Index of the query protein in embeddings.')
    parser.add_argument('--topk', type=int, default=5, help='Number of most similar proteins to find.')
    return parser.parse_args()


def find_topk_similar(X: np.ndarray, output_path: str, query_idx: int = 0, topk: int = 5):
    """
    Find the top-k most similar proteins to a given query using cosine similarity,
    and save results as a CSV.

    Args:
        X (ndarray): Embeddings array of shape [N, D].
        query_idx (int): Index of the query protein in X.
        topk (int): Number of most similar proteins to return.
        output_path (str): Path to save the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing indices and similarity scores.
    """
    query = X[query_idx:query_idx+1]
    sims = cosine_similarity(query, X)[0]
    top_indices = sims.argsort()[::-1][:topk]
    
    results = [(i, float(sims[i])) for i in top_indices]
    df = pd.DataFrame(results, columns=["protein_index", "similarity"])
    
    df.to_csv(output_path, index=False)
    print(f"\nTop-{topk} similar proteins to protein {query_idx}:")
    print(df)
    print(f"\nSaved results -> {output_path}")
    
    return df

def plot_embeddings_pca(X: np.ndarray, output_path: str, query_idx: int = 0):
    """
    Create a 2D PCA scatter plot of protein embeddings.

    Args:
        X (ndarray): Embeddings array of shape [N, D].
        query_idx (int): Index of the query protein to highlight.
        output_path (str): Path to save the output PNG file.
    """
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)

    plt.figure(figsize=(7, 6))
    plt.scatter(X2[:, 0], X2[:, 1], s=20, alpha=0.6, label="Proteins")
    plt.scatter(X2[query_idx, 0], X2[query_idx, 1],
                c="red", s=80, marker="x", label=f"Query {query_idx}")
    plt.title("Protein embeddings (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nSaved PCA plot -> {output_path}")



if __name__ == "__main__":
    args = parse_args()

    data_archive = valohai.inputs('embeddings').path(process_archives=False)
    
    extract_dir = os.path.join(os.path.dirname(data_archive), "extracted_data")

    # Unzip the dataset
    shutil.unpack_archive(data_archive, extract_dir, format='zip')
    print(f"Dataset extracted to: {extract_dir}")

    # Find and load the .pt file
    pt_files = glob.glob(os.path.join(extract_dir, "*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {extract_dir}")
    if len(pt_files) > 1:
        print(f"Warning: Multiple .pt files found, using the first one: {pt_files[0]}")
    
    pt_file = pt_files[0]
    print(f"Loading embeddings from: {os.path.basename(pt_file)}")
    pt = torch.load(pt_file, map_location="cpu")
    embs = pt["embeddings"]   # [N, D] or [N, L, D]

    # Pool if per-residue embeddings
    if embs.ndim == 3:
        embs = embs.mean(dim=1)   # [N, D]

    X = embs.numpy()
    print("Embeddings:", X.shape)

    topk_similarities_path = valohai.outputs("my-output").path("topk_similarities.csv")

    # Find top-k similar proteins and save to CSV
    find_topk_similar(X, output_path=topk_similarities_path, query_idx=args.query_idx, topk=args.topk)

    pca_plot_path = valohai.outputs("my-output").path("embeddings_pca.png")

    # Create and save PCA plot
    plot_embeddings_pca(X, output_path=pca_plot_path, query_idx=args.query_idx)