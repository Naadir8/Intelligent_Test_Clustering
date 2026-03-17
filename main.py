"""Main entry point — launches the entire pipeline."""

import pandas as pd
import numpy as np
from pathlib import Path

from src.data.loader import TestCaseLoader
from src.embedding.embedder import TestCaseEmbedder
from src.clustering.clusterer import TestCaseClusterer


def main():
    print("=== Intelligent Test Case Clustering System ===\n")

    # Run data generation or loading.
    loader = TestCaseLoader()
    df = loader.generate_synthetic(n_samples=1000, n_clusters=8)

    print("\nDataset preview:")
    print(df.head(3))
    print("\nTrue cluster distribution:")
    print(df["true_cluster"].value_counts().sort_index())

    # Load embeddings if exist, otherwise generate and save
    embeddings_path = Path("data/processed/embeddings.npy")

    if embeddings_path.exists():
        print(f"Завантажуємо готові embeddings з {embeddings_path}")
        embeddings = np.load(embeddings_path)
        df["embedding"] = list(embeddings)
    else:
        print("Embeddings не знайдено → генеруємо заново")
        embedder = TestCaseEmbedder(model_name='all-MiniLM-L6-v2')
        df = embedder.embed_dataframe(df, text_column="description")

        embeddings = np.array(df["embedding"].tolist())
        np.save(embeddings_path, embeddings)
        print(f"Embeddings збережено: {embeddings_path}")

    print(f"\nEmbeddings готові. Shape: {embeddings.shape}")

    # Perform clustering and evaluate results.
    clusterer = TestCaseClusterer(n_clusters=8)
    predicted_labels = clusterer.fit_predict(embeddings)
    df["predicted_cluster"] = predicted_labels

    print("\nPredicted cluster distribution:")
    print(pd.Series(predicted_labels).value_counts().sort_index())

    # Evaluate clustering performance.
    clusterer.evaluate(
        predicted_labels=predicted_labels,
        true_labels=df["true_cluster"],
        embeddings=embeddings
    )

    print("\nDone! You can now analyze df['predicted_cluster'] vs df['true_cluster']")


if __name__ == "__main__":
    main()