"""Provide clustering and evaluation utilities for embedding vectors.

This module is responsible for:
- grouping embeddings into clusters using unsupervised algorithms,
- evaluating clustering quality using standard metrics,
- supporting comparison with ground-truth labels (if available).
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


class TestCaseClusterer:
    """Performs clustering on embeddings and evaluates quality.

    By default uses KMeans clustering and computes both external (ARI, NMI)
    and internal (Silhouette) metrics.

    Attributes:
        n_clusters (int): Number of clusters.
        random_state (int): Seed for reproducibility.
        model (KMeans): Fitted clustering model.
    """

    def __init__(self, n_clusters: int = 8, random_state: int = 42) -> None:
        """Initialize clustering configuration.

        Args:
            n_clusters (int): Number of clusters for KMeans.
            random_state (int): Random seed for deterministic results.
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit KMeans model and assign cluster labels.

        Args:
            embeddings (np.ndarray): Input feature matrix of shape
                (n_samples, embedding_dim).

        Returns:
            np.ndarray: Predicted cluster labels for each sample.
        """
        print(f"Running KMeans with {self.n_clusters} clusters ...")

        # Initialize KMeans clustering model
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,  # multiple initializations to improve stability
        )

        # Fit model and assign cluster labels
        labels = self.model.fit_predict(embeddings)

        print(f"Predicted {len(set(labels))} clusters")

        return labels

    @staticmethod
    def evaluate(
        predicted_labels: np.ndarray, true_labels: pd.Series | np.ndarray, embeddings: np.ndarray
    ) -> dict:
        """Evaluate clustering quality using multiple metrics.

        Computes external metrics (ARI, NMI) against ground truth and
        internal metric (Silhouette Score) based on embedding distances.

        Args:
            predicted_labels: Cluster labels predicted by the model.
            true_labels: Ground-truth cluster labels.
            embeddings: Feature matrix used for clustering.

        Returns:
            dict: Dictionary with metric names and values:
                - Adjusted Rand Index (ARI)
                - Normalized Mutual Information (NMI)
                - Silhouette Score

        Raises:
            ValueError: If labels have different lengths or too few clusters.

        Example:
            >>> metrics = clusterer.evaluate(labels, df["true_cluster"], emb)
            >>> print(metrics["Adjusted Rand Index (ARI)"])
        """
        # Convert categorical labels to numeric codes if needed
        if isinstance(true_labels, pd.Series):
            true_labels = true_labels.astype("category").cat.codes.values

        # External evaluation: compare predicted clusters with ground truth
        ari = adjusted_rand_score(true_labels, predicted_labels)
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)

        # Internal evaluation: measure cluster cohesion and separation
        sil = (
            silhouette_score(embeddings, predicted_labels)
            if len(set(predicted_labels)) > 1
            else -1.0  # undefined for a single cluster
        )

        # Aggregate metrics into a structured dictionary
        metrics = {
            "Adjusted Rand Index (ARI)": ari,
            "Normalized Mutual Information (NMI)": nmi,
            "Silhouette Score": sil,
        }

        print("\nClustering quality metrics:")

        # Pretty-print metrics for quick inspection
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")

        return metrics
