"""Provide utilities for generating and loading test case datasets.

This module is responsible for:
- creating synthetic test case data with predefined semantic clusters,
- organizing project data directories,
- persisting generated datasets to disk for reuse.
"""

import pandas as pd
import numpy as np
from pathlib import Path


class TestCaseLoader:
    """Handle dataset generation and loading operations.

    This class encapsulates:
    - directory structure initialization,
    - synthetic dataset generation with labeled clusters,
    - loading previously generated datasets from disk.

    Attributes:
        data_dir (Path): Root directory for all data.
        raw_dir (Path): Directory for raw datasets.
        processed_dir (Path): Directory for processed artifacts.
    """

    def __init__(self, data_dir: str = "data"):
        """Initialize data directories and ensure they exist.

        Args:
            data_dir (str): Base directory for storing project data.
        """
        # Root data directory
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Directory for raw (original) datasets
        self.raw_dir = self.data_dir / "raw"
        self.raw_dir.mkdir(exist_ok=True)

        # Directory for processed data (e.g., embeddings)
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(exist_ok=True)

    def generate_synthetic(
        self,
        n_samples: int = 1000,
        n_clusters: int = 8,
        seed: int = 42
    ) -> pd.DataFrame:
        """Generate a synthetic dataset of test cases with semantic clusters.

        Each cluster represents a functional domain (e.g., Authentication, API),
        and contains multiple variations of test case descriptions.

        Args:
            n_samples (int): Total number of test cases to generate.
            n_clusters (int): Number of clusters (domains) to include.
            seed (int): Random seed for reproducibility.

        Returns:
            pd.DataFrame: Generated dataset with columns:
                - id: unique test case identifier
                - title: short test case title
                - description: detailed description
                - true_cluster: ground-truth cluster label
        """
        # Ensure reproducibility of random generation
        np.random.seed(seed)

        # Predefined templates describing typical test case actions per domain
        cluster_templates = {
            "Authentication": ["login", "registration", "password recovery", "two-factor authentication"],
            "Database": ["record creation", "update", "deletion", "search", "aggregation"],
            "UI/UX": ["button click", "form validation", "navigation", "responsive design"],
            "API": ["GET request", "POST creation", "PUT update", "DELETE", "rate limiting"],
            "Payment": ["card payment", "refund", "3D Secure", "payment error handling"],
            "Reporting": ["report generation", "PDF export", "filtering", "dashboard"],
            "Integration": ["external API integration", "webhook", "data synchronization"],
            "Performance": ["load testing", "stress test", "response time"],
        }

        # Select only the required number of clusters
        clusters = list(cluster_templates.keys())[:n_clusters]

        data = []

        # Evenly distribute samples across clusters
        samples_per_cluster = n_samples // n_clusters

        for cluster_name in clusters:
            templates = cluster_templates[cluster_name]

            for i in range(samples_per_cluster):
                # Randomly select a base action template
                base = np.random.choice(templates)

                # Construct structured test case title
                title = f"TC-{cluster_name[:3].upper()}-{i+1:04d}: Verify {base}"

                # Construct natural language description
                desc = (
                    f"User performs {base} in the {cluster_name.lower()} module. "
                    f"Checks correctness, error handling and requirement compliance."
                )

                # Periodically inject edge-case scenarios for diversity
                if i % 5 == 0:
                    desc += " Additional edge-case verification."

                # Append structured test case record
                data.append({
                    "id": f"TC-{len(data)+1:05d}",
                    "title": title,
                    "description": desc,
                    "true_cluster": cluster_name
                })

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Persist dataset to disk for reuse
        output_path = self.raw_dir / "synthetic_test_cases.csv"
        df.to_csv(output_path, index=False, encoding="utf-8")

        print(f"Generated {len(df)} test cases → saved to {output_path}")

        return df

    def load(self) -> pd.DataFrame:
        """Load previously generated dataset from disk.

        Returns:
            pd.DataFrame: Loaded dataset.

        Raises:
            FileNotFoundError: If dataset file does not exist.
        """
        path = self.raw_dir / "synthetic_test_cases.csv"

        # Validate dataset existence before loading
        if path.exists():
            return pd.read_csv(path)

        # Explicit failure with clear instruction
        raise FileNotFoundError("Run generate_synthetic() first or check path")