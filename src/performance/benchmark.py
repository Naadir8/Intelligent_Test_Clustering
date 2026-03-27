"""Performance benchmarking and profiling module for the Intelligent Test Case Clustering system.

This module provides standardized benchmarks for measuring CPU time and memory usage
across the full pipeline: dataset generation, embedding computation and clustering.
"""

import cProfile
import pstats
import time
from typing import Tuple

import numpy as np
import pandas as pd
from memory_profiler import profile as memory_profile

from src.data.loader import TestCaseLoader
from src.embedding.embedder import TestCaseEmbedder
from src.clustering.clusterer import TestCaseClusterer


def run_benchmark(n_samples: int = 3000, n_clusters: int = 8) -> Tuple[pd.DataFrame, np.ndarray, float]:
    """Run the full pipeline benchmark and measure total execution time.

    Args:
        n_samples: Number of synthetic test cases to generate.
        n_clusters: Number of clusters for KMeans.

    Returns:
        Tuple of (DataFrame, embeddings array, total execution time in seconds).
    """
    start_time = time.perf_counter()

    # 1. Generate synthetic dataset
    loader = TestCaseLoader()
    df = loader.generate_synthetic(n_samples=n_samples, n_clusters=n_clusters)

    # 2. Generate embeddings
    embedder = TestCaseEmbedder(model_name="all-MiniLM-L6-v2")
    df = embedder.embed_dataframe(df, text_column="description", batch_size=128)

    embeddings = np.array(df["embedding"].tolist())

    # 3. Perform clustering
    clusterer = TestCaseClusterer(n_clusters=n_clusters)
    _ = clusterer.fit_predict(embeddings)

    total_time = time.perf_counter() - start_time
    return df, embeddings, total_time


@memory_profile
def run_memory_profiled_embedding(n_samples: int = 2000):
    """Memory-profiled version of the embedding generation stage."""
    loader = TestCaseLoader()
    df = loader.generate_synthetic(n_samples=n_samples, n_clusters=8)

    embedder = TestCaseEmbedder(model_name="all-MiniLM-L6-v2")
    df = embedder.embed_dataframe(df, text_column="description", batch_size=128)
    return df


def profile_with_cprofile(output_file: str = "profile_results.prof", n_samples: int = 3000):
    """Run cProfile on the full benchmark and save results to file."""
    profiler = cProfile.Profile()
    profiler.enable()

    run_benchmark(n_samples=n_samples)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.dump_stats(output_file)
    print(f"Profile results saved to {output_file}")
    stats.print_stats(15)  # Show top 15 hot functions