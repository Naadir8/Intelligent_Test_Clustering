"""Main entry point — launches the entire intelligent test case clustering pipeline.

This module coordinates the full workflow including data generation,
embedding computation (with caching), clustering, and evaluation.
"""

import sys
from pathlib import Path

import numpy as np

from src.clustering.clusterer import TestCaseClusterer
from src.data.loader import TestCaseLoader
from src.embedding.embedder import TestCaseEmbedder
from src.utils.logger import logger


def main() -> None:
    """Execute the full intelligent test case clustering pipeline.

    This function serves as the primary orchestration layer of the system.
    It performs the following sequential steps:
    1. Generate a synthetic dataset of test cases
    2. Load cached embeddings or compute new ones
    3. Cluster embeddings into groups
    4. Evaluate clustering performance using ground truth labels

    The embedding stage is optimized via caching to avoid recomputation
    across multiple runs.

    Returns:
        None

    Side Effects:
        - Reads/writes embeddings to disk ("data/processed/embeddings.npy")
        - Logs progress and errors via the global logger
        - Terminates the process with non-zero exit code on failure

    Raises:
        SystemExit: If a critical error occurs during execution

    Notes:
        - Uses a fixed embedding model ("all-MiniLM-L6-v2")
        - Assumes the dataset contains a "true_cluster" column for evaluation
        - Logging is centrally configured via src.utils.logger
    """
    logger.info("Starting Intelligent Test Case Clustering System")

    try:
        logger.info("Generating synthetic test cases...")
        loader = TestCaseLoader()
        df = loader.generate_synthetic(n_samples=1000, n_clusters=8)
        logger.info(f"Successfully generated {len(df)} test cases")

        embeddings_path = Path("data/processed/embeddings.npy")

        if embeddings_path.exists():
            logger.info(f"Loading cached embeddings from {embeddings_path}")
            embeddings = np.load(embeddings_path)
        else:
            logger.info("No cached embeddings found. Generating new embeddings...")
            embedder = TestCaseEmbedder(model_name="all-MiniLM-L6-v2")
            df = embedder.embed_dataframe(df, text_column="description")

            embeddings = np.array(df["embedding"].tolist())
            np.save(embeddings_path, embeddings)
            logger.info(f"Embeddings successfully saved to {embeddings_path}")

        logger.info(f"Embeddings ready. Shape: {embeddings.shape}")

        logger.info("Starting clustering process...")
        clusterer = TestCaseClusterer(n_clusters=8)
        predicted_labels = clusterer.fit_predict(embeddings)
        df["predicted_cluster"] = predicted_labels

        logger.info("Evaluating clustering quality...")
        metrics = clusterer.evaluate(
            predicted_labels=predicted_labels,
            true_labels=df["true_cluster"],
            embeddings=embeddings
        )

        logger.info("Pipeline completed successfully")

    except FileNotFoundError as e:
        error_id = log_error("file_not_found", exc_info=True, extra={"path": str(e)})
        print(f"\nERROR [{error_id}]: Required file is missing. Check the data directory.")
        sys.exit(1)

    except Exception as e:
        error_id = log_error("critical_error", exc_info=True)
        print(f"\nCRITICAL ERROR [{error_id}]: An unexpected error occurred.")
        print("Please check the log file (logs/app.log) for detailed information.")
        sys.exit(1)

if __name__ == "__main__":
    main()