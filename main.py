"""Main entry point — launches the intelligent test case clustering pipeline.

This module orchestrates the full workflow including data generation,
embedding computation, clustering, evaluation, and structured error handling.
"""

import sys
import argparse
from pathlib import Path

import numpy as np

from src.clustering.clusterer import TestCaseClusterer
from src.data.loader import TestCaseLoader
from src.embedding.embedder import TestCaseEmbedder
from src.utils.logger import logger, log_error


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed CLI arguments containing:
            - log_level (str): Logging verbosity level.

    Notes:
        - Currently supports only logging configuration.
        - Can be extended to control pipeline parameters.
    """
    parser = argparse.ArgumentParser(
        description="Intelligent Test Case Clustering System"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level"
    )
    return parser.parse_args()


def main() -> None:
    """Execute the full intelligent test case clustering pipeline.

    This function coordinates the complete workflow:
    1. CLI argument parsing
    2. Synthetic dataset generation
    3. Embedding loading or computation (with caching)
    4. Clustering of embeddings
    5. Evaluation of clustering quality

    It also implements structured error handling using unique error IDs.

    Returns:
        None

    Side Effects:
        - Reads/writes embeddings to disk ("data/processed/embeddings.npy")
        - Writes logs to console and log file
        - Prints user-friendly error messages to stdout
        - Terminates the process with a non-zero exit code on failure

    Raises:
        SystemExit: If a critical error occurs during execution

    Notes:
        - Embedding model: "all-MiniLM-L6-v2"
        - Requires "true_cluster" column for evaluation
        - Errors are logged with unique IDs for traceability
    """
    args = parse_args()
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

        logger.info(f"Evaluation metrics: {metrics}")
        logger.info("Pipeline completed successfully")

    except FileNotFoundError as e:
        error_id = log_error(
            "Required file not found",
            exc_info=True,
            extra={"path": str(e)}
        )
        print(f"\nERROR [{error_id}]: Critical file is missing. Check data directory.")
        sys.exit(1)

    except Exception:
        error_id = log_error(
            "Unexpected error during pipeline execution",
            exc_info=True
        )
        print(f"\nCRITICAL ERROR [{error_id}]: An unexpected error occurred.")
        print("Please check the log file (logs/app.log) for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()