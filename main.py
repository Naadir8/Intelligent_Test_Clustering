"""Main entry point — launches the intelligent test case clustering pipeline.

This module parses CLI arguments, configures logging, and orchestrates
the full workflow including data generation, embedding computation,
clustering, and evaluation.
"""

import sys
import argparse
from pathlib import Path

import numpy as np

from src.clustering.clusterer import TestCaseClusterer
from src.data.loader import TestCaseLoader
from src.embedding.embedder import TestCaseEmbedder
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for pipeline configuration.

    Returns:
        argparse.Namespace: Parsed CLI arguments containing:
            - log_level (str): Logging verbosity level.

    Notes:
        - Restricts log levels to standard logging options.
        - Can be extended with additional pipeline parameters.
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

    This function acts as the main orchestration layer and performs:
    1. CLI argument parsing and logger configuration
    2. Synthetic dataset generation
    3. Embedding loading or generation (with caching)
    4. Clustering of embeddings
    5. Evaluation of clustering quality

    Returns:
        None

    Side Effects:
        - Reads/writes embeddings to disk ("data/processed/embeddings.npy")
        - Writes logs to console and log files
        - Prints error messages to stdout on failure
        - Terminates the process with a non-zero exit code on errors

    Raises:
        SystemExit: If a recoverable or critical error occurs

    Notes:
        - Embedding generation uses model "all-MiniLM-L6-v2"
        - Assumes dataset contains "true_cluster" for evaluation
        - Logging is dynamically configured via CLI argument
    """
    args = parse_args()

    # Reconfigure logger with chosen level
    logger = setup_logger(level=args.log_level)

    logger.info("Starting Intelligent Test Case Clustering System")
    logger.info(f"Log level set to: {args.log_level}")

    try:
        logger.info("Step 1: Generating synthetic test cases...")
        loader = TestCaseLoader()
        df = loader.generate_synthetic(n_samples=1000, n_clusters=8)
        logger.info(f"Successfully generated {len(df)} test cases")

        embeddings_path = Path("data/processed/embeddings.npy")

        if embeddings_path.exists():
            logger.info(f"Loading cached embeddings from {embeddings_path}")
            embeddings = np.load(embeddings_path)
        else:
            logger.info("Step 2: Generating new embeddings...")
            embedder = TestCaseEmbedder(model_name="all-MiniLM-L6-v2")
            df = embedder.embed_dataframe(df, text_column="description")

            embeddings = np.array(df["embedding"].tolist())
            np.save(embeddings_path, embeddings)
            logger.info(f"Embeddings saved to {embeddings_path}")

        logger.info(f"Embeddings ready. Shape: {embeddings.shape}")

        logger.info("Step 3: Starting clustering process...")
        clusterer = TestCaseClusterer(n_clusters=8)
        predicted_labels = clusterer.fit_predict(embeddings)
        df["predicted_cluster"] = predicted_labels

        logger.info("Step 4: Evaluating clustering quality...")
        metrics = clusterer.evaluate(
            predicted_labels=predicted_labels,
            true_labels=df["true_cluster"],
            embeddings=embeddings
        )

        logger.info(f"Evaluation metrics: {metrics}")
        logger.info("Pipeline completed successfully")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}", exc_info=True)
        print("ERROR: Required file is missing. Check the data directory.")
        sys.exit(1)

    except Exception:
        logger.critical(
            "Unexpected critical error during pipeline execution",
            exc_info=True
        )
        print("CRITICAL ERROR: An unexpected error occurred. Please check logs/app.log")
        sys.exit(1)


if __name__ == "__main__":
    main()