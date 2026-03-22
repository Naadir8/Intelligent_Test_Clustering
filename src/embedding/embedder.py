"""Provide utilities for generating sentence embeddings from test case texts.

This module handles:
- loading and configuring a transformer-based embedding model,
- converting text data into dense vector representations,
- integrating embeddings into pandas DataFrames,
- optionally persisting embeddings for reuse.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


class TestCaseEmbedder:
    """Generates dense vector embeddings for test case texts.

        Uses a pretrained SentenceTransformer model to convert natural language
        descriptions into fixed-size vectors suitable for clustering.

        Attributes:
            model (SentenceTransformer): Loaded embedding model.
            device (str): Device used for inference ('cpu' or 'cuda').
        """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize embedding model and configure execution device.

        Args:
            model_name (str): Pretrained SentenceTransformer model identifier.
        """
        # Load pretrained transformer model (may download on first run)
        print(f"Loading embedding model: {model_name} ... (first run may take 1–2 min)")
        self.model = SentenceTransformer(model_name)

        # Select GPU if available for faster inference
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

        print(f"Model loaded on {self.device}")

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Convert a list of texts into normalized embedding vectors.

        Args:
            texts (List[str]): Input text samples.
            batch_size (int): Number of texts processed per batch.

        Returns:
            np.ndarray: Array of shape (n_samples, embedding_dim).
        """
        print(f"Generating embeddings for {len(texts)} texts ...")

        # Encode texts into dense vectors
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # ensures cosine similarity ≈ dot product
        )

        print(f"Embeddings shape: {embeddings.shape}")

        return embeddings

    def embed_dataframe(
        self, df: pd.DataFrame, text_column: str = "description", save_path: str | None = None
    ) -> pd.DataFrame:
        """Generate embeddings for a DataFrame column and attach them.

                Args:
                    df: Input DataFrame containing test cases.
                    text_column: Name of the column with text to embed.
                    save_path: Optional path to save embeddings as .npy file.

                Returns:
                    pd.DataFrame: Copy of input DataFrame with added 'embedding' column
                        containing numpy arrays of shape (embedding_dim,).

                Raises:
                    KeyError: If text_column does not exist in DataFrame.
                    RuntimeError: If model fails to encode texts.

                Example:
                    >>> embedder = TestCaseEmbedder()
                    >>> df_with_emb = embedder.embed_dataframe(df)
                    >>> print(df_with_emb["embedding"].iloc[0].shape)
                    (384)
                """
        # Extract text data from DataFrame
        texts = df[text_column].tolist()

        # Generate embeddings using the model
        embeddings = self.embed(texts)

        # Create a copy to avoid mutating original data
        df = df.copy()

        # Store embeddings as a column (each row contains a vector)
        df["embedding"] = list(embeddings)

        # Optionally persist embeddings for faster reuse
        if save_path:
            path = Path(save_path)

            # Ensure directory exists before saving
            path.parent.mkdir(parents=True, exist_ok=True)

            np.save(path, embeddings)
            print(f"Embeddings saved as NumPy array: {path}")

        return df
