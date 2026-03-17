"""Provide utilities for generating sentence embeddings from test case texts.

This module handles:
- loading and configuring a transformer-based embedding model,
- converting text data into dense vector representations,
- integrating embeddings into pandas DataFrames,
- optionally persisting embeddings for reuse.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from typing import List, Optional
import torch
from pathlib import Path


class TestCaseEmbedder:
    """Generate dense vector representations (embeddings) for test case texts.

    This class wraps a SentenceTransformer model and provides methods for:
    - batch encoding of text data,
    - normalization of embeddings for similarity-based tasks,
    - seamless integration with tabular datasets.

    Attributes:
        model (SentenceTransformer): Loaded embedding model.
        device (str): Execution device ("cuda" or "cpu").
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
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

    def embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
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
            normalize_embeddings=True  # ensures cosine similarity ≈ dot product
        )

        print(f"Embeddings shape: {embeddings.shape}")

        return embeddings

    def embed_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "description",
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Generate embeddings for a DataFrame column and attach them.

        Args:
            df (pd.DataFrame): Input dataset.
            text_column (str): Column containing text to embed.
            save_path (Optional[str]): Optional path to save embeddings (.npy).

        Returns:
            pd.DataFrame: Copy of the input DataFrame with an added
                "embedding" column containing vector representations.
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