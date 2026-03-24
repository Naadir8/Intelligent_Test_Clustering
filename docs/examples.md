# Examples and Usage

This section provides executable examples of how to use the main components of the project. 
These examples serve as **living documentation** — they can be copied and run directly.

## 1. Generating Synthetic Test Cases

```python
from src.data.loader import TestCaseLoader

# Create loader and generate dataset
loader = TestCaseLoader()
df = loader.generate_synthetic(n_samples=800, n_clusters=8, seed=42)

print(f"Generated {len(df)} test cases")
print("\nCluster distribution:")
print(df["true_cluster"].value_counts())
```
## 2. Generating Embeddings

```python
from src.embedding.embedder import TestCaseEmbedder

embedder = TestCaseEmbedder(model_name="all-MiniLM-L6-v2")

# Generate embeddings from descriptions
df = embedder.embed_dataframe(df, text_column="description", save_path="data/processed/embeddings.npy")

print(f"Embeddings shape: {df['embedding'].iloc[0].shape}")
```

## 3. Clustering and Evaluation

```python
from src.clustering.clusterer import TestCaseClusterer
import numpy as np

# Prepare embeddings
embeddings = np.array(df["embedding"].tolist())

clusterer = TestCaseClusterer(n_clusters=8)
predicted_labels = clusterer.fit_predict(embeddings)

# Evaluate quality
metrics = clusterer.evaluate(
    predicted_labels=predicted_labels,
    true_labels=df["true_cluster"],
    embeddings=embeddings
)

print(metrics)
```

## 4.Full Pipeline (Recommended way)

```python
from src.main import main

# Run the entire pipeline
main()
```

These examples are kept up-to-date with the actual codebase and serve as both documentation and practical usage guides.