# Architecture

## Overview

`Intelligent Test Case Clustering` is a system designed to automatically group test cases based on their semantic similarity. 
The main goal is to help testing teams identify duplicate or near-duplicate test scenarios, reduce redundancy, and improve the maintainability of automated test suites.

By converting test case descriptions into vector embeddings and applying clustering algorithms, the system reveals hidden semantic groups that are difficult to detect manually.

## High-Level Architecture

The project follows a clean, layered architecture:

- **Data Layer** (`src/data/`)  
  Responsible for generating and loading test case datasets. Currently uses synthetic data with predefined semantic clusters.

- **Embedding Layer** (`src/embedding/`)  
  Converts natural language test case titles and descriptions into dense numerical vectors using SentenceTransformer models.

- **Clustering Layer** (`src/clustering/`)  
  Performs clustering on the generated embeddings and evaluates the quality of the resulting groups using multiple metrics.

- **Application Layer** (`main.py`)  
  Orchestrates the entire pipeline: data loading → embedding generation → clustering → evaluation.

## Business Logic

The core business value lies in **semantic deduplication** of test cases.  
Instead of manually reviewing hundreds or thousands of test scenarios, the system automatically detects groups of semantically similar tests. This allows test engineers to:

- Remove redundant test cases
- Maintain better coverage with fewer tests
- Focus manual effort on unique scenarios
- Improve long-term test suite maintainability

## Key Design Decisions

- **Embedding Caching**: Generated embeddings are saved to `data/processed/embeddings.npy` to avoid recomputing them on every run.
- **Modular Design**: Each layer is independent, making it easy to replace the embedding model or clustering algorithm in the future.
- **Synthetic Data**: Used at the current stage to have full control over ground-truth clusters for reliable evaluation.
- **Google-style Docstrings**: Chosen as the documentation standard for better readability and Sphinx compatibility.

## Component Interaction

```mermaid
graph TD
    A[main.py] --> B[TestCaseLoader.generate_synthetic()]
    B --> C[DataFrame with test cases]
    C --> D[TestCaseEmbedder.embed_dataframe()]
    D --> E[Embeddings]
    E --> F[TestCaseClusterer.fit_predict() + evaluate()]
    F --> G[Predicted clusters + quality metrics]
```

## Future Architecture Improvements

- Integration with real-world test case repositories;
- Support for multiple embedding models;
- Advanced clustering algorithms (HDBSCAN, Agglomerative);
- Visualization layer (UMAP 2D/3D plots);
- Configuration management via YAML.