# Intelligent Test Case Clustering

This project demonstrates an intelligent system for clustering and evaluating test cases in automated software testing using semantic embeddings and unsupervised machine learning.

**Goal**
Group semantically similar test cases (by title and description) to detect duplicates, improve test suite quality, and support automated testing maintenance.

**Main tasks solved in this version**
- Generate synthetic test cases with predefined semantic clusters  
- Convert test case descriptions into dense vector embeddings  
- Perform clustering (KMeans) on embeddings  
- Evaluate clustering quality using internal and external metrics  
- Cache embeddings for faster repeated runs

**Stack and requirements**
- Python 3.13  
- sentence-transformers (all-MiniLM-L6-v2)  
- scikit-learn (KMeans + metrics)  
- pandas, numpy  
- matplotlib (planned for visualization in future versions)

**Key components**
- `loader.py` — creates 1000 synthetic test cases divided into 8 clear semantic domains (Authentication, API, UI/UX, etc.)  
- `embedder.py` — converts titles + descriptions into 384-dimensional vectors using Sentence-BERT  
- `clusterer.py` — applies KMeans clustering and computes ARI, NMI, Silhouette Score  
- `main.py` — full pipeline: generate → embed → cluster → evaluate

**Planned improvements**
- Add text preprocessing (stopword removal, lemmatization, normalization) — preprocessing/
- Implement 2D/3D cluster visualization (UMAP + scatter plots with true vs predicted labels) — visualization/
- Compare multiple clustering algorithms: KMeans, HDBSCAN, AgglomerativeClustering, DBSCAN
- Add comparison of different embedding models (paraphrase-mpnet, multilingual, distiluse, etc.)
- Implement duplicate detection within clusters and test set reduction metrics
- Integrate a real-world test case dataset (e.g., from Kaggle or other open sources)
- Add YAML-based configuration (model, number of clusters, parameters, etc.)
- Add logging and a CLI interface (argparse or click)
- Write unit tests (pytest) for core modules
- Add export of clustering results to CSV/Excel for analysis