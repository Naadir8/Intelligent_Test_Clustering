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

# Code Documentation Standards

All public classes, methods, and functions must be documented using **Google-style docstrings** (PEP 257 + Google Python Style Guide).

**Mandatory sections**:
- Short description (first line)
- Args: (parameters with types and descriptions)
- Returns: (return type and description)
- Raises: (possible exceptions)
- Example: (short usage example — strongly recommended)

**Example**:
```python
def example_function(param1: int, param2: str) -> bool:
    """Short one-line description.

    Longer explanation if needed.

    Args:
        param1: Description of first parameter.
        param2: Description of second parameter.

    Returns:
        bool: Meaning of the return value.

    Raises:
        ValueError: When input is invalid.

    Example:
        >>> example_function(42, "test")
        True
    """
```
**Rules for contributors:**
- Document all public interfaces (classes, methods, functions)
- Use type hints (PEP 484)
- Add examples for complex logic
- Keep docstrings concise but informative
- Use consistent style across the project

Future documentation will be auto-generated with Sphinx (see docs/generate_docs.md).