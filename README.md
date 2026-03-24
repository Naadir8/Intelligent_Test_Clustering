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


## Quick Start for New Developers

This guide is intended for developers with a **freshly installed OS** with no required software installed yet.

### 1. Prerequisites

Install the following software:

- **Git**
- **Python 3.13** (or newer)
- **pip** (usually comes with Python)

### 2. Clone the Repository

```bash
git clone https://github.com/Naadir8/Intelligent_Test_Clustering.git
cd Intelligent_Test_Clustering
```

### 3. Set Up Development Environment

- Create virtual environment
```bash
python -m venv .venv
```
- Activate the environment (Windows)
```bash
.venv\Scripts\activate
```
- Activate the environment (Linux/macOS)
```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Project

```bash
python main.py
```

The system will:
- Generate a synthetic test case dataset
- Create text embeddings
- Perform clustering
- Display clustering quality metrics

### 6. Basic Commands

- Run main pipeline
```bash
python main.py
```
- Check code quality
```bash
ruff check .
```
- Format code
```bash
ruff format .
```
- Generate documentation
```bash
cd docs && make html
```

## Project Structure

- data/raw/ — raw data (synthetic dataset)
- data/processed/ — processed data (embeddings)
- src/ — main source code
- docs/ — project documentation

## For DevOps / Release Engineers

Detailed instructions for production deployment, updates, and backup procedures are located in the docs/ folder:
- deployment.md — Production deployment guide
- update.md — System update procedure
- backup.md — Backup and restore strategy