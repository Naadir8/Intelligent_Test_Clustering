# Deployment Guide (Production Environment)

## Hardware Requirements

- **CPU**: 2+ cores (4+ recommended)
- **RAM**: Minimum 4 GB (8 GB+ recommended)
- **Disk**: At least 10 GB of free space
- **Architecture**: x86_64

## Required Software

- Python 3.13+
- pip
- Git
- Virtual environment tool (venv recommended)

## Deployment Steps

1. Clone the repository:
```bash
git clone https://github.com/Naadir8/Intelligent_Test_Clustering.git
cd Intelligent_Test_Clustering
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate     # Linux/macOS
.venv\Scripts\activate        # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python main.py
```

## Health Check
- No errors during startup
- data/processed/embeddings.npy file exists
- Clustering completes successfully
- Quality metrics (ARI, NMI, Silhouette Score) are displayed
- Process exits with code 0

## Production Recommendations
- Run as a system service (systemd / supervisor)
- Configure proper logging
- Monitor memory usage
- Use virtual environment isolation