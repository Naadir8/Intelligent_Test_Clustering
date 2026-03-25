# Backup and Restore Guide

## Backup Strategy

- **Type**: Full backups
- **Frequency**:
  - Daily backup of `data/processed/`
  - Full project backup after major updates
- **Retention**: Keep last 7 daily backups + monthly archives

## What to Backup

- `data/processed/embeddings.npy` (most critical)
- `data/raw/synthetic_test_cases.csv`
- Entire project source code
- Configuration files (`pyproject.toml`, `requirements.txt`)

## Backup Procedure

```bash
DATE=$(date +%Y%m%d_%H%M)
mkdir -p backups/$DATE
```

- Backup processed data
```bash
cp -r data/processed/ backups/$DATE/
```
- Backup raw data
```bash
cp -r data/raw/ backups/$DATE/
```
- Backup source code
```bash
tar -czf backups/$DATE/code_backup.tar.gz --exclude='data' --exclude='.git' --exclude='__pycache__' .
```

## Backup Integrity Check

After backup:
- Verify presence of key files
- Test restoration in a separate environment
- Run python main.py and compare clustering metrics

## Restore Procedure

1. Extract backup
2. Restore data/processed/ and data/raw/ folders
3. Run python main.py to verify functionality

## Testing Restoration

After restore, always run the full pipeline and check that:
- Embeddings load correctly
- Clustering completes without errors
- Quality metrics are consistent with previous runs