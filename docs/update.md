# Update Guide

## Preparation

1. Create a backup before updating:
```bash
git tag backup-before-update-$(date +%Y%m%d)
cp -r data/processed/ backups/backup-before-update-$(date +%Y%m%d)/
```

2. Review changes in requirements.txt and code for breaking changes.

## Update Process

1. Stop the current process (if running as a service).
2. Pull the latest code:
```bash
git pull origin main
```

3. Update dependencies:
```bash
pip install -r requirements.txt --upgrade
```

4. Restart the application:
```bash
python main.py
```

## Rollback Procedure

In case of issues, revert to the previous version:
```bash
git checkout backup-before-update-YYYYMMDD
pip install -r requirements.txt
python main.py
```

It is strongly recommended to create a tagged backup before every update.