# Scalable Taxi Trip Pattern Mining

K-Means and MiniBatch K-Means on January 2025 NYC Yellow Taxi trips.

## Windows PowerShell setup

Install Python 3.11 or 3.12, then from the repository:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\ruff check .
.\.venv\Scripts\ruff format --check .
```

The synthetic fixture is exclusively for testing. Real results require the official dataset.
Data and model artifacts are deliberately excluded from Git.
