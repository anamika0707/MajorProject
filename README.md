# ML + Black-Litterman Portfolio Frontend

This project includes an ML model for generating investor views and a Black-Litterman mean-variance pipeline. This repository now contains a Streamlit frontend to interact with the models and run the pipeline.

Files of interest:
- `machine_learning_strategies_revised.py` — ML model, feature creation, and `generate_investor_views()`.
- `main5.py` — data fetching, optimization (MVO & Black-Litterman), backtest and pipeline orchestration.
- `app.py` — Streamlit frontend (new): generate views, run full pipeline, display results and charts.
- `requirements.txt` — dependencies to install.

Quick start (PowerShell):

1. (Optional) Create & activate a virtual environment:
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:
```powershell
pip install -r .\requirements.txt
```

3. Run the Streamlit app:
```powershell
streamlit run .\app.py
```

Notes & caveats:
- The app downloads price data via `yfinance`; network access is required.
- The Black-Litterman pipeline uses `pypfopt` and can take several minutes depending on the number of tickers and model complexity.
- If you see import errors for `xgboost` or `pypfopt`, install them separately or use conda if on Windows with compilation issues:
    - `pip install xgboost` or `conda install -c conda-forge xgboost`
    - `pip install pypfopt`
- The Streamlit UI uses caching to reduce repeated downloads and model runs; if you change code in the ML module, clear Streamlit cache (`streamlit cache clear` or use the UI) to force a fresh run.

If you want, I can attempt a smoke test here to verify the app starts and generates the expected files. Reply with `run` to try a local run (may require network access and time).
