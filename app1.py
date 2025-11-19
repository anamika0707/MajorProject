import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import date
from typing import Dict, List, Optional

import main5
from machine_learning_strategies_revised import generate_investor_views, download_stock_data


st.set_page_config(page_title="ML Portfolio Frontend", layout="wide")
st.title("ML + Black-Litterman Portfolio Frontend")


@st.cache_data(show_spinner=False)
def cached_download(ticker: str, start: str, end: str):
    return download_stock_data(ticker, start, end)


@st.cache_data(show_spinner=False)
def cached_generate_view(ticker: str, start: str, end: str, model_type: str, forward_days: int):
    return generate_investor_views(ticker, start, end, model_type=model_type, forward_days=forward_days)


def validate_inputs(tickers: List[str], market_rep: List[str], start_date: date, end_date: date, backtest_start: date, backtest_end: date) -> Optional[str]:
    if not tickers:
        return "Please provide at least one ticker."
    if not market_rep or len(market_rep) != 1:
        return "Provide a single market representative ticker (e.g. ^GSPC)."
    if start_date >= end_date:
        return "Start date must be before end date."
    if backtest_start >= backtest_end:
        return "Backtest start must be before backtest end."
    if backtest_start < start_date or backtest_end > end_date:
        return "Backtest window should be within the overall data window."
    return None


with st.sidebar:
    st.header("Inputs")
    tickers_input = st.text_input("Tickers (comma-separated)", value="AAPL,MSFT,GOOGL")
    market_rep_input = st.text_input("Market representative ticker (single, e.g. ^GSPC)", value="^GSPC")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", value=date(2018, 1, 1))
        end_date = st.date_input("End date", value=date(2023, 12, 31))
    with col2:
        backtest_start = st.date_input("Backtest start", value=date(2020, 1, 1))
        backtest_end = st.date_input("Backtest end", value=date(2022, 12, 31))
    post_bt_end = st.date_input("Post backtest retrain end", value=date(2023, 12, 31))

    model_type = st.selectbox("Model", options=["XGBoost", "Random Forest", "Gradient Boosting", "Linear Regression"], index=0)
    forward_days = st.number_input("Forward days (for features)", min_value=1, max_value=252, value=20)

    risk_free_rate = st.number_input("Risk-free rate (annual)", value=0.04, format="%.4f")
    target_volatility = st.number_input("Target annual volatility", value=0.3, format="%.3f")
    min_weight = st.number_input("Min weight per asset", value=0.01, format="%.3f")
    max_weight = st.number_input("Max weight per asset", value=0.2, format="%.3f")

    allocations_input = st.text_input("Unoptimized allocations (comma floats; optional)", value="")

    run_views = st.button("Generate Views")
    run_pipeline = st.button("Run Full Pipeline")


tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
market_rep = [market_rep_input.strip()]


def parse_allocations(tickers: List[str], allocations_input: str) -> Optional[Dict[str, float]]:
    if allocations_input.strip() == "":
        w = 1.0 / len(tickers)
        return {t: w for t in tickers}
    parts = [p.strip() for p in allocations_input.split(",") if p.strip()]
    if len(parts) != len(tickers):
        st.error("Number of allocations must match number of tickers or be left empty for equal weights.")
        return None
    try:
        vals = [float(p) for p in parts]
    except Exception:
        st.error("Allocations must be numeric floats.")
        return None
    s = sum(vals)
    if s == 0:
        st.error("Allocations sum must be non-zero.")
        return None
    vals = [v / s for v in vals]
    return {t: v for t, v in zip(tickers, vals)}


log = st.empty()

if run_views:
    st.subheader("Generated ML Views (forward returns)")
    err = validate_inputs(tickers, market_rep, start_date, end_date, backtest_start, backtest_end)
    if err:
        st.error(err)
        st.stop()

    results = []
    progress = st.progress(0)
    total = len(tickers)
    for i, t in enumerate(tickers, start=1):
        log.info(f"Generating view for {t} ({i}/{total})")
        try:
            pred_ret, conf = cached_generate_view(t, start_date.isoformat(), end_date.isoformat(), model_type, forward_days)
            pred_ret = float(pred_ret)
            conf = float(conf)
        except Exception as e:
            st.error(f"Error generating view for {t}: {e}")
            pred_ret, conf = np.nan, 0.0

        try:
            df_price = cached_download(t, start_date.isoformat(), end_date.isoformat())
            if df_price is not None and not df_price.empty:
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    st.write(f"Price series for {t}")
                    st.line_chart(df_price['Adj Close'] if 'Adj Close' in df_price.columns else df_price)
                with col_b:
                    st.metric(label=f"Predicted {forward_days}d Return", value=f"{pred_ret*100:.2f}%", delta=f"Conf: {conf:.2f}")
            else:
                st.write(f"No price data for {t}")
        except Exception:
            pass

        results.append({"ticker": t, "predicted_forward_return": pred_ret, "confidence": conf})
        progress.progress(int(i / total * 100))

    df_views = pd.DataFrame(results)
    st.dataframe(df_views)
    if not df_views.empty:
        st.bar_chart(df_views.set_index('ticker')['predicted_forward_return'])


if run_pipeline:
    st.subheader("Running full pipeline — this may take several minutes")
    err = validate_inputs(tickers, market_rep, start_date, end_date, backtest_start, backtest_end)
    if err:
        st.error(err)
        st.stop()

    allocations = parse_allocations(tickers, allocations_input)
    if allocations is None:
        st.stop()

    # --- START: Min/Max weight check ---
    n_tickers = len(tickers)
    if min_weight * n_tickers > 1:
        st.warning(f"min_weight ({min_weight}) too high for {n_tickers} tickers. Adjusting to {1/n_tickers:.3f}")
        min_weight = 1.0 / n_tickers
    if max_weight * n_tickers < 1:
        st.warning(f"max_weight ({max_weight}) too low for {n_tickers} tickers. Adjusting to {1/n_tickers:.3f}")
        max_weight = 1.0 / n_tickers
    # --- END: Min/Max weight check ---

    try:
        progress_bar = st.progress(0)
        status = st.empty()
        status.info("Starting pipeline — fetching data and generating views...")

        for i, t in enumerate(tickers, start=1):
            status.info(f"Generating view for {t} ({i}/{len(tickers)})")
            try:
                _ = cached_generate_view(t, start_date.isoformat(), end_date.isoformat(), model_type, forward_days)
            except Exception:
                pass
            progress_bar.progress(int(i / len(tickers) * 10))

        status.info("Calling full pipeline. This step may take several minutes.")
        with st.spinner("Executing pipeline..."):
            results = main5.full_pipeline(
                tickers=tickers,
                allocations=allocations,
                market_rep=market_rep,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                backtest_start=backtest_start.isoformat(),
                backtest_end=backtest_end.isoformat(),
                post_bt_end=post_bt_end.isoformat(),
                risk_free_rate=risk_free_rate,
                target_volatility=target_volatility,
                min_weight=min_weight,
                max_weight=max_weight,
                forward_days=forward_days,
                model_type=model_type
            )

        progress_bar.progress(100)
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        st.stop()

    st.success("Pipeline finished — results saved to workspace")

    st.subheader("Results JSON")
    try:
        st.json(results)
    except Exception:
        try:
            with open('portfolio_results.json', 'r') as f:
                st.json(json.load(f))
        except Exception as e:
            st.error(f"Could not load results JSON: {e}")

    img_path = 'portfolio_comparison.png'
    if os.path.exists(img_path):
        from PIL import Image
        st.subheader("Saved Cumulative Returns Chart")
        img = Image.open(img_path)
        st.image(img, use_column_width=True)
    else:
        st.info("No chart found (expected portfolio_comparison.png)")

    st.info("Pipeline created `portfolio_results.json` and `portfolio_comparison.png` in workspace root.")


st.markdown("---")
st.write("Tip: Start by using 'Generate Views' to inspect model outputs, then run the full pipeline.")
