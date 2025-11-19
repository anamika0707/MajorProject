import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import date
from typing import Dict, List, Optional
from PIL import Image

import main5
from machine_learning_strategies_revised import generate_investor_views, download_stock_data

st.set_page_config(page_title="ML Portfolio Dashboard", layout="wide")

st.title("📈 ML + Black-Litterman Portfolio Dashboard")

# ------------------ CACHED FUNCTIONS ------------------
@st.cache_data(show_spinner=False)
def cached_download(ticker: str, start: str, end: str):
    return download_stock_data(ticker, start, end)

@st.cache_data(show_spinner=False)
def cached_generate_view(ticker: str, start: str, end: str, model_type: str, forward_days: int):
    return generate_investor_views(ticker, start, end, model_type=model_type, forward_days=forward_days)

# ------------------ VALIDATE INPUTS ------------------
def validate_inputs(tickers: List[str], market_rep: List[str], start_date: date, end_date: date,
                    backtest_start: date, backtest_end: date, min_weight: float, max_weight: float) -> Optional[str]:
    n = len(tickers)
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
    if min_weight * n > 1:
        return f"Min weight too high! For {n} tickers, max min_weight = {1/n:.3f}"
    if max_weight * n < 1:
        return f"Max weight too low! For {n} tickers, min max_weight = {1/n:.3f}"
    return None

# ------------------ PARSE ALLOCATIONS ------------------
def parse_allocations(tickers: List[str], allocations_input: str, min_weight: float, max_weight: float) -> Optional[Dict[str, float]]:
    n = len(tickers)
    if allocations_input.strip() == "":
        w = 1.0 / n
        return {t: w for t in tickers}
    parts = [p.strip() for p in allocations_input.split(",") if p.strip()]
    if len(parts) != n:
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
    vals = [max(min_weight, min(max_weight, v / s)) for v in vals]
    return {t: v for t, v in zip(tickers, vals)}

# ------------------ SIDEBAR INPUTS ------------------
with st.sidebar:
    st.header("Portfolio Inputs")
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
    model_type = st.selectbox("ML Model", options=["XGBoost", "Random Forest", "Gradient Boosting", "Linear Regression"], index=0)
    forward_days = st.number_input("Forward days (for features)", min_value=1, max_value=252, value=20)

    risk_free_rate = st.number_input("Risk-free rate (annual)", value=0.04, format="%.4f")
    target_volatility = st.number_input("Target annual volatility", value=0.3, format="%.3f")
    min_weight = st.number_input("Min weight per asset", value=0.01, format="%.3f")
    max_weight = st.number_input("Max weight per asset", value=0.4, format="%.3f")
    allocations_input = st.text_input("Unoptimized allocations (comma floats; optional)", value="")

    run_views = st.button("Generate Views")
    run_pipeline = st.button("Run Full Pipeline")

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
market_rep = [market_rep_input.strip()]

# ------------------ TABS ------------------
tabs = st.tabs(["📊 ML Views", "🗂️ Full Pipeline Results"])

# ------------------ ML VIEWS ------------------
with tabs[0]:
    if run_views:
        st.subheader("Generated ML Views (Forward Returns)")
        err = validate_inputs(tickers, market_rep, start_date, end_date, backtest_start, backtest_end, min_weight, max_weight)
        if err:
            st.error(err)
            st.stop()

        results = []
        total = len(tickers)
        for i, t in enumerate(tickers, start=1):
            st.info(f"Generating view for {t} ({i}/{total})")
            try:
                pred_ret, conf = cached_generate_view(t, start_date.isoformat(), end_date.isoformat(), model_type, forward_days)
                pred_ret = float(pred_ret)
                conf = float(conf)
            except Exception as e:
                st.error(f"Error generating view for {t}: {e}")
                pred_ret, conf = np.nan, 0.0

            df_price = cached_download(t, start_date.isoformat(), end_date.isoformat())
            if df_price is not None and not df_price.empty:
                with st.container():
                    st.markdown(f"### {t}")
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.line_chart(df_price['Adj Close'] if 'Adj Close' in df_price.columns else df_price, height=200)
                    with col2:
                        st.metric(label="Predicted Return", value=f"{pred_ret*100:.2f}%", delta=f"Conf: {conf:.2f}")

            results.append({"ticker": t, "predicted_forward_return": pred_ret, "confidence": conf})

        df_views = pd.DataFrame(results)
        st.dataframe(df_views)
        if not df_views.empty:
            st.bar_chart(df_views.set_index('ticker')['predicted_forward_return'])

# ------------------ FULL PIPELINE ------------------# ------------------ FULL PIPELINE ------------------
with tabs[1]:
    if run_pipeline:
        st.subheader("Full Pipeline Execution")
        err = validate_inputs(tickers, market_rep, start_date, end_date, backtest_start, backtest_end, min_weight, max_weight)
        if err:
            st.error(err)
            st.stop()

        allocations = parse_allocations(tickers, allocations_input, min_weight, max_weight)
        if allocations is None:
            st.stop()

        try:
            status = st.empty()
            status.info("Running pipeline...")
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
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.stop()

        st.success("Pipeline finished successfully!")

        # ---------- Portfolio Statistics ----------
        st.subheader("📈 Portfolio Statistics")
        try:
            stats = results.get('portfolio_statistics', {})
            if stats:
                col1, col2, col3 = st.columns(3)
                col4, col5 = st.columns(2)
                
                col1.metric("Annualized Return", f"{stats.get('annualized_return',0)*100:.2f}%")
                col2.metric("Annualized Volatility", f"{stats.get('annualized_volatility',0)*100:.2f}%")
                col3.metric("Sharpe Ratio", f"{stats.get('sharpe_ratio',0):.2f}")
                col4.metric("Sortino Ratio", f"{stats.get('sortino_ratio',0):.2f}")
                col5.metric("Max Drawdown", f"{stats.get('max_drawdown',0)*100:.2f}%")
            else:
                st.info("Portfolio statistics not found in results.")
        except Exception as e:
            st.error(f"Error displaying stats: {e}")

        # ---------- Results JSON ----------
        st.subheader("Results JSON")
        st.json(results)

        # ---------- Portfolio Chart ----------
        img_path = 'portfolio_comparison.png'
        if os.path.exists(img_path):
            st.subheader("Cumulative Returns Chart")
            img = Image.open(img_path)
            st.image(img, use_container_width=True)
        else:
            st.info("No chart found (expected portfolio_comparison.png)")

