import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import json

from machine_learning_strategies_revised import generate_investor_views, download_stock_data
import portfolio_statistics as ps


# ========================= BLACK-LITTERMAN =========================

def black_litterman_expected_returns(prior_weights, sigma, tau, P, Q, omega):
    pi = prior_weights
    ts = tau * sigma
    middle = np.linalg.inv(P @ ts @ P.T + omega)
    adj = ts @ P.T @ middle @ (Q - P @ pi)
    return pi + adj


# ===================== OPTIMIZATION =====================

def optimize_portfolio(expected_returns, cov_matrix, target_volatility, min_weight, max_weight):
    n = len(expected_returns)

    def objective(weights):
        ret = weights @ expected_returns
        vol = np.sqrt(weights.T @ cov_matrix @ weights)
        penalty = 1000 * max(0, vol - target_volatility)**2
        return -ret + penalty

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(min_weight, max_weight) for _ in range(n)]
    x0 = np.array([1/n]*n)

    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")

    return result.x


# ===================== FULL PIPELINE =====================

def full_pipeline(
    tickers, allocations, market_rep, start_date, end_date,
    backtest_start, backtest_end, post_bt_end,
    risk_free_rate, target_volatility, min_weight, max_weight,
    forward_days, model_type
):

    # 1. DOWNLOAD PRICE DATA
    price_data = {}
    for t in tickers:
        price_data[t] = download_stock_data(t, start_date, end_date)['Adj Close']

    price_df = pd.DataFrame(price_data)
    returns_df = price_df.pct_change().dropna()

    # 2. MACHINE LEARNING VIEWS
    ml_views = []
    for t in tickers:
        pred_ret, conf = generate_investor_views(
            t, start_date, end_date,
            model_type=model_type, forward_days=forward_days
        )
        ml_views.append(pred_ret)

    Q = np.array(ml_views)

    # 3. BLACK-LITTERMAN MERGING
    n = len(tickers)
    market_weights = np.array([1/n]*n)
    sigma = returns_df.cov().values
    tau = 0.05
    P = np.eye(n)
    omega = np.diag([0.0001]*n)

    bl_expected_returns = black_litterman_expected_returns(
        market_weights, sigma, tau, P, Q, omega
    )

    # 4. OPTIMIZATION
    optimized_weights = optimize_portfolio(
        bl_expected_returns, sigma, target_volatility, min_weight, max_weight
    )

    # 5. PORTFOLIO RETURNS
    portfolio_returns = (returns_df @ optimized_weights)
    cumulative_returns = (1 + portfolio_returns).cumprod()

    # 6. CALCULATE PORTFOLIO STATISTICS
    stats = {
        "annualized_return": float(ps.annualized_return(portfolio_returns)),
        "annualized_volatility": float(ps.annualized_volatility(portfolio_returns)),
        "sharpe_ratio": float(ps.sharpe_ratio(portfolio_returns, risk_free_rate)),
        "sortino_ratio": float(ps.sortino_ratio(portfolio_returns, risk_free_rate)),
        "max_drawdown": float(ps.max_drawdown(cumulative_returns)),
        "calmar_ratio": float(ps.calmar_ratio(portfolio_returns))
    }

    # 7. SAVE RESULTS JSON
    results = {
        "tickers": tickers,
        "optimized_weights": {t: float(w) for t, w in zip(tickers, optimized_weights)},
        "ml_views": {t: float(v) for t, v in zip(tickers, Q)},
        "daily_returns": portfolio_returns.tolist(),
        "cumulative_returns": cumulative_returns.tolist(),
        "portfolio_statistics": stats
    }

    with open("portfolio_results.json", "w") as f:
        json.dump(results, f, indent=4)

    # 8. SAVE CHART
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns, label="Portfolio")
    plt.title("Portfolio Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("portfolio_comparison.png")
    plt.close()

    return results
