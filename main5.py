import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import json
from machine_learning_strategies_revised import generate_investor_views, download_stock_data

def black_litterman_expected_returns(prior_weights, sigma, tau, P, Q, omega):
    """
    Black-Litterman formula:
    E[R] = Pi + tau * sigma * P.T * (P * tau * sigma * P.T + omega)^-1 * (Q - P * Pi)
    """
    pi = prior_weights  # equilibrium returns
    ts = tau * sigma
    middle = np.linalg.inv(P @ ts @ P.T + omega)
    adj = ts @ P.T @ middle @ (Q - P @ pi)
    return pi + adj


def optimize_portfolio(expected_returns, cov_matrix, target_volatility, min_weight, max_weight):
    n = len(expected_returns)
    
    # Objective: minimize negative Sharpe (maximize returns for given volatility)
    def objective(weights):
        ret = weights @ expected_returns
        vol = np.sqrt(weights.T @ cov_matrix @ weights)
        # Penalize deviation from target volatility
        penalty = 1000 * max(0, vol - target_volatility)**2
        return -ret + penalty

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(min_weight, max_weight) for _ in range(n)]

    # Initial guess
    x0 = np.array([1/n]*n)

    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")

    return result.x


def full_pipeline(
    tickers, allocations, market_rep, start_date, end_date,
    backtest_start, backtest_end, post_bt_end,
    risk_free_rate, target_volatility, min_weight, max_weight,
    forward_days, model_type
):
    """
    Full ML + Black-Litterman portfolio pipeline.
    """
    # 1. Download all stock data
    price_data = {}
    for t in tickers:
        price_data[t] = download_stock_data(t, start_date, end_date)['Adj Close']

    price_df = pd.DataFrame(price_data)
    returns_df = price_df.pct_change().dropna()

    # 2. Generate ML views
    ml_views = []
    for t in tickers:
        pred_ret, conf = generate_investor_views(t, start_date, end_date, model_type=model_type, forward_days=forward_days)
        ml_views.append(pred_ret)
    Q = np.array(ml_views)

    # 3. Black-Litterman
    market_weights = np.array([1/len(tickers)]*len(tickers))
    sigma = returns_df.cov().values
    tau = 0.05

    # P matrix: identity (each view on a single asset)
    P = np.eye(len(tickers))
    omega = np.diag([0.0001]*len(tickers))  # small uncertainty

    bl_expected_returns = black_litterman_expected_returns(market_weights, sigma, tau, P, Q, omega)

    # 4. Portfolio optimization
    optimized_weights = optimize_portfolio(bl_expected_returns, sigma, target_volatility, min_weight, max_weight)

    # 5. Calculate portfolio returns
    portfolio_returns = (returns_df @ optimized_weights)
    cumulative_returns = (1 + portfolio_returns).cumprod()

    # 6. Save JSON
    results = {
        "tickers": tickers,
        "optimized_weights": {t: float(w) for t, w in zip(tickers, optimized_weights)},
        "ml_views": {t: float(v) for t, v in zip(tickers, Q)},
        "daily_returns": portfolio_returns.tolist(),
        "cumulative_returns": cumulative_returns.tolist()
    }

    with open("portfolio_results.json", "w") as f:
        json.dump(results, f, indent=4)

    # 7. Save chart
    plt.figure(figsize=(10,6))
    cumulative_returns.plot()
    plt.title("Portfolio Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("portfolio_comparison.png")
    plt.close()

    return results
