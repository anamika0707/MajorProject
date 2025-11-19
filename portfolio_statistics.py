import numpy as np
import pandas as pd

# === Sharpe Ratio ===
def sharpe_ratio(returns, risk_free_rate=0.0):
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]
    excess_returns = returns - (risk_free_rate / 252)
    annualized_excess_return = np.mean(excess_returns) * 252
    annualized_std_dev = np.std(excess_returns, ddof=1) * np.sqrt(252)
    if annualized_std_dev == 0:
        return np.nan
    return annualized_excess_return / annualized_std_dev


# === Sortino Ratio ===
def sortino_ratio(returns, risk_free_rate=0.0):
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]
    excess_returns = returns - (risk_free_rate / 252)
    downside_returns = np.minimum(excess_returns, 0)
    annualized_excess_return = np.mean(excess_returns) * 252
    annualized_downside_std_dev = np.std(downside_returns, ddof=1) * np.sqrt(252)
    if annualized_downside_std_dev == 0:
        return np.nan
    return annualized_excess_return / annualized_downside_std_dev


# === Information Ratio ===
def information_ratio(returns, benchmark_returns):
    returns = np.array(returns)
    benchmark_returns = np.array(benchmark_returns)
    mask = ~np.isnan(returns) & ~np.isnan(benchmark_returns)
    returns = returns[mask]
    benchmark_returns = benchmark_returns[mask]
    active_returns = returns - benchmark_returns
    annualized_active_return = np.mean(active_returns) * 252
    tracking_error = np.std(active_returns, ddof=1) * np.sqrt(252)
    if tracking_error == 0:
        return np.nan
    return annualized_active_return / tracking_error


# === Correlation with Market ===
def calculate_correlation_with_market(portfolio_data, market_data):
    common_dates = portfolio_data.index.intersection(market_data.index)
    portfolio_data = portfolio_data.loc[common_dates]
    market_data = market_data.loc[common_dates]
    return portfolio_data.corrwith(market_data)


# === Annualized Return ===
def annualized_return(returns):
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]
    return (1 + np.mean(returns)) ** 252 - 1


# === Annualized Volatility ===
def annualized_volatility(returns):
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]
    return np.std(returns, ddof=1) * np.sqrt(252)


# === Maximum Drawdown ===
def max_drawdown(cumulative_returns):
    cumulative_returns = np.array(cumulative_returns)
    cumulative_returns = cumulative_returns[~np.isnan(cumulative_returns)]
    roll_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = cumulative_returns / roll_max - 1.0
    return drawdowns.min() if len(drawdowns) > 0 else np.nan


# === Calmar Ratio ===
def calmar_ratio(returns, risk_free_rate=0.0):
    cumulative = (1 + np.array(returns))  # daily returns
    cumulative = np.cumprod(cumulative)
    max_dd = max_drawdown(cumulative)
    ann_return = annualized_return(returns)
    if max_dd == 0 or np.isnan(max_dd):
        return np.nan
    return ann_return / abs(max_dd)
