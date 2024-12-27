import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import argparse
from scipy.optimize import minimize
from portfolio_risk_kit import plot_ef_ax

"""
 Author: Jordi Corbilla

 Date: 26/12/2024

 Project Description:
   This project demonstrates how to optimize a portfolio through Monte Carlo simulation.
   By randomly generating thousands of possible portfolio allocations, we identify an
   optimal investment mix that seeks to maximize returns and minimize risk. This approach
   is rooted in Modern Portfolio Theory and illustrates the concept of the Efficient Frontier.

 Important Note:
   Past performance does not guarantee future results. To evaluate risk mitigation, compare
   the volatility of the optimized portfolio against either the original portfolio or a
   chosen benchmark.
"""

TRADING_DAYS = 252

def download_data(assets, start_date, end_date):
    """
    Downloads the adjusted close price data from Yahoo Finance for the given assets
    between the specified date range.

    :param assets: List of asset tickers.
    :param start_date: Start date for data in 'YYYY-MM-DD' format.
    :param end_date: End date for data in 'YYYY-MM-DD' format.
    :return: A pandas DataFrame of adjusted close prices.
    """
    print(assets)
    data = yf.download(assets, start=start_date, end=end_date)
    print(data)
    return data['Close']


def calculate_statistics(data, risk_free_rate=0.0):
    """
    Calculates daily returns, covariance matrix, mean daily returns, 
    annualized returns, annualized volatility, and Sharpe ratio 
    for the entire dataset.

    :param data: A pandas DataFrame of adjusted close prices.
    :param risk_free_rate: The risk-free rate, default is 0.0 (for simplicity).
    :return: daily_returns (DataFrame), cov_matrix (DataFrame)
    """
    daily_returns = data.pct_change().dropna()
    
    cov_matrix = daily_returns.cov()
    
    return daily_returns, cov_matrix


def run_monte_carlo_simulation(daily_returns, cov_matrix, num_portfolios=100_000, 
                               risk_free_rate=0.0):
    """
    Runs the Monte Carlo simulation to generate a large number of random portfolios,
    calculates their performance metrics (annualized return, volatility, Sharpe ratio),
    and returns a DataFrame of results as well as an array of the weight vectors.

    :param daily_returns: DataFrame of asset daily returns.
    :param cov_matrix: Covariance matrix of asset daily returns.
    :param num_portfolios: Number of random portfolios to simulate.
    :param risk_free_rate: Risk-free rate to be used in Sharpe ratio calculation.
    :return: (simulated_portfolios, weights_record)
    """
    
    results = np.zeros((4, num_portfolios))
    weights_record = np.zeros((len(daily_returns.columns), num_portfolios))
    
    print(f'Running {num_portfolios} Monte Carlo Simulation')
    for i in range(num_portfolios):
        weights = np.random.random(len(daily_returns.columns))
        weights /= np.sum(weights)
        weights_record[:, i] = weights

        portfolio_return = np.sum(weights * daily_returns.mean()) * TRADING_DAYS

        portfolio_stddev = np.sqrt(
            np.dot(weights.T, np.dot(cov_matrix, weights))
        ) * np.sqrt(TRADING_DAYS)

        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_stddev

        results[0, i] = portfolio_return
        results[1, i] = portfolio_stddev
        results[2, i] = sharpe_ratio
        results[3, i] = i

    columns = ['Return', 'Volatility', 'Sharpe Ratio', 'Simulation']
    simulated_portfolios = pd.DataFrame(results.T, columns=columns)
    
    return simulated_portfolios, weights_record


def get_market_statistics(market_ticker, start_date, end_date, risk_free_rate=0.0):
    """
    Downloads data for a market index (e.g., SPY), then calculates its
    annualized return, annualized volatility, and Sharpe ratio.
    """
    market_data = yf.download([market_ticker], start=start_date, end=end_date)['Close']
    
    if isinstance(market_data, pd.DataFrame):
        market_data = market_data[market_ticker] 
    
    market_daily_returns = market_data.pct_change().dropna()

    market_return = market_daily_returns.mean() * TRADING_DAYS
    market_volatility = market_daily_returns.std() * np.sqrt(TRADING_DAYS)
    market_sharpe_ratio = (market_return - risk_free_rate) / market_volatility

    if hasattr(market_return, 'iloc'):
        market_return = market_return.iloc[0]
    if hasattr(market_volatility, 'iloc'):
        market_volatility = market_volatility.iloc[0]
    if hasattr(market_sharpe_ratio, 'iloc'):
        market_sharpe_ratio = market_sharpe_ratio.iloc[0]

    return market_return, market_volatility, market_sharpe_ratio


def plot_efficient_frontier(simulated_portfolios, weights_record, assets, 
                            market_return, market_volatility, market_sharpe, 
                            daily_returns, cov_matrix,
                            risk_free_rate=0.0, title='Efficient Frontier'):
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    fig.subplots_adjust(right=0.95)

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.1f}%'.format(x * 100)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1f}%'.format(y * 100)))

    sc = ax.scatter(
        simulated_portfolios['Volatility'], 
        simulated_portfolios['Return'], 
        c=simulated_portfolios['Sharpe Ratio'], 
        cmap='plasma', 
        alpha=0.5,
        label='Simulated Portfolios'
    )

    fig.colorbar(sc, ax=ax, label='Sharpe Ratio')

    ax.set_xlabel('Volatility')
    ax.set_ylabel('Return')
    ax.set_title(title)

    ax.scatter(
        market_volatility, market_return,
        color='red', marker='o', s=100,
        label='Market'
    )

    optimal_idx = simulated_portfolios['Sharpe Ratio'].idxmax()
    optimal_portfolio = simulated_portfolios.loc[optimal_idx]
    optimal_weights = weights_record[:, optimal_idx]

    annual_returns = daily_returns.mean() * TRADING_DAYS
    annual_cov = daily_returns.cov() * TRADING_DAYS
    
    n_points = 50
    show_cml = True
    show_ew = True
    show_gmv = True
    
    plot_ef_ax(
        n_points=n_points,
        expected_returns=annual_returns,
        cov=annual_cov,
        style='.-',
        legend=False,
        show_cml=show_cml,
        riskfree_rate=risk_free_rate,
        show_ew=show_ew,
        show_gmv=show_gmv,
        ax=ax
    )

    ax.scatter(
        optimal_portfolio['Volatility'], 
        optimal_portfolio['Return'],
        color='green', marker='*', s=200,
        label='Optimal Portfolio'
    )

    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.08),
        fancybox=True,
        shadow=True,
        ncol=3
    )

    weight_text = (
        "Optimal Weights:\n" +
        "\n".join([f"-{asset}: {weight*100:.2f}%" for asset, weight in zip(assets, optimal_weights)])
    )
    ax.text(
        1.19, 1.0,
        weight_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", edgecolor='green', facecolor='white'),
        ha='left'
    )

    optimal_text = (
        f"Optimized Portfolio\n"
        f"- Return: {optimal_portfolio['Return']*100:.2f}%\n"
        f"- Volatility: {optimal_portfolio['Volatility']*100:.2f}%\n"
        f"- Sharpe Ratio: {optimal_portfolio['Sharpe Ratio']:.2f}"
    )
    ax.text(
        1.19, 0.52,
        optimal_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", edgecolor='green', facecolor='white'),
        ha='left'
    )

    market_text = (
        f"Market\n"
        f"- Return: {market_return*100:.2f}%\n"
        f"- Volatility: {market_volatility*100:.2f}%\n"
        f"- Risk Free Rate: {risk_free_rate*100:.2f}%\n"
        f"- Sharpe Ratio: {market_sharpe:.2f}"
    )
    ax.text(
        1.19, 0.35,
        market_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", edgecolor='green', facecolor='white'),
        ha='left'
    )

    plt.tight_layout()
    plt.savefig("efficient_frontier_monter_carlo.png", dpi=300, bbox_inches='tight')
    plt.show()


def main(assets=None, start_date='2020-01-01', end_date='2023-01-01',
         market='SPY', risk_free_rate=0.0, num_portfolios=100_000):
    """
    Main function to run the portfolio optimization using Monte Carlo simulation.

    :param assets: List of asset tickers.
    :param start_date: Start date for data collection (string).
    :param end_date: End date for data collection (string).
    :param market: Market ticker for benchmarking (string).
    :param risk_free_rate: Risk-free rate to use in Sharpe ratio.
    :param num_portfolios: Number of portfolios to simulate in Monte Carlo.
    """
    if assets is None:
        assets = ['MO', 'NWN', 'BKH', 'ED', 'PEP', 'NFG', 'KO', 'FRT', 'GPC', 'MSEX']

    asset_data = download_data(assets, start_date, end_date)
    asset_data.to_csv('market_data.csv')

    daily_returns, cov_matrix = calculate_statistics(asset_data, risk_free_rate)

    simulated_portfolios, weights_record = run_monte_carlo_simulation(
        daily_returns, cov_matrix, 
        num_portfolios=num_portfolios, 
        risk_free_rate=risk_free_rate
    )

    market_return, market_volatility, market_sharpe = get_market_statistics(
        market, start_date, end_date, risk_free_rate
    )

    plot_efficient_frontier(
        simulated_portfolios,
        weights_record,
        assets,
        market_return,
        market_volatility,
        market_sharpe,
        daily_returns, cov_matrix,
        risk_free_rate=risk_free_rate,
        title='Efficient Frontier - Monte Carlo Simulation'
    )


def portfolio_performance(weights, mean_returns, cov_matrix, trading_days=252):
    """
    Given weights, return annualized portfolio return and volatility.
    """
    returns = np.sum(mean_returns * weights) * trading_days
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(trading_days)
    return returns, volatility

def min_volatility(weights, mean_returns, cov_matrix):
    """
    Objective function: we want to minimize volatility.
    """
    return portfolio_performance(weights, mean_returns, cov_matrix)[1]

def efficient_frontier(mean_returns, cov_matrix, num_points=50):
    """
    Calculates the Efficient Frontier by iterating over possible target returns
    and finding the portfolio with minimum volatility for each target return.
    Returns arrays of frontier volatilities, returns, and the corresponding weights.
    """
    results = []
    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), num_points)
    
    num_assets = len(mean_returns)
    init_guess = num_assets * [1. / num_assets,]
    bounds = tuple((0,1) for _ in range(num_assets))
    
    for ret in target_returns:
        constraints = (
            {'type':'eq', 'fun': lambda w: np.sum(w) - 1}, 
            {'type':'eq', 'fun': lambda w: portfolio_performance(w, mean_returns, cov_matrix)[0] - ret}
        )
        
        result = minimize(min_volatility, 
                          init_guess, 
                          args=(mean_returns, cov_matrix),
                          method='SLSQP',
                          bounds=bounds,
                          constraints=constraints)
        if result.success:
            vol = portfolio_performance(result.x, mean_returns, cov_matrix)[1]
            results.append((vol, ret, result.x))
    
    results = sorted(results, key=lambda x: x[0])
    
    frontier_volatility = [res[0] for res in results]
    frontier_returns = [res[1] for res in results]
    frontier_weights = [res[2] for res in results]
    
    return frontier_volatility, frontier_returns, frontier_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Monte Carlo Portfolio Optimization')
    parser.add_argument('--assets', nargs='+', default=None,
                        help='List of assets to include in the portfolio')
    parser.add_argument('--start', type=str, default='2020-01-01',
                        help='Start date for historical data (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2025-01-01',
                        help='End date for historical data (YYYY-MM-DD)')
    parser.add_argument('--market', type=str, default='SPY',
                        help='Market ticker for benchmarking')
    parser.add_argument('--risk_free_rate', type=float, default=0.05,
                        help='Risk-free rate for Sharpe ratio calculation')
    parser.add_argument('--portfolios', type=int, default=100_000,
                        help='Number of random portfolios to simulate')

    args = parser.parse_args()

    main(
        assets=args.assets,
        start_date=args.start,
        end_date=args.end,
        market=args.market,
        risk_free_rate=args.risk_free_rate,
        num_portfolios=args.portfolios
    )