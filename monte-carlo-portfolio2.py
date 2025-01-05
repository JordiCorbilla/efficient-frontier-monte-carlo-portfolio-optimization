import numpy as np
import argparse
from riskoptima import RiskOptima

"""
 Author: Jordi Corbilla

 Date: 05/01/2024

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

def main(assets=None, start_date='2020-01-01', end_date='2023-01-01',
         market='SPY', risk_free_rate=0.05, num_portfolios=100_000):
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
        assets = ['TSLA', 'AMZN', 'AAPL', 'MSFT', 'NVDA']

    asset_data = RiskOptima.download_data_yfinance(assets, start_date, end_date)
    asset_data.to_csv('market_data.csv')

    daily_returns, cov_matrix = RiskOptima.calculate_statistics(asset_data, risk_free_rate)

    simulated_portfolios, weights_record = RiskOptima.run_monte_carlo_simulation(
        daily_returns, cov_matrix, 
        num_portfolios=num_portfolios, 
        risk_free_rate=risk_free_rate
    )

    market_return, market_volatility, market_sharpe = RiskOptima.get_market_statistics(
        market, start_date, end_date, risk_free_rate
    )
    
    # example of randomly assigning weights to the current portfolio
    my_current_weights = np.array([
        0.20, 
        0.20, 
        0.20, 
        0.20, 
        0.20
    ])
    
    my_current_labels = np.array([
        'Tesla', 
        'Amazon', 
        'Apple', 
        'Microsoft', 
        'NVIDIA'
    ])

    RiskOptima.plot_efficient_frontier(
        simulated_portfolios,
        weights_record,
        assets,
        market_return,
        market_volatility,
        market_sharpe,
        daily_returns, cov_matrix,
        risk_free_rate=risk_free_rate,
        title=f'Efficient Frontier - Monte Carlo Simulation {start_date} to {end_date}',
        current_weights=my_current_weights,
        current_labels=my_current_labels,
        start_date=start_date,
        end_date=end_date,
        set_ticks=False,
        x_pos_table=1.20,
        y_pos_table=0.56
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Monte Carlo Portfolio Optimization')
    parser.add_argument('--assets', nargs='+', default=None,
                        help='List of assets to include in the portfolio')
    parser.add_argument('--start', type=str, default='2024-01-01',
                        help='Start date for historical data (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=RiskOptima.get_previous_working_day(),
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