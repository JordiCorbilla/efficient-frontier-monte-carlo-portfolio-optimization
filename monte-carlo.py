import argparse
import pandas as pd
from riskoptima import RiskOptima

"""
 Author: Jordi Corbilla

 Date: 02/02/2025

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

def main(assets, weights, labels,
         start_date='2020-01-01', end_date='2023-01-01',
         market_benchmark='SPY', risk_free_rate=0.05, num_portfolios=100_000):
    """
    Main function to run the portfolio optimization using Monte Carlo simulation.

    :param assets: Comma-separated asset tickers.
    :param weights: Comma-separated asset weights.
    :param labels: Comma-separated asset labels.
    :param start_date: Start date for historical data (YYYY-MM-DD).
    :param end_date: End date for historical data (YYYY-MM-DD).
    :param market_benchmark: Market ticker for benchmarking.
    :param risk_free_rate: Risk-free rate for Sharpe ratio calculation.
    :param num_portfolios: Number of random portfolios to simulate.
    """
    # Convert string arguments into lists
    if isinstance(assets, str):
        assets = [x.strip() for x in assets.split(",")]
    if isinstance(weights, str):
        weights = [float(x.strip()) for x in weights.split(",")]
    if isinstance(labels, str):
        labels = [x.strip() for x in labels.split(",")]

    # Create the asset table DataFrame
    asset_table = pd.DataFrame(list(zip(assets, weights, labels)), columns=['Asset', 'Weight', 'Label'])

    # Call the new RiskOptima function with the asset table and other parameters
    RiskOptima.plot_efficient_frontier_monte_carlo(
        asset_table,
        start_date=start_date,
        end_date=end_date,
        risk_free_rate=risk_free_rate,
        num_portfolios=num_portfolios,
        market_benchmark=market_benchmark,
        set_ticks=False,
        x_pos_table=1.15,    # Position for the weight table on the plot
        y_pos_table=0.52,    # Position for the weight table on the plot
        title=f'Efficient Frontier - Monte Carlo Simulation {start_date} to {end_date}'
    )

if __name__ == "__main__":
    # Define default data as comma-separated strings.
    default_assets = "MO,NWN,BKH,ED,PEP,NFG,KO,FRT,GPC,MSEX"
    default_weights = "0.04,0.14,0.01,0.01,0.09,0.16,0.06,0.28,0.16,0.05"
    default_labels = ("Altria Group Inc.,Northwest Natural Gas,Black Hills Corp.,Con Edison,"
                      "PepsiCo Inc.,National Fuel Gas,Coca-Cola Company,Federal Realty Inv. Trust,"
                      "Genuine Parts Co.,Middlesex Water Co.")

    parser = argparse.ArgumentParser(description='Monte Carlo Portfolio Optimization')
    parser.add_argument('--assets', type=str, default=default_assets,
                        help='Comma-separated list of asset tickers')
    parser.add_argument('--weights', type=str, default=default_weights,
                        help='Comma-separated list of asset weights')
    parser.add_argument('--labels', type=str, default=default_labels,
                        help='Comma-separated list of asset labels (company names)')
    parser.add_argument('--start', type=str, default='2024-01-01',
                        help='Start date for historical data (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=RiskOptima.get_previous_working_day(),
                        help='End date for historical data (YYYY-MM-DD)')
    parser.add_argument('--market_benchmark', type=str, default='SPY',
                        help='Market ticker for benchmarking')
    parser.add_argument('--risk_free_rate', type=float, default=0.05,
                        help='Risk-free rate for Sharpe ratio calculation')
    parser.add_argument('--portfolios', type=int, default=100_000,
                        help='Number of random portfolios to simulate')

    args = parser.parse_args()

    main(
        assets=args.assets,
        weights=args.weights,
        labels=args.labels,
        start_date=args.start,
        end_date=args.end,
        market_benchmark=args.market_benchmark,
        risk_free_rate=args.risk_free_rate,
        num_portfolios=args.portfolios
    )