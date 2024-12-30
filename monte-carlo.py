import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import argparse
from datetime import datetime
from riskoptima import RiskOptima

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

def plot_efficient_frontier(simulated_portfolios, weights_record, assets, 
                            market_return, market_volatility, market_sharpe, 
                            daily_returns, cov_matrix,
                            risk_free_rate=0.0, title='Efficient Frontier'):
    
    fig, ax = plt.subplots(figsize=(17, 8))
    
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
        label='Market (benchmark S&P 500)'
    )

    optimal_idx = simulated_portfolios['Sharpe Ratio'].idxmax()
    optimal_portfolio = simulated_portfolios.loc[optimal_idx]
    optimal_weights = weights_record[:, optimal_idx]

    annual_returns = daily_returns.mean() * RiskOptima.get_trading_days()
    annual_cov = daily_returns.cov() * RiskOptima.get_trading_days()
    
    n_points = 50
    show_cml = True
    show_ew = True
    show_gmv = True
    
    RiskOptima.plot_ef_ax(
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
        "Optimal Portfolio Weights:\n" +
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
        f"US Market (benchmark S&P 500)\n"
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"efficient_frontier_monter_carlo_{timestamp}.png", dpi=300, bbox_inches='tight')
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Monte Carlo Portfolio Optimization')
    parser.add_argument('--assets', nargs='+', default=None,
                        help='List of assets to include in the portfolio')
    parser.add_argument('--start', type=str, default='2020-01-01',
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