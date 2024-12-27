# Monte Carlo Portfolio Optimization

This repository demonstrates how to **simulate and optimize a portfolio** of assets using a **Monte Carlo** approach and a **mathematical Efficient Frontier** calculation. It identifies the portfolio with the highest Sharpe Ratio and compares its performance to a market benchmark (e.g., SPY).

## Table of Contents

1. [Project Description](#project-description)  
2. [Features](#features)  
3. [Installation](#installation)  
4. [Usage](#usage)  
5. [File Structure](#file-structure)  
6. [Configuration](#configuration)  
7. [Disclaimer](#disclaimer)  
8. [License](#license)  
9. [Contributing](#contributing)

---

## Project Description

This project leverages **Modern Portfolio Theory** and **Monte Carlo Simulation** to explore different portfolio allocations and find an optimal mix of assets (i.e., the **highest Sharpe ratio**). Additionally, it plots the **Efficient Frontier** both from random portfolios and a mathematical optimizer:

- **Monte Carlo Simulation**: Randomly allocates weights among assets thousands of times to observe a broad range of risk-return outcomes.
- **Analytical Efficient Frontier**: Uses an optimizer (via `scipy.optimize`) to systematically compute the theoretical minimum volatility portfolio for each level of return.

By comparing the random results and the analytical frontier, you get a clearer picture of your portfolio’s potential risk and return.

![](https://github.com/JordiCorbilla/monte-carlo-portfolio-optimization/raw/main/efficient_frontier_monter_carlo.png)

---

## Features

1. **Data Retrieval**: Automatically fetches historical data from Yahoo Finance.  
2. **Monte Carlo Simulation**: Generates up to hundreds of thousands of random portfolio allocations.  
3. **Efficient Frontier**: An analytical solution using constrained optimization.  
4. **Benchmark Comparison**: Compares the optimal portfolio to a chosen market index (e.g., `SPY`).  
5. **Visualization**: Plots the Monte Carlo cloud, the Efficient Frontier, and highlights key portfolios (Market, Optimal, etc.).  
6. **Export**: Saves the final plot as an image (`.png`) and can export data to CSV.

---

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/JordiCorbilla/monte-carlo-portfolio-optimization.git
cd monte-carlo-portfolio-optimization
```

## Usage

You can directly run the script from the command line:

```bash
python monte_carlo_portfolio.py \
    --assets AAPL MSFT TSLA \
    --start 2020-01-01 \
    --end 2025-01-01 \
    --market SPY \
    --risk_free_rate 0.05 \
    --portfolios 100000
```

Command-Line Arguments:
--assets: List of tickers (space-separated). If not specified, the script defaults to a set of Dividend Kings (source: DividendGrowthInvestor.com).
--start: Start date for historical data (YYYY-MM-DD). Default: 2020-01-01.
--end: End date for historical data (YYYY-MM-DD). Default: 2025-01-01.
--market: Benchmark market ticker. Default: SPY.
--risk_free_rate: Risk-free rate (annualized). Default: 0.05.
--portfolios: Number of random portfolio simulations. Default: 100000.

When the script finishes:

- A PNG file named efficient_frontier_monter_carlo.png is saved locally.
- A CSV file called market_data.csv is saved with the downloaded prices.
