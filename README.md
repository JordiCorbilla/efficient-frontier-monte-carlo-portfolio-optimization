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
   git clone https://github.com/YourUsername/YourRepoName.git
   cd YourRepoName
