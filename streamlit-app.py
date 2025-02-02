# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 13:51:54 2025

@author: jordi
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import riskoptima as RiskOptima

st.set_page_config(page_title="Monte Carlo Portfolio Optimisation", layout="wide")
st.title("Monte Carlo Portfolio Optimisation with RiskOptima")


st.sidebar.header("Input Parameters")

assets_str = st.sidebar.text_input(
    "Enter asset tickers (comma separated)",
    value="APPL,TSLA"
)

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-01-01"))
market = st.sidebar.text_input("Market Ticker for Benchmarking", value="SPY")
risk_free_rate = st.sidebar.number_input("Risk-Free Rate", value=0.05, step=0.01, format="%.2f")
num_portfolios = st.sidebar.number_input("Number of Portfolios to Simulate", value=100000, step=1000)

use_custom_portfolio = st.sidebar.checkbox("Use custom current portfolio?", value=False)
if use_custom_portfolio:
    current_weights_str = st.sidebar.text_input(
        "Current Portfolio Weights (comma separated)",
        value="0.5,0.5"
    )
    current_labels_str = st.sidebar.text_input(
        "Current Portfolio Labels (comma separated)",
        value="Apple,Tesla"
    )

run_button = st.sidebar.button("Run Optimisation")


if run_button:
    # Parse the asset tickers from the input string
    assets = [asset.strip() for asset in assets_str.split(",") if asset.strip() != ""]
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    # Parse custom portfolio inputs if provided
    if use_custom_portfolio:
        try:
            current_weights = [float(x.strip()) for x in current_weights_str.split(",") if x.strip() != ""]
            current_labels = [label.strip() for label in current_labels_str.split(",") if label.strip() != ""]
            if len(current_weights) != len(assets) or len(current_labels) != len(assets):
                st.warning("The number of custom weights or labels does not match the number of assets. Ignoring custom portfolio inputs.")
                current_weights = None
                current_labels = None
        except Exception as e:
            st.error("Error parsing custom portfolio inputs. Please ensure weights are numeric and labels are valid.")
            current_weights = None
            current_labels = None
    else:
        current_weights = None
        current_labels = None

    st.write("Running Monte Carlo simulation...")
    with st.spinner("Fetching data and running simulation..."):
        # Download asset data via yfinance
        asset_data = RiskOptima.download_data_yfinance(assets, start_date_str, end_date_str)
        # Compute daily returns and covariance matrix
        daily_returns, cov_matrix = RiskOptima.calculate_statistics(asset_data, risk_free_rate)
        # Run the Monte Carlo simulation
        simulated_portfolios, weights_record = RiskOptima.run_monte_carlo_simulation(
            daily_returns, cov_matrix, num_portfolios=num_portfolios, risk_free_rate=risk_free_rate
        )
        # Get market statistics for benchmarking
        market_return, market_volatility, market_sharpe = RiskOptima.get_market_statistics(
            market, start_date_str, end_date_str, risk_free_rate
        )
        
        # Plot the efficient frontier using RiskOptima
        plt.figure()  # Create a new figure
        RiskOptima.plot_efficient_frontier(
            simulated_portfolios,
            weights_record,
            assets,
            market_return,
            market_volatility,
            market_sharpe,
            daily_returns,
            cov_matrix,
            risk_free_rate=risk_free_rate,
            title=f'Efficient Frontier - Monte Carlo Simulation {start_date_str} to {end_date_str}',
            current_weights=current_weights,
            current_labels=current_labels,
            start_date=start_date_str,
            end_date=end_date_str,
            set_ticks=False
        )
        st.pyplot(plt.gcf())
        plt.clf()
    st.success("Optimisation complete.")
