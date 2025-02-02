# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 13:51:54 2025

@author: jordi
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from riskoptima import RiskOptima

st.set_page_config(page_title="Portfolio Optimisation", layout="wide")
st.title("Portfolio Optimisation with an Interactive Asset Grid")

default_assets = [
    {"Asset": "MO",    "Weight": 0.04, "Label": "Altria Group Inc."},
    {"Asset": "NWN",   "Weight": 0.14, "Label": "Northwest Natural Gas"},
    {"Asset": "BKH",   "Weight": 0.01, "Label": "Black Hills Corp."},
    {"Asset": "ED",    "Weight": 0.01, "Label": "Con Edison"},
    {"Asset": "PEP",   "Weight": 0.09, "Label": "PepsiCo Inc."},
    {"Asset": "NFG",   "Weight": 0.16, "Label": "National Fuel Gas"},
    {"Asset": "KO",    "Weight": 0.06, "Label": "Coca-Cola Company"},
    {"Asset": "FRT",   "Weight": 0.28, "Label": "Federal Realty Inv. Trust"},
    {"Asset": "GPC",   "Weight": 0.16, "Label": "Genuine Parts Co."},
    {"Asset": "MSEX",  "Weight": 0.05, "Label": "Middlesex Water Co."}
]

st.subheader("Asset Details")
asset_df = st.data_editor(pd.DataFrame(default_assets), num_rows="dynamic", key="asset_editor")
st.write("Current asset table:")
st.dataframe(asset_df)

st.sidebar.header("Optimisation Parameters")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-01-01"))
default_end = str(RiskOptima.get_previous_working_day())
end_date = st.sidebar.text_input("End Date (YYYY-MM-DD)", value=default_end)
market_benchmark = st.sidebar.text_input("Market Benchmark Ticker", value="SPY")
risk_free_rate = st.sidebar.number_input("Risk Free Rate", value=0.05, step=0.01, format="%.2f")
num_portfolios = st.sidebar.number_input("Number of Simulations", value=10000, step=1000)

if st.sidebar.button("Run Optimisation"):
    st.write("### Running Monte Carlo Simulation...")
    start_date_str = start_date.strftime("%Y-%m-%d")
    
    RiskOptima.plot_efficient_frontier_monte_carlo(
        asset_df,
        start_date=start_date_str,
        end_date=end_date,
        risk_free_rate=risk_free_rate,
        num_portfolios=num_portfolios,
        market_benchmark=market_benchmark,
        set_ticks=False,
        x_pos_table=1.15,  # Position for the weight table on the plot
        y_pos_table=0.52,  # Position for the weight table on the plot
        title=f'Efficient Frontier - Monte Carlo Simulation {start_date_str} to {end_date}'
    )
    
    st.pyplot(plt.gcf())
    plt.clf()
