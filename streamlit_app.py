import pandas as pd
import streamlit as st 
import yfinance as yf

st.title(' 📈 InvestQuest')
st.info('Start investing now!') 

# Define the stock ticker and the time period
ticker = "GOOGL"  # Example: Google stock
period = "1y"  # Example: 1 year of data

# Fetch the historical market data
stock_data = yf.Ticker(ticker)
df = stock_data.history(period=period)

# Display the data
print(df)
