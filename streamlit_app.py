import pandas as pd
import streamlit as st 
import yfinance as yf

from sklearn.preprocessing import StandardScaler

# Drop any rows with missing values
df.dropna(inplace=True)

# Create additional features (e.g., moving averages)
df['MA10'] = df['Close'].rolling(window=10).mean()
df['MA50'] = df['Close'].rolling(window=50).mean()

# Normalize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['Close', 'MA10', 'MA50']])

# Convert the scaled data back to a DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=['Close', 'MA10', 'MA50'], index=df.index)
'''
st.title(' 📈 InvestQuest')
st.info('Start investing now!') 

# Define the stock ticker and the time period
ticker = "GOOGL"  # Example: Google stock
period = "1y"  # Example: 1 year of data

# Fetch the historical market data
stock_data = yf.Ticker(ticker)
df = stock_data.history(period=period)

# Display the data
print(df)'''
