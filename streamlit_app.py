import streamlit as st
st.title(' 📈 InvestQuest')
st.info('Start investing now!') 


import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Function to fetch data
def fetch_data(ticker):
    stock_data = yf.Ticker(ticker)
    df = stock_data.history(period="10y")
    return df
#reprocess the data
def preprocess_data(df):
    # Select the 'Close', 'Open', and 'Volume' columns
    df = df[['Close', 'Open', 'Volume']]
    # Calculate the 10-day and 50-day moving averages for the 'Close' column
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    # Drop rows with missing values
    df.dropna(inplace=True)
    return df

ticker = st.selectbox ("Choose a stock ticker", ["GOOGL", "AAPL", "MSFT", "AMZN", "NVDA", "WIT", "ORCL", "IBM", "META", "TSLA"])
n_years = st.slider("Years of prediction: " , 1 , 10) #this creates a slider for how long you want to predict the stocks for, this particular one i have mentioned 1-4 years.
period = n_years * 365 

# To run the app, use the command: streamlit run streamlit_app.py

# Create additional features (e.g., moving averages)
df['MA10'] = df['Close'].rolling(window=10).mean()
df['MA50'] = df['Close'].rolling(window=50).mean()

# Normalize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['Close', 'MA10', 'MA50']])

# Convert the scaled data back to a DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=['Close', 'MA10', 'MA50'], index=df.index)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Split the data into features (X) and target (y)
X = df_scaled[['MA10', 'MA50']]
y = df_scaled['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

import matplotlib.pyplot as plt

plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Close'], label='Actual Close Price')
plt.plot(df.index[-len(y_test):], predictions, label='Predicted Close Price')
plt.legend()
plt.show()

# Drop any rows with missing values
df.dropna(inplace=True)
# Check if df is a DataFrame
if isinstance(df, pd.DataFrame):
    df.dropna(inplace=True)
else:
    print("Error: df is not a DataFrame")

# Define the stock ticker and the time period
ticker = "GOOGL"  # Example: Google stock
period = "1y"  # Example: 1 year of data

# Fetch the historical market data
stock_data = yf.Ticker(ticker)
df = stock_data.history(period=period)

# Display the data
print(df)


