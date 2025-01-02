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

# Train a linear regression model
def train_model(df):
    X = df[['MA10', 'MA50']]
    y = df['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_test, y_pred

ticker = st.selectbox ("Choose a stock ticker", ["GOOGL", "AAPL", "MSFT", "AMZN", "NVDA", "WIT", "ORCL", "IBM", "META", "TSLA"])
n_years = st.slider("Years of prediction: " , 1 , 10) #this creates a slider for how long you want to predict the stocks for, this particular one i have mentioned 1-4 years.
period = n_years * 365 

#loading the data
@st.cache_data
def load_data(ticker):  #ticker is the stock name thing
  data = yf.download(ticker, start_date, TODAY) #downloading from start date to today date
  data.reset_index(inplace=True) #put the date in the first column
  return data

data_load_state = st.text("Loading data...")
data = load_data(selected_stock)
data_load_state.text("Loading data...done!")
if ticker:
    df = fetch_data(ticker)
    df = preprocess_data(df)
    model, y_test, y_pred = train_model(df)
    
    st.subheader("Historical Stock Prices")
    st.line_chart(df['Close'])
    
    st.subheader("Predictions vs Actual")
    plt.figure(figsize=(14, 7))
    plt.plot(y_test.index, y_test, label='Actual Close Price')
    plt.plot(y_test.index, y_pred, label='Predicted Close Price')
    plt.legend()
    st.pyplot(plt)

    st.write("Mean Squared Error:", ((y_test - y_pred) ** 2).mean())


