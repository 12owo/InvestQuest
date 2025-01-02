import pandas as pd
import streamlit as st 
import yfinance as yf
from sklearn.preprocessing import StandardScaler
st.title(' 📈 InvestQuest')
st.info('Start investing now!') 
import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Function to fetch data
def fetch_data(ticker):
    stock_data = yf.Ticker(ticker)
    df = stock_data.history(period="1y")
    return df

# Preprocess the data
def preprocess_data(df):
    df = df[['Close']]
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
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

ticker = st.text_input("Enter Stock Ticker", value="GOOGL", "TSLA", "AAPL")
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


