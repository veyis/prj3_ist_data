# https://eodhd.com/api/fundamentals/AAPL.US?api_token=650ede01c112b4.91562153
# https://eodhd.com/api/insider-transactions?api_token=YOUR_API_TOKEN

import requests
import pandas as pd
import streamlit as st
from api_info import api_key

st.set_page_config(page_title="EOD Historical Data", layout="wide")
st.header("EODHD Detailed Stock Screener")
ticker = st.sidebar.text_input("Ticker", "AAPL.US")  
data_type = st.sidebar.selectbox("Data Type", options= ["Fundamental Data", "Insider Transactions", "Stock News and Social Media Sentiment"])


if data_type == "Fundamental Data":

    #url = f"https://eodhd.com/api/fundamentals/{ticker}?api_token={api_key}"
    url = f"https://eodhd.com/api/fundamentals/{ticker}?api_token=demo"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError if the HTTP request returned an unsuccessful status code.
        data = response.json()
        fundemental_data = st.sidebar.selectbox("Financial Data", options= list(data.keys()))
        if fundemental_data == "Financials":
            statement = st.sidebar.selectbox("Financial Statement", options=["Balance Sheet", "Income Statement", "Cash Flow Statement"])
            df=pd.DataFrame(data["Financials"]['statement']['quarterly'])
            st.write(df)
        else:
            try:
                df = pd.DataFrame(data[fundemental_data])
                st.write(df)
            except:
                st.write(data[fundemental_data])

        
    except requests.RequestException as e:
        print(f"An error occurred: {e}")


if data_type == "Insider Transactions":
    url = f"https://eodhd.com/api/insider-transactions?api_token={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError if the HTTP request returned an unsuccessful status code.
        data = response.json()
        df = pd.DataFrame(data)
        st.write(df)
    except requests.RequestException as e:
        print(f"An error occurred: {e}")


if data_type == "Stock News and Social Media Sentiment":
    url = f"https://eodhd.com/api/insider-holdings?api_token={api_key}"
    url="https://eodhd.com/api/sentiments?s=btc-usd.cc,aapl&from=2022-01-01&to=2022-04-22&&api_token=650ede01c112b4.91562153"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError if the HTTP request returned an unsuccessful status code.
        data = response.json()
        df = pd.DataFrame(data)
        st.write(df)
    except requests.RequestException as e:
        print(f"An error occurred: {e}")

