import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from database_module import StockDB, DatabaseError

st.set_page_config(page_title="Yahoo Finance Stock Data Viewer", layout="wide")

def fetch_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    return yf.download(ticker, start=start_date, end=end_date)

def create_stock_plot(df: pd.DataFrame, ticker: str, close_color: str, volume_color: str) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12, 
                        row_heights=[0.7, 0.3], subplot_titles=('', ''))
def main():
    st.title("Yahoo Finance Stock Data Viewer")

    # Fetch the stock list from the database
    stock_db = StockDB()
    try:
        all_stocks = stock_db.get_all_stock_symbols()
    except DatabaseError as e:
        st.sidebar.text(f"Failed to fetch stock list. Error: {e}")
        return

    # Display fetched stocks in the sidebar
    selected_stock = st.sidebar.selectbox("Select a Stock", all_stocks)

    # Add date pickers for selecting the start and end dates
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2022-01-01'))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime('2023-01-01'))

    # Get the yfinance data for the selected stock and display it
    df = fetch_data(selected_stock, start_date, end_date)

    # Colors for closing price and volume
    close_color = "#1f77b4"
    volume_color = "#ff7f00"

    fig = create_stock_plot(df, selected_stock, close_color, volume_color)
    st.plotly_chart(fig, use_container_width=True)

# Execute the main function
if __name__ == '__main__':
    main()
