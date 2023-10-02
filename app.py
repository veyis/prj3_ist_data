import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Import necessary classes from the database module
from database_module import StockDB, DatabaseError

# Set the Streamlit Page Configurations
st.set_page_config(page_title="Stock Data Viewer", layout="wide")


close_color = "#1f77b4"
volume_color = "#5D6D7E" #"#ff7f00"


def fetch_data(ticker: str, selected_range: str) -> pd.DataFrame:
    current_date = datetime.now()
    st.write(ticker, selected_range)

    range_dict = {
        "1W": timedelta(weeks=1),
        "1M": timedelta(weeks=4),
        "3M": timedelta(weeks=12),
        "6M": timedelta(weeks=24),
        "YTD": datetime(current_date.year, 1, 1) - datetime(current_date.year - 1, 12, 31),
        "1Y": timedelta(weeks=52),
        "2Y": timedelta(weeks=104),
        "3Y": timedelta(weeks=156),
        "5Y": timedelta(weeks=260),
        "MAX": timedelta(weeks=0)
    }

    # Correcting for "YTD"
    if selected_range == "YTD":
        start_date = datetime(current_date.year, 1, 1)
    elif selected_range == "MAX":
        start_date = datetime.now() 
    else:
        start_date = current_date - range_dict.get(selected_range, timedelta(weeks=1))



    stock_db = StockDB()
    end_date = datetime.now()
    df = pd.DataFrame()

    try:
        df = stock_db.get_stock_data_between_dates(ticker, start_date, end_date)
        return df
    except DatabaseError as e:
        st.sidebar.text(f"Failed to fetch stock data. Error: {e}")
        return pd.DataFrame()
    finally:
        stock_db.close()




def create_stock_plot(df: pd.DataFrame, ticker: str, close_color: str, volume_color: str) -> go.Figure:
   
    df = df.set_index('date') # Note: Ensure the column name is 'Date' and not 'date'
    df.sort_index(inplace=True)
    df.reset_index(inplace=True)
    df['date2'] = df['date']



    #st.write(df['date2'])





    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12, 
                        row_heights=[0.7, 0.3], subplot_titles=('', ''))
    

    
    # Adding Scatter plot for Closing Price
    fig.add_trace(go.Candlestick(x=df["date2"], open=df['openingprice'], high=df['high'], low=df['low'], close=df['closingprice'], 
                    increasing_line_color='green', 
                    decreasing_line_color='red',name="closingprice"))

    # Adding Bar plot for Volume
  
    fig.add_trace(go.Bar(x=df['date2'],y=df['volume'], name="volume", marker_color='gray',opacity=0.5), row=2, col=1)
  

    # Updating the layout
    fig.update_layout(title=f"Closing Price & Volume for {ticker}",
                      height=500, width=1000)

    # Update x and y axes
    #fig.update_xaxes(title_text="Date", row=1, col=1)  # Update for the first subplot
    #fig.update_xaxes(title_text="Date", row=2, col=1)  # Update for the second subplot
    fig.update_yaxes(title_text="Closing Price", row=1, col=1)
    fig.update_yaxes(title_text="volume", row=2, col=1)



    # # Hide the range slider
    fig.update_layout(xaxis_rangeslider_visible=False)

    return fig


    # st.write("Candlestick Chart")
    # st.write(df.columns)
    # st.write(df.index)
    # st.write(df)

    # fig = go.Figure(data=[go.Candlestick(x=df.index,
    #             open=df['openingprice'],
    #             high=df['high'],
    #             low=df['low'],
    #             close=df['closingprice'])


def main():
    st.title("Stock Data Viewer")
    close_color = "#1f77b4"
    volume_color = "#ff7f00"

    # Fetch the stock list from the database
    stock_db = StockDB()
    try:
        all_stocks = stock_db.get_all_stock_symbols()
    except DatabaseError as e:
        st.sidebar.text(f"Failed to fetch stock list. Error: {e}")
        return

    selected_stock = st.sidebar.selectbox("Select a Stock", all_stocks)

    # Add date pickers for selecting the start and end dates
  
    data_period = st.sidebar.selectbox("Select a Data Range", ["1M", "3M", "6M","YTD", "1Y", "2Y", "3Y", "5Y", "MAX"], index=7)


    selected_model = st.sidebar.selectbox("Select a Model", ["Linear Regression", "Logistic Regression", "Decision Trees", "Support Vector Machines (SVM)", "K-Nearest Neighbors (KNN)", "Random Forest", "XGBoost", "K-Means Clustering", "Convolutional Neural Networks (CNN)", "Recurrent Neural Networks (RNN)", "Long Short-Term Memory (LSTM)", "BERT", "Q-Learning"])


    

    tb1, tb2, tb3,tb4,tb5,tb6, tb7, tb8, tb9, tb10, tb11,tb12,tb13,tb14,tb15,tb16,tb17,tb18  = st.tabs(["1W","1M", "3M", "6M","YTD", "1Y", "2Y", "3Y", "5Y", "MAX", "Data","Financial","Income","Balance","Cashflow","Earnings","Analysts","Predictions"])
    

    with tb1:
        time_range = "1M"
        df = fetch_data(selected_stock, time_range)
       # fig = create_stock_plot(df, selected_stock, close_color, volume_color)
       # st.plotly_chart(fig, use_container_width=True) 


    with tb2:
        time_range = "1M"
        df = fetch_data(selected_stock, time_range)
        fig = create_stock_plot(df, selected_stock, close_color, volume_color)
        st.plotly_chart(fig, use_container_width=True)

    with tb3:
        time_range = "3M"
        df = fetch_data(selected_stock, time_range)
        fig = create_stock_plot(df, selected_stock, close_color, volume_color)
        st.plotly_chart(fig, use_container_width=True)

        #st.image("https://static.streamlit.io/examples/owl.jpg", width=200)

    with tb4:
        time_range = "6M"
        df = fetch_data(selected_stock, time_range)
        fig = create_stock_plot(df, selected_stock, close_color, volume_color)
        st.plotly_chart(fig, use_container_width=True)

    with tb5:
        time_range = "YTD"
        df = fetch_data(selected_stock, time_range)
        fig = create_stock_plot(df, selected_stock, close_color, volume_color)
        st.plotly_chart(fig, use_container_width=True)

    with tb6:
        time_range="1Y"
        df = fetch_data(selected_stock, time_range)
        fig = create_stock_plot(df, selected_stock, close_color, volume_color)
        st.plotly_chart(fig, use_container_width=True)

    with tb7:
        time_range="2Y"
        df = fetch_data(selected_stock, time_range)
        fig = create_stock_plot(df, selected_stock, close_color, volume_color)
        st.plotly_chart(fig, use_container_width=True)

    with tb8:
        time_range="3Y"
        df = fetch_data(selected_stock, time_range)
        fig = create_stock_plot(df, selected_stock, close_color, volume_color)
        st.plotly_chart(fig, use_container_width=True)
     
    with tb9:
        time_range="5Y"
        df = fetch_data(selected_stock, time_range)
        fig = create_stock_plot(df, selected_stock, close_color, volume_color)
        st.plotly_chart(fig, use_container_width=True)

    with tb10:
        time_range="MAX"
        df = fetch_data(selected_stock, time_range)
        fig = create_stock_plot(df, selected_stock, close_color, volume_color)
        st.plotly_chart(fig, use_container_width=True)

    with tb11:
        st.write("Data")
        df = fetch_data(selected_stock, time_range)
        st.write(df)    


        fig2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=450,
            title={'text': "Speed"},
            domain={'x': [0.25, 0.75], 'y': [0.25, 0.75]}  # Adjusted domain to center the gauge
        ))

        fig2.update_layout(margin=dict(t=20, b=20, l=20, r=20))  # Reduce margins

        st.plotly_chart(fig2, use_container_width=True, height=120, width=120)  # Adjust height and width


        st.write(df)

    with tb12:
        st.write("Financial")
        df = fetch_data(selected_stock, time_range)
        st.write(df)


    with tb18:
        #st.write("Predictions")
        
        options = [
            "Linear Regression",
            "Logistic Regression",
            "Decision Trees",
            "Support Vector Machines (SVM)",
            "K-Nearest Neighbors (KNN)",
            "Random Forest",
            "XGBoost",
            "K-Means Clustering",
            "Convolutional Neural Networks (CNN)",
            "Recurrent Neural Networks (RNN)",
            "Long Short-Term Memory (LSTM)",
            "BERT",
            "Q-Learning",
        ]

        # Create the selectbox
        selected_option = st.selectbox("Select a Machine Learning Algorithm", options)

        if selected_option=="Linear Regression":
            st.write("Linear Regression")
            st.write(selected_stock, time_range)
            
             
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error
            import matplotlib.pyplot as plt        

            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df['daily_return'] = df['closingprice'].pct_change()
            df.dropna(inplace=True)  # Drop NaN values after computing return

            # Select features (X) and target (y)
            X = df[['openingprice', 'high', 'low', 'volume', 'daily_return']]
            y = df['closingprice']

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            model = LinearRegression()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            st.write(f'Mean Squared Error: {mse}')

            fig, ax = plt.subplots()
            ax.plot(y_test.index, y_test.values, label='True')
            ax.plot(y_test.index, y_pred, label='Predicted')
            ax.legend()
            st.pyplot(fig)


            st.write(df)
        


   # fig = create_stock_plot(df, selected_stock, close_color, volume_color)

   # st.plotly_chart(fig, use_container_width=True)


if __name__ == '__main__':
    main()
