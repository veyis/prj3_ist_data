import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.dummy import DummyRegressor
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime
from datetime import timedelta

from database_module import StockDB, DatabaseError  # adjust with your actual module and class



class StockAnalysis:
    def __init__(self, ticker: str, start_date: str, end_date: str = None):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date or datetime.now()
        self.df = self.fetch_data()

    def fetch_data(self) -> pd.DataFrame:
        stock_db = StockDB()
        df = pd.DataFrame()

        try:
            df = stock_db.get_stock_data_between_dates(self.ticker, self.start_date, self.end_date)
        except DatabaseError as e:
            print(f"Failed to fetch stock data. Error: {e}")
        finally:
            stock_db.close()

        return df

    def preprocess(self):
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df.set_index('date', inplace=True)  # Setting index to date for correct plotting
        self.df['daily_return'] = self.df['closingprice'].pct_change()
        self.df['next_day_closingprice'] = self.df['closingprice'].shift(-1)
        self.df['day_of_week'] = self.df.index.dayofweek
        self.df.dropna(inplace=True)

    def train_and_evaluate(self):
        self.preprocess()
        features = ['openingprice', 'closingprice', 'high', 'low', 'volume', 'day_of_week', 'daily_return']
        target = 'next_day_closingprice'

        X = self.df[features]
        y = self.df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Cross-Validation
        scores = cross_val_score(model, X, y, cv=5)
        print("Cross-validated Scores:", scores)
        print("Mean CV Score:", np.mean(scores))

        # Additional Metrics
        mae_test = mean_absolute_error(y_test, model.predict(X_test))
        r2_test = r2_score(y_test, model.predict(X_test))
        print(f'Mean Absolute Error for Test Data: {mae_test}')
        print(f'R-squared for Test Data: {r2_test}')

        # Dummy Model
        dummy_model = DummyRegressor(strategy='mean')
        dummy_model.fit(X_train, y_train)
        dummy_mse = mean_squared_error(y_test, dummy_model.predict(X_test))
        print(f'Mean Squared Error for Dummy Model: {dummy_mse}')

        self.df['actual_direction'] = self.df['next_day_closingprice'] > self.df['closingprice']
        self.df['predicted_price'] = model.predict(X)
        self.df['predicted_direction'] = self.df['predicted_price'] > self.df['closingprice']
        
        correct_predictions = sum(self.df['actual_direction'] == self.df['predicted_direction'])
        total_predictions = len(self.df)
        accuracy = correct_predictions / total_predictions * 100
        mse_test = mean_squared_error(y_test, model.predict(X_test))

        print(f'Model correctly predicted the direction {correct_predictions} times out of {total_predictions} ({accuracy:.2f}%)')
        print(f'Mean Squared Error for Test Data: {mse_test}')
        self.visualize_predictions()

    def visualize_predictions(self):
        fig = make_subplots(rows=3, cols=1,
                            subplot_titles=('Actual vs Predicted Closing Prices',
                                            'Scatter Plot: Actual vs Predicted Closing Prices',
                                            'Correct and Incorrect Direction Predictions'),
                            shared_xaxes=True)

        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['next_day_closingprice'], mode='lines', name='Actual'), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['predicted_price'], mode='lines', name='Predicted', line=dict(dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.df['next_day_closingprice'], y=self.df['predicted_price'], mode='markers', name='Actual vs Predicted'), row=2, col=1)
        
        correct_predictions_df = self.df[self.df['actual_direction'] == self.df['predicted_direction']]
        incorrect_predictions_df = self.df[self.df['actual_direction'] != self.df['predicted_direction']]
        
        fig.add_trace(go.Scatter(x=correct_predictions_df.index, y=[1]*len(correct_predictions_df), mode='markers', name='Correct', marker=dict(color='Green', size=10)), row=3, col=1)
        fig.add_trace(go.Scatter(x=incorrect_predictions_df.index, y=[2]*len(incorrect_predictions_df), mode='markers', name='Incorrect', marker=dict(color='Red', size=10)), row=3, col=1)
        fig.update_yaxes(row=3, col=1, tickvals=[1, 2], ticktext=['Correct', 'Incorrect'], range=[0.5, 2.5])
        
        fig.update_layout(height=1200, width=1200, title_text="Analysis of Stock Price Predictions", showlegend=True)
        fig.show()


    

if __name__ == "__main__":
    ticker = "AAPL"
    end_date = datetime.now()
    start_date = end_date - timedelta(weeks=260)

    stock_analysis = StockAnalysis(ticker, start_date, end_date)
    stock_analysis.train_and_evaluate()