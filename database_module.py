# database_module.py
import pandas as pd
import psycopg2


DB_PARAMS = {
    'dbname': 'stock_app',
    'user': 'postgres',
    'password': 'Psql_3875',
    'host': 'localhost',
    'port': 5432
}

class DatabaseError(Exception):
    """Custom exception for database related errors."""
    pass

class BaseDatabase:
    def __init__(self, db_params=DB_PARAMS):
        self.db_params = db_params
        self.connection = None
        self.cursor = None

    def connect(self):
        try:
            self.connection = psycopg2.connect(**self.db_params)
            self.cursor = self.connection.cursor()
        except psycopg2.Error as e:
            raise DatabaseError(f"Connection error: {e}")

    def close(self):
        if self.connection:
            self.cursor.close()
            self.connection.close()

    def commit(self):
        try:
            self.connection.commit()
        except psycopg2.Error as e:
            raise DatabaseError(f"Commit error: {e}")

class UserDB(BaseDatabase):
    def add_user(self, username, password, email):
        try:
            self.connect()
            query = "INSERT INTO Users (Username, Password, Email) VALUES (%s, %s, %s)"
            self.cursor.execute(query, (username, password, email))
            self.commit()
        except psycopg2.Error as e:
            raise DatabaseError(f"Error adding user: {e}")
        finally:
            self.close()

    def get_user(self, user_id):
        try:
            self.connect()
            query = "SELECT * FROM Users WHERE UserID = %s"
            self.cursor.execute(query, (user_id,))
            return self.cursor.fetchone()
        except psycopg2.Error as e:
            raise DatabaseError(f"Error getting user data: {e}")
        finally:
            self.close()

class StockDB(BaseDatabase):
    def add_stock(self, symbol, name, market_cap, country, ipo_year, sector, industry):
        try:
            self.connect()
            query = """
            INSERT INTO Stocks (Symbol, Name, MarketCap, Country, IPOYear, Sector, Industry) 
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            self.cursor.execute(query, (symbol, name, market_cap, country, ipo_year, sector, industry))
            self.commit()
        except psycopg2.Error as e:
            raise DatabaseError(f"Error adding stock: {e}")
        finally:
            self.close()

    def get_stock(self, symbol):
        try:
            self.connect()
            query = "SELECT * FROM Stocks WHERE Symbol = %s"
            self.cursor.execute(query, (symbol,))
            return self.cursor.fetchone()
        except psycopg2.Error as e:
            raise DatabaseError(f"Error getting stock data: {e}")
        finally:
            self.close()
    
    def get_all_stock_symbols(self):
        try:
            self.connect()
            query = "SELECT Symbol FROM Stocks"
            self.cursor.execute(query)
            return [row[0] for row in self.cursor.fetchall()]
        except psycopg2.Error as e:
            raise DatabaseError(f"Error fetching stock symbols: {e}")
        finally:
            self.close()


    def stock_exists(self, symbol):
        try:
            self.connect()
            query = "SELECT 1 FROM Stocks WHERE Symbol = %s"
            self.cursor.execute(query, (symbol,))
            return self.cursor.fetchone() is not None
        except psycopg2.Error as e:
            raise DatabaseError(f"Error checking if stock exists: {e}")
        finally:
            self.close()



    def get_stock_data_between_dates(self, symbol, start_date=None, end_date=None):
        try:
            self.connect()
            
            if start_date == end_date:
                query = """
                SELECT * FROM StockPrices
                WHERE Symbol = %s
                ORDER BY Date DESC
                """
                params = (symbol,)
            else:
                query = """
                SELECT * FROM StockPrices
                WHERE Symbol = %s AND Date BETWEEN %s AND %s
                ORDER BY Date DESC
                """
                params = (symbol, start_date, end_date)
            
            self.cursor.execute(query, params)
            
            # Get the column names from the cursor description
            column_names = [desc[0] for desc in self.cursor.description]
            
            # Convert the fetched data to a DataFrame
            df = pd.DataFrame(self.cursor.fetchall(), columns=column_names)
           # df = df.set_index('date') # Note: Ensure the column name is 'Date' and not 'date'          
            return df
            
        except psycopg2.Error as e:
            raise DatabaseError(f"Error fetching stock data between dates: {e}")
            
        finally:
            self.close()

class SuggestionDB(BaseDatabase):
    def add_suggestion(self, stock_symbol, suggested_price, recommendation, rationale):
        try:
            self.connect()
            query = """
            INSERT INTO Suggestions (StockID, SuggestedPrice, Recommendation, Rationale) 
            VALUES (%s, %s, %s, %s)
            """
            self.cursor.execute(query, (stock_symbol, suggested_price, recommendation, rationale))
            self.commit()
        except psycopg2.Error as e:
            raise DatabaseError(f"Error adding suggestion: {e}")
        finally:
            self.close()

    def get_suggestion(self, suggestion_id):
        try:
            self.connect()
            query = "SELECT * FROM Suggestions WHERE SuggestionID = %s"
            self.cursor.execute(query, (suggestion_id,))
            return self.cursor.fetchone()
        except psycopg2.Error as e:
            raise DatabaseError(f"Error getting suggestion data: {e}")
        finally:
            self.close()
