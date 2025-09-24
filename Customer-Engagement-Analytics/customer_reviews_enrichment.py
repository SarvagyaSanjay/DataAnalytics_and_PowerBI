# pip install pandas nltk mysql-connector-python SQLAlchemy

import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sqlalchemy import create_engine

# Download the VADER lexicon if not already installed
nltk.download('vader_lexicon')

# Define a function to fetch data from a MySQL database
def fetch_data_from_mysql():
    # Replace with your own credentials
    user = "your_username"
    password = "your_password"
    host = "localhost"   # or your MySQL server IP
    port = "3306"        # default MySQL port
    database = "PortfolioProject_MarketingAnalytics"
    
    # Create the SQLAlchemy engine for MySQL
    engine = create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}")
    
    # Define your query
    query = "SELECT ReviewID, CustomerID, ProductID, ReviewDate, Rating, ReviewText FROM fact_customer_reviews"
    
    # Fetch the data into a DataFrame
    df = pd.read_sql(query, engine)
    
    return df

# Fetch the customer reviews data
customer_reviews_df = fetch_data_from_mysql()

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to calculate sentiment scores
def calculate_sentiment(review):
    sentiment = sia.polarity_scores(str(review))  # str() to handle NULL values safely
    return sentiment['compound']

# Function to categorize sentiment using both score and rating
def categorize_sentiment(score, rating):
    if score > 0.05:
        if rating >= 4:
            return 'Positive'
        elif rating == 3:
            return 'Mixed Positive'
        else:
            return 'Mixed Negative'
    elif score < -0.05:
        if rating <= 2:
            return 'Negative'
        elif rating == 3:
            return 'Mixed Negative'
        else:
            return 'Mixed Positive'
    else:
        if rating >= 4:
            return 'Positive'
        elif rating <= 2:
            return 'Negative'
        else:
            return 'Neutral'

# Function to bucket sentiment scores
def sentiment_bucket(score):
    if score >= 0.5:
        return '0.5 to 1.0'
    elif 0.0 <= score < 0.5:
        return '0.0 to 0.49'
    elif -0.5 <= score < 0.0:
        return '-0.49 to 0.0'
    else:
        return '-1.0 to -0.5'

# Apply sentiment analysis
customer_reviews_df['SentimentScore'] = customer_reviews_df['ReviewText'].apply(calculate_sentiment)
customer_reviews_df['SentimentCategory'] = customer_reviews_df.apply(
    lambda row: categorize_sentiment(row['SentimentScore'], row['Rating']), axis=1)
customer_reviews_df['SentimentBucket'] = customer_reviews_df['SentimentScore'].apply(sentiment_bucket)

# Display results
print(customer_reviews_df.head())

# Save to CSV
customer_reviews_df.to_csv('fact_customer_reviews_with_sentiment.csv', index=False)
