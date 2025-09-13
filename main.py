import requests
import pandas as pd
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
import calendar
from sqlalchemy import create_engine, text
from config import EOD_API_KEY, TICKERS, EXCHANGE, QUESTDB_URI, START_YEAR, END_YEAR, START_MONTH, END_MONTH, START_DAY, END_DAY

# Connect to QuestDB via PostgreSQL protocol
engine = create_engine(QUESTDB_URI)

# Initialize a dictionary to log missing data
missing_data_log = {"missing_stock_data": [], "missing_news_data": []}

def fetch_daily_stock_data(ticker, exchange, api_key, start_date, end_date):
    url = (
        f"https://eodhistoricaldata.com/api/eod/{ticker}.{exchange}"
        f"?api_token={api_key}&from={start_date}&to={end_date}&period=d&fmt=json"
    )
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    # Defensive check for empty or malformed response
    if not isinstance(data, list) or len(data) == 0:
        return pd.DataFrame(columns=['ticker', 'time', 'open', 'high', 'low', 'close', 'volume'])

    df = pd.DataFrame(data)
    if 'date' not in df.columns:
        raise KeyError(f"'date' column missing in API response for {ticker} from {start_date} to {end_date}.")

    df['ticker'] = ticker
    df.rename(columns={"date": "time"}, inplace=True)
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    return df[['ticker', 'time', 'open', 'high', 'low', 'close', 'volume']]

def fetch_news_data(ticker, api_key, start_date, end_date):
    url = (
        f"https://eodhistoricaldata.com/api/news"
        f"?s={ticker}&from={start_date}&to={end_date}&api_token={api_key}&fmt=json"
    )
    response = requests.get(url)
    response.raise_for_status()
    news = response.json()

    if not isinstance(news, list) or len(news) == 0:
        return pd.DataFrame(columns=[
            'ticker', 'time', 'title', 'content', 'link', 'symbols', 'tags',
            'sentiment_polarity', 'sentiment_neg', 'sentiment_neu', 'sentiment_pos'
        ])

    records = []
    for item in news:
        sentiment = item.get("sentiment", {})
        record = {
            "ticker": ticker,
            "time": item.get("date"),
            "title": item.get("title", ""),
            "content": item.get("content", ""),
            "link": item.get("link", ""),
            "symbols": ",".join(item.get("symbols", [])) if item.get("symbols") else "",
            "tags": ",".join(item.get("tags", [])) if item.get("tags") else "",
            "sentiment_polarity": sentiment.get("polarity", None),
            "sentiment_neg": sentiment.get("neg", None),
            "sentiment_neu": sentiment.get("neu", None),
            "sentiment_pos": sentiment.get("pos", None)
        }
        records.append(record)

    df = pd.DataFrame(records)
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    return df

def write_to_questdb(df, table_name):
    with engine.begin() as conn:
        df = df.sort_values("time").reset_index(drop=True)
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            row_dict['time'] = pd.to_datetime(row_dict['time']).strftime('%Y-%m-%dT%H:%M:%S')

            if table_name == "stock_prices":
                query = text("""
                    INSERT INTO stock_prices (ticker, time, open, high, low, close, volume)
                    VALUES (:ticker, :time, :open, :high, :low, :close, :volume)
                """)
            elif table_name == "stock_news":
                query = text("""
                    INSERT INTO stock_news (
                        ticker, time, title, content, link, symbols, tags,
                        sentiment_polarity, sentiment_neg, sentiment_neu, sentiment_pos
                    )
                    VALUES (
                        :ticker, :time, :title, :content, :link, :symbols, :tags,
                        :sentiment_polarity, :sentiment_neg, :sentiment_neu, :sentiment_pos
                    )
                """)
            conn.execute(query, row_dict)

def iterate_months(start_year, end_year, start_month=1, end_month=1, start_day=1, end_day=1):
    current = datetime(start_year, start_month, start_day)
    end = datetime(end_year + 1, end_month, end_day)

    while current < end:
        start_date = current.date()
        last_day = calendar.monthrange(current.year, current.month)[1]
        end_date = datetime(current.year, current.month, last_day).date()
        
        yield start_date, end_date
        current += relativedelta(months=1)

if __name__ == "__main__":
    for TICKER in TICKERS:
        print(f"Processing ticker: {TICKER}")
        for start, end in iterate_months(START_YEAR, END_YEAR, START_MONTH, END_MONTH, START_DAY, END_DAY):
            try:
                print(f"Fetching stock data for {TICKER} from {start} to {end}...")
                stock_df = fetch_daily_stock_data(TICKER, EXCHANGE, EOD_API_KEY, start, end)
                if stock_df.empty:
                    missing_data_log["missing_stock_data"].append({"ticker": TICKER, "start_date": str(start), "end_date": str(end)})
                else:
                    write_to_questdb(stock_df, "stock_prices")

                print(f"Fetching news data for {TICKER} from {start} to {end}...")
                news_df = fetch_news_data(TICKER, EOD_API_KEY, start, end)
                if news_df.empty:
                    missing_data_log["missing_news_data"].append({"ticker": TICKER, "start_date": str(start), "end_date": str(end)})
                else:
                    write_to_questdb(news_df, "stock_news")

            except Exception as e:
                print(f"Error processing {TICKER} from {start} to {end}: {e}")
                missing_data_log["missing_stock_data"].append({"ticker": TICKER, "start_date": str(start), "end_date": str(end), "error": str(e)})

    # Write missing data log to a JSON file
    with open("missing_data_log.json", "w") as log_file:
        json.dump(missing_data_log, log_file, indent=4)

    print("Data pipeline completed.")