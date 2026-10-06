from pathlib import Path
 
import pandas as pd
import yfinance as yf
 
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = DATA_DIR / "ohlcv.parquet"

TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "TITAN.NS", "WIPRO.NS", "ULTRACEMCO.NS", "NESTLEIND.NS",
]

start = "2014-06-01"
end = "2024-12-31"

def download(tickers=TICKERS, start = start, end = end ):
    raw = yf.download(tickers , start = start , end = end , auto_adjust=True , progress = False)
    df = raw.stack(level=1, future_stack=True).rename_axis(["date" , "ticker"]).reset_index()
    df.columns = [c.lower() for c in df.columns]
    return df.dropna(subset=["close"]).sort_values(["ticker" , "date"]).reset_index(drop=True)

def load():
    return pd.read_parquet(OUT_PATH)

if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    df = download()
    df.to_parquet(OUT_PATH)
    print(df.groupby("ticker")["date"].agg(["min" , "max", "count"]))
    

