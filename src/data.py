from pathlib import Path

import pandas as pd
import yfinance as yf


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = DATA_DIR / "ohlcv.parquet"


TICKERS = [
    # original 20
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "TITAN.NS", "WIPRO.NS", "ULTRACEMCO.NS", "NESTLEIND.NS",

    # expansion
    "ABB.NS", "ADANIENT.NS", "ADANIGREEN.NS", "ADANIPORTS.NS", "ADANIPOWER.NS",
    "ATGL.NS", "AMBUJACEM.NS", "APOLLOHOSP.NS", "DMART.NS", "BAJAJ-AUTO.NS",
    "BAJAJFINSV.NS", "BAJAJHLDNG.NS", "BANKBARODA.NS", "BEL.NS", "BHEL.NS",
    "BOSCHLTD.NS", "BPCL.NS", "BRITANNIA.NS", "CANBK.NS", "CHOLAFIN.NS",
    "CIPLA.NS", "COALINDIA.NS", "DABUR.NS", "DIVISLAB.NS", "DLF.NS",
    "DRREDDY.NS", "EICHERMOT.NS", "ETERNAL.NS", "GAIL.NS", "GODREJCP.NS",
    "GRASIM.NS", "HAVELLS.NS", "HCLTECH.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS",
    "HINDALCO.NS", "HAL.NS", "ICICIGI.NS", "ICICIPRULI.NS", "INDUSINDBK.NS",
    "NAUKRI.NS", "INDIGO.NS", "IOC.NS", "IRCTC.NS", "IRFC.NS",
    "JINDALSTEL.NS", "JSWENERGY.NS", "JSWSTEEL.NS", "LICI.NS", "M&M.NS",
    "LODHA.NS", "NHPC.NS", "NTPC.NS", "ONGC.NS", "PIDILITIND.NS",
    "PNB.NS", "PFC.NS", "POWERGRID.NS", "RECLTD.NS", "MOTHERSON.NS",
    "SBILIFE.NS", "SHREECEM.NS", "SIEMENS.NS", "TATACONSUM.NS",
    "TATAMOTORS.NS", "TATASTEEL.NS", "TECHM.NS", "TORNTPHARM.NS",
    "TRENT.NS", "UNIONBANK.NS", "VBL.NS", "VEDL.NS",
]


START = "2014-06-01"
END = "2024-12-31"


def download(tickers=TICKERS, start=START, end=END):

    successful = []
    failed = []

    frames = []

    for ticker in tickers:
        try:
            print(f"Downloading {ticker}...")

            raw = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )

            if raw.empty:
                failed.append(ticker)
                print(f"  FAILED: no data returned")
                continue

            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            raw = raw.reset_index()

            raw.columns = [str(c).lower() for c in raw.columns]

            raw["ticker"] = ticker

            frames.append(raw)
            successful.append(ticker)

            print(f"  OK: {len(raw)} rows")

        except Exception as e:
            failed.append(ticker)
            print(f"  FAILED: {e}")

    if not frames:
        raise RuntimeError("No ticker data could be downloaded.")

    df = pd.concat(frames, ignore_index=True)

    columns = [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    df = df[[c for c in columns if c in df.columns]]

    df = df.dropna(subset=["close"])

    df["date"] = pd.to_datetime(df["date"])

    df = (
        df.sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    print("\n" + "=" * 50)
    print(f"Successfully downloaded: {len(successful)}/{len(tickers)}")
    print(f"Failed: {len(failed)}/{len(tickers)}")

    if failed:
        print("\nFailed tickers:")
        for ticker in failed:
            print(f"  - {ticker}")

    return df


def load():
    return pd.read_parquet(OUT_PATH)


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = download()

    df.to_parquet(OUT_PATH, index=False)

    print("\nDataset saved to:")
    print(OUT_PATH)

    print("\nDataset summary:")
    print(
        df.groupby("ticker")["date"]
        .agg(["min", "max", "count"])
    )