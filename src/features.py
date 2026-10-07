import numpy as np 
import pandas as pd

FEATURES = ["ret_5" , "ret_20" , "ret_60" , "vol_20" , "relvol_20" , "rsi_14" , "macd_norm"]

def build_features(df):
    df = df.sort_values(["ticker" , "date"]).reset_index(drop = True)
    tk = df["ticker"]
    close = df["close"]
    out = pd.DataFrame({"date": df["date"] , "ticker" : tk})

    for k in (5 , 20 , 60):
        out[f"return{k}"] = close.groupby(tk).pct_change(k)

    logreturn = np.log(close).groupby(tk).diff()
    out["vol_20"] = logreturn.groupby(tk).transform(lambda s: s.rolling(20).std())

    avg_vol = df["volume"].groupby(tk).transform(lambda s: s.mean())
    out["relvol_20"] = np.log(df["volume"]/avg_vol).replace([np.inf , -np.inf] , np.nan)

    delta = close.groupby(tk).diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.groupby(tk).transform(lambda s: s.ewn(alpha = 1/14 , adjust = False , min_periods = 14).mean())
    avg_loss = loss.groupby(tk).transform(lambda s: s.ewn(alpha = 1/14 , adjust = False , min_periods = 14).mean())

    rs = avg_gain/avg_loss
    out["rsi_14"] = 100 - 100/(1+rs)

    ema12 = close.groupby(tk).transform(lambda s: s.ewn(span = 12 , adjust = False , min_periods = 12).mean())
    ema26 = close.groupby(tk).transform(lambda s: s.ewn(span = 26 , adjust = True , min_periods = 12).mean())

    out["macd_norm"] = (ema12-ema26)/close
    return out.set_index(["date" , "ticker"])[FEATURES]

