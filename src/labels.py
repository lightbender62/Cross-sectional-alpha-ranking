import numpy as np
import pandas as pd

def rebalance_dates(dates):
    s = pd.Series(pd.DatetimeIndex(dates).unique().sort_values())
    return pd.DatetimeIndex(s.groupby([s.dt.year, s.dt.month]).max().to_numpy())

def make_labels(prices , rb_dates):
    rb = pd.DatetimeIndex(rb_dates)
    close = prices.pivot(index = "date" , columns = "ticker" , values = "close").reindex(rb)
    fwd = close.shift(-1)/close - 1
    end = pd.Series(rb[1:].append(pd.DatetimeIndex([pd.NaT])) , index = rb)
    out = fwd.stack(future_stack = True).dropna().rename("fwd_ret").to_frame()
    out.index.names = ["date", "ticker"]
    g=out.groupby(level="date")["fwd_ret"]
    out["fwd_ret_z"] = (out["fwd_ret"] - g.transform("mean"))/g.transform("std")
    out["beat_median"] = (out["fwd_ret"] > g.transform("median")).astype(float)
    out["label_end"] = end.reindex(out.index.get_level_values("date")).to_numpy()

    return out[["label_end", "fwd_ret", "fwd_ret_z", "beat_median"]]