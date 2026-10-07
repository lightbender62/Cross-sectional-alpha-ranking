import numpy as np
import pandas as pd

def winsorize_cs(df , lower=0.05 , upper = 0.95):
    g = df.groupby(level = "date")
    lo = g.transform("quantile" , lower)
    hi = g.transform("quantile" , upper)

    return df.clip(lower = lo , upper = hi)

def zscore_cs(df):
    g = df.groupby(level = "date")
    mu = g.transform("mean")
    sd = g.transform("std").replace(0 , np.nan)

    return(df -mu)/sd

def preprocess(df , lower = 0.05 , upper = 0.95):
    return zscore_cs(winsorize_cs(df , lower , upper))

