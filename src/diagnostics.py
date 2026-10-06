import warnings
 
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

def stationarity(series , alpha = 0.05):
    s = series.dropna()
    adf_p = adfuller(s , autolag ="AIC")[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore" , InterpolationWarning)
        kpss_p = kpss(s , regression="c" , nlags = "auto")[1]
    adf_rej , kpss_rej = adf_p < alpha , kpss_p < alpha

    if adf_rej and not kpss_rej:
        verdict = "stationary"
    elif not adf_rej and kpss_rej:
        verdict = "non-stationary"
    else:
        verdict = "ambiguous"
    return {
        "adf_p" : adf_p,
        "kpss_p" : kpss_p,
        "verdict": verdict
    }

def stationarity_table(df , col , group = "ticker"):
    rows = {t: stationarity(g[col]) for t,g in df.groupby(group)}
    return pd.DataFrame(rows).T

def normality(series):
    s = series.dropna()
    jb,p = stats.jarque_bera(s)
    return{
        "skew": stats.skew(s),
        "excess_kurtosis": stats.kurtosis(s),
        "jb": jb,
        "jb_p": p,
    }

def normality_table(df , cols):
    return pd.DataFrame({c: normality(df[c]) for c in cols}).T

def cross_sectional_moments(df , col , date = "date"):
    g = df.groupby(date)[col]
    return pd.DataFrame({"skew": g.skew() ,"excess_kurtosis": g.apply(pd.Series.kurt)})

def vif(df , cols):
    X = df[cols].dropna()
    X = (X- X.mean())/X.std()
    return pd.Series(
        [variance_inflation_factor(X.values , i) for i in range (X.shape[1])],
        index = cols,
        name = "vif",
    )