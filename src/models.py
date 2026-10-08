import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
 
from splits import walk_forward_splits

Target = {"ridge" : "fwd_ret_z" , "logit" : "beat_median" , "rf": "beat_median"}

DEFAULTS = {
    "ridge" : dict(alpha = 10.0),
    "logit" : dict(C = 1.0 , max_iter = 1000),
    "rf" : dict(n_estimators = 200, max_depth = 4 , min_samples_leaf = 50 , random_state = 0 , n_jobs= -1),
}

_CLASSES = {"ridge": Ridge, "logit": LogisticRegression, "rf": RandomForestClassifier}

def make_dataset(feats , labels):
    return feats.join(labels, how = "inner").dropna()

def make_model(name , **param):
    if name not in _CLASSES:
        raise ValueError(f"unknown model {name!r}")
    return _CLASSES[name](**{**DEFAULTS[name] , **param})

def _score(model , name , X):
    if name == "ridge":
        return model.predict(X)
    return model.predict_proba(X)[: , 1]

def _importances(model , name):
    if name == "ridge":
        return np.ravel(model.coef_)
    if name == "logit":
        return np.ravel(model.coef_)
    return model.feature_importances_

def walk_forward_predict(data , feature_cols , label_end , name , eval_start , eval_end , min_train = 12 , **params):
    target = Target[name]
    dates = data.index.get_level_values("date")
    preds , log = [] , []

    for train_dates, t in walk_forward_splits(label_end , eval_start , eval_end , min_train):
        tr = data[dates.isin(train_dates)]
        te = data[dates==t]
        if te.empty:
            continue
        model = make_model(name , **params).fit(tr[feature_cols] , tr[target])
        preds.append(
            pd.DataFrame(
                {"score": _score(model , name , te[feature_cols]) , "y" : te[target] , "fwd_ret": te["fwd_ret"]},
                index= te.index
            )
        )
        row = {"n_train" : len(tr) , "train_score": model.score(tr[feature_cols] , tr[target])}
        row.update({f"imp_{c}" : v for c,v in zip(feature_cols, _importances(model , name))})

        log.append(pd.Series(row , name=t))

    if not preds:
        raise ValueError("no test dates in the evaluation window")
    return pd.concat(preds) , pd.DataFrame(log)

def rank_ic(pred , score = "score" , ret = "fwd_ret"):
    ic = {d: spearmanr(g[score] , g[ret])[0] for d,g in pred.groupby(level = "date")}
    return pd.Series(ic , name = "rank_ic").rename_axis("date")

def ic_summary(ic):
    ic = ic.dropna()
    n,m,s = len(ic) , ic.mean() , ic.std()

    return {
        "mean_ic": m,
        "std_ic": s,
        "ic_ir": m / s,
        "t_stat": m / (s / np.sqrt(n)),
        "hit_rate": (ic > 0).mean(),
        "n_dates": n,
    }

def oos_r2(pred):
    return r2_score(pred["y"], pred["score"])
 
 
def oos_accuracy(pred):
    return accuracy_score(pred["y"], pred["score"] > 0.5)