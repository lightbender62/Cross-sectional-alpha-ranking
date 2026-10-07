import pandas as pd

def walk_forward_splits(label_end, eval_start , eval_end, min_train=12):
    label_end = label_end.sort_index()
    for t in label_end.index:
        if t < pd.Timestamp(eval_start) or t> pd.Timestamp(eval_end):
            continue
        train = label_end.index[(label_end.index < t) & (label_end <=t)]
        if len(train) < min_train:
            continue
        yield train, t