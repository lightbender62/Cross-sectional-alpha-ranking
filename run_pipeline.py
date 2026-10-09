import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from src.backtester import month_end_trading_dates, run_backtest
from src.data import load
from src.features import FEATURES, build_features
from src.labels import make_labels, rebalance_dates
from src.models import walk_forward_predict
from src.risk import risk_adapter

REPORTS = os.path.join(ROOT, "reports")
FIGURES = os.path.join(REPORTS, "figures")

WINDOW_METRICS = [
    "n_periods", "mean_IC", "ICIR", "IC_hit_rate", "sharpe", "sortino",
    "max_drawdown", "ann_return", "mean_turnover",
]
SUMMARY_METRICS = [
    "ann_vol", "max_drawdown", "mean_turnover", "sharpe", "sortino",
    "mean_IC", "final_nav",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="rf")
    parser.add_argument("--eval-start", default="2023-01-01")
    parser.add_argument("--eval-end", default="2024-12-31")
    return parser.parse_args()


def line_plot(path, title, ylabel, x, baseline, risk_on, zero_line=False):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, baseline, label="Baseline")
    ax.plot(x, risk_on, linestyle="--", label="Risk-adjusted")

    if zero_line:
        ax.axhline(0, linewidth=1, color="grey")

    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    tag = f"{args.model}_{args.eval_start[:4]}"

    df = load()
    df["date"] = pd.to_datetime(df["date"])
    prices = df[["date", "ticker", "close"]].copy()
    print(f"universe: {df['ticker'].nunique()} names | model: {args.model}")

    rb = rebalance_dates(df["date"])
    feats = build_features(df)
    labels = make_labels(df, rb)
    data = feats.join(labels, how="inner").dropna()

    label_end = labels["label_end"].groupby(level="date").first()
    label_end = pd.Series(
        pd.to_datetime(label_end.values),
        index=pd.DatetimeIndex(label_end.index),
    )

    pred, _ = walk_forward_predict(
        data,
        FEATURES,
        label_end,
        name=args.model,
        eval_start=args.eval_start,
        eval_end=args.eval_end,
    )

    scores = pred.reset_index()[["date", "ticker", "score"]].copy()
    scores["date"] = pd.to_datetime(scores["date"])
    print(f"prediction rows: {len(pred)} | scored dates: {scores['date'].nunique()}")

    backtest_kwargs = dict(
        rebalance_dates=month_end_trading_dates(df["date"]),
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        strict=True,
    )

    baseline = run_backtest(prices, scores, risk_fn=None, **backtest_kwargs)
    risk_on = run_backtest(prices, scores, risk_fn=risk_adapter, **backtest_kwargs)

    if not np.allclose(
        baseline.periods["rank_ic"].to_numpy(),
        risk_on.periods["rank_ic"].to_numpy(),
        equal_nan=True,
    ):
        raise RuntimeError("risk layer changed rank IC")

    windows = {
        "validation_2023": ("2023-01-01", "2023-12-31"),
        "test_2024": ("2024-01-01", "2024-12-31"),
    }

    if pd.Timestamp(args.eval_start) < pd.Timestamp("2023-01-01"):
        windows = {"pre_2023": (args.eval_start, "2022-12-31"), **windows}

    by_window = baseline.by_window(windows)
    comparison = pd.DataFrame(
        {
            "Baseline": baseline.summary(),
            "Risk-adjusted": risk_on.summary(),
        }
    )

    print(
        f"\nbaseline: {len(baseline.periods)} periods, "
        f"{len(baseline.skipped)} skipped | "
        f"risk-adjusted: {len(risk_on.periods)} periods, "
        f"{len(risk_on.skipped)} skipped"
    )

    if not by_window.empty:
        print("\nBaseline by window")
        print(by_window.loc[WINDOW_METRICS].round(4).to_string())

    print("\nBaseline vs risk-adjusted (full evaluation period)")
    print(comparison.loc[SUMMARY_METRICS].round(4).to_string())

    os.makedirs(FIGURES, exist_ok=True)

    baseline.periods.to_csv(
        os.path.join(REPORTS, f"backtest_periods_{tag}.csv"), index=False
    )
    risk_on.periods.to_csv(
        os.path.join(REPORTS, f"backtest_periods_{tag}_risk.csv"), index=False
    )
    comparison.to_csv(os.path.join(REPORTS, f"backtest_summary_{tag}.csv"))

    line_plot(
        os.path.join(FIGURES, f"backtest_rank_ic_{tag}.png"),
        f"Monthly rank IC ({tag})",
        "Spearman rank IC",
        baseline.periods["date"],
        baseline.periods["rank_ic"],
        risk_on.periods["rank_ic"],
        zero_line=True,
    )

    line_plot(
        os.path.join(FIGURES, f"backtest_equity_curve_{tag}.png"),
        f"Equity curve ({tag})",
        "NAV",
        baseline.periods["date"],
        baseline.periods["nav"],
        risk_on.periods["nav"],
    )

    print(f"\nsaved results to {REPORTS}")


if __name__ == "__main__":
    main()