"""Does the reserve uplift actually cover what it claims to?

Repeats, over several synthetic books, three comparisons that decide how
ClaimSight sets reserves:

1. honest coverage — uplift fitted on a calibration slice, coverage measured on
   a disjoint test slice, at every offered confidence level;
2. the trap — fitting the same uplift on the training fold instead, which is
   the tempting shortcut when you only cut the book in two;
3. per-claim-type uplifts vs one global uplift, which is the "condition the
   quantile on the segment" idea the README used to list as a limitation.

Usage:
    uv run python scripts/reserve_calibration_report.py [--books 4] [--claims 20000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_claims import generate

from claimsight.features.extractor import extract_log_residuals
from claimsight.models.triage import FEATURES, TYPE_DUMMIES, prepare

COLS = FEATURES + TYPE_DUMMIES
LEVELS = (0.50, 0.75, 0.90, 0.95, 0.99)
PRIMARY = 0.75


def one_book(n_claims: int, seed: int) -> dict:
    raw = generate(n_claims, seed)
    frame = prepare(raw)
    y_log = np.log1p(raw["final_severity"]).to_numpy()
    train_idx, rest = train_test_split(raw.index, test_size=0.4, random_state=seed)
    cal_idx, test_idx = train_test_split(rest, test_size=0.5, random_state=seed)

    model = LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=42
    ).fit(frame.loc[train_idx, COLS], y_log[frame.index.get_indexer(train_idx)])

    def residuals(idx):
        return extract_log_residuals(
            model, frame.loc[idx], y_log[frame.index.get_indexer(idx)], COLS
        )

    _, r_cal = residuals(cal_idx)
    _, r_train = residuals(train_idx)
    pred_test, r_test = residuals(test_idx)
    incurred = float(raw.loc[test_idx, "final_severity"].sum())

    out: dict[str, float] = {}
    for level in LEVELS:
        uplift = float(np.quantile(r_cal, level))
        out[f"cov@{level}"] = float((r_test <= uplift).mean())
        out[f"cap@{level}"] = float(np.expm1(pred_test + uplift).sum()) / incurred

    out["insample_uplift"] = float(np.quantile(r_train, PRIMARY))
    out["honest_uplift"] = float(np.quantile(r_cal, PRIMARY))
    out["insample_cov"] = float((r_test <= out["insample_uplift"]).mean())

    # Global vs per-claim-type uplift, both fitted on the calibration slice.
    global_uplift = out["honest_uplift"]
    by_type = (
        pd.Series(r_cal, index=cal_idx).groupby(raw.loc[cal_idx, "claim_type"]).quantile(PRIMARY)
    )
    test_types = raw.loc[test_idx, "claim_type"]
    per_type_uplift = test_types.map(by_type).to_numpy()
    covered_global = pd.Series(r_test <= global_uplift, index=test_idx)
    covered_typed = pd.Series(r_test <= per_type_uplift, index=test_idx)
    g = covered_global.groupby(test_types).mean()
    t = covered_typed.groupby(test_types).mean()
    out["global_maxdev"] = float((g - PRIMARY).abs().max())
    out["typed_maxdev"] = float((t - PRIMARY).abs().max())
    out["global_worst"] = float(g.min())
    out["typed_worst"] = float(t.min())
    out["min_type_support"] = int(raw.loc[cal_idx, "claim_type"].value_counts().min())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--books", type=int, default=4)
    parser.add_argument("--claims", type=int, default=20000)
    args = parser.parse_args()

    seeds = [42, 7, 11, 2024, 5, 99, 123, 8][: args.books]
    table = pd.DataFrame([one_book(args.claims, seed) for seed in seeds], index=seeds)

    print(f"\n{args.books} synthetic books of {args.claims:,} claims (60/20/20 split)\n")
    print("1. honest coverage - uplift fitted on calibration, measured on test")
    print(f"   {'nominal':>8}  {'realized':>9}  {'spread':>13}  reserves/incurred")
    for level in LEVELS:
        cov = table[f"cov@{level}"]
        print(
            f"   {level:>8.2f}  {cov.mean():>9.4f}  "
            f"[{cov.min():.4f},{cov.max():.4f}]  {table[f'cap@{level}'].mean():>8.3f}x"
        )

    print("\n2. the trap - same uplift fitted on the training fold instead")
    print(
        f"   uplift {table['insample_uplift'].mean():.4f} vs honest "
        f"{table['honest_uplift'].mean():.4f}; realized coverage "
        f"{table['insample_cov'].mean():.4f} "
        f"(worst {table['insample_cov'].min():.4f}) against nominal {PRIMARY:.2f} - "
        f"{(PRIMARY - table['insample_cov'].mean()) * 100:.1f} points of silent under-reserving"
    )

    print("\n3. per-claim-type uplift vs one global uplift, at nominal 0.75")
    print(
        f"   max deviation from nominal: global {table['global_maxdev'].mean():.4f}"
        f" vs per-type {table['typed_maxdev'].mean():.4f}"
    )
    print(
        f"   worst segment coverage:     global {table['global_worst'].mean():.4f}"
        f" vs per-type {table['typed_worst'].mean():.4f}"
    )
    print(
        f"   thinnest claim type has {table['min_type_support'].min()}-"
        f"{table['min_type_support'].max()} calibration claims - the per-type quantile's own"
        " sampling error swamps the bias it removes."
    )
    print()


if __name__ == "__main__":
    main()
