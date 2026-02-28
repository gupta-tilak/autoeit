"""Generate data/transcripts/test_mini.csv – a 40-row mini test set.

Sampling strategy
-----------------
* 10 from Nebrija-INMIGRA  (2-3 per L1 group, spread across L1s)
* 10 from Nebrija-WOCAE
* 20 from SPLLOC1           (4-5 per task type)

Prefers rows with ``has_timestamps=True`` for cleaner evaluation.
Requires ``processed_manifest.csv`` to filter to usable rows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    test_path = ROOT / "data" / "transcripts" / "test.csv"
    pm_path = ROOT / "data" / "transcripts" / "processed_manifest.csv"
    out_path = ROOT / "data" / "transcripts" / "test_mini.csv"

    test = pd.read_csv(test_path)
    pm = pd.read_csv(pm_path)

    # Merge to get processed_path and filter to usable rows
    test = test.merge(
        pm[["utterance_id", "processed_path", "rejected"]],
        on="utterance_id",
        how="left",
    )
    test = test[
        (test["processed_path"].notna()) & (test["rejected"] != True)
    ].copy()

    # Prefer has_timestamps=True
    ts_true = test[test["has_timestamps"] == True]
    if len(ts_true) >= 40:
        test = ts_true

    rng = pd.np if hasattr(pd, "np") else __import__("numpy")
    seed = 42

    samples = []

    # ── Nebrija-INMIGRA: 10 rows, spread across L1 groups ────────────
    inmigra = test[test["dataset"] == "Nebrija-INMIGRA"]
    l1_groups = sorted(inmigra["l1_group"].dropna().unique())
    per_l1 = max(1, 10 // len(l1_groups)) if l1_groups else 10
    inmigra_sample = (
        inmigra.groupby("l1_group", group_keys=False)
        .apply(lambda g: g.sample(n=min(per_l1, len(g)), random_state=seed))
    )
    # Top up to 10 if needed
    if len(inmigra_sample) < 10:
        remaining = inmigra[~inmigra.index.isin(inmigra_sample.index)]
        extra = remaining.sample(
            n=min(10 - len(inmigra_sample), len(remaining)),
            random_state=seed,
        )
        inmigra_sample = pd.concat([inmigra_sample, extra])
    inmigra_sample = inmigra_sample.head(10)
    samples.append(inmigra_sample)

    # ── Nebrija-WOCAE: 10 rows ───────────────────────────────────────
    wocae = test[test["dataset"] == "Nebrija-WOCAE"]
    wocae_sample = wocae.sample(n=min(10, len(wocae)), random_state=seed)
    samples.append(wocae_sample)

    # ── SPLLOC1: 20 rows, spread across task types ───────────────────
    splloc = test[test["dataset"] == "SPLLOC1"]
    task_types = sorted(splloc["task_type"].dropna().unique())
    per_task = max(1, 20 // len(task_types)) if task_types else 20
    splloc_sample = (
        splloc.groupby("task_type", group_keys=False)
        .apply(lambda g: g.sample(n=min(per_task, len(g)), random_state=seed))
    )
    if len(splloc_sample) < 20:
        remaining = splloc[~splloc.index.isin(splloc_sample.index)]
        extra = remaining.sample(
            n=min(20 - len(splloc_sample), len(remaining)),
            random_state=seed,
        )
        splloc_sample = pd.concat([splloc_sample, extra])
    splloc_sample = splloc_sample.head(20)
    samples.append(splloc_sample)

    # ── Combine & save ────────────────────────────────────────────────
    mini = pd.concat(samples, ignore_index=True)

    # Drop the merge columns – the inference code will re-merge
    drop_cols = [c for c in ("processed_path", "rejected", "rejection_reason") if c in mini.columns]
    mini_out = mini.drop(columns=drop_cols)

    mini_out.to_csv(out_path, index=False)
    print(f"Saved {len(mini_out)} rows to {out_path}")
    print(f"  Nebrija-INMIGRA: {(mini_out['dataset']=='Nebrija-INMIGRA').sum()}")
    print(f"  Nebrija-WOCAE:   {(mini_out['dataset']=='Nebrija-WOCAE').sum()}")
    print(f"  SPLLOC1:         {(mini_out['dataset']=='SPLLOC1').sum()}")
    print(f"  L1 groups (INMIGRA): {sorted(mini_out[mini_out['dataset']=='Nebrija-INMIGRA']['l1_group'].unique())}")


if __name__ == "__main__":
    main()
