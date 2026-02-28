"""AutoEIT – Run baseline ASR benchmark (no fine-tuning).

Evaluates five off-the-shelf models on the mini (or full) test split
and prints comparison tables.

Usage
-----
    # Quick run on 40-row mini test set:
    python scripts/run_baseline.py

    # Full evaluation on the complete test split:
    python scripts/run_baseline.py --full

    # Single model only:
    python scripts/run_baseline.py --models openai/whisper-large-v3
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Ensure project root is on sys.path so ``src`` is importable.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluate import ASREvaluator
from src.infer import ASRInferencer, normalize_text

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("autoeit.run_baseline")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(_h)

# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------
DEFAULT_MODELS: List[str] = [
    "openai/whisper-base",
    "openai/whisper-medium",
    "openai/whisper-large-v3",
    "facebook/wav2vec2-large-xlsr-53-spanish",
    "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
]

# Short display names for the table
MODEL_SHORT_NAMES: Dict[str, str] = {
    "openai/whisper-base": "whisper-base",
    "openai/whisper-medium": "whisper-medium",
    "openai/whisper-large-v3": "whisper-large-v3",
    "facebook/wav2vec2-large-xlsr-53-spanish": "wav2vec2-xlsr-53-es",
    "jonatasgrosman/wav2vec2-large-xlsr-53-spanish": "wav2vec2-xlsr-53-es (community)",
}

PRIMARY_MODEL = "openai/whisper-large-v3"


# ═══════════════════════════════════════════════════════════════════════════
# Pretty-print helpers
# ═══════════════════════════════════════════════════════════════════════════

def _fmt_pct(val: float) -> str:
    """Format a fraction as a percentage string."""
    return f"{val * 100:6.2f}%"


def _print_primary_table(rows: List[dict]) -> None:
    """Print the model comparison table."""
    header = f"{'Model':<40s} {'WER':>7s} {'CER':>7s} {'90%Agree':>9s} {'RTF':>7s}"
    sep = "─" * len(header)
    print()
    print("┌" + sep + "┐")
    print("│" + header + "│")
    print("├" + sep + "┤")
    for r in rows:
        star = " ★" if r["model"] == PRIMARY_MODEL else "  "
        name = MODEL_SHORT_NAMES.get(r["model"], r["model"])
        line = (
            f"{name + star:<40s} "
            f"{_fmt_pct(r['wer']):>7s} "
            f"{_fmt_pct(r['cer']):>7s} "
            f"{_fmt_pct(r['pct_90_agree']):>9s} "
            f"{r['avg_rtf']:>6.2f}x"
        )
        print("│" + line + "│")
    print("└" + sep + "┘")
    print()


def _print_dataset_table(by_dataset: pd.DataFrame) -> None:
    """Print per-dataset breakdown for a single model."""
    header = f"{'Dataset':<22s} {'WER':>7s} {'90%Agree':>9s}"
    sep = "─" * len(header)
    print("┌" + sep + "┐")
    print("│" + header + "│")
    print("├" + sep + "┤")
    for _, row in by_dataset.iterrows():
        line = (
            f"{str(row.iloc[0]):<22s} "
            f"{_fmt_pct(row['wer']):>7s} "
            f"{_fmt_pct(row['pct_90_agree']):>9s}"
        )
        print("│" + line + "│")
    print("└" + sep + "┘")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Core benchmark logic
# ═══════════════════════════════════════════════════════════════════════════

def run_benchmark(
    test_df: pd.DataFrame,
    models: List[str],
    device: str = "auto",
    language: str = "es",
    batch_size: int = 8,
    results_dir: str = "results",
) -> pd.DataFrame:
    """Run all models on *test_df* and return a summary DataFrame.

    Parameters
    ----------
    test_df : pd.DataFrame
        Test data with ``processed_path`` and ``transcript`` columns.
    models : list[str]
        HuggingFace model identifiers.
    device : str
        Torch device.
    language : str
        Target language code.
    batch_size : int
        Pipeline batch size.
    results_dir : str
        Directory for per-model CSV outputs.

    Returns
    -------
    pd.DataFrame
        Summary with one row per model.
    """
    evaluator = ASREvaluator()
    summary_rows: List[dict] = []
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    for model_name in models:
        logger.info("=" * 60)
        logger.info("Benchmarking: %s", model_name)
        logger.info("=" * 60)

        try:
            inferencer = ASRInferencer(
                model_name=model_name,
                device=device,
                language=language,
                batch_size=batch_size,
            )
        except Exception as exc:
            logger.error("Failed to load model %s: %s", model_name, exc)
            summary_rows.append(
                {
                    "model": model_name,
                    "wer": float("nan"),
                    "cer": float("nan"),
                    "pct_90_agree": float("nan"),
                    "avg_rtf": float("nan"),
                    "n_utterances": 0,
                    "error": str(exc),
                }
            )
            continue

        t0 = time.perf_counter()
        result_df = inferencer.transcribe_batch(test_df)
        wall_time = time.perf_counter() - t0

        # ── Compute metrics ───────────────────────────────────────────
        refs = result_df["reference_norm"].fillna("").tolist()
        hyps = result_df["hypothesis_norm"].fillna("").tolist()

        wer = evaluator.compute_wer(refs, hyps)
        cer = evaluator.compute_cer(refs, hyps)
        pct90 = evaluator.compute_90pct_agreement(refs, hyps)
        avg_rtf = result_df["rtf"].mean() if "rtf" in result_df.columns else float("nan")

        logger.info(
            "%s → WER=%.2f%%  CER=%.2f%%  90%%Agree=%.2f%%  RTF=%.2fx",
            model_name, wer * 100, cer * 100, pct90 * 100, avg_rtf,
        )

        # Sanity checks
        if wer > 0.85:
            logger.warning(
                "WER > 85%% for %s — check audio/transcript alignment!",
                model_name,
            )
        if wer < 0.10:
            logger.warning(
                "WER < 10%% for %s — possible data leakage, check splits!",
                model_name,
            )

        summary_rows.append(
            {
                "model": model_name,
                "wer": wer,
                "cer": cer,
                "pct_90_agree": pct90,
                "avg_rtf": avg_rtf,
                "n_utterances": len(refs),
                "wall_time_sec": wall_time,
                "error": "",
            }
        )

        # Save per-utterance results
        model_short = model_name.replace("/", "_")
        per_utt_path = results_path / f"baseline_{model_short}.csv"
        result_df.to_csv(per_utt_path, index=False)
        logger.info("Per-utterance results → %s", per_utt_path)

        # Free GPU memory
        del inferencer
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    return pd.DataFrame(summary_rows)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AutoEIT – Baseline ASR benchmark (pre-fine-tuning).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full test.csv instead of test_mini.csv.",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default=None,
        help="Override test CSV path.",
    )
    parser.add_argument(
        "--processed_manifest",
        type=str,
        default="data/transcripts/processed_manifest.csv",
        help="Path to the processed manifest.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model(s) to benchmark.  Defaults to all 5 models.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="es",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Resolve test CSV ──────────────────────────────────────────────
    if args.test_csv:
        test_csv = args.test_csv
    elif args.full:
        test_csv = "data/transcripts/test.csv"
    else:
        test_csv = "data/transcripts/test_mini.csv"

    logger.info("Loading test data: %s", test_csv)
    test_df = pd.read_csv(test_csv)

    # ── Merge with processed manifest ─────────────────────────────────
    if "processed_path" not in test_df.columns:
        logger.info("Merging with processed manifest …")
        pm = pd.read_csv(args.processed_manifest)
        merge_cols = ["utterance_id", "processed_path", "rejected"]
        if "rejection_reason" in pm.columns:
            merge_cols.append("rejection_reason")
        test_df = test_df.merge(pm[merge_cols], on="utterance_id", how="left")
        before = len(test_df)
        test_df = test_df[
            (test_df["processed_path"].notna()) & (test_df["rejected"] != True)
        ].copy()
        logger.info("Usable rows: %d / %d", len(test_df), before)

    # ── Select models ─────────────────────────────────────────────────
    models = args.models if args.models else DEFAULT_MODELS

    # ── Run benchmark ─────────────────────────────────────────────────
    summary_df = run_benchmark(
        test_df,
        models=models,
        device=args.device,
        language=args.language,
        batch_size=args.batch_size,
        results_dir=args.results_dir,
    )

    # ── Save summary ──────────────────────────────────────────────────
    summary_path = Path(args.results_dir) / "baseline_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Summary saved to %s", summary_path)

    # ── Print primary comparison table ────────────────────────────────
    _print_primary_table(summary_df.to_dict("records"))

    # ── Print per-dataset breakdown for best model ────────────────────
    valid = summary_df[summary_df["wer"].notna()]
    if len(valid) > 0:
        best_model = valid.loc[valid["wer"].idxmin(), "model"]
        best_short = best_model.replace("/", "_")
        best_csv = Path(args.results_dir) / f"baseline_{best_short}.csv"
        if best_csv.exists():
            best_df = pd.read_csv(best_csv)
            evaluator = ASREvaluator()
            if "dataset" in best_df.columns:
                print(f"Per-dataset breakdown for best model: {best_model}")
                by_ds = evaluator.compute_wer_by_group(best_df, "dataset")
                _print_dataset_table(by_ds)

            if "l1_group" in best_df.columns and best_df["l1_group"].notna().any():
                print(f"Per-L1-group breakdown ({best_model}):")
                by_l1 = evaluator.compute_wer_by_group(
                    best_df[best_df["l1_group"].notna()], "l1_group",
                )
                print(by_l1.to_string(index=False))
                print()

            if "task_type" in best_df.columns and best_df["task_type"].notna().any():
                print(f"Per-task-type breakdown ({best_model}):")
                by_task = evaluator.compute_wer_by_group(
                    best_df[best_df["task_type"].notna()], "task_type",
                )
                print(by_task.to_string(index=False))
                print()


if __name__ == "__main__":
    main()
