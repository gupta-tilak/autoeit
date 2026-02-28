"""AutoEIT – Evaluation metrics (WER, CER, 90 %% agreement, breakdowns).

Provides :class:`ASREvaluator` with corpus-level and per-group metric
computation for ASR output evaluation.

Metrics
-------
* **WER** – Word Error Rate (jiwer).
* **CER** – Character Error Rate (jiwer).
* **90 % agreement** – fraction of utterances where per-utterance
  accuracy ``(1 − utt_wer) ≥ 0.90``.  This is the project's
  **primary** metric because it directly measures "usable" transcripts.
* **Error breakdown** – substitutions, insertions, deletions.
* **Per-group WER** – WER sliced by ``dataset``, ``l1_group``, or
  ``task_type``.
"""

from __future__ import annotations

import logging
import sys
from typing import Dict, List, Optional, Sequence, Union

import jiwer
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("autoeit.evaluate")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(_h)


# ═══════════════════════════════════════════════════════════════════════════
# ASREvaluator
# ═══════════════════════════════════════════════════════════════════════════

class ASREvaluator:
    """Compute ASR evaluation metrics.

    All public methods accept plain Python lists of reference /
    hypothesis strings.  Text should already be normalised (via
    :func:`src.infer.normalize_text`) before calling these methods.
    """

    # ------------------------------------------------------------------
    # Corpus-level WER / CER
    # ------------------------------------------------------------------
    @staticmethod
    def compute_wer(refs: Sequence[str], hyps: Sequence[str]) -> float:
        """Compute corpus-level Word Error Rate.

        Parameters
        ----------
        refs : sequence of str
            Reference (gold) transcripts.
        hyps : sequence of str
            Hypothesis transcripts.

        Returns
        -------
        float
            WER in ``[0, ∞)``.  Values > 1.0 are possible if the
            hypothesis is much longer than the reference.
        """
        refs_clean = [r if r.strip() else "<empty>" for r in refs]
        hyps_clean = [h if h.strip() else "<empty>" for h in hyps]
        return float(jiwer.wer(refs_clean, hyps_clean))

    @staticmethod
    def compute_cer(refs: Sequence[str], hyps: Sequence[str]) -> float:
        """Compute corpus-level Character Error Rate.

        Parameters
        ----------
        refs : sequence of str
            Reference transcripts.
        hyps : sequence of str
            Hypothesis transcripts.

        Returns
        -------
        float
            CER in ``[0, ∞)``.
        """
        refs_clean = [r if r.strip() else "<empty>" for r in refs]
        hyps_clean = [h if h.strip() else "<empty>" for h in hyps]
        return float(jiwer.cer(refs_clean, hyps_clean))

    # ------------------------------------------------------------------
    # 90 % agreement (primary metric)
    # ------------------------------------------------------------------
    @staticmethod
    def compute_90pct_agreement(
        refs: Sequence[str],
        hyps: Sequence[str],
    ) -> float:
        """Fraction of utterances where per-utterance accuracy ≥ 90 %.

        For each ``(ref, hyp)`` pair the utterance-level WER is
        computed.  An utterance "agrees" when ``(1 − utt_wer) ≥ 0.90``,
        i.e. ``utt_wer ≤ 0.10``.

        Parameters
        ----------
        refs : sequence of str
            Reference transcripts.
        hyps : sequence of str
            Hypothesis transcripts.

        Returns
        -------
        float
            Fraction in ``[0, 1]``.
        """
        if len(refs) == 0:
            return 0.0
        agree = 0
        for ref, hyp in zip(refs, hyps):
            r = ref if ref.strip() else "<empty>"
            h = hyp if hyp.strip() else "<empty>"
            utt_wer = jiwer.wer(r, h)
            if (1.0 - utt_wer) >= 0.90:
                agree += 1
        return agree / len(refs)

    # ------------------------------------------------------------------
    # Detailed error breakdown
    # ------------------------------------------------------------------
    @staticmethod
    def compute_error_breakdown(
        refs: Sequence[str],
        hyps: Sequence[str],
    ) -> Dict[str, float]:
        """Return a dict with substitution / insertion / deletion counts.

        Parameters
        ----------
        refs : sequence of str
            Reference transcripts.
        hyps : sequence of str
            Hypothesis transcripts.

        Returns
        -------
        dict
            Keys: ``substitutions``, ``insertions``, ``deletions``,
            ``total_words``, ``wer``, ``cer``.
        """
        refs_clean = [r if r.strip() else "<empty>" for r in refs]
        hyps_clean = [h if h.strip() else "<empty>" for h in hyps]

        # jiwer >= 3.0 uses process_words; fall back to compute_measures
        if hasattr(jiwer, "process_words"):
            out = jiwer.process_words(refs_clean, hyps_clean)
            subs = out.substitutions
            ins = out.insertions
            dels = out.deletions
            hits = out.hits
            wer_val = out.wer
        else:
            # Legacy jiwer < 3.0
            out = jiwer.compute_measures(refs_clean, hyps_clean)  # type: ignore[attr-defined]
            subs = out["substitutions"]
            ins = out["insertions"]
            dels = out["deletions"]
            hits = out["hits"]
            wer_val = out["wer"]

        return {
            "substitutions": subs,
            "insertions": ins,
            "deletions": dels,
            "total_words": subs + dels + hits,
            "wer": float(wer_val),
            "cer": float(jiwer.cer(refs_clean, hyps_clean)),
        }

    # ------------------------------------------------------------------
    # Per-group breakdown
    # ------------------------------------------------------------------
    @staticmethod
    def compute_wer_by_group(
        df: pd.DataFrame,
        group_col: str,
        ref_col: str = "reference_norm",
        hyp_col: str = "hypothesis_norm",
    ) -> pd.DataFrame:
        """Compute WER, CER, 90 % agreement per unique value of *group_col*.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain *group_col*, *ref_col*, *hyp_col*.
        group_col : str
            Column name to group by (e.g. ``"dataset"``, ``"l1_group"``).
        ref_col : str
            Column with normalised reference text.
        hyp_col : str
            Column with normalised hypothesis text.

        Returns
        -------
        pd.DataFrame
            One row per unique group value with columns:
            ``group_col``, ``n_utterances``, ``wer``, ``cer``,
            ``pct_90_agree``.
        """
        evaluator = ASREvaluator()
        rows = []
        for group_val, grp in df.groupby(group_col):
            refs = grp[ref_col].fillna("").tolist()
            hyps = grp[hyp_col].fillna("").tolist()
            rows.append(
                {
                    group_col: group_val,
                    "n_utterances": len(grp),
                    "wer": evaluator.compute_wer(refs, hyps),
                    "cer": evaluator.compute_cer(refs, hyps),
                    "pct_90_agree": evaluator.compute_90pct_agreement(refs, hyps),
                }
            )
        return pd.DataFrame(rows).sort_values("wer")

    # ------------------------------------------------------------------
    # Convenience: evaluate a full results DataFrame
    # ------------------------------------------------------------------
    def evaluate_all(
        self,
        df: pd.DataFrame,
        ref_col: str = "reference_norm",
        hyp_col: str = "hypothesis_norm",
    ) -> Dict[str, object]:
        """Run all metrics on *df* and return a summary dict.

        Parameters
        ----------
        df : pd.DataFrame
            Inference results with normalised ref / hyp columns.
        ref_col, hyp_col : str
            Column names.

        Returns
        -------
        dict
            ``overall`` (dict), ``by_dataset`` (DataFrame),
            ``by_l1_group`` (DataFrame or None),
            ``by_task_type`` (DataFrame or None).
        """
        refs = df[ref_col].fillna("").tolist()
        hyps = df[hyp_col].fillna("").tolist()

        overall = self.compute_error_breakdown(refs, hyps)
        overall["pct_90_agree"] = self.compute_90pct_agreement(refs, hyps)
        overall["n_utterances"] = len(refs)

        result: Dict[str, object] = {"overall": overall}

        if "dataset" in df.columns:
            result["by_dataset"] = self.compute_wer_by_group(
                df, "dataset", ref_col, hyp_col,
            )

        if "l1_group" in df.columns and df["l1_group"].notna().any():
            result["by_l1_group"] = self.compute_wer_by_group(
                df[df["l1_group"].notna()], "l1_group", ref_col, hyp_col,
            )

        if "task_type" in df.columns and df["task_type"].notna().any():
            result["by_task_type"] = self.compute_wer_by_group(
                df[df["task_type"].notna()], "task_type", ref_col, hyp_col,
            )

        return result
