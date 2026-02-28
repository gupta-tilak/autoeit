"""Unit tests for src.evaluate and src.infer.normalize_text."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluate import ASREvaluator
from src.infer import normalize_text


# ═══════════════════════════════════════════════════════════════════════════
# normalize_text
# ═══════════════════════════════════════════════════════════════════════════

class TestNormalizeText:
    """Tests for :func:`normalize_text`."""

    def test_punctuation_removal_preserves_diacritics(self) -> None:
        """¿Cómo estás? → 'cómo estás' (diacritics kept, punctuation gone)."""
        assert normalize_text("¿Cómo estás?") == "cómo estás"

    def test_unicode_nfc(self) -> None:
        """Combined and decomposed forms should produce the same output."""
        # 'é' as single codepoint vs e + combining acute
        assert normalize_text("caf\u00e9") == normalize_text("cafe\u0301")

    def test_unintelligible_removal(self) -> None:
        assert normalize_text("hola <unintelligible> mundo") == "hola mundo"

    def test_whitespace_collapse(self) -> None:
        assert normalize_text("  hola   mundo  ") == "hola mundo"

    def test_empty_and_none(self) -> None:
        assert normalize_text("") == ""
        assert normalize_text(None) == ""  # type: ignore[arg-type]

    def test_apostrophe_preserved(self) -> None:
        assert normalize_text("it's fine") == "it's fine"

    def test_lowercase(self) -> None:
        assert normalize_text("HOLA MUNDO") == "hola mundo"


# ═══════════════════════════════════════════════════════════════════════════
# ASREvaluator.compute_wer
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeWER:
    """Tests for :meth:`ASREvaluator.compute_wer`."""

    def test_perfect_match(self) -> None:
        assert ASREvaluator.compute_wer(["hola mundo"], ["hola mundo"]) == 0.0

    def test_diacritics_matter(self) -> None:
        """'niño' vs 'nino' should produce WER > 0."""
        wer = ASREvaluator.compute_wer(["el niño"], ["el nino"])
        assert wer > 0.0, "Diacritics should be significant"

    def test_complete_mismatch(self) -> None:
        wer = ASREvaluator.compute_wer(["uno dos tres"], ["cuatro cinco seis"])
        assert wer == 1.0

    def test_empty_refs_and_hyps(self) -> None:
        """Empty strings are replaced with <empty> sentinel."""
        wer = ASREvaluator.compute_wer([""], [""])
        assert wer == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# ASREvaluator.compute_cer
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeCER:
    def test_perfect_match(self) -> None:
        assert ASREvaluator.compute_cer(["abc"], ["abc"]) == 0.0

    def test_one_char_diff(self) -> None:
        cer = ASREvaluator.compute_cer(["abc"], ["aXc"])
        assert cer > 0.0


# ═══════════════════════════════════════════════════════════════════════════
# ASREvaluator.compute_90pct_agreement
# ═══════════════════════════════════════════════════════════════════════════

class TestCompute90PctAgreement:
    """Tests for the primary metric."""

    def test_all_perfect(self) -> None:
        refs = ["hola"] * 10
        hyps = ["hola"] * 10
        assert ASREvaluator.compute_90pct_agreement(refs, hyps) == 1.0

    def test_known_fraction(self) -> None:
        """9 perfect + 1 at ~50% WER → agreement should be 0.90."""
        refs = ["hola mundo"] * 9 + ["uno dos"]
        hyps = ["hola mundo"] * 9 + ["tres cuatro"]  # 100% WER on last
        result = ASREvaluator.compute_90pct_agreement(refs, hyps)
        assert result == pytest.approx(0.90, abs=0.01)

    def test_all_wrong(self) -> None:
        refs = ["a"] * 5
        hyps = ["b"] * 5
        assert ASREvaluator.compute_90pct_agreement(refs, hyps) == 0.0

    def test_empty(self) -> None:
        assert ASREvaluator.compute_90pct_agreement([], []) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# ASREvaluator.compute_error_breakdown
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeErrorBreakdown:
    def test_keys_present(self) -> None:
        bd = ASREvaluator.compute_error_breakdown(["a b c"], ["a b c"])
        assert set(bd.keys()) == {
            "substitutions", "insertions", "deletions",
            "total_words", "wer", "cer",
        }

    def test_perfect_zero_errors(self) -> None:
        bd = ASREvaluator.compute_error_breakdown(["uno dos"], ["uno dos"])
        assert bd["substitutions"] == 0
        assert bd["insertions"] == 0
        assert bd["deletions"] == 0
        assert bd["wer"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# ASREvaluator.compute_wer_by_group
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeWERByGroup:
    """Test per-group breakdown returns one row per unique group value."""

    def test_one_row_per_group(self) -> None:
        df = pd.DataFrame(
            {
                "dataset": ["A", "A", "B", "B", "C"],
                "reference_norm": ["hola", "mundo", "uno", "dos", "tres"],
                "hypothesis_norm": ["hola", "mundo", "uno", "xxx", "tres"],
            }
        )
        by_group = ASREvaluator.compute_wer_by_group(df, "dataset")
        assert len(by_group) == 3, "Should have one row per unique dataset"
        assert set(by_group["dataset"]) == {"A", "B", "C"}

    def test_columns_present(self) -> None:
        df = pd.DataFrame(
            {
                "l1_group": ["Chinese", "Chinese", "Romanian"],
                "reference_norm": ["a", "b", "c"],
                "hypothesis_norm": ["a", "b", "c"],
            }
        )
        by_group = ASREvaluator.compute_wer_by_group(df, "l1_group")
        expected_cols = {"l1_group", "n_utterances", "wer", "cer", "pct_90_agree"}
        assert expected_cols.issubset(set(by_group.columns))


# ═══════════════════════════════════════════════════════════════════════════
# ASREvaluator.evaluate_all
# ═══════════════════════════════════════════════════════════════════════════

class TestEvaluateAll:
    def test_returns_overall_and_by_dataset(self) -> None:
        df = pd.DataFrame(
            {
                "dataset": ["Nebrija-INMIGRA", "SPLLOC1"],
                "l1_group": ["Chinese", "English"],
                "reference_norm": ["hola", "hello"],
                "hypothesis_norm": ["hola", "hello"],
            }
        )
        evaluator = ASREvaluator()
        out = evaluator.evaluate_all(df)
        assert "overall" in out
        assert out["overall"]["wer"] == 0.0
        assert "by_dataset" in out
