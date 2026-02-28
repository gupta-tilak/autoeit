"""AutoEIT – Inference pipeline.

Provides :class:`ASRInferencer` which wraps **whisper.cpp** (via
``pywhispercpp``) for Whisper models and HuggingFace ``transformers``
for Wav2Vec2 models, enabling fast batch transcription of preprocessed
utterance WAVs.

Two transcription modes are supported:

* **Mode A** (``has_timestamps=True``): each row already has an individual
  utterance WAV clip → transcribe directly.
* **Mode B** (``has_timestamps=False``): multiple rows share the same
  session-level WAV → transcribe once, then align the full hypothesis to
  individual reference transcripts using sliding-window difflib matching.

CLI
---
    python src/infer.py \\
        --test_csv data/transcripts/test_mini.csv \\
        --processed_manifest data/transcripts/processed_manifest.csv \\
        --model openai/whisper-large-v3 \\
        --output results/baseline_whisper-large-v3.csv
"""

from __future__ import annotations

import argparse
import difflib
import logging
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("autoeit.infer")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(_h)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_SR: int = 16_000

# Regex helpers for normalize_text
_PUNCT_RE = re.compile(r"[^\w\s']", re.UNICODE)
_UNINTELLIGIBLE_RE = re.compile(r"<unintelligible>", re.IGNORECASE)
_MULTI_WS_RE = re.compile(r"\s{2,}")


# ═══════════════════════════════════════════════════════════════════════════
# Text normalisation
# ═══════════════════════════════════════════════════════════════════════════

def normalize_text(text: str) -> str:
    """Normalise text for WER/CER evaluation.

    Steps
    -----
    1. Lowercase.
    2. NFC unicode normalisation (preserves Spanish diacritics).
    3. Remove punctuation **except** apostrophes.
    4. Remove ``<unintelligible>`` tokens.
    5. Collapse whitespace and strip.

    Parameters
    ----------
    text : str
        Raw reference or hypothesis text.

    Returns
    -------
    str
        Normalised text suitable for metric computation.
    """
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = unicodedata.normalize("NFC", text)
    # Remove <unintelligible> BEFORE punctuation stripping (so angle
    # brackets are still present for the regex to match).
    text = _UNINTELLIGIBLE_RE.sub("", text)
    text = _PUNCT_RE.sub("", text)
    text = _MULTI_WS_RE.sub(" ", text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════
# Sliding-window session alignment (Mode B)
# ═══════════════════════════════════════════════════════════════════════════

def _align_session_hypothesis(
    full_hyp: str,
    ref_utterances: List[str],
) -> List[str]:
    """Align a full-session hypothesis to individual utterance references.

    Uses :class:`difflib.SequenceMatcher` with a sliding window to
    extract the best-matching substring of *full_hyp* for each
    reference utterance.

    Parameters
    ----------
    full_hyp : str
        Full hypothesis transcript for the entire session.
    ref_utterances : list[str]
        Ordered list of reference transcripts (one per utterance).

    Returns
    -------
    list[str]
        Per-utterance hypothesis fragments, same length as
        *ref_utterances*.
    """
    hyp_words = full_hyp.split()
    per_utt_hyps: List[str] = []
    cursor = 0  # track position in hypothesis

    for ref in ref_utterances:
        ref_words = ref.split()
        if not ref_words:
            per_utt_hyps.append("")
            continue

        window_size = max(len(ref_words) * 3, 20)
        search_start = max(0, cursor - len(ref_words))
        search_end = min(len(hyp_words), cursor + window_size)
        window = hyp_words[search_start:search_end]

        best_ratio = 0.0
        best_start = 0
        best_len = len(ref_words)

        # Slide through window
        for i in range(max(1, len(window) - len(ref_words) + 1)):
            candidate = window[i : i + len(ref_words)]
            if not candidate:
                continue
            ratio = difflib.SequenceMatcher(
                None,
                " ".join(ref_words),
                " ".join(candidate),
            ).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = i
                best_len = len(ref_words)

        abs_start = search_start + best_start
        matched = hyp_words[abs_start : abs_start + best_len]
        per_utt_hyps.append(" ".join(matched))
        cursor = abs_start + best_len

    return per_utt_hyps


# ═══════════════════════════════════════════════════════════════════════════
# ASRInferencer
# ═══════════════════════════════════════════════════════════════════════════

class ASRInferencer:
    """Wrapper around ASR models for batch inference.

    * **Whisper models** are served via ``pywhispercpp`` (whisper.cpp)
      which uses native Metal acceleration on Apple Silicon and is
      significantly faster than the HuggingFace pipeline.
    * **Wav2Vec2 models** use HuggingFace ``transformers`` as before.

    The backend is auto-detected from the *model_name* string.

    Parameters
    ----------
    model_name : str
        Model identifier, e.g. ``"openai/whisper-large-v3"`` or
        ``"jonatasgrosman/wav2vec2-large-xlsr-53-spanish"``.
    device : str
        Device string (``"auto"``, ``"cpu"``, ``"cuda"``, ``"mps"``).
        Only used for Wav2Vec2; whisper.cpp handles device selection
        internally (Metal on macOS, CPU otherwise).
    language : str
        BCP-47 language code passed to Whisper.
    batch_size : int
        Batch size (only used for Wav2Vec2 / HuggingFace pipeline).
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        language: str = "es",
        batch_size: int = 8,
    ) -> None:
        self.model_name = model_name
        self.language = language
        self.batch_size = batch_size
        self.device = self._resolve_device(device)
        self._backend = self._detect_backend(model_name)
        self._whisper_model = None  # pywhispercpp Model
        self._pipeline = None          # (unused, kept for compat)
        self._model = None             # HF model (Wav2Vec2)
        self._processor = None         # HF processor (Wav2Vec2)

        logger.info(
            "ASRInferencer: model=%s  backend=%s  device=%s  batch_size=%d",
            model_name,
            self._backend,
            self.device,
            batch_size,
        )
        self._load_model()

    # ------------------------------------------------------------------
    # Device helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve ``'auto'`` to the best available device."""
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _detect_backend(model_name: str) -> str:
        """Return ``'whisper'`` or ``'wav2vec2'``."""
        name_lower = model_name.lower()
        if "whisper" in name_lower:
            return "whisper"
        if "wav2vec2" in name_lower:
            return "wav2vec2"
        raise ValueError(
            f"Cannot detect backend for model '{model_name}'. "
            "Expected 'whisper' or 'wav2vec2' in model name."
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """Load model and processor/pipeline."""
        if self._backend == "whisper":
            self._load_whisper()
        else:
            self._load_wav2vec2()

    # ------------------------------------------------------------------
    # Whisper model-name mapping
    # ------------------------------------------------------------------
    _WHISPER_NAME_MAP: Dict[str, str] = {
        "openai/whisper-tiny": "tiny",
        "openai/whisper-base": "base",
        "openai/whisper-small": "small",
        "openai/whisper-medium": "medium",
        "openai/whisper-large": "large",
        "openai/whisper-large-v2": "large-v2",
        "openai/whisper-large-v3": "large-v3",
        "openai/whisper-large-v3-turbo": "large-v3-turbo",
    }

    def _load_whisper(self) -> None:
        """Load Whisper via ``pywhispercpp`` (whisper.cpp).

        whisper.cpp uses Metal acceleration on Apple Silicon and CPU
        SIMD elsewhere, bypassing all PyTorch device/dtype issues.
        GGML model files are auto-downloaded on first use to
        ``~/Library/Application Support/pywhispercpp/models/``.
        """
        from pywhispercpp.model import Model as WhisperCppModel

        # Map HuggingFace-style name → whisper.cpp model name
        cpp_name = self._WHISPER_NAME_MAP.get(self.model_name)
        if cpp_name is None:
            # Fallback: try using the part after the last '/'
            cpp_name = self.model_name.rsplit("/", 1)[-1]
            # Strip "whisper-" prefix if present
            if cpp_name.startswith("whisper-"):
                cpp_name = cpp_name[len("whisper-"):]
            logger.warning(
                "Model '%s' not in _WHISPER_NAME_MAP; guessing cpp name '%s'",
                self.model_name, cpp_name,
            )

        logger.info("Loading whisper.cpp model: %s", cpp_name)
        self._whisper_model = WhisperCppModel(cpp_name)
        logger.info("whisper.cpp model loaded: %s", cpp_name)

    def _load_wav2vec2(self) -> None:
        """Load Wav2Vec2ForCTC + Wav2Vec2Processor."""
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self._processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self._model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()
        logger.info("Wav2Vec2 model loaded: %s", self.model_name)

    # ------------------------------------------------------------------
    # Single-file transcription
    # ------------------------------------------------------------------
    def _transcribe_file(self, audio_path: str) -> Tuple[str, float]:
        """Transcribe a single audio file.

        Parameters
        ----------
        audio_path : str
            Path to a 16 kHz mono WAV.

        Returns
        -------
        tuple[str, float]
            ``(hypothesis_text, processing_seconds)``
        """
        t0 = time.perf_counter()

        if self._backend == "whisper":
            # pywhispercpp accepts a file path directly; it handles
            # loading, resampling, and Metal/CPU dispatch internally.
            segments = self._whisper_model.transcribe(
                audio_path, language=self.language,
            )
            hyp = " ".join(seg.text.strip() for seg in segments).strip()
        else:
            audio, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
            # Wav2Vec2
            inputs = self._processor(
                audio,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs.input_values.to(self.device)
            with torch.no_grad():
                logits = self._model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            hyp = self._processor.batch_decode(predicted_ids)[0]

        elapsed = time.perf_counter() - t0
        return hyp.strip(), elapsed

    # ------------------------------------------------------------------
    # Batch transcription
    # ------------------------------------------------------------------
    def transcribe_batch(
        self,
        df: pd.DataFrame,
        audio_col: str = "processed_path",
        ref_col: str = "transcript",
    ) -> pd.DataFrame:
        """Transcribe all rows in *df* and return results.

        Handles both Mode A (per-utterance clips) and Mode B (session
        files shared across rows) automatically based on the
        ``has_timestamps`` column.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain *audio_col*, *ref_col*, and ``has_timestamps``.
        audio_col : str
            Column with paths to processed WAV files.
        ref_col : str
            Column with reference transcripts.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with added columns:
            ``hypothesis``, ``hypothesis_raw``, ``reference_norm``,
            ``hypothesis_norm``, ``rtf``, ``aligned_from_session``.
        """
        result = df.copy()
        result["hypothesis_raw"] = ""
        result["hypothesis"] = ""
        result["reference_norm"] = ""
        result["hypothesis_norm"] = ""
        result["rtf"] = np.nan
        result["aligned_from_session"] = False

        # --- Identify Mode A vs Mode B rows ---
        has_ts_col = "has_timestamps"
        if has_ts_col not in result.columns:
            result[has_ts_col] = True  # default to Mode A

        mode_a = result[result[has_ts_col] == True]
        mode_b = result[result[has_ts_col] != True]

        # ── Mode A: per-utterance clips ──────────────────────────────
        if len(mode_a) > 0:
            logger.info("Mode A: transcribing %d individual utterance clips", len(mode_a))
            total = len(mode_a)
            for i, (idx, row) in enumerate(mode_a.iterrows()):
                audio_path = row[audio_col]
                if pd.isna(audio_path) or not Path(audio_path).exists():
                    logger.warning("Missing audio: %s  (row %s)", audio_path, idx)
                    result.at[idx, "hypothesis_raw"] = ""
                    result.at[idx, "hypothesis"] = ""
                    continue
                hyp_raw, elapsed = self._transcribe_file(str(audio_path))
                # Compute RTF
                try:
                    audio_dur = librosa.get_duration(path=str(audio_path))
                except Exception:
                    audio_dur = 1.0
                rtf = elapsed / max(audio_dur, 0.01)

                result.at[idx, "hypothesis_raw"] = hyp_raw
                result.at[idx, "hypothesis"] = normalize_text(hyp_raw)
                result.at[idx, "reference_norm"] = normalize_text(
                    str(row[ref_col]) if pd.notna(row[ref_col]) else ""
                )
                result.at[idx, "hypothesis_norm"] = normalize_text(hyp_raw)
                result.at[idx, "rtf"] = rtf

                if (i + 1) % 50 == 0 or (i + 1) == total:
                    logger.info(
                        "  Mode A progress: %d / %d  (%.1f%%)",
                        i + 1, total, 100 * (i + 1) / total,
                    )

        # ── Mode B: session-level files ──────────────────────────────
        if len(mode_b) > 0:
            logger.info("Mode B: transcribing %d rows (session-level)", len(mode_b))
            grouped = mode_b.groupby(audio_col)
            for session_path, group in grouped:
                if pd.isna(session_path) or not Path(str(session_path)).exists():
                    logger.warning("Missing session audio: %s", session_path)
                    continue

                hyp_raw, elapsed = self._transcribe_file(str(session_path))
                hyp_norm = normalize_text(hyp_raw)

                # Align to per-utterance references
                refs = [
                    normalize_text(str(r) if pd.notna(r) else "")
                    for r in group[ref_col]
                ]
                aligned_hyps = _align_session_hypothesis(hyp_norm, refs)

                try:
                    audio_dur = librosa.get_duration(path=str(session_path))
                except Exception:
                    audio_dur = 1.0
                rtf = elapsed / max(audio_dur, 0.01)

                for (row_idx, _), aligned_hyp, ref_norm in zip(
                    group.iterrows(), aligned_hyps, refs
                ):
                    result.at[row_idx, "hypothesis_raw"] = hyp_raw
                    result.at[row_idx, "hypothesis"] = aligned_hyp
                    result.at[row_idx, "reference_norm"] = ref_norm
                    result.at[row_idx, "hypothesis_norm"] = aligned_hyp
                    result.at[row_idx, "rtf"] = rtf
                    result.at[row_idx, "aligned_from_session"] = True

        # ── Normalise references for any remaining rows ──────────────
        mask = result["reference_norm"] == ""
        result.loc[mask, "reference_norm"] = result.loc[mask, ref_col].apply(
            lambda x: normalize_text(str(x) if pd.notna(x) else "")
        )

        return result


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AutoEIT – ASR inference on test splits.",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="data/transcripts/test_mini.csv",
        help="Path to the test CSV (test.csv or test_mini.csv).",
    )
    parser.add_argument(
        "--processed_manifest",
        type=str,
        default="data/transcripts/processed_manifest.csv",
        help="Processed manifest with processed_path column.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/whisper-large-v3",
        help="HuggingFace model identifier.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path. Default: results/baseline_{model_short}.csv",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="es",
        help="Language code for Whisper.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for the pipeline.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cpu, cuda, mps.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry-point: load test CSV, run inference, save results."""
    args = _parse_args()

    # ── Load test data ────────────────────────────────────────────────
    logger.info("Loading test CSV: %s", args.test_csv)
    test_df = pd.read_csv(args.test_csv)

    # ── Merge with processed manifest to get processed_path ──────────
    if "processed_path" not in test_df.columns:
        logger.info("Merging with processed manifest: %s", args.processed_manifest)
        pm = pd.read_csv(args.processed_manifest)
        merge_cols = ["utterance_id", "processed_path", "rejected"]
        if "rejection_reason" in pm.columns:
            merge_cols.append("rejection_reason")
        test_df = test_df.merge(
            pm[merge_cols],
            on="utterance_id",
            how="left",
        )
        before = len(test_df)
        test_df = test_df[
            (test_df["processed_path"].notna())
            & (test_df["rejected"] != True)
        ].copy()
        logger.info(
            "After filtering: %d / %d rows with valid processed audio",
            len(test_df),
            before,
        )

    # ── Initialise inferencer ─────────────────────────────────────────
    inferencer = ASRInferencer(
        model_name=args.model,
        device=args.device,
        language=args.language,
        batch_size=args.batch_size,
    )

    # ── Run inference ─────────────────────────────────────────────────
    logger.info("Starting transcription of %d utterances…", len(test_df))
    results_df = inferencer.transcribe_batch(test_df)

    # ── Save results ──────────────────────────────────────────────────
    if args.output is None:
        model_short = args.model.replace("/", "_")
        output_path = f"results/baseline_{model_short}.csv"
    else:
        output_path = args.output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info("Results saved to %s  (%d rows)", output_path, len(results_df))


if __name__ == "__main__":
    main()
