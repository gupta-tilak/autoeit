"""AutoEIT – Data augmentation for training audio.

Augments only the **train** split.  Each dataset has a configurable
multiplier and a list of augmentation strategy presets (speed perturbation,
noise injection, reverb, pitch shift, volume jitter).

Pipeline
--------
1. Load the train manifest, merge with ``processed_manifest.csv`` to get
   ``processed_path`` and ``rejected`` columns.
2. Filter to non-rejected rows.
3. For each utterance, generate *(multiplier − 1)* augmented copies using
   the dataset's strategy list.
4. Save augmented WAV files to ``data/augmented/{dataset}/``.
5. Write ``data/transcripts/train_augmented.csv`` with all original +
   augmented rows.

CLI
---
    python src/augment.py \\
        --input data/transcripts/train.csv \\
        --processed_manifest data/transcripts/processed_manifest.csv \\
        --output data/augmented \\
        --config configs/config.yaml \\
        --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from scipy.signal import fftconvolve

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("autoeit.augment")
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
TARGET_DBFS: float = -20.0


# ---------------------------------------------------------------------------
# Helper – RMS normalisation (shared with preprocess)
# ---------------------------------------------------------------------------

def _rms_db(audio: np.ndarray) -> float:
    """Return RMS level in dBFS."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-10:
        return -120.0
    return float(20 * np.log10(rms))


def _normalize_rms(audio: np.ndarray,
                   target_dbfs: float = TARGET_DBFS) -> np.ndarray:
    """RMS-normalise *audio* to *target_dbfs* and peak-clip to ±0.99."""
    current_rms = np.sqrt(np.mean(audio ** 2))
    if current_rms < 1e-10:
        return audio
    target_rms = 10 ** (target_dbfs / 20.0)
    audio = audio * (target_rms / current_rms)
    return np.clip(audio, -0.99, 0.99)


# ═══════════════════════════════════════════════════════════════════════════
# DataAugmentor
# ═══════════════════════════════════════════════════════════════════════════

class DataAugmentor:
    """Audio data augmentor driven by ``configs/config.yaml``.

    Parameters
    ----------
    config_path : Path to the YAML config file.
    seed        : Global random seed for reproducibility.
    """

    def __init__(self, config_path: str = "configs/config.yaml",
                 seed: int = 42) -> None:
        self.config_path = Path(config_path)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.config: Dict[str, Any] = {}
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as fh:
                self.config = yaml.safe_load(fh) or {}

        self.aug_cfg: Dict[str, Any] = self.config.get("augmentation", {})
        self.target_dbfs: float = float(
            self.aug_cfg.get("target_dbfs", TARGET_DBFS)
        )
        logger.info(
            "DataAugmentor initialised (config=%s, seed=%d)",
            self.config_path, self.seed,
        )

    # ------------------------------------------------------------------
    # Private augmentation primitives
    # ------------------------------------------------------------------

    def _speed_perturb(self, audio: np.ndarray, sr: int,
                       factor: float) -> np.ndarray:
        """Time-stretch by *factor* and re-normalise.

        factor < 1 → slower (longer output), factor > 1 → faster (shorter).
        """
        stretched = librosa.effects.time_stretch(audio, rate=factor)
        return _normalize_rms(stretched, self.target_dbfs)

    def _pitch_shift(self, audio: np.ndarray, sr: int,
                     steps: int) -> np.ndarray:
        """Shift pitch by *steps* semitones."""
        shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps)
        return shifted.astype(np.float32)

    def _add_white_noise(self, audio: np.ndarray,
                         snr_db: float) -> np.ndarray:
        """Add white Gaussian noise at target *snr_db*."""
        noise = self.rng.standard_normal(len(audio)).astype(np.float32)
        sig_rms = np.sqrt(np.mean(audio ** 2))
        noise_target_rms = sig_rms / (10 ** (snr_db / 20.0))
        noise_rms = np.sqrt(np.mean(noise ** 2))
        if noise_rms < 1e-10:
            return audio
        noise = noise * (noise_target_rms / noise_rms)
        return np.clip(audio + noise, -1.0, 1.0).astype(np.float32)

    def _add_pink_noise(self, audio: np.ndarray,
                        snr_db: float) -> np.ndarray:
        """Add 1/f (pink) noise at target *snr_db* via FFT shaping."""
        n = len(audio)
        white = self.rng.standard_normal(n).astype(np.float64)
        # Shape in frequency domain: scale by 1/sqrt(f)
        spectrum = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n, d=1.0 / TARGET_SR)
        freqs[0] = 1.0  # avoid div-by-zero at DC
        spectrum /= np.sqrt(freqs)
        pink = np.fft.irfft(spectrum, n=n).astype(np.float32)

        sig_rms = np.sqrt(np.mean(audio ** 2))
        noise_target_rms = sig_rms / (10 ** (snr_db / 20.0))
        pink_rms = np.sqrt(np.mean(pink ** 2))
        if pink_rms < 1e-10:
            return audio
        pink = pink * (noise_target_rms / pink_rms)
        return np.clip(audio + pink, -1.0, 1.0).astype(np.float32)

    def _add_synthetic_reverb(self, audio: np.ndarray, sr: int,
                              room_size: str = "small") -> np.ndarray:
        """Convolve with an analytical exponential-decay RIR.

        Parameters
        ----------
        room_size : ``"small"`` (RT60 = 0.3 s) or ``"medium"`` (RT60 = 0.6 s).
        """
        rt60_map = {"small": 0.3, "medium": 0.6}
        rt60 = rt60_map.get(room_size, 0.3)
        t = np.arange(0, rt60, 1.0 / sr)
        rir = np.exp(-3.0 * t / rt60) * self.rng.standard_normal(len(t))
        rir = (rir / np.max(np.abs(rir) + 1e-10)).astype(np.float32)
        reverbed = fftconvolve(audio, rir, mode="full")[: len(audio)]
        return _normalize_rms(reverbed.astype(np.float32), self.target_dbfs)

    def _volume_jitter(self, audio: np.ndarray,
                       gain_db_range: Tuple[float, float] = (-4.0, 4.0)
                       ) -> np.ndarray:
        """Apply a random uniform gain in dB."""
        gain_db = self.rng.uniform(gain_db_range[0], gain_db_range[1])
        gain = 10 ** (gain_db / 20.0)
        return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Strategy dispatcher
    # ------------------------------------------------------------------

    def _apply_strategy(self, audio: np.ndarray, sr: int,
                        strategy: Dict[str, Any]) -> np.ndarray:
        """Apply all transforms in a single strategy preset, in order.

        The strategy dict may contain any combination of:
          speed_factor, pitch_steps, noise_type + noise_snr_db,
          reverb, volume_jitter.
        """
        out = audio.copy()

        if "speed_factor" in strategy:
            out = self._speed_perturb(out, sr, strategy["speed_factor"])

        if "pitch_steps" in strategy:
            out = self._pitch_shift(out, sr, int(strategy["pitch_steps"]))

        if "noise_type" in strategy:
            snr = float(strategy.get("noise_snr_db", 15))
            if strategy["noise_type"] == "white":
                out = self._add_white_noise(out, snr)
            elif strategy["noise_type"] == "pink":
                out = self._add_pink_noise(out, snr)

        if "reverb" in strategy:
            out = self._add_synthetic_reverb(out, sr, strategy["reverb"])

        if strategy.get("volume_jitter"):
            out = self._volume_jitter(out)

        return out

    # ------------------------------------------------------------------
    # Public batch method
    # ------------------------------------------------------------------

    def augment_dataset(
        self,
        train_df: pd.DataFrame,
        output_dir: Path,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate augmented copies for every row in *train_df*.

        Parameters
        ----------
        train_df   : Train-split DataFrame **already filtered** to
                     non-rejected rows containing a ``processed_path`` col.
        output_dir : Root directory for augmented WAV files.
        seed       : Deterministic seed (also resets ``self.rng``).

        Returns
        -------
        DataFrame with all original rows (``is_augmented=False``) and
        augmented rows (``is_augmented=True``), plus extra columns
        ``is_augmented``, ``augmentation_type``, ``source_utterance_id``.
        """
        self.rng = np.random.default_rng(seed)
        output_dir = Path(output_dir)

        ds_cfg = self.aug_cfg.get("datasets", {})
        all_rows: List[Dict[str, Any]] = []

        # --- L1 balance check for INMIGRA ---------------------------------
        inmigra_low_l1: set = set()
        inmigra_cfg = ds_cfg.get("Nebrija-INMIGRA", {})
        low_res_mult = int(inmigra_cfg.get("low_resource_multiplier", 6))
        if len(train_df[train_df.dataset == "Nebrija-INMIGRA"]):
            l1_counts = (
                train_df[train_df.dataset == "Nebrija-INMIGRA"]
                .groupby("l1_group")["audio_path"]
                .nunique()
            )
            for l1, cnt in l1_counts.items():
                if cnt < 3:
                    inmigra_low_l1.add(l1)
                    logger.info(
                        "L1 '%s' has only %d sessions → multiplier=%d",
                        l1, cnt, low_res_mult,
                    )

        total_original = 0
        total_augmented = 0

        for ds_name in sorted(train_df.dataset.unique()):
            sub = train_df[train_df.dataset == ds_name].copy()
            cfg = ds_cfg.get(ds_name, {})
            base_multiplier = int(cfg.get("multiplier", 2))
            strategies: List[Dict] = cfg.get("strategies", [])

            ds_out = output_dir / ds_name
            ds_out.mkdir(parents=True, exist_ok=True)

            logger.info(
                "Augmenting %s: %d rows, multiplier=%dx, %d strategies",
                ds_name, len(sub), base_multiplier, len(strategies),
            )

            for _, row in sub.iterrows():
                # Original row
                orig = row.to_dict()
                orig["is_augmented"] = False
                orig["augmentation_type"] = "original"
                orig["source_utterance_id"] = orig["utterance_id"]
                all_rows.append(orig)
                total_original += 1

                # Decide multiplier (may be higher for low-resource L1)
                mult = base_multiplier
                if (ds_name == "Nebrija-INMIGRA"
                        and row.get("l1_group") in inmigra_low_l1):
                    mult = max(mult, low_res_mult)

                n_copies = mult - 1
                if n_copies <= 0 or not strategies:
                    continue

                # Load audio once
                pp = str(row["processed_path"])
                try:
                    audio, sr = librosa.load(pp, sr=TARGET_SR, mono=True)
                except Exception as exc:
                    logger.warning("Cannot load %s: %s", pp, exc)
                    continue

                # Generate copies cycling through strategies
                for copy_idx in range(n_copies):
                    strat = strategies[copy_idx % len(strategies)]
                    tag = strat.get("tag", f"aug{copy_idx}")

                    aug_audio = self._apply_strategy(audio, sr, strat)

                    # Build output path
                    stem = Path(pp).stem
                    out_path = ds_out / f"{stem}_{tag}.wav"
                    sf.write(str(out_path), aug_audio, sr, subtype="PCM_16")

                    aug_row = row.to_dict()
                    aug_row["is_augmented"] = True
                    aug_row["augmentation_type"] = tag
                    aug_row["source_utterance_id"] = row["utterance_id"]
                    aug_row["utterance_id"] = (
                        f"{row['utterance_id']}_{tag}"
                    )
                    aug_row["processed_path"] = str(out_path)
                    all_rows.append(aug_row)
                    total_augmented += 1

        result_df = pd.DataFrame(all_rows)

        # Save
        out_csv = Path("data/transcripts/train_augmented.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(out_csv, index=False)
        logger.info("Saved %s (%d rows)", out_csv, len(result_df))

        # Summary table
        self._print_summary(result_df, ds_cfg)
        return result_df

    # ------------------------------------------------------------------
    @staticmethod
    def _print_summary(df: pd.DataFrame,
                       ds_cfg: Dict[str, Any]) -> None:
        """Print a per-dataset augmentation summary table."""
        border = "─"
        col_w = [19, 10, 10, 10, 14]
        headers = ["Dataset", "Original", "Augmented", "Total", "Multiplier"]

        def _row(vals: List[str]) -> str:
            return "│ " + " │ ".join(
                v.ljust(w) for v, w in zip(vals, col_w)
            ) + " │"

        top = "┌" + "┬".join(border * (w + 2) for w in col_w) + "┐"
        mid = "├" + "┼".join(border * (w + 2) for w in col_w) + "┤"
        bot = "└" + "┴".join(border * (w + 2) for w in col_w) + "┘"

        lines = [top, _row(headers), mid]
        tot_orig = tot_aug = tot_all = 0
        for ds in sorted(df.dataset.unique()):
            sub = df[df.dataset == ds]
            n_orig = int((~sub.is_augmented).sum())
            n_aug = int(sub.is_augmented.sum())
            n_tot = len(sub)
            mult = ds_cfg.get(ds, {}).get("multiplier", "?")
            lines.append(
                _row([ds, str(n_orig), str(n_aug), str(n_tot),
                      f"{mult}x"])
            )
            tot_orig += n_orig
            tot_aug += n_aug
            tot_all += n_tot
        lines.append(mid)
        lines.append(
            _row(["TOTAL", str(tot_orig), str(tot_aug), str(tot_all), ""])
        )
        lines.append(bot)
        logger.info("Augmentation summary:\n%s", "\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AutoEIT data augmentation – expand training set."
    )
    parser.add_argument(
        "--input", type=str, default="data/transcripts/train.csv",
        help="Path to the train-split CSV.",
    )
    parser.add_argument(
        "--processed_manifest", type=str,
        default="data/transcripts/processed_manifest.csv",
        help="Path to processed_manifest.csv (from Phase 2).",
    )
    parser.add_argument(
        "--output", type=str, default="data/augmented",
        help="Root output dir for augmented WAV files.",
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to project config YAML.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    logger.info("AutoEIT data augmentation started")
    logger.info("  input               : %s", args.input)
    logger.info("  processed_manifest  : %s", args.processed_manifest)
    logger.info("  output              : %s", args.output)
    logger.info("  config              : %s", args.config)
    logger.info("  seed                : %d", args.seed)

    # Load train split
    train_df = pd.read_csv(args.input)
    logger.info("Loaded train CSV with %d rows", len(train_df))

    # Merge processed columns if not already present
    if "processed_path" not in train_df.columns:
        pm = pd.read_csv(args.processed_manifest)
        merge_cols = ["utterance_id", "processed_path", "snr_db",
                      "processed_duration_sec", "was_chunked",
                      "rejected", "rejection_reason"]
        merge_cols = [c for c in merge_cols if c in pm.columns]
        train_df = train_df.merge(
            pm[merge_cols], on="utterance_id", how="left",
        )
        logger.info("Merged processed columns from %s", args.processed_manifest)

    # Filter to non-rejected rows
    if "rejected" in train_df.columns:
        before = len(train_df)
        train_df = train_df[train_df.rejected != True].copy()
        logger.info(
            "Filtered rejected rows: %d → %d", before, len(train_df),
        )

    # Speaker leakage check against dev and test
    dev_path = Path("data/transcripts/dev.csv")
    test_path = Path("data/transcripts/test.csv")
    if dev_path.exists() and test_path.exists():
        dev_df = pd.read_csv(dev_path)
        test_df = pd.read_csv(test_path)
        train_speakers = set(train_df.speaker_id.unique())
        dev_speakers = set(dev_df.speaker_id.unique())
        test_speakers = set(test_df.speaker_id.unique())
        overlap_dev = train_speakers & dev_speakers
        overlap_test = train_speakers & test_speakers
        if overlap_dev:
            logger.warning(
                "Speaker overlap train↔dev: %s", overlap_dev,
            )
        if overlap_test:
            logger.warning(
                "Speaker overlap train↔test: %s", overlap_test,
            )
        if not overlap_dev and not overlap_test:
            logger.info("No speaker overlap between train and dev/test.")

    augmentor = DataAugmentor(config_path=args.config, seed=args.seed)
    augmentor.augment_dataset(
        train_df=train_df,
        output_dir=Path(args.output),
        seed=args.seed,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
