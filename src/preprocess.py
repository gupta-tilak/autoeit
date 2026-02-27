"""AutoEIT - Audio preprocessing (resample, denoise, segment).

Pipeline
--------
1. Load raw MP3 (librosa), convert stereo → mono, optionally extract
   utterance clip via start_ms / end_ms.
2. Resample to 16 kHz (kaiser_best).
3. Noise reduction (noisereduce) + Butterworth band-pass 80-7600 Hz.
4. VAD trim of leading/trailing silence (webrtcvad, aggressiveness=2).
5. RMS normalisation to - 20 dBFS, peak-clip ±0.99.
6. Quality gate: reject if SNR < threshold, duration < 0.5 s or > 30 s
   (>30 s triggers chunking instead of rejection).
7. Chunking: 25 s segments with 2 s overlap for long files.
8. Save 16-bit PCM WAV.

CLI
---
    python src/preprocess.py \\
        --manifest data/transcripts/master.csv \\
        --output   data/processed \\
        --config   configs/config.yaml \\
        --jobs     4
"""

from __future__ import annotations

import argparse
import logging
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import noisereduce as nr
import pandas as pd
import soundfile as sf
import yaml
from joblib import Parallel, delayed
from scipy.signal import butter, sosfilt

try:
    import webrtcvad
    _HAS_WEBRTCVAD = True
except (ImportError, ModuleNotFoundError):
    _HAS_WEBRTCVAD = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("autoeit.preprocess")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_h)

# ---------------------------------------------------------------------------
# Default dataset parameters
# ---------------------------------------------------------------------------
DATASET_DEFAULTS: Dict[str, Dict[str, float]] = {
    "Nebrija-INMIGRA": {"denoise_strength": 0.80, "snr_threshold": 5.0},
    "Nebrija-WOCAE":   {"denoise_strength": 0.75, "snr_threshold": 4.0},
    "SPLLOC1":         {"denoise_strength": 0.70, "snr_threshold": 5.0},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TARGET_SR: int = 16_000
TARGET_DBFS: float = -20.0
CHUNK_LEN_SEC: float = 25.0
CHUNK_OVERLAP_SEC: float = 2.0
MIN_DURATION_SEC: float = 0.5
MAX_DURATION_SEC: float = 30.0
PAD_SEC: float = 0.15


def _rms_db(audio: np.ndarray) -> float:
    """Return RMS in dBFS."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-10:
        return -120.0
    return float(20 * np.log10(rms))


def _estimate_snr(audio: np.ndarray, sr: int) -> float:
    """Estimate SNR by comparing speech RMS to noise-floor RMS.

    Uses a simple energy-based partitioning: frames above the median
    energy are treated as signal; the rest as noise.
    """
    frame_len = int(0.025 * sr)  # 25 ms frames
    hop = frame_len
    n_frames = max(1, len(audio) // hop)
    energies = np.array(
        [np.mean(audio[i * hop: i * hop + frame_len] ** 2)
         for i in range(n_frames)]
    )
    median_e = np.median(energies)
    signal_e = energies[energies >= median_e]
    noise_e = energies[energies < median_e]
    sig_rms = np.sqrt(np.mean(signal_e)) if len(signal_e) else 1e-10
    noise_rms = np.sqrt(np.mean(noise_e)) if len(noise_e) else 1e-10
    noise_rms = max(noise_rms, 1e-10)
    return float(20 * np.log10(sig_rms / noise_rms))


def _butterworth_bandpass(audio: np.ndarray, sr: int,
                          low: float = 80.0,
                          high: float = 7600.0,
                          order: int = 4) -> np.ndarray:
    """Apply Butterworth band-pass (high-pass + low-pass) filter."""
    nyq = sr / 2.0
    # High-pass
    sos_hp = butter(order, low / nyq, btype="high", output="sos")
    audio = sosfilt(sos_hp, audio)
    # Low-pass
    sos_lp = butter(order, high / nyq, btype="low", output="sos")
    audio = sosfilt(sos_lp, audio)
    return audio


def _vad_trim(audio: np.ndarray, sr: int,
              aggressiveness: int = 2,
              frame_ms: int = 30) -> np.ndarray:
    """Remove leading and trailing silence using webrtcvad (preferred)
    or an energy-based fallback.

    Parameters
    ----------
    audio : 1-D float32 in [-1, 1]
    sr : must be 8000, 16000, 32000, or 48000 for webrtcvad
    aggressiveness : 0–3
    frame_ms : 10, 20, or 30
    """
    if not _HAS_WEBRTCVAD:
        return _energy_vad_trim(audio, sr, frame_ms=frame_ms)

    vad = webrtcvad.Vad(aggressiveness)
    frame_len = int(sr * frame_ms / 1000)
    # Convert float → 16-bit PCM bytes
    pcm_int16 = np.clip(audio, -1.0, 1.0)
    pcm_int16 = (pcm_int16 * 32767).astype(np.int16)
    raw_bytes = pcm_int16.tobytes()
    n_frames = len(pcm_int16) // frame_len

    voiced = []
    for i in range(n_frames):
        start = i * frame_len * 2  # 2 bytes per sample
        end = start + frame_len * 2
        chunk = raw_bytes[start:end]
        if len(chunk) < frame_len * 2:
            break
        try:
            voiced.append(vad.is_speech(chunk, sr))
        except Exception:
            voiced.append(True)  # keep frame on error

    if not voiced or not any(voiced):
        return audio  # nothing to trim

    # Find first and last voiced frame
    first = next(i for i, v in enumerate(voiced) if v)
    last = len(voiced) - 1 - next(i for i, v in enumerate(reversed(voiced)) if v)
    start_sample = first * frame_len
    end_sample = min((last + 1) * frame_len, len(audio))
    return audio[start_sample:end_sample]


def _energy_vad_trim(audio: np.ndarray, sr: int,
                     frame_ms: int = 30,
                     energy_thresh_db: float = -40.0) -> np.ndarray:
    """Energy-based VAD trim fallback when webrtcvad is unavailable."""
    frame_len = int(sr * frame_ms / 1000)
    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return audio

    energies = np.array([
        np.mean(audio[i * frame_len:(i + 1) * frame_len] ** 2)
        for i in range(n_frames)
    ])
    db = 10 * np.log10(energies + 1e-10)
    voiced = db > energy_thresh_db

    if not any(voiced):
        return audio
    first = next(i for i, v in enumerate(voiced) if v)
    last = len(voiced) - 1 - next(i for i, v in enumerate(reversed(voiced)) if v)
    return audio[first * frame_len: min((last + 1) * frame_len, len(audio))]


def _normalize_rms(audio: np.ndarray, target_dbfs: float = TARGET_DBFS) -> np.ndarray:
    """RMS-normalise to *target_dbfs* and peak-clip to ±0.99."""
    current_rms = np.sqrt(np.mean(audio ** 2))
    if current_rms < 1e-10:
        return audio
    target_rms = 10 ** (target_dbfs / 20.0)
    gain = target_rms / current_rms
    audio = audio * gain
    audio = np.clip(audio, -0.99, 0.99)
    return audio


def _chunk_audio(audio: np.ndarray, sr: int,
                 chunk_sec: float = CHUNK_LEN_SEC,
                 overlap_sec: float = CHUNK_OVERLAP_SEC) -> List[np.ndarray]:
    """Split audio into fixed-length chunks with overlap."""
    chunk_len = int(chunk_sec * sr)
    hop = int((chunk_sec - overlap_sec) * sr)
    chunks: List[np.ndarray] = []
    start = 0
    while start < len(audio):
        end = start + chunk_len
        chunk = audio[start:end]
        if len(chunk) >= int(MIN_DURATION_SEC * sr):
            chunks.append(chunk)
        start += hop
    return chunks


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AudioPreprocessor:
    """End-to-end audio preprocessor for AutoEIT corpora.

    Parameters
    ----------
    config_path : path to the project YAML config file.
    """

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as fh:
                self.config = yaml.safe_load(fh) or {}
        self.dataset_params = self._load_dataset_params()
        logger.info("AudioPreprocessor initialised (config=%s)", self.config_path)

    # ------------------------------------------------------------------
    def _load_dataset_params(self) -> Dict[str, Dict[str, float]]:
        """Merge dataset params from config with built-in defaults."""
        params = {k: dict(v) for k, v in DATASET_DEFAULTS.items()}
        cfg_ds = self.config.get("datasets", {})
        for name, vals in cfg_ds.items():
            if name not in params:
                params[name] = {}
            if isinstance(vals, dict):
                if "denoise_strength" in vals:
                    params[name]["denoise_strength"] = float(vals["denoise_strength"])
                if "snr_threshold" in vals:
                    params[name]["snr_threshold"] = float(vals["snr_threshold"])
        # Ensure every dataset at least has fallback values
        for name in params:
            params[name].setdefault("denoise_strength", 0.75)
            params[name].setdefault("snr_threshold", 5.0)
        return params

    # ------------------------------------------------------------------
    def _params_for(self, dataset: str) -> Dict[str, float]:
        return self.dataset_params.get(dataset, {"denoise_strength": 0.75, "snr_threshold": 5.0})

    # ------------------------------------------------------------------
    # MAIN PROCESSING METHOD
    # ------------------------------------------------------------------
    def process_file(
        self,
        input_path: Path,
        output_path: Path,
        dataset: str,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Process a single audio file through the full pipeline.

        Parameters
        ----------
        input_path  : Path to the raw audio file (.mp3 or any librosa-readable).
        output_path : Desired output .wav path (before potential chunking suffix).
        dataset     : Dataset name used to look up denoise / SNR params.
        start_ms    : If provided, extract utterance starting at this ms offset.
        end_ms      : If provided, extract utterance ending at this ms offset.

        Returns
        -------
        dict with keys: output_path, dataset, snr_db, original_duration,
             processed_duration, was_chunked, n_chunks, rejected,
             rejection_reason, chunk_paths
        """
        result: Dict[str, Any] = {
            "output_path": str(output_path),
            "dataset": dataset,
            "snr_db": 0.0,
            "original_duration": 0.0,
            "processed_duration": 0.0,
            "was_chunked": False,
            "n_chunks": 0,
            "rejected": False,
            "rejection_reason": None,
            "chunk_paths": [],
        }
        params = self._params_for(dataset)

        # STEP 1 — LOAD ------------------------------------------------
        try:
            audio, native_sr = librosa.load(str(input_path), sr=None, mono=False)
        except Exception as exc:
            logger.error("Failed to load %s: %s", input_path, exc)
            result["rejected"] = True
            result["rejection_reason"] = f"load_error: {exc}"
            return result

        # Stereo → mono
        if audio.ndim > 1:
            audio = audio.mean(axis=0)

        result["original_duration"] = float(len(audio) / native_sr)

        # Extract utterance clip if timestamps provided
        if start_ms is not None and end_ms is not None:
            pad = int(PAD_SEC * native_sr)
            sample_start = int(start_ms / 1000.0 * native_sr)
            sample_end = int(end_ms / 1000.0 * native_sr)
            audio = audio[max(0, sample_start - pad): sample_end + pad]

        # Reject too short
        duration_sec = len(audio) / native_sr
        if duration_sec < MIN_DURATION_SEC:
            result["rejected"] = True
            result["rejection_reason"] = f"too_short_after_extract ({duration_sec:.2f}s)"
            result["processed_duration"] = duration_sec
            return result

        # STEP 2 — RESAMPLE -------------------------------------------
        if native_sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=native_sr, target_sr=TARGET_SR,
                                     res_type="kaiser_best")
        audio = audio.astype(np.float32)

        # STEP 3 — NOISE REDUCTION + BANDPASS --------------------------
        audio = nr.reduce_noise(
            y=audio, sr=TARGET_SR, stationary=False,
            prop_decrease=params["denoise_strength"],
        )
        audio = _butterworth_bandpass(audio, TARGET_SR).astype(np.float32)

        # STEP 4 — VAD TRIM (full sessions only, not utterance clips)
        if start_ms is None:
            audio = _vad_trim(audio, TARGET_SR)

        # STEP 5 — RMS NORMALISATION -----------------------------------
        audio = _normalize_rms(audio, TARGET_DBFS)

        # STEP 6 — QUALITY CHECK --------------------------------------
        snr = _estimate_snr(audio, TARGET_SR)
        result["snr_db"] = round(snr, 2)
        processed_dur = len(audio) / TARGET_SR
        result["processed_duration"] = round(processed_dur, 3)

        if snr < params["snr_threshold"]:
            result["rejected"] = True
            result["rejection_reason"] = f"low_snr ({snr:.1f} dB < {params['snr_threshold']} dB)"
            return result

        if processed_dur < MIN_DURATION_SEC:
            result["rejected"] = True
            result["rejection_reason"] = f"too_short ({processed_dur:.2f}s)"
            return result

        # STEP 7 — CHUNKING (>30 s) -----------------------------------
        if processed_dur > MAX_DURATION_SEC:
            chunks = _chunk_audio(audio, TARGET_SR)
            result["was_chunked"] = True
            result["n_chunks"] = len(chunks)
            stem = output_path.stem
            parent = output_path.parent
            parent.mkdir(parents=True, exist_ok=True)
            chunk_paths: List[str] = []
            for idx, chunk in enumerate(chunks):
                cp = parent / f"{stem}_chunk{idx:03d}.wav"
                sf.write(str(cp), chunk, TARGET_SR, subtype="PCM_16")
                chunk_paths.append(str(cp))
            result["chunk_paths"] = chunk_paths
            result["output_path"] = chunk_paths[0] if chunk_paths else str(output_path)
            logger.info("Chunked %s → %d chunks", input_path.name, len(chunks))
            return result

        # STEP 8 — SAVE -----------------------------------------------
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio, TARGET_SR, subtype="PCM_16")
        result["output_path"] = str(output_path)
        logger.info("Saved %s (%.1fs, SNR=%.1f dB)", output_path.name,
                     processed_dur, snr)
        return result

    # ------------------------------------------------------------------
    # BATCH PROCESSING
    # ------------------------------------------------------------------
    def process_from_manifest(
        self,
        df: pd.DataFrame,
        output_root: Path,
        n_jobs: int = 4,
    ) -> pd.DataFrame:
        """Process every audio file referenced in *df* and return an
        updated DataFrame with processing metadata.

        Parameters
        ----------
        df          : DataFrame with columns audio_path, dataset,
                      has_timestamps, start_ms, end_ms, speaker_id,
                      utterance_index, etc.
        output_root : Root output directory (e.g. ``data/processed``).
        n_jobs      : Parallel workers passed to joblib.

        Returns
        -------
        Updated DataFrame with new columns:
          processed_path, snr_db, processed_duration_sec,
          was_chunked, rejected, rejection_reason
        """
        output_root = Path(output_root)

        # Initialise new columns
        df = df.copy()
        df["processed_path"] = None
        df["snr_db"] = np.nan
        df["processed_duration_sec"] = np.nan
        df["was_chunked"] = False
        df["rejected"] = False
        df["rejection_reason"] = None

        # Group by audio_path so each session file is processed ONCE
        groups = df.groupby("audio_path", sort=False)

        tasks: List[Tuple[str, pd.DataFrame]] = list(groups)
        logger.info("Processing %d unique audio sessions …", len(tasks))

        def _process_session(audio_path_str: str, rows: pd.DataFrame) -> List[Dict]:
            """Process one audio session and return per-row results."""
            audio_path = Path(audio_path_str)
            dataset = rows.iloc[0]["dataset"]
            out_dir = output_root / dataset

            has_ts = rows.iloc[0].get("has_timestamps", False)
            # Normalise bool-like string / actual bool
            if isinstance(has_ts, str):
                has_ts = has_ts.strip().lower() == "true"

            per_row_results: List[Dict] = []

            if has_ts:
                # Process each utterance individually
                for _, row in rows.iterrows():
                    s_ms = row.get("start_ms")
                    e_ms = row.get("end_ms")
                    if pd.isna(s_ms) or pd.isna(e_ms):
                        per_row_results.append({
                            "idx": row.name,
                            "processed_path": None,
                            "snr_db": np.nan,
                            "processed_duration_sec": np.nan,
                            "was_chunked": False,
                            "rejected": True,
                            "rejection_reason": "missing_timestamps",
                        })
                        continue
                    s_ms = int(float(s_ms))
                    e_ms = int(float(e_ms))
                    spk = row.get("speaker_id", "unknown")
                    utt_idx = row.get("utterance_index", 0)
                    out_name = f"{spk}_utt{int(utt_idx):04d}.wav"
                    out_path = out_dir / out_name

                    res = self.process_file(
                        input_path=audio_path,
                        output_path=out_path,
                        dataset=dataset,
                        start_ms=s_ms,
                        end_ms=e_ms,
                    )
                    per_row_results.append({
                        "idx": row.name,
                        "processed_path": res["output_path"],
                        "snr_db": res["snr_db"],
                        "processed_duration_sec": res["processed_duration"],
                        "was_chunked": res["was_chunked"],
                        "rejected": res["rejected"],
                        "rejection_reason": res["rejection_reason"],
                    })
            else:
                # Full session — process once, assign to all rows
                media_stem = audio_path.stem
                out_path = out_dir / f"{media_stem}.wav"
                res = self.process_file(
                    input_path=audio_path,
                    output_path=out_path,
                    dataset=dataset,
                )
                for _, row in rows.iterrows():
                    per_row_results.append({
                        "idx": row.name,
                        "processed_path": res["output_path"],
                        "snr_db": res["snr_db"],
                        "processed_duration_sec": res["processed_duration"],
                        "was_chunked": res["was_chunked"],
                        "rejected": res["rejected"],
                        "rejection_reason": res["rejection_reason"],
                    })
            return per_row_results

        # Run in parallel -----------------------------------------------
        all_results: List[List[Dict]] = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_process_session)(ap, rows) for ap, rows in tasks
        )

        # Merge back into DataFrame
        for session_results in all_results:
            for r in session_results:
                idx = r["idx"]
                df.at[idx, "processed_path"] = r["processed_path"]
                df.at[idx, "snr_db"] = r["snr_db"]
                df.at[idx, "processed_duration_sec"] = r["processed_duration_sec"]
                df.at[idx, "was_chunked"] = r["was_chunked"]
                df.at[idx, "rejected"] = r["rejected"]
                df.at[idx, "rejection_reason"] = r["rejection_reason"]

        # Save manifest
        manifest_out = Path("data/transcripts/processed_manifest.csv")
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(manifest_out, index=False)
        logger.info("Saved processed manifest → %s", manifest_out)

        # Per-dataset summary table
        self._print_summary(df)
        return df

    # ------------------------------------------------------------------
    @staticmethod
    def _print_summary(df: pd.DataFrame) -> None:
        """Print a per-dataset processing summary table."""
        border = "─"
        datasets = sorted(df["dataset"].unique())
        col_w = [19, 10, 10, 10, 10]
        headers = ["Dataset", "Sessions", "Accepted", "Rejected", "Avg SNR"]

        def _row(vals: List[str]) -> str:
            return "│ " + " │ ".join(v.ljust(w) for v, w in zip(vals, col_w)) + " │"

        top = "┌" + "┬".join(border * (w + 2) for w in col_w) + "┐"
        mid = "├" + "┼".join(border * (w + 2) for w in col_w) + "┤"
        bot = "└" + "┴".join(border * (w + 2) for w in col_w) + "┘"

        lines = [top, _row(headers), mid]
        for ds in datasets:
            sub = df[df["dataset"] == ds]
            n_sessions = sub["audio_path"].nunique() if "audio_path" in sub.columns else len(sub)
            n_rejected = int(sub["rejected"].sum())
            n_accepted = n_sessions - n_rejected  # approximate per-session
            # Actually, rejected is per-utterance; compute per-session more carefully
            if "audio_path" in sub.columns:
                session_rejected = sub.groupby("audio_path")["rejected"].any().sum()
                n_accepted = n_sessions - int(session_rejected)
                n_rejected = int(session_rejected)
            avg_snr = sub["snr_db"].mean()
            avg_snr_str = f"{avg_snr:.1f} dB" if not pd.isna(avg_snr) else "N/A"
            lines.append(_row([
                ds, str(n_sessions), str(n_accepted),
                str(n_rejected), avg_snr_str,
            ]))
        lines.append(bot)
        summary = "\n".join(lines)
        logger.info("Processing summary:\n%s", summary)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AutoEIT audio preprocessor – convert raw MP3 to clean 16 kHz WAV."
    )
    parser.add_argument(
        "--manifest", type=str, default="data/transcripts/master.csv",
        help="Path to input manifest CSV (master or per-split).",
    )
    parser.add_argument(
        "--output", type=str, default="data/processed",
        help="Root output directory for processed WAVs.",
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to project config YAML.",
    )
    parser.add_argument(
        "--jobs", type=int, default=4,
        help="Number of parallel workers.",
    )
    args = parser.parse_args()

    logger.info("AutoEIT audio preprocessing started")
    logger.info("  manifest : %s", args.manifest)
    logger.info("  output   : %s", args.output)
    logger.info("  config   : %s", args.config)
    logger.info("  jobs     : %d", args.jobs)

    preprocessor = AudioPreprocessor(config_path=args.config)
    df = pd.read_csv(args.manifest)
    logger.info("Loaded manifest with %d rows", len(df))

    preprocessor.process_from_manifest(
        df=df,
        output_root=Path(args.output),
        n_jobs=args.jobs,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
