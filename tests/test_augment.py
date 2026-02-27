"""Unit tests for src.augment – DataAugmentor pipeline.

Tests
-----
1. Speed 0.9x → output length ≈ input_length / 0.9 (±2 %).
2. White noise at target SNR → measured SNR ≈ target ±3 dB.
3. Mini-DataFrame (5 rows, 2 datasets) → correct original + augmented count.
4. Same seed → bit-identical output (md5 check).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.augment import DataAugmentor, _normalize_rms, _rms_db


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def augmentor(tmp_path: Path) -> DataAugmentor:
    """Create an augmentor with a minimal YAML config."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "augmentation:\n"
        "  seed: 42\n"
        "  target_dbfs: -20.0\n"
        "  datasets:\n"
        "    DS_A:\n"
        "      multiplier: 3\n"
        "      strategies:\n"
        "        - tag: sp090_wn15\n"
        "          speed_factor: 0.90\n"
        "          noise_type: white\n"
        "          noise_snr_db: 15\n"
        "        - tag: sp110_pn18\n"
        "          speed_factor: 1.10\n"
        "          noise_type: pink\n"
        "          noise_snr_db: 18\n"
        "    DS_B:\n"
        "      multiplier: 2\n"
        "      strategies:\n"
        "        - tag: sp092_vj\n"
        "          speed_factor: 0.92\n"
        "          volume_jitter: true\n"
    )
    return DataAugmentor(config_path=str(cfg), seed=42)


def _make_sine(path: Path, duration: float = 3.0,
               sr: int = 16000, freq: float = 440.0) -> Path:
    """Write a 16 kHz mono sine WAV file and return *path*."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False,
                    dtype=np.float32)
    audio = 0.3 * np.sin(2 * np.pi * freq * t)
    sf.write(str(path), audio, sr, subtype="PCM_16")
    return path


# ---------------------------------------------------------------------------
# 1. Speed perturbation length
# ---------------------------------------------------------------------------

def test_speed_perturb_length(augmentor: DataAugmentor) -> None:
    """0.9x time-stretch → output ≈ input / 0.9 in length (±2 %)."""
    sr = 16000
    dur = 3.0
    n = int(sr * dur)
    audio = 0.3 * np.sin(
        2 * np.pi * 440.0 * np.linspace(0, dur, n, endpoint=False,
                                         dtype=np.float32)
    )

    out = augmentor._speed_perturb(audio, sr, factor=0.9)
    expected = n / 0.9
    ratio = len(out) / expected
    assert 0.98 <= ratio <= 1.02, (
        f"Length ratio {ratio:.4f} outside ±2 % of expected"
    )


# ---------------------------------------------------------------------------
# 2. White noise SNR accuracy
# ---------------------------------------------------------------------------

def test_white_noise_snr(augmentor: DataAugmentor) -> None:
    """Added white noise at 15 dB → measured SNR ≈ 15 ±3 dB."""
    sr = 16000
    dur = 5.0
    n = int(sr * dur)
    signal = 0.3 * np.sin(
        2 * np.pi * 440.0 * np.linspace(0, dur, n, endpoint=False,
                                         dtype=np.float32)
    )

    noisy = augmentor._add_white_noise(signal.copy(), snr_db=15.0)
    noise_only = noisy - signal
    sig_rms = np.sqrt(np.mean(signal ** 2))
    noise_rms = np.sqrt(np.mean(noise_only ** 2))
    measured_snr = 20 * np.log10(sig_rms / max(noise_rms, 1e-10))
    assert 12.0 <= measured_snr <= 18.0, (
        f"Measured SNR {measured_snr:.1f} dB outside 15 ±3 dB"
    )


# ---------------------------------------------------------------------------
# 3. Mini-DataFrame row counts
# ---------------------------------------------------------------------------

def test_mini_dataframe_counts(augmentor: DataAugmentor,
                               tmp_path: Path) -> None:
    """5-row DF (3 DS_A + 2 DS_B) → correct original and augmented counts."""
    import pandas as pd

    wavs = []
    for i in range(5):
        p = _make_sine(tmp_path / f"utt{i:04d}.wav")
        wavs.append(str(p))

    rows = [
        {"utterance_id": f"A_{i}", "audio_path": f"a{i}.mp3",
         "transcript": "hola", "speaker_id": f"spkA{i}",
         "dataset": "DS_A", "l1_group": "X", "task_type": "oral",
         "location": "lab", "has_timestamps": True,
         "start_ms": 0, "end_ms": 3000, "duration_str": "",
         "utterance_index": i, "processed_path": wavs[i],
         "snr_db": 20.0, "processed_duration_sec": 3.0,
         "was_chunked": False, "rejected": False,
         "rejection_reason": None}
        for i in range(3)
    ] + [
        {"utterance_id": f"B_{i}", "audio_path": f"b{i}.mp3",
         "transcript": "hello", "speaker_id": f"spkB{i}",
         "dataset": "DS_B", "l1_group": "Y", "task_type": "oral",
         "location": "lab", "has_timestamps": True,
         "start_ms": 0, "end_ms": 3000, "duration_str": "",
         "utterance_index": i, "processed_path": wavs[3 + i],
         "snr_db": 20.0, "processed_duration_sec": 3.0,
         "was_chunked": False, "rejected": False,
         "rejection_reason": None}
        for i in range(2)
    ]
    df = pd.DataFrame(rows)

    out_dir = tmp_path / "augmented"
    result = augmentor.augment_dataset(df, out_dir, seed=42)

    # Original rows
    n_orig = int((~result.is_augmented).sum())
    assert n_orig == 5, f"Expected 5 original rows, got {n_orig}"

    # DS_A: multiplier=3 → 2 copies each → 3 × 2 = 6 augmented
    ds_a = result[result.dataset == "DS_A"]
    n_a_aug = int(ds_a.is_augmented.sum())
    assert n_a_aug == 6, f"DS_A: expected 6 augmented, got {n_a_aug}"

    # DS_B: multiplier=2 → 1 copy each → 2 × 1 = 2 augmented
    ds_b = result[result.dataset == "DS_B"]
    n_b_aug = int(ds_b.is_augmented.sum())
    assert n_b_aug == 2, f"DS_B: expected 2 augmented, got {n_b_aug}"

    # Total
    assert len(result) == 5 + 6 + 2, f"Total rows: {len(result)}"


# ---------------------------------------------------------------------------
# 4. Determinism – same seed → identical md5
# ---------------------------------------------------------------------------

def test_determinism_same_seed(tmp_path: Path) -> None:
    """Two runs with the same seed produce byte-identical output."""
    import pandas as pd

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "augmentation:\n"
        "  seed: 99\n"
        "  target_dbfs: -20.0\n"
        "  datasets:\n"
        "    DS_X:\n"
        "      multiplier: 2\n"
        "      strategies:\n"
        "        - tag: sp090_wn15\n"
        "          speed_factor: 0.90\n"
        "          noise_type: white\n"
        "          noise_snr_db: 15\n"
    )

    wav_in = _make_sine(tmp_path / "src.wav", duration=2.0)
    row = {
        "utterance_id": "X_0", "audio_path": "x.mp3",
        "transcript": "hola", "speaker_id": "spkX",
        "dataset": "DS_X", "l1_group": "Z", "task_type": "oral",
        "location": "lab", "has_timestamps": True,
        "start_ms": 0, "end_ms": 2000, "duration_str": "",
        "utterance_index": 0, "processed_path": str(wav_in),
        "snr_db": 20.0, "processed_duration_sec": 2.0,
        "was_chunked": False, "rejected": False,
        "rejection_reason": None,
    }
    df = pd.DataFrame([row])

    def _run(out_name: str) -> str:
        out_dir = tmp_path / out_name
        aug = DataAugmentor(config_path=str(cfg), seed=99)
        aug.augment_dataset(df, out_dir, seed=99)
        # Find the augmented wav
        wavs = list(out_dir.rglob("*.wav"))
        assert len(wavs) == 1, f"Expected 1 augmented wav, got {len(wavs)}"
        return hashlib.md5(wavs[0].read_bytes()).hexdigest()

    h1 = _run("run1")
    h2 = _run("run2")
    assert h1 == h2, f"MD5 mismatch: {h1} vs {h2}"
