"""Unit tests for src.preprocess – AudioPreprocessor pipeline.

Tests
-----
1. Synthetic 5 s stereo sine at 44100 Hz + noise → 16 kHz mono, RMS ≈ −20 dBFS ±3 dB.
2. Utterance extraction with start_ms=1000, end_ms=4000 → ≈ 3 s ±0.3 s.
3. 35 s file → chunked into ≥ 2 files.
4. Near-silent file → rejected.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.preprocess import AudioPreprocessor, _rms_db

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def preprocessor(tmp_path: Path) -> AudioPreprocessor:
    """Create a preprocessor with a minimal in-memory config."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "audio:\n  sample_rate: 16000\n"
        "datasets:\n"
        "  test_ds:\n"
        "    denoise_strength: 0.70\n"
        "    snr_threshold: 0.0\n"
    )
    return AudioPreprocessor(config_path=str(cfg))


def _make_sine_wav(path: Path, duration: float, sr: int = 44100,
                   stereo: bool = False, freq: float = 440.0,
                   noise_level: float = 0.02) -> Path:
    """Generate a synthetic WAV file with a sine tone + optional noise."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False).astype(np.float32)
    tone = 0.5 * np.sin(2 * np.pi * freq * t)
    noise = noise_level * np.random.default_rng(42).standard_normal(len(t)).astype(np.float32)
    audio = tone + noise
    if stereo:
        audio = np.stack([audio, audio * 0.8], axis=0)  # (2, N)
        # soundfile expects (N, 2)
        audio = audio.T
    sf.write(str(path), audio, sr)
    return path


# ---------------------------------------------------------------------------
# Test 1: basic pipeline — stereo 44100 → 16 kHz mono, RMS ≈ −20 dBFS
# ---------------------------------------------------------------------------

def test_basic_pipeline_stereo_to_mono(preprocessor: AudioPreprocessor, tmp_path: Path) -> None:
    """5 s stereo 44.1 kHz sine → 16 kHz mono WAV with RMS ≈ −20 dBFS ±3."""
    inp = _make_sine_wav(tmp_path / "input.wav", duration=5.0, sr=44100, stereo=True)
    out = tmp_path / "output.wav"

    result = preprocessor.process_file(
        input_path=inp, output_path=out, dataset="test_ds",
    )

    assert not result["rejected"], f"Rejected: {result['rejection_reason']}"
    assert Path(result["output_path"]).exists()

    # Read back and verify
    audio, sr = sf.read(str(result["output_path"]))
    assert sr == 16000, f"Expected 16 kHz, got {sr}"
    assert audio.ndim == 1, "Expected mono"

    rms = _rms_db(audio)
    assert -23.0 <= rms <= -17.0, f"RMS {rms:.1f} dBFS outside ±3 dB of −20"


# ---------------------------------------------------------------------------
# Test 2: timestamp extraction — output duration ≈ 3 s ±0.3 s
# ---------------------------------------------------------------------------

def test_utterance_extraction(preprocessor: AudioPreprocessor, tmp_path: Path) -> None:
    """Extract start_ms=1000 → end_ms=4000; output ≈ 3 s."""
    inp = _make_sine_wav(tmp_path / "input.wav", duration=10.0, sr=44100, stereo=False)
    out = tmp_path / "utt.wav"

    result = preprocessor.process_file(
        input_path=inp, output_path=out, dataset="test_ds",
        start_ms=1000, end_ms=4000,
    )

    assert not result["rejected"], f"Rejected: {result['rejection_reason']}"
    audio, sr = sf.read(str(result["output_path"]))
    dur = len(audio) / sr
    # 3 s core + up to 0.3 s padding → expect ~3.0–3.3 s
    assert 2.7 <= dur <= 3.6, f"Duration {dur:.2f}s outside expected range"


# ---------------------------------------------------------------------------
# Test 3: chunking — 35 s file → ≥ 2 chunks
# ---------------------------------------------------------------------------

def test_chunking_long_file(preprocessor: AudioPreprocessor, tmp_path: Path) -> None:
    """A 35 s file exceeds 30 s limit and must be chunked into ≥ 2 pieces."""
    inp = _make_sine_wav(tmp_path / "input.wav", duration=35.0, sr=16000, stereo=False)
    out = tmp_path / "long.wav"

    result = preprocessor.process_file(
        input_path=inp, output_path=out, dataset="test_ds",
    )

    assert not result["rejected"], f"Rejected: {result['rejection_reason']}"
    assert result["was_chunked"] is True
    assert result["n_chunks"] >= 2, f"Expected ≥2 chunks, got {result['n_chunks']}"
    for cp in result["chunk_paths"]:
        assert Path(cp).exists(), f"Chunk not found: {cp}"


# ---------------------------------------------------------------------------
# Test 4: near-silent file → rejected
# ---------------------------------------------------------------------------

def test_silent_file_rejected(preprocessor: AudioPreprocessor, tmp_path: Path) -> None:
    """A near-silent file should be rejected for low SNR."""
    # Create a nearly silent file (very low amplitude noise only)
    sr = 16000
    dur = 3.0
    n = int(sr * dur)
    silence = np.random.default_rng(0).standard_normal(n).astype(np.float32) * 1e-5
    inp = tmp_path / "silent.wav"
    sf.write(str(inp), silence, sr)
    out = tmp_path / "out_silent.wav"

    # Use a high SNR threshold to ensure rejection
    preprocessor.dataset_params["test_ds"]["snr_threshold"] = 30.0

    result = preprocessor.process_file(
        input_path=inp, output_path=out, dataset="test_ds",
    )

    assert result["rejected"] is True, "Near-silent file should be rejected"
    assert result["rejection_reason"] is not None
