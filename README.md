# AutoEIT — Fine-tuning Whisper on L2 Spanish EIT Data

AutoEIT is a pipeline for automatically scoring Elicited Imitation Tests (EIT) in L2 Spanish using a fine-tuned Whisper model. It covers three SLABANK corpora of learner speech, applies corpus-specific audio preprocessing and augmentation, and fine-tunes `openai/whisper-large-v2` with LoRA on Kaggle GPUs.

| Split | Utterances | Audio hours |
|-------|-----------|-------------|
| Train (incl. augmentation) | 56,090 | ~65.4 h |
| Dev | 4,033 | ~4.6 h |
| Test | 4,698 | ~5.3 h |

| Corpus | L1 groups | Utterances |
|--------|----------|-----------|
| SPLLOC1 | English | 15,760 |
| Nebrija-INMIGRA | Brazilian-Portuguese, Chinese, French, etc. | 11,721 |
| Nebrija-WOCAE | Chinese | 922 |

---

## Project Structure

```
autoeit/
├── configs/
│   └── config.yaml                # Hyperparameters and corpus-specific settings
├── data/
│   ├── manifest.csv               # One row per .cha session file
│   ├── transcripts/
│   │   ├── processed_manifest.csv # 28,403 cleaned utterances
│   │   ├── train_augmented.csv    # 56,090 training rows (original + augmented)
│   │   ├── dev.csv                # 4,033 dev rows
│   │   ├── test.csv               # 4,698 test rows
│   │   └── test_mini.csv          # Small subset for quick testing
│   ├── processed/                 # Preprocessed 16 kHz WAV files
│   ├── augmented/                 # Augmented WAV copies
│   └── mel_cache/                 # Pre-computed mel spectrograms (.npy)
├── models/
│   ├── adapters/                  # Saved LoRA adapter weights
│   └── checkpoints/               # Training checkpoints
├── results/
│   └── baseline_openai_whisper-large-v3.csv
├── scripts/
│   ├── run_baseline.py            # Whisper-large-v3 baseline evaluation
│   └── generate_test_mini.py      # Creates test_mini.csv for quick testing
├── src/
│   ├── data_loader.py             # CHAT parser, manifest loader, train/dev/test split
│   ├── preprocess.py              # 8-step audio preprocessor
│   ├── augment.py                 # Speed/pitch/noise/reverb augmentation
│   ├── train.py                   # MLX LoRA training loop (Apple Silicon)
│   ├── infer.py                   # Inference helper, text normalisation
│   └── evaluate.py                # WER / CER / 90% agreement evaluation
├── tests/
│   ├── test_preprocess.py
│   ├── test_augment.py
│   └── test_evaluate.py
├── whisper-eit-kaggle-v2.ipynb    # Kaggle finetuning notebook (main training script)
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.10+
- macOS with Apple Silicon for local preprocessing (MLX is Apple-only)
- FFmpeg: `brew install ffmpeg`
- Kaggle account with GPU runtime for finetuning

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Data Processing Pipeline

The data processing pipeline runs locally on your machine. It transforms raw SLABANK corpora (CHAT files + audio) into clean, augmented training data.

### Step 1 — Generate the Source Manifest

```bash
python data/generate_manifest.py
```

Scans raw corpus directories (`data/Nebrija-INMIGRA/`, `data/Nebrija-WOCAE/`, `data/SPLLOC1/`) and writes `data/manifest.csv` — one row per `.cha` session file with audio path, participant IDs, and metadata.

**Output:** `data/manifest.csv`

### Step 2 — Parse CHAT Files and Build the Utterance Manifest

```bash
python -c "
from src.data_loader import ManifestLoader
ml = ManifestLoader()
ml.load_corpus()
ml.split()
"
```

Reads every `.cha` file from the manifest, extracts PAR-tier utterances with timestamps, assigns L1 labels per corpus, and creates speaker-stratified splits (70/15/15, no speaker leakage).

CHAT cleaning includes stripping 20+ annotation patterns (`@s`/`@l` code-switch markers, `[*]`, `&`, `[/]`, etc.).

**Outputs:**
- `data/transcripts/processed_manifest.csv` — all 28,403 utterances
- `data/transcripts/train.csv`, `dev.csv`, `test.csv` — stratified splits

### Step 3 — Preprocess Audio

```bash
python -c "
from src.preprocess import AudioPreprocessor
ap = AudioPreprocessor()
ap.process_all()
"
```

Each audio file goes through an 8-step pipeline:

1. **Load** audio (MP3/WAV) with librosa
2. **Resample** to 16 kHz mono (kaiser_best)
3. **Denoise** (Wiener) + bandpass filter 80–7500 Hz
4. **VAD trim** leading/trailing silence
5. **RMS normalise** to -20 dBFS
6. **Peak clip** to ±0.99
7. **Quality gate** — reject if SNR < threshold or duration < 0.5 s
8. **Chunk** sessions > 30 s into 25 s segments with 2 s overlap

Corpus-specific denoising settings (from `configs/config.yaml`):

| Corpus | `denoise_strength` | `snr_threshold` |
|--------|------------------:|---------------:|
| Nebrija-INMIGRA | 0.80 | 5.0 dB |
| Nebrija-WOCAE | 0.75 | 4.0 dB |
| SPLLOC1 | 0.70 | 5.0 dB |

**Output:** `data/processed/` — 16 kHz mono 16-bit WAV files

### Step 4 — Augment Training Data

```bash
python -c "
from src.augment import DataAugmentor
da = DataAugmentor()
da.augment_all()
"
```

Augmentation strategies per corpus:

| Corpus | Tag | Transform |
|--------|-----|-----------|
| INMIGRA | `sp090_wn15` | speed x0.90 + white noise SNR 15 dB |
| INMIGRA | `sp110_pn18` | speed x1.10 + pink noise SNR 18 dB |
| INMIGRA | `sp095_revsm` | speed x0.95 + small reverb |
| WOCAE | `sp085_wn12` | speed x0.85 + white noise SNR 12 dB |
| WOCAE | `sp115_ps1_pn16` | speed x1.15 + pitch +1 st + pink noise SNR 16 dB |
| WOCAE | `sp105_revmd` | speed x1.05 + medium reverb |
| SPLLOC1 | `sp092_vj` | speed x0.92 + volume jitter |

INMIGRA L1 groups with < 3 sessions get a 6x low-resource multiplier.

**Output:** `data/augmented/` — augmented WAV copies; `train_augmented.csv` updated with `is_augmented` rows (~56K total).

### Step 5 — Pre-compute Mel Spectrograms (optional)

```bash
python -c "
from src.data_loader import MelCache
MelCache().precompute_all()
"
```

Caches mel spectrograms as `.npy` files in `data/mel_cache/` (shape `(3000, 80)`, time-first). The training loop transposes to Whisper's `(80, T)` convention automatically.

This step is **required before uploading to Kaggle** — the notebook reads from the mel cache for faster data loading.

---

## Fine-tuning on Kaggle

The finetuning runs on Kaggle using `whisper-eit-kaggle-v2.ipynb`. This notebook fine-tunes `openai/whisper-large-v2` with LoRA using HuggingFace Transformers + PEFT.

### Prepare the Data Upload

After running the full local pipeline (Steps 1–5), zip the data directory:

```bash
rm -rf data/mel_cache/
python -c "from src.data_loader import MelCache; MelCache().precompute_all()"
zip -r eit-data.zip data/ \
    --exclude "data/transcripts/backups/*" \
    --exclude "*.pyc" --exclude "__pycache__/*"
```

Upload `eit-data.zip` as a Kaggle Dataset named **`eit-data`**.

### Kaggle Setup

1. Go to Kaggle > New Notebook
2. Upload `whisper-eit-kaggle-v2.ipynb`
3. Add `eit-data` as an input dataset
4. Set **Accelerator: GPU T4 x2** (or P100/A100) and enable internet access

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `openai/whisper-large-v2` (1.55B params) |
| LoRA rank | 8 (~10.5M trainable params) |
| LoRA targets | `q_proj`, `v_proj`, `k_proj`, `out_proj` (encoder + decoder + cross-attn) |
| Batch size | 4 per device, grad_accum 8 (effective batch 32) |
| Learning rate | 5e-6 with cosine decay |
| Warmup | 5% of total steps |
| Epochs | 10 |
| Eval / save interval | 500 steps |
| Early stopping | 3 eval intervals patience |
| SpecAugment | 2 time masks (100 frames), 2 freq masks (20 bins) |
| Repetition penalty | 1.3 + no-repeat 4-gram |
| Eval samples | 1000 / 4033 dev samples |

### Run

Run all cells in order. The notebook handles:
1. Installing dependencies (`transformers`, `peft`, `accelerate`, etc.)
2. Loading data from the Kaggle dataset input
3. Building mel-cached datasets with in-flight SpecAugment
4. Loading whisper-large-v2 and applying LoRA adapters
5. Training with weighted sampling (35% INMIGRA, 20% WOCAE, 45% SPLLOC1)
6. Evaluating on dev set (WER/CER) at each checkpoint
7. Fusing LoRA into base weights and exporting a standalone model
8. Zipping the fused model to `/kaggle/working/whisper_eit_lora.zip`

**Expected training time:** ~8-10 h on T4 x2, ~4-5 h on A100.

### Download Results

After training completes, download `whisper_eit_lora.zip` from the Kaggle Output panel. This contains the fused model (base + LoRA weights merged).

---

## Local Training (Apple Silicon)

For local training/debugging on Apple Silicon, use `src/train.py` with MLX:

```bash
python src/train.py \
  --config configs/config.yaml \
  --train-csv data/transcripts/train_augmented.csv \
  --dev-csv data/transcripts/dev.csv \
  --output-dir models/checkpoints
```

This uses `whisper-small` with memory optimizations for 8 GB unified memory (batch_size=2, gradient accumulation, Metal cache clearing).

---

## Evaluation

### Run the Whisper-large-v3 Baseline

```bash
python scripts/run_baseline.py \
  --model openai/whisper-large-v3 \
  --test-csv data/transcripts/test.csv \
  --output results/baseline_v3.csv
```

### Metrics

The evaluation module (`src/evaluate.py`) computes:
- **WER** — Word Error Rate
- **CER** — Character Error Rate
- **90% Agreement** — fraction of utterances with per-utterance accuracy >= 90% (primary metric)
- Error breakdown: substitutions, insertions, deletions
- Per-group metrics by dataset, L1, and task type

---

## Testing

```bash
python -m pytest tests/ -v
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'mlx'` | MLX is Apple Silicon only: `pip install mlx mlx-whisper` |
| Out of memory during local training | Reduce `batch_size` to 1 in `configs/config.yaml` |
| `librosa.load` is slow | Install `soundfile` for 10-50x faster WAV loading |
| Kaggle OOM on T4 | Reduce `batch_size` to 2 in the notebook's CFG dict |

---

## Citation

If you use this pipeline or the cleaned corpus splits, please cite the SLABANK corpora:
- SPLLOC1: [Dominguez et al., 2013]
- Nebrija-WOCAE / Nebrija-INMIGRA: [Perlmutter et al., 2018-2022]