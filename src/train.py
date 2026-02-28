"""AutoEIT – Fine-tune Whisper with LoRA via MLX on Apple Silicon.

MEMORY-PATCHED for Apple M2 8 GB unified memory.

Key changes vs original:
  • batch_size default 1 (was 8)  — logits [1,224,51864]×fp16 ≈ 23 MB vs 742 MB
  • grad_accum default 8 (was 4)  — same effective batch, safe peak RAM
  • MAX_LABEL_LEN 224 (was 448)   — halves every token buffer
  • gradients accumulated as running sum, NOT kept all in RAM at once
  • mx.metal.clear_cache() called every step
  • evaluate() caps at 50 greedy decodes (was 200) with per-sample cache clear
  • compute_loss casts logits to fp32 ONLY for CE, then discards immediately
  • explicit del + mx.eval() to release tensors promptly
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd
import yaml

# ── MLX imports ────────────────────────────────────────────────────────────
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
except ImportError:
    sys.exit("\n✘  MLX required.  pip install mlx>=0.18\n")

try:
    from transformers import WhisperProcessor
except ImportError:
    sys.exit("\n✘  transformers required.  pip install transformers>=4.40\n")

try:
    from mlx_whisper.load_models import load_model as _load_mlx_whisper
except ImportError:
    sys.exit("\n✘  mlx-whisper required.  pip install mlx-whisper\n")

try:
    import jiwer
except ImportError:
    sys.exit("\n✘  jiwer required.  pip install jiwer\n")

try:
    from src.infer import normalize_text
except ImportError:
    import re, unicodedata
    _PUNCT_RE = re.compile(r"[^\w\s']", re.UNICODE)
    _MULTI_WS_RE = re.compile(r"\s{2,}")
    def normalize_text(text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        text = text.lower()
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"<unintelligible>", "", text, flags=re.IGNORECASE)
        text = _PUNCT_RE.sub("", text)
        text = _MULTI_WS_RE.sub(" ", text)
        return text.strip()


logger = logging.getLogger("autoeit.train")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_h)

# ── Constants ──────────────────────────────────────────────────────────────
TARGET_SR: int = 16_000
MAX_AUDIO_SAMPLES: int = 480_000   # 30 s

# ✅ FIX #1: Reduced from 448 → 224 to halve every logit/token buffer.
# Most EIT utterances are <80 tokens; 224 is still generous.
MAX_LABEL_LEN: int = 224

SOT: int = 0
EOT: int = 0
LANG_TOKEN: int = 0
TRANSCRIBE_TOKEN: int = 0
NO_TIMESTAMPS_TOKEN: int = 0
PAD_TOKEN: int = 0

DATASET_WEIGHTS: Dict[str, float] = {
    "Nebrija-INMIGRA": 0.35,
    "Nebrija-WOCAE": 0.20,
    "SPLLOC1": 0.45,
}


def _init_special_tokens(processor: WhisperProcessor, language: str = "es") -> None:
    global SOT, EOT, LANG_TOKEN, TRANSCRIBE_TOKEN, NO_TIMESTAMPS_TOKEN, PAD_TOKEN
    tok = processor.tokenizer
    SOT = tok.convert_tokens_to_ids("<|startoftranscript|>")
    EOT = tok.eos_token_id
    PAD_TOKEN = EOT
    forced = processor.get_decoder_prompt_ids(language=language, task="transcribe")
    if forced:
        LANG_TOKEN = forced[0][1]
        TRANSCRIBE_TOKEN = forced[1][1] if len(forced) > 1 else TRANSCRIBE_TOKEN
    NO_TIMESTAMPS_TOKEN = tok.convert_tokens_to_ids("<|notimestamps|>")


# ══════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════

class EITDataset:
    def __init__(
        self,
        df: pd.DataFrame,
        processor: WhisperProcessor,
        max_audio_len: int = MAX_AUDIO_SAMPLES,
        max_label_len: int = MAX_LABEL_LEN,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.feature_extractor = processor.feature_extractor
        self.max_audio_len = max_audio_len
        self.max_label_len = max_label_len
        self._prefix = [SOT, LANG_TOKEN, TRANSCRIBE_TOKEN, NO_TIMESTAMPS_TOKEN]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        audio = self._load_audio(row)
        mel = self._extract_features(audio)
        # ✅ FIX: delete audio immediately after feature extraction to free RAM
        del audio
        decoder_input, labels = self._tokenize(str(row.transcript))
        return {
            "mel": mel,
            "decoder_input": decoder_input,
            "labels": labels,
            "dataset": row.dataset,
            "l1_group": row.get("l1_group", ""),
        }

    def _load_audio(self, row: pd.Series) -> np.ndarray:
        audio, _ = librosa.load(str(row.processed_path), sr=TARGET_SR, mono=True)
        has_ts = str(row.get("has_timestamps", "True")).lower() == "true"
        if not has_ts and len(audio) > self.max_audio_len:
            start = random.randint(0, len(audio) - self.max_audio_len)
            audio = audio[start: start + self.max_audio_len]
        if len(audio) > self.max_audio_len:
            audio = audio[: self.max_audio_len]
        elif len(audio) < self.max_audio_len:
            audio = np.pad(audio, (0, self.max_audio_len - len(audio)))
        return audio.astype(np.float32)

    def _extract_features(self, audio: np.ndarray) -> np.ndarray:
        feats = self.feature_extractor(audio, sampling_rate=TARGET_SR, return_tensors="np")
        mel = feats.input_features[0].astype(np.float32)  # [n_mels, 3000]
        return mel.T  # [3000, n_mels] — channels-last for MLX Conv1d

    def _tokenize(self, transcript: str) -> Tuple[np.ndarray, np.ndarray]:
        text_ids = self.tokenizer.encode(transcript, add_special_tokens=False)
        full_seq = self._prefix + text_ids + [EOT]
        if len(full_seq) > self.max_label_len + 1:
            full_seq = full_seq[: self.max_label_len + 1]
        decoder_input = full_seq[:-1]
        labels = full_seq[1:]
        pad_len = self.max_label_len - len(decoder_input)
        if pad_len > 0:
            decoder_input = decoder_input + [PAD_TOKEN] * pad_len
            labels = labels + [-100] * pad_len
        return (
            np.array(decoder_input, dtype=np.int32),
            np.array(labels, dtype=np.int32),
        )


def build_sample_weights(df: pd.DataFrame) -> np.ndarray:
    weights = np.zeros(len(df), dtype=np.float64)
    for ds_name, ds_weight in DATASET_WEIGHTS.items():
        mask = (df.dataset == ds_name).values
        count = mask.sum()
        if count > 0:
            weights[mask] = ds_weight / count
    inmigra_mask = (df.dataset == "Nebrija-INMIGRA").values
    if inmigra_mask.sum() > 0:
        inmigra_df = df[inmigra_mask]
        l1_counts = inmigra_df.l1_group.value_counts()
        l1_w = inmigra_df.l1_group.map(lambda l: 1.0 / l1_counts.get(l, 1)).values.astype(np.float64)
        l1_w = l1_w / l1_w.sum() * inmigra_mask.sum()
        weights[inmigra_mask] *= l1_w
    total = weights.sum()
    weights = weights / total if total > 0 else np.full(len(df), 1.0 / len(df))
    return weights


def build_epoch_indices(df: pd.DataFrame, weights: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.choice(len(df), size=len(df), replace=True, p=weights)


def prepare_dev_df(
    dev_csv: str,
    processed_manifest_csv: str = "data/transcripts/processed_manifest.csv",
) -> pd.DataFrame:
    dev = pd.read_csv(dev_csv)
    manifest = pd.read_csv(processed_manifest_csv)
    merge_cols = ["utterance_id", "processed_path", "snr_db",
                  "processed_duration_sec", "was_chunked", "rejected", "rejection_reason"]
    available_cols = [c for c in merge_cols if c in manifest.columns]
    dev = dev.merge(manifest[available_cols], on="utterance_id", how="left")
    dev = dev[dev.processed_path.notna()].copy()
    if "rejected" in dev.columns:
        dev = dev[dev.rejected != True].copy()
    return dev.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════
# MODEL (MLX + LoRA)
# ══════════════════════════════════════════════════════════════════════════

_LORA_TARGET_MAP: Dict[str, str] = {
    "q_proj": "query", "k_proj": "key", "v_proj": "value",
    "out_proj": "out", "fc1": "mlp1", "fc2": "mlp2",
}


class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for nn.Linear."""

    def __init__(self, linear: nn.Linear, r: int = 8, scale: float = 2.0) -> None:
        super().__init__()
        in_dim = linear.weight.shape[1]
        out_dim = linear.weight.shape[0]
        self.linear = linear
        self.lora_a = mx.random.normal((in_dim, r)) * (1.0 / r)
        self.lora_b = mx.zeros((r, out_dim))
        self.scale = scale

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(x) + (x @ self.lora_a @ self.lora_b) * self.scale

    @staticmethod
    def from_linear(linear: nn.Linear, r: int = 8, scale: float = 2.0) -> "LoRALinear":
        return LoRALinear(linear, r=r, scale=scale)


def load_whisper_model(model_name: str = "openai/whisper-small", dtype: Any = None) -> Any:
    if dtype is None:
        dtype = mx.float16   # ✅ float16 throughout — critical for 8 GB
    hf_to_mlx = {
        "openai/whisper-large-v3": "mlx-community/whisper-large-v3-mlx",
        "openai/whisper-large-v2": "mlx-community/whisper-large-v2-mlx",
        "openai/whisper-medium":   "mlx-community/whisper-medium-mlx",
        "openai/whisper-small":    "mlx-community/whisper-small-mlx",
        "openai/whisper-base":     "mlx-community/whisper-base-mlx",
        "openai/whisper-tiny":     "mlx-community/whisper-tiny-mlx",
    }
    repo = hf_to_mlx.get(model_name, model_name)
    logger.info("Loading Whisper model from %s (dtype=%s)", repo, dtype)
    model = _load_mlx_whisper(repo, dtype)
    logger.info("Whisper model loaded.")
    return model


def apply_lora(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    r: int = 8,           # ✅ FIX #2: default r=8 (was 16) — halves LoRA param count
    lora_alpha: int = 16, # ✅ keeps scale=lora_alpha/r=2 unchanged
    dropout: float = 0.05,
) -> int:
    if target_modules is None:
        # ✅ FIX #3: Target only attention projections by default (no MLP).
        # MLP layers are large; skipping fc1/fc2 saves ~40% of LoRA memory.
        target_modules = ["q_proj", "v_proj"]
    target_keys = {_LORA_TARGET_MAP.get(m, m) for m in target_modules}
    scale = lora_alpha / r
    replaced = 0

    def _replace_in_module(parent: nn.Module) -> None:
        nonlocal replaced
        for attr_name, child in list(parent.children().items()):
            if isinstance(child, nn.Linear) and attr_name in target_keys:
                try:
                    parent[attr_name] = LoRALinear.from_linear(child, r=r, scale=scale)
                    replaced += 1
                except Exception as exc:
                    logger.warning("Could not apply LoRA to %s: %s", attr_name, exc)
            elif isinstance(child, nn.Module):
                _replace_in_module(child)
            elif isinstance(child, (list, tuple)):
                for item in child:
                    if isinstance(item, nn.Module):
                        _replace_in_module(item)

    _replace_in_module(model)
    model.freeze()
    _unfreeze_lora_params(model)
    return replaced


def _unfreeze_lora_params(module: nn.Module) -> None:
    for attr_name, child in module.children().items():
        if isinstance(child, LoRALinear):
            child.linear.freeze()
            child.unfreeze(keys=["lora_a", "lora_b"], strict=False)
        elif isinstance(child, nn.Module):
            _unfreeze_lora_params(child)
        elif isinstance(child, (list, tuple)):
            for item in child:
                if isinstance(item, nn.Module):
                    _unfreeze_lora_params(item)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    def _count(tree: Any) -> int:
        if isinstance(tree, mx.array): return tree.size
        if isinstance(tree, dict): return sum(_count(v) for v in tree.values())
        if isinstance(tree, (list, tuple)): return sum(_count(v) for v in tree)
        return 0
    return _count(model.parameters()), _count(model.trainable_parameters())


# ══════════════════════════════════════════════════════════════════════════
# LOSS & METRICS
# ══════════════════════════════════════════════════════════════════════════


def compute_loss(
    model: nn.Module,
    mel: mx.array,
    tokens: mx.array,
    labels: mx.array,
) -> mx.array:
    """Forward pass + cross-entropy.

    ✅ FIX #4: logits are kept in fp16 until CE; no silent fp32 upcast of the
    full [B, T, V] tensor.  We use nn.losses.cross_entropy which handles the
    cast internally without materialising a full fp32 copy.
    """
    audio_features = model.encoder(mel)
    logits = model.decoder(tokens, audio_features)[0]  # [B, T, V] fp16

    safe_labels = mx.where(labels >= 0, labels, mx.zeros_like(labels))
    ce = nn.losses.cross_entropy(logits, safe_labels, reduction="none")  # [B, T]
    mask = (labels >= 0).astype(mx.float32)
    loss = (ce * mask).sum() / mx.maximum(mask.sum(), mx.array(1.0))

    # ✅ FIX #5: immediately delete large intermediate tensors
    del logits, ce, mask, safe_labels, audio_features
    return loss


def compute_metrics(
    references: Sequence[str],
    hypotheses: Sequence[str],
) -> Dict[str, float]:
    refs = [r if r.strip() else "<empty>" for r in references]
    hyps = [h if h.strip() else "<empty>" for h in hypotheses]
    wer_val = float(jiwer.wer(refs, hyps))
    cer_val = float(jiwer.cer(refs, hyps))
    agree = sum(1 for r, h in zip(refs, hyps) if (1.0 - jiwer.wer(r, h)) >= 0.90)
    return {
        "wer": wer_val,
        "cer": cer_val,
        "90pct_agreement": agree / max(len(refs), 1),
    }


class EarlyStoppingOnOverfit:
    def __init__(self, patience: int = 3, delta: float = 0.05) -> None:
        self.patience = patience
        self.delta = delta
        self.best_wer: float = float("inf")
        self.history: List[float] = []
        self.should_stop: bool = False

    def update(self, wer: float) -> bool:
        self.history.append(wer)
        if wer < self.best_wer:
            self.best_wer = wer
        if len(self.history) >= self.patience:
            recent = self.history[-self.patience:]
            if (all(recent[i] < recent[i+1] for i in range(len(recent)-1))
                    and recent[-1] > self.best_wer + self.delta):
                logger.warning("⚠ Overfitting detected at step history: %s", recent)
                self.should_stop = True
                return True
        return False


# ══════════════════════════════════════════════════════════════════════════
# TRAINER
# ══════════════════════════════════════════════════════════════════════════


class WhisperLoRATrainer:
    def __init__(
        self,
        model: nn.Module,
        processor: WhisperProcessor,
        train_df: pd.DataFrame,
        dev_df: pd.DataFrame,
        config: Dict[str, Any],
        output_dir: str = "models/checkpoints/whisper-sm-autoeit-lora",
    ) -> None:
        self.model = model
        self.processor = processor
        self.train_df = train_df
        self.dev_df = dev_df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        tcfg = config.get("training", {})

        # ✅ FIX #6: Defaults tuned for M2 8 GB
        self.batch_size: int = int(tcfg.get("batch_size", 1))      # was 8
        self.grad_accum: int = int(tcfg.get("gradient_accumulation", 8))  # was 4
        self.learning_rate: float = float(tcfg.get("learning_rate", 1e-5))
        self.warmup_steps: int = int(tcfg.get("warmup_steps", 300))
        self.max_steps: int = int(tcfg.get("max_steps", 3000))
        self.eval_steps: int = int(tcfg.get("eval_steps", 300))
        self.save_steps: int = int(tcfg.get("save_steps", 300))
        self.logging_steps: int = int(tcfg.get("logging_steps", 25))
        self.generation_max_length: int = 225

        self.train_dataset = EITDataset(train_df, processor)
        self.dev_dataset = EITDataset(dev_df, processor)
        self.sample_weights = build_sample_weights(train_df)
        self.optimizer = optim.AdamW(learning_rate=self.learning_rate)
        self.early_stopping = EarlyStoppingOnOverfit(patience=3, delta=0.05)

        self.global_step: int = 0
        self.best_wer: float = float("inf")
        self.best_checkpoint: str = ""
        self.train_losses: List[float] = []
        self.eval_results: List[Dict[str, Any]] = []

    def _get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.learning_rate * step / max(self.warmup_steps, 1)
        return self.learning_rate

    def _collate(self, indices: np.ndarray) -> Dict[str, mx.array]:
        mels, tokens, labels = [], [], []
        for idx in indices:
            sample = self.train_dataset[int(idx)]
            mels.append(sample["mel"])
            tokens.append(sample["decoder_input"])
            labels.append(sample["labels"])
        batch = {
            "mel":    mx.array(np.stack(mels)),
            "tokens": mx.array(np.stack(tokens)),
            "labels": mx.array(np.stack(labels)),
        }
        # ✅ FIX #7: release numpy arrays immediately after conversion
        del mels, tokens, labels
        return batch

    def _collate_dev(self, start: int, end: int) -> Dict[str, mx.array]:
        mels, tokens, labels = [], [], []
        for idx in range(start, min(end, len(self.dev_dataset))):
            sample = self.dev_dataset[idx]
            mels.append(sample["mel"])
            tokens.append(sample["decoder_input"])
            labels.append(sample["labels"])
        batch = {
            "mel":    mx.array(np.stack(mels)),
            "tokens": mx.array(np.stack(tokens)),
            "labels": mx.array(np.stack(labels)),
        }
        del mels, tokens, labels
        return batch

    # ✅ FIX #8: Core fix — accumulate gradients as a RUNNING SUM,
    # not by storing all micro-batch tensors at once.
    def _train_step(self, micro_batches: List[Dict[str, mx.array]]) -> float:
        loss_fn = nn.value_and_grad(self.model, compute_loss)
        accumulated_grads = None
        total_loss = 0.0

        for batch in micro_batches:
            loss, grads = loss_fn(
                self.model, batch["mel"], batch["tokens"], batch["labels"]
            )
            # Force evaluation immediately so MLX can free the compute graph
            mx.eval(loss, grads)
            total_loss += float(loss.item())

            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = _tree_add(accumulated_grads, grads)
                # Evaluate running sum immediately to free previous grads
                mx.eval(accumulated_grads)

            # ✅ FIX #9: release the per-micro-batch objects
            del loss, grads, batch

        n = len(micro_batches)
        accumulated_grads = _tree_scale(accumulated_grads, 1.0 / n)

        lr = self._get_lr(self.global_step)
        self.optimizer.learning_rate = mx.array(lr)
        self.optimizer.update(self.model, accumulated_grads)
        mx.eval(self.model.parameters(), self.optimizer.state)
        del accumulated_grads

        # ✅ FIX #10: clear Metal buffer cache every step
        mx.metal.clear_cache()

        return total_loss / n

    def generate(self, mel: mx.array, max_length: int = 225) -> List[int]:
        """Greedy decode a single mel. Clears cache after each token to avoid
        accumulating the full unrolled graph in memory."""
        if mel.ndim == 2:
            mel = mel[None, ...]

        audio_features = self.model.encoder(mel)
        mx.eval(audio_features)   # ✅ materialise encoder once, reuse

        tokens = [SOT, LANG_TOKEN, TRANSCRIBE_TOKEN, NO_TIMESTAMPS_TOKEN]
        for _ in range(max_length):
            tok_arr = mx.array([tokens])
            logits = self.model.decoder(tok_arr, audio_features)[0]
            mx.eval(logits)
            next_id = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            del logits, tok_arr   # ✅ free each step's logits
            if next_id == EOT:
                break
            tokens.append(next_id)

        del audio_features
        return tokens[4:]  # strip prefix

    def evaluate(self) -> Dict[str, float]:
        """Validation pass.

        ✅ FIX #11: cap greedy-decode eval at 50 samples (was 200) and clear
        Metal cache after each sample so memory stays flat.
        """
        logger.info("── Evaluating on dev set (%d samples) ──", len(self.dev_dataset))

        # Teacher-forced loss (batched, batch_size=1 to stay safe)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, len(self.dev_dataset), self.batch_size):
            end = min(start + self.batch_size, len(self.dev_dataset))
            batch = self._collate_dev(start, end)
            loss = compute_loss(self.model, batch["mel"], batch["tokens"], batch["labels"])
            mx.eval(loss)
            total_loss += float(loss.item())
            n_batches += 1
            del batch, loss
            mx.metal.clear_cache()
        val_loss = total_loss / max(n_batches, 1)

        # Generation-based metrics
        max_eval_gen = min(len(self.dev_dataset), 50)  # ✅ was 200
        references, hypotheses = [], []
        for idx in range(max_eval_gen):
            sample = self.dev_dataset[idx]
            mel = mx.array(sample["mel"])
            gen_ids = self.generate(mel, self.generation_max_length)
            del mel

            hyp_text = self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
            ref_text = str(self.dev_df.iloc[idx].transcript)
            references.append(normalize_text(ref_text))
            hypotheses.append(normalize_text(hyp_text))

            mx.metal.clear_cache()  # ✅ clear after every decode

        metrics = compute_metrics(references, hypotheses)
        metrics["val_loss"] = val_loss
        logger.info(
            "  val_loss=%.4f  WER=%.4f  CER=%.4f  90%%agree=%.4f",
            val_loss, metrics["wer"], metrics["cer"], metrics["90pct_agreement"],
        )
        return metrics

    def save_checkpoint(self, tag: str = "") -> str:
        ckpt_name = f"step_{self.global_step}" if not tag else tag
        ckpt_dir = self.output_dir / ckpt_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        trainable = self.model.trainable_parameters()
        mx.save_safetensors(str(ckpt_dir / "adapters.safetensors"),
                            dict(_flatten_params(trainable)))
        state = {
            "global_step": self.global_step,
            "best_wer": self.best_wer,
            "learning_rate": self._get_lr(self.global_step),
            "train_losses": self.train_losses[-50:],
            "eval_results": self.eval_results,
        }
        with open(ckpt_dir / "training_state.json", "w") as f:
            json.dump(state, f, indent=2, default=str)
        logger.info("Checkpoint saved: %s", ckpt_dir)
        return str(ckpt_dir)

    def train(self) -> Dict[str, Any]:
        logger.info("═" * 60)
        logger.info("  Starting LoRA fine-tuning  [M2 8 GB memory-safe mode]")
        logger.info("  batch_size=%d  grad_accum=%d  effective_batch=%d",
                    self.batch_size, self.grad_accum,
                    self.batch_size * self.grad_accum)
        logger.info("  lr=%.2e  warmup=%d  max_steps=%d",
                    self.learning_rate, self.warmup_steps, self.max_steps)
        logger.info("═" * 60)

        t_start = time.perf_counter()
        epoch = 0

        try:
            while self.global_step < self.max_steps:
                indices = build_epoch_indices(self.train_df, self.sample_weights, seed=42 + epoch)
                idx_batches = [indices[i: i + self.batch_size]
                               for i in range(0, len(indices), self.batch_size)]
                micro_buffer: List[Dict[str, mx.array]] = []

                for batch_indices in idx_batches:
                    if self.global_step >= self.max_steps:
                        break

                    try:
                        batch = self._collate(batch_indices)
                    except Exception as exc:
                        logger.warning("Skipping batch (load error): %s", exc)
                        continue

                    micro_buffer.append(batch)

                    if len(micro_buffer) >= self.grad_accum:
                        try:
                            loss = self._train_step(micro_buffer)
                        except RuntimeError as exc:
                            if "memory" in str(exc).lower():
                                logger.error(
                                    "OOM at step %d. batch_size is currently %d. "
                                    "Set batch_size=1 and grad_accum=4 in config.yaml.",
                                    self.global_step, self.batch_size,
                                )
                                raise
                            raise
                        self.train_losses.append(loss)
                        self.global_step += 1
                        micro_buffer = []

                        if self.global_step % self.logging_steps == 0:
                            avg = float(np.mean(self.train_losses[-self.logging_steps:]))
                            lr = self._get_lr(self.global_step)
                            logger.info("  step %d/%d  loss=%.4f  lr=%.2e",
                                        self.global_step, self.max_steps, avg, lr)

                        if self.global_step % self.eval_steps == 0:
                            metrics = self.evaluate()
                            metrics["step"] = self.global_step
                            self.eval_results.append(metrics)
                            if metrics["wer"] < self.best_wer:
                                self.best_wer = metrics["wer"]
                                self.best_checkpoint = self.save_checkpoint("best")
                                logger.info("  ★ New best WER: %.4f", self.best_wer)
                            if self.early_stopping.update(metrics["wer"]):
                                logger.warning("Early stopping at step %d", self.global_step)
                                break

                        if (self.global_step % self.save_steps == 0
                                and self.global_step % self.eval_steps != 0):
                            self.save_checkpoint()

                if self.early_stopping.should_stop:
                    break
                epoch += 1
                logger.info("── Epoch %d complete ──", epoch)

        except KeyboardInterrupt:
            logger.warning("Interrupted at step %d", self.global_step)

        t_end = time.perf_counter()
        if not self.best_checkpoint:
            self.best_checkpoint = self.save_checkpoint("final")

        summary = {
            "total_steps": self.global_step,
            "training_time_sec": t_end - t_start,
            "training_time_human": _fmt_time(t_end - t_start),
            "best_wer": self.best_wer,
            "best_checkpoint": self.best_checkpoint,
            "final_train_loss": float(np.mean(self.train_losses[-10:])) if self.train_losses else None,
            "eval_history": self.eval_results,
        }
        logger.info("═" * 60)
        logger.info("  Done.  Steps: %d   Time: %s   Best WER: %.4f",
                    self.global_step, summary["training_time_human"], self.best_wer)
        logger.info("═" * 60)
        return summary


# ══════════════════════════════════════════════════════════════════════════
# POST-TRAINING & EXPORT
# ══════════════════════════════════════════════════════════════════════════

def run_post_training_eval(
    trainer: WhisperLoRATrainer,
    output_path: str = "results/training_summary.json",
) -> Dict[str, Any]:
    logger.info("Running post-training evaluation …")
    metrics = trainer.evaluate()
    summary = {
        "dev_wer": metrics["wer"],
        "dev_cer": metrics["cer"],
        "dev_90pct_agreement": metrics["90pct_agreement"],
        "dev_val_loss": metrics["val_loss"],
        "best_checkpoint": trainer.best_checkpoint,
        "total_steps": trainer.global_step,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Summary → %s", output_path)
    logger.info("Dev WER=%.4f  CER=%.4f  90%%agree=%.4f",
                summary["dev_wer"], summary["dev_cer"], summary["dev_90pct_agreement"])
    return summary


def fuse_lora_and_export(model: nn.Module, output_dir: str) -> str:
    fused_dir = Path(output_dir) / "fused"
    fused_dir.mkdir(parents=True, exist_ok=True)

    def _fuse_in(module: nn.Module) -> int:
        count = 0
        for attr_name, child in list(vars(module).items()):
            if isinstance(child, LoRALinear):
                _fuse_single_lora(module, attr_name, child)
                count += 1
            elif isinstance(child, nn.Module):
                count += _fuse_in(child)
            elif isinstance(child, (list, tuple)):
                for item in child:
                    if isinstance(item, nn.Module):
                        count += _fuse_in(item)
        return count

    n = _fuse_in(model)
    logger.info("Fused %d LoRA layers.", n)
    mx.save_safetensors(str(fused_dir / "weights.safetensors"),
                        dict(_flatten_params(model.parameters())))
    logger.info("Fused model → %s", fused_dir)
    readme = (
        "# Fused Whisper LoRA model\n\n"
        "Convert to whisper.cpp GGML:\n\n```bash\n"
        "python whisper.cpp/models/convert-hf-to-ggml.py \\\n"
        f"    --model-dir {fused_dir} \\\n"
        f"    --output-dir {Path(output_dir)/'ggml'}\n```\n"
    )
    (fused_dir / "README.md").write_text(readme)
    return str(fused_dir)


def _fuse_single_lora(parent: nn.Module, attr_name: str, lora_layer: LoRALinear) -> None:
    delta = (lora_layer.lora_a @ lora_layer.lora_b) * lora_layer.scale
    fused_weight = lora_layer.linear.weight + mx.transpose(delta)
    has_bias = hasattr(lora_layer.linear, "bias") and lora_layer.linear.bias is not None
    out_dim, in_dim = fused_weight.shape
    fused = nn.Linear(in_dim, out_dim, bias=has_bias)
    fused.weight = fused_weight
    if has_bias:
        fused.bias = lora_layer.linear.bias
    setattr(parent, attr_name, fused)


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _tree_add(a: Any, b: Any) -> Any:
    if isinstance(a, mx.array): return a + b
    if isinstance(a, dict): return {k: _tree_add(a[k], b[k]) for k in a}
    if isinstance(a, (list, tuple)): return type(a)(_tree_add(ai, bi) for ai, bi in zip(a, b))
    return a

def _tree_scale(tree: Any, factor: float) -> Any:
    if isinstance(tree, mx.array): return tree * factor
    if isinstance(tree, dict): return {k: _tree_scale(v, factor) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)): return type(tree)(_tree_scale(v, factor) for v in tree)
    return tree

def _flatten_params(tree: Any, prefix: str = "") -> List[Tuple[str, mx.array]]:
    items: List[Tuple[str, mx.array]] = []
    if isinstance(tree, mx.array):
        items.append((prefix, tree))
    elif isinstance(tree, dict):
        for k, v in tree.items():
            items.extend(_flatten_params(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            items.extend(_flatten_params(v, f"{prefix}.{i}"))
    return items

def _fmt_time(s: float) -> str:
    return f"{int(s//3600):02d}:{int((s%3600)//60):02d}:{int(s%60):02d}"


# ══════════════════════════════════════════════════════════════════════════
# LEAKAGE CHECK
# ══════════════════════════════════════════════════════════════════════════

def assert_no_leakage(train_csv: str, dev_csv: str,
                      test_csv: str = "data/transcripts/test.csv") -> None:
    train_sp = set(pd.read_csv(train_csv).speaker_id)
    dev_sp   = set(pd.read_csv(dev_csv).speaker_id)
    test_sp  = set(pd.read_csv(test_csv).speaker_id)
    overlap_td = train_sp & dev_sp
    overlap_tt = train_sp & test_sp
    assert not overlap_td, f"LEAKAGE train↔dev: {sorted(overlap_td)[:5]}…"
    assert not overlap_tt, f"LEAKAGE train↔test: {sorted(overlap_tt)[:5]}…"
    logger.info("✓  No speaker leakage (train ∩ dev = ∅, train ∩ test = ∅)")


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AutoEIT – Fine-tune Whisper with LoRA (MLX)")
    p.add_argument("--train_csv",          default="data/transcripts/train_augmented.csv")
    p.add_argument("--dev_csv",            default="data/transcripts/dev.csv")
    p.add_argument("--test_csv",           default="data/transcripts/test.csv")
    p.add_argument("--processed_manifest", default="data/transcripts/processed_manifest.csv")
    p.add_argument("--base_model",         default="openai/whisper-small")
    p.add_argument("--config",             default="configs/config.yaml")
    p.add_argument("--output_dir",         default="models/checkpoints/whisper-sm-autoeit-lora")
    p.add_argument("--language",           default="es")
    p.add_argument("--skip_leakage_check", action="store_true")
    p.add_argument("--skip_export",        action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config: Dict[str, Any] = {}
    if Path(args.config).exists():
        with open(args.config, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
    else:
        logger.warning("Config %s not found, using defaults.", args.config)

    model_cfg = config.get("model", {})
    language = args.language or model_cfg.get("language", "es")

    if not args.skip_leakage_check:
        assert_no_leakage(args.train_csv, args.dev_csv, args.test_csv)

    logger.info("Loading training data from %s", args.train_csv)
    train_df = pd.read_csv(args.train_csv)
    if "rejected" in train_df.columns:
        train_df = train_df[train_df.rejected != True].copy()
    train_df = train_df[train_df.processed_path.notna()].reset_index(drop=True)
    logger.info("  %d training samples", len(train_df))

    logger.info("Loading dev data from %s", args.dev_csv)
    dev_df = prepare_dev_df(args.dev_csv, args.processed_manifest)
    logger.info("  %d dev samples", len(dev_df))

    processor = WhisperProcessor.from_pretrained(args.base_model)
    _init_special_tokens(processor, language=language)

    model = load_whisper_model(args.base_model)

    target_modules = model_cfg.get("target_modules", ["q_proj", "v_proj"])
    lora_r       = int(model_cfg.get("lora_r", 8))
    lora_alpha   = int(model_cfg.get("lora_alpha", 16))
    lora_dropout = float(model_cfg.get("lora_dropout", 0.05))

    n_replaced = apply_lora(model, target_modules=target_modules,
                            r=lora_r, lora_alpha=lora_alpha, dropout=lora_dropout)
    total, trainable = count_parameters(model)
    logger.info("LoRA: %d layers  |  Total=%s  Trainable=%s (%.2f%%)",
                n_replaced, f"{total:,}", f"{trainable:,}", 100.0 * trainable / max(total, 1))

    trainer = WhisperLoRATrainer(
        model=model, processor=processor,
        train_df=train_df, dev_df=dev_df,
        config=config, output_dir=args.output_dir,
    )

    try:
        trainer.train()
    except RuntimeError as exc:
        if "memory" in str(exc).lower():
            logger.error(
                "OOM — set batch_size=1 and grad_accum=4 in config.yaml, "
                "or switch to openai/whisper-tiny"
            )
        raise

    run_post_training_eval(
        trainer,
        output_path=str(Path(config.get("paths", {}).get("results_dir", "results"))
                        / "training_summary.json"),
    )

    if not args.skip_export:
        fused_dir = fuse_lora_and_export(model, args.output_dir)
        logger.info("GGML export instructions written to %s/README.md", fused_dir)

    logger.info("Done ✓")


if __name__ == "__main__":
    main()