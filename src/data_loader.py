"""
AutoEIT Data Loader Module
===========================

Parses the manifest CSV, filters usable audio+transcript pairs,
parses all CHAT (.cha) files, and produces speaker-stratified
train / dev / test splits for the AutoEIT Spanish L2 speech
transcription pipeline.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

# ──────────────────────────────────────────────────────────────
# WOCAE l1 mapping
# ──────────────────────────────────────────────────────────────
_WOCAE_L1_MAP: dict[str, str] = {
    "2018": "Chinese",
    "2019": "Chinese",
    "Chinese": "Chinese",
}


# ──────────────────────────────────────────────────────────────
# CHAT Code Stripping Helpers
# ──────────────────────────────────────────────────────────────

# Timestamp pattern: digits_digits at end of an utterance line
_TS_PATTERN = re.compile(r"\s*(\d+)_(\d+)\s*$")
# Bullet-style timestamps  •NUMBER_NUMBER•  or  \x15NUMBER_NUMBER\x15
_BULLET_TS_PATTERN = re.compile(r"[\u2022\x15](\d+)_(\d+)[\u2022\x15]")

# Dependent tier prefixes to skip entirely
_DEPENDENT_TIER_RE = re.compile(r"^%(mor|gra|com|err|xdb|act|exp):")

# CHAT code patterns (order matters for some)
_REPLACEMENT_CORRECTION = re.compile(r"\[:\s*([^\]]+)\]")  # [: word] → word
_ERROR_MARKER = re.compile(r"\[\*\]")                       # [*] → remove
_UNINTELLIGIBLE = re.compile(r"\b(xxx|yyy|www)\b")          # xxx/yyy/www
_ANGLE_BRACKETS = re.compile(r"<([^>]*)>")                  # <...> → keep inner
_PHONOLOGICAL_FRAGMENT = re.compile(r"&\w+")                # &word (but NOT &- or &=)
_PARALINGUISTIC = re.compile(r"&=\w+")                      # &=word
_FILLER_PREFIX = re.compile(r"&-(\w+)")                     # &-eh → eh
_REPETITION_MARKERS = re.compile(r"\[/+\]")                 # [/] [//]
_PAUSE_MARKERS = re.compile(r"\(\.\.*\)")                   # (.) (..) (...)
_TRAILING_MARKERS = re.compile(r"\+\.\.\.|\+/+")            # +... +/ +//
_ZERO_PREFIX = re.compile(r"\b0(\w+)")                      # 0word → word
_CODE_SWITCH = re.compile(r"@s:\w+")                        # @s:lang
_CARET = re.compile(r"\^")                                  # ^
_LANGUAGE_MARKERS = re.compile(r"\[-\s*\w+\]")              # [- eng]
_OVERLAP_MARKERS = re.compile(r"\[<\]|\[>\]")               # overlap markers
_GRAMMAR_MARKERS = re.compile(r"\[\+\s*[^\]]*\]")           # [+ gram] etc.
_EXTRA_BRACKETS = re.compile(r"\[[^\]]*\]")                 # leftover brackets
_MULTI_SPACES = re.compile(r"\s{2,}")                       # collapse whitespace


def _clean_utterance(text: str) -> str:
    """Apply CHAT code stripping rules to utterance text.

    Parameters
    ----------
    text : str
        Raw utterance text (timestamps already removed).

    Returns
    -------
    str
        Cleaned utterance text.
    """
    # 1. Replace [: corrected_form] – keep correction, remove preceding word
    #    Pattern: word [: correction] → correction
    text = re.sub(r"(\S+)\s*\[:\s*([^\]]+)\]", r"\2", text)

    # 2. [*] – remove the marker, KEEP the preceding word
    text = _ERROR_MARKER.sub("", text)

    # 3. xxx / yyy / www → <unintelligible>
    text = _UNINTELLIGIBLE.sub("<unintelligible>", text)

    # 4. <...> → keep words inside, remove angle brackets
    text = _ANGLE_BRACKETS.sub(r"\1", text)

    # 5. &=word → remove (paralinguistic)
    text = _PARALINGUISTIC.sub("", text)

    # 6. &-word → keep as filler text (eh, um)
    text = _FILLER_PREFIX.sub(r"\1", text)

    # 7. &word (phonological fragment) – must come AFTER &- and &= rules
    text = _PHONOLOGICAL_FRAGMENT.sub("", text)

    # 8. [/] [//] repetition/revision markers
    text = _REPETITION_MARKERS.sub("", text)

    # 9. (.) (..) (...) pause markers
    text = _PAUSE_MARKERS.sub("", text)

    # 10. +... +/ +// trailing off / interruption markers
    text = _TRAILING_MARKERS.sub("", text)

    # 11. 0word → word
    text = _ZERO_PREFIX.sub(r"\1", text)

    # 12. @s:lang code-switching markers
    text = _CODE_SWITCH.sub("", text)

    # 13. ^
    text = _CARET.sub("", text)

    # 14. [- eng] language markers
    text = _LANGUAGE_MARKERS.sub("", text)

    # 15. [>] [<] overlap markers
    text = _OVERLAP_MARKERS.sub("", text)

    # 16. [+ gram] etc.
    text = _GRAMMAR_MARKERS.sub("", text)

    # 17. Any remaining bracket annotations
    text = _EXTRA_BRACKETS.sub("", text)

    # 18. Strip terminal punctuation symbols used in CHAT
    text = text.replace("‡", "").replace("„", "")

    # 19. Collapse whitespace & strip
    text = _MULTI_SPACES.sub(" ", text).strip()

    # 20. Remove trailing punctuation-only residue (. ? !)
    text = text.strip(" .?!,")
    text = _MULTI_SPACES.sub(" ", text).strip()

    return text


# ──────────────────────────────────────────────────────────────
# CHAT File Parser
# ──────────────────────────────────────────────────────────────


def parse_cha_file(
    cha_path: Path,
    target_roles: list[str],
) -> list[dict]:
    """Parse a CHAT (.cha) file and extract utterances for *target_roles*.

    Parameters
    ----------
    cha_path : Path
        Absolute or relative path to the ``.cha`` file.
    target_roles : list[str]
        Participant codes whose utterances to extract
        (e.g. ``["PAR"]``, ``["D21", "D22"]``, ``["STU"]``).

    Returns
    -------
    list[dict]
        Each dict contains:
        ``utterance_text``, ``role_code``, ``utterance_index``,
        ``has_timestamps``, ``start_ms``, ``end_ms``.

    Raises
    ------
    ValueError
        If the file cannot be decoded at all.
    """
    cha_path = Path(cha_path)
    if not cha_path.exists():
        raise FileNotFoundError(f"CHAT file not found: {cha_path}")

    # Step 1 – read with encoding fallback
    text: str = ""
    for enc in ("utf-8", "latin-1"):
        try:
            text = cha_path.read_text(encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if not text:
        raise ValueError(f"Cannot decode CHAT file with utf-8 or latin-1: {cha_path}")

    lines = text.splitlines()

    # Step 2 – extract header metadata (light – we only need it for context)
    session_metadata: dict = {}
    for line in lines:
        if line.startswith("@Languages:"):
            session_metadata["languages"] = line.split(":", 1)[1].strip()
        elif line.startswith("@ID:"):
            session_metadata.setdefault("ids", []).append(line.split(":", 1)[1].strip())

    # Step 3-4 – collect utterances with multi-line joining
    # A speaker line starts with *ROLE:\t
    # Continuation lines start with \t (tab)
    raw_utterances: list[tuple[str, str]] = []  # (role_code, full_text)
    current_role: Optional[str] = None
    current_text: str = ""
    in_utterance = False

    for line in lines:
        # Skip header lines and dependent tiers
        if line.startswith("@"):
            if in_utterance:
                raw_utterances.append((current_role, current_text))  # type: ignore[arg-type]
                in_utterance = False
            continue
        if _DEPENDENT_TIER_RE.match(line):
            continue
        # New speaker line
        if line.startswith("*"):
            if in_utterance and current_role is not None:
                raw_utterances.append((current_role, current_text))
            colon_idx = line.index(":")
            role = line[1:colon_idx]
            text_part = line[colon_idx + 1 :].strip()
            if role in target_roles:
                current_role = role
                current_text = text_part
                in_utterance = True
            else:
                current_role = None
                current_text = ""
                in_utterance = False
        elif line.startswith("\t") and in_utterance:
            # Continuation line
            current_text += " " + line.strip()
        else:
            # Any other line – if we were in an utterance, we may need to close
            # (dependent tiers start with %, already handled above)
            pass

    # Flush last utterance
    if in_utterance and current_role is not None:
        raw_utterances.append((current_role, current_text))

    # Step 5-8 – process each utterance
    results: list[dict] = []
    file_has_timestamps = False
    utterance_index = 0

    for role_code, raw_text in raw_utterances:
        # Extract timestamps
        start_ms: Optional[int] = None
        end_ms: Optional[int] = None

        # Check for trailing NUMBER_NUMBER
        ts_match = _TS_PATTERN.search(raw_text)
        if ts_match:
            start_ms = int(ts_match.group(1))
            end_ms = int(ts_match.group(2))
            raw_text = raw_text[: ts_match.start()]
            file_has_timestamps = True

        # Check for bullet timestamps  •N_N•
        bullet_match = _BULLET_TS_PATTERN.search(raw_text)
        if bullet_match and start_ms is None:
            start_ms = int(bullet_match.group(1))
            end_ms = int(bullet_match.group(2))
            raw_text = _BULLET_TS_PATTERN.sub("", raw_text)
            file_has_timestamps = True

        # Clean the utterance
        cleaned = _clean_utterance(raw_text)

        # Skip empty or only-unintelligible utterances
        if not cleaned or cleaned == "<unintelligible>":
            continue

        results.append(
            {
                "utterance_text": cleaned,
                "role_code": role_code,
                "utterance_index": utterance_index,
                "has_timestamps": start_ms is not None,
                "start_ms": start_ms,
                "end_ms": end_ms,
            }
        )
        utterance_index += 1

    # Retroactively set file-level flag
    for r in results:
        r["has_timestamps"] = file_has_timestamps

    return results


# ──────────────────────────────────────────────────────────────
# Statistics Helper
# ──────────────────────────────────────────────────────────────


def generate_stats(df: pd.DataFrame) -> None:
    """Log corpus statistics to the console.

    Parameters
    ----------
    df : pd.DataFrame
        The full corpus DataFrame produced by :py:meth:`ManifestLoader.load_corpus`.
    """
    logger.info("=" * 60)
    logger.info("CORPUS STATISTICS")
    logger.info("=" * 60)

    for ds in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == ds]
        n_files = sub["audio_path"].nunique()
        n_speakers = sub["speaker_id"].nunique()
        n_utt = len(sub)
        n_ts = int(sub["has_timestamps"].sum())
        logger.info(
            "  %-20s  files=%4d  speakers=%4d  utterances=%5d  with_ts=%5d",
            ds, n_files, n_speakers, n_utt, n_ts,
        )

    logger.info("-" * 60)
    logger.info("Per-L1 utterance counts:")
    for l1, cnt in df["l1_group"].value_counts().sort_index().items():
        logger.info("  %-20s  %5d", l1, cnt)

    unintelligible_frac = (
        df["transcript"].str.contains("<unintelligible>", na=False).mean()
    )
    logger.info(
        "Unintelligible utterance fraction: %.4f", unintelligible_frac,
    )
    logger.info("=" * 60)


# ──────────────────────────────────────────────────────────────
# Manifest Loader
# ──────────────────────────────────────────────────────────────


class ManifestLoader:
    """Load and process the AutoEIT manifest for model training.

    Parameters
    ----------
    manifest_path : str
        Path to the ``manifest.csv`` file (relative or absolute).
    data_root : str
        Root directory that transcript and audio paths are relative to.
    """

    def __init__(
        self,
        manifest_path: str = "manifest.csv",
        data_root: str = ".",
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root)

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest CSV not found at {self.manifest_path.resolve()}. "
                "Make sure you are running from the project root or "
                "pass the correct path."
            )

    # ──────────────── load_corpus ────────────────

    def load_corpus(self) -> pd.DataFrame:
        """Read the manifest, parse all CHAT files, and return a flat
        utterance-level DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: ``utterance_id``, ``audio_path``, ``transcript``,
            ``speaker_id``, ``dataset``, ``l1_group``, ``task_type``,
            ``location``, ``has_timestamps``, ``start_ms``, ``end_ms``,
            ``duration_str``, ``utterance_index``.
        """
        logger.info("Reading manifest from %s", self.manifest_path)
        manifest = pd.read_csv(self.manifest_path, dtype=str)

        # Normalise boolean column
        manifest["audio_exists"] = manifest["audio_exists"].str.strip().str.lower() == "true"

        # Step 2 – filter
        mask = (manifest["audio_exists"] == True) & (  # noqa: E712
            manifest["modality"].str.strip().str.lower() == "oral"
        )
        manifest = manifest[mask].reset_index(drop=True)
        logger.info(
            "After filtering: %d oral rows with audio", len(manifest),
        )

        per_dataset = manifest["dataset"].value_counts()
        for ds, cnt in per_dataset.items():
            logger.info("  %-20s  %d files", ds, cnt)

        # Step 3 – parse each CHAT file
        all_rows: list[dict] = []
        for _, row in manifest.iterrows():
            cha_path = self.data_root / row["transcript_path"]
            participant_ids_raw: str = str(row.get("participant_ids", ""))
            target_roles = [r.strip() for r in participant_ids_raw.split("|") if r.strip()]

            if not target_roles:
                logger.warning(
                    "No participant_ids for %s – skipping", row["transcript_path"],
                )
                continue

            try:
                utterances = parse_cha_file(cha_path, target_roles)
            except (FileNotFoundError, ValueError) as exc:
                logger.warning("Skipping %s: %s", cha_path, exc)
                continue

            if not utterances:
                logger.debug("No learner utterances in %s", cha_path)
                continue

            dataset: str = str(row["dataset"])
            media_stem: str = str(row["media_stem"])
            l1_group_raw: str = str(row.get("l1_group", ""))
            task_type: str = str(row.get("task_type", ""))
            location: str = str(row.get("location", ""))
            duration_str: str = str(row.get("duration", ""))
            audio_path_str = str(self.data_root / row["audio_path"])

            # Build l1_group
            if dataset == "Nebrija-INMIGRA":
                l1_label = l1_group_raw
            elif dataset == "Nebrija-WOCAE":
                l1_label = _WOCAE_L1_MAP.get(l1_group_raw, "Chinese")
            elif dataset == "SPLLOC1":
                l1_label = "English"
            else:
                l1_label = l1_group_raw

            for utt in utterances:
                role = utt["role_code"]
                # Build speaker_id
                if dataset == "SPLLOC1":
                    speaker_id = f"{media_stem}_{role}"
                else:
                    speaker_id = media_stem

                utterance_id = (
                    f"{dataset}_{media_stem}_{role}_{utt['utterance_index']}"
                )

                all_rows.append(
                    {
                        "utterance_id": utterance_id,
                        "audio_path": audio_path_str,
                        "transcript": utt["utterance_text"],
                        "speaker_id": speaker_id,
                        "dataset": dataset,
                        "l1_group": l1_label,
                        "task_type": task_type,
                        "location": location,
                        "has_timestamps": utt["has_timestamps"],
                        "start_ms": utt["start_ms"],
                        "end_ms": utt["end_ms"],
                        "duration_str": duration_str,
                        "utterance_index": utt["utterance_index"],
                    }
                )

        df = pd.DataFrame(all_rows)
        logger.info("Total utterances parsed: %d", len(df))

        generate_stats(df)
        return df

    # ──────────────── split ────────────────

    def split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Produce speaker-stratified train / dev / test splits.

        Parameters
        ----------
        df : pd.DataFrame
            Full corpus DataFrame from :py:meth:`load_corpus`.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            ``(train_df, dev_df, test_df)``
        """
        logger.info("Splitting corpus (70 / 15 / 15) by speaker …")

        # Build a stratification key per speaker
        # Combine dataset + (task_type for SPLLOC1, l1_group for INMIGRA)
        speaker_info = (
            df.groupby("speaker_id")
            .agg(
                dataset=("dataset", "first"),
                l1_group=("l1_group", "first"),
                task_type=("task_type", "first"),
            )
            .reset_index()
        )
        speaker_info["strat_key"] = speaker_info.apply(
            lambda r: (
                f"{r['dataset']}_{r['task_type']}"
                if r["dataset"] == "SPLLOC1"
                else f"{r['dataset']}_{r['l1_group']}"
            ),
            axis=1,
        )

        # Merge strat key back
        df = df.merge(
            speaker_info[["speaker_id", "strat_key"]],
            on="speaker_id",
            how="left",
        )

        # First split: 70% train, 30% remainder
        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
        speakers = df["speaker_id"].values
        train_idx, rest_idx = next(gss1.split(df, groups=speakers))

        train_df = df.iloc[train_idx].copy()
        rest_df = df.iloc[rest_idx].copy()

        # Second split: 50/50 of remainder → 15% dev, 15% test
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
        rest_speakers = rest_df["speaker_id"].values
        dev_idx, test_idx = next(gss2.split(rest_df, groups=rest_speakers))

        dev_df = rest_df.iloc[dev_idx].copy()
        test_df = rest_df.iloc[test_idx].copy()

        # Drop helper column
        for _df in (train_df, dev_df, test_df, df):
            if "strat_key" in _df.columns:
                _df.drop(columns=["strat_key"], inplace=True)

        # ── Leakage assertion ──
        train_speakers = set(train_df["speaker_id"])
        dev_speakers = set(dev_df["speaker_id"])
        test_speakers = set(test_df["speaker_id"])

        assert train_speakers.isdisjoint(dev_speakers), "Speaker leakage: train ∩ dev"
        assert train_speakers.isdisjoint(test_speakers), "Speaker leakage: train ∩ test"
        assert dev_speakers.isdisjoint(test_speakers), "Speaker leakage: dev ∩ test"
        logger.info("✓ No speaker leakage across splits.")

        # ── Save CSVs ──
        transcripts_dir = self.data_root / "transcripts"
        transcripts_dir.mkdir(parents=True, exist_ok=True)

        df.to_csv(transcripts_dir / "master.csv", index=False)
        train_df.to_csv(transcripts_dir / "train.csv", index=False)
        dev_df.to_csv(transcripts_dir / "dev.csv", index=False)
        test_df.to_csv(transcripts_dir / "test.csv", index=False)
        logger.info("Saved master / train / dev / test CSVs to %s", transcripts_dir)

        # ── Print summary table ──
        self._print_split_summary(train_df, dev_df, test_df)
        self._print_l1_distribution(train_df, dev_df, test_df)

        return train_df, dev_df, test_df

    # ──────────────── helpers ────────────────

    @staticmethod
    def _print_split_summary(
        train_df: pd.DataFrame,
        dev_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        header = (
            "┌─────────────────┬──────────┬──────────────┬─────────────────────────────┐"
        )
        sep = (
            "├─────────────────┼──────────┼──────────────┼─────────────────────────────┤"
        )
        footer = (
            "└─────────────────┴──────────┴──────────────┴─────────────────────────────┘"
        )
        logger.info(header)
        logger.info(
            "│ %-15s │ %8s │ %12s │ %-27s │",
            "Split", "Speakers", "Utterances", "Datasets",
        )
        logger.info(sep)
        for name, sdf in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
            datasets = ", ".join(sorted(sdf["dataset"].unique()))
            logger.info(
                "│ %-15s │ %8d │ %12d │ %-27s │",
                name,
                sdf["speaker_id"].nunique(),
                len(sdf),
                datasets,
            )
        logger.info(footer)

    @staticmethod
    def _print_l1_distribution(
        train_df: pd.DataFrame,
        dev_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        logger.info("Per-L1 distribution across splits:")
        all_l1 = sorted(
            set(train_df["l1_group"].unique())
            | set(dev_df["l1_group"].unique())
            | set(test_df["l1_group"].unique())
        )
        logger.info(
            "  %-20s  %8s  %8s  %8s", "L1", "train", "dev", "test",
        )
        for l1 in all_l1:
            tr = int((train_df["l1_group"] == l1).sum())
            dv = int((dev_df["l1_group"] == l1).sum())
            te = int((test_df["l1_group"] == l1).sum())
            logger.info("  %-20s  %8d  %8d  %8d", l1, tr, dv, te)


# ──────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AutoEIT data loader")
    parser.add_argument(
        "--manifest",
        default="data/manifest.csv",
        help="Path to manifest.csv",
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Root directory for transcript/audio paths",
    )
    args = parser.parse_args()

    loader = ManifestLoader(
        manifest_path=args.manifest,
        data_root=args.data_root,
    )
    corpus = loader.load_corpus()
    train, dev, test = loader.split(corpus)
    logger.info("Done. train=%d  dev=%d  test=%d", len(train), len(dev), len(test))
