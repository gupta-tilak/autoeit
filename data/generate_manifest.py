"""
generate_manifest.py
====================
Generates a unified CSV manifest for all CHAT (.cha) transcripts and their
corresponding audio files across all datasets under data/.

Supported datasets (extensible to more):
  - Nebrija-INMIGRA
  - Nebrija-WOCAE
  - SPLLOC1

Output columns
--------------
dataset             Name of the corpus (e.g. SPLLOC1)
transcript_path     Path to .cha file, relative to data/
audio_path          Path to audio file, relative to data/ (empty if not found)
audio_exists        True / False
pid                 @PID value from CHAT header
media_stem          Stem declared in @Media header line
languages           Pipe-separated language codes (e.g. spa|eng)
task_type           @Situation value
location            @Location value
modality            oral | written
l1_group            Nationality / cohort group folder (e.g. Brazilian, 2018)
speaker_roles       Pipe-separated "CODE:Role" from @Participants (e.g. PAR:Participant)
participant_ids     Pipe-separated speaker codes for non-Investigator participants
participant_l1      L1 language code(s) taken from @ID lines of learner participants
transcriber         @Transcriber value
duration            @Time Duration value (if present)
comment             @Comment value (if present)
"""

import csv
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_ROOT = Path(__file__).parent / "data"
OUTPUT_CSV = Path(__file__).parent / "manifest.csv"

# Audio file extensions to look for, in priority order
AUDIO_EXTENSIONS = [".mp3", ".wav", ".flac", ".ogg", ".m4a"]

# Roles we do NOT treat as learner participants
INVESTIGATOR_ROLES = {"investigator", "interviewer", "teacher", "instructor", "native"}

# ---------------------------------------------------------------------------
# CHAT header parsing helpers
# ---------------------------------------------------------------------------

def parse_chat_header(cha_path: Path) -> dict:
    """
    Read the CHAT header block (lines starting with @) and return a dict of
    the key metadata fields we care about.
    """
    meta = {
        "pid": "",
        "languages": [],
        "participants": [],   # list of (code, role) tuples
        "id_lines": [],       # raw @ID values
        "media_stem": "",
        "location": "",
        "task_type": "",
        "transcriber": "",
        "duration": "",
        "comment": "",
    }

    try:
        with open(cha_path, encoding="utf-8", errors="replace") as fh:
            for raw_line in fh:
                line = raw_line.rstrip("\n")

                # Stop once we hit actual transcript utterances
                if line.startswith("*") or line.startswith("@End"):
                    break

                if not line.startswith("@"):
                    continue

                # Strip the leading "@" and split on the first tab or colon
                content = line[1:]
                if ":" not in content:
                    continue
                tag, _, value = content.partition(":")
                tag = tag.strip().upper()
                value = value.strip()

                if tag == "PID":
                    meta["pid"] = value

                elif tag == "LANGUAGES":
                    meta["languages"] = [l.strip() for l in re.split(r"[,\s]+", value) if l.strip()]

                elif tag == "PARTICIPANTS":
                    # Format: CODE Role, CODE Role, ...
                    for part in re.split(r",\s*", value):
                        tokens = part.strip().split()
                        if len(tokens) >= 2:
                            meta["participants"].append((tokens[0], tokens[1]))
                        elif len(tokens) == 1:
                            meta["participants"].append((tokens[0], ""))

                elif tag == "ID":
                    meta["id_lines"].append(value)

                elif tag == "MEDIA":
                    # Format: stem, type  (e.g. "2_030_O, audio")
                    stem = value.split(",")[0].strip()
                    meta["media_stem"] = stem

                elif tag == "LOCATION":
                    meta["location"] = value

                elif tag == "SITUATION":
                    meta["task_type"] = value

                elif tag == "TRANSCRIBER":
                    meta["transcriber"] = value

                elif tag == "TIME DURATION":
                    meta["duration"] = value

                elif tag == "COMMENT":
                    meta["comment"] = value

    except OSError as exc:
        print(f"  [WARN] Cannot read {cha_path}: {exc}", file=sys.stderr)

    return meta


def extract_participant_l1(id_lines: list[str], participant_codes: set[str]) -> list[str]:
    """
    From @ID lines pick the L1 language of non-investigator participants.

    @ID format: language|corpus|code|age|sex|group|SES|role|education|custom
    We return the 'language' field (index 0) for participants whose code
    appears in participant_codes.
    """
    l1s = []
    for line in id_lines:
        parts = line.split("|")
        if len(parts) < 3:
            continue
        lang_code = parts[0].strip()
        speaker_code = parts[2].strip()
        role = parts[7].strip().lower() if len(parts) > 7 else ""

        if speaker_code in participant_codes and role not in INVESTIGATOR_ROLES:
            l1s.append(lang_code)
    return l1s


# ---------------------------------------------------------------------------
# Audio resolution
# ---------------------------------------------------------------------------

def find_audio(transcript_dir: Path, media_stem: str) -> Path | None:
    """
    Given the directory that contains the .cha file, look for an audio file
    whose stem matches `media_stem`.

    Search order:
      1. Sibling *_audio directory at the same level as transcript_dir
         e.g. Brazilian/ → Brazilian_audio/
      2. Sibling *_audio directory one level up (handles escrito/ subdir)
         e.g. Brazilian/escrito/ → Brazilian_audio/
      3. The transcript_dir itself (audio co-located with transcripts)
    """
    if not media_stem:
        return None

    search_dirs: list[Path] = []

    # Level 1: sibling of transcript_dir
    parent = transcript_dir.parent
    sibling = parent / (transcript_dir.name + "_audio")
    if sibling.is_dir():
        search_dirs.append(sibling)

    # Level 2: sibling of transcript_dir's parent (covers escrito/ case)
    grandparent = parent.parent
    if grandparent != transcript_dir.parent:  # guard against filesystem root
        uncle = grandparent / (parent.name + "_audio")
        if uncle.is_dir():
            search_dirs.append(uncle)

    # Level 3: co-located
    search_dirs.append(transcript_dir)

    for audio_dir in search_dirs:
        for ext in AUDIO_EXTENSIONS:
            candidate = audio_dir / (media_stem + ext)
            if candidate.exists():
                return candidate

    return None


# ---------------------------------------------------------------------------
# Path → metadata inference helpers
# ---------------------------------------------------------------------------

def infer_dataset(rel_parts: tuple[str, ...]) -> str:
    """First component of the relative path is the dataset folder."""
    return rel_parts[0] if rel_parts else ""


def infer_l1_group(cha_path: Path, dataset: str) -> str:
    """
    Infer nationality/speaker-group from directory structure.

    Nebrija-INMIGRA  → direct parent folder (Brazilian, Chinese, ...)
    Nebrija-WOCAE    → second-level folder under oral/ or written/
                       (2018, 2019, Chinese, eng, spa, ...)
    SPLLOC1          → direct parent folder (Discussion, Picture, ...)
    """
    parts = cha_path.relative_to(DATA_ROOT).parts  # e.g. ('SPLLOC1', 'Discussion', 'foo.cha')

    if dataset == "Nebrija-INMIGRA":
        # parts: ('Nebrija-INMIGRA', 'Brazilian', 'file.cha')
        return parts[1] if len(parts) > 2 else ""

    if dataset == "Nebrija-WOCAE":
        # parts: ('Nebrija-WOCAE', 'oral', '2018', 'file.cha')  → '2018'
        # parts: ('Nebrija-WOCAE', 'written', 'eng', '2', 'file.cha') → 'eng/2'
        if len(parts) > 3:
            return "/".join(parts[2:-1])
        elif len(parts) > 2:
            return parts[2]
        return ""

    if dataset == "SPLLOC1":
        # parts: ('SPLLOC1', 'Discussion', 'file.cha')
        return parts[1] if len(parts) > 2 else ""

    # Generic fallback: immediate parent
    return cha_path.parent.name


def infer_modality(cha_path: Path, dataset: str) -> str:
    """
    Infer whether this is an oral or written sample.

    Nebrija-WOCAE  → explicit oral/ and written/ top-level subtrees.
    Nebrija-INMIGRA → groups contain an escrito/ subdirectory for written
                      production tasks; everything else is oral.
    SPLLOC1        → all oral.
    """
    parts = cha_path.relative_to(DATA_ROOT).parts

    if dataset == "Nebrija-WOCAE":
        if len(parts) > 1:
            if parts[1] == "written":
                return "written"
            if parts[1] == "oral":
                return "oral"

    if dataset == "Nebrija-INMIGRA":
        # Parts: ('Nebrija-INMIGRA', 'Brazilian', 'escrito', 'file.cha')
        if "escrito" in parts:
            return "written"

    return "oral"


# ---------------------------------------------------------------------------
# Core manifest builder
# ---------------------------------------------------------------------------

def build_manifest(data_root: Path) -> list[dict]:
    rows = []

    for cha_path in sorted(data_root.rglob("*.cha")):
        rel = cha_path.relative_to(data_root)
        rel_parts = rel.parts

        dataset = infer_dataset(rel_parts)

        # Skip metadata / non-transcript .cha files at root level of a dataset
        if len(rel_parts) < 3 and dataset in ("Nebrija-INMIGRA", "SPLLOC1"):
            # These datasets keep .cha files at depth ≥ 3
            pass  # still process, audio search will simply fail gracefully

        # ---- Parse CHAT header ----
        meta = parse_chat_header(cha_path)

        # ---- Resolve audio ----
        # Fallback: if @Media is absent, use the .cha filename stem itself.
        # (Some SPLLOC1 files omit @Media but audio is named after the file.)
        effective_stem = meta["media_stem"] or cha_path.stem
        audio_path = find_audio(cha_path.parent, effective_stem)
        # Persist the resolved stem back so the CSV always has a value
        if not meta["media_stem"]:
            meta["media_stem"] = effective_stem

        # ---- Build participant info ----
        # Separate investigator vs. learner participants
        learner_codes = set()
        speaker_roles_parts = []
        for code, role in meta["participants"]:
            speaker_roles_parts.append(f"{code}:{role}")
            if role.lower() not in INVESTIGATOR_ROLES:
                learner_codes.add(code)

        participant_l1_list = extract_participant_l1(meta["id_lines"], learner_codes)

        # ---- Assemble row ----
        row = {
            "dataset":          dataset,
            "transcript_path":  str(rel).replace("\\", "/"),
            "audio_path":       str(audio_path.relative_to(data_root)).replace("\\", "/")
                                if audio_path else "",
            "audio_exists":     audio_path is not None,
            "pid":              meta["pid"],
            "media_stem":       meta["media_stem"],
            "languages":        "|".join(meta["languages"]),
            "task_type":        meta["task_type"],
            "location":         meta["location"],
            "modality":         infer_modality(cha_path, dataset),
            "l1_group":         infer_l1_group(cha_path, dataset),
            "speaker_roles":    "|".join(speaker_roles_parts),
            "participant_ids":  "|".join(sorted(learner_codes)),
            "participant_l1":   "|".join(participant_l1_list),
            "transcriber":      meta["transcriber"],
            "duration":         meta["duration"],
            "comment":          meta["comment"],
        }
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "dataset",
    "transcript_path",
    "audio_path",
    "audio_exists",
    "pid",
    "media_stem",
    "languages",
    "task_type",
    "location",
    "modality",
    "l1_group",
    "speaker_roles",
    "participant_ids",
    "participant_l1",
    "transcriber",
    "duration",
    "comment",
]


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Scanning: {DATA_ROOT}")
    rows = build_manifest(DATA_ROOT)

    if not rows:
        print("No .cha files found. Check DATA_ROOT path.", file=sys.stderr)
        sys.exit(1)

    write_csv(rows, OUTPUT_CSV)

    # ---- Summary ----
    total = len(rows)
    with_audio = sum(1 for r in rows if r["audio_exists"])
    by_dataset: dict[str, int] = {}
    by_modality: dict[str, int] = {}
    for r in rows:
        by_dataset[r["dataset"]] = by_dataset.get(r["dataset"], 0) + 1
        by_modality[r["modality"]] = by_modality.get(r["modality"], 0) + 1

    print(f"\nManifest written → {OUTPUT_CSV}")
    print(f"  Total transcripts : {total}")
    print(f"  With audio        : {with_audio} ({100*with_audio//total}%)")
    print(f"  By dataset        :")
    for ds, count in sorted(by_dataset.items()):
        print(f"    {ds:<25} {count}")
    print(f"  By modality       :")
    for mod, count in sorted(by_modality.items()):
        print(f"    {mod:<25} {count}")


if __name__ == "__main__":
    main()
