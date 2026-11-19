#!/usr/bin/env python3
"""Validate aligned line files, package them, and upload a corpus or split to the HF Hub.

Usage and organization permissions are documented in DATASET_UPLOAD_GUIDE.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError


ORG_DEFAULT = "transRacoons"
TARGET_LANGUAGE = "english"
SPLITS = ("corpus", "train", "validation", "test")
REVIEW_STATUSES = ("unreviewed", "automatically-filtered", "human-reviewed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate one or more pairs of one-line-per-example text files, "
            "convert them to the standard JSONL layout, and upload a corpus or split."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--language", required=True,
                        help="Lowercase language slug, e.g. rajasthani, haryanvi, odia, bengali")
    parser.add_argument("--pair", required=True, action="append", nargs=2, type=Path,
                        metavar=("SOURCE_FILE", "TARGET_FILE"),
                        help="UTF-8 source and English files aligned line by line; repeat to combine corpora")
    parser.add_argument("--source-ids-file", action="append", type=Path, default=None,
                        help="Optional one-ID-per-line file for each --pair, repeated in the same order")
    parser.add_argument("--split", choices=SPLITS, default="train",
                        help="Use corpus for an unsplit archive, or an existing train/validation/test split; never split or shuffle here")
    parser.add_argument("--org", default=ORG_DEFAULT,
                        help="Hugging Face organization that owns the dataset repositories")
    parser.add_argument("--visibility", choices=("private", "public"), default="private",
                        help="Visibility used when creating a new repository")
    parser.add_argument("--review-status", choices=REVIEW_STATUSES, default="unreviewed",
                        help="Corpus-level review status recorded in every row and the manifest")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow replacing this split's existing JSONL and manifest files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and package the input without logging in or uploading anything")
    return parser.parse_args()


def validate_slug(value: str, label: str) -> str:
    slug = value.strip().lower()
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug):
        raise ValueError(f"{label} must be a lowercase slug using letters, numbers, and single hyphens: {value!r}")
    return slug


def read_lines(path: Path, label: str) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} file does not exist: {path}")
    # utf-8-sig accepts ordinary UTF-8 and removes a possible BOM at file start.
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_rows(pairs: list[tuple[Path, Path]], id_files: list[Path] | None,
                 language: str, split: str, review_status: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if id_files is not None and len(id_files) != len(pairs):
        raise ValueError("Provide exactly one --source-ids-file for each --pair, in the same order.")

    rows: list[dict[str, object]] = []
    input_manifest: list[dict[str, object]] = []
    for pair_number, (source_path, target_path) in enumerate(pairs):
        source = read_lines(source_path, "Source")
        target = read_lines(target_path, "Target")
        if not source or not target:
            raise ValueError(f"Input pair {pair_number + 1} contains no examples.")
        if len(source) != len(target):
            raise ValueError(
                f"Pair {pair_number + 1} line counts do not match: source has {len(source)}, "
                f"target has {len(target)}. No files were uploaded."
            )
        blank_rows = [index for index, (src, tgt) in enumerate(zip(source, target), start=1)
                      if not src or not tgt]
        if blank_rows:
            preview = ", ".join(map(str, blank_rows[:20]))
            suffix = " ..." if len(blank_rows) > 20 else ""
            raise ValueError(
                f"Pair {pair_number + 1} has {len(blank_rows)} blank row(s) (line(s): {preview}{suffix}). "
                "Fix or explicitly review these rows before uploading."
            )

        ids: list[str] | None = None
        id_path: Path | None = None
        if id_files is not None:
            id_path = id_files[pair_number]
            ids = read_lines(id_path, "Source ID")
            if len(ids) != len(source) or any(not value for value in ids):
                raise ValueError(f"Source ID file {id_path} must have one nonblank ID per source row.")

        for local_index, (src, tgt) in enumerate(zip(source, target), start=1):
            rows.append({
                "id": f"{language}-{split}-{len(rows) + 1:08d}",
                "source": src,
                "target": tgt,
                "source_language": language,
                "target_language": TARGET_LANGUAGE,
                "split": split,
                "source_id": ids[local_index - 1] if ids is not None else f"{source_path.name}:{local_index}",
                "review_status": review_status,
            })
        input_manifest.append({
            "source_file": source_path.name,
            "target_file": target_path.name,
            "source_id_file": id_path.name if id_path else None,
            "row_count": len(source),
            "source_sha256": sha256_file(source_path),
            "target_sha256": sha256_file(target_path),
            "source_id_sha256": sha256_file(id_path) if id_path else None,
        })
    return rows, input_manifest


def write_upload_bundle(directory: Path, rows: list[dict[str, object]],
                        manifest: dict[str, object], language: str,
                        repo_id: str) -> None:
    data_dir = directory / "data"
    metadata_dir = directory / "metadata"
    data_dir.mkdir(parents=True)
    metadata_dir.mkdir()

    split = str(manifest["split"])
    manifest_path = f"metadata/{split}_manifest.json"
    data_path = f"data/{split}.jsonl"
    with (data_dir / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    (metadata_dir / f"{split}_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    readme = f"""---
pretty_name: {language.title()} to English Parallel Corpus
task_categories:
  - translation
tags:
  - parallel-corpus
  - {language}-english
---

# {language.title()}–English Parallel Corpus

This repository stores line-aligned {language}–English examples. Each JSONL row
contains `source`, `target`, language labels, a corpus/split name, a stable row
ID, a source ID, and the declared review status. The source ID comes from the
provided ID file when one is supplied; otherwise it is the source filename and
line number.

## Layout

- `data/corpus.jsonl`: optional unsplit corpus archive.
- `data/train.jsonl`, `data/validation.jsonl`, `data/test.jsonl`: optional prepared splits.
- `metadata/<name>_manifest.json`: input filenames, hashes, row count, and upload metadata.

This upload added `{data_path}` with {len(rows)} pairs and
`{manifest_path}`. Review files are not uploaded.

The uploader checks file existence, equal line counts, and blank rows. It does
not judge translation correctness, remove duplicates, normalize dialect, or
create train/validation/test splits. Review the corpus locally before using it
for training. See the project upload guide for the preparation process.

Repository: `{repo_id}`
"""
    (directory / "README.md").write_text(readme, encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        language = validate_slug(args.language, "--language")
        org = args.org.strip()
        if not org or "/" in org or " " in org:
            raise ValueError("--org must be a Hugging Face organization slug, without slashes or spaces")

        pairs = [(Path(source_path), Path(target_path)) for source_path, target_path in args.pair]
        rows, input_manifest = prepare_rows(
            pairs, args.source_ids_file, language, args.split, args.review_status,
        )
        repo_id = f"{org}/{language}-en"
        jsonl_path = f"data/{args.split}.jsonl"
        manifest_path = f"metadata/{args.split}_manifest.json"
        manifest = {
            "schema_version": 1,
            "dataset_repo": repo_id,
            "source_language": language,
            "target_language": TARGET_LANGUAGE,
            "split": args.split,
            "row_count": len(rows),
            "inputs": input_manifest,
            "review_status": args.review_status,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "note": "The uploader validates alignment structure; it does not verify translation quality or data rights.",
        }

        if args.dry_run:
            with tempfile.TemporaryDirectory(prefix="parallel-corpus-dry-run-") as temp_dir:
                write_upload_bundle(Path(temp_dir), rows, manifest, language, repo_id)
            print(f"Dry run passed: {len(rows)} aligned pairs would be uploaded to {repo_id}")
            print(f"Hub path: data/{args.split}.jsonl | visibility: {args.visibility}")
            print(f"Input pairs: {len(input_manifest)} | review sidecars: not uploaded")
            print("No Hugging Face login or upload was performed.")
            return 0

        api = HfApi()
        # Fail early with an actionable auth message; the token remains in the
        # user's Hugging Face cache and is never requested as a CLI argument.
        try:
            api.whoami()
        except Exception as exc:
            raise RuntimeError("Not logged in to Hugging Face. Run `hf auth login` as this uploader.") from exc

        # Reuse an existing repository without requesting namespace-level repo
        # creation rights. This supports fine-grained tokens scoped to a dataset
        # repo created separately by an organization admin.
        try:
            info = api.repo_info(repo_id=repo_id, repo_type="dataset")
        except RepositoryNotFoundError:
            api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=(args.visibility == "private"),
                exist_ok=True,
            )
            info = api.repo_info(repo_id=repo_id, repo_type="dataset")
        actual_visibility = "private" if info.private else "public"
        if actual_visibility != args.visibility:
            raise RuntimeError(
                f"Repository {repo_id} already exists as {actual_visibility}, but --visibility "
                f"is {args.visibility}. Change repository visibility deliberately on Hugging Face "
                "or rerun with the matching --visibility. No files were uploaded."
            )

        existing_files = {item.rfilename for item in (info.siblings or [])}
        existing_targets = existing_files.intersection({jsonl_path, manifest_path})
        if existing_targets and not args.overwrite:
            files = ", ".join(sorted(existing_targets))
            raise FileExistsError(
                f"The repository already contains {files}. Use --overwrite only if replacing "
                "that split is intentional."
            )

        with tempfile.TemporaryDirectory(prefix="parallel-corpus-upload-") as temp_dir:
            bundle = Path(temp_dir)
            write_upload_bundle(bundle, rows, manifest, language, repo_id)
            api.upload_folder(
                repo_id=repo_id,
                repo_type="dataset",
                folder_path=str(bundle),
                commit_message=f"Upload {language}-English {args.split} corpus ({len(rows)} pairs)",
            )

        print(f"Uploaded {len(rows)} aligned pairs to https://huggingface.co/datasets/{repo_id}")
        print(f"Split: {args.split} | visibility: {actual_visibility} | review status: {args.review_status}")
        if args.overwrite and existing_targets:
            print("Existing split files were replaced; prior versions remain in repository history.")
        print("The script checked structure and alignment only; it did not evaluate translation quality.")
        return 0
    except (OSError, UnicodeError, ValueError, RuntimeError, HfHubHTTPError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
