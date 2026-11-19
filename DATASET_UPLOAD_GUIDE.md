# Uploading language corpora to Hugging Face

This guide defines the shared upload format for Rajasthani, Haryanvi, Odia,
Bengali, and any additional source language. Each language gets its own
Hugging Face **dataset** repository under the organization. The uploader
converts two aligned text files into the same JSONL schema, checks row counts
and blanks, records source hashes, and uploads the requested split.

The uploader does not translate, judge linguistic quality, deduplicate, split,
or repair data. A successful upload means the files were structurally aligned,
not that the translations are correct. Human review and permission to
redistribute the source material remain the dataset team's responsibility.

## 1. Organization access and tokens

The organization settings shown in the project screenshot have the invite-link
option enabled and the default role set to **Write**. Hugging Face's `Write`
role allows changes to all repositories in the organization. Before inviting
the team:

1. Turn off the public/shareable invitation link unless you actively need it.
2. Set the default role for new members to **Contributor**. A contributor can
   create and update repositories they created, which works with the
   one-repository-per-language arrangement below.
3. Invite only the specific Hugging Face usernames who will upload data.
4. Assign one uploader/maintainer per language repository. A contributor who
   did not create a repository cannot update that repository. If you instead
   want someone to upload to a repository created by you, they need `Write`
   access; that role applies across the organization's repositories. Use it
   only for trusted collaborators. Resource Groups can scope access to selected
   repositories, but Hugging Face documents that feature for Team and
   Enterprise plans.

Each uploader must create and use their **own** Hugging Face user access token
with write permission. Do not send your personal token to the team, put a token
in this GitHub repository, or paste a token into a command. The script uses the
token saved by the Hugging Face CLI. Hugging Face's token permissions and
organization role both limit what an uploader can change.

On each uploader's computer, after their account has been invited and accepted:

```bash
hf auth login
hf auth whoami
```

They should confirm that `whoami` shows the account with the intended
organization membership. If a write fails, check both the user's organization
role and the token's write permissions.

## 2. Repository naming and contents

Use the exact organization slug `transRacoons` unless the organization owner
changes it. The script creates a dataset repository named
`<language>-en` under that organization:

| Language slug passed to the script | Dataset repository |
| --- | --- |
| `rajasthani` | `transRacoons/rajasthani-en` |
| `haryanvi` | `transRacoons/haryanvi-en` |
| `odia` | `transRacoons/odia-en` |
| `bengali` | `transRacoons/bengali-en` |

The language slug must be lowercase letters/numbers with optional single
hyphens. It is independent of the local filenames: teams do **not** need to
rename their files. Pass each actual source/target path pair with `--pair`.

The script creates the following structure for each upload. Use `corpus` for
an unsplit source collection. Use `train`, `validation`, and `test` only after
the research split has been decided; the script does not create splits.

```text
README.md
data/
  corpus.jsonl
  train.jsonl
  validation.jsonl
  test.jsonl
metadata/
  corpus_manifest.json
  train_manifest.json
  validation_manifest.json
  test_manifest.json
```

Only the selected split and its manifest are uploaded in a run. The JSONL
schema is fixed and must not be changed by individual teams:

```json
{"id":"rajasthani-train-00000001","source":"राजस्थानी वाक्य","target":"English sentence.","source_language":"rajasthani","target_language":"english","split":"train","source_id":"my_source.raj:1","review_status":"unreviewed"}
```

Each input file must be UTF-8, with exactly one source/target pair per line and
matching line order. Do not add headers, blank separator lines, or JSON around
the text. The uploader rejects unequal line counts and blank rows rather than
silently dropping or pairing them. It trims leading/trailing whitespace only;
it does not rewrite text, remove duplicates, or verify a translation.

## 3. Prepare and upload

From the project root, make a small virtual environment for the uploader and
install only its Hub dependency. They do not need to install PyTorch or the
model-training packages just to upload a corpus:

```bash
uv venv .venv-upload
uv pip install --python .venv-upload/bin/python -r requirements-dataset-upload.txt
source .venv-upload/bin/activate
hf auth login
hf auth whoami
```

If using a standard Python-created virtual environment with pip available,
`python -m pip install -r requirements-dataset-upload.txt` works as well.

Example with local files that have different names from the examples:

```bash
python "scripts for fintuning/upload_parallel_corpus.py" \
  --language odia \
  --pair "/path/to/team_files/odia_sentences.txt" "/path/to/team_files/english_translations.txt" \
  --split train \
  --review-status unreviewed
```

Repeat `--pair SOURCE TARGET` for each aligned input corpus in the same
language. The script concatenates their records in the order supplied and
stores the original filename and line number in `source_id`. If the corpus has
stable source IDs, pass one `--source-ids-file` per pair in the same order;
each ID file must have one ID per source line. Review TSV/CSV files are not
uploaded by this script and should remain local for human review.

Before uploading, add `--dry-run` to check all input paths, row counts, blank
rows, IDs, and bundle serialization without logging into Hugging Face or
creating a repository.

Run again for the same language with `--split validation` or `--split test`
and those split's paired files. The script does not choose or randomize
splits; all teams must use the split protocol agreed for the study. For a
source corpus that has not had individual human checking, keep
`--review-status unreviewed`. Use `human-reviewed` only after a competent
speaker has checked the translations. Use `automatically-filtered` only when
an actual documented automated filter was applied.

Repositories are created **private by default**. Keep them private unless the
team has checked that it has permission to redistribute every source and
translation. To create a public repository intentionally, add:

```bash
--visibility public
```

The script will refuse to upload if a repository already exists with a
different visibility. Decide visibility deliberately in the Hugging Face
repository settings or rerun with the repository's current visibility.

The script also refuses to replace a corpus or split file that already exists.
To intentionally publish a corrected version of that same corpus/split, rerun with `--overwrite`; the
Hub keeps the earlier version in the repository history. Check the dataset
page after upload and verify row counts, sample pairs, and the manifest.

## 4. Adding and loading datasets

The language repositories stay separate to preserve provenance and allow
language-specific review. An unsplit `corpus.jsonl` is an archive of pairs, not
a training split. Training code should select rows, review them, and create
consistent train/validation/test splits before training. Never use test rows
for training.

Hub upload documentation:

- [Upload files and folders](https://huggingface.co/docs/huggingface_hub/guides/upload)
- [Create and manage repositories](https://huggingface.co/docs/huggingface_hub/guides/repository)
- [Organization access roles](https://huggingface.co/docs/hub/organizations-security)
- [User access tokens](https://huggingface.co/docs/hub/security-tokens)
