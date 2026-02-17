# Data Layout

## Local Paths

- `data/raw/images/`: local tile cache (optional when using S3-first mode)
- `data/raw/labels.csv`: weak labels from heuristic scoring
- `data/raw/labels_human.csv`: manual annotations (`scenic_human`, confidence, notes)
- `data/processed/`: reports, feature exports, run metadata, regression artifacts

## S3 Canonical Paths

Keep tile keys under:

- `raw/images/satellite/z{zoom}/{region}/{x}_{y}.png`
- `raw/images/terrain/z{zoom}/{region}/{x}_{y}.png`

No extra nested zoom folder under region.

## Annotation Workflow Notes

- For local tile loading in annotator: `raw_dir = data/raw`
- For S3 tile loading in annotator: `raw_dir = s3://<bucket>/raw`
- Manual labels are appended/upserted to `data/raw/labels_human.csv`

## Current Active Working Sets

- `masswhites` at `z14` (primary region)
- `amherst_ma` at `z16` (secondary/local validation)

## Regression Artifacts (Step 3)

- Feature exports may include:
  - `sample_weights` (for mixed supervision, e.g. human=3.0 / heuristic=1.0)
  - `image_paths` (row alignment/audit)
- Recommended mixed labels source:
  - `data/processed/regression/labels_masswhites_z14_mixed5000.csv`

## Git Policy

Large datasets and generated artifacts are ignored via `.gitignore`.
