import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import os
    import shlex
    import subprocess
    import sys
    from dataclasses import dataclass
    from pathlib import Path

    return Path, dataclass, json, os, shlex, subprocess, sys


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Learned Scenic Scoring (Step 3)

        This notebook orchestrates the Step 3 pipeline using the script CLIs:
        1. Export regression features (`scripts/modeling/export_regression_dataset.py`)
        2. Train baseline regressor (`scripts/modeling/train_regression_baseline.py`)
        3. Evaluate baseline (`scripts/modeling/evaluate_regression_baseline.py`)
        """
    )
    return


@app.cell
def _(Path, dataclass, os):
    @dataclass
    class Config:
        labels_csv: str = "data/raw/labels.csv"
        raw_dir: str = "data/raw"
        dataset_npz: str = "data/processed/regression/features_v1.npz"
        checkpoint: str = "models/scenic_regression_baseline.pt"
        metrics_json: str = "data/processed/regression/baseline_metrics.json"

        skip_missing: bool = True
        max_samples: int | None = None
        sample_weight_column: str | None = None
        label_source_column: str = "label_source"
        human_weight: float = 4.0
        heuristic_weight: float = 1.0
        default_weight: float = 1.0

        epochs: int = 40
        batch_size: int = 128
        lr: float = 1e-3
        val_split: float = 0.15
        seed: int = 42
        device: str = "auto"
        use_sample_weights: bool = True

        run_export: bool = True
        run_train: bool = True
        run_eval: bool = True

    cfg = Config()

    _s3_bucket = os.getenv("SCENIC_S3_BUCKET")
    _s3_only = os.getenv("SCENIC_S3_ONLY", "").strip().lower() in {"1", "true", "yes"}
    if _s3_only and _s3_bucket:
        cfg.raw_dir = f"s3://{_s3_bucket}/raw"

    Path(cfg.dataset_npz).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.metrics_json).parent.mkdir(parents=True, exist_ok=True)
    cfg
    return (cfg,)


@app.cell
def _(Path, cfg, shlex, subprocess, sys):
    def _run(cmd: list[str]) -> None:
        print("$", " ".join(shlex.quote(p) for p in cmd))
        subprocess.run(cmd, check=True)

    export_cmd = [
        sys.executable,
        "scripts/modeling/export_regression_dataset.py",
        "--labels-csv",
        cfg.labels_csv,
        "--raw-dir",
        cfg.raw_dir,
        "--output",
        cfg.dataset_npz,
        "--device",
        cfg.device,
    ]
    if cfg.skip_missing:
        export_cmd.append("--skip-missing")
    if cfg.max_samples is not None:
        export_cmd.extend(["--max-samples", str(cfg.max_samples)])
    if cfg.sample_weight_column:
        export_cmd.extend(["--sample-weight-column", cfg.sample_weight_column])
    else:
        export_cmd.extend(
            [
                "--label-source-column",
                cfg.label_source_column,
                "--human-weight",
                str(cfg.human_weight),
                "--heuristic-weight",
                str(cfg.heuristic_weight),
                "--default-weight",
                str(cfg.default_weight),
            ]
        )

    train_cmd = [
        sys.executable,
        "scripts/modeling/train_regression_baseline.py",
        "--dataset",
        cfg.dataset_npz,
        "--output",
        cfg.checkpoint,
        "--epochs",
        str(cfg.epochs),
        "--batch-size",
        str(cfg.batch_size),
        "--lr",
        str(cfg.lr),
        "--val-split",
        str(cfg.val_split),
        "--seed",
        str(cfg.seed),
        "--device",
        cfg.device,
    ]
    if cfg.use_sample_weights:
        train_cmd.append("--use-sample-weights")
    else:
        train_cmd.append("--no-use-sample-weights")

    eval_cmd = [
        sys.executable,
        "scripts/modeling/evaluate_regression_baseline.py",
        "--dataset",
        cfg.dataset_npz,
        "--checkpoint",
        cfg.checkpoint,
        "--val-split",
        str(cfg.val_split),
        "--seed",
        str(cfg.seed),
        "--batch-size",
        str(cfg.batch_size),
        "--device",
        cfg.device,
        "--metrics-json",
        cfg.metrics_json,
    ]

    if cfg.run_export:
        _run(export_cmd)
    else:
        print("Skipping export step (cfg.run_export=False)")

    if cfg.run_train:
        if not Path(cfg.dataset_npz).exists():
            raise FileNotFoundError(f"Dataset not found: {cfg.dataset_npz}")
        _run(train_cmd)
    else:
        print("Skipping train step (cfg.run_train=False)")

    if cfg.run_eval:
        if not Path(cfg.checkpoint).exists():
            raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint}")
        _run(eval_cmd)
    else:
        print("Skipping eval step (cfg.run_eval=False)")

    return


@app.cell
def _(Path, cfg, json):
    metrics_path = Path(cfg.metrics_json)
    if metrics_path.exists():
        print(f"Metrics: {metrics_path}")
        print(json.dumps(json.loads(metrics_path.read_text(encoding='utf-8')), indent=2))
    else:
        print(f"No metrics file yet: {metrics_path}")
    return


if __name__ == "__main__":
    app.run()
