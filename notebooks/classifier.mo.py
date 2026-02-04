import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    """Header."""
    import marimo as _mo

    _mo.md(
        """
        # Stage 1: RESISC45 Classifier
        - Trains a ViT classifier on RESISC45.
        - Outputs to `models/classifier/`.
        """
    )


@app.cell
def _():
    """Config."""
    from dataclasses import dataclass

    @dataclass
    class Config:
        resisc45_dir: str = "data/NWPU-RESISC45"
        output_dir: str = "models/classifier"
        epochs: int = 50
        batch_size: int = 32
        lr: float = 1e-4
        num_workers: int = 0
        use_resisc45_stats: bool = True
        seed: int = 42

    cfg = Config()
    cfg
    return (cfg,)


@app.cell
def _(cfg):
    """Train classifier."""
    from pathlib import Path
    import sys as _sys

    _project_root = Path.cwd()
    if str(_project_root) not in _sys.path:
        _sys.path.insert(0, str(_project_root))

    from src.classifier.train import train

    _resisc_dir = Path(cfg.resisc45_dir)
    if not _resisc_dir.exists():
        raise FileNotFoundError(f"RESISC45 dataset not found at: {_resisc_dir}")

    train(
        data_dir=_resisc_dir,
        output_dir=Path(cfg.output_dir),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        num_workers=cfg.num_workers,
        use_resisc45_stats=cfg.use_resisc45_stats,
        seed=cfg.seed,
    )


if __name__ == "__main__":
    app.run()
