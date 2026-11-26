import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    """Overview."""
    import marimo as _mo

    _mo.md(
        """
        # Training Hub (Marimo-only)
        This hub links four supported training and annotation notebooks:
        - `notebooks/classifier.mo.py` (RESISC45 landscape classifier training)
        - `notebooks/regression.mo.py` (heuristic-label generation + multitask regression/classification training)
        - `notebooks/learned_scoring.mo.py` (feature export + baseline regressor training and evaluation with benchmark-bound splits)
        - `notebooks/annotate_scenic.mo.py` (manual scenic labeling)

        Quick start:
        - `uv run marimo edit notebooks/classifier.mo.py`
        - `uv run marimo edit notebooks/regression.mo.py`
        - `uv run marimo edit notebooks/learned_scoring.mo.py`
        - `uv run marimo edit notebooks/annotate_scenic.mo.py`
        """
    )


if __name__ == "__main__":
    app.run()
