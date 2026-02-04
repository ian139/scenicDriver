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
        This repo now uses two separate notebooks for training:
        - `notebooks/classifier.mo.py` (Stage 1: RESISC45 classifier)
        - `notebooks/regression.mo.py` (Stage 2/3: heuristic labels + multitask)

        Quick start:
        - `uv run marimo edit notebooks/classifier.mo.py`
        - `uv run marimo edit notebooks/regression.mo.py`
        """
    )


if __name__ == "__main__":
    app.run()
