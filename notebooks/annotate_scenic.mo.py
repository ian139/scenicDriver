import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import sys as _sys
    from dataclasses import dataclass
    from pathlib import Path

    import pandas as pd
    from PIL import Image

    _project_root = Path.cwd()
    if str(_project_root) not in _sys.path:
        _sys.path.insert(0, str(_project_root))

    from scripts.annotation.annotate_scenic_web import (
        AnnotatorConfig as WebAnnotatorConfig,
        AnnotatorState,
        ApiError,
        _safe_image_path as safe_image_path,
        _sample_stratified as sample_stratified,
    )
    from scenicdriver.active_learning.finalize import UNUSABLE_REASONS

    return (
        AnnotatorState,
        ApiError,
        Image,
        Path,
        WebAnnotatorConfig,
        UNUSABLE_REASONS,
        dataclass,
        mo,
        os,
        pd,
        safe_image_path,
        sample_stratified,
    )


@app.cell
def _(mo):
    mo.md("""
    # Manual Scenic Annotation
    Annotate tiles with human scenic ratings (0-10) for calibration and benchmark evaluation.
    Nothing is read until **Load Annotation Batch** is submitted. The whole labels CSV is
    registered with the shared web-annotator store as the approved batch; the notebook then
    samples **Sample Size** tiles the selected Annotator ID has not annotated yet. Saves go
    through that store (sidecar-locked, atomic CSV writes; one row per tile and annotator),
    and completion counts only rows saved under the selected Annotator ID.
    """)
    return


@app.cell
def _(Path, dataclass, os):
    @dataclass
    class AnnotatorConfig:
        labels_csv: str = "data/processed/annotation_batches/masswhites_z14_flat_5k_seamfix_batch500/labels_batch.csv"
        raw_dir: str = "data/raw"
        annotations_csv: str = "data/raw/labels_human.csv"
        batch_size: int = 500
        sample_seed: int = 42
        stratify_by_class: bool = True
        annotator_id: str = os.getenv("USER", "annotator")

    cfg = AnnotatorConfig()
    Path(cfg.annotations_csv).parent.mkdir(parents=True, exist_ok=True)
    return (cfg,)


@app.cell
def _(cfg, mo):
    ui_labels_csv = mo.ui.text(label="Labels CSV", value=cfg.labels_csv)
    ui_raw_dir = mo.ui.text(label="Raw Dir", value=cfg.raw_dir)
    ui_annotations_csv = mo.ui.text(label="Annotations CSV", value=cfg.annotations_csv)
    ui_batch_size = mo.ui.number(label="Sample Size", value=cfg.batch_size, step=1)
    ui_sample_seed = mo.ui.number(label="Seed", value=cfg.sample_seed, step=1)
    ui_stratify = mo.ui.checkbox(label="Stratify by class_id", value=cfg.stratify_by_class)
    ui_annotator_id = mo.ui.text(label="Annotator ID", value=cfg.annotator_id)

    config_inputs = mo.ui.dictionary(
        {
            "labels_csv": ui_labels_csv,
            "raw_dir": ui_raw_dir,
            "annotations_csv": ui_annotations_csv,
            "sample_size": ui_batch_size,
            "seed": ui_sample_seed,
            "stratify_by_class": ui_stratify,
            "annotator_id": ui_annotator_id,
        }
    )
    ui_config_form = mo.ui.form(config_inputs, label="Load Annotation Batch")
    mo.vstack([ui_config_form])
    return (ui_config_form,)


@app.cell
def _(AnnotatorState, ApiError, WebAnnotatorConfig, mo, ui_config_form):
    mo.stop(
        ui_config_form.value is None,
        mo.md("Submit **Load Annotation Batch** to start."),
    )
    config = dict(ui_config_form.value)
    config["annotator_id"] = str(config["annotator_id"]).strip() or "annotator"
    config["batch_csv"] = str(config["labels_csv"])
    try:
        annotator_state = AnnotatorState(
            WebAnnotatorConfig(
                labels_csv=str(config["labels_csv"]),
                batch_csv=config["batch_csv"],
                raw_dir=str(config["raw_dir"]),
                annotations_csv=str(config["annotations_csv"]),
                sample_size=int(config["sample_size"]),
                seed=int(config["seed"]),
                stratify_by_class=bool(config["stratify_by_class"]),
                annotator_id=config["annotator_id"],
            )
        )
        # The full labels pool is the approved save membership; display sampling happens below.
        loaded_batch = annotator_state.load_batch(config)
    except ApiError as _exc:
        mo.stop(True, mo.callout(f"Could not load batch: {_exc.message}", kind="danger"))
    return annotator_state, config, loaded_batch


@app.cell
def _(Path, annotator_state, config, loaded_batch, pd, safe_image_path, sample_stratified):
    annotator_id = annotator_state.config.annotator_id
    _ann_path = Path(annotator_state.config.annotations_csv)
    _ann = pd.read_csv(_ann_path) if _ann_path.exists() and _ann_path.stat().st_size else pd.DataFrame()
    if _ann.empty or "annotator_id" not in _ann.columns:
        done_paths = set()
    else:
        _own = _ann["annotator_id"].notna() & (_ann["annotator_id"].astype(str) == annotator_id)
        done_paths = set(_ann.loc[_own, "image_path"].astype(str))

    # Same rows load_batch approved, keeping every source column (scorer metadata included).
    labels_df = pd.read_csv(annotator_state.config.batch_csv).dropna(subset=["image_path"])
    labels_df["image_path"] = labels_df["image_path"].astype(str).map(safe_image_path)
    labels_df = labels_df.drop_duplicates(subset=["image_path"], keep="first")
    _approved = {str(row["image_path"]) for row in loaded_batch["batch"]}
    labels_df = labels_df.loc[labels_df["image_path"].isin(_approved)]

    _n = int(config["sample_size"])
    _seed = int(config["seed"])
    unlabeled_df = labels_df.loc[~labels_df["image_path"].isin(done_paths)].copy()
    if unlabeled_df.empty:
        batch_df = unlabeled_df.reset_index(drop=True)
    elif bool(config["stratify_by_class"]):
        batch_df = sample_stratified(unlabeled_df, _n, _seed)
    else:
        batch_df = unlabeled_df.sample(n=min(_n, len(unlabeled_df)), random_state=_seed).reset_index(drop=True)
    return annotator_id, batch_df, done_paths, labels_df


@app.cell
def _(UNUSABLE_REASONS, annotator_id, batch_df, done_paths, labels_df, mo):
    mo.md(
        f"Labels pool: **{len(labels_df)}** tiles | Sampled pending for `{annotator_id}`: "
        f"**{len(batch_df)}** | Annotated by `{annotator_id}` (any batch): **{len(done_paths)}**"
    )
    if batch_df.empty:
        mo.callout("No unlabeled samples found for this config.", kind="warn")

    max_index = max(0, len(batch_df) - 1)
    current_index, set_current_index = mo.state(0, allow_self_loops=True)
    if current_index() < 0:
        set_current_index(0)
    if current_index() > max_index:
        set_current_index(max_index)

    def _read_value(widget):
        return widget.value() if callable(widget.value) else widget.value

    ui_index = mo.ui.number(
        label="Tile Index",
        value=int(current_index()),
        step=1,
        start=0,
        stop=max_index,
    )
    prev_btn = mo.ui.button(
        label="Previous",
        on_click=lambda _value: set_current_index(max(0, int(current_index()) - 1)),
        disabled=bool(current_index() <= 0),
    )
    next_btn = mo.ui.button(
        label="Next",
        on_click=lambda _value: set_current_index(min(max_index, int(current_index()) + 1)),
        disabled=bool(current_index() >= max_index),
    )
    go_btn = mo.ui.button(
        label="Go",
        on_click=lambda _value: set_current_index(min(max_index, max(0, int(_read_value(ui_index))))),
    )

    nav = mo.hstack([prev_btn, next_btn, ui_index, go_btn], widths="equal")

    ui_score = mo.ui.number(label="Scenic Score (0-10)", value=5.0, step=0.1, start=0, stop=10)
    ui_conf = mo.ui.dropdown(label="Confidence", options=["high", "medium", "low"], value="medium")
    ui_skip = mo.ui.checkbox(label="Skip", value=False)
    ui_unusable_reason = mo.ui.dropdown(
        label="Unusable Reason (required when skipping)",
        options=["", *UNUSABLE_REASONS],
        value="",
    )
    ui_notes = mo.ui.text_area(label="Notes", value="", rows=3)

    ui_save_click = mo.ui.run_button(label="Save Annotation", kind="success")
    controls = mo.vstack(
        [
            mo.md("### Navigation"),
            nav,
            mo.md("### Annotation"),
            ui_score,
            ui_conf,
            ui_skip,
            ui_unusable_reason,
            ui_notes,
            ui_save_click,
        ]
    )
    controls
    return (
        current_index,
        ui_conf,
        ui_notes,
        ui_save_click,
        ui_score,
        ui_skip,
        ui_unusable_reason,
    )


@app.cell
def _(Image, Path, annotator_state, batch_df, current_index, mo):
    image_path = ""
    if batch_df.empty:
        _panel_current = mo.callout("No samples to annotate.", kind="warn")
    else:
        i = int(current_index())
        i = min(max(0, i), len(batch_df) - 1)
        row = batch_df.iloc[i]
        image_path = str(row["image_path"])
        image_abs = Path(annotator_state.config.raw_dir) / image_path

        meta = [f"Index: {i}/{len(batch_df)-1}", f"image_path: {image_path}"]
        for k in ["scenic_score", "class_id", "lat", "lon"]:
            if k in row:
                meta.append(f"{k}: {row[k]}")
        header = mo.md("### Current Tile\n" + "\n".join([f"- {m}" for m in meta]))

        if image_abs.exists():
            body = mo.image(Image.open(image_abs).convert("RGB"))
        else:
            body = mo.callout(f"Image not found: {image_abs}", kind="warn")

        _panel_current = mo.vstack([header, body])
    _panel_current
    return (image_path,)


@app.cell
def _(
    ApiError,
    annotator_state,
    image_path,
    mo,
    ui_conf,
    ui_notes,
    ui_save_click,
    ui_score,
    ui_skip,
    ui_unusable_reason,
):
    def _value(widget):
        return widget.value() if callable(widget.value) else widget.value

    save_result = None
    if bool(_value(ui_save_click)) and image_path:
        _skip = bool(_value(ui_skip))
        _reason = _value(ui_unusable_reason)
        try:
            save_result = annotator_state.save_annotation(
                {
                    "image_path": image_path,
                    "scenic_human": float(_value(ui_score)),
                    "confidence": str(_value(ui_conf)),
                    "skip": _skip,
                    "unusable_reason": _reason if _skip else None,
                    "notes": str(_value(ui_notes)),
                }
            )
        except ApiError as _exc:
            save_result = {"saved": False, "error": _exc.message}
    if save_result is None:
        _panel_save = None
    elif save_result["saved"]:
        _panel_save = mo.callout(
            f"Saved {image_path} to {annotator_state.config.annotations_csv} (rows={save_result['row_count']})",
            kind="success",
        )
    else:
        _panel_save = mo.callout(f"Save failed: {save_result['error']}", kind="danger")
    _panel_save
    return (save_result,)


@app.cell
def _(Path, annotator_id, annotator_state, mo, pd, save_result):
    _ = save_result  # re-render after every save
    _ann_path = Path(annotator_state.config.annotations_csv)
    _progress = annotator_state.summary()
    _completion = mo.md(
        f"### Completion for `{annotator_id}`\n"
        f"**{_progress['completed']}** / {_progress['batch_size']} labels-pool tiles "
        f"(skipped: {_progress['skipped']}, remaining: {_progress['remaining']})"
    )
    if not _ann_path.exists():
        _panel_summary = mo.vstack(
            [_completion, mo.callout(f"No annotations file yet: {_ann_path}", kind="warn")]
        )
    else:
        ann = pd.read_csv(_ann_path)
        if ann.empty:
            _panel_summary = mo.vstack(
                [_completion, mo.callout("Annotation file exists but has no rows yet.", kind="warn")]
            )
        else:
            summary = pd.DataFrame(
                [
                    {"metric": "total_rows", "value": int(len(ann))},
                    {"metric": "unique_tiles", "value": int(ann["image_path"].nunique())},
                    {"metric": "annotators", "value": int(ann["annotator_id"].nunique())},
                ]
            )
            by_annotator = (
                ann.groupby("annotator_id", dropna=False)
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
            )
            recent = ann.tail(20)

            _panel_summary = mo.vstack(
                [
                    mo.md("## Annotation Summary"),
                    _completion,
                    mo.md("### Totals"),
                    summary,
                    mo.md("### By Annotator"),
                    by_annotator,
                    mo.md("### Recent (last 20 rows)"),
                    recent,
                ]
            )
    _panel_summary
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
