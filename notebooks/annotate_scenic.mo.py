import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    from dataclasses import dataclass
    from datetime import datetime, timezone
    from pathlib import Path

    import pandas as pd
    from PIL import Image

    return Image, Path, dataclass, datetime, mo, os, pd, timezone


@app.cell
def _(mo):
    mo.md("""
    # Manual Scenic Annotation
    Annotate tiles with human scenic ratings (0-10) for calibration and benchmark evaluation.
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
def _(Path, pd, ui_config_form):
    if ui_config_form.value is None:
        print("Submit 'Load Annotation Batch' to start.")
        config = {
            "labels_csv": "data/raw/labels.csv",
            "raw_dir": "data/raw",
            "annotations_csv": "data/raw/labels_human.csv",
            "sample_size": 500,
            "seed": 42,
            "stratify_by_class": True,
            "annotator_id": "annotator",
        }
    else:
        config = ui_config_form.value
    labels_path = Path(config["labels_csv"])
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.csv not found: {labels_path}")

    labels_df = pd.read_csv(labels_path)
    if labels_df.empty:
        raise ValueError("labels.csv is empty")
    if "image_path" not in labels_df.columns:
        raise ValueError("labels.csv must contain 'image_path'")

    labels_df = labels_df.dropna(subset=["image_path"]).copy()
    labels_df["image_path"] = labels_df["image_path"].astype(str)

    ann_path = Path(config["annotations_csv"])
    if ann_path.exists():
        ann_df = pd.read_csv(ann_path)
    else:
        ann_df = pd.DataFrame(
            columns=[
                "image_path",
                "scenic_human",
                "confidence",
                "skip",
                "annotator_id",
                "timestamp",
                "notes",
            ]
        )
    return ann_df, config, labels_df


@app.cell
def _(ann_df, config, labels_df, pd):
    def sample_stratified(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
        if n <= 0:
            return df.iloc[0:0].copy()
        if "class_id" not in df.columns or df["class_id"].isna().all():
            return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)

        classes = sorted(df["class_id"].dropna().unique().tolist())
        per_class = max(1, n // max(1, len(classes)))
        chunks = []
        for class_id in classes:
            class_df = df[df["class_id"] == class_id]
            if class_df.empty:
                continue
            take = min(per_class, len(class_df))
            chunks.append(class_df.sample(n=take, random_state=seed))

        sampled = pd.concat(chunks, ignore_index=True) if chunks else df.iloc[0:0].copy()
        needed = min(n, len(df)) - len(sampled)
        if needed > 0:
            remainder = df.loc[~df["image_path"].isin(sampled["image_path"])]
            if not remainder.empty:
                sampled = pd.concat(
                    [sampled, remainder.sample(n=min(needed, len(remainder)), random_state=seed)],
                    ignore_index=True,
                )
        return sampled.drop_duplicates(subset=["image_path"]).reset_index(drop=True)

    n = int(config["sample_size"])
    seed = int(config["seed"])
    stratify = bool(config["stratify_by_class"])

    done_paths = set(ann_df["image_path"].astype(str).tolist()) if not ann_df.empty else set()
    unlabeled_df = labels_df.loc[~labels_df["image_path"].isin(done_paths)].copy()
    if unlabeled_df.empty:
        batch_df = unlabeled_df
    elif stratify:
        batch_df = sample_stratified(unlabeled_df, n=n, seed=seed)
    else:
        batch_df = unlabeled_df.sample(n=min(n, len(unlabeled_df)), random_state=seed).reset_index(drop=True)
    return batch_df, done_paths


@app.cell
def _(batch_df, done_paths, mo):
    mo.md(f"Loaded batch: **{len(batch_df)}** unlabeled | Existing annotations: **{len(done_paths)}**")
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
            ui_notes,
            ui_save_click,
        ]
    )
    controls
    return current_index, ui_conf, ui_notes, ui_save_click, ui_score, ui_skip


@app.cell
def _(Image, Path, batch_df, config, current_index, mo):
    image_path = ""
    if batch_df.empty:
        _panel_current = mo.callout("No samples to annotate.", kind="warn")
    else:
        i = int(current_index())
        i = min(max(0, i), len(batch_df) - 1)
        row = batch_df.iloc[i]
        image_path = str(row["image_path"])
        image_abs = Path(config["raw_dir"]) / image_path

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
    Path,
    config,
    datetime,
    image_path,
    pd,
    timezone,
    ui_conf,
    ui_notes,
    ui_save_click,
    ui_score,
    ui_skip,
):
    save_clicked = ui_save_click.value() if callable(ui_save_click.value) else ui_save_click.value
    if bool(save_clicked) and image_path:
        out_path = Path(config["annotations_csv"])
        if out_path.exists():
            updated = pd.read_csv(out_path)
        else:
            updated = pd.DataFrame(
                columns=[
                    "image_path",
                    "scenic_human",
                    "confidence",
                    "skip",
                    "annotator_id",
                    "timestamp",
                    "notes",
                ]
            )
        record = {
            "image_path": image_path,
            "scenic_human": float(ui_score.value() if callable(ui_score.value) else ui_score.value),
            "confidence": str(ui_conf.value() if callable(ui_conf.value) else ui_conf.value),
            "skip": bool(ui_skip.value() if callable(ui_skip.value) else ui_skip.value),
            "annotator_id": str(config["annotator_id"]),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "notes": str(ui_notes.value() if callable(ui_notes.value) else ui_notes.value),
        }
        if not updated.empty:
            mask = (updated["image_path"].astype(str) == record["image_path"]) & (
                updated["annotator_id"].astype(str) == record["annotator_id"]
            )
            updated = updated.loc[~mask].copy()
        updated = pd.concat([updated, pd.DataFrame([record])], ignore_index=True)
        updated.to_csv(out_path, index=False)
        print(f"Saved annotation to {out_path} (rows={len(updated)})")
    return


@app.cell
def _(Path, config, mo, pd):
    _ann_path = Path(config["annotations_csv"])
    if not _ann_path.exists():
        _panel_summary = mo.callout(f"No annotations file yet: {_ann_path}", kind="warn")
    else:
        ann = pd.read_csv(_ann_path)
        if ann.empty:
            _panel_summary = mo.callout("Annotation file exists but has no rows yet.", kind="warn")
        else:
            summary = pd.DataFrame(
                [
                    {"metric": "total_rows", "value": int(len(ann))},
                    {"metric": "unique_tiles", "value": int(ann["image_path"].nunique())},
                    {
                        "metric": "annotators",
                        "value": int(ann["annotator_id"].nunique()) if "annotator_id" in ann.columns else 0,
                    },
                ]
            )
            by_annotator = (
                ann.groupby("annotator_id", dropna=False)
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
                if "annotator_id" in ann.columns
                else pd.DataFrame()
            )
            recent = ann.tail(20)

            _panel_summary = mo.vstack(
                [
                    mo.md("## Annotation Summary"),
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
