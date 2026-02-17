"""
Run a lightweight local web app for manual scenic annotation.

This mirrors the marimo notebook workflow, but provides a browser-first UI
with procedural navigation (prev/next/go) and CSV-backed persistence.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd


DEFAULT_COLUMNS = [
    "image_path",
    "scenic_human",
    "confidence",
    "skip",
    "annotator_id",
    "timestamp",
    "notes",
]


@dataclass
class AnnotatorConfig:
    labels_csv: str = "data/raw/labels.csv"
    batch_csv: str = ""
    raw_dir: str = "data/raw"
    annotations_csv: str = "data/raw/labels_human.csv"
    sample_size: int = 500
    seed: int = 42
    stratify_by_class: bool = True
    annotator_id: str = os.getenv("USER", "annotator")


class AnnotatorState:
    def __init__(self, config: AnnotatorConfig):
        self._lock = threading.Lock()
        self.config = config
        self.batch: list[dict[str, Any]] = []

    def load_batch(self, config_data: dict[str, Any]) -> dict[str, Any]:
        config = AnnotatorConfig(
            labels_csv=_expand_vars(str(config_data.get("labels_csv", self.config.labels_csv))),
            batch_csv=_expand_vars(str(config_data.get("batch_csv", self.config.batch_csv))),
            raw_dir=_expand_vars(str(config_data.get("raw_dir", self.config.raw_dir))),
            annotations_csv=_expand_vars(str(config_data.get("annotations_csv", self.config.annotations_csv))),
            sample_size=max(1, int(config_data.get("sample_size", self.config.sample_size))),
            seed=int(config_data.get("seed", self.config.seed)),
            stratify_by_class=_to_bool(config_data.get("stratify_by_class", self.config.stratify_by_class)),
            annotator_id=str(config_data.get("annotator_id", self.config.annotator_id)),
        )

        labels_df = pd.DataFrame(columns=["image_path"])
        labels_path = Path(config.labels_csv)
        if labels_path.exists():
            labels_df = pd.read_csv(labels_path)
            if "image_path" not in labels_df.columns:
                raise ValueError("labels.csv must contain 'image_path'")
            labels_df = labels_df.dropna(subset=["image_path"]).copy()
            labels_df["image_path"] = labels_df["image_path"].astype(str)
        elif not config.batch_csv.strip():
            raise FileNotFoundError(f"labels.csv not found: {labels_path}")

        batch_source = "labels_csv"
        candidate_df = labels_df
        if config.batch_csv.strip():
            batch_path = Path(config.batch_csv)
            if not batch_path.exists():
                raise FileNotFoundError(f"batch CSV not found: {batch_path}")
            batch_df = pd.read_csv(batch_path)
            if batch_df.empty:
                raise ValueError("batch CSV is empty")
            if "image_path" not in batch_df.columns:
                raise ValueError("batch CSV must contain 'image_path'")
            batch_df = batch_df.dropna(subset=["image_path"]).copy()
            batch_df["image_path"] = batch_df["image_path"].astype(str)
            candidate_df = batch_df.drop_duplicates(subset=["image_path"], keep="first")

            label_cols = [c for c in labels_df.columns if c != "image_path" and c not in candidate_df.columns]
            if label_cols:
                candidate_df = candidate_df.merge(
                    labels_df[["image_path", *label_cols]],
                    on="image_path",
                    how="left",
                )
            batch_source = "batch_csv"
        elif candidate_df.empty:
            raise ValueError("labels.csv is empty")

        ann_df = _read_annotations(Path(config.annotations_csv))
        if ann_df.empty:
            done_paths_for_annotator: set[str] = set()
            done_paths_all: set[str] = set()
        else:
            ann_df["image_path"] = ann_df["image_path"].astype(str)
            ann_df["annotator_id"] = ann_df["annotator_id"].astype(str)
            done_paths_for_annotator = set(
                ann_df.loc[ann_df["annotator_id"] == config.annotator_id, "image_path"].tolist()
            )
            done_paths_all = set(ann_df["image_path"].tolist())
        unlabeled_df = candidate_df.loc[~candidate_df["image_path"].isin(done_paths_for_annotator)].copy()

        if unlabeled_df.empty:
            batch_df = unlabeled_df
        elif config.stratify_by_class:
            batch_df = _sample_stratified(unlabeled_df, n=config.sample_size, seed=config.seed)
        else:
            batch_df = (
                unlabeled_df.sample(n=min(config.sample_size, len(unlabeled_df)), random_state=config.seed)
                .reset_index(drop=True)
            )

        with self._lock:
            self.config = config
            self.batch = batch_df.to_dict(orient="records")

        return {
            "config": asdict(config),
            "batch_size": len(self.batch),
            "batch_source": batch_source,
            "candidate_pool": int(len(candidate_df)),
            "existing_annotations": int(len(done_paths_for_annotator)),
            "existing_annotations_total": int(len(done_paths_all)),
            "remaining_unlabeled": int(len(unlabeled_df)),
            "batch": self.batch,
        }

    def save_annotation(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            config = self.config

        image_path = str(payload.get("image_path", "")).strip()
        if not image_path:
            raise ValueError("image_path is required")

        score = float(payload.get("scenic_human"))
        if score < 0.0 or score > 10.0:
            raise ValueError("scenic_human must be in [0, 10]")

        record = {
            "image_path": image_path,
            "scenic_human": score,
            "confidence": str(payload.get("confidence", "medium")),
            "skip": _to_bool(payload.get("skip", False)),
            "annotator_id": str(payload.get("annotator_id", config.annotator_id)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "notes": str(payload.get("notes", "")),
        }

        out_path = Path(config.annotations_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ann_df = _read_annotations(out_path)
        if not ann_df.empty:
            mask = (ann_df["image_path"].astype(str) == record["image_path"]) & (
                ann_df["annotator_id"].astype(str) == record["annotator_id"]
            )
            ann_df = ann_df.loc[~mask].copy()
        ann_df = pd.concat([ann_df, pd.DataFrame([record])], ignore_index=True)
        ann_df.to_csv(out_path, index=False)

        return {
            "saved": True,
            "annotations_csv": str(out_path),
            "rows": int(len(ann_df)),
            "record": record,
        }

    def summary(self) -> dict[str, Any]:
        with self._lock:
            config = self.config
            batch_size = len(self.batch)

        ann_path = Path(config.annotations_csv)
        if not ann_path.exists():
            return {
                "exists": False,
                "annotations_csv": str(ann_path),
                "batch_size": batch_size,
                "totals": {"total_rows": 0, "unique_tiles": 0, "annotators": 0},
                "by_annotator": [],
                "recent": [],
            }

        ann = _read_annotations(ann_path)
        totals = {
            "total_rows": int(len(ann)),
            "unique_tiles": int(ann["image_path"].nunique()) if "image_path" in ann.columns else 0,
            "annotators": int(ann["annotator_id"].nunique()) if "annotator_id" in ann.columns else 0,
        }

        by_annotator = []
        if "annotator_id" in ann.columns:
            by_annotator = (
                ann.groupby("annotator_id", dropna=False)
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
                .to_dict(orient="records")
            )

        return {
            "exists": True,
            "annotations_csv": str(ann_path),
            "batch_size": batch_size,
            "totals": totals,
            "by_annotator": by_annotator,
            "recent": ann.tail(20).to_dict(orient="records"),
        }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _expand_vars(text: str) -> str:
    return os.path.expandvars(text.strip())


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "t"}


def _read_annotations(path: Path) -> pd.DataFrame:
    if path.exists():
        ann_df = pd.read_csv(path)
    else:
        ann_df = pd.DataFrame(columns=DEFAULT_COLUMNS)
    for col in DEFAULT_COLUMNS:
        if col not in ann_df.columns:
            ann_df[col] = None
    return ann_df[DEFAULT_COLUMNS].copy()


def _sample_stratified(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
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


def _parse_s3(raw_dir: str) -> tuple[str, str] | None:
    if not raw_dir.startswith("s3://"):
        return None
    rest = raw_dir.replace("s3://", "", 1)
    parts = rest.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix


def _serve_image_bytes(raw_dir: str, image_path: str, *, s3_client: Any | None) -> tuple[bytes, str]:
    s3_info = _parse_s3(raw_dir)
    if s3_info is not None:
        client = s3_client
        if client is None:
            try:
                import boto3
            except ImportError as exc:
                raise RuntimeError("boto3 is required for s3:// raw_dir") from exc
            client = boto3.client("s3")
        bucket, prefix = s3_info
        key = f"{prefix.rstrip('/')}/{image_path.lstrip('/')}" if prefix else image_path.lstrip("/")
        obj = client.get_object(Bucket=bucket, Key=key)
        body = obj["Body"].read()
        content_type = obj.get("ContentType") or mimetypes.guess_type(image_path)[0] or "application/octet-stream"
        return body, content_type

    root = Path(raw_dir).resolve()
    path = (root / image_path).resolve()
    if root not in path.parents and path != root:
        raise ValueError("Invalid image path outside raw_dir")
    if not path.exists():
        raise FileNotFoundError(path)

    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return path.read_bytes(), content_type


def make_handler(state: AnnotatorState, s3_client: Any | None):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ScenicAnnotator/0.1"

        def _send_json(self, obj: dict[str, Any], status: int = 200) -> None:
            data = json.dumps(_json_ready(obj), allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                html = HTML_PAGE.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
                return

            if parsed.path == "/api/default-config":
                self._send_json({"config": asdict(state.config)})
                return

            if parsed.path == "/api/summary":
                self._send_json(state.summary())
                return

            if parsed.path == "/api/image":
                query = parse_qs(parsed.query)
                image_path = (query.get("image_path") or [""])[0]
                raw_dir = (query.get("raw_dir") or [state.config.raw_dir])[0]
                if not image_path:
                    self._send_json({"error": "image_path is required"}, status=400)
                    return
                try:
                    body, content_type = _serve_image_bytes(raw_dir, image_path, s3_client=s3_client)
                except FileNotFoundError:
                    self._send_json({"error": f"Image not found: {image_path}"}, status=404)
                    return
                except Exception as exc:
                    self._send_json({"error": str(exc)}, status=400)
                    return

                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            self._send_json({"error": "Not found"}, status=404)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                payload = self._read_json_body()
            except json.JSONDecodeError:
                self._send_json({"error": "Invalid JSON"}, status=400)
                return

            if parsed.path == "/api/load-batch":
                try:
                    result = state.load_batch(payload.get("config", {}))
                except Exception as exc:
                    self._send_json({"error": str(exc)}, status=400)
                    return
                self._send_json(result)
                return

            if parsed.path == "/api/save-annotation":
                try:
                    result = state.save_annotation(payload)
                except Exception as exc:
                    self._send_json({"error": str(exc)}, status=400)
                    return
                self._send_json(result)
                return

            self._send_json({"error": "Not found"}, status=404)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    return Handler


HTML_PAGE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Scenic Annotator</title>
  <style>
    :root {
      --bg: #f3f5f7;
      --panel: #ffffff;
      --text: #0f172a;
      --muted: #475569;
      --accent: #0f766e;
      --border: #dbe2ea;
    }
    body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background: var(--bg); color: var(--text); }
    .wrap { max-width: 1300px; margin: 0 auto; padding: 16px; display: grid; gap: 12px; }
    .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 12px; }
    .grid { display: grid; gap: 12px; grid-template-columns: 380px 1fr; }
    .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    label { font-size: 12px; color: var(--muted); display: block; margin-bottom: 4px; }
    input, select, textarea, button { width: 100%; box-sizing: border-box; border: 1px solid var(--border); border-radius: 8px; padding: 8px; font-size: 14px; }
    textarea { min-height: 80px; resize: vertical; }
    button { background: #fff; cursor: pointer; }
    button.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    .row { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 8px; }
    .meta { white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; color: #1e293b; background: #f8fafc; border: 1px solid var(--border); border-radius: 8px; padding: 8px; }
    .status { font-size: 13px; color: #0f172a; margin-top: 6px; }
    .muted { color: var(--muted); }
    img { width: 100%; max-height: 76vh; object-fit: contain; background: #0b1220; border-radius: 8px; border: 1px solid var(--border); }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { border: 1px solid var(--border); padding: 6px; text-align: left; }
    @media (max-width: 1100px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="panel">
      <h2 style="margin:0 0 8px 0;">Scenic Annotation Web UI</h2>
      <div class="form-grid">
        <div><label>Labels CSV</label><input id="labels_csv" /></div>
        <div><label>Batch CSV (optional)</label><input id="batch_csv" placeholder="data/processed/regression/overlap_batch.csv" /></div>
        <div><label>Raw Dir (local or s3://...)</label><input id="raw_dir" /></div>
        <div><label>Annotations CSV</label><input id="annotations_csv" /></div>
        <div><label>Annotator ID</label><input id="annotator_id" /></div>
        <div><label>Sample Size</label><input id="sample_size" type="number" min="1" step="1" /></div>
        <div><label>Seed</label><input id="seed" type="number" step="1" /></div>
        <div><label>Stratify by class_id</label><select id="stratify_by_class"><option value="true">true</option><option value="false">false</option></select></div>
      </div>
      <div style="margin-top:10px; display:flex; gap:8px;">
        <button class="primary" id="load_btn">Load Annotation Batch</button>
        <button id="refresh_summary_btn">Refresh Summary</button>
      </div>
      <div class="status" id="load_status"></div>
    </div>

    <div class="grid">
      <div class="panel">
        <h3 style="margin-top:0;">Navigation</h3>
        <div class="row">
          <button id="prev_btn">Previous</button>
          <button id="next_btn">Next</button>
          <input id="go_idx" type="number" min="0" step="1" />
          <button id="go_btn">Go</button>
        </div>
        <div class="status" id="progress"></div>

        <h3>Annotation</h3>
        <label>Scenic Score (0-10)</label>
        <input id="score" type="number" min="0" max="10" step="0.1" value="5.0" />
        <label>Confidence</label>
        <select id="confidence"><option>high</option><option selected>medium</option><option>low</option></select>
        <label>Skip</label>
        <select id="skip"><option value="false" selected>false</option><option value="true">true</option></select>
        <label>Notes</label>
        <textarea id="notes"></textarea>
        <button class="primary" id="save_btn">Save Annotation</button>
        <div class="status" id="save_status"></div>

        <h3>Tile Metadata</h3>
        <div class="meta" id="meta">Load a batch to begin.</div>
      </div>

      <div class="panel">
        <img id="tile_img" alt="tile preview" />
      </div>
    </div>

    <div class="panel">
      <h3 style="margin-top:0;">Annotation Summary</h3>
      <div class="status" id="summary_status"></div>
      <table id="summary_table"></table>
    </div>
  </div>

  <script>
    let batch = [];
    let idx = 0;

    const cfgIds = [
      "labels_csv", "batch_csv", "raw_dir", "annotations_csv", "sample_size", "seed", "stratify_by_class", "annotator_id"
    ];

    function readConfig() {
      return {
        labels_csv: document.getElementById("labels_csv").value.trim(),
        batch_csv: document.getElementById("batch_csv").value.trim(),
        raw_dir: document.getElementById("raw_dir").value.trim(),
        annotations_csv: document.getElementById("annotations_csv").value.trim(),
        sample_size: Number(document.getElementById("sample_size").value || 500),
        seed: Number(document.getElementById("seed").value || 42),
        stratify_by_class: document.getElementById("stratify_by_class").value === "true",
        annotator_id: document.getElementById("annotator_id").value.trim() || "annotator",
      };
    }

    function writeConfig(cfg) {
      for (const id of cfgIds) {
        const el = document.getElementById(id);
        if (el && cfg[id] !== undefined) el.value = String(cfg[id]);
      }
    }

    function currentTile() {
      if (!batch.length) return null;
      if (idx < 0) idx = 0;
      if (idx >= batch.length) idx = batch.length - 1;
      return batch[idx];
    }

    function render() {
      const tile = currentTile();
      const progress = document.getElementById("progress");
      const meta = document.getElementById("meta");
      const img = document.getElementById("tile_img");
      const goIdx = document.getElementById("go_idx");
      goIdx.max = Math.max(0, batch.length - 1);
      goIdx.value = String(idx);
      if (!tile) {
        progress.textContent = "No batch loaded.";
        meta.textContent = "Load a batch to begin.";
        img.removeAttribute("src");
        return;
      }
      progress.textContent = `Index ${idx + 1}/${batch.length} | image_path: ${tile.image_path}`;
      const lines = Object.entries(tile).map(([k, v]) => `${k}: ${v}`);
      meta.textContent = lines.join("\\n");
      const rawDir = encodeURIComponent(document.getElementById("raw_dir").value.trim());
      const imagePath = encodeURIComponent(tile.image_path);
      img.src = `/api/image?raw_dir=${rawDir}&image_path=${imagePath}&ts=${Date.now()}`;
    }

    async function loadDefaults() {
      const res = await fetch("/api/default-config");
      const data = await res.json();
      if (data.config) writeConfig(data.config);
    }

    async function loadBatch() {
      const loadStatus = document.getElementById("load_status");
      loadStatus.textContent = "Loading batch...";
      const res = await fetch("/api/load-batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ config: readConfig() }),
      });
      const data = await res.json();
      if (!res.ok) {
        loadStatus.textContent = `Error: ${data.error || "failed to load"}`;
        return;
      }
      writeConfig(data.config || {});
      batch = data.batch || [];
      idx = 0;
      loadStatus.textContent = `Loaded ${data.batch_size} tiles from ${data.batch_source || "labels_csv"} | Candidate pool: ${data.candidate_pool || 0} | Existing (this annotator): ${data.existing_annotations || 0} | Existing (all annotators): ${data.existing_annotations_total || 0} | Remaining for this annotator: ${data.remaining_unlabeled || 0}`;
      render();
      refreshSummary();
    }

    async function saveAnnotation() {
      const tile = currentTile();
      if (!tile) return;
      const payload = {
        image_path: tile.image_path,
        scenic_human: Number(document.getElementById("score").value),
        confidence: document.getElementById("confidence").value,
        skip: document.getElementById("skip").value === "true",
        notes: document.getElementById("notes").value,
        annotator_id: document.getElementById("annotator_id").value.trim() || "annotator",
      };
      const saveStatus = document.getElementById("save_status");
      saveStatus.textContent = "Saving...";
      const res = await fetch("/api/save-annotation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) {
        saveStatus.textContent = `Error: ${data.error || "save failed"}`;
        return;
      }
      saveStatus.textContent = `Saved (${data.rows} rows).`;
      if (idx < batch.length - 1) idx += 1;
      render();
      refreshSummary();
    }

    async function refreshSummary() {
      const el = document.getElementById("summary_status");
      const table = document.getElementById("summary_table");
      el.textContent = "Loading summary...";
      const res = await fetch("/api/summary");
      const data = await res.json();
      if (!res.ok) {
        el.textContent = `Error: ${data.error || "summary failed"}`;
        return;
      }
      const totals = data.totals || {};
      const rows = [
        ["annotations_csv", data.annotations_csv || ""],
        ["batch_size", data.batch_size || 0],
        ["total_rows", totals.total_rows || 0],
        ["unique_tiles", totals.unique_tiles || 0],
        ["annotators", totals.annotators || 0],
      ];
      table.innerHTML = "<tr><th>Metric</th><th>Value</th></tr>" + rows.map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join("");
      el.textContent = data.exists ? "Summary loaded." : "No annotation file yet.";
    }

    document.getElementById("load_btn").addEventListener("click", loadBatch);
    document.getElementById("refresh_summary_btn").addEventListener("click", refreshSummary);
    document.getElementById("save_btn").addEventListener("click", saveAnnotation);
    document.getElementById("prev_btn").addEventListener("click", () => { idx = Math.max(0, idx - 1); render(); });
    document.getElementById("next_btn").addEventListener("click", () => { idx = Math.min(Math.max(0, batch.length - 1), idx + 1); render(); });
    document.getElementById("go_btn").addEventListener("click", () => {
      const n = Number(document.getElementById("go_idx").value || 0);
      idx = Math.max(0, Math.min(Math.max(0, batch.length - 1), n));
      render();
    });

    loadDefaults().then(refreshSummary);
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local web annotator for scenic tiles")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--labels-csv", type=str, default="data/raw/labels.csv")
    parser.add_argument("--batch-csv", type=str, default="")
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--annotations-csv", type=str, default="data/raw/labels_human.csv")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratify-by-class", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--annotator-id", type=str, default=os.getenv("USER", "annotator"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AnnotatorConfig(
        labels_csv=_expand_vars(args.labels_csv),
        batch_csv=_expand_vars(args.batch_csv),
        raw_dir=_expand_vars(args.raw_dir),
        annotations_csv=_expand_vars(args.annotations_csv),
        sample_size=args.sample_size,
        seed=args.seed,
        stratify_by_class=args.stratify_by_class,
        annotator_id=args.annotator_id,
    )

    s3_client = None
    if config.raw_dir.startswith("s3://"):
        try:
            import boto3
        except ImportError as exc:
            raise ImportError("boto3 is required when raw_dir uses s3://") from exc
        s3_client = boto3.client("s3")

    Path(config.annotations_csv).parent.mkdir(parents=True, exist_ok=True)
    state = AnnotatorState(config)
    handler_cls = make_handler(state, s3_client=s3_client)
    server = ThreadingHTTPServer((args.host, args.port), handler_cls)
    print(f"Scenic annotator running on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
