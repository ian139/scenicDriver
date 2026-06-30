import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    """Header + marimo import."""
    import marimo as mo

    mo.md(
        """
        # Heuristic Labeling + Report UI
        Use this notebook to run per-region heuristic labeling, generate reports,
        and launch the local report viewer.
        """
    )
    return (mo,)


@app.cell
def _():
    """Imports + helpers."""
    from pathlib import Path
    import subprocess
    import shlex
    import re
    import os

    def run_command(cmd: str) -> tuple[int, str]:
        proc = subprocess.run(
            shlex.split(cmd),
            capture_output=True,
            text=True,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, output.strip()

    def find_regions() -> list[str]:
        root = Path("data/raw/images/satellite/z16")
        if not root.exists():
            return []
        return sorted([p.name for p in root.iterdir() if p.is_dir()])

    def extract_run_name(output: str) -> str | None:
        match = re.search(r"heuristic_runs/([^/]+)/report", output)
        return match.group(1) if match else None

    def normalize_region(name: str) -> str:
        return re.sub(r"[^a-z0-9_\\-]", "_", name.strip().lower())
    return extract_run_name, find_regions, normalize_region, os, run_command


@app.cell
def _(find_regions, mo):
    """Region selection."""
    regions = find_regions()
    region_dropdown = mo.ui.dropdown(
        label="Region",
        options=regions,
        value=regions[0] if regions else None,
    )
    region_custom = mo.ui.text(label="Or enter a new region name", value="")

    mo.md("## Region Selection")
    if not regions:
        mo.callout(
            "No regions found under data/raw/images/satellite/z16. "
            "Use the Download Tiles section to create one.",
            kind="warn",
        )
    mo.vstack([region_dropdown, region_custom])
    return region_custom, region_dropdown


@app.cell
def _(mo):
    """Heuristic report options."""
    preview = mo.ui.checkbox(label="Preview (limit tiles)", value=True)
    report_max_tiles = mo.ui.number(label="Max tiles (optional)", value=None, step=1)
    write_raw_labels = mo.ui.checkbox(label="Write data/raw/labels.csv", value=False)
    device = mo.ui.dropdown(label="Device", options=["auto", "cpu", "cuda"], value="auto")

    mo.md("## Report Options")
    mo.vstack([preview, report_max_tiles, write_raw_labels, device])
    return device, preview, report_max_tiles, write_raw_labels


@app.cell
def _(mo):
    """Download options."""
    dl_region = mo.ui.text(label="Region name", value="")
    min_lat = mo.ui.number(label="Min lat", value=None)
    min_lon = mo.ui.number(label="Min lon", value=None)
    max_lat = mo.ui.number(label="Max lat", value=None)
    max_lon = mo.ui.number(label="Max lon", value=None)
    zoom = mo.ui.number(label="Zoom", value=16, step=1)
    download_max_tiles = mo.ui.number(label="Max tiles (optional)", value=None, step=1)

    mo.md("## Download Tiles (Optional)")
    mo.md("Requires `MAPBOX_ACCESS_TOKEN` in your environment.")
    mo.vstack([dl_region, min_lat, min_lon, max_lat, max_lon, zoom, download_max_tiles])
    return (
        dl_region,
        download_max_tiles,
        max_lat,
        max_lon,
        min_lat,
        min_lon,
        zoom,
    )


@app.cell
def _(mo):
    """Actions."""
    port = mo.ui.number(label="Viewer port", value=8001, step=1)

    mo.md("## Actions")
    mo.vstack([port])
    return (port,)


@app.cell
def _(
    device,
    dl_region,
    download_max_tiles,
    extract_run_name,
    max_lat,
    max_lon,
    min_lat,
    min_lon,
    mo,
    normalize_region,
    os,
    port,
    preview,
    region_custom,
    region_dropdown,
    report_max_tiles,
    run_command,
    write_raw_labels,
    zoom,
):
    """Execution + output (form-driven)."""
    log_text = "Idle."
    run_name = ""

    region_custom_val = region_custom.value() if callable(region_custom.value) else region_custom.value
    region_dropdown_val = region_dropdown.value() if callable(region_dropdown.value) else region_dropdown.value
    region_raw = (region_custom_val or "").strip() or region_dropdown_val or ""
    region = normalize_region(region_raw)

    report_form = mo.ui.form(
        [region_dropdown, region_custom, preview, report_max_tiles, write_raw_labels, device],
        label="Run Heuristic Report",
    )

    download_style = mo.ui.dropdown(
        label="Download style",
        options=["mapbox.satellite", "mapbox.terrain-rgb"],
        value="mapbox.satellite",
    )
    download_form = mo.ui.form(
        [dl_region, min_lat, min_lon, max_lat, max_lon, zoom, download_max_tiles, download_style],
        label="Download Tiles",
    )

    mo.md("## Run Report")
    mo.vstack([report_form])
    mo.md("## Download Tiles")
    mo.vstack([download_form])

    if report_form.value is not None:
        if not region:
            log_text = "Run report requires a region selection."
        else:
            if region != region_raw and region_raw:
                log_text = f"Normalized region name to '{region}'."
            cmd_parts = [
                "uv",
                "run",
                "python",
                "scripts/heuristic_report_region.py",
                "--region",
                region,
            ]
            if preview.value:
                cmd_parts.append("--preview")
            if report_max_tiles.value is not None:
                cmd_parts.extend(["--max-tiles", str(int(report_max_tiles.value))])
            if write_raw_labels.value:
                cmd_parts.append("--write-raw-labels")
            if device.value:
                cmd_parts.extend(["--device", device.value])

            cmd = " ".join(cmd_parts)
            code, output = run_command(cmd)
            log_text = f"$ {cmd}\n\n{output}"
            if code == 0:
                extracted = extract_run_name(output)
                if extracted:
                    run_name = extracted

    if download_form.value is not None:
        dl_region_val = dl_region.value() if callable(dl_region.value) else dl_region.value or ""
        if not dl_region_val:
            log_text = "Download requires a region name."
        elif not os.environ.get("MAPBOX_ACCESS_TOKEN"):
            log_text = "MAPBOX_ACCESS_TOKEN is not set in the environment."
        elif None in (min_lat.value, min_lon.value, max_lat.value, max_lon.value):
            log_text = "Download requires min/max lat/lon."
        else:
            dl_region_clean = normalize_region(dl_region_val)
            style = download_style.value() if callable(download_style.value) else download_style.value
            out_root = (
                "data/raw/images/satellite/z16"
                if style == "mapbox.satellite"
                else "data/raw/images/terrain/z16"
            )
            out_dir = f"{out_root}/{dl_region_clean}"
            cmd = (
                "uv run python scripts/download_bbox_tiles.py "
                f"--min-lat {min_lat.value} --min-lon {min_lon.value} "
                f"--max-lat {max_lat.value} --max-lon {max_lon.value} "
                f"--zoom {int(zoom.value)} --style {style} --output {out_dir}"
            )
            if download_max_tiles.value is not None:
                cmd += f" --max-tiles {int(download_max_tiles.value)}"
            code, output = run_command(cmd)
            log_text = f"$ {cmd}\n\n{output}"

    if run_name:
        server_cmd = (
            "uv run python scripts/heuristic_report_server.py "
            f"--run-name {run_name} --no-open --port {int(port.value)}"
        )
        server_url = f"http://127.0.0.1:{int(port.value)}/index.html"
    else:
        server_cmd = "Run a report to get the viewer command."
        server_url = ""

    mo.md("## Output")
    mo.vstack(
        [
            mo.md("### Last Command Output"),
            mo.ui.text_area(value=str(log_text or ""), label="log", rows=12),
            mo.md("### Last Run Name"),
            mo.ui.text(value=str(run_name or "")),
            mo.md("### Viewer Command"),
            mo.ui.text(value=server_cmd),
            mo.md("### Viewer URL"),
            mo.ui.text(value=server_url),
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
