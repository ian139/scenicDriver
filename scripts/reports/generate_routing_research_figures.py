"""Generate deterministic, accessible routing-performance research figures.

The only input is docs/assets/research/routing/figure-data.json.  The renderer
uses the Python standard library and emits six self-contained SVG files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "docs/assets/research/routing/figure-data.json"
OUTPUT_DIR = ROOT / "docs/assets/research/routing"
CANONICAL_FILENAMES = (
    "system-boundary.svg",
    "autoresearch-loop.svg",
    "experiment-outcomes.svg",
    "latency-distributions.svg",
    "profile-bottlenecks.svg",
    "two-worker-execution.svg",
)
WIDTH = 1200
HEIGHT = 720

COLORS = {
    "ink": "#1f2933",
    "muted": "#52616b",
    "paper": "#f7f5ef",
    "panel": "#fffdf8",
    "line": "#c8c4ba",
    "grid": "#e1ddd3",
    "navy": "#315b7d",
    "teal": "#287271",
    "orange": "#b85c38",
    "gold": "#9b6c16",
    "red": "#a63d40",
    "gray": "#71808c",
    "white": "#fffdf8",
}
FONT = "Georgia, 'Times New Roman', serif"
SANS = "Arial, Helvetica, sans-serif"


def load_data(path: Path = DATA_PATH) -> dict[str, Any]:
    """Load and validate the auditable research figure input."""
    with path.open(encoding="utf-8") as source:
        data = json.load(source)
    validate_data(data)
    return data


def validate_data(data: dict[str, Any]) -> None:
    """Reject incomplete or misleading research figure data."""
    if data.get("schema_version") != 1:
        raise ValueError("figure data must declare schema_version 1")
    evidence = data.get("source", {}).get("evidence")
    if not isinstance(evidence, dict) or not evidence:
        raise ValueError("figure data must contain source evidence references")
    figures = data.get("figures")
    expected = {name.removesuffix(".svg") for name in CANONICAL_FILENAMES}
    if not isinstance(figures, dict) or set(figures) != expected:
        raise ValueError("figure data must define every canonical figure exactly once")
    for name, figure in figures.items():
        if not figure.get("title") or not figure.get("description"):
            raise ValueError(f"{name} must have a title and description")
        reference = figure.get("evidence")
        if reference not in evidence:
            raise ValueError(f"{name} references unknown evidence {reference!r}")
    for key in ("target", "request", "cache_policy", "host"):
        reference = data["study"][key].get("evidence")
        if reference not in evidence:
            raise ValueError(f"study {key} lacks a valid evidence reference")
    experiments = figures["experiment-outcomes"]["experiments"]
    if any(item["status"] == "timeout" and "median_seconds" in item for item in experiments):
        raise ValueError("timeouts must be categorical, not measured latency values")
    classes = {item["class"] for item in figures["latency-distributions"]["series"]}
    required_classes = {"uncached", "accepted_candidate", "final_confirmation", "response_cache_hit"}
    if not required_classes.issubset(classes):
        raise ValueError("latency evidence must distinguish cache and confirmation contexts")
    if data["study"]["target"]["met"]:
        raise ValueError("the documented uncached target was not met")


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def _text(x: float, y: float, text: str, *, size: int = 16, fill: str = COLORS["ink"],
          weight: str = "normal", anchor: str = "start", family: str = SANS,
          italic: bool = False) -> str:
    style = "font-style:italic;" if italic else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'style="font-family:{family};font-size:{size}px;font-weight:{weight};'
        f'fill:{fill};{style}">{escape(text)}</text>'
    )


def _rect(x: float, y: float, width: float, height: float, *, fill: str, stroke: str = "none",
          radius: float = 0, extra: str = "") -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'rx="{radius:.1f}" fill="{fill}" stroke="{stroke}" {extra}/>'
    )


def _line(x1: float, y1: float, x2: float, y2: float, *, stroke: str = COLORS["line"],
          width: float = 1.5, dash: str = "", marker: str = "") -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    marker_attr = f' marker-end="url(#{marker})"' if marker else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width:.1f}"{dash_attr}{marker_attr}/>'
    )


def _circle(cx: float, cy: float, radius: float, *, fill: str, stroke: str = "none", extra: str = "") -> str:
    return (
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" fill="{fill}" '
        f'stroke="{stroke}" {extra}/>'
    )


def _title_block(title: str, subtitle: str) -> list[str]:
    return [
        _text(64, 66, title, size=29, weight="bold", family=FONT),
        _text(64, 96, subtitle, size=15, fill=COLORS["muted"]),
        _line(64, 116, 1136, 116, stroke=COLORS["line"]),
    ]


def _notes(data: dict[str, Any], figure: dict[str, Any]) -> list[str]:
    study = data["study"]
    return [
        _rect(64, 650, 1072, 42, fill=COLORS["paper"], stroke=COLORS["line"], radius=4),
        _text(80, 675, f"Source: {data['source']['document']} ({figure['evidence']}).", size=12, fill=COLORS["muted"]),
        _text(1120, 675, study["host"]["label"], size=12, fill=COLORS["muted"], anchor="end"),
    ]


def _svg(filename: str, figure: dict[str, Any], body: list[str], data: dict[str, Any]) -> str:
    title_id = f"{filename}-title"
    description_id = f"{filename}-desc"
    defs = (
        "<defs>"
        "<pattern id=\"diagonal-hatch\" width=\"8\" height=\"8\" patternUnits=\"userSpaceOnUse\" patternTransform=\"rotate(45)\">"
        f"<line x1=\"0\" y1=\"0\" x2=\"0\" y2=\"8\" stroke=\"{COLORS['ink']}\" stroke-width=\"2\"/></pattern>"
        "<marker id=\"arrow\" markerWidth=\"9\" markerHeight=\"9\" refX=\"8\" refY=\"4.5\" orient=\"auto\">"
        f"<path d=\"M0,0 L9,4.5 L0,9 z\" fill=\"{COLORS['ink']}\"/></marker>"
        "</defs>"
    )
    return "\n".join([
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="{title_id} {description_id}">',
        f"<title id=\"{title_id}\">{escape(figure['title'])}</title>",
        f"<desc id=\"{description_id}\">{escape(figure['description'])}</desc>",
        defs,
        _rect(0, 0, WIDTH, HEIGHT, fill=COLORS["panel"]),
        '<g aria-label="Research figure">',
        *body,
        *_notes(data, figure),
        "</g>",
        "</svg>",
        "",
    ])


def render_system_boundary(data: dict[str, Any]) -> str:
    figure = data["figures"]["system-boundary"]
    study = data["study"]
    body = _title_block(figure["title"], "A complete request is the unit of the uncached research claim.")
    boxes = [
        (80, 180, 212, 120, "Request", "Burlington → Pittsburgh", COLORS["navy"]),
        (336, 180, 212, 120, "Service planner", "normal scenic + baseline path", COLORS["teal"]),
        (592, 180, 212, 120, "Native search", "four multiplier searches", COLORS["orange"]),
        (848, 180, 212, 120, "Semantic oracle", "complete response comparison", COLORS["gold"]),
    ]
    for x, y, w, h, label, detail, color in boxes:
        body.extend([_rect(x, y, w, h, fill=COLORS["paper"], stroke=color, radius=6), _rect(x, y, 9, h, fill=color),
                     _text(x + 26, y + 48, label, size=18, weight="bold"), _text(x + 26, y + 77, detail, size=13, fill=COLORS["muted"])])
    for x in (292, 548, 804):
        body.append(_line(x + 8, 240, x + 36, 240, stroke=COLORS["ink"], width=2, marker="arrow"))
    body.extend([
        _rect(80, 356, 700, 156, fill="#eef4f2", stroke=COLORS["teal"], radius=6),
        _text(104, 392, "UNCACHED SEARCH EVIDENCE", size=14, fill=COLORS["teal"], weight="bold"),
        _text(104, 425, "Complete response cache cleared before every run", size=21, weight="bold"),
        _text(104, 455, f"Target: median complete plan_routes request < {_fmt(study['target']['threshold_seconds'])} s — not met", size=16, fill=COLORS["orange"]),
        _text(104, 485, f"Deadline: {_fmt(study['request']['deadline_seconds'])} s; preload is measured separately", size=14, fill=COLORS["muted"]),
        _rect(816, 356, 244, 156, fill=COLORS["paper"], stroke=COLORS["gray"], radius=6),
        _text(840, 392, "EXCLUDED FROM CLAIM", size=14, fill=COLORS["gray"], weight="bold"),
        _circle(848, 431, 8, fill=COLORS["gray"]),
        _text(868, 436, "Preload", size=16),
        _circle(848, 470, 8, fill=COLORS["gray"], extra='stroke-dasharray="2 2"'),
        _text(868, 475, "Response-cache hit", size=16),
        _text(840, 499, "operational evidence only", size=12, fill=COLORS["muted"], italic=True),
    ])
    return _svg("system-boundary", figure, body, data)


def render_autoresearch_loop(data: dict[str, Any]) -> str:
    figure = data["figures"]["autoresearch-loop"]
    body = _title_block(figure["title"], "One immutable baseline, eleven measured candidates, and a fixed stop condition.")
    steps = [
        (84, "Immutable baseline", "cache-empty", COLORS["navy"]),
        (302, "One hypothesis", "one candidate", COLORS["orange"]),
        (520, "Semantic oracle", "complete response", COLORS["teal"]),
        (738, "Measured decision", "retain or reject", COLORS["gold"]),
        (956, "Cap reached", "stop and document", COLORS["gray"]),
    ]
    for index, (x, label, detail, color) in enumerate(steps):
        body.extend([_circle(x + 72, 260, 50, fill=COLORS["paper"], stroke=color, extra='stroke-width="4"'),
                     _text(x + 72, 253, str(index + 1), size=21, weight="bold", fill=color, anchor="middle"),
                     _text(x + 72, 335, label, size=16, weight="bold", anchor="middle"),
                     _text(x + 72, 357, detail, size=13, fill=COLORS["muted"], anchor="middle")])
        if index < len(steps) - 1:
            body.append(_line(x + 126, 260, x + 163, 260, stroke=COLORS["ink"], width=2, marker="arrow"))
    body.extend([
        _rect(150, 440, 900, 120, fill=COLORS["paper"], stroke=COLORS["line"], radius=6),
        _text(180, 476, "Acceptance record", size=15, weight="bold", fill=COLORS["teal"]),
        _text(180, 506, "Two native workers: 62.682 s isolated median; 18.03% below the fresh 76.470 s baseline.", size=18),
        _text(180, 534, "All rejected candidates were removed. Timeouts are deadline outcomes, not 120-second measurements.", size=14, fill=COLORS["muted"]),
        _text(1020, 476, "STOP CONDITION", size=13, weight="bold", fill=COLORS["orange"], anchor="end"),
        _text(1020, 506, "1 baseline + 11 candidates", size=17, weight="bold", anchor="end"),
        _text(1020, 534, "uncached median target < 20 s not met", size=13, fill=COLORS["muted"], anchor="end"),
    ])
    return _svg("autoresearch-loop", figure, body, data)


def render_experiment_outcomes(data: dict[str, Any]) -> str:
    figure = data["figures"]["experiment-outcomes"]
    experiments = figure["experiments"]
    body = _title_block(figure["title"], "Median wall times for measured candidates; deadline outcomes are categorical.")
    x0, y0, chart_w, chart_h, max_value = 360, 150, 700, 410, 120
    body.extend([_text(64, 162, "Run / hypothesis", size=14, weight="bold"), _text(x0, 142, "median seconds", size=13, fill=COLORS["muted"])])
    for tick in range(0, 121, 20):
        x = x0 + chart_w * tick / max_value
        body.extend([_line(x, y0, x, y0 + chart_h, stroke=COLORS["grid"]), _text(x, y0 + chart_h + 22, str(tick), size=12, fill=COLORS["muted"], anchor="middle")])
    row_h = 32
    styles = {"baseline": (COLORS["navy"], "circle"), "retain": (COLORS["teal"], "square"), "reject": (COLORS["orange"], "diamond")}
    for index, item in enumerate(experiments):
        y = y0 + 18 + index * row_h
        if index % 2 == 0:
            body.append(_rect(56, y - 18, 1030, row_h, fill=COLORS["paper"]))
        body.append(_text(64, y + 5, f"{item['run']}  {item['label']}", size=13))
        if item["status"] == "timeout":
            body.extend([_rect(x0 + 10, y - 10, 18, 18, fill="url(#diagonal-hatch)", stroke=COLORS["red"]),
                         _text(x0 + 42, y + 5, "TIMEOUT — warm-up exceeded deadline", size=13, fill=COLORS["red"], weight="bold")])
            continue
        value = item["median_seconds"]
        color, shape = styles[item["status"]]
        x = x0 + chart_w * value / max_value
        body.append(_line(x0, y, x, y, stroke=color, width=7))
        if shape == "circle":
            body.append(_circle(x, y, 7, fill=color))
        elif shape == "square":
            body.append(_rect(x - 7, y - 7, 14, 14, fill=color))
        else:
            body.append(f'<path d="M{x:.1f},{y - 8:.1f} L{x + 8:.1f},{y:.1f} L{x:.1f},{y + 8:.1f} L{x - 8:.1f},{y:.1f} Z" fill="{color}"/>')
        body.append(_text(min(x + 13, 1055), y + 5, f"{_fmt(value)} s", size=12, fill=color, weight="bold"))
    body.extend([_circle(80, 598, 6, fill=COLORS["navy"]), _text(94, 603, "baseline", size=12),
                 _rect(180, 592, 12, 12, fill=COLORS["teal"]), _text(200, 603, "retained", size=12),
                 f'<path d="M300,592 L306,598 L300,604 L294,598 Z" fill="{COLORS["orange"]}"/>', _text(314, 603, "rejected", size=12),
                 _rect(410, 592, 12, 12, fill="url(#diagonal-hatch)", stroke=COLORS["red"]), _text(430, 603, "timeout (categorical)", size=12)])
    return _svg("experiment-outcomes", figure, body, data)


def render_latency_distributions(data: dict[str, Any]) -> str:
    figure = data["figures"]["latency-distributions"]
    series = figure["series"]
    body = _title_block(figure["title"], "Cache-empty distributions and the separate response-cache-hit observation share an honest zero-based axis.")
    x0, y0, chart_w, chart_h, maximum = 190, 155, 860, 340, 120
    for tick in range(0, 121, 20):
        x = x0 + chart_w * tick / maximum
        body.extend([_line(x, y0, x, y0 + chart_h, stroke=COLORS["grid"]), _text(x, y0 + chart_h + 23, str(tick), size=12, fill=COLORS["muted"], anchor="middle")])
    body.append(_text(x0 + chart_w / 2, y0 + chart_h + 50, "wall time (seconds)", size=14, fill=COLORS["muted"], anchor="middle"))
    target_x = x0 + chart_w * data["study"]["target"]["threshold_seconds"] / maximum
    body.extend([_line(target_x, y0 - 8, target_x, y0 + chart_h, stroke=COLORS["red"], width=2, dash="6 5"),
                 _text(target_x, y0 - 16, "20 s research target", size=12, fill=COLORS["red"], anchor="middle")])
    colors = [COLORS["navy"], COLORS["teal"], COLORS["orange"], COLORS["gray"]]
    shapes = ["circle", "square", "diamond", "cross"]
    row_positions = [195, 275, 355, 435]
    for index, (item, color, shape, y) in enumerate(zip(series, colors, shapes, row_positions, strict=True)):
        body.append(_text(64, y - 8, item["label"], size=14, weight="bold"))
        body.append(_text(64, y + 12, f"n={item['n']}", size=12, fill=COLORS["muted"]))
        values = item["values"]
        x_values = [x0 + chart_w * value / maximum for value in values]
        if len(x_values) > 1:
            body.append(_line(min(x_values), y, max(x_values), y, stroke=color, width=3))
        for x, value in zip(x_values, values, strict=True):
            if shape == "circle":
                body.append(_circle(x, y, 7, fill=color))
            elif shape == "square":
                body.append(_rect(x - 7, y - 7, 14, 14, fill=color))
            elif shape == "diamond":
                body.append(f'<path d="M{x:.1f},{y - 8:.1f} L{x + 8:.1f},{y:.1f} L{x:.1f},{y + 8:.1f} L{x - 8:.1f},{y:.1f} Z" fill="{color}"/>')
            else:
                body.extend([_line(x - 6, y - 6, x + 6, y + 6, stroke=color, width=3), _line(x - 6, y + 6, x + 6, y - 6, stroke=color, width=3)])
        values_text = ", ".join(_fmt(value, 4) for value in values)
        body.append(_text(x0, y + 28, values_text + " s", size=12, fill=COLORS["muted"]))
    body.extend([_rect(64, 535, 996, 76, fill=COLORS["paper"], stroke=COLORS["line"], radius=5),
                 _text(84, 564, "Interpretation boundary", size=14, weight="bold", fill=COLORS["orange"]),
                 _text(84, 589, data["study"]["host"]["caveat"], size=13, fill=COLORS["muted"]),
                 _text(84, 608, "Response-cache hit (0.2663 s) is operational evidence only, not an uncached-search improvement.", size=13, fill=COLORS["muted"])])
    return _svg("latency-distributions", figure, body, data)


def render_profile_bottlenecks(data: dict[str, Any]) -> str:
    figure = data["figures"]["profile-bottlenecks"]
    observations = figure["observations"]
    body = _title_block(figure["title"], "Observed sample shares identify where further uncached-search research must begin.")
    x0, y0, chart_w, chart_h = 310, 180, 700, 230
    for tick in range(0, 101, 20):
        x = x0 + chart_w * tick / 100
        body.extend([_line(x, y0, x, y0 + chart_h, stroke=COLORS["grid"]), _text(x, y0 + chart_h + 24, f"{tick}%", size=12, fill=COLORS["muted"], anchor="middle")])
    for index, item in enumerate(observations):
        y = 240 + index * 115
        color = COLORS["teal"] if index == 0 else COLORS["orange"]
        width = chart_w * item["percent"] / 100
        body.extend([_text(64, y - 14, item["label"], size=19, weight="bold"),
                     _text(64, y + 10, "observed share", size=13, fill=COLORS["muted"]),
                     _rect(x0, y - 35, width, 50, fill=color, radius=3),
                     _rect(x0 + width - 5, y - 35, 5, 50, fill="url(#diagonal-hatch)"),
                     _text(x0 + width + 14, y - 2, f"{_fmt(item['percent'], 1)}%", size=22, weight="bold", fill=color)])
    body.extend([_rect(64, 480, 970, 104, fill=COLORS["paper"], stroke=COLORS["line"], radius=6),
                 _text(88, 513, "Do not add these shares", size=16, weight="bold"),
                 _text(88, 541, "Heap pop is a component of native compact search; these are sampled attributions, not a partition of all work.", size=14, fill=COLORS["muted"]),
                 _text(88, 568, "No remaining non-artifact method had measured evidence for a repeatable 5% gain when the cap fired.", size=14, fill=COLORS["muted"])])
    return _svg("profile-bottlenecks", figure, body, data)


def render_two_worker_execution(data: dict[str, Any]) -> str:
    figure = data["figures"]["two-worker-execution"]
    two_worker = figure
    body = _title_block(figure["title"], "Concurrency is deliberately bounded: four independent searches, at most two executing together.")
    body.extend([_text(64, 180, "Planner", size=17, weight="bold"), _text(230, 153, "Worker 1", size=16, weight="bold", anchor="middle"), _text(490, 153, "Worker 2", size=16, weight="bold", anchor="middle"), _text(790, 153, "Ordered join", size=16, weight="bold", anchor="middle")])
    searches = [(190, 195, "λ1"), (450, 195, "λ2"), (190, 330, "λ3"), (450, 330, "λ4")]
    for x, y, label in searches:
        body.extend([_rect(x, y, 120, 68, fill=COLORS["paper"], stroke=COLORS["teal"], radius=5), _text(x + 60, y + 42, label + " native search", size=14, weight="bold", anchor="middle")])
    body.extend([_rect(64, 220, 76, 92, fill=COLORS["navy"], radius=5), _text(102, 258, "request", size=14, fill=COLORS["white"], weight="bold", anchor="middle"), _text(102, 280, "context", size=14, fill=COLORS["white"], anchor="middle"),
                 _line(140, 245, 186, 225, stroke=COLORS["ink"], width=2, marker="arrow"), _line(140, 285, 186, 355, stroke=COLORS["ink"], width=2, marker="arrow"),
                 _line(310, 229, 704, 258, stroke=COLORS["ink"], width=2, marker="arrow"), _line(570, 229, 704, 258, stroke=COLORS["ink"], width=2, marker="arrow"),
                 _line(310, 364, 704, 292, stroke=COLORS["ink"], width=2, marker="arrow"), _line(570, 364, 704, 292, stroke=COLORS["ink"], width=2, marker="arrow"),
                 _rect(704, 220, 172, 92, fill=COLORS["gold"], radius=5), _text(790, 258, "restore original", size=14, fill=COLORS["white"], weight="bold", anchor="middle"), _text(790, 280, "multiplier order", size=14, fill=COLORS["white"], anchor="middle"),
                 _line(876, 266, 966, 266, stroke=COLORS["ink"], width=2, marker="arrow"), _rect(966, 220, 120, 92, fill=COLORS["orange"], radius=5), _text(1026, 258, "response", size=15, fill=COLORS["white"], weight="bold", anchor="middle"), _text(1026, 280, "or failure", size=14, fill=COLORS["white"], anchor="middle")])
    body.extend([_rect(64, 470, 1022, 112, fill=COLORS["paper"], stroke=COLORS["line"], radius=6),
                 _text(88, 505, "Measured isolated result", size=15, weight="bold", fill=COLORS["teal"]),
                 _text(88, 535, f"{_fmt(two_worker['isolated_candidate_median_seconds'])} s median vs {_fmt(two_worker['isolated_baseline_median_seconds'])} s baseline ({_fmt(two_worker['isolated_improvement_percent'], 2)}% lower; n=3 each).", size=18),
                 _text(88, 562, "Workers join on success or failure; three and four workers were rejected under host contention.", size=14, fill=COLORS["muted"])])
    return _svg("two-worker-execution", figure, body, data)


RENDERERS = {
    "system-boundary": render_system_boundary,
    "autoresearch-loop": render_autoresearch_loop,
    "experiment-outcomes": render_experiment_outcomes,
    "latency-distributions": render_latency_distributions,
    "profile-bottlenecks": render_profile_bottlenecks,
    "two-worker-execution": render_two_worker_execution,
}


def generate(output_dir: Path = OUTPUT_DIR, data_path: Path = DATA_PATH) -> list[Path]:
    """Render all canonical figures to *output_dir* in deterministic order."""
    data = load_data(data_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for filename in CANONICAL_FILENAMES:
        key = filename.removesuffix(".svg")
        output = output_dir / filename
        output.write_text(RENDERERS[key](data), encoding="utf-8")
        generated.append(output)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    args = parser.parse_args()
    for path in generate(args.output_dir, args.data):
        print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


if __name__ == "__main__":
    main()
