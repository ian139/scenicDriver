from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "docs/assets/research/routing/figure-data.json"
SVG_NAMESPACE = "{http://www.w3.org/2000/svg}"


def load_script(name: str):
    path = ROOT / "scripts/reports" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


figures = load_script("generate_routing_research_figures")
cleanup = load_script("autoresearch_cleanup_metric")


class RoutingResearchFigureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = figures.load_data(DATA_PATH)

    def generate(self, data):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "data.json"
            source.write_text(json.dumps(data), encoding="utf-8")
            return {
                path.name: ElementTree.parse(path).getroot()
                for path in figures.generate(root / "output", source)
            }

    @staticmethod
    def text(root):
        return " ".join(root.itertext())

    @staticmethod
    def seconds(root):
        return {
            float(value)
            for value in re.findall(r"(?<!\w)(\d+(?:\.\d+)?)\s+s\b", " ".join(root.itertext()))
        }

    def changed_measurements(self, candidate=40):
        data = copy.deepcopy(self.data)
        series = data["figures"]["latency-distributions"]["series"]
        values = {
            "uncached": [46, 48, 50, 52, 54],
            "accepted_candidate": [candidate - 2, candidate, candidate + 2],
            "final_confirmation": [30, 34, 38],
            "response_cache_hit": [0.75],
        }
        for item in series:
            item["values"] = values[item["class"]]
            item["n"] = len(item["values"])
        for item in data["figures"]["experiment-outcomes"]["experiments"]:
            if item["status"] == "baseline":
                item["median_seconds"] = 50
            elif item["status"] == "retain":
                item["median_seconds"] = candidate
        percent = 100 * (50 - candidate) / 50
        loop = data["figures"]["autoresearch-loop"]
        loop["acceptance_improvement_percent"] = percent
        loop["candidate_cap"] = 12
        workers = data["figures"]["two-worker-execution"]
        workers["isolated_baseline_median_seconds"] = 50
        workers["isolated_candidate_median_seconds"] = candidate
        workers["isolated_improvement_percent"] = percent
        data["study"]["target"].update(threshold_seconds=35, met=True)
        data["figures"]["profile-bottlenecks"]["observations"][0]["percent"] = 10
        data["figures"]["profile-bottlenecks"]["observations"][1]["percent"] = 4
        return data

    def test_annotations_follow_samples_threshold_counts_and_profile_shares(self) -> None:
        rendered = self.generate(self.changed_measurements())
        for name in ("autoresearch-loop.svg", "two-worker-execution.svg"):
            root = rendered[name]
            self.assertTrue({40, 50}.issubset(self.seconds(root)))
            self.assertRegex(self.text(root), r"20%\s+lower")
        worker_text = self.text(rendered["two-worker-execution.svg"])
        self.assertRegex(worker_text, r"n=3\s+candidate")
        self.assertRegex(worker_text, r"n=5\s+baseline")
        for name in ("system-boundary.svg", "autoresearch-loop.svg", "latency-distributions.svg"):
            root = rendered[name]
            self.assertIn(35, self.seconds(root))
            self.assertNotIn("not met", self.text(root))
        self.assertIn(0.75, self.seconds(rendered["latency-distributions.svg"]))
        loop_text = self.text(rendered["autoresearch-loop.svg"])
        self.assertRegex(loop_text, r"11\s+(?:measured\s+)?candidates")
        self.assertNotIn("Cap reached", loop_text)
        profile = rendered["profile-bottlenecks.svg"]
        self.assertRegex(self.text(profile), r"10%")
        self.assertRegex(self.text(profile), r"4%")
        description = profile.find(SVG_NAMESPACE + "desc").text
        self.assertIn("10%", description)
        self.assertIn("4%", description)

    def test_slower_candidate_is_not_reported_as_an_improvement(self) -> None:
        rendered = self.generate(self.changed_measurements(candidate=60))
        for name in ("autoresearch-loop.svg", "two-worker-execution.svg"):
            self.assertRegex(self.text(rendered[name]), r"20%\s+higher")
            self.assertNotRegex(self.text(rendered[name]), r"20%\s+lower")

    def test_target_uses_final_confirmation_and_strict_threshold_not_cache_hit(self) -> None:
        data = self.changed_measurements()
        data["study"]["target"].update(threshold_seconds=34, met=False)
        rendered = self.generate(data)
        for name in ("system-boundary.svg", "autoresearch-loop.svg", "latency-distributions.svg"):
            self.assertIn("not met", self.text(rendered[name]))
        data["study"]["target"]["met"] = True
        with self.assertRaises(ValueError):
            self.generate(data)

    def test_invalid_quantitative_data_is_rejected_before_writing_figures(self) -> None:
        supplied = self.data["figures"]
        series = supplied["latency-distributions"]["series"]
        experiments = supplied["experiment-outcomes"]["experiments"]
        loop = supplied["autoresearch-loop"]
        workers = supplied["two-worker-execution"]
        profile = supplied["profile-bottlenecks"]["observations"]
        production = self.data["production_context"]
        mutations = [
            (("figures", "latency-distributions", "series", 0, "n"), series[0]["n"] + 1),
            (("figures", "latency-distributions", "series", 0, "n"), True),
            (("figures", "latency-distributions", "series", 0, "n"), 3.0),
            (("figures", "latency-distributions", "series", 0, "values"), []),
            (("figures", "latency-distributions", "series", 0, "values", 0), float("nan")),
            (("figures", "latency-distributions", "series", 0, "values", 0), float("inf")),
            (("figures", "latency-distributions", "series", 0, "values", 0), -1),
            (("figures", "latency-distributions", "series", 0, "values", 0), True),
            (("figures", "latency-distributions", "series", 1, "class"), "uncached"),
            (("figures", "experiment-outcomes", "experiments", 0, "median_seconds"), experiments[0]["median_seconds"] + 1),
            (("figures", "experiment-outcomes", "experiments", 1, "median_seconds"), 1),
            (("figures", "experiment-outcomes", "experiments", 2, "status"), "success"),
            (("figures", "experiment-outcomes", "experiments", 2, "run"), experiments[0]["run"]),
            (("figures", "autoresearch-loop", "baseline_count"), 2),
            (("figures", "autoresearch-loop", "candidate_cap"), len(experiments) - loop["baseline_count"] - 1),
            (("figures", "autoresearch-loop", "acceptance_improvement_percent"), loop["acceptance_improvement_percent"] + 1),
            (("figures", "two-worker-execution", "isolated_improvement_percent"), workers["isolated_improvement_percent"] + 1),
            (("figures", "two-worker-execution", "isolated_candidate_median_seconds"), workers["isolated_candidate_median_seconds"] + 1),
            (("figures", "two-worker-execution", "max_workers"), 3),
            (("figures", "system-boundary", "search_count"), 3),
            (("figures", "profile-bottlenecks", "observations", 0, "percent"), 101),
            (("figures", "profile-bottlenecks", "observations", 1, "percent"), profile[0]["percent"] + 1),
            (("study", "target", "threshold_seconds"), 0),
            (("study", "target", "met"), not self.data["study"]["target"]["met"]),
            (("study", "request", "deadline_seconds"), float("inf")),
            (("production_context", "cases"), 0),
            (("production_context", "baseline_timeouts"), production["cases"] + 1),
            (("production_context", "candidate_timeouts"), -1),
            (("production_context", "candidate_p95_ms"), production["candidate_median_ms"] / 2),
        ]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, (keys, value) in enumerate(mutations):
                with self.subTest(keys=keys, value=value):
                    invalid = copy.deepcopy(self.data)
                    container = invalid
                    for key in keys[:-1]:
                        container = container[key]
                    container[keys[-1]] = value
                    source = root / "data.json"
                    source.write_text(json.dumps(invalid), encoding="utf-8")
                    output = root / f"output-{index}"
                    with self.assertRaises(ValueError):
                        figures.generate(output, source)
                    self.assertFalse(output.exists())

    def test_larger_latency_and_threshold_remain_inside_chart(self) -> None:
        data = self.changed_measurements()
        confirmation = next(item for item in data["figures"]["latency-distributions"]["series"] if item["class"] == "final_confirmation")
        confirmation["values"] = [160, 180, 200]
        data["study"]["target"].update(threshold_seconds=150, met=False)
        root = self.generate(data)["latency-distributions.svg"]
        self.assertIn(150, self.seconds(root))
        ticks = [
            float(item.text) for item in root.iter(SVG_NAMESPACE + "text")
            if re.fullmatch(r"\d+(?:\.\d+)?", item.text or "")
        ]
        self.assertGreaterEqual(max(ticks), 200)
        width = float(root.attrib["width"])
        for circle in root.iter(SVG_NAMESPACE + "circle"):
            center = float(circle.attrib["cx"])
            radius = float(circle.attrib["r"])
            self.assertGreaterEqual(center - radius, 0)
            self.assertLessEqual(center + radius, width)

    def test_generation_is_deterministic_and_accessible(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first = figures.generate(directory, DATA_PATH)
            hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in first}
            second = figures.generate(directory, DATA_PATH)
            self.assertEqual(hashes, {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in second})
            self.assertEqual(set(hashes), set(figures.CANONICAL_FILENAMES))
            for path in second:
                root = ElementTree.parse(path).getroot()
                self.assertEqual(root.tag, SVG_NAMESPACE + "svg")
                self.assertEqual(root.attrib["role"], "img")
                title = root.find(SVG_NAMESPACE + "title")
                description = root.find(SVG_NAMESPACE + "desc")
                self.assertEqual(root.attrib["aria-labelledby"].split(), [title.attrib["id"], description.attrib["id"]])
                self.assertTrue(title.text.strip())
                self.assertTrue(description.text.strip())


class MaintenanceMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        subprocess.run(["git", "init", "--quiet", str(self.root)], check=True, capture_output=True)
        self.patch_root = patch.object(cleanup, "ROOT", self.root)
        self.patch_root.start()
        self.addCleanup(self.patch_root.stop)
        self.put(".ruff.toml", '[lint]\nselect = ["F"]\n', tracked=False)

    def put(self, name, content, *, tracked=True):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        if tracked:
            subprocess.run(["git", "add", "--", name], cwd=self.root, check=True, capture_output=True)
        return path

    def metrics(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            cleanup.main()
        return {
            name: int(value)
            for line in output.getvalue().splitlines()
            for name, value in [line.removeprefix("METRIC ").split("=")]
        }

    def test_native_shell_lines_and_lint_penalty_keep_existing_objective(self) -> None:
        self.put("src/search.c", "int search(void) {\n    return 0; }\n")
        self.put("src/search.h", "int search(void);\n")
        self.put("scripts/task.sh", "#!/bin/sh\nset -e\necho ready\n")
        self.put("autoresearch.sh", "#!/bin/sh\necho run\n")
        self.put("src/tracked.py", "import os\n")  # One actual F401 violation.
        self.put("README.md", "Title\nDescription\n")
        self.put("tests/test_tracked.py", "value = 1\nother = 2\n")
        self.put("src/blob.bin", "not maintenance text\n")
        result = self.metrics()
        self.assertEqual(result, {
            "maintenance_debt": 10011,
            "ruff_violations": 1,
            "maintenance_lines": 11,
            "active_files": 6,
            "test_lines": 2,
        })

    def test_untracked_outputs_and_tests_do_not_change_inventory_or_lint_debt(self) -> None:
        tracked = self.put("src/file with spaces.py", "value = 1\n")
        self.put("tests/test_tracked.py", "value = 1\n")
        before = self.metrics()
        self.put("src/untracked.py", "import missing_dependency\n", tracked=False)
        self.put("docs/generated.md", "generated\n" * 100, tracked=False)
        self.put("scripts/generated.sh", "echo generated\n", tracked=False)
        self.put("tests/test_untracked.py", "import missing_dependency\n", tracked=False)
        self.assertEqual(self.metrics(), before)
        tracked.write_text("value = 1\nother = 2\n", encoding="utf-8")
        after = self.metrics()
        self.assertEqual(after["maintenance_lines"], before["maintenance_lines"] + 1)
        self.assertEqual(after["maintenance_debt"], before["maintenance_debt"] + 1)
        self.assertEqual(after["active_files"], before["active_files"])

    def test_staged_additions_and_worktree_deletions_affect_source_debt(self) -> None:
        source = self.put("src/search.cpp", "int value = 1;\n")
        self.assertEqual(self.metrics()["maintenance_lines"], 1)
        self.put("scripts/task.sh", "echo one\necho two\n")
        self.assertEqual(self.metrics()["maintenance_lines"], 3)
        source.unlink()
        result = self.metrics()
        self.assertEqual(result["maintenance_lines"], 2)
        self.assertEqual(result["active_files"], 1)

    def test_missing_git_inventory_fails_instead_of_guessing_filesystem(self) -> None:
        with TemporaryDirectory() as temporary, patch.object(cleanup, "ROOT", Path(temporary)):
            with self.assertRaises(RuntimeError):
                cleanup.main()


if __name__ == "__main__":
    unittest.main()
