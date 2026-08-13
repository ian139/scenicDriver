from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "scripts/reports/generate_routing_research_figures.py"
DATA_PATH = ROOT / "docs/assets/research/routing/figure-data.json"

spec = importlib.util.spec_from_file_location("routing_research_figures", GENERATOR_PATH)
assert spec is not None and spec.loader is not None
figures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figures)

SVG_NAMESPACE = "{http://www.w3.org/2000/svg}"


class RoutingResearchFigureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = figures.load_data(DATA_PATH)

    def test_source_schema_and_exact_research_values(self) -> None:
        self.assertEqual(self.data["schema_version"], 1)
        self.assertEqual(
            self.data["source"]["document"],
            "docs/research/routing-performance-autoresearch.md",
        )
        self.assertEqual(self.data["study"]["target"], {
            "metric": "median uncached complete plan_routes request",
            "threshold_seconds": 20,
            "met": False,
            "evidence": "uncached_decision",
        })
        self.assertEqual(self.data["study"]["request"]["deadline_seconds"], 120)
        self.assertEqual(self.data["study"]["host"]["final_load_averages"], [16.95, 31.52, 30.53])
        self.assertEqual(
            self.data["figures"]["latency-distributions"]["series"][0]["values"],
            [76.4699, 78.9445, 74.2697],
        )
        self.assertEqual(
            self.data["figures"]["latency-distributions"]["series"][1]["values"],
            [60.8398, 62.6824, 65.7794],
        )
        self.assertEqual(
            self.data["figures"]["latency-distributions"]["series"][2]["values"],
            [82.3048, 114.4851, 92.0760, 70.9841, 61.6257],
        )
        self.assertEqual(
            self.data["figures"]["latency-distributions"]["series"][3]["values"],
            [0.2663],
        )
        self.assertEqual(self.data["figures"]["profile-bottlenecks"]["observations"], [
            {"label": "Native compact search", "percent": 89.6},
            {"label": "Heap pop", "percent": 57.5},
        ])
        self.assertEqual(self.data["production_context"]["cases"], 2256)
        self.assertEqual(self.data["production_context"]["baseline_median_ms"], 6449.169)
        self.assertEqual(self.data["production_context"]["candidate_median_ms"], 1238.558)

    def test_every_research_figure_value_has_evidence_reference(self) -> None:
        evidence = self.data["source"]["evidence"]
        self.assertTrue(all(isinstance(value, str) and value for value in evidence.values()))
        self.assertEqual(
            set(self.data["figures"]),
            {name.removesuffix(".svg") for name in figures.CANONICAL_FILENAMES},
        )
        for figure in self.data["figures"].values():
            self.assertIn(figure["evidence"], evidence)
        for item in self.data["study"].values():
            self.assertIn(item["evidence"], evidence)
        self.assertIn(self.data["production_context"]["evidence"], evidence)

    def test_timeouts_and_response_cache_hits_are_not_misrepresented(self) -> None:
        experiments = self.data["figures"]["experiment-outcomes"]["experiments"]
        timeout_runs = [item["run"] for item in experiments if item["status"] == "timeout"]
        self.assertEqual(timeout_runs, [112, 119, 122])
        self.assertTrue(all("median_seconds" not in item for item in experiments if item["status"] == "timeout"))
        classes = {
            item["class"]
            for item in self.data["figures"]["latency-distributions"]["series"]
        }
        self.assertIn("response_cache_hit", classes)
        self.assertIn("uncached", classes)
        self.assertNotEqual("response_cache_hit", "uncached")
        invalid = copy.deepcopy(self.data)
        invalid["figures"]["experiment-outcomes"]["experiments"][1]["median_seconds"] = 120
        with self.assertRaisesRegex(ValueError, "categorical"):
            figures.validate_data(invalid)

    def test_generation_is_deterministic_and_matches_committed_assets(self) -> None:
        with TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            first = figures.generate(output_dir, DATA_PATH)
            first_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in first}
            second = figures.generate(output_dir, DATA_PATH)
            second_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in second}
            self.assertEqual(first_hashes, second_hashes)
            committed_hashes = {
                name: hashlib.sha256((ROOT / "docs/assets/research/routing" / name).read_bytes()).hexdigest()
                for name in figures.CANONICAL_FILENAMES
            }
            self.assertEqual(first_hashes, committed_hashes)

    def test_svg_xml_accessibility_and_required_annotations(self) -> None:
        required_annotations = {
            "system-boundary.svg": ["Target:", "Complete response cache cleared", "Deadline:"],
            "autoresearch-loop.svg": ["1 baseline + 11 candidates", "20 s not met", "Timeouts are deadline outcomes"],
            "experiment-outcomes.svg": ["median seconds", "TIMEOUT", "warm-up exceeded deadline"],
            "latency-distributions.svg": ["20 s research target", "n=3", "n=5", "Response-cache hit"],
            "profile-bottlenecks.svg": ["89.6%", "57.5%", "Do not add these shares"],
            "two-worker-execution.svg": ["at most two", "62.682 s", "76.47 s", "n=3 each"],
        }
        for filename in figures.CANONICAL_FILENAMES:
            path = ROOT / "docs/assets/research/routing" / filename
            root = ElementTree.parse(path).getroot()
            self.assertEqual(root.tag, SVG_NAMESPACE + "svg")
            self.assertEqual(root.attrib["role"], "img")
            labelled_by = root.attrib["aria-labelledby"].split()
            self.assertEqual(len(labelled_by), 2)
            titles = root.findall(SVG_NAMESPACE + "title")
            descriptions = root.findall(SVG_NAMESPACE + "desc")
            self.assertEqual(len(titles), 1)
            self.assertEqual(len(descriptions), 1)
            self.assertEqual([titles[0].attrib["id"], descriptions[0].attrib["id"]], labelled_by)
            text = path.read_text(encoding="utf-8")
            for annotation in required_annotations[filename]:
                self.assertIn(annotation, text)


if __name__ == "__main__":
    unittest.main()
