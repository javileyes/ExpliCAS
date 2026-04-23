import argparse
import importlib.util
import pathlib
import sys
import unittest


SCRIPT_PATH = pathlib.Path(__file__).resolve().parent / "engine_improvement_scorecard.py"
SPEC = importlib.util.spec_from_file_location("engine_improvement_scorecard", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


SAMPLE_CORPUS_OUTPUT = """Total cases: 8
Passed: 8
Failed: 0
Elapsed: 17.10ms
Distinct wrappers: 4
Distinct families: 2
Distinct complexity levels: 1
Distinct shell depths: 2
Largest wrapper share: 25.0%
Largest wrapper x complexity share: 25.0%
Max shell depth: 2
Max expression depth: 5
Average wrapper overhead nodes: 10.00
Wrappers: additive_passthrough_zero, shifted_quotient_one

By composition:
  sum: total=3 passed=3 failed=0 elapsed=12.00ms avg_case_ms=4.00
  product: total=5 passed=5 failed=0 elapsed=5.10ms avg_case_ms=1.02

By window:
  sum@0+3: total=3 passed=3 failed=0 elapsed=12.00ms avg_case_ms=4.00
  product@0+5: total=5 passed=5 failed=0 elapsed=5.10ms avg_case_ms=1.02

By shell depth:
  depth 1: total=2 passed=2 failed=0
  depth 2: total=6 passed=6 failed=0

By complexity level:
  l2_wrapper_plus_noise: total=8 passed=8 failed=0 avg_wrapper_overhead_nodes=10.00 avg_shell_depth=1.75 max_shell_depth=2

Top wrapper x complexity buckets:
  additive_passthrough_zero x l2_wrapper_plus_noise: total=5 passed=5 failed=0 avg_wrapper_overhead_nodes=11.20 avg_shell_depth=2.00 max_shell_depth=2
  shifted_quotient_one x l2_wrapper_plus_noise: total=3 passed=3 failed=0 avg_wrapper_overhead_nodes=8.00 avg_shell_depth=1.33 max_shell_depth=2

Orchestrator Profiling Report
──────────────────────────────────────────────────────────────────────────────────────────────
Section                                          Attempts     Hits   Misses     Total ms       Avg us
──────────────────────────────────────────────────────────────────────────────────────────────
pipeline.phase.core                                     6        6        0        0.444        74.08
root.div.01.shifted_quotient_exact_one_gate             2        0        2        0.124        61.98
TOTAL                                                   8        6        2        0.568        71.00
──────────────────────────────────────────────────────────────────────────────────────────────
Sample expressions
──────────────────────────────────────────────────────────────────────────────────────────────
pipeline.phase.core
  - 0
root.div.01.shifted_quotient_exact_one_gate
  - x + x + 1  ||  2 * x + 1
"""


class EngineImprovementScorecardTests(unittest.TestCase):
    def test_orchestrator_profile_env_and_command_only_apply_to_embedded_suite(self):
        args = argparse.Namespace(orchestrator_profile=True)

        embedded_env = MODULE.orchestrator_profile_env(
            MODULE.SUITES["embedded_equivalence_context"], args
        )
        embedded_command = MODULE.orchestrator_profile_command(
            MODULE.SUITES["embedded_equivalence_context"]
        )
        pressure_env = MODULE.orchestrator_profile_env(
            MODULE.SUITES["simplify_zero_mixed"], args
        )
        pressure_command = MODULE.orchestrator_profile_command(
            MODULE.SUITES["simplify_zero_mixed"]
        )

        self.assertEqual(embedded_env["CAS_PROFILE_ORCHESTRATOR_SHORTCUTS"], "1")
        self.assertEqual(
            embedded_env["CAS_PROFILE_ORCHESTRATOR_SHORTCUT_FILTER"], "pipeline.,root."
        )
        self.assertEqual(embedded_command[-2:], ["--limit", str(MODULE.EMBEDDED_ORCHESTRATOR_PROFILE_LIMIT)])
        self.assertIsNone(pressure_env)
        self.assertIsNone(pressure_command)

    def test_parse_corpus_extracts_complexity_and_orchestrator_profile_metrics(self):
        metrics = MODULE.parse_corpus(SAMPLE_CORPUS_OUTPUT)

        self.assertEqual(metrics["complexity_level_count"], 1)
        self.assertEqual(metrics["shell_depth_count"], 2)
        self.assertEqual(metrics["max_shell_depth"], 2)
        self.assertEqual(metrics["max_expression_depth"], 5)
        self.assertEqual(metrics["average_wrapper_overhead_nodes"], 10.0)
        self.assertEqual(
            metrics["complexity_rows"]["l2_wrapper_plus_noise"]["avg_shell_depth"], 1.75
        )
        self.assertEqual(metrics["shell_depth_rows"][2]["total"], 6)
        self.assertEqual(metrics["wrapper_complexity_rows"][0]["wrapper"], "additive_passthrough_zero")
        self.assertEqual(
            metrics["wrapper_complexity_rows"][0]["avg_wrapper_overhead_nodes"], 11.2
        )
        self.assertEqual(metrics["composition_rows"]["sum"]["total"], 3)
        self.assertAlmostEqual(metrics["composition_rows"]["sum"]["elapsed_seconds"], 0.012)
        self.assertEqual(metrics["composition_rows"]["sum"]["avg_case_ms"], 4.0)
        self.assertEqual(metrics["window_rows"]["sum@0+3"]["total"], 3)
        self.assertAlmostEqual(metrics["window_rows"]["product@0+5"]["elapsed_seconds"], 0.0051)

        profile = metrics["orchestrator_profile"]
        self.assertEqual(profile["section_count"], 2)
        self.assertEqual(profile["totals"]["attempts"], 8)
        self.assertEqual(profile["totals"]["misses"], 2)
        self.assertEqual(profile["top_hot_sections"][0]["section"], "pipeline.phase.core")
        self.assertEqual(
            profile["top_no_match_cost_sections"][0]["section"],
            "root.div.01.shifted_quotient_exact_one_gate",
        )
        self.assertEqual(profile["top_hot_sections"][0]["samples"], ["0"])

    def test_render_markdown_includes_embedded_orchestrator_profile_summary(self):
        metrics = MODULE.parse_corpus(SAMPLE_CORPUS_OUTPUT)
        scorecard = {
            "generated_at": "2026-04-20T00:00:00+00:00",
            "profile": "guardrail",
            "git": {"branch": "main", "commit": "abc123"},
            "suites": {
                "embedded_equivalence_context": {
                    "status": "pass",
                    "elapsed_seconds": 0.017,
                    "metrics": metrics,
                    "guardrail": {
                        "assessment": "stable",
                        "baseline_elapsed_seconds": 0.02,
                        "delta_seconds": -0.003,
                        "delta_ratio": -0.15,
                        "threshold_seconds": 5.0,
                    },
                    "delta": {},
                }
            },
        }

        markdown = MODULE.render_markdown(scorecard)

        self.assertIn("## Embedded Orchestrator Profile", markdown)
        self.assertIn("Shell-depth mix", markdown)
        self.assertIn("Dominant wrapper x complexity buckets", markdown)
        self.assertIn("pipeline.phase.core", markdown)
        self.assertIn("No-match hotspot 1", markdown)

    def test_render_markdown_includes_mixed_pressure_and_proof_shape_caveat(self):
        metrics = MODULE.parse_corpus(SAMPLE_CORPUS_OUTPUT)
        scorecard = {
            "generated_at": "2026-04-20T00:00:00+00:00",
            "profile": "pressure",
            "git": {"branch": "main", "commit": "abc123"},
            "suites": {
                "simplify_zero_mixed": {
                    "status": "pass",
                    "elapsed_seconds": 2.5,
                    "metrics": metrics,
                    "delta": {},
                },
                "simplify_strict": {
                    "status": "pass",
                    "elapsed_seconds": 40.0,
                    "metrics": {
                        "total_combos": 100,
                        "nf_convergent": 0,
                        "proved_symbolic": 100,
                        "numeric_only": 0,
                        "inconclusive": 0,
                        "failed": 0,
                        "timeouts": 0,
                        "cycles": 0,
                        "skipped": 0,
                        "suite_rows": {},
                        "proved_breakdown": {
                            "quotient": 8,
                            "difference": 2,
                            "composed": 90,
                        },
                        "proof_shape_mix": {
                            "non_composed_symbolic": 10,
                            "non_composed_share_percent": 10.0,
                            "composed_share_percent": 90.0,
                        },
                    },
                    "delta": {},
                },
            },
        }

        markdown = MODULE.render_markdown(scorecard)

        self.assertIn("## Simplify Benchmark Interpretation", markdown)
        self.assertIn("composed 90.0%", markdown)
        self.assertIn("## Mixed Zero Pressure", markdown)
        self.assertIn("Composition hotspots", markdown)
        self.assertIn("Window slices", markdown)
        self.assertIn("sum total=3 failed=0", markdown)


if __name__ == "__main__":
    unittest.main()
