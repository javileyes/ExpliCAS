import argparse
import contextlib
import io
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
Distinct wrappers: 3
Distinct families: 2
Distinct complexity levels: 1
Distinct shell depths: 2
Largest wrapper share: 25.0%
Largest wrapper x complexity share: 25.0%
Max shell depth: 2
Max expression depth: 5
Average wrapper overhead nodes: 10.00
Wrappers: additive_passthrough_zero, combined_additive_zero, shifted_quotient_one

By wrapper:
  additive_passthrough_zero: total=5 passed=5 failed=0
  combined_additive_zero: total=1 passed=1 failed=0
  shifted_quotient_one: total=3 passed=3 failed=0

By composition:
  sum: total=3 passed=3 failed=0 elapsed=12.00ms avg_case_ms=4.00 wire_elapsed=3.00ms avg_wire_ms=1.00 parse_elapsed=1.20ms avg_parse_ms=0.40 simplify_elapsed=0.90ms avg_simplify_ms=0.30
  product: total=5 passed=5 failed=0 elapsed=5.10ms avg_case_ms=1.02 wire_elapsed=1.60ms avg_wire_ms=0.32 parse_elapsed=0.55ms avg_parse_ms=0.11 simplify_elapsed=0.45ms avg_simplify_ms=0.09

By window:
  sum@0+3: total=3 passed=3 failed=0 elapsed=12.00ms avg_case_ms=4.00 wire_elapsed=3.00ms avg_wire_ms=1.00 parse_elapsed=1.20ms avg_parse_ms=0.40 simplify_elapsed=0.90ms avg_simplify_ms=0.30
  product@0+5: total=5 passed=5 failed=0 elapsed=5.10ms avg_case_ms=1.02 wire_elapsed=1.60ms avg_wire_ms=0.32 parse_elapsed=0.55ms avg_parse_ms=0.11 simplify_elapsed=0.45ms avg_simplify_ms=0.09

By shell depth:
  depth 1: total=2 passed=2 failed=0
  depth 2: total=6 passed=6 failed=0

By complexity level:
  l2_wrapper_plus_noise: total=8 passed=8 failed=0 avg_wrapper_overhead_nodes=10.00 avg_shell_depth=1.75 max_shell_depth=2

Top wrapper x complexity buckets:
  additive_passthrough_zero x l2_wrapper_plus_noise: total=5 passed=5 failed=0 avg_wrapper_overhead_nodes=11.20 avg_shell_depth=2.00 max_shell_depth=2
  combined_additive_zero x l0_root_pair: total=1 passed=1 failed=0 avg_wrapper_overhead_nodes=0.00 avg_shell_depth=0.00 max_shell_depth=0
  shifted_quotient_one x l2_wrapper_plus_noise: total=3 passed=3 failed=0 avg_wrapper_overhead_nodes=8.00 avg_shell_depth=1.33 max_shell_depth=2

Steady-state engine-heavy reruns:
  [sum@0+3 #9 sum] runs=3 median_simplify=0.25ms median_wire=0.40ms median_parse=0.05ms median_elapsed=1.20ms expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (1/(x - 1) - 1/(x + 1) - 2/(x^2 - 1))
  [product@0+5 #12 product] runs=3 median_simplify=0.09ms median_wire=0.12ms median_parse=0.03ms median_elapsed=0.55ms expr=a*b - b*a

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

    def test_pressure_profile_stays_bounded_and_full_keeps_nf_first(self):
        pressure_args = argparse.Namespace(profile="pressure", suite=[])
        full_args = argparse.Namespace(profile="full", suite=[])

        pressure_names = [spec.name for spec in MODULE.selected_suites(pressure_args)]
        full_names = [spec.name for spec in MODULE.selected_suites(full_args)]

        self.assertEqual(pressure_names, ["simplify_zero_mixed"])
        self.assertIn("simplify_nf_first", full_names)
        self.assertIn("simplify_zero_mixed", full_names)
        self.assertEqual(
            MODULE.SUITES["simplify_nf_first"].timeout_seconds,
            MODULE.NF_FIRST_FULL_TIMEOUT_SECONDS,
        )

    def test_run_command_timeout_marks_timeout_and_captures_output(self):
        spec = MODULE.SuiteSpec(
            name="synthetic_timeout",
            category="test",
            profile_tags=("test",),
            command=[
                sys.executable,
                "-c",
                "import time; print('started', flush=True); time.sleep(5)",
            ],
            env={},
            parser="cargo_test_basic",
            description="Synthetic timeout command.",
            timeout_seconds=0.2,
        )

        captured_stdout = io.StringIO()
        with contextlib.redirect_stdout(captured_stdout):
            returncode, output, elapsed, timed_out = MODULE.run_command(spec)

        self.assertTrue(timed_out)
        self.assertNotEqual(returncode, 0)
        self.assertLess(elapsed, 3.0)
        self.assertIn("started", output)
        self.assertIn("suite timeout after 0.2s", output)
        self.assertEqual(captured_stdout.getvalue(), output)

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
        self.assertEqual(metrics["wrapper_rows"]["shifted_quotient_one"]["total"], 3)
        self.assertEqual(metrics["wrapper_complexity_rows"][0]["wrapper"], "additive_passthrough_zero")
        self.assertEqual(
            metrics["wrapper_complexity_rows"][0]["avg_wrapper_overhead_nodes"], 11.2
        )
        self.assertEqual(metrics["composition_rows"]["sum"]["total"], 3)
        self.assertAlmostEqual(metrics["composition_rows"]["sum"]["elapsed_seconds"], 0.012)
        self.assertEqual(metrics["composition_rows"]["sum"]["avg_case_ms"], 4.0)
        self.assertAlmostEqual(
            metrics["composition_rows"]["sum"]["simplify_elapsed_seconds"], 0.0009
        )
        self.assertEqual(metrics["composition_rows"]["sum"]["avg_simplify_ms"], 0.30)
        self.assertEqual(metrics["window_rows"]["sum@0+3"]["total"], 3)
        self.assertAlmostEqual(metrics["window_rows"]["product@0+5"]["elapsed_seconds"], 0.0051)
        self.assertAlmostEqual(
            metrics["window_rows"]["product@0+5"]["parse_elapsed_seconds"], 0.00055
        )
        self.assertEqual(metrics["steady_engine_heavy_rows"][0]["case_number"], 9)
        self.assertEqual(metrics["steady_engine_heavy_rows"][0]["window_label"], "sum@0+3")
        self.assertEqual(metrics["steady_engine_heavy_rows"][0]["runs"], 3)
        self.assertAlmostEqual(
            metrics["steady_engine_heavy_rows"][0]["median_simplify_seconds"], 0.00025
        )

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
        self.assertIn("Sparse wrappers", markdown)
        self.assertIn("combined_additive_zero total=1 failed=0", markdown)
        self.assertIn("Sparse wrapper x complexity buckets", markdown)
        self.assertIn("combined_additive_zero x l0_root_pair total=1 failed=0", markdown)
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
        self.assertIn("Engine hotspots", markdown)
        self.assertIn("Window slices", markdown)
        self.assertIn("Steady-state engine reruns", markdown)
        self.assertIn("#9 sum", markdown)
        self.assertIn("sum total=3 failed=0", markdown)


if __name__ == "__main__":
    unittest.main()
