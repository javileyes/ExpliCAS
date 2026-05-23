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
Distinct shell depths: 5
Largest wrapper share: 25.0%
Largest wrapper x complexity share: 25.0%
Max shell depth: 4
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
  depth 0: total=1 passed=1 failed=0
  depth 1: total=2 passed=2 failed=0
  depth 2: total=3 passed=3 failed=0
  depth 3: total=1 passed=1 failed=0
  depth 4: total=1 passed=1 failed=0

Sparse wrapper x shell-depth buckets:
  combined_additive_zero x depth 0: total=1 passed=1 failed=0
  shifted_quotient_one x depth 1: total=1 passed=1 failed=0
  shifted_quotient_one x depth 2: total=2 passed=2 failed=0
  combined_additive_zero x depth 4: total=1 passed=1 failed=0

Sparse wrapper x shell-depth family buckets:
  combined_additive_zero x depth 0 x simplify: total=1 passed=1 failed=0
  shifted_quotient_one x depth 1 x log_expand: total=1 passed=1 failed=0
  shifted_quotient_one x depth 2 x expand: total=2 passed=2 failed=0
  combined_additive_zero x depth 4 x simplify: total=1 passed=1 failed=0

Sparse wrapper noise-budget rows:
  combined_additive_zero: total=1 passed=1 failed=0 avg_wrapper_overhead_nodes=0.00 max_wrapper_overhead_nodes=0 avg_shell_depth=0.00 max_shell_depth=0
  shifted_quotient_one: total=3 passed=3 failed=0 avg_wrapper_overhead_nodes=8.00 max_wrapper_overhead_nodes=12 avg_shell_depth=1.33 max_shell_depth=2

By complexity level:
  l2_wrapper_plus_noise: total=8 passed=8 failed=0 avg_wrapper_overhead_nodes=10.00 avg_shell_depth=1.75 max_shell_depth=2

Top wrapper x complexity buckets:
  additive_passthrough_zero x l2_wrapper_plus_noise: total=5 passed=5 failed=0 avg_wrapper_overhead_nodes=11.20 avg_shell_depth=2.00 max_shell_depth=2
  combined_additive_zero x l0_root_pair: total=1 passed=1 failed=0 avg_wrapper_overhead_nodes=0.00 avg_shell_depth=0.00 max_shell_depth=0
  shifted_quotient_one x l2_wrapper_plus_noise: total=3 passed=3 failed=0 avg_wrapper_overhead_nodes=8.00 avg_shell_depth=1.33 max_shell_depth=2

Sparse wrapper x complexity family buckets:
  combined_additive_zero x l0_root_pair x simplify: total=1 passed=1 failed=0
  combined_additive_zero x l3_nested_or_composed x simplify: total=1 passed=1 failed=0
  shifted_quotient_one x l2_wrapper_plus_noise x expand: total=2 passed=2 failed=0
  shifted_quotient_one x l2_wrapper_plus_noise x log_expand: total=1 passed=1 failed=0

Sparse wrapper x family buckets:
  combined_additive_zero x simplify: total=1 passed=1 failed=0
  shifted_quotient_one x expand: total=2 passed=2 failed=0
  shifted_quotient_one x log_expand: total=1 passed=1 failed=0

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

SAMPLE_DISCOVERY_LEDGER = """# Engine Combination Ledger

## Current Entries

### 2026-04-27: `integrate_prep` Dirichlet Timeout Discovery

- area:
  - generated discovery / embedded equivalence candidate smoke
  - `combined_additive_zero` x `integrate_prep`
- status:
  - `observe-only discovery`

### 2026-04-27: `radical_power` Passthrough Discovery

- area:
  - generated discovery / embedded equivalence candidate smoke
  - `combined_additive_zero` x `radical_power`
- status:
  - `observe-only discovery`

### 2026-04-24: Local Runtime Patch

- area:
  - orchestrator
- status:
  - `rejected`
"""


class EngineImprovementScorecardTests(unittest.TestCase):
    def test_orchestrator_profile_env_and_command_apply_to_embedded_and_mixed_suites(self):
        args = argparse.Namespace(
            orchestrator_profile=True,
            orchestrator_profile_filter="pipeline.,root.",
            orchestrator_profile_limit=17,
        )

        embedded_env = MODULE.orchestrator_profile_env(
            MODULE.SUITES["embedded_equivalence_context"], args
        )
        embedded_command = MODULE.orchestrator_profile_command(
            MODULE.SUITES["embedded_equivalence_context"], args
        )
        pressure_env = MODULE.orchestrator_profile_env(
            MODULE.SUITES["simplify_zero_mixed"], args
        )
        pressure_command = MODULE.orchestrator_profile_command(
            MODULE.SUITES["simplify_zero_mixed"], args
        )

        self.assertEqual(embedded_env["CAS_PROFILE_ORCHESTRATOR_SHORTCUTS"], "1")
        self.assertEqual(
            embedded_env["CAS_PROFILE_ORCHESTRATOR_SHORTCUT_FILTER"], "pipeline.,root."
        )
        self.assertEqual(embedded_command[-2:], ["--limit", "17"])
        self.assertEqual(pressure_env["CAS_PROFILE_ORCHESTRATOR_SHORTCUTS"], "1")
        self.assertEqual(
            pressure_env["CAS_PROFILE_ORCHESTRATOR_SHORTCUT_FILTER"], "pipeline.,root."
        )
        self.assertEqual(
            pressure_command,
            MODULE.SUITES["simplify_zero_mixed"].command,
        )
        self.assertIsNone(
            MODULE.orchestrator_profile_case_limit(
                MODULE.SUITES["simplify_zero_mixed"], args
            )
        )

    def test_orchestrator_profile_env_and_command_accept_custom_filter_and_limit(self):
        args = argparse.Namespace(
            orchestrator_profile=True,
            orchestrator_profile_filter="rule.direct_identity.",
            orchestrator_profile_limit=9,
        )

        embedded_env = MODULE.orchestrator_profile_env(
            MODULE.SUITES["embedded_equivalence_context"], args
        )
        embedded_command = MODULE.orchestrator_profile_command(
            MODULE.SUITES["embedded_equivalence_context"], args
        )
        mixed_env = MODULE.orchestrator_profile_env(
            MODULE.SUITES["simplify_zero_mixed"], args
        )

        self.assertEqual(
            embedded_env["CAS_PROFILE_ORCHESTRATOR_SHORTCUT_FILTER"],
            "rule.direct_identity.",
        )
        self.assertEqual(embedded_command[-2:], ["--limit", "9"])
        self.assertEqual(
            mixed_env["CAS_PROFILE_ORCHESTRATOR_SHORTCUT_FILTER"],
            "rule.direct_identity.",
        )

    def test_pressure_profile_stays_bounded_and_full_keeps_nf_first(self):
        pressure_args = argparse.Namespace(profile="pressure", suite=[])
        full_args = argparse.Namespace(profile="full", suite=[])

        pressure_names = [spec.name for spec in MODULE.selected_suites(pressure_args)]
        full_names = [spec.name for spec in MODULE.selected_suites(full_args)]

        self.assertEqual(
            pressure_names,
            [
                "simplify_zero_mixed",
                "calculus_diff_exhaustive_contract",
                "calculus_integrate_exhaustive_contract",
            ],
        )
        self.assertIn("simplify_nf_first", full_names)
        self.assertIn("simplify_zero_mixed", full_names)
        self.assertEqual(
            MODULE.SUITES["simplify_nf_first"].timeout_seconds,
            MODULE.NF_FIRST_FULL_TIMEOUT_SECONDS,
        )

    def test_fast_profile_includes_specialized_contextual_radical_lane(self):
        fast_args = argparse.Namespace(profile="fast", suite=[])
        fast_embedded_args = argparse.Namespace(profile="fast_embedded", suite=[])

        fast_names = [spec.name for spec in MODULE.selected_suites(fast_args)]
        fast_embedded_names = [
            spec.name for spec in MODULE.selected_suites(fast_embedded_args)
        ]

        self.assertEqual(
            fast_names,
            [
                "simplify_add_small",
                "contextual_strict_fast",
                "contextual_radical_fast",
                "calculus_diff_contract",
                "calculus_integrate_compact_contract",
                "calculus_residual_matrix_smoke",
            ],
        )
        self.assertIn("contextual_radical_fast", fast_embedded_names)
        self.assertIn("calculus_diff_contract", fast_embedded_names)
        self.assertIn("calculus_integrate_compact_contract", fast_embedded_names)
        self.assertIn("calculus_residual_matrix_smoke", fast_embedded_names)
        self.assertNotIn("calculus_limit_contract", fast_names)
        self.assertNotIn("calculus_limit_presimplify_contract", fast_names)
        self.assertNotIn("calculus_integrate_contract", fast_names)
        self.assertNotIn("calculus_diff_exhaustive_contract", fast_names)
        self.assertNotIn("calculus_integrate_exhaustive_contract", fast_names)

    def test_calculus_contracts_are_visible_in_guardrail_and_full_profiles(self):
        guardrail_args = argparse.Namespace(profile="guardrail", suite=[])
        full_args = argparse.Namespace(profile="full", suite=[])

        guardrail_names = [
            spec.name for spec in MODULE.selected_suites(guardrail_args)
        ]
        full_names = [spec.name for spec in MODULE.selected_suites(full_args)]

        self.assertIn("calculus_diff_contract", guardrail_names)
        self.assertIn("calculus_diff_contract", full_names)
        self.assertNotIn("calculus_diff_exhaustive_contract", guardrail_names)
        self.assertIn("calculus_diff_exhaustive_contract", full_names)
        self.assertIn("calculus_limit_contract", guardrail_names)
        self.assertIn("calculus_limit_contract", full_names)
        self.assertIn("calculus_limit_presimplify_contract", guardrail_names)
        self.assertIn("calculus_limit_presimplify_contract", full_names)
        self.assertIn("calculus_residual_matrix_smoke", guardrail_names)
        self.assertIn("calculus_residual_matrix_smoke", full_names)
        self.assertIn("calculus_integrate_contract", guardrail_names)
        self.assertIn("calculus_integrate_contract", full_names)
        self.assertNotIn("calculus_integrate_exhaustive_contract", guardrail_names)
        self.assertIn("calculus_integrate_exhaustive_contract", full_names)
        self.assertEqual(MODULE.SUITES["calculus_diff_contract"].category, "calculus")
        self.assertEqual(
            MODULE.SUITES["calculus_diff_exhaustive_contract"].category,
            "calculus",
        )
        self.assertEqual(MODULE.SUITES["calculus_limit_contract"].category, "calculus")
        self.assertEqual(
            MODULE.SUITES["calculus_limit_presimplify_contract"].category,
            "calculus",
        )
        self.assertEqual(
            MODULE.SUITES["calculus_integrate_compact_contract"].category,
            "calculus",
        )
        self.assertEqual(
            MODULE.SUITES["calculus_integrate_contract"].category,
            "calculus",
        )
        self.assertEqual(
            MODULE.SUITES["calculus_integrate_exhaustive_contract"].category,
            "calculus",
        )
        self.assertEqual(
            MODULE.SUITES["calculus_residual_matrix_smoke"].category,
            "calculus",
        )
        integrate_compact_command = MODULE.SUITES[
            "calculus_integrate_compact_contract"
        ].command
        self.assertIn("integrate_contract_tests", integrate_compact_command)
        self.assertIn(
            "integrate_contract_polynomial_derivative_over_fractional_denominator_power_substitution",
            integrate_compact_command,
        )
        integrate_command = MODULE.SUITES["calculus_integrate_contract"].command
        self.assertIn("integrate_contract_tests", integrate_command)
        self.assertNotIn("test_enhanced_integration", integrate_command)
        diff_exhaustive_command = MODULE.SUITES[
            "calculus_diff_exhaustive_contract"
        ].command
        self.assertIn("diff_step_contract_tests", diff_exhaustive_command)
        self.assertIn(
            "inverse_reciprocal_trig_diff_evaluates_with_explicit_domain_conditions_exhaustive",
            diff_exhaustive_command,
        )
        self.assertIn("--ignored", diff_exhaustive_command)
        integrate_exhaustive_command = MODULE.SUITES[
            "calculus_integrate_exhaustive_contract"
        ].command
        self.assertIn("integrate_contract_tests", integrate_exhaustive_command)
        self.assertIn(
            "integrate_contract_supported_antiderivatives_verify_by_differentiation_exhaustive",
            integrate_exhaustive_command,
        )
        self.assertIn("--ignored", integrate_exhaustive_command)
        residual_command = MODULE.SUITES["calculus_residual_matrix_smoke"].command
        self.assertIn("engine_calculus_residual_probe_smoke.py", residual_command[1])
        self.assertIn("--default-matrix", residual_command)
        self.assertIn("--ensure-release-cas-cli", residual_command)
        self.assertIn("--summary-json", residual_command)

    def test_parse_calculus_residual_matrix_extracts_status_counts(self):
        metrics = MODULE.parse_calculus_residual_matrix(
            """
{"status":"pass","total":247,"status_counts":{"pass":247,"slow":0,"fail":0,"timeout":0},"issue_kind_counts":{},"expected_required_condition_case_count":42,"distinct_expected_required_conditions":3,"expected_required_condition_counts":{"x + 1":10,"x + 2":20,"sin(x)":12}}
"""
        )

        self.assertEqual(metrics["matrix_status"], "pass")
        self.assertEqual(metrics["total_cases"], 247)
        self.assertEqual(metrics["passed"], 247)
        self.assertEqual(metrics["failed"], 0)
        self.assertEqual(metrics["slow"], 0)
        self.assertEqual(metrics["timeouts"], 0)
        self.assertEqual(metrics["problem_case_count"], 0)
        self.assertEqual(metrics["problem_cases"], [])
        self.assertEqual(metrics["expected_required_condition_case_count"], 42)
        self.assertEqual(metrics["distinct_expected_required_conditions"], 3)
        self.assertEqual(
            metrics["expected_required_condition_counts"],
            {"x + 1": 10, "x + 2": 20, "sin(x)": 12},
        )
        self.assertEqual(
            MODULE.suite_status("calculus_residual_matrix_smoke", metrics, 0),
            "pass",
        )

    def test_parse_calculus_residual_matrix_counts_slow_and_timeout_as_failures(self):
        metrics = MODULE.parse_calculus_residual_matrix(
            """
{"status":"slow","total":3,"status_counts":{"pass":1,"slow":1,"fail":0,"timeout":1},"issue_kind_counts":{"slow":1,"timeout":1},"problem_case_count":2,"problem_cases":[{"name":"slow_case","status":"slow","error_kind":"slow","wall_elapsed_seconds":9.1,"result":"1","required_conditions":["x + 2"],"verbose_payload":{"ignored":true}},{"name":"timeout_case","status":"timeout","error_kind":"timeout","error":"timeout"}]}
"""
        )

        self.assertEqual(metrics["passed"], 1)
        self.assertEqual(metrics["failed"], 2)
        self.assertEqual(metrics["slow"], 1)
        self.assertEqual(metrics["timeouts"], 1)
        self.assertEqual(metrics["problem_case_count"], 2)
        self.assertEqual(
            metrics["problem_cases"],
            [
                {
                    "name": "slow_case",
                    "status": "slow",
                    "error_kind": "slow",
                    "result": "1",
                    "wall_elapsed_seconds": 9.1,
                    "required_conditions": ["x + 2"],
                },
                {
                    "name": "timeout_case",
                    "status": "timeout",
                    "error_kind": "timeout",
                    "error": "timeout",
                },
            ],
        )
        self.assertEqual(metrics["issue_kind_counts"], {"slow": 1, "timeout": 1})
        self.assertEqual(
            MODULE.suite_status("calculus_residual_matrix_smoke", metrics, 0),
            "fail",
        )

    def test_format_runtime_duration_preserves_subsecond_signal(self):
        self.assertEqual(MODULE.format_runtime_duration(0.00025), "0.25ms")
        self.assertEqual(MODULE.format_runtime_duration(0.15), "150.00ms")
        self.assertEqual(MODULE.format_runtime_duration(1.234), "1.23s")

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

    def test_cargo_test_basic_closure_rates_are_parsed_and_rendered(self):
        metrics = MODULE.parse_cargo_test_basic(
            """
running 1 test
✅ Double combinations [add]: 435 passed, 0 failed, 0 skipped (timeout), 0 inconclusive
   📐 NF-convergent: 0 | 🔢 Proved-symbolic: 435 (quotient: 0, diff: 0, composed: 435) | 🌡️ Numeric-only: 0 | ◐ Inconclusive: 0
.
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 138 filtered out; finished in 0.51s
"""
        )
        scorecard = {
            "generated_at": "2026-04-20T00:00:00+00:00",
            "profile": "fast",
            "git": {"branch": "main", "commit": "abc123"},
            "suites": {
                "simplify_add_small": {
                    "status": "pass",
                    "elapsed_seconds": 0.5,
                    "metrics": metrics,
                    "delta": {},
                },
            },
        }

        self.assertEqual(metrics["total_combos"], 435)
        self.assertEqual(metrics["effective_combos"], 435)
        self.assertEqual(metrics["symbolic_closure"], 435)
        self.assertEqual(metrics["symbolic_closure_rate_percent"], 100.0)
        self.assertEqual(metrics["nf_rate_percent"], 0.0)
        self.assertEqual(metrics["proved_symbolic_rate_percent"], 100.0)
        self.assertIn(
            "closure=100.0% nf=0 (0.0%) proved=435 (100.0%)",
            MODULE.render_markdown(scorecard),
        )

    def test_parse_ignored_rust_tests_extracts_names_and_reasons(self):
        source = """
#[test]
#[ignore = "debug slow"]
fn slow_case() {}

    #[ignore]
    fn reasonless_case() {}
"""

        self.assertEqual(
            MODULE.parse_ignored_rust_tests(source),
            [
                {"name": "slow_case", "reason": "debug slow"},
                {"name": "reasonless_case", "reason": ""},
            ],
        )

    def test_parse_derive_extracts_strategy_specificity_metrics(self):
        metrics = MODULE.parse_derive(
            """
derive corpus summary: derived=291 unsupported=0 not_equivalent=1
derive stats: reachability_rate=1.000 supported_equiv_rate=1.000 mean_step_count=1.05 long_path_rate=0.000
derive path signal: single_step_successes=284 multi_step_successes=7 max_step_count=3
derive multi-step-ids: expand_two_stage:2,contract_log_chain:3
derive strategy specificity: generic_simplify_expected=0 distinct_expected_strategies=27
derive expected-strategy-counts: {"expand": 30}
derive derived-by-family: {"expand": 30}
derive unsupported-equivalent-by-family: {}
derive not-equivalent-by-family: {"negative": 1}
"""
        )

        self.assertEqual(metrics["derived"], 291)
        self.assertEqual(metrics["generic_simplify_expected"], 0)
        self.assertEqual(metrics["distinct_expected_strategies"], 27)
        self.assertEqual(metrics["single_step_successes"], 284)
        self.assertEqual(metrics["multi_step_successes"], 7)
        self.assertEqual(
            metrics["multi_step_success_ids"],
            ["expand_two_stage:2", "contract_log_chain:3"],
        )
        self.assertEqual(metrics["max_step_count"], 3)
        self.assertEqual(metrics["expected_strategy_counts"], {"expand": 30})
        self.assertEqual(metrics["derived_by_family"]["expand"], 30)
        self.assertEqual(metrics["unsupported_by_family"], {})
        self.assertEqual(metrics["not_equivalent_by_family"], {"negative": 1})

    def test_parse_derive_shadow_extracts_strategy_specificity_metrics(self):
        metrics = MODULE.parse_derive_shadow(
            """
derive shadow pressure summary: sampled=16 derived=16 unsupported=0 not_equivalent=0
derive shadow pressure stats: reachability_rate=1.000 mean_step_count=1.19 single_step_successes=13 multi_step_successes=3
derive shadow pressure strategy specificity: generic_simplify_strategy_successes=2 distinct_actual_strategies=9
derive shadow pressure generic-simplify-ids: identity_a,identity_b
derive shadow pressure multi-step-ids: identity_a:2,embedded_calculus_diff_arctan_sqrt_compact:4,identity_c:3
derive shadow pressure actual-strategy-counts: {"simplify": 2, "contract logs": 1}
derive shadow pressure derived-by-family: {"log_contract": 1, "simplify": 2}
derive shadow pressure unsupported-equivalent-by-family: {"branch_sensitive": 1}
derive shadow pressure not-equivalent-by-family: {}
"""
        )

        self.assertEqual(metrics["sampled"], 16)
        self.assertEqual(metrics["generic_simplify_strategy_successes"], 2)
        self.assertEqual(
            metrics["generic_simplify_strategy_ids"], ["identity_a", "identity_b"]
        )
        self.assertEqual(
            metrics["multi_step_success_ids"],
            [
                "identity_a:2",
                "embedded_calculus_diff_arctan_sqrt_compact:4",
                "identity_c:3",
            ],
        )
        self.assertEqual(
            metrics["multi_step_success_step_counts"],
            {
                "identity_a": 2,
                "embedded_calculus_diff_arctan_sqrt_compact": 4,
                "identity_c": 3,
            },
        )
        self.assertEqual(metrics["max_step_count"], 4)
        self.assertEqual(metrics["distinct_actual_strategies"], 9)
        self.assertEqual(metrics["actual_strategy_counts"]["contract logs"], 1)
        self.assertEqual(metrics["derived_by_family"]["log_contract"], 1)
        self.assertEqual(metrics["unsupported_by_family"]["branch_sensitive"], 1)
        self.assertEqual(metrics["not_equivalent_by_family"], {})

    def test_parse_simplify_didactic_extracts_trace_quality_metrics(self):
        metrics = MODULE.parse_simplify_didactic(
            """
simplify didactic audit summary: cases=14 flagged=1 no_wire_substeps=1 single_step_no_substeps=0 missing_math_sides=0 total_wire_substeps=20 mean_step_count=2.43
didactic audit report written to docs/generated/DIDACTIC_STEP_QUALITY_AUDIT_REPORT.md
"""
        )

        self.assertEqual(metrics["cases"], 14)
        self.assertEqual(metrics["flagged_cases"], 1)
        self.assertAlmostEqual(metrics["flagged_rate"], 1 / 14)
        self.assertEqual(metrics["no_wire_substeps"], 1)
        self.assertEqual(metrics["single_step_no_substeps"], 0)
        self.assertEqual(metrics["missing_math_sides"], 0)
        self.assertEqual(metrics["total_wire_substeps"], 20)
        self.assertEqual(metrics["mean_step_count"], 2.43)

    def test_parse_derive_didactic_extracts_trace_quality_metrics(self):
        metrics = MODULE.parse_derive_didactic(
            """
derive didactic audit summary: cases=403 flagged=82 no_web_substeps=82 no_web_steps=0 total_web_substeps=320 mean_step_count=1.06
wrote docs/generated/DERIVE_DIDACTIC_AUDIT.md
"""
        )

        self.assertEqual(metrics["cases"], 403)
        self.assertEqual(metrics["flagged_cases"], 82)
        self.assertAlmostEqual(metrics["flagged_rate"], 82 / 403)
        self.assertEqual(metrics["no_web_substeps"], 82)
        self.assertEqual(metrics["no_web_steps"], 0)
        self.assertEqual(metrics["total_web_substeps"], 320)
        self.assertEqual(metrics["mean_step_count"], 1.06)

    def test_parse_corpus_extracts_complexity_and_orchestrator_profile_metrics(self):
        metrics = MODULE.parse_corpus(SAMPLE_CORPUS_OUTPUT)

        self.assertEqual(metrics["complexity_level_count"], 1)
        self.assertEqual(metrics["shell_depth_count"], 5)
        self.assertEqual(metrics["max_shell_depth"], 4)
        self.assertEqual(metrics["max_expression_depth"], 5)
        self.assertEqual(metrics["average_wrapper_overhead_nodes"], 10.0)
        self.assertAlmostEqual(metrics["reported_elapsed_per_case_ms"], 2.1375)
        self.assertEqual(
            metrics["complexity_rows"]["l2_wrapper_plus_noise"]["avg_shell_depth"], 1.75
        )
        self.assertEqual(metrics["shell_depth_rows"][2]["total"], 3)
        self.assertEqual(metrics["shell_depth_rows"][4]["total"], 1)
        self.assertEqual(metrics["wrapper_shell_depth_rows"][0]["wrapper"], "combined_additive_zero")
        self.assertEqual(metrics["wrapper_shell_depth_rows"][0]["shell_depth"], 0)
        self.assertEqual(metrics["wrapper_shell_depth_rows"][2]["total"], 2)
        self.assertEqual(metrics["wrapper_shell_depth_rows"][3]["shell_depth"], 4)
        self.assertEqual(
            metrics["wrapper_shell_depth_family_rows"][0]["wrapper"],
            "combined_additive_zero",
        )
        self.assertEqual(metrics["wrapper_shell_depth_family_rows"][0]["shell_depth"], 0)
        self.assertEqual(metrics["wrapper_shell_depth_family_rows"][2]["family"], "expand")
        self.assertEqual(metrics["wrapper_shell_depth_family_rows"][2]["total"], 2)
        self.assertEqual(metrics["wrapper_shell_depth_family_rows"][3]["shell_depth"], 4)
        self.assertEqual(metrics["sparse_wrapper_noise_budget_rows"][0]["wrapper"], "combined_additive_zero")
        self.assertEqual(
            metrics["sparse_wrapper_noise_budget_rows"][1]["avg_wrapper_overhead_nodes"],
            8.0,
        )
        self.assertEqual(
            metrics["sparse_wrapper_noise_budget_rows"][1]["max_wrapper_overhead_nodes"],
            12,
        )
        self.assertEqual(metrics["wrapper_rows"]["shifted_quotient_one"]["total"], 3)
        self.assertEqual(metrics["wrapper_complexity_rows"][0]["wrapper"], "additive_passthrough_zero")
        self.assertEqual(
            metrics["wrapper_complexity_rows"][0]["avg_wrapper_overhead_nodes"], 11.2
        )
        self.assertEqual(
            metrics["wrapper_complexity_family_rows"][0]["wrapper"],
            "combined_additive_zero",
        )
        self.assertEqual(
            metrics["wrapper_complexity_family_rows"][1]["level"],
            "l3_nested_or_composed",
        )
        self.assertEqual(metrics["wrapper_complexity_family_rows"][2]["family"], "expand")
        self.assertEqual(metrics["wrapper_family_rows"][0]["wrapper"], "combined_additive_zero")
        self.assertEqual(metrics["wrapper_family_rows"][0]["family"], "simplify")
        self.assertEqual(metrics["wrapper_family_rows"][1]["total"], 2)
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

    def test_shell_depth_summary_keeps_small_intermediate_buckets(self):
        rows = {depth: {"total": depth + 1, "failed": 0} for depth in range(6)}

        summary = MODULE.shell_depth_summary_rows(rows)

        self.assertEqual([depth for depth, _row in summary], [0, 1, 2, 3, 4, 5])

    def test_parse_orchestrator_profile_recovers_truncated_section_samples(self):
        output = """Orchestrator Profiling Report
──────────────────────────────────────────────────────────────────────────────────────────────
Section                                          Attempts     Hits   Misses     Total ms       Avg us
──────────────────────────────────────────────────────────────────────────────────────────────
root.direct_small_zero_composition.candidate.th…       75       19       56       71.683       955.78
TOTAL                                                  75       19       56       71.683       955.78
──────────────────────────────────────────────────────────────────────────────────────────────
Sample expressions
──────────────────────────────────────────────────────────────────────────────────────────────
root.direct_small_zero_composition.candidate.three_core_groups
  - sub(add, add)
"""

        profile = MODULE.parse_orchestrator_profile(output)

        assert profile is not None
        row = profile["top_hot_sections"][0]
        self.assertEqual(
            row["section"],
            "root.direct_small_zero_composition.candidate.three_core_groups",
        )
        self.assertEqual(row["samples"], ["sub(add, add)"])

    def test_parse_orchestrator_profile_splits_outcome_labeled_samples(self):
        output = """Orchestrator Profiling Report
──────────────────────────────────────────────────────────────────────────────────────────────
Section                                          Attempts     Hits   Misses     Total ms       Avg us
──────────────────────────────────────────────────────────────────────────────────────────────
root.direct_small_zero_composition.candidate.th…        2        1        1        2.000      1000.00
TOTAL                                                   2        1        1        2.000      1000.00
──────────────────────────────────────────────────────────────────────────────────────────────
Sample expressions
──────────────────────────────────────────────────────────────────────────────────────────────
root.direct_small_zero_composition.candidate.three_core_groups
  - hit: terms=6 [+sin:function -cos:function]
  - miss: terms=6 [+log:function -div:div]
"""

        profile = MODULE.parse_orchestrator_profile(output)

        assert profile is not None
        row = profile["top_no_match_cost_sections"][0]
        self.assertEqual(
            row["section"],
            "root.direct_small_zero_composition.candidate.three_core_groups",
        )
        self.assertEqual(row["hit_samples"], ["terms=6 [+sin:function -cos:function]"])
        self.assertEqual(row["miss_samples"], ["terms=6 [+log:function -div:div]"])
        self.assertEqual(
            MODULE.orchestrator_profile_sample_suffix(row, prefer_miss=True),
            " miss_sample=`terms=6 [+log:function -div:div]`",
        )

    def test_render_markdown_includes_embedded_orchestrator_profile_summary(self):
        metrics = MODULE.parse_corpus(SAMPLE_CORPUS_OUTPUT)
        metrics["orchestrator_profile_slice"] = {
            "elapsed_seconds": 0.043,
            "family_count": 2,
            "filter": "pipeline.,root.",
            "limit": 17,
            "total_cases": 8,
            "wrapper_count": 3,
        }
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

        self.assertIn("- Per-case runtime: 2.138ms/case", markdown)
        self.assertIn("## Embedded Orchestrator Profile", markdown)
        self.assertIn(
            "Profiled slice: 8 cases (limit 17), 3 wrappers, 2 families, 0.04s elapsed, filter `pipeline.,root.`.",
            markdown,
        )
        self.assertIn("Shell-depth mix", markdown)
        self.assertIn("depth 4 total=1 failed=0", markdown)
        self.assertIn("Sparse wrapper x shell-depth buckets", markdown)
        self.assertIn("shifted_quotient_one x depth 2 total=2 failed=0", markdown)
        self.assertIn("Sparse wrapper x shell-depth family buckets", markdown)
        self.assertIn("shifted_quotient_one x depth 2 x expand total=2 failed=0", markdown)
        self.assertIn("Sparse wrapper depth 4 family breadth", markdown)
        self.assertIn(
            "combined_additive_zero depth4_families=1/2 missing=1 cases=1",
            markdown,
        )
        self.assertIn("Sparse wrapper noise budgets", markdown)
        self.assertIn("shifted_quotient_one total=3 failed=0 avg_overhead=8.00 max_overhead=12", markdown)
        self.assertIn("Sparse wrappers", markdown)
        self.assertIn("combined_additive_zero total=1 failed=0", markdown)
        self.assertIn("Sparse wrapper x complexity buckets", markdown)
        self.assertIn("combined_additive_zero x l0_root_pair total=1 failed=0", markdown)
        self.assertIn("Dominant wrapper x complexity buckets", markdown)
        self.assertIn("Sparse wrapper l3 family breadth", markdown)
        self.assertIn("combined_additive_zero l3_families=1/2 missing=1 cases=1", markdown)
        self.assertIn("Sparse wrapper family breadth", markdown)
        self.assertIn("combined_additive_zero families=1/2 cases=1", markdown)
        self.assertIn("Sparse wrapper family gaps", markdown)
        self.assertIn(
            "combined_additive_zero missing_families=1/2 covered=1 cases=1",
            markdown,
        )
        self.assertIn("Sparse wrapper x family buckets", markdown)
        self.assertIn("combined_additive_zero x simplify total=1 failed=0", markdown)
        self.assertIn("pipeline.phase.core", markdown)
        self.assertIn("No-match hotspot 1", markdown)

    def test_render_markdown_does_not_truncate_sparse_wrapper_family_rows(self):
        metrics = MODULE.parse_corpus(SAMPLE_CORPUS_OUTPUT)
        metrics["wrapper_family_rows"] = [
            {
                "wrapper": "combined_additive_zero",
                "family": f"family_{idx:02}",
                "total": 1,
                "passed": 1,
                "failed": 0,
            }
            for idx in range(13)
        ]
        metrics["family_count"] = 13
        scorecard = {
            "generated_at": "2026-04-20T00:00:00+00:00",
            "profile": "guardrail",
            "git": {"branch": "main", "commit": "abc123"},
            "suites": {
                "embedded_equivalence_context": {
                    "status": "pass",
                    "elapsed_seconds": 0.017,
                    "metrics": metrics,
                    "guardrail": None,
                    "delta": {},
                }
            },
        }

        markdown = MODULE.render_markdown(scorecard)

        self.assertIn("combined_additive_zero families=13/13 cases=13", markdown)
        self.assertIn(
            "combined_additive_zero missing_families=0/13 covered=13 cases=13",
            markdown,
        )
        self.assertIn("combined_additive_zero x family_00 total=1 failed=0", markdown)
        self.assertIn("combined_additive_zero x family_12 total=1 failed=0", markdown)

    def test_combined_additive_summary_names_low_families_when_many_are_at_min(self):
        summary = MODULE.combined_additive_structure_summary(
            {
                "family_count": 10,
                "combined_additive_zero": {
                    "total": 40,
                    "family_count": 10,
                    "collapse_rows": 30,
                    "collapse_family_count": 10,
                    "depth4_rows": 10,
                    "depth4_family_count": 10,
                    "orientation_rows": 10,
                    "orientation_family_count": 10,
                    "multi_core_rows": 10,
                    "multi_core_family_count": 10,
                    "min_family_case_count": 4,
                    "target_family_case_count": 6,
                    "low_family_count": 9,
                    "low_family_counts": {
                        f"family_{idx:02}": 4 for idx in range(9)
                    },
                    "under_target_family_count": 9,
                    "under_target_family_counts": {
                        f"family_{idx:02}": 4 for idx in range(9)
                    },
                    "above_min_family_counts": {"family_09": 5},
                },
            }
        )

        self.assertIn("families_at_min=9/10", summary)
        self.assertIn("target_family_case_count=6", summary)
        self.assertIn("families_under_target=9/10", summary)
        self.assertIn("low_family_counts=family_00:4", summary)
        self.assertIn("under_target_family_counts=family_00:4", summary)
        self.assertIn("family_08:4", summary)
        self.assertIn("above_min_family_counts=family_09:5", summary)

    def test_embedded_coverage_saturation_marks_balanced_axes(self):
        metrics = MODULE.parse_corpus(SAMPLE_CORPUS_OUTPUT)
        metrics.update(
            {
                "corpus_structure": {
                    "family_count": 2,
                    "combined_additive_zero": {
                        "family_counts": {
                            "calculus_diff": 6,
                            "calculus_integrate": 6,
                        },
                        "under_target_family_counts": {},
                        "depth4_missing_families": [],
                        "orientation_missing_families": [],
                        "multi_core_missing_families": [],
                    },
                },
                "wrapper_shell_depth_family_rows": [
                    {
                        "wrapper": "reciprocal_shifted_difference_zero",
                        "family": "calculus_diff",
                        "shell_depth": 4,
                        "total": 1,
                        "passed": 1,
                        "failed": 0,
                    },
                    {
                        "wrapper": "reciprocal_shifted_difference_zero",
                        "family": "calculus_integrate",
                        "shell_depth": 4,
                        "total": 1,
                        "passed": 1,
                        "failed": 0,
                    },
                ],
            }
        )

        saturation = MODULE.embedded_coverage_saturation_metrics(metrics)

        self.assertEqual(saturation["status"], "balanced")
        self.assertEqual(saturation["checked_family_count"], 2)
        self.assertEqual(saturation["balanced_check_count"], 5)
        self.assertEqual(saturation["open_check_count"], 0)
        self.assertIn("defer embedded corpus padding", saturation["recommendation"])

        scorecard = {
            "generated_at": "2026-04-20T00:00:00+00:00",
            "profile": "guardrail",
            "git": {"branch": "main", "commit": "abc123"},
            "suites": {
                "embedded_equivalence_context": {
                    "status": "pass",
                    "elapsed_seconds": 1.0,
                    "metrics": {**metrics, "coverage_saturation": saturation},
                    "delta": {},
                },
            },
        }

        markdown = MODULE.render_markdown(scorecard)

        self.assertIn("Embedded coverage saturation: balanced (5/5 checks)", markdown)
        self.assertIn("prefer calculus, robustness, or presentation", markdown)

    def test_embedded_coverage_saturation_reports_reciprocal_depth4_gap(self):
        metrics = {
            "corpus_structure": {
                "family_count": 2,
                "combined_additive_zero": {
                    "family_counts": {
                        "calculus_diff": 6,
                        "calculus_integrate": 6,
                    },
                    "under_target_family_counts": {},
                    "depth4_missing_families": [],
                    "orientation_missing_families": [],
                    "multi_core_missing_families": [],
                },
            },
            "wrapper_shell_depth_family_rows": [
                {
                    "wrapper": "reciprocal_shifted_difference_zero",
                    "family": "calculus_diff",
                    "shell_depth": 4,
                    "total": 1,
                    "passed": 1,
                    "failed": 0,
                },
            ],
        }

        saturation = MODULE.embedded_coverage_saturation_metrics(metrics)

        self.assertEqual(saturation["status"], "needs_attention")
        self.assertEqual(saturation["balanced_check_count"], 4)
        self.assertEqual(saturation["open_check_count"], 1)
        reciprocal_check = saturation["checks"][-1]
        self.assertEqual(
            reciprocal_check["name"], "reciprocal_shifted_difference_depth4"
        )
        self.assertEqual(
            reciprocal_check["missing_families"], ["calculus_integrate"]
        )

    def test_generated_discovery_ledger_metrics_and_markdown(self):
        metrics = MODULE.parse_generated_discovery_ledger(SAMPLE_DISCOVERY_LEDGER)

        self.assertEqual(metrics["observe_only_discoveries"], 2)
        self.assertEqual(
            metrics["areas"]["generated discovery / embedded equivalence candidate smoke"],
            2,
        )
        self.assertEqual(metrics["families"]["integrate_prep"], 1)
        self.assertEqual(metrics["families"]["radical_power"], 1)
        self.assertEqual(metrics["wrappers"]["combined_additive_zero"], 2)
        self.assertEqual(
            metrics["recent"][0]["area"],
            "generated discovery / embedded equivalence candidate smoke",
        )
        self.assertEqual(metrics["recent"][0]["family"], "integrate_prep")

        scorecard = {
            "generated_at": "2026-04-20T00:00:00+00:00",
            "profile": "guardrail",
            "git": {"branch": "main", "commit": "abc123"},
            "generated_discovery": metrics,
            "suites": {},
        }
        markdown = MODULE.render_markdown(scorecard)

        self.assertIn("## Generated Discovery Ledger", markdown)
        self.assertIn("Observe-only discoveries: total=2", markdown)
        self.assertIn(
            "generated discovery / embedded equivalence candidate smoke:2",
            markdown,
        )
        self.assertIn("integrate_prep:1", markdown)
        self.assertIn("combined_additive_zero:2", markdown)
        self.assertIn("Recent 1: `integrate_prep`", markdown)

    def test_generated_discovery_ledger_parses_current_heading_and_status_variants(self):
        text = """# Engine Combination Ledger

## Current Entries

## 2026-05-21 - Discovery observe-only: inline sqrt-variable tan-root presentation residual timeout

- area:
  - calculus / differentiation
- status:
  - `discovery/observe-only`

## 2026-05-20 - Observe-only discovery: non-corpus rational quadratic wrapper warning

- area:
  - generated discovery
  - `combined_additive_zero` x `integrate_prep`
- status:
  - `observe-only`

## 2026-05-20 - Retained robustness: resolved discovery

- status:
  - `retained`
- retained learning:
  - mentions a previous observe-only discovery without reopening it

## 2026-05-19 - Discovery observe-only: closed generated candidate

- area:
  - generated discovery
  - `combined_additive_zero` x `collect`
- status:
  - `discovery/observe-only`
- resolved by:
  - later retained coverage made this candidate obsolete

## 2026-05-18 - Discovery observe-only: superseded generated candidate

- area:
  - generated discovery
  - `combined_additive_zero` x `log_contract`
- status:
  - `superseded`

## 2026-04-27: `radical_power` Passthrough Discovery

- area:
  - generated discovery
  - `squared_passthrough_zero` x `radical_power`
- status:
  - `discovery-observe-only`
"""
        metrics = MODULE.parse_generated_discovery_ledger(text)

        self.assertEqual(metrics["observe_only_discoveries"], 3)
        self.assertEqual(metrics["areas"]["calculus / differentiation"], 1)
        self.assertEqual(metrics["areas"]["generated discovery"], 2)
        self.assertEqual(metrics["families"]["integrate_prep"], 1)
        self.assertEqual(metrics["families"]["radical_power"], 1)
        self.assertNotIn("log_contract", metrics["families"])
        self.assertEqual(metrics["wrappers"]["combined_additive_zero"], 1)
        self.assertEqual(metrics["wrappers"]["squared_passthrough_zero"], 1)
        self.assertEqual(metrics["recent"][0]["area"], "calculus / differentiation")
        self.assertEqual(metrics["recent"][0]["family"], "unknown")
        self.assertEqual(metrics["recent"][1]["family"], "integrate_prep")

        scorecard = {
            "generated_at": "2026-04-20T00:00:00+00:00",
            "profile": "guardrail",
            "git": {"branch": "main", "commit": "abc123"},
            "generated_discovery": metrics,
            "suites": {},
        }
        markdown = MODULE.render_markdown(scorecard)

        self.assertIn(
            "Recent 1: `calculus / differentiation` - 2026-05-21 - Discovery observe-only",
            markdown,
        )
        self.assertIn("Recent 2: `integrate_prep` in `combined_additive_zero`", markdown)
        self.assertNotIn("`unknown` in `unknown`", markdown)

    def test_generated_discovery_area_bucket_normalizes_calculus_command_aliases(self):
        self.assertEqual(
            MODULE.discovery_area_bucket("calculus / integrate / residual verification"),
            "calculus / integration",
        )
        self.assertEqual(
            MODULE.discovery_area_bucket("calculus / diff / post-calculus presentation"),
            "calculus / differentiation",
        )
        self.assertEqual(
            MODULE.discovery_area_bucket("calculus / diff runtime / inverse tangent"),
            "calculus / diff runtime",
        )
        self.assertEqual(
            MODULE.discovery_area_bucket("calculus / integration contract / nested log"),
            "calculus / integration contract",
        )

    def test_generated_discovery_ledger_markdown_reports_clean_state(self):
        metrics = MODULE.parse_generated_discovery_ledger(
            "### 2026-04-27: Resolved Discovery\n\n"
            "- status:\n"
            "  - `resolved`\n"
        )

        self.assertEqual(metrics["observe_only_discoveries"], 0)

        scorecard = {
            "generated_at": "2026-04-20T00:00:00+00:00",
            "profile": "guardrail",
            "git": {"branch": "main", "commit": "abc123"},
            "generated_discovery": metrics,
            "suites": {},
        }
        markdown = MODULE.render_markdown(scorecard)

        self.assertIn("## Generated Discovery Ledger", markdown)
        self.assertIn("Observe-only discoveries: total=0", markdown)
        self.assertIn("no open observe-only generated discoveries", markdown)
        self.assertNotIn("- By family:", markdown)
        self.assertNotIn("- Recent 1:", markdown)

    def test_generated_discovery_pressure_marks_low_family_intersection(self):
        metrics = MODULE.parse_generated_discovery_ledger(SAMPLE_DISCOVERY_LEDGER)
        scorecard = {
            "generated_at": "2026-04-20T00:00:00+00:00",
            "profile": "guardrail",
            "git": {"branch": "main", "commit": "abc123"},
            "generated_discovery": metrics,
            "suites": {
                "embedded_equivalence_context": {
                    "status": "pass",
                    "elapsed_seconds": 1.0,
                    "metrics": {
                        "total_cases": 2,
                        "passed": 2,
                        "failed": 0,
                        "corpus_structure": {
                            "family_count": 3,
                            "combined_additive_zero": {
                                "low_family_counts": {
                                    "collect": 4,
                                    "integrate_prep": 4,
                                }
                            },
                        },
                    },
                    "delta": {},
                }
            },
        }

        pressure = MODULE.generated_discovery_pressure_metrics(scorecard)

        self.assertEqual(pressure["low_family_count"], 2)
        self.assertEqual(pressure["blocked_low_family_count"], 1)
        self.assertEqual(
            pressure["blocked_low_families"]["integrate_prep"]["live_count"],
            4,
        )
        self.assertEqual(
            pressure["blocked_low_families"]["integrate_prep"][
                "observe_only_discoveries"
            ],
            1,
        )
        self.assertEqual(pressure["unblocked_low_families"]["collect"], 4)

        markdown = MODULE.render_markdown(scorecard)

        self.assertIn("Low-family discovery pressure: blocked=1/2", markdown)
        self.assertIn("integrate_prep:live=4,observe_only=1", markdown)

    def test_generated_discovery_pressure_ignores_balanced_floor_families(self):
        metrics = MODULE.parse_generated_discovery_ledger(SAMPLE_DISCOVERY_LEDGER)
        scorecard = {
            "generated_at": "2026-04-20T00:00:00+00:00",
            "profile": "guardrail",
            "git": {"branch": "main", "commit": "abc123"},
            "generated_discovery": metrics,
            "suites": {
                "embedded_equivalence_context": {
                    "status": "pass",
                    "elapsed_seconds": 1.0,
                    "metrics": {
                        "total_cases": 2,
                        "passed": 2,
                        "failed": 0,
                        "corpus_structure": {
                            "family_count": 3,
                            "combined_additive_zero": {
                                "low_family_counts": {
                                    "collect": 6,
                                    "integrate_prep": 6,
                                },
                                "under_target_family_counts": {},
                            },
                        },
                    },
                    "delta": {},
                }
            },
        }

        pressure = MODULE.generated_discovery_pressure_metrics(scorecard)

        self.assertEqual(pressure["low_family_count"], 0)
        self.assertEqual(pressure["blocked_low_family_count"], 0)

        markdown = MODULE.render_markdown(scorecard)

        self.assertNotIn("Low-family discovery pressure", markdown)

    def test_render_markdown_includes_derive_shadow_count_maps(self):
        scorecard = {
            "generated_at": "2026-04-20T00:00:00+00:00",
            "profile": "guardrail",
            "git": {"branch": "main", "commit": "abc123"},
            "suites": {
                "derive_shadow_pressure": {
                    "status": "pass",
                    "elapsed_seconds": 0.1,
                    "metrics": {
                        "sampled": 3,
                        "derived": 2,
                        "unsupported": 1,
                        "not_equivalent": 0,
                        "mean_step_count": 1.5,
                        "single_step_successes": 1,
                        "multi_step_successes": 1,
                        "multi_step_success_ids": ["nested_fraction_case:4"],
                        "multi_step_success_step_counts": {
                            "nested_fraction_case": 4,
                        },
                        "max_step_count": 4,
                        "generic_simplify_strategy_successes": 0,
                        "generic_simplify_strategy_ids": [],
                        "distinct_actual_strategies": 2,
                        "actual_strategy_counts": {
                            "contract logs": 1,
                            "nested fraction": 1,
                        },
                        "derived_by_family": {
                            "log_contract": 1,
                            "nested_fraction": 1,
                        },
                        "unsupported_by_family": {"branch_sensitive": 1},
                        "not_equivalent_by_family": {},
                    },
                    "delta": {},
                },
            },
        }

        markdown = MODULE.render_markdown(scorecard)

        self.assertIn("## Derive Shadow Pressure", markdown)
        self.assertIn("Actual strategy counts", markdown)
        self.assertIn("contract logs:1", markdown)
        self.assertIn("max_step_count=4", markdown)
        self.assertIn("multi_step_ids=1 max_step=4", markdown)
        self.assertIn("Derived family counts", markdown)
        self.assertIn("log_contract:1", markdown)
        self.assertIn("unsupported=branch_sensitive:1", markdown)
        self.assertIn("not_equivalent=none", markdown)

    def test_render_markdown_labels_derive_expected_negative_controls(self):
        scorecard = {
            "generated_at": "2026-04-20T00:00:00+00:00",
            "profile": "guardrail",
            "git": {"branch": "main", "commit": "abc123"},
            "suites": {
                "derive_contract": {
                    "status": "pass",
                    "elapsed_seconds": 0.2,
                    "metrics": {
                        "derived": 10,
                        "unsupported": 0,
                        "not_equivalent": 1,
                        "reachability_rate": 1.0,
                        "supported_equiv_rate": 1.0,
                        "mean_step_count": 1.2,
                        "long_path_rate": 0.0,
                        "single_step_successes": 8,
                        "multi_step_successes": 2,
                        "multi_step_success_ids": ["case_a:2", "case_b:3"],
                        "max_step_count": 3,
                        "generic_simplify_expected": 0,
                        "distinct_expected_strategies": 4,
                        "expected_strategy_counts": {"expand trig": 4, "factor": 2},
                        "unsupported_by_family": {},
                        "not_equivalent_by_family": {"negative": 1},
                    },
                    "delta": {},
                },
            },
        }

        markdown = MODULE.render_markdown(scorecard)

        self.assertIn("## Derive Reachability Guardrail", markdown)
        self.assertIn(
            "Expected-status breakdown: derived=10 unsupported=0 not_equivalent=1",
            markdown,
        )
        self.assertIn(
            "Non-derived expected families: unsupported=none not_equivalent=negative:1",
            markdown,
        )
        self.assertIn(
            "Expected strategy counts: expand trig:4, factor:2",
            markdown,
        )
        self.assertIn(
            "Path quality: mean_step_count=1.20 long_path_rate=0.00 "
            "single_step_successes=8 multi_step_successes=2 max_step_count=3",
            markdown,
        )
        self.assertIn(
            "| `derive_contract` | `pass` | 0.20s | derived=10 unsupported=0 "
            "expected_not_equivalent=1 mean_step_count=1.20 "
            "generic_simplify_expected=0 single_step=8 multi_step_ids=2 max_step=3",
            markdown,
        )

    def test_render_markdown_includes_simplify_didactic_trace_audit(self):
        scorecard = {
            "generated_at": "2026-04-20T00:00:00+00:00",
            "profile": "guardrail",
            "git": {"branch": "main", "commit": "abc123"},
            "suites": {
                "simplify_didactic_audit": {
                    "status": "warn",
                    "elapsed_seconds": 1.2,
                    "metrics": {
                        "cases": 14,
                        "flagged_cases": 1,
                        "flagged_rate": 1 / 14,
                        "no_wire_substeps": 1,
                        "single_step_no_substeps": 0,
                        "missing_math_sides": 0,
                        "total_wire_substeps": 20,
                        "mean_step_count": 2.43,
                    },
                    "delta": {},
                },
            },
        }

        markdown = MODULE.render_markdown(scorecard)

        self.assertIn("## Simplify Didactic Trace Audit", markdown)
        self.assertIn("cases=14 flagged=1", markdown)
        self.assertIn("flagged_rate=7.1%", markdown)
        self.assertIn("total_wire_substeps=20", markdown)
        self.assertIn("no_wire_substeps=1", markdown)

    def test_render_markdown_includes_derive_didactic_trace_audit(self):
        scorecard = {
            "generated_at": "2026-04-20T00:00:00+00:00",
            "profile": "guardrail",
            "git": {"branch": "main", "commit": "abc123"},
            "suites": {
                "derive_didactic_audit": {
                    "status": "warn",
                    "elapsed_seconds": 2.8,
                    "metrics": {
                        "cases": 403,
                        "flagged_cases": 82,
                        "flagged_rate": 82 / 403,
                        "no_web_substeps": 82,
                        "no_web_steps": 0,
                        "total_web_substeps": 320,
                        "mean_step_count": 1.06,
                    },
                    "delta": {},
                },
            },
        }

        markdown = MODULE.render_markdown(scorecard)

        self.assertIn("## Derive Didactic Trace Audit", markdown)
        self.assertIn("cases=403 flagged=82", markdown)
        self.assertIn("flagged_rate=20.3%", markdown)
        self.assertIn("total_web_substeps=320", markdown)
        self.assertIn("no_web_substeps=82", markdown)

    def test_render_markdown_includes_calculus_diff_contract_signal(self):
        scorecard = {
            "generated_at": "2026-04-20T00:00:00+00:00",
            "profile": "guardrail",
            "git": {"branch": "main", "commit": "abc123"},
            "suites": {
                "calculus_diff_contract": {
                    "status": "pass",
                    "elapsed_seconds": 0.1,
                    "metrics": {
                        "cargo_status": "ok",
                        "passed": 30,
                        "failed": 0,
                        "ignored": 1,
                        "measured": 0,
                        "filtered_out": 0,
                        "timeouts": 0,
                        "ignored_tests": [
                            {
                                "name": "inverse_reciprocal_trig_diff_exhaustive",
                                "reason": "debug-slow",
                            }
                        ],
                    },
                    "delta": {},
                },
                "calculus_diff_exhaustive_contract": {
                    "status": "pass",
                    "elapsed_seconds": 0.7,
                    "metrics": {
                        "cargo_status": "ok",
                        "passed": 1,
                        "failed": 0,
                        "ignored": 0,
                        "measured": 0,
                        "filtered_out": 256,
                        "timeouts": 0,
                    },
                    "delta": {},
                },
                "calculus_integrate_contract": {
                    "status": "pass",
                    "elapsed_seconds": 0.2,
                    "metrics": {
                        "cargo_status": "ok",
                        "passed": 23,
                        "failed": 0,
                        "ignored": 0,
                        "measured": 0,
                        "filtered_out": 0,
                        "timeouts": 0,
                    },
                    "delta": {},
                },
                "calculus_integrate_exhaustive_contract": {
                    "status": "pass",
                    "elapsed_seconds": 1.7,
                    "metrics": {
                        "cargo_status": "ok",
                        "passed": 1,
                        "failed": 0,
                        "ignored": 0,
                        "measured": 0,
                        "filtered_out": 321,
                        "timeouts": 0,
                    },
                    "delta": {},
                },
                "calculus_limit_contract": {
                    "status": "pass",
                    "elapsed_seconds": 0.2,
                    "metrics": {
                        "cargo_status": "ok",
                        "passed": 6,
                        "failed": 0,
                        "ignored": 0,
                        "measured": 0,
                        "filtered_out": 0,
                        "timeouts": 0,
                    },
                    "delta": {},
                },
                "calculus_limit_presimplify_contract": {
                    "status": "pass",
                    "elapsed_seconds": 0.2,
                    "metrics": {
                        "cargo_status": "ok",
                        "passed": 8,
                        "failed": 0,
                        "ignored": 0,
                        "measured": 0,
                        "filtered_out": 0,
                        "timeouts": 0,
                    },
                    "delta": {},
                },
                "calculus_residual_matrix_smoke": {
                    "status": "pass",
                    "elapsed_seconds": 3.2,
                    "metrics": {
                        "matrix_status": "pass",
                        "total_cases": 247,
                        "passed": 247,
                        "failed": 0,
                        "raw_failed": 0,
                        "slow": 0,
                        "timeouts": 0,
                        "problem_case_count": 0,
                        "problem_cases": [],
                        "issue_kind_counts": {},
                        "expected_required_condition_case_count": 42,
                        "distinct_expected_required_conditions": 3,
                        "expected_required_condition_counts": {
                            "x + 1": 10,
                            "x + 2": 20,
                            "sin(x)": 12,
                        },
                    },
                    "delta": {},
                },
            },
        }

        markdown = MODULE.render_markdown(scorecard)

        self.assertIn("## Calculus Contract Signal", markdown)
        self.assertIn("public calculus behavior", markdown)
        self.assertIn("`diff`: passed=30 failed=0", markdown)
        self.assertIn(
            "`diff` ignored tests: `inverse_reciprocal_trig_diff_exhaustive` (debug-slow)",
            markdown,
        )
        self.assertIn(
            "`diff_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=256",
            markdown,
        )
        self.assertIn("`limit`: passed=6 failed=0", markdown)
        self.assertIn("`limit_presimplify_safe`: passed=8 failed=0", markdown)
        self.assertIn(
            "`residual_matrix`: passed=247 failed=0 total=247 slow=0 timeouts=0 "
            "conditioned_cases=42 distinct_conditions=3",
            markdown,
        )
        self.assertIn(
            "`residual_matrix` sparse expected conditions: x + 1=10, sin(x)=12, x + 2=20",
            markdown,
        )
        self.assertNotIn("`residual_matrix` problem cases:", markdown)
        self.assertIn("`integrate`: passed=23 failed=0", markdown)
        self.assertIn(
            "`integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=321",
            markdown,
        )
        self.assertIn(
            "| `calculus_diff_contract` | `pass` | 0.10s | passed=30 failed=0 ignored=1 |",
            markdown,
        )
        self.assertIn(
            "| `calculus_residual_matrix_smoke` | `pass` | 3.20s | "
            "passed=247 failed=0 total=247 conditioned=42 conditions=3 |",
            markdown,
        )

    def test_render_markdown_reports_residual_problem_cases(self):
        scorecard = {
            "generated_at": "2026-04-20T00:00:00+00:00",
            "profile": "guardrail",
            "git": {"branch": "main", "commit": "abc123"},
            "generated_discovery": {},
            "suites": {
                "calculus_residual_matrix_smoke": {
                    "status": "fail",
                    "elapsed_seconds": 3.2,
                    "metrics": {
                        "matrix_status": "slow",
                        "total_cases": 3,
                        "passed": 1,
                        "failed": 2,
                        "raw_failed": 0,
                        "slow": 1,
                        "timeouts": 1,
                        "problem_case_count": 2,
                        "problem_cases": [
                            {
                                "name": "slow_case",
                                "status": "slow",
                                "error_kind": "slow",
                            },
                            {
                                "name": "timeout_case",
                                "status": "timeout",
                                "error_kind": "timeout",
                            },
                        ],
                        "issue_kind_counts": {"slow": 1, "timeout": 1},
                    },
                    "delta": {},
                },
            },
        }

        markdown = MODULE.render_markdown(scorecard)

        self.assertIn(
            "`residual_matrix`: passed=1 failed=2 total=3 slow=1 timeouts=1",
            markdown,
        )
        self.assertIn(
            "`residual_matrix` problem cases: "
            "slow_case status=slow kind=slow; "
            "timeout_case status=timeout kind=timeout",
            markdown,
        )

    def test_render_markdown_includes_mixed_pressure_and_proof_shape_caveat(self):
        metrics = MODULE.parse_corpus(SAMPLE_CORPUS_OUTPUT)
        metrics["orchestrator_profile_slice"] = {
            "elapsed_seconds": 0.568,
            "family_count": 0,
            "filter": "root.div.",
            "limit": None,
            "total_cases": 4,
            "wrapper_count": 0,
        }
        strict_metrics = {
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
        }
        MODULE.add_unified_benchmark_rate_metrics(strict_metrics)
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
                    "metrics": strict_metrics,
                    "delta": {},
                },
            },
        }

        markdown = MODULE.render_markdown(scorecard)

        self.assertIn("## Simplify Closure Signal", markdown)
        self.assertIn(
            "symbolic=100/100 (100.0%), NF=0 (0.0%), proved-only=100 (100.0%)",
            markdown,
        )
        self.assertIn(
            "closure=100.0% nf=0 (0.0%) proved=100 (100.0%)",
            markdown,
        )
        self.assertIn("## Simplify Benchmark Interpretation", markdown)
        self.assertIn("composed 90.0%", markdown)
        self.assertIn("## Mixed Zero Pressure", markdown)
        self.assertIn("Composition hotspots", markdown)
        self.assertIn("Engine hotspots", markdown)
        self.assertIn("Window slices", markdown)
        self.assertIn("Steady-state engine reruns", markdown)
        self.assertIn("Steady-state dominant expressions", markdown)
        self.assertIn("#9 sum", markdown)
        self.assertIn(
            "expr=(ln(x^3) + ln(y^2) - ln(x^3 * y^2))",
            markdown,
        )
        self.assertIn("expr=a*b - b*a", markdown)
        self.assertIn("## Mixed Zero Orchestrator Profile", markdown)
        self.assertIn(
            "Profiled slice: 4 cases, 0.57s elapsed, filter `root.div.`.",
            markdown,
        )
        self.assertIn("Hot 1: `pipeline.phase.core`", markdown)
        self.assertIn(
            "No-match hotspot 1: `root.div.01.shifted_quotient_exact_one_gate`",
            markdown,
        )
        self.assertIn("sum total=3 failed=0 elapsed=12.00ms", markdown)
        self.assertIn("simplify=0.90ms", markdown)
        self.assertIn("sum simplify=0.90ms", markdown)
        self.assertIn("wall=12.00ms", markdown)
        self.assertIn("sum@0+3 failed=0 elapsed=12.00ms", markdown)
        self.assertIn("median_simplify=0.25ms", markdown)
        self.assertIn("median_wire=0.40ms", markdown)
        self.assertIn("median_wall=1.20ms", markdown)
        self.assertNotIn("simplify=0.00s", markdown)
        self.assertNotIn("median_simplify=0.00s", markdown)
        self.assertIn("sum total=3 failed=0", markdown)


if __name__ == "__main__":
    unittest.main()
