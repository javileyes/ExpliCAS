import argparse
import contextlib
import io
import importlib.util
import json
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
                "calculus_diff_command_matrix_smoke",
                "calculus_limit_compact_contract",
                "calculus_limit_presimplify_contract",
                "calculus_limit_command_matrix_smoke",
                "calculus_integrate_compact_contract",
                "calculus_integrate_backend_observability",
                "calculus_integrate_command_matrix_smoke",
                "calculus_residual_matrix_smoke",
            ],
        )
        self.assertIn("contextual_radical_fast", fast_embedded_names)
        self.assertIn("calculus_diff_contract", fast_embedded_names)
        self.assertIn("calculus_diff_command_matrix_smoke", fast_embedded_names)
        self.assertIn("calculus_limit_compact_contract", fast_embedded_names)
        self.assertIn("calculus_limit_presimplify_contract", fast_embedded_names)
        self.assertIn("calculus_limit_command_matrix_smoke", fast_embedded_names)
        self.assertIn("calculus_integrate_compact_contract", fast_embedded_names)
        self.assertIn("calculus_integrate_backend_observability", fast_embedded_names)
        self.assertIn("calculus_integrate_command_matrix_smoke", fast_embedded_names)
        self.assertIn("calculus_residual_matrix_smoke", fast_embedded_names)
        self.assertNotIn("calculus_limit_contract", fast_names)
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
        self.assertIn("calculus_diff_command_matrix_smoke", guardrail_names)
        self.assertIn("calculus_diff_command_matrix_smoke", full_names)
        self.assertNotIn("calculus_diff_exhaustive_contract", guardrail_names)
        self.assertIn("calculus_diff_exhaustive_contract", full_names)
        self.assertIn("calculus_limit_contract", guardrail_names)
        self.assertIn("calculus_limit_contract", full_names)
        self.assertIn("calculus_limit_presimplify_contract", guardrail_names)
        self.assertIn("calculus_limit_presimplify_contract", full_names)
        self.assertIn("calculus_limit_command_matrix_smoke", guardrail_names)
        self.assertIn("calculus_limit_command_matrix_smoke", full_names)
        self.assertIn("calculus_integrate_backend_observability", guardrail_names)
        self.assertIn("calculus_integrate_backend_observability", full_names)
        self.assertIn("calculus_integrate_command_matrix_smoke", guardrail_names)
        self.assertIn("calculus_integrate_command_matrix_smoke", full_names)
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
        self.assertEqual(
            MODULE.SUITES["calculus_diff_command_matrix_smoke"].category,
            "calculus",
        )
        self.assertEqual(MODULE.SUITES["calculus_limit_contract"].category, "calculus")
        self.assertEqual(
            MODULE.SUITES["calculus_limit_compact_contract"].category,
            "calculus",
        )
        self.assertEqual(
            MODULE.SUITES["calculus_limit_presimplify_contract"].category,
            "calculus",
        )
        self.assertEqual(
            MODULE.SUITES["calculus_limit_command_matrix_smoke"].category,
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
            MODULE.SUITES["calculus_integrate_command_matrix_smoke"].category,
            "calculus",
        )
        self.assertEqual(
            MODULE.SUITES["calculus_integrate_backend_observability"].category,
            "observability",
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
        diff_matrix_command = MODULE.SUITES[
            "calculus_diff_command_matrix_smoke"
        ].command
        self.assertIn("engine_diff_command_matrix_smoke.py", diff_matrix_command[1])
        self.assertIn("--ensure-release-cas-cli", diff_matrix_command)
        self.assertIn("--summary-json", diff_matrix_command)
        limit_compact_command = MODULE.SUITES[
            "calculus_limit_compact_contract"
        ].command
        self.assertIn("limit_contract_tests", limit_compact_command)
        self.assertIn(
            "test_limit_sqrt_quadratic_over_noisy_scaled_linear_denominator",
            limit_compact_command,
        )
        limit_matrix_command = MODULE.SUITES[
            "calculus_limit_command_matrix_smoke"
        ].command
        self.assertIn("engine_limit_command_matrix_smoke.py", limit_matrix_command[1])
        self.assertIn("--ensure-release-cas-cli", limit_matrix_command)
        self.assertIn("--summary-json", limit_matrix_command)
        integrate_matrix_command = MODULE.SUITES[
            "calculus_integrate_command_matrix_smoke"
        ].command
        self.assertIn(
            "engine_integrate_command_matrix_smoke.py",
            integrate_matrix_command[1],
        )
        self.assertIn("--ensure-release-cas-cli", integrate_matrix_command)
        self.assertIn("--summary-json", integrate_matrix_command)
        integrate_backend_command = MODULE.SUITES[
            "calculus_integrate_backend_observability"
        ].command
        self.assertIn("cas_math", integrate_backend_command)
        self.assertIn(
            "backend_observability_reports_boundary_metrics",
            integrate_backend_command,
        )
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
{"status":"pass","total":247,"status_counts":{"pass":247,"slow":0,"fail":0,"timeout":0},"issue_kind_counts":{},"matrix_base_count":26,"matrix_wrapped_base_count":21,"matrix_standalone_base_count":5,"matrix_wrapper_count":12,"matrix_wrapped_case_count":242,"matrix_standalone_case_count":5,"matrix_expected_wrapped_case_count":252,"matrix_missing_wrapped_pair_count":10,"matrix_full_wrapper_base_count":20,"matrix_partial_wrapper_base_count":1,"matrix_largest_wrapper_gap_count":10,"matrix_wrapper_gap_examples":[{"base":"wide_gap","missing_count":10,"missing_wrappers":["w1","w2"]}],"expected_required_condition_case_count":42,"distinct_expected_required_conditions":3,"expected_required_condition_counts":{"x + 1":10,"x + 2":20,"sin(x)":12}}
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
        self.assertEqual(metrics["matrix_base_count"], 26)
        self.assertEqual(metrics["matrix_wrapped_base_count"], 21)
        self.assertEqual(metrics["matrix_standalone_base_count"], 5)
        self.assertEqual(metrics["matrix_wrapper_count"], 12)
        self.assertEqual(metrics["matrix_wrapped_case_count"], 242)
        self.assertEqual(metrics["matrix_standalone_case_count"], 5)
        self.assertEqual(metrics["matrix_expected_wrapped_case_count"], 252)
        self.assertEqual(metrics["matrix_missing_wrapped_pair_count"], 10)
        self.assertEqual(metrics["matrix_full_wrapper_base_count"], 20)
        self.assertEqual(metrics["matrix_partial_wrapper_base_count"], 1)
        self.assertEqual(metrics["matrix_largest_wrapper_gap_count"], 10)
        self.assertEqual(
            metrics["matrix_wrapper_gap_examples"],
            [
                {
                    "base": "wide_gap",
                    "missing_count": 10,
                    "missing_wrappers": ["w1", "w2"],
                }
            ],
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

    def test_parse_calculus_limit_command_matrix_extracts_policy_axes(self):
        metrics = MODULE.parse_calculus_limit_command_matrix(
            """
{"status":"pass","total":11,"status_counts":{"pass":11,"slow":0,"fail":0,"timeout":0},"issue_kind_counts":{},"problem_case_count":0,"problem_cases":[],"supported_case_count":8,"residual_case_count":3,"residual_case_names":["limit_residual_a","limit_residual_b","limit_residual_c"],"warning_expected_case_count":3,"required_display_case_count":6,"step_checked_case_count":11,"supported_step_unchecked_case_count":0,"expected_step_substring_count":11,"distinct_required_display_count":6,"required_display_counts":{"x > -3":1,"x ≠ -2":1,"x ≠ 1":1},"family_count":7,"point_regime_counts":{"finite":6,"infinity":5},"domain_regime_counts":{"unconditional":4,"required_condition":3,"removable_hole":1,"endpoint_residual":1,"discontinuous_residual":1,"domain_path_conflict":1},"required_condition_regime_counts":{"finite_source_definedness":1,"infinity_path_conflict":1,"none":9},"outcome_counts":{"supported":8,"residual":3},"residual_cause_counts":{"finite_endpoint_or_boundary_policy":2,"infinity_domain_path_conflict":1},"residual_family_counts":{"endpoint_family":2,"infinity_family":1},"residual_cause_family_counts":{"finite_endpoint_or_boundary_policy/endpoint_family":2,"infinity_domain_path_conflict/infinity_family":1},"residual_cases_by_cause":{"finite_endpoint_or_boundary_policy":["limit_residual_a","limit_residual_b"],"infinity_domain_path_conflict":["limit_residual_c"]},"calculus_maturity_block_counts":{"block3_real_domain_limits":8,"block9_residuals_and_non_goals":3},"calculus_block_gate_counts":{"didactic_trace_and_limit_policy":2,"domain_conditions_and_limit_policy":6,"safe_residual_policy":3},"trace_regime_counts":{"substitution":1},"presentation_regime_counts":{"canonical":7,"infinity":1,"residual":3},"cli_simplify_runtime_distribution":{"timed_case_count":2,"total_elapsed_seconds":0.033,"avg_case_ms":16.5,"p95_case_ms":21.0,"max_case_ms":21.0},"cli_total_runtime_distribution":{"timed_case_count":2,"total_elapsed_seconds":0.037,"avg_case_ms":18.5,"p95_case_ms":23.0,"max_case_ms":23.0},"cli_public_overhead_runtime_distribution":{"timed_case_count":2,"total_elapsed_seconds":0.012,"avg_case_ms":6.0,"p95_case_ms":7.0,"max_case_ms":7.0},"slowest_cli_simplify_evaluations":[{"name":"limit_residual_c","family":"infinity_family","point_regime":"infinity","domain_regime":"domain_path_conflict","trace_regime":"infinity_residual_policy","cli_simplify_elapsed_seconds":0.021}],"slowest_cli_total_evaluations":[{"name":"limit_residual_c","family":"infinity_family","point_regime":"infinity","domain_regime":"domain_path_conflict","trace_regime":"infinity_residual_policy","cli_total_elapsed_seconds":0.023}],"slowest_cli_public_overhead_evaluations":[{"name":"finite_supported_a","family":"polynomial","point_regime":"finite","domain_regime":"unconditional","trace_regime":"substitution","cli_public_overhead_seconds":0.007}]}
"""
        )

        self.assertEqual(metrics["matrix_status"], "pass")
        self.assertEqual(metrics["total_cases"], 11)
        self.assertEqual(metrics["passed"], 11)
        self.assertEqual(metrics["failed"], 0)
        self.assertEqual(metrics["limit_supported_case_count"], 8)
        self.assertEqual(metrics["limit_residual_case_count"], 3)
        self.assertEqual(
            metrics["limit_residual_case_names"],
            ["limit_residual_a", "limit_residual_b", "limit_residual_c"],
        )
        self.assertEqual(metrics["limit_warning_expected_case_count"], 3)
        self.assertEqual(metrics["limit_required_display_case_count"], 6)
        self.assertEqual(metrics["limit_step_checked_case_count"], 11)
        self.assertEqual(metrics["limit_supported_step_unchecked_case_count"], 0)
        self.assertEqual(metrics["limit_expected_step_substring_count"], 11)
        self.assertEqual(metrics["limit_distinct_required_display_count"], 6)
        self.assertEqual(metrics["limit_family_count"], 7)
        self.assertEqual(
            metrics["limit_required_display_counts"],
            {"x > -3": 1, "x ≠ -2": 1, "x ≠ 1": 1},
        )
        self.assertEqual(
            metrics["limit_required_condition_regime_counts"],
            {
                "finite_source_definedness": 1,
                "infinity_path_conflict": 1,
                "none": 9,
            },
        )
        self.assertEqual(metrics["limit_point_regime_counts"], {"finite": 6, "infinity": 5})
        self.assertEqual(metrics["limit_outcome_counts"], {"supported": 8, "residual": 3})
        self.assertEqual(
            metrics["limit_residual_cause_counts"],
            {
                "finite_endpoint_or_boundary_policy": 2,
                "infinity_domain_path_conflict": 1,
            },
        )
        self.assertEqual(
            metrics["limit_residual_family_counts"],
            {
                "endpoint_family": 2,
                "infinity_family": 1,
            },
        )
        self.assertEqual(
            metrics["limit_residual_cause_family_counts"],
            {
                "finite_endpoint_or_boundary_policy/endpoint_family": 2,
                "infinity_domain_path_conflict/infinity_family": 1,
            },
        )
        self.assertEqual(
            metrics["limit_residual_cases_by_cause"],
            {
                "finite_endpoint_or_boundary_policy": [
                    "limit_residual_a",
                    "limit_residual_b",
                ],
                "infinity_domain_path_conflict": ["limit_residual_c"],
            },
        )
        self.assertEqual(
            metrics["limit_calculus_maturity_block_counts"],
            {
                "block3_real_domain_limits": 8,
                "block9_residuals_and_non_goals": 3,
            },
        )
        self.assertEqual(
            metrics["limit_calculus_block_gate_counts"],
            {
                "didactic_trace_and_limit_policy": 2,
                "domain_conditions_and_limit_policy": 6,
                "safe_residual_policy": 3,
            },
        )
        self.assertEqual(
            metrics["limit_cli_simplify_runtime_distribution"]["max_case_ms"],
            21.0,
        )
        self.assertEqual(
            metrics["limit_slowest_cli_simplify_evaluations"][0]["name"],
            "limit_residual_c",
        )
        self.assertEqual(
            MODULE.suite_status("calculus_limit_command_matrix_smoke", metrics, 0),
            "pass",
        )

    def test_parse_calculus_diff_command_matrix_extracts_policy_axes(self):
        metrics = MODULE.parse_calculus_diff_command_matrix(
            """
{"status":"pass","total":13,"status_counts":{"pass":13,"slow":0,"fail":0,"timeout":0},"issue_kind_counts":{},"problem_case_count":0,"problem_cases":[],"supported_case_count":12,"residual_case_count":1,"residual_case_names":["diff_residual_case"],"warning_expected_case_count":0,"required_display_case_count":9,"step_checked_case_count":13,"supported_step_unchecked_case_count":0,"expected_step_substring_count":32,"distinct_required_display_count":4,"required_display_counts":{"x > 0":4,"cos(2·x + 1) ≠ 0":1},"family_count":10,"argument_regime_counts":{"variable":4,"product":2,"polynomial_inner":1,"rational_expression":1,"nested_root":1,"scaled_nested_root":1,"nested_bounded_root":1,"negated_nested_root":1,"variable_power":1},"domain_regime_counts":{"unconditional":3,"required_condition":8,"interval_required":1,"discontinuous_residual":1},"outcome_counts":{"supported":12,"residual":1},"calculus_maturity_block_counts":{"block2_real_domain_differentiation":12,"block9_residuals_and_non_goals":1},"calculus_block_gate_counts":{"didactic_trace_and_diff_policy":3,"domain_conditions_and_diff_policy":9,"safe_residual_policy":1},"symbolic_radius_policy_cluster_counts":{"block2_symbolic_radius_arctan_positive_quadratic":2},"positive_quadratic_policy_cluster_counts":{"block2_positive_quadratic_log_abs_pole_primitive":3,"block2_positive_quadratic_log_arctan_primitive":3,"block2_symbolic_radius_arctan_positive_quadratic":2},"trace_regime_counts":{"chain_rule":4,"constant_multiple_chain_rule":1,"logarithmic_derivative":1,"negative_argument_chain_rule":1,"piecewise_abs":1,"power_rule":1,"product_rule":1,"product_rule_log":1,"quotient_rule":1,"residual_policy":1},"presentation_regime_counts":{"canonical":2,"compact_quotient":1,"factored":2,"post_calculus_compact":1,"quotient_abs":1,"reciprocal_root":1,"residual":1,"scaled_post_calculus_compact":1,"signed_post_calculus_compact":1,"signed_reciprocal_root_interval":1,"trig_power_difference":1}}
"""
        )

        self.assertEqual(metrics["matrix_status"], "pass")
        self.assertEqual(metrics["total_cases"], 13)
        self.assertEqual(metrics["passed"], 13)
        self.assertEqual(metrics["failed"], 0)
        self.assertEqual(metrics["diff_supported_case_count"], 12)
        self.assertEqual(metrics["diff_residual_case_count"], 1)
        self.assertEqual(metrics["diff_residual_case_names"], ["diff_residual_case"])
        self.assertEqual(metrics["diff_warning_expected_case_count"], 0)
        self.assertEqual(metrics["diff_required_display_case_count"], 9)
        self.assertEqual(metrics["diff_step_checked_case_count"], 13)
        self.assertEqual(metrics["diff_supported_step_unchecked_case_count"], 0)
        self.assertEqual(metrics["diff_expected_step_substring_count"], 32)
        self.assertEqual(metrics["diff_distinct_required_display_count"], 4)
        self.assertEqual(metrics["diff_family_count"], 10)
        self.assertEqual(
            metrics["diff_required_display_counts"],
            {"cos(2·x + 1) ≠ 0": 1, "x > 0": 4},
        )
        self.assertEqual(
            metrics["diff_argument_regime_counts"],
            {
                "variable": 4,
                "product": 2,
                "polynomial_inner": 1,
                "rational_expression": 1,
                "nested_root": 1,
                "scaled_nested_root": 1,
                "nested_bounded_root": 1,
                "negated_nested_root": 1,
                "variable_power": 1,
            },
        )
        self.assertEqual(
            metrics["diff_outcome_counts"], {"supported": 12, "residual": 1}
        )
        self.assertEqual(
            metrics["diff_calculus_maturity_block_counts"],
            {
                "block2_real_domain_differentiation": 12,
                "block9_residuals_and_non_goals": 1,
            },
        )
        self.assertEqual(
            metrics["diff_calculus_block_gate_counts"],
            {
                "didactic_trace_and_diff_policy": 3,
                "domain_conditions_and_diff_policy": 9,
                "safe_residual_policy": 1,
            },
        )
        self.assertEqual(
            metrics["diff_symbolic_radius_policy_cluster_counts"],
            {"block2_symbolic_radius_arctan_positive_quadratic": 2},
        )
        self.assertEqual(
            metrics["diff_symbolic_radius_consolidated_policy_cluster_counts"],
            {"block2_symbolic_radius_arctan_positive_quadratic": 2},
        )
        self.assertEqual(
            metrics["diff_positive_quadratic_policy_cluster_counts"],
            {
                "block2_positive_quadratic_log_abs_pole_primitive": 3,
                "block2_positive_quadratic_log_arctan_primitive": 3,
                "block2_symbolic_radius_arctan_positive_quadratic": 2,
            },
        )
        self.assertEqual(
            metrics["diff_positive_quadratic_consolidated_policy_cluster_counts"],
            {
                "block2_positive_quadratic_log_abs_pole_primitive": 3,
                "block2_positive_quadratic_log_arctan_primitive": 3,
                "block2_symbolic_radius_arctan_positive_quadratic": 2,
            },
        )
        self.assertNotIn(
            "diff_positive_quadratic_consolidation_candidate_counts",
            metrics,
        )
        self.assertEqual(
            MODULE.suite_status("calculus_diff_command_matrix_smoke", metrics, 0),
            "pass",
        )

    def test_parse_calculus_integrate_command_matrix_extracts_policy_axes(self):
        metrics = MODULE.parse_calculus_integrate_command_matrix(
            """
{"status":"pass","total":18,"status_counts":{"pass":18,"slow":0,"fail":0,"timeout":0},"issue_kind_counts":{},"problem_case_count":0,"problem_cases":[],"supported_case_count":17,"residual_case_count":1,"residual_case_names":["integrate_residual_case"],"warning_expected_case_count":0,"required_display_case_count":9,"step_checked_case_count":18,"supported_step_unchecked_case_count":0,"antiderivative_verification_case_count":17,"verified_supported_case_count":17,"direct_diff_integrate_case_count":2,"direct_diff_integrate_exact_case_count":2,"direct_diff_integrate_equivalence_case_count":0,"expected_step_substring_count":46,"distinct_required_display_count":7,"required_display_counts":{"-1 < x < 1":2,"x > 0":2,"x ≠ -1/2":1},"family_count":18,"argument_regime_counts":{"variable_power":1,"sum":1,"affine_argument":1,"affine_bounded_rational":1,"bounded_rational_expression":1,"bounded_variable_radical":1,"unbounded_variable_radical":1,"nonlinear_polynomial_derivative":1,"nonlinear_polynomial_base":1,"rational_expression":2,"affine_radical":1,"affine_hyperbolic":1,"product":1,"affine_product":1,"sqrt_chain":2,"unsupported_core":1},"domain_regime_counts":{"unconditional":9,"nonzero_required":1,"radical_interval":2,"rational_interval":2,"positive_required":2,"sqrt_chain_nonzero_positive":2},"outcome_counts":{"supported":17,"residual":1},"residual_cause_counts":{"special_function_method_required":1},"residual_family_counts":{"log_rational_residual":1},"residual_cause_family_counts":{"special_function_method_required/log_rational_residual":1},"residual_cases_by_cause":{"special_function_method_required":["integrate_residual_case"]},"verification_regime_counts":{"residual_not_verified":1,"verified_by_diff":15,"verified_by_diff_and_direct_diff_integrate":2},"calculus_maturity_block_counts":{"block4_base_integration":2,"block5_generalized_substitution":3,"block7_trig_hyperbolic_integration":8,"block8_radical_inverse_families":4,"block9_residuals_and_non_goals":1},"calculus_block_gate_counts":{"didactic_trace_and_verified_antiderivative":8,"domain_conditions_and_verified_antiderivative":9,"safe_residual_policy":1},"trig_hyperbolic_policy_cluster_counts":{"block7_explicit_reciprocal_trig_log_substitution":7,"block7_sqrt_chain_reciprocal_trig_product":7,"block7_hyperbolic_reciprocal_square":1},"base_integration_policy_cluster_counts":{"block4_log_by_parts":2},"radical_inverse_policy_cluster_counts":{"block8_inverse_trig_root_reciprocal":6},"direct_diff_integrate_calculus_maturity_block_counts":{"block7_trig_hyperbolic_integration":1,"block8_radical_inverse_families":1},"direct_diff_integrate_calculus_block_gate_counts":{"didactic_trace_and_verified_antiderivative":1,"domain_conditions_and_verified_antiderivative":1},"direct_diff_integrate_trig_hyperbolic_policy_cluster_counts":{"block7_hyperbolic_reciprocal_square":1},"direct_diff_integrate_radical_inverse_policy_cluster_counts":{"block8_inverse_sqrt_tables":1},"direct_diff_integrate_gap_cases_by_calculus_maturity_block":{"block7_trig_hyperbolic_integration":["gap_case_a","gap_case_b","gap_case_c","gap_case_d"]},"direct_diff_integrate_gap_cases_by_trig_hyperbolic_policy_cluster":{"block7_sqrt_chain_reciprocal_trig_product":["gap_case_a","gap_case_b","gap_case_c","gap_case_d"]},"direct_diff_integrate_equivalence_calculus_maturity_block_counts":{},"direct_diff_integrate_equivalence_trig_hyperbolic_policy_cluster_counts":{},"trace_regime_counts":{"inverse_hyperbolic_rational_affine_table":1,"inverse_hyperbolic_rational_direct_table":1,"inverse_sqrt_direct_table":1,"inverse_sqrt_hyperbolic_direct_table":1,"linearity":1,"power_rule":1},"presentation_regime_counts":{"affine_atanh":1,"compact_power":1,"inverse_hyperbolic":2,"inverse_trig":2,"residual":1},"cli_simplify_runtime_distribution":{"timed_case_count":2,"total_elapsed_seconds":0.075,"avg_case_ms":37.5,"p95_case_ms":49.0,"max_case_ms":49.0},"cli_total_runtime_distribution":{"timed_case_count":2,"total_elapsed_seconds":0.081,"avg_case_ms":40.5,"p95_case_ms":52.0,"max_case_ms":52.0},"cli_public_overhead_runtime_distribution":{"timed_case_count":2,"total_elapsed_seconds":0.022,"avg_case_ms":11.0,"p95_case_ms":13.0,"max_case_ms":13.0},"slowest_cli_simplify_evaluations":[{"name":"slow_integrate_case","family":"hyperbolic_reciprocal_fourth","trace_regime":"hyperbolic_reciprocal_fourth","calculus_maturity_block":"block7_trig_hyperbolic_integration","calculus_block_gate":"didactic_trace_and_verified_antiderivative","antiderivative_verification_mode":"residual_equivalence","cli_simplify_elapsed_seconds":0.049}],"slowest_cli_total_evaluations":[{"name":"slow_integrate_case","family":"hyperbolic_reciprocal_fourth","trace_regime":"hyperbolic_reciprocal_fourth","calculus_maturity_block":"block7_trig_hyperbolic_integration","calculus_block_gate":"didactic_trace_and_verified_antiderivative","antiderivative_verification_mode":"residual_equivalence","cli_total_elapsed_seconds":0.052}],"slowest_cli_public_overhead_evaluations":[{"name":"slow_integrate_case","family":"hyperbolic_reciprocal_fourth","trace_regime":"hyperbolic_reciprocal_fourth","calculus_maturity_block":"block7_trig_hyperbolic_integration","calculus_block_gate":"didactic_trace_and_verified_antiderivative","antiderivative_verification_mode":"residual_equivalence","cli_public_overhead_seconds":0.013}],"slowest_integrate_evaluations":[{"name":"slow_integrate_case","family":"hyperbolic_reciprocal_fourth","integrate_elapsed_seconds":0.42,"antiderivative_verification_mode":"residual_equivalence"}],"slowest_antiderivative_verifications":[{"name":"slow_verify_case","family":"hyperbolic_reciprocal_square","antiderivative_verification_elapsed_seconds":0.21,"antiderivative_verification_mode":"residual_equivalence"}],"slowest_direct_diff_integrate_checks":[{"name":"direct_nested_case","family":"inverse_sqrt_table","direct_diff_integrate_elapsed_seconds":0.13,"antiderivative_verification_mode":"direct_derivative"}],"largest_stdout_payload_cases":[{"name":"large_integrate_output","family":"rational_partial_fraction","stdout_bytes":8192,"required_display_count":3,"expected_step_substring_count":4}],"largest_step_trace_cases":[{"name":"large_integrate_steps","family":"by_parts_affine_log","step_text_char_count":4096,"required_display_count":1,"expected_step_substring_count":6}],"runtime_by_antiderivative_verification_mode":[{"mode":"residual_equivalence","case_count":10,"total_elapsed_seconds":1.23,"avg_case_ms":123.0,"max_elapsed_seconds":0.21,"slowest_case":"slow_verify_case"}],"runtime_by_residual_cause":[{"cause":"special_function_method_required","case_count":1,"total_elapsed_seconds":0.42,"avg_case_ms":420.0,"max_elapsed_seconds":0.42,"slowest_case":"integrate_residual_case"}],"runtime_by_residual_cause_family":[{"cause_family":"special_function_method_required/log_rational_residual","case_count":1,"total_elapsed_seconds":0.42,"avg_case_ms":420.0,"max_elapsed_seconds":0.42,"slowest_case":"integrate_residual_case"}],"residual_public_phase_by_cause":[{"cause":"special_function_method_required","case_count":1,"integrate_total_seconds":0.42,"cli_total_seconds":0.31,"public_overhead_total_seconds":0.11,"public_overhead_share_percent":26.2,"avg_required_display_count":4.0,"avg_step_text_char_count":610.0,"slowest_case":"integrate_residual_case"}],"residual_public_phase_by_cause_family":[{"cause_family":"special_function_method_required/log_rational_residual","case_count":1,"integrate_total_seconds":0.42,"cli_total_seconds":0.31,"public_overhead_total_seconds":0.11,"public_overhead_share_percent":26.2,"avg_required_display_count":4.0,"avg_step_text_char_count":610.0,"slowest_case":"integrate_residual_case"}],"residual_public_phase_slowest_cases":[{"name":"integrate_residual_case","integrate_elapsed_seconds":0.42,"cli_total_seconds":0.31,"cli_simplify_ms":309.0,"public_overhead_seconds":0.11,"public_overhead_share_percent":26.2,"required_display_count":4,"step_text_char_count":610,"stdout_bytes":2048,"residual_cause":"special_function_method_required","family":"explicit_reciprocal_trig_residual_domain","trace_regime":"residual_presimplification_with_domain"}],"residual_shape_orientation_probes":[{"name":"shifted_tangent_log_factored_residual","status":"pass","expression_shape":"factored_residual","orientation":"tan_minus_offset","wall_elapsed_seconds":0.052,"cli_parse_us":100,"cli_simplify_us":49000,"cli_total_us":51000,"stdout_bytes":2048,"stderr_bytes":0,"required_display_count":4}]}
"""
        )

        self.assertEqual(metrics["matrix_status"], "pass")
        self.assertEqual(metrics["total_cases"], 18)
        self.assertEqual(metrics["passed"], 18)
        self.assertEqual(metrics["failed"], 0)
        self.assertEqual(metrics["integrate_supported_case_count"], 17)
        self.assertEqual(metrics["integrate_residual_case_count"], 1)
        self.assertEqual(
            metrics["integrate_residual_case_names"], ["integrate_residual_case"]
        )
        self.assertEqual(metrics["integrate_warning_expected_case_count"], 0)
        self.assertEqual(metrics["integrate_required_display_case_count"], 9)
        self.assertEqual(metrics["integrate_step_checked_case_count"], 18)
        self.assertEqual(metrics["integrate_supported_step_unchecked_case_count"], 0)
        self.assertEqual(
            metrics["integrate_antiderivative_verification_case_count"], 17
        )
        self.assertEqual(metrics["integrate_verified_supported_case_count"], 17)
        self.assertEqual(metrics["integrate_direct_diff_integrate_case_count"], 2)
        self.assertEqual(metrics["integrate_direct_diff_integrate_exact_case_count"], 2)
        self.assertEqual(
            metrics["integrate_direct_diff_integrate_equivalence_case_count"], 0
        )
        self.assertEqual(
            metrics[
                "integrate_direct_diff_integrate_gap_cases_by_calculus_maturity_block"
            ],
            {
                "block7_trig_hyperbolic_integration": [
                    "gap_case_a",
                    "gap_case_b",
                    "gap_case_c",
                    "gap_case_d",
                ]
            },
        )
        self.assertEqual(
            metrics[
                "integrate_direct_diff_integrate_gap_cases_by_trig_hyperbolic_policy_cluster"
            ],
            {
                "block7_sqrt_chain_reciprocal_trig_product": [
                    "gap_case_a",
                    "gap_case_b",
                    "gap_case_c",
                    "gap_case_d",
                ]
            },
        )
        self.assertEqual(metrics["integrate_expected_step_substring_count"], 46)
        self.assertEqual(metrics["integrate_distinct_required_display_count"], 7)
        self.assertEqual(metrics["integrate_family_count"], 18)
        self.assertEqual(
            metrics["integrate_required_display_counts"],
            {"-1 < x < 1": 2, "x > 0": 2, "x ≠ -1/2": 1},
        )
        self.assertEqual(
            metrics["integrate_argument_regime_counts"],
            {
                "variable_power": 1,
                "sum": 1,
                "affine_argument": 1,
                "affine_bounded_rational": 1,
                "bounded_rational_expression": 1,
                "bounded_variable_radical": 1,
                "unbounded_variable_radical": 1,
                "nonlinear_polynomial_derivative": 1,
                "nonlinear_polynomial_base": 1,
                "rational_expression": 2,
                "affine_radical": 1,
                "affine_hyperbolic": 1,
                "product": 1,
                "affine_product": 1,
                "sqrt_chain": 2,
                "unsupported_core": 1,
            },
        )
        self.assertEqual(
            metrics["integrate_outcome_counts"], {"supported": 17, "residual": 1}
        )
        self.assertEqual(
            metrics["integrate_residual_cause_counts"],
            {"special_function_method_required": 1},
        )
        self.assertEqual(
            metrics["integrate_residual_family_counts"],
            {"log_rational_residual": 1},
        )
        self.assertEqual(
            metrics["integrate_residual_cause_family_counts"],
            {"special_function_method_required/log_rational_residual": 1},
        )
        self.assertEqual(
            metrics["integrate_residual_cases_by_cause"],
            {"special_function_method_required": ["integrate_residual_case"]},
        )
        self.assertEqual(
            metrics["integrate_verification_regime_counts"],
            {
                "residual_not_verified": 1,
                "verified_by_diff": 15,
                "verified_by_diff_and_direct_diff_integrate": 2,
            },
        )
        self.assertEqual(
            metrics["integrate_calculus_maturity_block_counts"],
            {
                "block4_base_integration": 2,
                "block5_generalized_substitution": 3,
                "block7_trig_hyperbolic_integration": 8,
                "block8_radical_inverse_families": 4,
                "block9_residuals_and_non_goals": 1,
            },
        )
        self.assertEqual(
            metrics["integrate_calculus_block_gate_counts"],
            {
                "didactic_trace_and_verified_antiderivative": 8,
                "domain_conditions_and_verified_antiderivative": 9,
                "safe_residual_policy": 1,
            },
        )
        self.assertEqual(
            metrics["integrate_trig_hyperbolic_policy_cluster_counts"],
            {
                "block7_explicit_reciprocal_trig_log_substitution": 7,
                "block7_hyperbolic_reciprocal_square": 1,
                "block7_sqrt_chain_reciprocal_trig_product": 7,
            },
        )
        self.assertEqual(
            metrics["integrate_trig_hyperbolic_consolidated_policy_cluster_counts"],
            {
                "block7_explicit_reciprocal_trig_log_substitution": 7,
                "block7_hyperbolic_reciprocal_square": 1,
                "block7_sqrt_chain_reciprocal_trig_product": 7,
            },
        )
        self.assertNotIn(
            "integrate_trig_hyperbolic_consolidation_candidate_counts",
            metrics,
        )
        self.assertEqual(
            metrics["integrate_radical_inverse_policy_cluster_counts"],
            {"block8_inverse_trig_root_reciprocal": 6},
        )
        self.assertEqual(
            metrics["integrate_radical_inverse_consolidated_policy_cluster_counts"],
            {"block8_inverse_trig_root_reciprocal": 6},
        )
        self.assertEqual(
            metrics["integrate_base_integration_policy_cluster_counts"],
            {"block4_log_by_parts": 2},
        )
        self.assertEqual(
            metrics["integrate_base_integration_consolidated_policy_cluster_counts"],
            {"block4_log_by_parts": 2},
        )
        self.assertEqual(
            metrics["integrate_direct_diff_integrate_calculus_maturity_block_counts"],
            {
                "block7_trig_hyperbolic_integration": 1,
                "block8_radical_inverse_families": 1,
            },
        )
        self.assertEqual(
            metrics["integrate_direct_diff_integrate_calculus_block_gate_counts"],
            {
                "didactic_trace_and_verified_antiderivative": 1,
                "domain_conditions_and_verified_antiderivative": 1,
            },
        )
        self.assertEqual(
            metrics[
                "integrate_direct_diff_integrate_trig_hyperbolic_policy_cluster_counts"
            ],
            {"block7_hyperbolic_reciprocal_square": 1},
        )
        self.assertEqual(
            metrics[
                "integrate_direct_diff_integrate_trig_hyperbolic_shared_policy_cluster_counts"
            ],
            {"block7_hyperbolic_reciprocal_square": 1},
        )
        self.assertEqual(
            metrics[
                "integrate_direct_diff_integrate_radical_inverse_policy_cluster_counts"
            ],
            {"block8_inverse_sqrt_tables": 1},
        )
        self.assertEqual(
            metrics[
                "integrate_direct_diff_integrate_radical_inverse_shared_policy_cluster_counts"
            ],
            {"block8_inverse_sqrt_tables": 1},
        )
        self.assertNotIn(
            "integrate_direct_diff_integrate_base_integration_policy_cluster_counts",
            metrics,
        )
        self.assertNotIn(
            "integrate_direct_diff_integrate_base_integration_shared_policy_cluster_counts",
            metrics,
        )
        self.assertNotIn(
            "integrate_direct_diff_integrate_equivalence_calculus_maturity_block_counts",
            metrics,
        )
        self.assertNotIn(
            "integrate_direct_diff_integrate_equivalence_trig_hyperbolic_policy_cluster_counts",
            metrics,
        )
        self.assertNotIn(
            "integrate_direct_diff_integrate_equivalence_base_integration_policy_cluster_counts",
            metrics,
        )
        self.assertEqual(
            metrics["integrate_slowest_integrate_evaluations"][0]["name"],
            "slow_integrate_case",
        )
        self.assertEqual(
            metrics["integrate_cli_simplify_runtime_distribution"]["max_case_ms"],
            49.0,
        )
        self.assertEqual(
            metrics["integrate_slowest_cli_simplify_evaluations"][0]["name"],
            "slow_integrate_case",
        )
        self.assertEqual(
            metrics["integrate_slowest_antiderivative_verifications"][0][
                "antiderivative_verification_elapsed_seconds"
            ],
            0.21,
        )
        self.assertEqual(
            metrics["integrate_slowest_direct_diff_integrate_checks"][0][
                "direct_diff_integrate_elapsed_seconds"
            ],
            0.13,
        )
        self.assertEqual(
            metrics["integrate_largest_stdout_payload_cases"][0]["stdout_bytes"],
            8192,
        )
        self.assertEqual(
            metrics["integrate_largest_step_trace_cases"][0]["step_text_char_count"],
            4096,
        )
        self.assertEqual(
            metrics["integrate_runtime_by_antiderivative_verification_mode"][0][
                "mode"
            ],
            "residual_equivalence",
        )
        self.assertEqual(
            metrics["integrate_runtime_by_residual_cause"][0],
            {
                "cause": "special_function_method_required",
                "case_count": 1,
                "total_elapsed_seconds": 0.42,
                "avg_case_ms": 420.0,
                "max_elapsed_seconds": 0.42,
                "slowest_case": "integrate_residual_case",
            },
        )
        self.assertEqual(
            metrics["integrate_runtime_by_residual_cause_family"][0],
            {
                "cause_family": "special_function_method_required/log_rational_residual",
                "case_count": 1,
                "total_elapsed_seconds": 0.42,
                "avg_case_ms": 420.0,
                "max_elapsed_seconds": 0.42,
                "slowest_case": "integrate_residual_case",
            },
        )
        self.assertEqual(
            metrics["integrate_residual_public_phase_by_cause"][0]["cause"],
            "special_function_method_required",
        )
        self.assertEqual(
            metrics["integrate_residual_public_phase_by_cause_family"][0][
                "cause_family"
            ],
            "special_function_method_required/log_rational_residual",
        )
        self.assertEqual(
            metrics["integrate_residual_public_phase_by_cause"][0][
                "public_overhead_share_percent"
            ],
            26.2,
        )
        self.assertEqual(
            metrics["integrate_residual_public_phase_slowest_cases"][0][
                "cli_simplify_ms"
            ],
            309.0,
        )
        self.assertEqual(
            metrics["integrate_residual_shape_orientation_probes"][0],
            {
                "name": "shifted_tangent_log_factored_residual",
                "expression_shape": "factored_residual",
                "orientation": "tan_minus_offset",
                "status": "pass",
                "wall_elapsed_seconds": 0.052,
                "cli_parse_us": 100.0,
                "cli_simplify_us": 49000.0,
                "cli_total_us": 51000.0,
                "stdout_bytes": 2048,
                "stderr_bytes": 0,
                "required_display_count": 4,
            },
        )
        self.assertEqual(
            metrics["integrate_residual_shape_orientation_summary"],
            {
                "probe_count": 1,
                "counted_probe_count": 1,
                "max_required_display_count": 4,
                "avg_required_display_count": 4.0,
                "max_name": "shifted_tangent_log_factored_residual",
                "max_expression_shape": "factored_residual",
                "max_orientation": "tan_minus_offset",
                "max_status": "pass",
                "status_counts": {"pass": 1},
            },
        )
        self.assertNotIn(
            "integrate_radical_inverse_consolidation_candidate_counts",
            metrics,
        )
        self.assertEqual(
            MODULE.suite_status("calculus_integrate_command_matrix_smoke", metrics, 0),
            "pass",
        )

    def test_parse_algorithmic_backend_observability_extracts_boundary_metrics(
        self,
    ):
        metrics = MODULE.parse_algorithmic_backend_observability(
            """
running 1 test
algorithmic_backend_observability: {"assumption_exprs": 0, "attempts": 16, "constant_policy_counts": {"arbitrary_constant_omitted": 13, "unspecified": 3}, "failure_class_by_method": {"rational/budget_exceeded": 1, "table_reused/residual_affine_in_variable": 1, "table_reused/residual_function_of_variable": 1, "unsupported/budget_exceeded": 1, "unsupported/disabled_by_mode": 1, "unsupported/unsupported_method": 1}, "failure_class_counts": {"budget_exceeded": 2, "disabled_by_mode": 1, "residual_affine_in_variable": 1, "residual_function_of_variable": 1, "unsupported_method": 1}, "fallback_assumption_exprs": 0, "fallback_constant_policy_counts": {"arbitrary_constant_omitted": 1}, "fallback_eligible": 1, "fallback_max_verification_normalization_passes": 0, "fallback_status_by_method": {"hermite/blocked_by_mode": 4, "heurisch_probe/blocked_by_mode": 1, "rational/blocked_by_candidate_policy": 1, "rational/blocked_by_mode": 3, "rational/eligible": 1, "table_reused/blocked_by_candidate_policy": 2, "table_reused/blocked_by_mode": 1, "unsupported/blocked_by_candidate_policy": 3}, "fallback_status_counts": {"blocked_by_candidate_policy": 6, "blocked_by_mode": 9, "eligible": 1}, "fallback_trace_level_counts": {"algorithmic_summary": 1}, "fallback_verification_evidence_by_method": {"rational/direct_differentiation": 1}, "fallback_verification_evidence_counts": {"direct_differentiation": 1}, "fallback_verification_normalization_pass_count_by_method": {"rational/0": 1}, "fallback_verification_normalization_pass_count_counts": {"0": 1}, "fallback_verification_normalization_reason_by_method": {}, "fallback_verification_normalization_reason_counts": {}, "max_verification_normalization_passes": 2, "method_counts": {"hermite": 4, "heurisch_probe": 1, "rational": 5, "table_reused": 3, "unsupported": 3}, "method_probe_attempt_counts": {"hermite": 6, "heurisch_probe": 2, "rational": 11}, "method_probe_budget_exhausted": 1, "method_probe_candidate_counts": {"hermite": 4, "heurisch_probe": 1, "rational": 5}, "method_probe_no_match_counts": {"hermite": 2, "heurisch_probe": 1, "rational": 6}, "method_probe_no_match_reason_counts": {"hermite/denominator_policy_mismatch": 1, "hermite/shape_mismatch": 1, "heurisch_probe/shape_mismatch": 1, "rational/numerator_policy_mismatch": 5, "rational/shape_mismatch": 1}, "method_probe_usage_by_method": {"hermite": 8, "heurisch_probe": 3, "rational": 5, "unsupported": 3}, "method_probes_used_total": 19, "mode_counts": {"diagnostic_only": 14, "disabled": 1, "residual_fallback": 1}, "public_accepted": 10, "public_assumption_exprs": 0, "public_constant_policy_counts": {"arbitrary_constant_omitted": 10}, "public_max_verification_normalization_passes": 2, "public_trace_level_counts": {"algorithmic_summary": 10}, "public_verification_evidence_by_method": {"hermite/normalized_differentiation": 4, "heurisch_probe/direct_differentiation": 1, "rational/direct_differentiation": 2, "rational/normalized_differentiation": 2, "table_reused/direct_differentiation": 1}, "public_verification_evidence_counts": {"direct_differentiation": 4, "normalized_differentiation": 6}, "public_verification_normalization_pass_count_by_method": {"hermite/1": 3, "hermite/2": 1, "heurisch_probe/0": 1, "rational/0": 2, "rational/1": 1, "rational/2": 1, "table_reused/0": 1}, "public_verification_normalization_pass_count_counts": {"0": 4, "1": 4, "2": 2}, "public_verification_normalization_reason_by_method": {"hermite/numeric_scaled_quotient": 1, "hermite/power_one_elision": 2, "hermite/quotient_numeric_factor_cancellation": 1, "rational/numeric_scaled_quotient": 1, "rational/symbolic_scaled_quotient": 1}, "public_verification_normalization_reason_counts": {"numeric_scaled_quotient": 2, "power_one_elision": 2, "quotient_numeric_factor_cancellation": 1, "symbolic_scaled_quotient": 1}, "publication_status_by_method": {"hermite/accepted": 4, "heurisch_probe/accepted": 1, "rational/accepted": 4, "rational/rejected_residual_reason": 1, "table_reused/accepted": 1, "table_reused/rejected_residual_reason": 2, "unsupported/rejected_no_antiderivative": 3}, "publication_status_counts": {"accepted": 10, "rejected_no_antiderivative": 3, "rejected_residual_reason": 3}, "required_condition_counts": {"nonzero": 6}, "residual_reason_by_method": {"rational/budget_exceeded": 1, "table_reused/verification_failed": 2, "unsupported/budget_exceeded": 1, "unsupported/disabled_by_mode": 1, "unsupported/unsupported_method": 1}, "residual_reason_counts": {"budget_exceeded": 2, "disabled_by_mode": 1, "unsupported_method": 1, "verification_failed": 2}, "trace_level_counts": {"algorithmic_summary": 13, "diagnostic_only": 3}, "unverified_fallback_acceptances": 0, "unverified_public_acceptances": 0, "verification_blocker_by_method": {"rational/budget_exceeded": 1, "table_reused/derivative_mismatch": 2}, "verification_blocker_counts": {"budget_exceeded": 1, "derivative_mismatch": 2}, "verification_budget_exceeded": 1, "verification_check_usage_by_method": {"hermite": 4, "heurisch_probe": 1, "rational": 4}, "verification_checks_used_total": 9, "verification_elapsed_ms": 2.125, "verification_evidence_by_method": {"hermite/normalized_differentiation": 4, "heurisch_probe/direct_differentiation": 1, "rational/direct_differentiation": 2, "rational/none": 1, "rational/normalized_differentiation": 2, "table_reused/direct_differentiation": 1, "table_reused/failed_differentiation": 2, "unsupported/none": 3}, "verification_evidence_counts": {"direct_differentiation": 4, "failed_differentiation": 2, "none": 4, "normalized_differentiation": 6}, "verification_normalization_pass_count_by_method": {"hermite/1": 3, "hermite/2": 1, "heurisch_probe/0": 1, "rational/0": 3, "rational/1": 1, "rational/2": 1, "table_reused/0": 3, "unsupported/0": 3}, "verification_normalization_pass_count_counts": {"0": 10, "1": 4, "2": 2}, "verification_normalization_reason_by_method": {"hermite/numeric_scaled_quotient": 1, "hermite/power_one_elision": 2, "hermite/quotient_numeric_factor_cancellation": 1, "rational/numeric_scaled_quotient": 1, "rational/symbolic_scaled_quotient": 1}, "verification_normalization_reason_counts": {"numeric_scaled_quotient": 2, "power_one_elision": 2, "quotient_numeric_factor_cancellation": 1, "symbolic_scaled_quotient": 1}, "verification_residual_by_method": {"table_reused/derivative_minus_integrand": 2}, "verification_residual_counts": {"derivative_minus_integrand": 2}, "verification_residual_kind_by_method": {"table_reused/depends_on_variable": 2}, "verification_residual_kind_counts": {"depends_on_variable": 2}, "verification_residual_signature_by_method": {"table_reused/affine_in_variable": 1, "table_reused/function_of_variable": 1}, "verification_residual_signature_counts": {"affine_in_variable": 1, "function_of_variable": 1}, "verification_status_by_method": {"hermite/verified": 4, "heurisch_probe/verified_under_conditions": 1, "rational/inconclusive": 1, "rational/verified_under_conditions": 4, "table_reused/failed": 2, "table_reused/verified": 1, "unsupported/inconclusive": 1, "unsupported/not_attempted": 2}, "verification_status_counts": {"failed": 2, "inconclusive": 2, "not_attempted": 2, "verified": 5, "verified_under_conditions": 5}}
test result: ok. 1 passed
"""
        )

        self.assertEqual(metrics["total_cases"], 16)
        self.assertEqual(metrics["passed"], 16)
        self.assertEqual(metrics["failed"], 0)
        self.assertEqual(metrics["backend_attempts"], 16)
        self.assertEqual(metrics["backend_public_accepted"], 10)
        self.assertEqual(metrics["backend_unverified_public_acceptances"], 0)
        self.assertEqual(metrics["backend_fallback_eligible"], 1)
        self.assertEqual(metrics["backend_unverified_fallback_acceptances"], 0)
        self.assertEqual(metrics["backend_verified_count"], 10)
        self.assertEqual(metrics["backend_failed_or_blocked_count"], 6)
        self.assertEqual(metrics["backend_required_condition_count"], 6)
        self.assertEqual(metrics["backend_budget_exceeded_count"], 2)
        self.assertEqual(metrics["backend_method_probe_budget_exhausted_count"], 1)
        self.assertEqual(metrics["backend_verification_budget_exceeded_count"], 1)
        self.assertEqual(metrics["backend_method_probes_used_total"], 19)
        self.assertEqual(metrics["backend_verification_checks_used_total"], 9)
        self.assertEqual(metrics["backend_verification_elapsed_ms"], 2.125)
        self.assertEqual(metrics["backend_verification_pressure_status"], "watch")
        self.assertEqual(
            metrics["backend_verification_pressure"],
            {
                "status": "watch",
                "primary_signal": "normalization_passes",
                "reason": "max_passes=2",
                "attempts": 16,
                "verification_checks_used_total": 9,
                "checks_per_attempt": 0.562,
                "max_verification_normalization_passes": 2,
                "verification_elapsed_ms": 2.125,
            },
        )
        self.assertEqual(metrics["backend_assumption_exprs"], 0)
        self.assertEqual(metrics["backend_public_assumption_exprs"], 0)
        self.assertEqual(metrics["backend_fallback_assumption_exprs"], 0)
        self.assertEqual(
            metrics["backend_verification_evidence_counts"],
            {
                "direct_differentiation": 4,
                "failed_differentiation": 2,
                "none": 4,
                "normalized_differentiation": 6,
            },
        )
        self.assertEqual(
            metrics["backend_public_verification_evidence_counts"],
            {"direct_differentiation": 4, "normalized_differentiation": 6},
        )
        self.assertEqual(
            metrics["backend_fallback_verification_evidence_counts"],
            {"direct_differentiation": 1},
        )
        self.assertEqual(
            metrics["backend_verification_evidence_by_method"],
            {
                "hermite/normalized_differentiation": 4,
                "heurisch_probe/direct_differentiation": 1,
                "rational/direct_differentiation": 2,
                "rational/none": 1,
                "rational/normalized_differentiation": 2,
                "table_reused/direct_differentiation": 1,
                "table_reused/failed_differentiation": 2,
                "unsupported/none": 3,
            },
        )
        self.assertEqual(
            metrics["backend_public_verification_evidence_by_method"],
            {
                "hermite/normalized_differentiation": 4,
                "heurisch_probe/direct_differentiation": 1,
                "rational/direct_differentiation": 2,
                "rational/normalized_differentiation": 2,
                "table_reused/direct_differentiation": 1,
            },
        )
        self.assertEqual(
            metrics["backend_fallback_verification_evidence_by_method"],
            {"rational/direct_differentiation": 1},
        )
        self.assertEqual(
            metrics["backend_verification_normalization_reason_counts"],
            {
                "numeric_scaled_quotient": 2,
                "power_one_elision": 2,
                "quotient_numeric_factor_cancellation": 1,
                "symbolic_scaled_quotient": 1,
            },
        )
        self.assertEqual(
            metrics["backend_public_verification_normalization_reason_counts"],
            {
                "numeric_scaled_quotient": 2,
                "power_one_elision": 2,
                "quotient_numeric_factor_cancellation": 1,
                "symbolic_scaled_quotient": 1,
            },
        )
        self.assertEqual(
            metrics["backend_fallback_verification_normalization_reason_counts"],
            {},
        )
        self.assertEqual(
            metrics["backend_verification_normalization_reason_by_method"],
            {
                "hermite/numeric_scaled_quotient": 1,
                "hermite/power_one_elision": 2,
                "hermite/quotient_numeric_factor_cancellation": 1,
                "rational/numeric_scaled_quotient": 1,
                "rational/symbolic_scaled_quotient": 1,
            },
        )
        self.assertEqual(
            metrics["backend_public_verification_normalization_reason_by_method"],
            {
                "hermite/numeric_scaled_quotient": 1,
                "hermite/power_one_elision": 2,
                "hermite/quotient_numeric_factor_cancellation": 1,
                "rational/numeric_scaled_quotient": 1,
                "rational/symbolic_scaled_quotient": 1,
            },
        )
        self.assertEqual(
            metrics["backend_fallback_verification_normalization_reason_by_method"],
            {},
        )
        self.assertEqual(metrics["backend_max_verification_normalization_passes"], 2)
        self.assertEqual(
            metrics["backend_public_max_verification_normalization_passes"], 2
        )
        self.assertEqual(
            metrics["backend_fallback_max_verification_normalization_passes"], 0
        )
        self.assertEqual(
            metrics["backend_verification_normalization_pass_count_counts"],
            {"0": 10, "1": 4, "2": 2},
        )
        self.assertEqual(
            metrics["backend_public_verification_normalization_pass_count_counts"],
            {"0": 4, "1": 4, "2": 2},
        )
        self.assertEqual(
            metrics["backend_fallback_verification_normalization_pass_count_counts"],
            {"0": 1},
        )
        self.assertEqual(
            metrics["backend_verification_normalization_pass_count_by_method"],
            {
                "hermite/1": 3,
                "hermite/2": 1,
                "heurisch_probe/0": 1,
                "rational/0": 3,
                "rational/1": 1,
                "rational/2": 1,
                "table_reused/0": 3,
                "unsupported/0": 3,
            },
        )
        self.assertEqual(
            metrics["backend_public_verification_normalization_pass_count_by_method"],
            {
                "hermite/1": 3,
                "hermite/2": 1,
                "heurisch_probe/0": 1,
                "rational/0": 2,
                "rational/1": 1,
                "rational/2": 1,
                "table_reused/0": 1,
            },
        )
        self.assertEqual(
            metrics["backend_fallback_verification_normalization_pass_count_by_method"],
            {"rational/0": 1},
        )
        self.assertEqual(
            metrics["backend_publication_status_counts"],
            {
                "accepted": 10,
                "rejected_no_antiderivative": 3,
                "rejected_residual_reason": 3,
            },
        )
        self.assertEqual(
            metrics["backend_publication_status_by_method"],
            {
                "hermite/accepted": 4,
                "heurisch_probe/accepted": 1,
                "rational/accepted": 4,
                "rational/rejected_residual_reason": 1,
                "table_reused/accepted": 1,
                "table_reused/rejected_residual_reason": 2,
                "unsupported/rejected_no_antiderivative": 3,
            },
        )
        self.assertEqual(
            metrics["backend_fallback_status_counts"],
            {
                "blocked_by_candidate_policy": 6,
                "blocked_by_mode": 9,
                "eligible": 1,
            },
        )
        self.assertEqual(
            metrics["backend_fallback_status_by_method"],
            {
                "hermite/blocked_by_mode": 4,
                "heurisch_probe/blocked_by_mode": 1,
                "rational/blocked_by_candidate_policy": 1,
                "rational/blocked_by_mode": 3,
                "rational/eligible": 1,
                "table_reused/blocked_by_candidate_policy": 2,
                "table_reused/blocked_by_mode": 1,
                "unsupported/blocked_by_candidate_policy": 3,
            },
        )
        self.assertEqual(
            metrics["backend_mode_counts"],
            {"diagnostic_only": 14, "disabled": 1, "residual_fallback": 1},
        )
        self.assertEqual(
            metrics["backend_method_counts"],
            {
                "hermite": 4,
                "heurisch_probe": 1,
                "rational": 5,
                "table_reused": 3,
                "unsupported": 3,
            },
        )
        self.assertEqual(
            metrics["backend_method_probe_usage_by_method"],
            {"hermite": 8, "heurisch_probe": 3, "rational": 5, "unsupported": 3},
        )
        self.assertEqual(
            metrics["backend_method_probe_attempt_counts"],
            {"hermite": 6, "heurisch_probe": 2, "rational": 11},
        )
        self.assertEqual(
            metrics["backend_method_probe_candidate_counts"],
            {"hermite": 4, "heurisch_probe": 1, "rational": 5},
        )
        self.assertEqual(
            metrics["backend_method_probe_no_match_counts"],
            {"hermite": 2, "heurisch_probe": 1, "rational": 6},
        )
        self.assertEqual(
            metrics["backend_method_probe_no_match_reason_counts"],
            {
                "hermite/denominator_policy_mismatch": 1,
                "hermite/shape_mismatch": 1,
                "heurisch_probe/shape_mismatch": 1,
                "rational/numerator_policy_mismatch": 5,
                "rational/shape_mismatch": 1,
            },
        )
        self.assertEqual(
            metrics["backend_verification_check_usage_by_method"],
            {"hermite": 4, "heurisch_probe": 1, "rational": 4},
        )
        self.assertEqual(
            metrics["backend_verification_status_by_method"],
            {
                "hermite/verified": 4,
                "heurisch_probe/verified_under_conditions": 1,
                "rational/inconclusive": 1,
                "rational/verified_under_conditions": 4,
                "table_reused/failed": 2,
                "table_reused/verified": 1,
                "unsupported/inconclusive": 1,
                "unsupported/not_attempted": 2,
            },
        )
        self.assertEqual(
            metrics["backend_residual_reason_by_method"],
            {
                "rational/budget_exceeded": 1,
                "table_reused/verification_failed": 2,
                "unsupported/budget_exceeded": 1,
                "unsupported/disabled_by_mode": 1,
                "unsupported/unsupported_method": 1,
            },
        )
        self.assertEqual(
            metrics["backend_verification_blocker_counts"],
            {"budget_exceeded": 1, "derivative_mismatch": 2},
        )
        self.assertEqual(
            metrics["backend_verification_blocker_by_method"],
            {
                "rational/budget_exceeded": 1,
                "table_reused/derivative_mismatch": 2,
            },
        )
        self.assertEqual(
            metrics["backend_failure_class_counts"],
            {
                "budget_exceeded": 2,
                "disabled_by_mode": 1,
                "residual_affine_in_variable": 1,
                "residual_function_of_variable": 1,
                "unsupported_method": 1,
            },
        )
        self.assertEqual(
            metrics["backend_failure_class_by_method"],
            {
                "rational/budget_exceeded": 1,
                "table_reused/residual_affine_in_variable": 1,
                "table_reused/residual_function_of_variable": 1,
                "unsupported/budget_exceeded": 1,
                "unsupported/disabled_by_mode": 1,
                "unsupported/unsupported_method": 1,
            },
        )
        self.assertEqual(
            metrics["backend_verification_residual_counts"],
            {"derivative_minus_integrand": 2},
        )
        self.assertEqual(
            metrics["backend_verification_residual_by_method"],
            {"table_reused/derivative_minus_integrand": 2},
        )
        self.assertEqual(
            metrics["backend_verification_residual_kind_counts"],
            {"depends_on_variable": 2},
        )
        self.assertEqual(
            metrics["backend_verification_residual_kind_by_method"],
            {"table_reused/depends_on_variable": 2},
        )
        self.assertEqual(
            metrics["backend_verification_residual_signature_counts"],
            {"affine_in_variable": 1, "function_of_variable": 1},
        )
        self.assertEqual(
            metrics["backend_verification_residual_signature_by_method"],
            {"table_reused/affine_in_variable": 1, "table_reused/function_of_variable": 1},
        )
        self.assertEqual(
            metrics["backend_trace_level_counts"],
            {"algorithmic_summary": 13, "diagnostic_only": 3},
        )
        self.assertEqual(
            metrics["backend_constant_policy_counts"],
            {"arbitrary_constant_omitted": 13, "unspecified": 3},
        )
        self.assertEqual(
            metrics["backend_public_trace_level_counts"],
            {"algorithmic_summary": 10},
        )
        self.assertEqual(
            metrics["backend_public_constant_policy_counts"],
            {"arbitrary_constant_omitted": 10},
        )
        self.assertEqual(
            metrics["backend_fallback_trace_level_counts"],
            {"algorithmic_summary": 1},
        )
        self.assertEqual(
            metrics["backend_fallback_constant_policy_counts"],
            {"arbitrary_constant_omitted": 1},
        )
        self.assertEqual(
            metrics["backend_verification_status_counts"],
            {
                "failed": 2,
                "inconclusive": 2,
                "not_attempted": 2,
                "verified": 5,
                "verified_under_conditions": 5,
            },
        )
        self.assertEqual(
            metrics["backend_residual_reason_counts"],
            {
                "budget_exceeded": 2,
                "disabled_by_mode": 1,
                "unsupported_method": 1,
                "verification_failed": 2,
            },
        )
        self.assertEqual(metrics["backend_required_condition_counts"], {"nonzero": 6})
        self.assertEqual(
            MODULE.suite_status(
                "calculus_integrate_backend_observability",
                metrics,
                0,
            ),
            "pass",
        )

        metrics["backend_unverified_public_acceptances"] = 1
        self.assertEqual(
            MODULE.suite_status(
                "calculus_integrate_backend_observability",
                metrics,
                0,
            ),
            "fail",
        )
        metrics["backend_unverified_public_acceptances"] = 0
        metrics["backend_unverified_fallback_acceptances"] = 1
        self.assertEqual(
            MODULE.suite_status(
                "calculus_integrate_backend_observability",
                metrics,
                0,
            ),
            "fail",
        )
        metrics["backend_unverified_fallback_acceptances"] = 0
        metrics["backend_public_trace_level_counts"] = {"diagnostic_only": 1}
        self.assertEqual(
            MODULE.suite_status(
                "calculus_integrate_backend_observability",
                metrics,
                0,
            ),
            "fail",
        )
        metrics["backend_public_trace_level_counts"] = {"algorithmic_summary": 10}
        metrics["backend_public_constant_policy_counts"] = {"unspecified": 1}
        self.assertEqual(
            MODULE.suite_status(
                "calculus_integrate_backend_observability",
                metrics,
                0,
            ),
            "fail",
        )
        metrics["backend_public_constant_policy_counts"] = {
            "arbitrary_constant_omitted": 9
        }
        metrics["backend_public_assumption_exprs"] = 1
        self.assertEqual(
            MODULE.suite_status(
                "calculus_integrate_backend_observability",
                metrics,
                0,
            ),
            "fail",
        )
        metrics["backend_public_assumption_exprs"] = 0
        metrics["backend_fallback_assumption_exprs"] = 1
        self.assertEqual(
            MODULE.suite_status(
                "calculus_integrate_backend_observability",
                metrics,
                0,
            ),
            "fail",
        )
        metrics["backend_fallback_assumption_exprs"] = 0
        metrics["backend_public_verification_evidence_counts"] = {"none": 1}
        self.assertEqual(
            MODULE.suite_status(
                "calculus_integrate_backend_observability",
                metrics,
                0,
            ),
            "fail",
        )
        metrics["backend_public_verification_evidence_counts"] = {
            "failed_differentiation": 1
        }
        self.assertEqual(
            MODULE.suite_status(
                "calculus_integrate_backend_observability",
                metrics,
                0,
            ),
            "fail",
        )
        metrics["backend_public_verification_evidence_counts"] = {
            "direct_differentiation": 4,
            "normalized_differentiation": 6,
        }
        metrics["backend_fallback_verification_evidence_counts"] = {"none": 1}
        self.assertEqual(
            MODULE.suite_status(
                "calculus_integrate_backend_observability",
                metrics,
                0,
            ),
            "fail",
        )
        metrics["backend_fallback_verification_evidence_counts"] = {
            "failed_differentiation": 1
        }
        self.assertEqual(
            MODULE.suite_status(
                "calculus_integrate_backend_observability",
                metrics,
                0,
            ),
            "fail",
        )

    def test_algorithmic_backend_observability_requires_residual_kind_counts(self):
        payload = self.minimal_algorithmic_backend_residual_payload()
        del payload["verification_residual_kind_counts"]
        del payload["verification_residual_kind_by_method"]

        with self.assertRaisesRegex(
            ValueError,
            "verification_residual_kind_counts missing verification residuals",
        ):
            MODULE.parse_algorithmic_backend_observability(
                self.algorithmic_backend_observability_output(payload)
            )

    def test_algorithmic_backend_observability_requires_failure_class_counts(self):
        payload = self.minimal_algorithmic_backend_residual_payload()
        del payload["failure_class_counts"]
        del payload["failure_class_by_method"]

        with self.assertRaisesRegex(
            ValueError,
            "failure_class_counts missing rejected candidates",
        ):
            MODULE.parse_algorithmic_backend_observability(
                self.algorithmic_backend_observability_output(payload)
            )

    def test_algorithmic_backend_observability_requires_failure_class_by_method(
        self,
    ):
        payload = self.minimal_algorithmic_backend_residual_payload()
        del payload["failure_class_by_method"]

        with self.assertRaisesRegex(
            ValueError,
            "failure_class_by_method missing rejected candidates",
        ):
            MODULE.parse_algorithmic_backend_observability(
                self.algorithmic_backend_observability_output(payload)
            )

    def test_algorithmic_backend_observability_requires_residual_kind_by_method(self):
        payload = self.minimal_algorithmic_backend_residual_payload()
        del payload["verification_residual_kind_by_method"]

        with self.assertRaisesRegex(
            ValueError,
            "verification_residual_kind_by_method missing verification residual methods",
        ):
            MODULE.parse_algorithmic_backend_observability(
                self.algorithmic_backend_observability_output(payload)
            )

    def test_algorithmic_backend_observability_requires_residual_signature_counts(self):
        payload = self.minimal_algorithmic_backend_residual_payload()
        del payload["verification_residual_signature_counts"]
        del payload["verification_residual_signature_by_method"]

        with self.assertRaisesRegex(
            ValueError,
            "verification_residual_signature_counts missing verification residuals",
        ):
            MODULE.parse_algorithmic_backend_observability(
                self.algorithmic_backend_observability_output(payload)
            )

    def test_algorithmic_backend_observability_requires_probe_attempt_counts(self):
        payload = self.minimal_algorithmic_backend_residual_payload()
        payload["method_probes_used_total"] = 1
        payload["method_probe_usage_by_method"] = {"table_reused": 1}

        with self.assertRaisesRegex(
            ValueError,
            "method_probe_attempt_counts missing method probes",
        ):
            MODULE.parse_algorithmic_backend_observability(
                self.algorithmic_backend_observability_output(payload)
            )

    def test_algorithmic_backend_observability_requires_probe_candidate_counts(self):
        payload = self.minimal_algorithmic_backend_residual_payload()
        payload["method_probes_used_total"] = 1
        payload["method_probe_usage_by_method"] = {"rational": 1}
        payload["method_probe_attempt_counts"] = {"rational": 1}
        payload["method_probe_no_match_counts"] = {"rational": 1}

        with self.assertRaisesRegex(
            ValueError,
            "method_probe_candidate_counts missing method probes",
        ):
            MODULE.parse_algorithmic_backend_observability(
                self.algorithmic_backend_observability_output(payload)
            )

    def test_algorithmic_backend_observability_requires_probe_no_match_counts(self):
        payload = self.minimal_algorithmic_backend_residual_payload()
        payload["method_probes_used_total"] = 1
        payload["method_probe_usage_by_method"] = {"rational": 1}
        payload["method_probe_attempt_counts"] = {"rational": 1}
        payload["method_probe_candidate_counts"] = {"rational": 1}

        with self.assertRaisesRegex(
            ValueError,
            "method_probe_no_match_counts missing method probes",
        ):
            MODULE.parse_algorithmic_backend_observability(
                self.algorithmic_backend_observability_output(payload)
            )

    def test_algorithmic_backend_observability_validates_probe_yield_split(self):
        payload = self.minimal_algorithmic_backend_residual_payload()
        payload["method_probes_used_total"] = 2
        payload["method_probe_usage_by_method"] = {"unsupported": 2}
        payload["method_probe_attempt_counts"] = {"rational": 2}
        payload["method_probe_candidate_counts"] = {"rational": 1}
        payload["method_probe_no_match_counts"] = {"rational": 2}

        with self.assertRaisesRegex(
            ValueError,
            "method_probe candidate/no-match split does not match attempts",
        ):
            MODULE.parse_algorithmic_backend_observability(
                self.algorithmic_backend_observability_output(payload)
            )

    def test_algorithmic_backend_observability_requires_probe_no_match_reasons(self):
        payload = self.minimal_algorithmic_backend_residual_payload()
        payload["method_probes_used_total"] = 1
        payload["method_probe_usage_by_method"] = {"unsupported": 1}
        payload["method_probe_attempt_counts"] = {"rational": 1}
        payload["method_probe_candidate_counts"] = {}
        payload["method_probe_no_match_counts"] = {"rational": 1}

        with self.assertRaisesRegex(
            ValueError,
            "method_probe_no_match_reason_counts missing method probe no-matches",
        ):
            MODULE.parse_algorithmic_backend_observability(
                self.algorithmic_backend_observability_output(payload)
            )

    def test_algorithmic_backend_observability_validates_probe_no_match_reasons(
        self,
    ):
        payload = self.minimal_algorithmic_backend_residual_payload()
        payload["method_probes_used_total"] = 1
        payload["method_probe_usage_by_method"] = {"unsupported": 1}
        payload["method_probe_attempt_counts"] = {"rational": 1}
        payload["method_probe_candidate_counts"] = {}
        payload["method_probe_no_match_counts"] = {"rational": 1}
        payload["method_probe_no_match_reason_counts"] = {
            "hermite/shape_mismatch": 1
        }

        with self.assertRaisesRegex(
            ValueError,
            "method_probe_no_match_reason_counts do not match "
            "method_probe_no_match_counts by method",
        ):
            MODULE.parse_algorithmic_backend_observability(
                self.algorithmic_backend_observability_output(payload)
            )

    def test_algorithmic_backend_observability_requires_residual_signature_by_method(
        self,
    ):
        payload = self.minimal_algorithmic_backend_residual_payload()
        del payload["verification_residual_signature_by_method"]

        with self.assertRaisesRegex(
            ValueError,
            "verification_residual_signature_by_method missing verification residual methods",
        ):
            MODULE.parse_algorithmic_backend_observability(
                self.algorithmic_backend_observability_output(payload)
            )

    @staticmethod
    def minimal_algorithmic_backend_residual_payload():
        return {
            "attempts": 1,
            "public_accepted": 0,
            "unverified_public_acceptances": 0,
            "fallback_eligible": 0,
            "unverified_fallback_acceptances": 0,
            "method_probe_budget_exhausted": 0,
            "verification_budget_exceeded": 0,
            "method_probes_used_total": 0,
            "verification_checks_used_total": 0,
            "verification_elapsed_ms": 0.0,
            "max_verification_normalization_passes": 0,
            "public_max_verification_normalization_passes": 0,
            "fallback_max_verification_normalization_passes": 0,
            "method_counts": {"table_reused": 1},
            "verification_status_counts": {"failed": 1},
            "verification_evidence_counts": {"failed_differentiation": 1},
            "verification_evidence_by_method": {
                "table_reused/failed_differentiation": 1
            },
            "verification_normalization_pass_count_counts": {"0": 1},
            "verification_normalization_pass_count_by_method": {"table_reused/0": 1},
            "failure_class_counts": {"residual_affine_in_variable": 1},
            "failure_class_by_method": {
                "table_reused/residual_affine_in_variable": 1
            },
            "verification_residual_counts": {"derivative_minus_integrand": 1},
            "verification_residual_by_method": {
                "table_reused/derivative_minus_integrand": 1
            },
            "verification_residual_kind_counts": {"depends_on_variable": 1},
            "verification_residual_kind_by_method": {
                "table_reused/depends_on_variable": 1
            },
            "verification_residual_signature_counts": {"affine_in_variable": 1},
            "verification_residual_signature_by_method": {
                "table_reused/affine_in_variable": 1
            },
            "residual_reason_counts": {"verification_failed": 1},
        }

    @staticmethod
    def algorithmic_backend_observability_output(payload):
        return (
            "running 1 test\n"
            f"algorithmic_backend_observability: {json.dumps(payload, sort_keys=True)}\n"
            "test result: ok. 1 passed\n"
        )

    def test_algorithmic_backend_observability_renders_backend_signal(self):
        metrics = {'backend_assumption_exprs': 0,
         'backend_attempts': 16,
         'backend_budget_exceeded_count': 2,
         'backend_constant_policy_counts': {'arbitrary_constant_omitted': 13, 'unspecified': 3},
         'backend_failure_class_by_method': {'rational/budget_exceeded': 1,
                                             'table_reused/residual_affine_in_variable': 1,
                                             'table_reused/residual_function_of_variable': 1,
                                             'unsupported/budget_exceeded': 1,
                                             'unsupported/disabled_by_mode': 1,
                                             'unsupported/unsupported_method': 1},
         'backend_failure_class_counts': {'budget_exceeded': 2,
                                          'disabled_by_mode': 1,
                                          'residual_affine_in_variable': 1,
                                          'residual_function_of_variable': 1,
                                          'unsupported_method': 1},
         'backend_failed_or_blocked_count': 6,
         'backend_fallback_assumption_exprs': 0,
         'backend_fallback_constant_policy_counts': {'arbitrary_constant_omitted': 1},
         'backend_fallback_eligible': 1,
         'backend_fallback_status_by_method': {'hermite/blocked_by_mode': 4,
                                               'heurisch_probe/blocked_by_mode': 1,
                                               'rational/blocked_by_candidate_policy': 1,
                                               'rational/blocked_by_mode': 3,
                                               'rational/eligible': 1,
                                               'table_reused/blocked_by_candidate_policy': 2,
                                               'table_reused/blocked_by_mode': 1,
                                               'unsupported/blocked_by_candidate_policy': 3},
         'backend_fallback_status_counts': {'blocked_by_candidate_policy': 6,
                                            'blocked_by_mode': 9,
                                            'eligible': 1},
         'backend_fallback_trace_level_counts': {'algorithmic_summary': 1},
         'backend_fallback_verification_evidence_by_method': {'rational/direct_differentiation': 1},
         'backend_fallback_verification_evidence_counts': {'direct_differentiation': 1},
         'backend_fallback_verification_normalization_reason_by_method': {},
         'backend_fallback_verification_normalization_reason_counts': {},
         'backend_method_counts': {'hermite': 4,
                                   'heurisch_probe': 1,
                                   'rational': 5,
                                   'table_reused': 3,
                                   'unsupported': 3},
         'backend_method_probe_attempt_counts': {'hermite': 6, 'heurisch_probe': 2, 'rational': 11},
         'backend_method_probe_budget_exhausted_count': 1,
         'backend_method_probe_candidate_counts': {'hermite': 4, 'heurisch_probe': 1, 'rational': 5},
         'backend_method_probe_no_match_counts': {'hermite': 2, 'heurisch_probe': 1, 'rational': 6},
         'backend_method_probe_no_match_reason_counts': {'hermite/denominator_policy_mismatch': 1,
                                                         'hermite/shape_mismatch': 1,
                                                         'heurisch_probe/shape_mismatch': 1,
                                                         'rational/numerator_policy_mismatch': 5,
                                                         'rational/shape_mismatch': 1},
         'backend_method_probe_usage_by_method': {'hermite': 8,
                                                  'heurisch_probe': 3,
                                                  'rational': 5,
                                                  'unsupported': 3},
         'backend_method_probes_used_total': 19,
         'backend_mode_counts': {'diagnostic_only': 14, 'disabled': 1, 'residual_fallback': 1},
         'backend_public_accepted': 10,
         'backend_public_assumption_exprs': 0,
         'backend_public_constant_policy_counts': {'arbitrary_constant_omitted': 10},
         'backend_public_trace_level_counts': {'algorithmic_summary': 10},
         'backend_public_verification_evidence_by_method': {'hermite/normalized_differentiation': 4,
                                                            'heurisch_probe/direct_differentiation': 1,
                                                            'rational/direct_differentiation': 2,
                                                            'rational/normalized_differentiation': 2,
                                                            'table_reused/direct_differentiation': 1},
         'backend_public_verification_evidence_counts': {'direct_differentiation': 4,
                                                         'normalized_differentiation': 6},
         'backend_public_verification_normalization_reason_by_method': {'hermite/numeric_scaled_quotient': 1,
                                                                        'hermite/power_one_elision': 2,
                                                                        'hermite/quotient_numeric_factor_cancellation': 1,
                                                                        'rational/numeric_scaled_quotient': 1,
                                                                        'rational/symbolic_scaled_quotient': 1},
         'backend_public_verification_normalization_reason_counts': {'numeric_scaled_quotient': 2,
                                                                     'power_one_elision': 2,
                                                                     'quotient_numeric_factor_cancellation': 1,
                                                                     'symbolic_scaled_quotient': 1},
         'backend_publication_status_by_method': {'hermite/accepted': 4,
                                                  'heurisch_probe/accepted': 1,
                                                  'rational/accepted': 4,
                                                  'rational/rejected_residual_reason': 1,
                                                  'table_reused/accepted': 1,
                                                  'table_reused/rejected_residual_reason': 2,
                                                  'unsupported/rejected_no_antiderivative': 3},
         'backend_publication_status_counts': {'accepted': 10,
                                               'rejected_no_antiderivative': 3,
                                               'rejected_residual_reason': 3},
         'backend_required_condition_count': 6,
         'backend_required_condition_counts': {'nonzero': 6},
         'backend_residual_reason_by_method': {'rational/budget_exceeded': 1,
                                               'table_reused/verification_failed': 2,
                                               'unsupported/budget_exceeded': 1,
                                               'unsupported/disabled_by_mode': 1,
                                               'unsupported/unsupported_method': 1},
         'backend_residual_reason_counts': {'budget_exceeded': 2,
                                            'disabled_by_mode': 1,
                                            'unsupported_method': 1,
                                            'verification_failed': 2},
         'backend_trace_level_counts': {'algorithmic_summary': 13, 'diagnostic_only': 3},
         'backend_unverified_fallback_acceptances': 0,
         'backend_unverified_public_acceptances': 0,
         'backend_verification_blocker_by_method': {'rational/budget_exceeded': 1,
                                                    'table_reused/derivative_mismatch': 2},
         'backend_verification_blocker_counts': {'budget_exceeded': 1, 'derivative_mismatch': 2},
         'backend_verification_budget_exceeded_count': 1,
         'backend_verification_check_usage_by_method': {'hermite': 4, 'heurisch_probe': 1, 'rational': 4},
         'backend_verification_checks_used_total': 9,
         'backend_verification_elapsed_ms': 2.125,
         'backend_verification_pressure': {'attempts': 16,
                                           'checks_per_attempt': 0.562,
                                           'max_verification_normalization_passes': 2,
                                           'primary_signal': 'normalization_passes',
                                           'reason': 'max_passes=2',
                                           'status': 'watch',
                                           'verification_checks_used_total': 9,
                                           'verification_elapsed_ms': 2.125},
         'backend_verification_pressure_status': 'watch',
         'backend_max_verification_normalization_passes': 2,
         'backend_public_max_verification_normalization_passes': 2,
         'backend_fallback_max_verification_normalization_passes': 0,
         'backend_verification_evidence_by_method': {'hermite/normalized_differentiation': 4,
                                                     'heurisch_probe/direct_differentiation': 1,
                                                     'rational/direct_differentiation': 2,
                                                     'rational/none': 1,
                                                     'rational/normalized_differentiation': 2,
                                                     'table_reused/direct_differentiation': 1,
                                                     'table_reused/failed_differentiation': 2,
                                                     'unsupported/none': 3},
         'backend_verification_evidence_counts': {'direct_differentiation': 4,
                                                  'failed_differentiation': 2,
                                                  'none': 4,
                                                  'normalized_differentiation': 6},
         'backend_verification_normalization_reason_by_method': {'hermite/numeric_scaled_quotient': 1,
                                                                 'hermite/power_one_elision': 2,
                                                                 'hermite/quotient_numeric_factor_cancellation': 1,
                                                                 'rational/numeric_scaled_quotient': 1,
                                                                 'rational/symbolic_scaled_quotient': 1},
         'backend_verification_normalization_reason_counts': {'numeric_scaled_quotient': 2,
                                                              'power_one_elision': 2,
                                                              'quotient_numeric_factor_cancellation': 1,
                                                              'symbolic_scaled_quotient': 1},
         'backend_verification_normalization_pass_count_counts': {'0': 10, '1': 4, '2': 2},
         'backend_public_verification_normalization_pass_count_counts': {'0': 4, '1': 4, '2': 2},
         'backend_fallback_verification_normalization_pass_count_counts': {'0': 1},
         'backend_verification_normalization_pass_count_by_method': {'hermite/1': 3,
                                                                     'hermite/2': 1,
                                                                     'heurisch_probe/0': 1,
                                                                     'rational/0': 3,
                                                                     'rational/1': 1,
                                                                     'rational/2': 1,
                                                                     'table_reused/0': 3,
                                                                     'unsupported/0': 3},
         'backend_public_verification_normalization_pass_count_by_method': {'hermite/1': 3,
                                                                            'hermite/2': 1,
                                                                            'heurisch_probe/0': 1,
                                                                            'rational/0': 2,
                                                                            'rational/1': 1,
                                                                            'rational/2': 1,
                                                                            'table_reused/0': 1},
         'backend_fallback_verification_normalization_pass_count_by_method': {'rational/0': 1},
         'backend_verification_residual_by_method': {'table_reused/derivative_minus_integrand': 2},
         'backend_verification_residual_counts': {'derivative_minus_integrand': 2},
         'backend_verification_residual_kind_by_method': {'table_reused/depends_on_variable': 2},
         'backend_verification_residual_kind_counts': {'depends_on_variable': 2},
         'backend_verification_residual_signature_by_method': {'table_reused/affine_in_variable': 1,
                                                               'table_reused/function_of_variable': 1},
         'backend_verification_residual_signature_counts': {'affine_in_variable': 1,
                                                            'function_of_variable': 1},
         'backend_verification_status_by_method': {'hermite/verified': 4,
                                                   'heurisch_probe/verified_under_conditions': 1,
                                                   'rational/inconclusive': 1,
                                                   'rational/verified_under_conditions': 4,
                                                   'table_reused/failed': 2,
                                                   'table_reused/verified': 1,
                                                   'unsupported/inconclusive': 1,
                                                   'unsupported/not_attempted': 2},
         'backend_verification_status_counts': {'failed': 2,
                                                'inconclusive': 2,
                                                'not_attempted': 2,
                                                'verified': 5,
                                                'verified_under_conditions': 5},
         'backend_verified_count': 10,
         'failed': 0,
         'passed': 16,
         'total_cases': 16}
        scorecard = {
            "generated_at": "2026-06-05T00:00:00+00:00",
            "profile": "fast",
            "git": {"branch": "main", "commit": "abc123"},
            "suites": {
                "calculus_integrate_backend_observability": {
                    "category": "observability",
                    "status": "pass",
                    "elapsed_seconds": 0.25,
                    "metrics": metrics,
                    "delta": {},
                },
            },
        }

        rendered = MODULE.render_markdown(scorecard)

        self.assertIn("integrate_backend_observability", rendered)
        self.assertIn("backend_attempts=16", rendered)
        self.assertIn("backend_public_accepted=10", rendered)
        self.assertIn("backend_fallback_eligible=1", rendered)
        self.assertIn("backend_unverified_fallback=0", rendered)
        self.assertIn("backend_required_conditions=6", rendered)
        self.assertIn("backend_budget_exceeded=2", rendered)
        self.assertIn("backend_method_budget_exhausted=1", rendered)
        self.assertIn("backend_verification_budget_exceeded=1", rendered)
        self.assertIn("backend_method_probes_used=19", rendered)
        self.assertIn("backend_verification_checks_used=9", rendered)
        self.assertIn("backend_verify=2.125ms", rendered)
        self.assertIn("backend_verification_pressure=watch", rendered)
        self.assertIn(
            "backend modes: diagnostic_only=14, disabled=1, residual_fallback=1",
            rendered,
        )
        self.assertIn(
            "backend methods: hermite=4, heurisch_probe=1, rational=5, table_reused=3, unsupported=3",
            rendered,
        )
        self.assertIn(
            "verification statuses: failed=2, inconclusive=2, not_attempted=2, verified=5, verified_under_conditions=5",
            rendered,
        )
        self.assertIn(
            "residual reasons: budget_exceeded=2, disabled_by_mode=1, unsupported_method=1, verification_failed=2",
            rendered,
        )
        self.assertIn(
            "budget split: method_probe_exhausted=1, verification_exceeded=1",
            rendered,
        )
        self.assertIn(
            "budget usage: method_probes_used=19, verification_checks_used=9",
            rendered,
        )
        self.assertIn(
            "backend verification pressure: status=watch primary=normalization_passes reason=max_passes=2 attempts=16 checks=9 checks_per_attempt=0.562 max_passes=2 elapsed_ms=2.125",
            rendered,
        )
        self.assertIn(
            "method-probe usage by method: hermite=8, heurisch_probe=3, rational=5, unsupported=3",
            rendered,
        )
        self.assertIn(
            "method-probe attempts: hermite=6, heurisch_probe=2, rational=11",
            rendered,
        )
        self.assertIn(
            "method-probe candidates: hermite=4, heurisch_probe=1, rational=5",
            rendered,
        )
        self.assertIn(
            "method-probe no-matches: hermite=2, heurisch_probe=1, rational=6",
            rendered,
        )
        self.assertIn(
            "method-probe no-match reasons: hermite/denominator_policy_mismatch=1, hermite/shape_mismatch=1, heurisch_probe/shape_mismatch=1, rational/numerator_policy_mismatch=5, rational/shape_mismatch=1",
            rendered,
        )
        self.assertIn(
            "verification-check usage by method: hermite=4, heurisch_probe=1, rational=4",
            rendered,
        )
        self.assertIn(
            "verification status by method: hermite/verified=4, heurisch_probe/verified_under_conditions=1, rational/inconclusive=1, rational/verified_under_conditions=4, table_reused/failed=2, table_reused/verified=1, unsupported/inconclusive=1, unsupported/not_attempted=2",
            rendered,
        )
        self.assertIn(
            "residual reason by method: rational/budget_exceeded=1, table_reused/verification_failed=2, unsupported/budget_exceeded=1, unsupported/disabled_by_mode=1, unsupported/unsupported_method=1",
            rendered,
        )
        self.assertIn(
            "verification blockers: budget_exceeded=1, derivative_mismatch=2",
            rendered,
        )
        self.assertIn(
            "verification blocker by method: rational/budget_exceeded=1, table_reused/derivative_mismatch=2",
            rendered,
        )
        self.assertIn(
            "failure classes: budget_exceeded=2, disabled_by_mode=1, residual_affine_in_variable=1, residual_function_of_variable=1, unsupported_method=1",
            rendered,
        )
        self.assertIn(
            "failure class by method: rational/budget_exceeded=1, table_reused/residual_affine_in_variable=1, table_reused/residual_function_of_variable=1, unsupported/budget_exceeded=1, unsupported/disabled_by_mode=1, unsupported/unsupported_method=1",
            rendered,
        )
        self.assertIn(
            "verification residuals: derivative_minus_integrand=2",
            rendered,
        )
        self.assertIn(
            "verification residual by method: table_reused/derivative_minus_integrand=2",
            rendered,
        )
        self.assertIn(
            "verification residual kinds: depends_on_variable=2",
            rendered,
        )
        self.assertIn(
            "verification residual kind by method: table_reused/depends_on_variable=2",
            rendered,
        )
        self.assertIn(
            "verification residual signatures: affine_in_variable=1, function_of_variable=1",
            rendered,
        )
        self.assertIn(
            "verification residual signature by method: table_reused/affine_in_variable=1, table_reused/function_of_variable=1",
            rendered,
        )
        self.assertIn(
            "publication statuses: accepted=10, rejected_no_antiderivative=3, rejected_residual_reason=3",
            rendered,
        )
        self.assertIn(
            "publication status by method: hermite/accepted=4, heurisch_probe/accepted=1, rational/accepted=4, rational/rejected_residual_reason=1, table_reused/accepted=1, table_reused/rejected_residual_reason=2, unsupported/rejected_no_antiderivative=3",
            rendered,
        )
        self.assertIn(
            "fallback statuses: blocked_by_candidate_policy=6, blocked_by_mode=9, eligible=1",
            rendered,
        )
        self.assertIn(
            "fallback status by method: hermite/blocked_by_mode=4, heurisch_probe/blocked_by_mode=1, rational/blocked_by_candidate_policy=1, rational/blocked_by_mode=3, rational/eligible=1, table_reused/blocked_by_candidate_policy=2, table_reused/blocked_by_mode=1, unsupported/blocked_by_candidate_policy=3",
            rendered,
        )
        self.assertIn(
            "trace levels: algorithmic_summary=13, diagnostic_only=3",
            rendered,
        )
        self.assertIn(
            "constant policies: arbitrary_constant_omitted=13, unspecified=3",
            rendered,
        )
        self.assertIn(
            "public trace levels: algorithmic_summary=10",
            rendered,
        )
        self.assertIn(
            "public constant policies: arbitrary_constant_omitted=10",
            rendered,
        )
        self.assertIn(
            "fallback trace levels: algorithmic_summary=1",
            rendered,
        )
        self.assertIn(
            "fallback constant policies: arbitrary_constant_omitted=1",
            rendered,
        )
        self.assertIn(
            "assumption exprs: total=0, public=0, fallback=0",
            rendered,
        )
        self.assertIn(
            "verification evidence: direct_differentiation=4, failed_differentiation=2, none=4, normalized_differentiation=6",
            rendered,
        )
        self.assertIn(
            "public verification evidence: direct_differentiation=4, normalized_differentiation=6",
            rendered,
        )
        self.assertIn(
            "fallback verification evidence: direct_differentiation=1",
            rendered,
        )
        self.assertIn(
            "verification evidence by method: hermite/normalized_differentiation=4, heurisch_probe/direct_differentiation=1, rational/direct_differentiation=2, rational/none=1, rational/normalized_differentiation=2, table_reused/direct_differentiation=1, table_reused/failed_differentiation=2, unsupported/none=3",
            rendered,
        )
        self.assertIn(
            "public verification evidence by method: hermite/normalized_differentiation=4, heurisch_probe/direct_differentiation=1, rational/direct_differentiation=2, rational/normalized_differentiation=2, table_reused/direct_differentiation=1",
            rendered,
        )
        self.assertIn(
            "fallback verification evidence by method: rational/direct_differentiation=1",
            rendered,
        )
        self.assertIn(
            "verification normalization reasons: numeric_scaled_quotient=2, power_one_elision=2, quotient_numeric_factor_cancellation=1, symbolic_scaled_quotient=1",
            rendered,
        )
        self.assertIn(
            "verification normalization passes: max=2; 0=10, 1=4, 2=2",
            rendered,
        )
        self.assertIn(
            "public verification normalization passes: max=2; 0=4, 1=4, 2=2",
            rendered,
        )
        self.assertIn(
            "fallback verification normalization passes: max=0; 0=1",
            rendered,
        )
        self.assertIn(
            "verification normalization passes by method: hermite/1=3, hermite/2=1, heurisch_probe/0=1, rational/0=3, rational/1=1, rational/2=1, table_reused/0=3, unsupported/0=3",
            rendered,
        )
        self.assertIn(
            "public verification normalization reasons: numeric_scaled_quotient=2, power_one_elision=2, quotient_numeric_factor_cancellation=1, symbolic_scaled_quotient=1",
            rendered,
        )
        self.assertIn(
            "verification normalization reason by method: hermite/numeric_scaled_quotient=1, hermite/power_one_elision=2, hermite/quotient_numeric_factor_cancellation=1, rational/numeric_scaled_quotient=1, rational/symbolic_scaled_quotient=1",
            rendered,
        )
        self.assertIn(
            "public verification normalization reason by method: hermite/numeric_scaled_quotient=1, hermite/power_one_elision=2, hermite/quotient_numeric_factor_cancellation=1, rational/numeric_scaled_quotient=1, rational/symbolic_scaled_quotient=1",
            rendered,
        )
        self.assertIn(
            "required conditions: nonzero=6",
            rendered,
        )

    def test_integrate_radical_inverse_sqrt_tables_are_shared_policy_not_candidate(
        self,
    ):
        metrics = MODULE.parse_calculus_integrate_command_matrix(
            """
{"status":"pass","total":8,"status_counts":{"pass":8,"slow":0,"fail":0,"timeout":0},"issue_kind_counts":{},"problem_case_count":0,"problem_cases":[],"supported_case_count":8,"residual_case_count":0,"residual_case_names":[],"warning_expected_case_count":0,"required_display_case_count":8,"step_checked_case_count":8,"supported_step_unchecked_case_count":0,"antiderivative_verification_case_count":8,"verified_supported_case_count":8,"direct_diff_integrate_case_count":7,"direct_diff_integrate_exact_case_count":7,"direct_diff_integrate_equivalence_case_count":0,"expected_step_substring_count":16,"distinct_required_display_count":3,"family_count":2,"radical_inverse_policy_cluster_counts":{"block8_inverse_sqrt_tables":8},"direct_diff_integrate_radical_inverse_policy_cluster_counts":{"block8_inverse_sqrt_tables":7}}
"""
        )

        self.assertEqual(
            metrics["integrate_radical_inverse_consolidated_policy_cluster_counts"],
            {"block8_inverse_sqrt_tables": 8},
        )
        self.assertEqual(
            metrics[
                "integrate_direct_diff_integrate_radical_inverse_shared_policy_cluster_counts"
            ],
            {"block8_inverse_sqrt_tables": 7},
        )
        self.assertNotIn(
            "integrate_radical_inverse_consolidation_candidate_counts",
            metrics,
        )

    def test_integrate_inverse_hyperbolic_rational_interval_is_shared_policy(
        self,
    ):
        metrics = MODULE.parse_calculus_integrate_command_matrix(
            """
{"status":"pass","total":2,"status_counts":{"pass":2,"slow":0,"fail":0,"timeout":0},"issue_kind_counts":{},"problem_case_count":0,"problem_cases":[],"supported_case_count":2,"residual_case_count":0,"residual_case_names":[],"warning_expected_case_count":0,"required_display_case_count":2,"step_checked_case_count":2,"supported_step_unchecked_case_count":0,"antiderivative_verification_case_count":2,"verified_supported_case_count":2,"direct_diff_integrate_case_count":2,"direct_diff_integrate_exact_case_count":2,"direct_diff_integrate_equivalence_case_count":0,"expected_step_substring_count":4,"distinct_required_display_count":2,"family_count":2,"radical_inverse_policy_cluster_counts":{"block8_inverse_hyperbolic_rational_interval":2},"direct_diff_integrate_radical_inverse_policy_cluster_counts":{"block8_inverse_hyperbolic_rational_interval":2}}
"""
        )

        self.assertEqual(
            metrics["integrate_radical_inverse_consolidated_policy_cluster_counts"],
            {"block8_inverse_hyperbolic_rational_interval": 2},
        )
        self.assertEqual(
            metrics[
                "integrate_direct_diff_integrate_radical_inverse_shared_policy_cluster_counts"
            ],
            {"block8_inverse_hyperbolic_rational_interval": 2},
        )
        self.assertNotIn(
            "integrate_radical_inverse_consolidation_candidate_counts",
            metrics,
        )

    def test_calculus_command_matrix_status_fails_on_unchecked_supported_steps(self):
        cases = (
            (
                "calculus_diff_command_matrix_smoke",
                "diff_supported_step_unchecked_case_count",
            ),
            (
                "calculus_limit_command_matrix_smoke",
                "limit_supported_step_unchecked_case_count",
            ),
            (
                "calculus_integrate_command_matrix_smoke",
                "integrate_supported_step_unchecked_case_count",
            ),
        )
        for suite_name, metric_key in cases:
            with self.subTest(suite_name=suite_name):
                metrics = {"matrix_status": "pass", metric_key: 1}
                self.assertEqual(
                    MODULE.suite_status(suite_name, metrics, 0),
                    "fail",
                )

    def test_calculus_command_matrix_status_allows_checked_supported_steps(self):
        cases = (
            (
                "calculus_diff_command_matrix_smoke",
                "diff_supported_step_unchecked_case_count",
            ),
            (
                "calculus_limit_command_matrix_smoke",
                "limit_supported_step_unchecked_case_count",
            ),
            (
                "calculus_integrate_command_matrix_smoke",
                "integrate_supported_step_unchecked_case_count",
            ),
        )
        for suite_name, metric_key in cases:
            with self.subTest(suite_name=suite_name):
                metrics = {"matrix_status": "pass", metric_key: 0}
                self.assertEqual(
                    MODULE.suite_status(suite_name, metrics, 0),
                    "pass",
                )

    def test_format_runtime_duration_preserves_subsecond_signal(self):
        self.assertEqual(MODULE.format_runtime_duration(0.00025), "0.25ms")
        self.assertEqual(MODULE.format_runtime_duration(0.15), "150.00ms")
        self.assertEqual(MODULE.format_runtime_duration(1.234), "1.23s")

    def test_elapsed_per_case_metric_preserves_reported_corpus_value(self):
        metrics = {
            "total_cases": 8,
            "reported_elapsed_per_case_ms": 2.1375,
        }

        MODULE.add_elapsed_per_case_metric(metrics, 99.0)

        self.assertEqual(metrics["reported_elapsed_per_case_ms"], 2.1375)

    def test_elapsed_per_case_metric_renders_for_calculus_matrix(self):
        metrics = {
            "matrix_status": "pass",
            "total_cases": 25,
            "passed": 25,
            "failed": 0,
            "diff_supported_case_count": 25,
        }
        MODULE.add_elapsed_per_case_metric(metrics, 0.5)
        scorecard = {
            "generated_at": "2026-06-05T00:00:00+00:00",
            "profile": "fast",
            "git": {"branch": "main", "commit": "abc123"},
            "suites": {
                "calculus_diff_command_matrix_smoke": {
                    "category": "calculus",
                    "status": "pass",
                    "elapsed_seconds": 0.5,
                    "metrics": metrics,
                    "delta": {},
                },
            },
        }

        rendered = MODULE.render_markdown(scorecard)

        self.assertEqual(metrics["reported_elapsed_per_case_ms"], 20.0)
        self.assertIn("avg_case=20.000ms", rendered)

    def test_calculus_matrix_runtime_hotspots_parse_and_render(self):
        metrics = MODULE.parse_calculus_diff_command_matrix(
            """
{"status":"pass","total":2,"status_counts":{"pass":2,"slow":0,"fail":0,"timeout":0},"issue_kind_counts":{},"problem_case_count":0,"problem_cases":[],"supported_case_count":2,"residual_case_count":0,"residual_case_names":[],"warning_expected_case_count":0,"required_display_case_count":1,"step_checked_case_count":2,"supported_step_unchecked_case_count":0,"expected_step_substring_count":4,"distinct_required_display_count":1,"family_count":2,"argument_regime_counts":{"variable":1,"affine_argument":1},"domain_regime_counts":{"unconditional":1,"required_condition":1},"outcome_counts":{"supported":2},"calculus_maturity_block_counts":{"block2_real_domain_differentiation":2},"calculus_block_gate_counts":{"didactic_trace_and_diff_policy":1,"domain_conditions_and_diff_policy":1},"trace_regime_counts":{"power_rule":1,"log_chain_rule":1},"presentation_regime_counts":{"canonical":1,"compact_quotient":1},"runtime_distribution":{"timed_case_count":2,"total_elapsed_seconds":0.143,"avg_case_ms":71.5,"p95_case_ms":123.0,"max_case_ms":123.0},"runtime_concentration":{"timed_case_count":2,"total_elapsed_seconds":0.143,"slowest_case":"log_affine_chain_required_domain","slowest_case_ms":123.0,"slowest_case_share_percent":86.0,"top_3_share_percent":100.0,"slowest_family":"log_affine_chain"},"warm_runtime_distribution":{"timed_case_count":1,"total_elapsed_seconds":0.123,"avg_case_ms":123.0,"p95_case_ms":123.0,"max_case_ms":123.0},"warm_runtime_concentration":{"timed_case_count":1,"total_elapsed_seconds":0.123,"slowest_case":"log_affine_chain_required_domain","slowest_case_ms":123.0,"slowest_case_share_percent":100.0,"top_3_share_percent":100.0,"slowest_family":"log_affine_chain"},"cold_start_case":{"name":"polynomial_power_direct","family":"polynomial","domain_regime":"unconditional","wall_elapsed_seconds":0.02,"status":"pass"},"slowest_cases":[{"name":"log_affine_chain_required_domain","family":"log_affine_chain","domain_regime":"required_condition","wall_elapsed_seconds":0.123,"status":"pass"}],"warm_slowest_cases":[{"name":"log_affine_chain_required_domain","family":"log_affine_chain","domain_regime":"required_condition","wall_elapsed_seconds":0.123,"status":"pass"}],"runtime_by_family":[{"axis":"family","group":"log_affine_chain","case_count":1,"total_elapsed_seconds":0.123,"avg_case_ms":123.0,"max_elapsed_seconds":0.123,"slowest_case":"log_affine_chain_required_domain"}],"warm_runtime_by_family":[{"axis":"family","group":"log_affine_chain","case_count":1,"total_elapsed_seconds":0.123,"avg_case_ms":123.0,"max_elapsed_seconds":0.123,"slowest_case":"log_affine_chain_required_domain"}],"runtime_by_calculus_maturity_block":[{"axis":"calculus_maturity_block","group":"block2_real_domain_differentiation","case_count":2,"total_elapsed_seconds":0.143,"avg_case_ms":71.5,"max_elapsed_seconds":0.123,"slowest_case":"log_affine_chain_required_domain"}],"warm_runtime_by_calculus_maturity_block":[{"axis":"calculus_maturity_block","group":"block2_real_domain_differentiation","case_count":1,"total_elapsed_seconds":0.123,"avg_case_ms":123.0,"max_elapsed_seconds":0.123,"slowest_case":"log_affine_chain_required_domain"}],"runtime_by_calculus_block_gate":[{"axis":"calculus_block_gate","group":"domain_conditions_and_diff_policy","case_count":1,"total_elapsed_seconds":0.123,"avg_case_ms":123.0,"max_elapsed_seconds":0.123,"slowest_case":"log_affine_chain_required_domain"}],"warm_runtime_by_calculus_block_gate":[{"axis":"calculus_block_gate","group":"domain_conditions_and_diff_policy","case_count":1,"total_elapsed_seconds":0.123,"avg_case_ms":123.0,"max_elapsed_seconds":0.123,"slowest_case":"log_affine_chain_required_domain"}],"runtime_by_positive_quadratic_policy_cluster":[{"axis":"positive_quadratic_policy_cluster","group":"block2_positive_quadratic_log_abs_pole_primitive","case_count":1,"total_elapsed_seconds":0.123,"avg_case_ms":123.0,"max_elapsed_seconds":0.123,"slowest_case":"log_affine_chain_required_domain"}],"warm_runtime_by_positive_quadratic_policy_cluster":[{"axis":"positive_quadratic_policy_cluster","group":"block2_positive_quadratic_log_abs_pole_primitive","case_count":1,"total_elapsed_seconds":0.123,"avg_case_ms":123.0,"max_elapsed_seconds":0.123,"slowest_case":"log_affine_chain_required_domain"}],"harness_check_runtime_distribution":{"timed_case_count":2,"total_elapsed_seconds":0.011,"avg_case_ms":5.5,"p95_case_ms":10.0,"max_case_ms":10.0},"slowest_process_evaluations":[{"name":"log_affine_chain_required_domain","family":"log_affine_chain","domain_regime":"required_condition","process_elapsed_seconds":0.123,"calculus_maturity_block":"block2_real_domain_differentiation","calculus_block_gate":"domain_conditions_and_diff_policy","positive_quadratic_policy_cluster":"block2_positive_quadratic_log_abs_pole_primitive"}],"slowest_harness_checks":[{"name":"polynomial_power_direct","family":"polynomial","domain_regime":"unconditional","harness_check_elapsed_seconds":0.010,"calculus_maturity_block":"block2_real_domain_differentiation","calculus_block_gate":"didactic_trace_and_diff_policy"}],"exact_square_runtime_pairs":[{"name":"inverse_hyperbolic_root_atanh_denominator_scale_exact_square","baseline_case":"inverse_hyperbolic_root_atanh_symbolic_denominator_scale_open_interval","exact_square_case":"inverse_hyperbolic_root_atanh_exact_square_denominator_scale_open_interval","baseline_case_ms":6.0,"exact_square_case_ms":96.0,"delta_ms":90.0,"ratio":16.0,"family":"inverse_hyperbolic_root"}],"largest_stdout_payload_cases":[{"name":"log_affine_chain_required_domain","family":"log_affine_chain","domain_regime":"required_condition","stdout_bytes":4096,"required_display_count":2,"expected_step_substring_count":3}],"largest_step_trace_cases":[{"name":"polynomial_power_direct","family":"polynomial","domain_regime":"unconditional","step_text_char_count":1200,"required_display_count":0,"expected_step_substring_count":1}]}
"""
        )
        scorecard = {
            "generated_at": "2026-06-05T00:00:00+00:00",
            "profile": "fast",
            "git": {"branch": "main", "commit": "abc123"},
            "suites": {
                "calculus_diff_command_matrix_smoke": {
                    "category": "calculus",
                    "status": "pass",
                    "elapsed_seconds": 0.2,
                    "metrics": metrics,
                    "delta": {},
                },
            },
        }

        rendered = MODULE.render_markdown(scorecard)

        self.assertEqual(
            metrics["diff_slowest_cases"][0]["name"],
            "log_affine_chain_required_domain",
        )
        self.assertEqual(metrics["diff_runtime_distribution"]["timed_case_count"], 2)
        self.assertEqual(metrics["diff_runtime_distribution"]["p95_case_ms"], 123.0)
        self.assertEqual(metrics["diff_runtime_pressure"]["status"], "watch")
        self.assertEqual(metrics["diff_runtime_pressure"]["primary_signal"], "p95")
        self.assertEqual(
            metrics["diff_runtime_measurement"]["status"],
            "actionable_pressure",
        )
        self.assertEqual(
            metrics["diff_runtime_pressure"]["reason"],
            "p95_case_ms=123.000",
        )
        self.assertEqual(metrics["diff_runtime_pressure"]["sample_basis"], "warm")
        self.assertEqual(
            metrics["diff_runtime_concentration"]["slowest_case"],
            "log_affine_chain_required_domain",
        )
        self.assertEqual(
            metrics["diff_runtime_concentration"]["slowest_case_share_percent"],
            86.0,
        )
        self.assertEqual(metrics["diff_warm_runtime_distribution"]["timed_case_count"], 1)
        self.assertEqual(
            metrics["diff_warm_runtime_concentration"]["top_3_share_percent"],
            100.0,
        )
        self.assertEqual(
            metrics["diff_slowest_process_evaluations"][0]["name"],
            "log_affine_chain_required_domain",
        )
        self.assertEqual(
            metrics["diff_slowest_process_evaluations"][0][
                "process_elapsed_seconds"
            ],
            0.123,
        )
        self.assertEqual(
            metrics["diff_exact_square_runtime_pairs"][0]["ratio"],
            16.0,
        )
        self.assertEqual(
            metrics["diff_harness_check_runtime_distribution"]["max_case_ms"],
            10.0,
        )
        self.assertEqual(
            metrics["diff_slowest_harness_checks"][0]["name"],
            "polynomial_power_direct",
        )
        self.assertEqual(
            metrics["diff_largest_stdout_payload_cases"][0]["stdout_bytes"],
            4096,
        )
        self.assertEqual(
            metrics["diff_largest_stdout_payload_cases"][0][
                "required_display_count"
            ],
            2,
        )
        self.assertEqual(
            metrics["diff_largest_step_trace_cases"][0]["step_text_char_count"],
            1200,
        )
        self.assertEqual(metrics["diff_cold_start_case"]["name"], "polynomial_power_direct")
        self.assertIn(
            "`diff_command_matrix` runtime distribution: "
            "timed_cases=2 avg=71.500ms p95=123.000ms max=123.000ms total=0.143s",
            rendered,
        )
        self.assertIn(
            "`diff_command_matrix` runtime concentration: "
            "slowest=log_affine_chain_required_domain slowest_ms=123.000 "
            "slowest_share=86.0% top3_share=100.0% total=0.143s "
            "family=log_affine_chain",
            rendered,
        )
        self.assertIn(
            "`diff_command_matrix` runtime pressure: "
            "status=watch primary=p95 reason=p95_case_ms=123.000 basis=warm "
            "timed_cases=1 p95=123.000ms max=123.000ms "
            "top3_share=100.0% slowest=log_affine_chain_required_domain "
            "family=log_affine_chain",
            rendered,
        )
        self.assertIn(
            "`diff_command_matrix` warm runtime distribution: "
            "timed_cases=1 avg=123.000ms p95=123.000ms max=123.000ms total=0.123s",
            rendered,
        )
        self.assertIn(
            "`diff_command_matrix` warm runtime concentration: "
            "slowest=log_affine_chain_required_domain slowest_ms=123.000 "
            "slowest_share=100.0% top3_share=100.0% total=0.123s "
            "family=log_affine_chain",
            rendered,
        )
        self.assertIn(
            "`diff_command_matrix` cold-start case: "
            "polynomial_power_direct=0.020s family=polynomial",
            rendered,
        )
        self.assertIn(
            "`diff_command_matrix` slowest process evaluations: "
            "log_affine_chain_required_domain=0.123s family=log_affine_chain",
            rendered,
        )
        self.assertIn(
            "`diff_command_matrix` exact-square runtime pairs: "
            "inverse_hyperbolic_root_atanh_denominator_scale_exact_square "
            "exact_square=96.000ms baseline=6.000ms delta=90.000ms "
            "ratio=16.000x "
            "pair=inverse_hyperbolic_root_atanh_exact_square_denominator_scale_open_interval/"
            "inverse_hyperbolic_root_atanh_symbolic_denominator_scale_open_interval "
            "family=inverse_hyperbolic_root",
            rendered,
        )
        self.assertIn(
            "`diff_command_matrix` harness check runtime distribution: "
            "timed_cases=2 avg=5.500ms p95=10.000ms max=10.000ms total=0.011s",
            rendered,
        )
        self.assertIn(
            "`diff_command_matrix` slowest harness checks: "
            "polynomial_power_direct=0.010s family=polynomial",
            rendered,
        )
        self.assertIn(
            "`diff_command_matrix` largest stdout payload cases: "
            "log_affine_chain_required_domain=4096B family=log_affine_chain "
            "required=2 expected_steps=3",
            rendered,
        )
        self.assertIn(
            "`diff_command_matrix` largest step trace cases: "
            "polynomial_power_direct=1200 chars family=polynomial "
            "required=0 expected_steps=1",
            rendered,
        )
        self.assertEqual(metrics["diff_runtime_by_family"][0]["avg_case_ms"], 123.0)
        self.assertEqual(metrics["diff_warm_runtime_by_family"][0]["avg_case_ms"], 123.0)
        self.assertEqual(
            metrics["diff_runtime_by_calculus_maturity_block"][0]["group"],
            "block2_real_domain_differentiation",
        )
        self.assertEqual(
            metrics["diff_runtime_by_calculus_block_gate"][0]["group"],
            "domain_conditions_and_diff_policy",
        )
        self.assertEqual(
            metrics["diff_runtime_by_positive_quadratic_policy_cluster"][0]["group"],
            "block2_positive_quadratic_log_abs_pole_primitive",
        )
        self.assertEqual(
            metrics["diff_warm_runtime_by_positive_quadratic_policy_cluster"][0][
                "avg_case_ms"
            ],
            123.0,
        )
        self.assertEqual(
            metrics["diff_slowest_process_evaluations"][0][
                "positive_quadratic_policy_cluster"
            ],
            "block2_positive_quadratic_log_abs_pole_primitive",
        )
        self.assertIn(
            "`diff_command_matrix` slowest cases: "
            "log_affine_chain_required_domain=0.123s family=log_affine_chain",
            rendered,
        )
        self.assertIn(
            "`diff_command_matrix` warm slowest cases: "
            "log_affine_chain_required_domain=0.123s family=log_affine_chain",
            rendered,
        )
        self.assertIn(
            "`diff_command_matrix` runtime by family: "
            "log_affine_chain total=0.123s avg=123.000ms cases=1",
            rendered,
        )
        self.assertIn(
            "`diff_command_matrix` warm runtime by family: "
            "log_affine_chain total=0.123s avg=123.000ms cases=1",
            rendered,
        )
        self.assertIn(
            "`diff_command_matrix` runtime by calculus maturity block: "
            "block2_real_domain_differentiation total=0.143s "
            "avg=71.500ms cases=2",
            rendered,
        )
        self.assertIn(
            "`diff_command_matrix` warm runtime by calculus block gate: "
            "domain_conditions_and_diff_policy total=0.123s "
            "avg=123.000ms cases=1",
            rendered,
        )
        self.assertIn(
            "`diff_command_matrix` runtime by positive quadratic policy cluster: "
            "block2_positive_quadratic_log_abs_pole_primitive total=0.123s "
            "avg=123.000ms cases=1",
            rendered,
        )
        self.assertIn(
            "`diff_command_matrix` runtime measurement: "
            "mode=process_per_case status=actionable_pressure pressure=watch "
            "p95=123.000ms max=123.000ms "
            "guidance=triage runtime pressure with focused probes",
            rendered,
        )
        self.assertIn(
            "cluster=block2_positive_quadratic_log_abs_pole_primitive",
            rendered,
        )

    def test_calculus_runtime_pressure_ignores_tiny_concentration(self):
        pressure = MODULE.classify_calculus_runtime_pressure(
            {
                "timed_case_count": 2,
                "avg_case_ms": 5.0,
                "p95_case_ms": 10.0,
                "max_case_ms": 10.0,
            },
            {
                "timed_case_count": 2,
                "top_3_share_percent": 100.0,
                "slowest_case": "tiny_matrix_hotspot",
            },
        )

        self.assertEqual(pressure["status"], "ok")
        self.assertEqual(pressure["primary_signal"], "none")
        self.assertEqual(pressure["sample_basis"], "cold_inclusive")

    def test_diff_positive_quadratic_runtime_candidates_require_material_runtime(self):
        metrics = MODULE.parse_calculus_diff_command_matrix(
            """
{"status":"pass","total":8,"status_counts":{"pass":8,"slow":0,"fail":0,"timeout":0},"issue_kind_counts":{},"problem_case_count":0,"problem_cases":[],"supported_case_count":8,"residual_case_count":0,"residual_case_names":[],"warning_expected_case_count":0,"required_display_case_count":0,"step_checked_case_count":8,"supported_step_unchecked_case_count":0,"expected_step_substring_count":8,"distinct_required_display_count":0,"family_count":3,"argument_regime_counts":{"quadratic":8},"domain_regime_counts":{"unconditional":8},"outcome_counts":{"supported":8},"calculus_maturity_block_counts":{"block2_real_domain_differentiation":8},"calculus_block_gate_counts":{"didactic_trace_and_diff_policy":8},"positive_quadratic_policy_cluster_counts":{"block2_positive_quadratic_log_abs_pole_primitive":3,"block2_positive_quadratic_log_arctan_primitive":3,"block2_symbolic_radius_arctan_positive_quadratic":2},"trace_regime_counts":{"chain_rule":8},"presentation_regime_counts":{"compact":8},"warm_runtime_by_positive_quadratic_policy_cluster":[{"axis":"positive_quadratic_policy_cluster","group":"block2_positive_quadratic_log_abs_pole_primitive","case_count":3,"total_elapsed_seconds":0.087,"avg_case_ms":29.0,"max_elapsed_seconds":0.032,"slowest_case":"positive_quadratic_log_abs_pole_scaled_quadratic_compact"},{"axis":"positive_quadratic_policy_cluster","group":"block2_symbolic_radius_arctan_positive_quadratic","case_count":2,"total_elapsed_seconds":0.053,"avg_case_ms":26.5,"max_elapsed_seconds":0.036,"slowest_case":"inverse_trig_symbolic_radius_non_unit_slope_arctan_primitive_compact"},{"axis":"positive_quadratic_policy_cluster","group":"block2_positive_quadratic_log_arctan_primitive","case_count":3,"total_elapsed_seconds":0.031,"avg_case_ms":10.333,"max_elapsed_seconds":0.012,"slowest_case":"positive_quadratic_log_arctan_surd_negative_orientation_compact"}]}
"""
        )

        self.assertNotIn(
            "diff_positive_quadratic_runtime_candidate_clusters",
            metrics,
        )

    def test_diff_positive_quadratic_runtime_candidates_parse_and_render(self):
        metrics = MODULE.parse_calculus_diff_command_matrix(
            """
{"status":"pass","total":8,"status_counts":{"pass":8,"slow":0,"fail":0,"timeout":0},"issue_kind_counts":{},"problem_case_count":0,"problem_cases":[],"supported_case_count":8,"residual_case_count":0,"residual_case_names":[],"warning_expected_case_count":0,"required_display_case_count":0,"step_checked_case_count":8,"supported_step_unchecked_case_count":0,"expected_step_substring_count":8,"distinct_required_display_count":0,"family_count":3,"argument_regime_counts":{"quadratic":8},"domain_regime_counts":{"unconditional":8},"outcome_counts":{"supported":8},"calculus_maturity_block_counts":{"block2_real_domain_differentiation":8},"calculus_block_gate_counts":{"didactic_trace_and_diff_policy":8},"positive_quadratic_policy_cluster_counts":{"block2_positive_quadratic_log_abs_pole_primitive":3,"block2_positive_quadratic_log_arctan_primitive":3,"block2_symbolic_radius_arctan_positive_quadratic":2},"trace_regime_counts":{"chain_rule":8},"presentation_regime_counts":{"compact":8},"warm_runtime_by_positive_quadratic_policy_cluster":[{"axis":"positive_quadratic_policy_cluster","group":"block2_positive_quadratic_log_abs_pole_primitive","case_count":3,"total_elapsed_seconds":0.36,"avg_case_ms":120.0,"max_elapsed_seconds":0.132,"slowest_case":"positive_quadratic_log_abs_pole_scaled_quadratic_compact"},{"axis":"positive_quadratic_policy_cluster","group":"block2_symbolic_radius_arctan_positive_quadratic","case_count":2,"total_elapsed_seconds":0.22,"avg_case_ms":110.0,"max_elapsed_seconds":0.116,"slowest_case":"inverse_trig_symbolic_radius_non_unit_slope_arctan_primitive_compact"},{"axis":"positive_quadratic_policy_cluster","group":"block2_positive_quadratic_log_arctan_primitive","case_count":3,"total_elapsed_seconds":0.031,"avg_case_ms":10.333,"max_elapsed_seconds":0.012,"slowest_case":"positive_quadratic_log_arctan_surd_negative_orientation_compact"}]}
"""
        )
        scorecard = {
            "generated_at": "2026-06-05T00:00:00+00:00",
            "profile": "fast",
            "git": {"branch": "main", "commit": "abc123"},
            "suites": {
                "calculus_diff_command_matrix_smoke": {
                    "category": "calculus",
                    "status": "pass",
                    "elapsed_seconds": 0.2,
                    "metrics": metrics,
                    "delta": {},
                },
            },
        }

        rendered = MODULE.render_markdown(scorecard)

        self.assertEqual(
            [
                row["group"]
                for row in metrics[
                    "diff_positive_quadratic_runtime_candidate_clusters"
                ]
            ],
            [
                "block2_positive_quadratic_log_abs_pole_primitive",
                "block2_symbolic_radius_arctan_positive_quadratic",
            ],
        )
        self.assertEqual(
            metrics["diff_positive_quadratic_runtime_candidate_clusters"][0][
                "avg_ratio"
            ],
            11.613,
        )
        self.assertIn(
            "`diff_command_matrix` positive-quadratic consolidated policy clusters: "
            "block2_positive_quadratic_log_abs_pole_primitive=3, "
            "block2_positive_quadratic_log_arctan_primitive=3, "
            "block2_symbolic_radius_arctan_positive_quadratic=2",
            rendered,
        )
        self.assertIn(
            "`diff_command_matrix` positive-quadratic runtime candidate clusters: "
            "block2_positive_quadratic_log_abs_pole_primitive avg=120.000ms "
            "ratio=11.613x cases=3",
            rendered,
        )

    def test_diff_variable_power_runtime_candidates_allow_single_hot_cluster(self):
        metrics = MODULE.parse_calculus_diff_command_matrix(
            """
{"status":"pass","total":3,"status_counts":{"pass":3,"slow":0,"fail":0,"timeout":0},"issue_kind_counts":{},"problem_case_count":0,"problem_cases":[],"supported_case_count":3,"residual_case_count":0,"residual_case_names":[],"warning_expected_case_count":0,"required_display_case_count":1,"step_checked_case_count":3,"supported_step_unchecked_case_count":0,"expected_step_substring_count":6,"distinct_required_display_count":1,"family_count":1,"argument_regime_counts":{"variable_power":3},"domain_regime_counts":{"unconditional":2,"disconnected_positive_base_required":1},"outcome_counts":{"supported":3},"calculus_maturity_block_counts":{"block2_real_domain_differentiation":3},"calculus_block_gate_counts":{"domain_conditions_and_diff_policy":3},"variable_power_policy_cluster_counts":{"block2_variable_power_logarithmic_derivative_domain":3},"trace_regime_counts":{"logarithmic_derivative":3},"presentation_regime_counts":{"canonical":2,"quadratic_base_fraction":1},"warm_runtime_by_variable_power_policy_cluster":[{"axis":"variable_power_policy_cluster","group":"block2_variable_power_logarithmic_derivative_domain","case_count":3,"total_elapsed_seconds":0.201,"avg_case_ms":67.0,"max_elapsed_seconds":0.182,"slowest_case":"quadratic_base_variable_power_disconnected_domain"}]}
"""
        )

        self.assertEqual(
            metrics["diff_variable_power_runtime_candidate_clusters"],
            [
                {
                    "group": "block2_variable_power_logarithmic_derivative_domain",
                    "case_count": 3,
                    "avg_case_ms": 67.0,
                    "avg_ratio": 1.0,
                    "total_elapsed_seconds": 0.201,
                    "max_elapsed_seconds": 0.182,
                    "slowest_case": (
                        "quadratic_base_variable_power_disconnected_domain"
                    ),
                }
            ],
        )

    def test_diff_cli_phase_runtime_parse_and_render(self):
        metrics = MODULE.parse_calculus_diff_command_matrix(
            """
{"status":"pass","total":3,"status_counts":{"pass":3,"slow":0,"fail":0,"timeout":0},"issue_kind_counts":{},"problem_case_count":0,"problem_cases":[],"supported_case_count":3,"residual_case_count":0,"residual_case_names":[],"warning_expected_case_count":0,"required_display_case_count":1,"step_checked_case_count":3,"supported_step_unchecked_case_count":0,"expected_step_substring_count":6,"distinct_required_display_count":1,"family_count":1,"argument_regime_counts":{"variable_power":3},"domain_regime_counts":{"unconditional":2,"disconnected_positive_base_required":1},"outcome_counts":{"supported":3},"calculus_maturity_block_counts":{"block2_real_domain_differentiation":3},"calculus_block_gate_counts":{"domain_conditions_and_diff_policy":3},"variable_power_policy_cluster_counts":{"block2_variable_power_logarithmic_derivative_domain":3},"trace_regime_counts":{"logarithmic_derivative":3},"presentation_regime_counts":{"canonical":2,"quadratic_base_fraction":1},"cli_simplify_runtime_distribution":{"timed_case_count":3,"total_elapsed_seconds":0.201,"avg_case_ms":67.0,"p95_case_ms":184.0,"max_case_ms":184.0},"cli_total_runtime_distribution":{"timed_case_count":3,"total_elapsed_seconds":0.205,"avg_case_ms":68.333,"p95_case_ms":185.0,"max_case_ms":185.0},"cli_public_overhead_runtime_distribution":{"timed_case_count":3,"total_elapsed_seconds":0.019,"avg_case_ms":6.333,"p95_case_ms":8.0,"max_case_ms":8.0},"slowest_cli_simplify_evaluations":[{"name":"quadratic_base_variable_power_disconnected_domain","family":"variable_power","domain_regime":"disconnected_positive_base_required","cli_simplify_elapsed_seconds":0.184,"calculus_maturity_block":"block2_real_domain_differentiation","calculus_block_gate":"domain_conditions_and_diff_policy","variable_power_policy_cluster":"block2_variable_power_logarithmic_derivative_domain","steps_count":3,"step_rule_names":["Calcular la derivada","Usar derivación logarítmica","Sumar fracciones"]}],"slowest_cli_total_evaluations":[{"name":"quadratic_base_variable_power_disconnected_domain","family":"variable_power","domain_regime":"disconnected_positive_base_required","cli_total_elapsed_seconds":0.185,"calculus_maturity_block":"block2_real_domain_differentiation","calculus_block_gate":"domain_conditions_and_diff_policy","variable_power_policy_cluster":"block2_variable_power_logarithmic_derivative_domain","steps_count":3,"step_rule_names":["Calcular la derivada","Usar derivación logarítmica","Sumar fracciones"]}],"slowest_cli_public_overhead_evaluations":[{"name":"quadratic_base_variable_power_disconnected_domain","family":"variable_power","domain_regime":"disconnected_positive_base_required","cli_public_overhead_seconds":0.008,"calculus_maturity_block":"block2_real_domain_differentiation","calculus_block_gate":"domain_conditions_and_diff_policy","variable_power_policy_cluster":"block2_variable_power_logarithmic_derivative_domain","steps_count":3,"step_rule_names":["Calcular la derivada","Usar derivación logarítmica","Sumar fracciones"]}]}
"""
        )
        scorecard = {
            "generated_at": "2026-06-05T00:00:00+00:00",
            "profile": "fast",
            "git": {"branch": "main", "commit": "abc123"},
            "suites": {
                "calculus_diff_command_matrix_smoke": {
                    "category": "calculus",
                    "status": "pass",
                    "elapsed_seconds": 0.2,
                    "metrics": metrics,
                    "delta": {},
                },
            },
        }

        rendered = MODULE.render_markdown(scorecard)

        self.assertEqual(
            metrics["diff_cli_simplify_runtime_distribution"]["max_case_ms"],
            184.0,
        )
        self.assertEqual(
            metrics["diff_slowest_cli_simplify_evaluations"][0][
                "variable_power_policy_cluster"
            ],
            "block2_variable_power_logarithmic_derivative_domain",
        )
        self.assertEqual(
            metrics["diff_slowest_cli_simplify_evaluations"][0]["steps_count"],
            3,
        )
        self.assertEqual(
            metrics["diff_slowest_cli_simplify_evaluations"][0]["step_rule_names"],
            [
                "Calcular la derivada",
                "Usar derivación logarítmica",
                "Sumar fracciones",
            ],
        )
        self.assertIn(
            "`diff_command_matrix` CLI simplify runtime distribution: "
            "timed_cases=3 avg=67.000ms p95=184.000ms max=184.000ms "
            "total=0.201s",
            rendered,
        )
        self.assertIn(
            "`diff_command_matrix` slowest CLI simplify evaluations: "
            "quadratic_base_variable_power_disconnected_domain=0.184s "
            "family=variable_power "
            "variable_power_cluster=block2_variable_power_logarithmic_derivative_domain "
            "steps=3 rules=Calcular la derivada|Usar derivación logarítmica|"
            "Sumar fracciones",
            rendered,
        )

    def test_calculus_runtime_pressure_prefers_warm_samples_when_available(self):
        metrics: dict[str, object] = {}

        MODULE.add_runtime_observability_metrics(
            metrics,
            {
                "runtime_distribution": {
                    "timed_case_count": 3,
                    "avg_case_ms": 84.0,
                    "p95_case_ms": 233.0,
                    "max_case_ms": 233.0,
                },
                "runtime_concentration": {
                    "timed_case_count": 3,
                    "slowest_case": "polynomial_power_direct",
                    "slowest_family": "polynomial",
                    "slowest_case_ms": 233.0,
                    "top_3_share_percent": 90.0,
                },
                "warm_runtime_distribution": {
                    "timed_case_count": 2,
                    "avg_case_ms": 12.0,
                    "p95_case_ms": 20.0,
                    "max_case_ms": 20.0,
                },
                "warm_runtime_concentration": {
                    "timed_case_count": 2,
                    "slowest_case": "log_affine_chain_required_domain",
                    "slowest_family": "log_affine_chain",
                    "slowest_case_ms": 20.0,
                    "top_3_share_percent": 40.0,
                },
                "cold_start_case": {
                    "name": "polynomial_power_direct",
                    "family": "polynomial",
                    "wall_elapsed_seconds": 0.233,
                    "status": "pass",
                },
            },
            prefix="diff",
            group_keys=(),
        )

        self.assertEqual(metrics["diff_runtime_pressure"]["status"], "ok")
        self.assertEqual(metrics["diff_runtime_pressure"]["primary_signal"], "none")
        self.assertEqual(metrics["diff_runtime_pressure"]["sample_basis"], "warm")
        self.assertEqual(
            metrics["diff_runtime_measurement"]["status"],
            "process_overhead_floor",
        )
        self.assertEqual(
            metrics["diff_runtime_measurement"]["guidance"],
            "require embedded or profiler evidence before treating this as an engine hotspot",
        )
        self.assertEqual(
            metrics["diff_cold_start_case"]["name"],
            "polynomial_power_direct",
        )
        self.assertEqual(metrics["diff_runtime_distribution"]["max_case_ms"], 233.0)
        self.assertEqual(metrics["diff_warm_runtime_distribution"]["max_case_ms"], 20.0)

    def test_calculus_runtime_guardrail_renders_cross_command_summary(self):
        scorecard = {
            "generated_at": "2026-06-05T00:00:00+00:00",
            "profile": "fast",
            "git": {"branch": "main", "commit": "abc123"},
            "suites": {
                "calculus_diff_command_matrix_smoke": {
                    "category": "calculus",
                    "status": "pass",
                    "elapsed_seconds": 0.2,
                    "metrics": {
                        "matrix_status": "pass",
                        "passed": 2,
                        "failed": 0,
                        "total_cases": 2,
                        "diff_supported_case_count": 2,
                        "diff_residual_case_count": 0,
                        "diff_warm_runtime_distribution": {
                            "timed_case_count": 2,
                            "avg_case_ms": 12.0,
                            "p95_case_ms": 20.0,
                            "max_case_ms": 20.0,
                            "total_elapsed_seconds": 0.024,
                        },
                        "diff_runtime_pressure": {
                            "status": "ok",
                            "primary_signal": "none",
                        },
                        "diff_runtime_measurement": {
                            "status": "process_overhead_floor",
                        },
                    },
                    "delta": {},
                },
                "calculus_limit_command_matrix_smoke": {
                    "category": "calculus",
                    "status": "pass",
                    "elapsed_seconds": 0.3,
                    "metrics": {
                        "matrix_status": "pass",
                        "passed": 3,
                        "failed": 0,
                        "total_cases": 3,
                        "limit_supported_case_count": 2,
                        "limit_residual_case_count": 1,
                        "limit_runtime_distribution": {
                            "timed_case_count": 3,
                            "avg_case_ms": 80.0,
                            "p95_case_ms": 120.0,
                            "max_case_ms": 150.0,
                            "total_elapsed_seconds": 0.24,
                        },
                        "limit_runtime_pressure": {
                            "status": "watch",
                            "primary_signal": "max",
                        },
                        "limit_runtime_measurement": {
                            "status": "actionable_pressure",
                        },
                    },
                    "delta": {},
                },
                "calculus_integrate_command_matrix_smoke": {
                    "category": "calculus",
                    "status": "pass",
                    "elapsed_seconds": 0.4,
                    "metrics": {
                        "matrix_status": "pass",
                        "passed": 4,
                        "failed": 0,
                        "total_cases": 4,
                        "integrate_supported_case_count": 3,
                        "integrate_residual_case_count": 1,
                        "integrate_warm_runtime_distribution": {
                            "timed_case_count": 4,
                            "avg_case_ms": 25.0,
                            "p95_case_ms": 30.0,
                            "max_case_ms": 35.0,
                            "total_elapsed_seconds": 0.1,
                        },
                        "integrate_runtime_pressure": {
                            "status": "ok",
                            "primary_signal": "none",
                        },
                        "integrate_runtime_measurement": {
                            "status": "process_overhead_floor",
                        },
                    },
                    "delta": {},
                },
                "calculus_residual_matrix_smoke": {
                    "category": "calculus",
                    "status": "pass",
                    "elapsed_seconds": 1.0,
                    "metrics": {
                        "matrix_status": "pass",
                        "passed": 5,
                        "failed": 0,
                        "timeouts": 0,
                        "total_cases": 5,
                    },
                    "delta": {},
                },
            },
        }

        rendered = MODULE.render_markdown(scorecard)

        self.assertIn(
            "- Calculus runtime guardrail: "
            "diff_command_matrix cases=2 avg=12.000ms p95=20.000ms "
            "max=20.000ms total=0.024s status=pass pressure=ok signal=none "
            "measurement=process_overhead_floor basis=warm; "
            "limit_command_matrix cases=3 avg=80.000ms p95=120.000ms "
            "max=150.000ms total=0.240s status=pass pressure=watch signal=max "
            "measurement=actionable_pressure basis=cold_inclusive; "
            "integrate_command_matrix cases=4 avg=25.000ms p95=30.000ms "
            "max=35.000ms total=0.100s status=pass pressure=ok signal=none "
            "measurement=process_overhead_floor basis=warm; "
            "residual_matrix cases=5 avg=200.000ms total=1.000s status=pass "
            "failed=0 timeouts=0 pressure=suite_elapsed_only "
            "measurement=suite_elapsed_per_case basis=suite_elapsed",
            rendered,
        )

    def test_calculus_runtime_pressure_flags_representative_concentration(self):
        pressure = MODULE.classify_calculus_runtime_pressure(
            {
                "timed_case_count": 20,
                "avg_case_ms": 5.0,
                "p95_case_ms": 10.0,
                "max_case_ms": 10.0,
            },
            {
                "timed_case_count": 20,
                "top_3_share_percent": 90.0,
                "slowest_case": "concentrated_hotspot",
                "slowest_family": "exp_log_chain",
            },
        )

        self.assertEqual(pressure["status"], "risk")
        self.assertEqual(pressure["primary_signal"], "top3")
        self.assertEqual(pressure["reason"], "top3_share_percent=90.0")

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
derive didactic audit timings: artifacts_seconds=41.250 cli_seconds=38.500 report_seconds=0.125 total_seconds=79.875 worker_count=4
derive didactic audit family hotspots: artifacts=trig_expand:12.500,simplify:9.250 cli=trig_expand:11.750,simplify:8.125
derive didactic audit case hotspots: artifacts=expand_trig_hot:1.500,log_hot:1.125 cli=expand_trig_hot:1.250,log_hot:1.000
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
        self.assertEqual(metrics["artifact_seconds"], 41.25)
        self.assertEqual(metrics["cli_seconds"], 38.5)
        self.assertEqual(metrics["report_seconds"], 0.125)
        self.assertEqual(metrics["reported_total_seconds"], 79.875)
        self.assertEqual(metrics["worker_count"], 4)
        self.assertEqual(
            metrics["artifact_family_hotspots"],
            "trig_expand:12.500,simplify:9.250",
        )
        self.assertEqual(
            metrics["cli_family_hotspots"],
            "trig_expand:11.750,simplify:8.125",
        )
        self.assertEqual(
            metrics["artifact_case_hotspots"],
            "expand_trig_hot:1.500,log_hot:1.125",
        )
        self.assertEqual(
            metrics["cli_case_hotspots"],
            "expand_trig_hot:1.250,log_hot:1.000",
        )

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

## 2026-05-18 - Discovery observe-only: closed by follow-up resolution wording

- area:
  - generated discovery
  - `combined_additive_zero` x `simplify`
- status:
  - `discovery/observe-only`
- follow-up resolution:
  - later retained coverage made this candidate obsolete

## 2026-05-18 - Discovery observe-only: superseded generated candidate

- area:
  - generated discovery
  - `combined_additive_zero` x `log_contract`
- status:
  - `superseded`

## 2026-05-18 - Discovery observe-only: resolved retained generated candidate

- area:
  - generated discovery
  - `combined_additive_zero` x `quotient_wrapper`
- status:
  - `resolved-retained`

## 2026-05-18 - Discovery observe-only: resolved by follow-up generated candidate

- area:
  - generated discovery
  - `combined_additive_zero` x `asinh_radius`
- status:
  - `resolved-by-follow-up`

## 2026-05-18 - Discovery observe-only: closed by retained follow-up wording

- area:
  - generated discovery
  - `combined_additive_zero` x `shifted_secant`
- status:
  - `discovery/observe-only`
- retained follow-up:
  - validated later; this entry should no longer be treated as an open observe-only candidate

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
        self.assertNotIn("quotient_wrapper", metrics["families"])
        self.assertNotIn("asinh_radius", metrics["families"])
        self.assertNotIn("shifted_secant", metrics["families"])
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
                        "artifact_seconds": 41.25,
                        "cli_seconds": 38.5,
                        "report_seconds": 0.125,
                        "reported_total_seconds": 79.875,
                        "worker_count": 4,
                        "artifact_family_hotspots": "trig_expand:12.500,simplify:9.250",
                        "cli_family_hotspots": "trig_expand:11.750,simplify:8.125",
                        "artifact_case_hotspots": "expand_trig_hot:1.500,log_hot:1.125",
                        "cli_case_hotspots": "expand_trig_hot:1.250,log_hot:1.000",
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
        self.assertIn("artifacts=41.25s cli=38.50s workers=4", markdown)
        self.assertIn(
            "Runtime family hotspots: artifacts=trig_expand:12.500,simplify:9.250",
            markdown,
        )
        self.assertIn(
            "Runtime case hotspots: artifacts=expand_trig_hot:1.500,log_hot:1.125",
            markdown,
        )
        self.assertIn("top_artifact_family=trig_expand:12.500", markdown)

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
                "calculus_diff_command_matrix_smoke": {
                    "status": "pass",
                    "elapsed_seconds": 0.3,
                    "metrics": {
                        "matrix_status": "pass",
                        "total_cases": 13,
                        "passed": 13,
                        "failed": 0,
                        "raw_failed": 0,
                        "slow": 0,
                        "timeouts": 0,
                        "problem_case_count": 0,
                        "problem_cases": [],
                        "issue_kind_counts": {},
                        "diff_supported_case_count": 12,
                        "diff_residual_case_count": 1,
                        "diff_residual_case_names": ["diff_residual_case"],
                        "diff_warning_expected_case_count": 0,
                        "diff_required_display_case_count": 9,
                        "diff_step_checked_case_count": 13,
                        "diff_supported_step_unchecked_case_count": 0,
                        "diff_expected_step_substring_count": 32,
                        "diff_distinct_required_display_count": 4,
                        "diff_required_display_counts": {
                            "x > 0": 4,
                            "cos(2·x + 1) ≠ 0": 1,
                        },
                        "diff_family_count": 10,
                        "diff_argument_regime_counts": {
                            "variable": 4,
                            "product": 2,
                            "polynomial_inner": 1,
                            "rational_expression": 1,
                            "nested_root": 1,
                            "scaled_nested_root": 1,
                            "nested_bounded_root": 1,
                            "negated_nested_root": 1,
                            "variable_power": 1,
                        },
                        "diff_domain_regime_counts": {
                            "unconditional": 3,
                            "required_condition": 8,
                            "interval_required": 1,
                            "discontinuous_residual": 1,
                        },
                        "diff_outcome_counts": {"supported": 12, "residual": 1},
                        "diff_calculus_maturity_block_counts": {
                            "block2_real_domain_differentiation": 12,
                            "block9_residuals_and_non_goals": 1,
                        },
                        "diff_calculus_block_gate_counts": {
                            "didactic_trace_and_diff_policy": 3,
                            "domain_conditions_and_diff_policy": 9,
                            "safe_residual_policy": 1,
                        },
                        "diff_trace_regime_counts": {
                            "chain_rule": 4,
                            "constant_multiple_chain_rule": 1,
                            "negative_argument_chain_rule": 1,
                            "power_rule": 1,
                            "product_rule": 1,
                            "residual_policy": 1,
                        },
                        "diff_presentation_regime_counts": {
                            "canonical": 2,
                            "factored": 2,
                            "residual": 1,
                            "scaled_post_calculus_compact": 1,
                            "signed_post_calculus_compact": 1,
                            "signed_reciprocal_root_interval": 1,
                        },
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
                "calculus_integrate_command_matrix_smoke": {
                    "status": "pass",
                    "elapsed_seconds": 0.35,
                    "metrics": {
                        "matrix_status": "pass",
                        "total_cases": 18,
                        "passed": 18,
                        "failed": 0,
                        "raw_failed": 0,
                        "slow": 0,
                        "timeouts": 0,
                        "problem_case_count": 0,
                        "problem_cases": [],
                        "issue_kind_counts": {},
                        "integrate_supported_case_count": 17,
                        "integrate_residual_case_count": 1,
                        "integrate_residual_case_names": ["integrate_residual_case"],
                        "integrate_warning_expected_case_count": 0,
                        "integrate_required_display_case_count": 9,
                        "integrate_step_checked_case_count": 18,
                        "integrate_supported_step_unchecked_case_count": 0,
                        "integrate_antiderivative_verification_case_count": 17,
                        "integrate_verified_supported_case_count": 17,
                        "integrate_direct_diff_integrate_case_count": 2,
                        "integrate_direct_diff_integrate_exact_case_count": 2,
                        "integrate_direct_diff_integrate_equivalence_case_count": 0,
                        "integrate_expected_step_substring_count": 46,
                        "integrate_distinct_required_display_count": 7,
                        "integrate_required_display_counts": {
                            "-1 < x < 1": 2,
                            "x > 0": 2,
                            "x ≠ -1/2": 1,
                        },
                        "integrate_family_count": 18,
                        "integrate_argument_regime_counts": {
                            "variable_power": 1,
                            "sum": 1,
                            "affine_argument": 1,
                            "affine_bounded_rational": 1,
                            "bounded_rational_expression": 1,
                            "bounded_variable_radical": 1,
                            "unbounded_variable_radical": 1,
                            "nonlinear_polynomial_derivative": 1,
                            "nonlinear_polynomial_base": 1,
                            "rational_expression": 2,
                            "affine_radical": 1,
                            "affine_hyperbolic": 1,
                            "product": 1,
                            "affine_product": 1,
                            "sqrt_chain": 2,
                            "unsupported_core": 1,
                        },
                        "integrate_domain_regime_counts": {
                            "unconditional": 9,
                            "nonzero_required": 1,
                            "radical_interval": 2,
                            "rational_interval": 2,
                            "positive_required": 2,
                            "sqrt_chain_nonzero_positive": 2,
                        },
                        "integrate_outcome_counts": {
                            "supported": 17,
                            "residual": 1,
                        },
                        "integrate_residual_cause_counts": {
                            "special_function_method_required": 1,
                        },
                        "integrate_residual_family_counts": {
                            "log_rational_residual": 1,
                        },
                        "integrate_residual_cause_family_counts": {
                            "special_function_method_required/log_rational_residual": 1,
                        },
                        "integrate_residual_cases_by_cause": {
                            "special_function_method_required": [
                                "integrate_residual_case",
                            ],
                        },
                        "integrate_verification_regime_counts": {
                            "residual_not_verified": 1,
                            "verified_by_diff": 15,
                            "verified_by_diff_and_direct_diff_integrate": 2,
                        },
                        "integrate_calculus_maturity_block_counts": {
                            "block4_base_integration": 2,
                            "block5_generalized_substitution": 3,
                            "block7_trig_hyperbolic_integration": 8,
                            "block8_radical_inverse_families": 4,
                            "block9_residuals_and_non_goals": 1,
                        },
                        "integrate_calculus_block_gate_counts": {
                            "didactic_trace_and_verified_antiderivative": 8,
                            "domain_conditions_and_verified_antiderivative": 9,
                            "safe_residual_policy": 1,
                        },
                        "integrate_trig_hyperbolic_policy_cluster_counts": {
                            "block7_explicit_reciprocal_trig_log_substitution": 7,
                            "block7_hyperbolic_reciprocal_square": 1,
                            "block7_sqrt_chain_reciprocal_trig_product": 7,
                        },
                        "integrate_trig_hyperbolic_consolidated_policy_cluster_counts": {
                            "block7_explicit_reciprocal_trig_log_substitution": 7,
                            "block7_hyperbolic_reciprocal_square": 1,
                            "block7_sqrt_chain_reciprocal_trig_product": 7,
                        },
                        "integrate_radical_inverse_policy_cluster_counts": {
                            "block8_inverse_trig_root_reciprocal": 6,
                        },
                        "integrate_radical_inverse_consolidated_policy_cluster_counts": {
                            "block8_inverse_trig_root_reciprocal": 6,
                        },
                        "integrate_base_integration_policy_cluster_counts": {
                            "block4_log_by_parts": 2,
                        },
                        "integrate_base_integration_consolidated_policy_cluster_counts": {
                            "block4_log_by_parts": 2,
                        },
                        "integrate_direct_diff_integrate_calculus_maturity_block_counts": {
                            "block7_trig_hyperbolic_integration": 1,
                            "block8_radical_inverse_families": 1,
                        },
                        "integrate_direct_diff_integrate_calculus_block_gate_counts": {
                            "didactic_trace_and_verified_antiderivative": 1,
                            "domain_conditions_and_verified_antiderivative": 1,
                        },
                        "integrate_direct_diff_integrate_trig_hyperbolic_policy_cluster_counts": {
                            "block7_hyperbolic_reciprocal_square": 1,
                        },
                        "integrate_direct_diff_integrate_trig_hyperbolic_shared_policy_cluster_counts": {
                            "block7_hyperbolic_reciprocal_square": 1,
                        },
                        "integrate_direct_diff_integrate_radical_inverse_policy_cluster_counts": {
                            "block8_inverse_sqrt_tables": 1,
                        },
                        "integrate_direct_diff_integrate_radical_inverse_shared_policy_cluster_counts": {
                            "block8_inverse_sqrt_tables": 1,
                        },
                        "integrate_direct_diff_integrate_gap_cases_by_calculus_maturity_block": {
                            "block7_trig_hyperbolic_integration": [
                                "gap_case_a",
                                "gap_case_b",
                                "gap_case_c",
                                "gap_case_d",
                            ],
                        },
                        "integrate_direct_diff_integrate_gap_cases_by_trig_hyperbolic_policy_cluster": {
                            "block7_sqrt_chain_reciprocal_trig_product": [
                                "gap_case_a",
                                "gap_case_b",
                                "gap_case_c",
                                "gap_case_d",
                            ],
                        },
                        "integrate_direct_diff_integrate_equivalence_calculus_maturity_block_counts": {},
                        "integrate_direct_diff_integrate_equivalence_base_integration_policy_cluster_counts": {},
                        "integrate_direct_diff_integrate_equivalence_trig_hyperbolic_policy_cluster_counts": {},
                        "integrate_trace_regime_counts": {
                            "by_parts_affine_log": 1,
                            "by_parts_log": 1,
                            "hyperbolic_substitution": 1,
                            "inverse_hyperbolic_rational_affine_table": 1,
                            "inverse_hyperbolic_rational_direct_table": 1,
                            "inverse_sqrt_affine_table": 1,
                            "inverse_sqrt_direct_table": 1,
                            "inverse_sqrt_hyperbolic_direct_table": 1,
                            "inverse_trig_table": 1,
                            "linear_substitution": 1,
                            "linearity": 1,
                            "log_reciprocal_derivative": 1,
                            "polynomial_base_substitution": 1,
                            "polynomial_derivative_substitution": 1,
                            "power_rule": 1,
                        },
                        "integrate_presentation_regime_counts": {
                            "abs_log": 1,
                            "abs_log_sqrt_chain": 1,
                            "affine_arcsin": 1,
                            "affine_atanh": 1,
                            "compact_power": 1,
                            "factored_by_parts": 2,
                            "exponential": 1,
                            "inverse_hyperbolic": 2,
                            "inverse_trig": 2,
                            "radical_power": 1,
                            "residual": 1,
                        },
                        "integrate_largest_stdout_payload_cases": [
                            {
                                "name": "large_integrate_output",
                                "family": "rational_partial_fraction",
                                "stdout_bytes": 8192,
                                "required_display_count": 3,
                                "expected_step_substring_count": 4,
                            }
                        ],
                        "integrate_largest_step_trace_cases": [
                            {
                                "name": "large_integrate_steps",
                                "family": "by_parts_affine_log",
                                "step_text_char_count": 4096,
                                "required_display_count": 1,
                                "expected_step_substring_count": 6,
                            }
                        ],
                        "integrate_runtime_by_residual_cause": [
                            {
                                "cause": "special_function_method_required",
                                "case_count": 1,
                                "total_elapsed_seconds": 0.42,
                                "avg_case_ms": 420.0,
                                "max_elapsed_seconds": 0.42,
                                "slowest_case": "integrate_residual_case",
                            }
                        ],
                        "integrate_runtime_by_residual_cause_family": [
                            {
                                "cause_family": (
                                    "special_function_method_required/"
                                    "log_rational_residual"
                                ),
                                "case_count": 1,
                                "total_elapsed_seconds": 0.42,
                                "avg_case_ms": 420.0,
                                "max_elapsed_seconds": 0.42,
                                "slowest_case": "integrate_residual_case",
                            }
                        ],
                        "integrate_residual_public_phase_by_cause": [
                            {
                                "cause": "special_function_method_required",
                                "case_count": 1,
                                "integrate_total_seconds": 0.42,
                                "cli_total_seconds": 0.31,
                                "public_overhead_total_seconds": 0.11,
                                "public_overhead_share_percent": 26.2,
                                "avg_required_display_count": 4.0,
                                "avg_step_text_char_count": 610.0,
                                "slowest_case": "integrate_residual_case",
                            }
                        ],
                        "integrate_residual_public_phase_by_cause_family": [
                            {
                                "cause_family": (
                                    "special_function_method_required/"
                                    "log_rational_residual"
                                ),
                                "case_count": 1,
                                "integrate_total_seconds": 0.42,
                                "cli_total_seconds": 0.31,
                                "public_overhead_total_seconds": 0.11,
                                "public_overhead_share_percent": 26.2,
                                "avg_required_display_count": 4.0,
                                "avg_step_text_char_count": 610.0,
                                "slowest_case": "integrate_residual_case",
                            }
                        ],
                        "integrate_residual_public_phase_slowest_cases": [
                            {
                                "name": "integrate_residual_case",
                                "integrate_elapsed_seconds": 0.42,
                                "cli_total_seconds": 0.31,
                                "cli_simplify_ms": 309.0,
                                "public_overhead_seconds": 0.11,
                                "public_overhead_share_percent": 26.2,
                                "required_display_count": 4,
                                "step_text_char_count": 610,
                                "stdout_bytes": 2048,
                                "residual_cause": "special_function_method_required",
                                "family": "explicit_reciprocal_trig_residual_domain",
                                "trace_regime": "residual_presimplification_with_domain",
                            }
                        ],
                        "integrate_residual_shape_orientation_probes": [
                            {
                                "name": "shifted_tangent_log_factored_residual",
                                "status": "pass",
                                "expression_shape": "factored_residual",
                                "orientation": "tan_minus_offset",
                                "steps_mode": "off",
                                "cli_simplify_us": 49000,
                                "cli_total_us": 51000,
                                "required_display_count": 4,
                            }
                        ],
                        "integrate_residual_shape_orientation_summary": {
                            "probe_count": 1,
                            "counted_probe_count": 1,
                            "max_required_display_count": 4,
                            "avg_required_display_count": 4.0,
                            "status_counts": {"pass": 1, "timeout": 1},
                            "max_name": "shifted_tangent_log_factored_residual",
                            "max_expression_shape": "factored_residual",
                            "max_orientation": "tan_minus_offset",
                            "max_steps_mode": "off",
                            "max_status": "pass",
                            "first_problem_name": "late_symbolic_probe",
                            "first_problem_status": "timeout",
                            "first_problem_orientation": "symbolic_offset_minus_tan",
                            "first_problem_steps_mode": "on",
                        },
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
                "calculus_limit_compact_contract": {
                    "status": "pass",
                    "elapsed_seconds": 0.05,
                    "metrics": {
                        "cargo_status": "ok",
                        "passed": 1,
                        "failed": 0,
                        "ignored": 0,
                        "measured": 0,
                        "filtered_out": 105,
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
                "calculus_limit_command_matrix_smoke": {
                    "status": "pass",
                    "elapsed_seconds": 0.4,
                    "metrics": {
                        "matrix_status": "pass",
                        "total_cases": 11,
                        "passed": 11,
                        "failed": 0,
                        "raw_failed": 0,
                        "slow": 0,
                        "timeouts": 0,
                        "problem_case_count": 0,
                        "problem_cases": [],
                        "issue_kind_counts": {},
                        "limit_supported_case_count": 8,
                        "limit_residual_case_count": 3,
                        "limit_residual_case_names": [
                            "limit_residual_a",
                            "limit_residual_b",
                            "limit_residual_c",
                        ],
                        "limit_warning_expected_case_count": 3,
                        "limit_required_display_case_count": 6,
                        "limit_step_checked_case_count": 11,
                        "limit_supported_step_unchecked_case_count": 0,
                        "limit_expected_step_substring_count": 11,
                        "limit_distinct_required_display_count": 6,
                        "limit_required_display_counts": {
                            "x > -3": 1,
                            "x ≠ -2": 1,
                            "x ≠ 1": 1,
                        },
                        "limit_family_count": 7,
                        "limit_point_regime_counts": {"finite": 6, "infinity": 5},
                        "limit_domain_regime_counts": {
                            "unconditional": 4,
                            "required_condition": 3,
                            "removable_hole": 1,
                            "endpoint_residual": 1,
                            "discontinuous_residual": 1,
                            "domain_path_conflict": 1,
                        },
                        "limit_required_condition_regime_counts": {
                            "finite_source_definedness": 1,
                            "infinity_path_conflict": 1,
                            "none": 9,
                        },
                        "limit_outcome_counts": {"supported": 8, "residual": 3},
                        "limit_residual_cause_counts": {
                            "finite_endpoint_or_boundary_policy": 2,
                            "infinity_domain_path_conflict": 1,
                        },
                        "limit_residual_family_counts": {
                            "endpoint_family": 2,
                            "infinity_family": 1,
                        },
                        "limit_residual_cause_family_counts": {
                            "finite_endpoint_or_boundary_policy/endpoint_family": 2,
                            "infinity_domain_path_conflict/infinity_family": 1,
                        },
                        "limit_residual_cases_by_cause": {
                            "finite_endpoint_or_boundary_policy": [
                                "limit_residual_a",
                                "limit_residual_b",
                            ],
                            "infinity_domain_path_conflict": ["limit_residual_c"],
                        },
                        "limit_calculus_maturity_block_counts": {
                            "block3_real_domain_limits": 8,
                            "block9_residuals_and_non_goals": 3,
                        },
                        "limit_calculus_block_gate_counts": {
                            "didactic_trace_and_limit_policy": 2,
                            "domain_conditions_and_limit_policy": 6,
                            "safe_residual_policy": 3,
                        },
                        "limit_trace_regime_counts": {
                            "finite_residual_policy": 2,
                            "substitution": 1,
                            "rational_degree_policy": 1,
                        },
                        "limit_presentation_regime_counts": {
                            "canonical": 7,
                            "infinity": 1,
                            "residual": 3,
                        },
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
                        "matrix_base_count": 26,
                        "matrix_wrapped_base_count": 21,
                        "matrix_standalone_base_count": 5,
                        "matrix_wrapper_count": 12,
                        "matrix_wrapped_case_count": 242,
                        "matrix_standalone_case_count": 5,
                        "matrix_expected_wrapped_case_count": 252,
                        "matrix_missing_wrapped_pair_count": 10,
                        "matrix_full_wrapper_base_count": 20,
                        "matrix_partial_wrapper_base_count": 1,
                        "matrix_largest_wrapper_gap_count": 10,
                        "matrix_wrapper_gap_examples": [
                            {
                                "base": "wide_gap",
                                "missing_count": 10,
                                "missing_wrappers": ["w1", "w2"],
                            }
                        ],
                        "expected_required_condition_case_count": 42,
                        "distinct_expected_required_conditions": 4,
                        "expected_required_condition_counts": {
                            "x + 1": 10,
                            "x + 2": 20,
                            "sin(x)": 12,
                            "x > 0": 24,
                        },
                    },
                    "delta": {},
                },
            },
        }

        markdown = MODULE.render_markdown(scorecard)

        self.assertIn("## Calculus Support Matrix Signal", markdown)
        self.assertIn("public calculus behavior", markdown)
        self.assertIn("support-matrix coverage", markdown)
        self.assertIn("Matrix axes: command, family, argument regime", markdown)
        self.assertNotIn("vertical slices", markdown)
        self.assertIn("`diff`: passed=30 failed=0", markdown)
        self.assertIn(
            "`diff_command_matrix`: passed=13 failed=0 total=13 slow=0 "
            "timeouts=0 supported_cases=12 residual_cases=1 warning_expected=0 "
            "required_display=9 step_checked=13 unchecked_supported_steps=0 "
            "families=10",
            markdown,
        )
        self.assertIn(
            "`diff_command_matrix` argument regimes: negated_nested_root=1, "
            "nested_bounded_root=1, nested_root=1, polynomial_inner=1, product=2, "
            "rational_expression=1, scaled_nested_root=1, variable=4",
            markdown,
        )
        self.assertIn(
            "`diff_command_matrix` outcomes: residual=1, supported=12",
            markdown,
        )
        self.assertIn(
            "`diff_command_matrix` residual case IDs: diff_residual_case",
            markdown,
        )
        self.assertIn(
            "`diff_command_matrix` calculus maturity blocks: "
            "block2_real_domain_differentiation=12, "
            "block9_residuals_and_non_goals=1",
            markdown,
        )
        self.assertIn(
            "`diff_command_matrix` calculus block gates: "
            "didactic_trace_and_diff_policy=3, "
            "domain_conditions_and_diff_policy=9, safe_residual_policy=1",
            markdown,
        )
        self.assertIn(
            "`diff_command_matrix` trace regimes: chain_rule=4, "
            "constant_multiple_chain_rule=1, negative_argument_chain_rule=1, "
            "power_rule=1, product_rule=1, residual_policy=1",
            markdown,
        )
        self.assertIn(
            "`diff_command_matrix` presentation regimes: canonical=2, "
            "factored=2, residual=1, scaled_post_calculus_compact=1, "
            "signed_post_calculus_compact=1, signed_reciprocal_root_interval=1",
            markdown,
        )
        self.assertIn(
            "`diff` ignored tests: `inverse_reciprocal_trig_diff_exhaustive` (debug-slow)",
            markdown,
        )
        self.assertIn(
            "`diff_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=256",
            markdown,
        )
        self.assertIn("`limit`: passed=6 failed=0", markdown)
        self.assertIn("`limit_compact`: passed=1 failed=0", markdown)
        self.assertIn("`limit_presimplify_safe`: passed=8 failed=0", markdown)
        self.assertIn(
            "`limit_command_matrix`: passed=11 failed=0 total=11 slow=0 "
            "timeouts=0 supported_cases=8 residual_cases=3 warning_expected=3 "
            "required_display=6 step_checked=11 unchecked_supported_steps=0 "
            "families=7",
            markdown,
        )
        self.assertIn("`limit_command_matrix` point regimes: finite=6, infinity=5", markdown)
        self.assertIn(
            "`limit_command_matrix` required condition regimes: "
            "finite_source_definedness=1, infinity_path_conflict=1, none=9",
            markdown,
        )
        self.assertIn(
            "`limit_command_matrix` required displays: "
            "x > -3=1, x ≠ -2=1, x ≠ 1=1",
            markdown,
        )
        self.assertIn(
            "`limit_command_matrix` outcomes: residual=3, supported=8",
            markdown,
        )
        self.assertIn(
            "`limit_command_matrix` residual case IDs: limit_residual_a, "
            "limit_residual_b, limit_residual_c",
            markdown,
        )
        self.assertIn(
            "`limit_command_matrix` residual causes: "
            "finite_endpoint_or_boundary_policy=2, "
            "infinity_domain_path_conflict=1",
            markdown,
        )
        self.assertIn(
            "`limit_command_matrix` residual families: "
            "endpoint_family=2, infinity_family=1",
            markdown,
        )
        self.assertIn(
            "`limit_command_matrix` residual cause-family buckets: "
            "finite_endpoint_or_boundary_policy/endpoint_family=2, "
            "infinity_domain_path_conflict/infinity_family=1",
            markdown,
        )
        self.assertIn(
            "`limit_command_matrix` residual examples by cause: "
            "finite_endpoint_or_boundary_policy: limit_residual_a, "
            "limit_residual_b; infinity_domain_path_conflict: "
            "limit_residual_c",
            markdown,
        )
        self.assertIn(
            "`limit_command_matrix` trace regimes: finite_residual_policy=2, "
            "rational_degree_policy=1, substitution=1",
            markdown,
        )
        self.assertIn(
            "`limit_command_matrix` presentation regimes: canonical=7, "
            "infinity=1, residual=3",
            markdown,
        )
        self.assertIn(
            "`residual_matrix`: passed=247 failed=0 total=247 slow=0 timeouts=0 "
            "total_bases=26 wrapped_bases=21 standalone_bases=5 wrappers=12 "
            "wrapped_cases=242 standalone_cases=5 "
            "conditioned_cases=42 distinct_conditions=4",
            markdown,
        )
        self.assertIn(
            "`residual_matrix` sparse expected conditions: x + 1=10, sin(x)=12, x + 2=20, x > 0=24",
            markdown,
        )
        self.assertIn(
            "`residual_matrix` domain expected conditions: x > 0=24",
            markdown,
        )
        self.assertIn(
            "`residual_matrix` wrapper coverage: expected_wrapped_cases=252 "
            "missing_wrapped_pairs=10 full_wrapper_bases=20 "
            "partial_wrapper_bases=1 largest_gap=10",
            markdown,
        )
        self.assertIn(
            "`residual_matrix` largest wrapper gaps: wide_gap missing=10",
            markdown,
        )
        self.assertNotIn("`residual_matrix` problem cases:", markdown)
        self.assertIn("`integrate`: passed=23 failed=0", markdown)
        self.assertIn(
            "`integrate_command_matrix`: passed=18 failed=0 total=18 slow=0 "
            "timeouts=0 supported_cases=17 residual_cases=1 warning_expected=0 "
            "required_display=9 step_checked=18 unchecked_supported_steps=0 "
            "antiderivative_verified=17 verified_supported=17 "
            "direct_diff_integrate=2 "
            "direct_exact=2 direct_equiv=0 "
            "families=18",
            markdown,
        )
        self.assertIn(
            "`integrate_command_matrix` argument regimes: affine_argument=1, "
            "affine_bounded_rational=1, affine_hyperbolic=1, "
            "affine_product=1, affine_radical=1, bounded_rational_expression=1, "
            "bounded_variable_radical=1, nonlinear_polynomial_base=1",
            markdown,
        )
        self.assertIn(
            "`integrate_command_matrix` domain regimes: nonzero_required=1, "
            "positive_required=2, radical_interval=2, rational_interval=2, "
            "sqrt_chain_nonzero_positive=2, unconditional=9",
            markdown,
        )
        self.assertIn(
            "`integrate_command_matrix` required displays: "
            "-1 < x < 1=2, x > 0=2, x ≠ -1/2=1",
            markdown,
        )
        self.assertIn(
            "`diff_command_matrix` required displays: "
            "cos(2·x + 1) ≠ 0=1, x > 0=4",
            markdown,
        )
        self.assertIn(
            "`integrate_command_matrix` outcomes: residual=1, supported=17",
            markdown,
        )
        self.assertIn(
            "`integrate_command_matrix` residual case IDs: integrate_residual_case",
            markdown,
        )
        self.assertIn(
            "`integrate_command_matrix` residual causes: "
            "special_function_method_required=1",
            markdown,
        )
        self.assertIn(
            "`integrate_command_matrix` residual families: "
            "log_rational_residual=1",
            markdown,
        )
        self.assertIn(
            "`integrate_command_matrix` residual cause-family buckets: "
            "special_function_method_required/log_rational_residual=1",
            markdown,
        )
        self.assertIn(
            "`integrate_command_matrix` residual examples by cause: "
            "special_function_method_required: integrate_residual_case",
            markdown,
        )
        self.assertIn(
            "`integrate_command_matrix` verification regimes: "
            "residual_not_verified=1, verified_by_diff=15, "
            "verified_by_diff_and_direct_diff_integrate=2",
            markdown,
        )
        self.assertIn(
            "`integrate_command_matrix` calculus maturity blocks: "
            "block4_base_integration=2, block5_generalized_substitution=3, "
            "block7_trig_hyperbolic_integration=8",
            markdown,
        )
        self.assertIn(
            "`integrate_command_matrix` calculus block gates: "
            "didactic_trace_and_verified_antiderivative=8, "
            "domain_conditions_and_verified_antiderivative=9, "
            "safe_residual_policy=1",
            markdown,
        )
        self.assertIn(
            "`integrate_command_matrix` trig/hyperbolic policy clusters: "
            "block7_explicit_reciprocal_trig_log_substitution=7, "
            "block7_hyperbolic_reciprocal_square=1, "
            "block7_sqrt_chain_reciprocal_trig_product=7",
            markdown,
        )
        self.assertIn(
            "`integrate_command_matrix` trig/hyperbolic consolidated policy clusters: "
            "block7_explicit_reciprocal_trig_log_substitution=7, "
            "block7_hyperbolic_reciprocal_square=1, "
            "block7_sqrt_chain_reciprocal_trig_product=7",
            markdown,
        )
        self.assertNotIn(
            "`integrate_command_matrix` trig/hyperbolic consolidation candidates:",
            markdown,
        )
        self.assertIn(
            "`integrate_command_matrix` base integration policy clusters: "
            "block4_log_by_parts=2",
            markdown,
        )
        self.assertIn(
            "`integrate_command_matrix` base integration consolidated policy clusters: "
            "block4_log_by_parts=2",
            markdown,
        )
        self.assertNotIn(
            "`integrate_command_matrix` base integration consolidation candidates:",
            markdown,
        )
        self.assertIn(
            "`integrate_command_matrix` direct diff(integrate) calculus maturity blocks: "
            "block7_trig_hyperbolic_integration=1, "
            "block8_radical_inverse_families=1",
            markdown,
        )
        self.assertIn(
            "`integrate_command_matrix` direct diff(integrate) calculus block gates: "
            "didactic_trace_and_verified_antiderivative=1, "
            "domain_conditions_and_verified_antiderivative=1",
            markdown,
        )
        self.assertIn(
            "`integrate_command_matrix` direct diff(integrate) trig/hyperbolic policy clusters:",
            markdown,
        )
        self.assertIn(
            "block7_hyperbolic_reciprocal_square=1",
            markdown,
        )
        self.assertIn(
            "`integrate_command_matrix` direct diff(integrate) trig/hyperbolic shared-policy clusters: "
            "block7_hyperbolic_reciprocal_square=1",
            markdown,
        )
        self.assertIn(
            "`integrate_command_matrix` direct diff(integrate) exactness: "
            "all 2 checks exact; no equivalence-backed fallback",
            markdown,
        )
        self.assertIn(
            "`integrate_command_matrix` direct diff(integrate) gap examples by calculus maturity block: "
            "block7_trig_hyperbolic_integration: gap_case_a, gap_case_b, gap_case_c, +1 more",
            markdown,
        )
        self.assertIn(
            "`integrate_command_matrix` direct diff(integrate) gap examples by trig/hyperbolic policy cluster: "
            "block7_sqrt_chain_reciprocal_trig_product: gap_case_a, gap_case_b, gap_case_c, +1 more",
            markdown,
        )
        self.assertNotIn(
            "`integrate_command_matrix` equivalence-backed diff(integrate) calculus maturity blocks:",
            markdown,
        )
        self.assertNotIn(
            "`integrate_command_matrix` equivalence-backed diff(integrate) base integration policy clusters:",
            markdown,
        )
        self.assertNotIn(
            "`integrate_command_matrix` equivalence-backed diff(integrate) trig/hyperbolic policy clusters:",
            markdown,
        )
        self.assertIn(
            "`integrate_command_matrix` radical/inverse policy clusters: "
            "block8_inverse_trig_root_reciprocal=6",
            markdown,
        )
        self.assertIn(
            "`integrate_command_matrix` radical/inverse consolidated policy clusters: "
            "block8_inverse_trig_root_reciprocal=6",
            markdown,
        )
        self.assertNotIn(
            "`integrate_command_matrix` radical/inverse consolidation candidates:",
            markdown,
        )
        self.assertIn(
            "`integrate_command_matrix` direct diff(integrate) radical/inverse policy clusters: "
            "block8_inverse_sqrt_tables=1",
            markdown,
        )
        self.assertIn(
            "`integrate_command_matrix` direct diff(integrate) radical/inverse shared-policy clusters: "
            "block8_inverse_sqrt_tables=1",
            markdown,
        )
        self.assertIn(
            "`integrate_command_matrix` trace regimes: by_parts_affine_log=1, "
            "by_parts_log=1, hyperbolic_substitution=1, "
            "inverse_hyperbolic_rational_affine_table=1, "
            "inverse_hyperbolic_rational_direct_table=1, "
            "inverse_sqrt_affine_table=1, inverse_sqrt_direct_table=1",
            markdown,
        )
        self.assertIn(
            "`integrate_command_matrix` presentation regimes: abs_log=1, "
            "abs_log_sqrt_chain=1, affine_arcsin=1, affine_atanh=1",
            markdown,
        )
        self.assertIn(
            "`integrate_command_matrix` largest stdout payload cases: "
            "large_integrate_output=8192B family=rational_partial_fraction "
            "required=3 expected_steps=4",
            markdown,
        )
        self.assertIn(
            "`integrate_command_matrix` largest step trace cases: "
            "large_integrate_steps=4096 chars family=by_parts_affine_log "
            "required=1 expected_steps=6",
            markdown,
        )
        self.assertIn(
            "`integrate_command_matrix` runtime by residual cause: "
            "special_function_method_required total=0.420s avg=420.000ms "
            "cases=1 slowest=integrate_residual_case",
            markdown,
        )
        self.assertIn(
            "`integrate_command_matrix` runtime by residual cause-family: "
            "special_function_method_required/log_rational_residual total=0.420s "
            "avg=420.000ms cases=1 slowest=integrate_residual_case",
            markdown,
        )
        self.assertIn(
            "`integrate_command_matrix` residual public phase by cause: "
            "special_function_method_required total=0.420s cli=0.310s "
            "overhead_share=26.2% cases=1 slowest=integrate_residual_case",
            markdown,
        )
        self.assertIn(
            "`integrate_command_matrix` residual public phase by cause-family: "
            "special_function_method_required/log_rational_residual total=0.420s "
            "cli=0.310s overhead_share=26.2% cases=1 "
            "slowest=integrate_residual_case",
            markdown,
        )
        self.assertIn(
            "`integrate_command_matrix` residual public phase slowest cases: "
            "integrate_residual_case total=0.420s cli_simplify=309.000ms "
            "overhead_share=26.2% cause=special_function_method_required",
            markdown,
        )
        self.assertIn(
            "`integrate_command_matrix` residual shape/orientation summary: "
            "probes=1 counted=1 max_required_display=4 "
            "avg_required_display=4.000 status_counts=pass=1,timeout=1 "
            "max_case=shifted_tangent_log_factored_residual "
            "shape=factored_residual orientation=tan_minus_offset steps=off "
            "first_problem=late_symbolic_probe:timeout "
            "orientation=symbolic_offset_minus_tan steps=on",
            markdown,
        )
        self.assertIn(
            "`integrate_command_matrix` residual shape/orientation probes: "
            "shifted_tangent_log_factored_residual status=pass "
            "shape=factored_residual orientation=tan_minus_offset steps=off "
            "required_display=4 cli_simplify=49.000ms cli_total=51.000ms",
            markdown,
        )
        self.assertIn(
            "`integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=321",
            markdown,
        )
        self.assertIn(
            "| `calculus_diff_contract` | `pass` | 0.10s | passed=30 failed=0 ignored=1 |",
            markdown,
        )
        self.assertIn(
            "| `calculus_diff_command_matrix_smoke` | `pass` | 0.30s | "
            "passed=13 failed=0 total=13 supported=12 residual=1 "
            "warning_expected=0 required_display=9 step_checked=13 "
            "unchecked_supported_steps=0 families=10 |",
            markdown,
        )
        self.assertIn(
            "| `calculus_residual_matrix_smoke` | `pass` | 3.20s | "
            "passed=247 failed=0 total=247 conditioned=42 conditions=4 "
            "total_bases=26 wrapped_bases=21 standalone_bases=5 wrappers=12 "
            "missing_wrapped_pairs=10 partial_wrapper_bases=1 |",
            markdown,
        )
        self.assertIn(
            "| `calculus_limit_command_matrix_smoke` | `pass` | 0.40s | "
            "passed=11 failed=0 total=11 supported=8 residual=3 "
            "warning_expected=3 required_display=6 step_checked=11 "
            "unchecked_supported_steps=0 families=7 |",
            markdown,
        )
        self.assertIn(
            "`limit_command_matrix` calculus maturity blocks: "
            "block3_real_domain_limits=8, block9_residuals_and_non_goals=3",
            markdown,
        )
        self.assertIn(
            "`limit_command_matrix` calculus block gates: "
            "didactic_trace_and_limit_policy=2, "
            "domain_conditions_and_limit_policy=6, safe_residual_policy=3",
            markdown,
        )
        self.assertIn(
            "| `calculus_integrate_command_matrix_smoke` | `pass` | 0.35s | "
            "passed=18 failed=0 total=18 supported=17 residual=1 "
            "warning_expected=0 required_display=9 step_checked=18 "
            "unchecked_supported_steps=0 antiderivative_verified=17 "
            "verified_supported=17 direct_diff_integrate=2 direct_exact=2 "
            "direct_equiv=0 families=18 |",
            markdown,
        )

    def test_render_markdown_does_not_truncate_calculus_matrix_regimes(self):
        scorecard = {
            "generated_at": "2026-04-20T00:00:00+00:00",
            "profile": "guardrail",
            "git": {"branch": "main", "commit": "abc123"},
            "generated_discovery": {},
            "suites": {
                "calculus_integrate_command_matrix_smoke": {
                    "status": "pass",
                    "elapsed_seconds": 0.35,
                    "metrics": {
                        "total_cases": 10,
                        "passed": 10,
                        "failed": 0,
                        "integrate_domain_regime_counts": {
                            "empty_real_domain": 1,
                            "explicit_denominator_source_condition": 1,
                            "explicit_hyperbolic_tangent_denominator_verified_substitution": 1,
                            "explicit_hyperbolic_tangent_presimplified_condition_dedupe": 1,
                            "explicit_tangent_denominator_source_condition": 1,
                            "explicit_tangent_denominator_verified_substitution": 1,
                            "linear_poles_required": 2,
                            "nonfinite_undefined": 1,
                            "nonzero_required": 3,
                            "positive_log_argument_required": 1,
                        },
                    },
                    "delta": {},
                },
            },
        }

        markdown = MODULE.render_markdown(scorecard)

        self.assertIn(
            "`integrate_command_matrix` domain regimes: empty_real_domain=1, "
            "explicit_denominator_source_condition=1",
            markdown,
        )
        self.assertIn("positive_log_argument_required=1", markdown)
        self.assertIn(
            "nonzero_required=3, positive_log_argument_required=1",
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
