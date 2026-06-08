import importlib.util
import stat
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "scripts" / "engine_diff_command_matrix_smoke.py"
SPEC = importlib.util.spec_from_file_location("engine_diff_command_matrix_smoke", MODULE_PATH)
SMOKE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = SMOKE
SPEC.loader.exec_module(SMOKE)


class DiffCommandMatrixSmokeTests(unittest.TestCase):
    def test_runtime_observability_summary_groups_hotspots(self) -> None:
        summary = SMOKE.runtime_observability_summary(
            [
                {
                    "name": "fast_case",
                    "family": "polynomial",
                    "domain_regime": "unconditional",
                    "wall_elapsed_seconds": 0.01,
                    "status": "pass",
                },
                {
                    "name": "slow_case",
                    "family": "log_chain",
                    "domain_regime": "required_condition",
                    "positive_quadratic_policy_cluster": "quadratic_cluster",
                    "variable_power_policy_cluster": "variable_power_cluster",
                    "wall_elapsed_seconds": 0.25,
                    "status": "pass",
                },
                {
                    "name": "medium_case",
                    "family": "log_chain",
                    "domain_regime": "required_condition",
                    "positive_quadratic_policy_cluster": "quadratic_cluster",
                    "variable_power_policy_cluster": "variable_power_cluster",
                    "wall_elapsed_seconds": 0.14,
                    "status": "pass",
                },
            ],
            group_keys=(
                "family",
                "domain_regime",
                "positive_quadratic_policy_cluster",
                "variable_power_policy_cluster",
            ),
        )

        self.assertEqual(summary["slowest_cases"][0]["name"], "slow_case")
        self.assertEqual(summary["runtime_concentration"]["slowest_case"], "slow_case")
        self.assertEqual(summary["runtime_concentration"]["slowest_case_ms"], 250.0)
        self.assertEqual(
            summary["runtime_concentration"]["slowest_case_share_percent"],
            62.5,
        )
        self.assertEqual(summary["runtime_concentration"]["top_3_share_percent"], 100.0)
        self.assertEqual(summary["cold_start_case"]["name"], "fast_case")
        self.assertEqual(summary["warm_slowest_cases"][0]["name"], "slow_case")
        self.assertEqual(summary["warm_runtime_distribution"]["timed_case_count"], 2)
        self.assertEqual(
            summary["warm_runtime_concentration"]["slowest_case_share_percent"],
            64.1,
        )
        self.assertEqual(
            summary["runtime_by_family"][0],
            {
                "axis": "family",
                "group": "log_chain",
                "case_count": 2,
                "total_elapsed_seconds": 0.39,
                "avg_case_ms": 195.0,
                "max_elapsed_seconds": 0.25,
                "slowest_case": "slow_case",
            },
        )
        self.assertEqual(
            summary["warm_runtime_by_family"][0],
            {
                "axis": "family",
                "group": "log_chain",
                "case_count": 2,
                "total_elapsed_seconds": 0.39,
                "avg_case_ms": 195.0,
                "max_elapsed_seconds": 0.25,
                "slowest_case": "slow_case",
            },
        )
        self.assertEqual(
            summary["runtime_by_positive_quadratic_policy_cluster"][0],
            {
                "axis": "positive_quadratic_policy_cluster",
                "group": "quadratic_cluster",
                "case_count": 2,
                "total_elapsed_seconds": 0.39,
                "avg_case_ms": 195.0,
                "max_elapsed_seconds": 0.25,
                "slowest_case": "slow_case",
            },
        )
        self.assertEqual(
            summary["runtime_by_variable_power_policy_cluster"][0],
            {
                "axis": "variable_power_policy_cluster",
                "group": "variable_power_cluster",
                "case_count": 2,
                "total_elapsed_seconds": 0.39,
                "avg_case_ms": 195.0,
                "max_elapsed_seconds": 0.25,
                "slowest_case": "slow_case",
            },
        )

    def test_phase_runtime_observability_summary_separates_process_and_harness(self) -> None:
        summary = SMOKE.phase_runtime_observability_summary(
            [
                {
                    "name": "engine_slow_case",
                    "family": "inverse_hyperbolic_root",
                    "domain_regime": "required_condition",
                    "process_elapsed_seconds": 0.18,
                    "harness_check_elapsed_seconds": 0.01,
                    "cli_parse_elapsed_seconds": 0.001,
                    "cli_simplify_elapsed_seconds": 0.17,
                    "cli_total_elapsed_seconds": 0.172,
                    "cli_public_overhead_seconds": 0.008,
                    "calculus_maturity_block": "block2_real_domain_differentiation",
                    "calculus_block_gate": "domain_conditions_and_diff_policy",
                    "positive_quadratic_policy_cluster": "quadratic_cluster",
                    "variable_power_policy_cluster": "variable_power_cluster",
                    "steps_count": 3,
                    "step_rule_names": [
                        "Calcular la derivada",
                        "Usar derivación logarítmica",
                        "Sumar fracciones",
                    ],
                },
                {
                    "name": "harness_slow_case",
                    "family": "log_over_sqrt",
                    "domain_regime": "required_condition",
                    "process_elapsed_seconds": 0.02,
                    "harness_check_elapsed_seconds": 0.04,
                    "cli_parse_elapsed_seconds": 0.002,
                    "cli_simplify_elapsed_seconds": 0.015,
                    "cli_total_elapsed_seconds": 0.018,
                    "cli_public_overhead_seconds": 0.002,
                    "calculus_maturity_block": "block2_real_domain_differentiation",
                    "calculus_block_gate": "didactic_trace_and_diff_policy",
                },
            ]
        )

        self.assertEqual(
            summary["slowest_process_evaluations"][0]["name"],
            "engine_slow_case",
        )
        self.assertEqual(
            summary["slowest_process_evaluations"][0]["process_elapsed_seconds"],
            0.18,
        )
        self.assertEqual(
            summary["slowest_process_evaluations"][0][
                "positive_quadratic_policy_cluster"
            ],
            "quadratic_cluster",
        )
        self.assertEqual(
            summary["slowest_process_evaluations"][0][
                "variable_power_policy_cluster"
            ],
            "variable_power_cluster",
        )
        self.assertEqual(
            summary["slowest_harness_checks"][0]["name"],
            "harness_slow_case",
        )
        self.assertEqual(
            summary["harness_check_runtime_distribution"]["max_case_ms"],
            40.0,
        )
        self.assertEqual(
            summary["cli_simplify_runtime_distribution"]["max_case_ms"],
            170.0,
        )
        self.assertEqual(
            summary["slowest_cli_simplify_evaluations"][0]["name"],
            "engine_slow_case",
        )
        self.assertEqual(
            summary["slowest_cli_simplify_evaluations"][0][
                "variable_power_policy_cluster"
            ],
            "variable_power_cluster",
        )
        self.assertEqual(
            summary["slowest_cli_simplify_evaluations"][0]["steps_count"],
            3,
        )
        self.assertEqual(
            summary["slowest_cli_simplify_evaluations"][0]["step_rule_names"],
            [
                "Calcular la derivada",
                "Usar derivación logarítmica",
                "Sumar fracciones",
            ],
        )
        self.assertEqual(
            summary["cli_public_overhead_runtime_distribution"]["max_case_ms"],
            8.0,
        )

    def test_exact_square_runtime_pair_rows_compare_sibling_cases(self) -> None:
        rows = SMOKE.exact_square_runtime_pair_rows(
            [
                {
                    "name": "inverse_hyperbolic_root_atanh_symbolic_denominator_scale_open_interval",
                    "family": "inverse_hyperbolic_root",
                    "argument_regime": "symbolic_denominator_scaled_shifted_root",
                    "presentation_regime": "symbolic_denominator_open_interval_compact",
                    "wall_elapsed_seconds": 0.006,
                },
                {
                    "name": "inverse_hyperbolic_root_atanh_exact_square_denominator_scale_open_interval",
                    "family": "inverse_hyperbolic_root",
                    "argument_regime": "exact_square_symbolic_denominator_scaled_shifted_root",
                    "presentation_regime": "exact_square_symbolic_denominator_open_interval_compact",
                    "wall_elapsed_seconds": 0.096,
                },
            ]
        )

        self.assertEqual(
            rows,
            [
                {
                    "name": "inverse_hyperbolic_root_atanh_denominator_scale_exact_square",
                    "baseline_case": "inverse_hyperbolic_root_atanh_symbolic_denominator_scale_open_interval",
                    "exact_square_case": "inverse_hyperbolic_root_atanh_exact_square_denominator_scale_open_interval",
                    "baseline_case_ms": 6.0,
                    "exact_square_case_ms": 96.0,
                    "delta_ms": 90.0,
                    "ratio": 16.0,
                    "baseline_argument_regime": "symbolic_denominator_scaled_shifted_root",
                    "baseline_presentation_regime": "symbolic_denominator_open_interval_compact",
                    "exact_square_argument_regime": "exact_square_symbolic_denominator_scaled_shifted_root",
                    "exact_square_presentation_regime": "exact_square_symbolic_denominator_open_interval_compact",
                    "family": "inverse_hyperbolic_root",
                },
            ],
        )

    def test_payload_observability_summary_reports_output_and_step_hotspots(self) -> None:
        summary = SMOKE.payload_observability_summary(
            [
                {
                    "name": "large_stdout_case",
                    "family": "inverse_trig_root",
                    "stdout_bytes": 4096,
                    "step_text_char_count": 900,
                    "required_display": ["a ≠ 0", "x > -1"],
                    "expected_step_substrings": ["u =", "du ="],
                    "calculus_maturity_block": "block2_real_domain_differentiation",
                    "calculus_block_gate": "domain_conditions_and_diff_policy",
                },
                {
                    "name": "large_steps_case",
                    "family": "inverse_hyperbolic_root",
                    "stdout_bytes": 1024,
                    "step_text_char_count": 1200,
                    "required_display": ["x > -1"],
                    "expected_step_substrings": ["chain"],
                    "calculus_maturity_block": "block2_real_domain_differentiation",
                    "calculus_block_gate": "didactic_trace_and_diff_policy",
                },
            ]
        )

        self.assertEqual(
            summary["largest_stdout_payload_cases"][0]["name"],
            "large_stdout_case",
        )
        self.assertEqual(
            summary["largest_stdout_payload_cases"][0]["required_display_count"],
            2,
        )
        self.assertEqual(
            summary["largest_step_trace_cases"][0]["name"],
            "large_steps_case",
        )
        self.assertEqual(
            summary["largest_step_trace_cases"][0]["step_text_char_count"],
            1200,
        )

    def test_default_matrix_covers_diff_policy_axes(self) -> None:
        cases = SMOKE.build_cases()

        self.assertEqual(len(cases), 80)
        names = {case.name for case in cases}
        self.assertIn(
            "log_quadratic_empty_positive_argument_domain_undefined",
            names,
        )
        self.assertIn(
            "log_abs_negative_scale_empty_positive_argument_domain_undefined",
            names,
        )
        self.assertIn("general_base_log_unit_base_domain_undefined", names)
        self.assertIn("polynomial_inner_chain_power", names)
        self.assertIn("elementary_exp_affine_chain_trace", names)
        self.assertIn("elementary_trig_affine_chain_trace", names)
        self.assertIn("elementary_trig_tan_affine_chain_required_condition", names)
        self.assertIn(
            "inverse_hyperbolic_acosh_named_positive_offset_satisfied_lower_bound",
            names,
        )
        self.assertIn(
            "elementary_reciprocal_trig_csc_affine_chain_sine_pole",
            names,
        )
        self.assertIn("elementary_hyperbolic_sinh_affine_chain_trace", names)
        self.assertIn(
            "elementary_hyperbolic_tanh_affine_chain_reciprocal_square",
            names,
        )
        self.assertIn("log_tangent_positive_source_domain", names)
        self.assertIn("log_cotangent_positive_source_domain", names)
        self.assertIn("sqrt_tangent_positive_dominates_nonnegative_domain", names)
        self.assertIn("product_log_root_domain_dedupe_compact", names)
        self.assertIn("log_over_sqrt_root_denominator_compact", names)
        self.assertIn("log_over_sqrt_denominator_scale_compact", names)
        self.assertIn("log_over_sqrt_symbolic_denominator_scale_compact", names)
        self.assertIn("log_affine_chain_required_domain", names)
        self.assertIn(
            "log_affine_chain_negative_orientation_required_domain",
            names,
        )
        self.assertIn("constant_base_exp_affine_chain_positive_base", names)
        self.assertIn("constant_base_exp_affine_chain_negative_base_undefined", names)
        self.assertIn(
            "constant_base_exp_affine_chain_zero_base_positive_exponent_domain",
            names,
        )
        self.assertIn(
            "constant_base_exp_quadratic_zero_base_empty_domain_undefined",
            names,
        )
        self.assertIn("nonfinite_constant_derivative_undefined", names)
        self.assertIn("quotient_root_over_log_domain_pole_compact", names)
        self.assertIn(
            "quotient_root_over_log_symbolic_denominator_scale_compact",
            names,
        )
        self.assertIn(
            "sqrt_quadratic_empty_positive_argument_domain_undefined",
            names,
        )
        self.assertIn("inverse_trig_root_compact_presentation", names)
        self.assertIn("inverse_trig_root_constant_multiple_trace", names)
        self.assertIn("inverse_trig_root_symbolic_denominator_scale", names)
        self.assertIn("inverse_trig_root_external_symbolic_denominator_scale", names)
        self.assertIn("inverse_trig_root_symbolic_rational_denominator_scale", names)
        self.assertIn(
            "inverse_trig_root_symbolic_rational_denominator_affine_radicand_shortcut",
            names,
        )
        self.assertIn(
            "inverse_trig_root_symbolic_rational_denominator_affine_radicand_external_scale_shortcut",
            names,
        )
        self.assertIn(
            "inverse_trig_root_exact_square_symbolic_denominator_scale_condition_dedupe",
            names,
        )
        self.assertIn(
            "inverse_trig_root_symbolic_numerator_scale_positive_gap",
            names,
        )
        self.assertIn("inverse_trig_root_symbolic_denominator_internal_scale", names)
        self.assertIn(
            "inverse_trig_root_symbolic_denominator_scale_dual_orientation",
            names,
        )
        self.assertIn("inverse_trig_root_interval_orientation", names)
        self.assertIn("inverse_trig_root_empty_open_interval_undefined", names)
        self.assertIn(
            "inverse_trig_shifted_quadratic_empty_open_interval_undefined",
            names,
        )
        self.assertIn(
            "inverse_trig_arcsin_point_domain_empty_derivative_undefined",
            names,
        )
        self.assertIn(
            "inverse_trig_symbolic_constant_empty_open_interval_undefined",
            names,
        )
        self.assertIn(
            "inverse_trig_named_constant_positive_offset_empty_open_interval_undefined",
            names,
        )
        self.assertIn("inverse_hyperbolic_atanh_empty_open_interval_undefined", names)
        self.assertIn(
            "inverse_hyperbolic_atanh_named_constant_positive_offset_empty_open_interval_undefined",
            names,
        )
        self.assertIn("inverse_hyperbolic_acosh_empty_lower_bound_undefined", names)
        self.assertIn(
            "inverse_hyperbolic_acosh_empty_derivative_domain_undefined",
            names,
        )
        self.assertIn(
            "inverse_hyperbolic_root_atanh_symbolic_numerator_scale_open_interval",
            names,
        )
        self.assertIn(
            "inverse_hyperbolic_root_atanh_symbolic_denominator_scale_open_interval",
            names,
        )
        self.assertIn(
            "inverse_hyperbolic_root_atanh_exact_square_denominator_scale_open_interval",
            names,
        )
        self.assertIn(
            "inverse_hyperbolic_root_symbolic_numerator_scale_positive_gap",
            names,
        )
        self.assertIn(
            "inverse_hyperbolic_root_asinh_symbolic_denominator_scale_positive_gap",
            names,
        )
        self.assertIn("inverse_trig_root_negative_argument", names)
        self.assertIn(
            "sqrt_chain_trig_log_presimplified_condition_dedupe",
            names,
        )
        self.assertIn(
            "positive_quadratic_log_arctan_polynomial_primitive_compact",
            names,
        )
        self.assertIn(
            "positive_quadratic_log_arctan_surd_primitive_compact",
            names,
        )
        self.assertIn(
            "positive_quadratic_log_arctan_surd_negative_orientation_compact",
            names,
        )
        self.assertIn(
            "inverse_trig_symbolic_radius_shifted_arctan_primitive_no_depth_overflow",
            names,
        )
        self.assertIn(
            "inverse_trig_symbolic_radius_non_unit_slope_arctan_primitive_compact",
            names,
        )
        self.assertIn("log_ratio_single_pole_scaled_shifted_linear_compact", names)
        self.assertIn(
            "log_ratio_single_pole_positive_scaled_abs_argument_compact",
            names,
        )
        self.assertIn("separated_log_abs_linear_pole_raw_preserved", names)
        self.assertIn("positive_quadratic_log_abs_pole_scaled_quadratic_compact", names)
        self.assertIn("positive_quadratic_log_abs_pole_scaled_linear_pole_compact", names)
        self.assertIn(
            "positive_quadratic_log_abs_pole_positive_orientation_combined_source_compact",
            names,
        )
        self.assertIn(
            "positive_quadratic_log_abs_pole_linear_wrapper_compact",
            names,
        )
        self.assertIn("abs_quadratic_factored_nondifferentiable_domain", names)
        self.assertIn("discontinuous_sign_polynomial_nonzero_domain", names)
        self.assertIn("discontinuous_sign_quadratic_factored_domain", names)
        self.assertIn("shifted_linear_base_variable_power_log_domain", names)
        self.assertIn("quadratic_base_variable_power_disconnected_domain", names)
        self.assertEqual(
            SMOKE.count_by(cases, "outcome"),
            {"supported": 64, "undefined": 16},
        )
        self.assertEqual(
            SMOKE.count_calculus_maturity_blocks(cases),
            {
                "block2_real_domain_differentiation": 64,
                "block9_residuals_and_non_goals": 16,
            },
        )
        self.assertEqual(
            SMOKE.count_calculus_block_gates(cases),
            {
                "didactic_trace_and_diff_policy": 11,
                "domain_conditions_and_diff_policy": 53,
                "explicit_undefined_domain_policy": 16,
            },
        )
        self.assertEqual(
            SMOKE.count_symbolic_radius_policy_clusters(cases),
            {"block2_symbolic_radius_arctan_positive_quadratic": 2},
        )
        self.assertEqual(
            SMOKE.count_positive_quadratic_policy_clusters(cases),
            {
                "block2_positive_quadratic_log_abs_pole_primitive": 4,
                "block2_positive_quadratic_log_arctan_primitive": 3,
                "block2_symbolic_radius_arctan_positive_quadratic": 2,
            },
        )
        self.assertEqual(
            SMOKE.count_variable_power_policy_clusters(cases),
            {"block2_variable_power_logarithmic_derivative_domain": 3},
        )
        step_checked = {
            case.name
            for case in cases
            if case.expected_step_substrings
        }
        supported_step_unchecked = {
            case.name
            for case in cases
            if case.outcome == "supported" and not case.expected_step_substrings
        }
        self.assertEqual(
            step_checked,
            {
                "product_log_required_condition",
                "product_log_root_domain_dedupe_compact",
                "log_over_sqrt_root_denominator_compact",
                "log_over_sqrt_denominator_scale_compact",
                "log_over_sqrt_symbolic_denominator_scale_compact",
                "log_quadratic_empty_positive_argument_domain_undefined",
                "log_abs_negative_scale_empty_positive_argument_domain_undefined",
                "general_base_log_unit_base_domain_undefined",
                "polynomial_power_direct",
                "polynomial_inner_chain_power",
                "elementary_exp_affine_chain_trace",
                "elementary_trig_affine_chain_trace",
                "elementary_trig_tan_affine_chain_required_condition",
                "elementary_reciprocal_trig_csc_affine_chain_sine_pole",
                "elementary_hyperbolic_sinh_affine_chain_trace",
                "elementary_hyperbolic_tanh_affine_chain_reciprocal_square",
                "log_tangent_positive_source_domain",
                "log_cotangent_positive_source_domain",
                "log_affine_chain_required_domain",
                "log_affine_chain_negative_orientation_required_domain",
                "constant_base_exp_affine_chain_positive_base",
                "constant_base_exp_affine_chain_negative_base_undefined",
                "constant_base_exp_affine_chain_zero_base_positive_exponent_domain",
                "constant_base_exp_quadratic_zero_base_empty_domain_undefined",
                "nonfinite_constant_derivative_undefined",
                "rational_quotient_required_condition",
                "quotient_root_over_log_domain_pole_compact",
                "quotient_root_over_log_symbolic_denominator_scale_compact",
                "positive_quadratic_log_arctan_polynomial_primitive_compact",
                "positive_quadratic_log_arctan_surd_primitive_compact",
                "positive_quadratic_log_arctan_surd_negative_orientation_compact",
                "inverse_trig_symbolic_radius_shifted_arctan_primitive_no_depth_overflow",
                "inverse_trig_symbolic_radius_non_unit_slope_arctan_primitive_compact",
                "log_ratio_single_pole_scaled_shifted_linear_compact",
                "log_ratio_single_pole_positive_scaled_abs_argument_compact",
                "separated_log_abs_linear_pole_raw_preserved",
                "positive_quadratic_log_abs_pole_scaled_quadratic_compact",
                "positive_quadratic_log_abs_pole_scaled_linear_pole_compact",
                "positive_quadratic_log_abs_pole_positive_orientation_combined_source_compact",
                "positive_quadratic_log_abs_pole_linear_wrapper_compact",
                "sqrt_variable_open_domain",
                "sqrt_tangent_positive_dominates_nonnegative_domain",
                "sqrt_quadratic_empty_positive_argument_domain_undefined",
                "inverse_trig_root_compact_presentation",
                "inverse_trig_root_constant_multiple_trace",
                "inverse_trig_root_symbolic_denominator_scale",
                "inverse_trig_root_external_symbolic_denominator_scale",
                "inverse_trig_root_symbolic_rational_denominator_scale",
                "inverse_trig_root_symbolic_rational_denominator_affine_radicand_shortcut",
                "inverse_trig_root_symbolic_rational_denominator_affine_radicand_external_scale_shortcut",
                "inverse_trig_root_exact_square_symbolic_denominator_scale_condition_dedupe",
                "inverse_trig_root_symbolic_numerator_scale_positive_gap",
                "inverse_trig_root_symbolic_denominator_internal_scale",
                "inverse_trig_root_symbolic_denominator_scale_dual_orientation",
                "inverse_trig_root_interval_orientation",
                "inverse_trig_root_empty_open_interval_undefined",
                "inverse_trig_shifted_quadratic_empty_open_interval_undefined",
                "inverse_trig_arcsin_point_domain_empty_derivative_undefined",
                "inverse_trig_symbolic_constant_empty_open_interval_undefined",
                "inverse_trig_named_constant_positive_offset_empty_open_interval_undefined",
                "inverse_hyperbolic_atanh_empty_open_interval_undefined",
                "inverse_hyperbolic_atanh_named_constant_positive_offset_empty_open_interval_undefined",
                "inverse_hyperbolic_acosh_empty_lower_bound_undefined",
                "inverse_hyperbolic_acosh_empty_derivative_domain_undefined",
                "inverse_hyperbolic_acosh_named_positive_offset_satisfied_lower_bound",
                "inverse_hyperbolic_root_atanh_symbolic_numerator_scale_open_interval",
                "inverse_hyperbolic_root_atanh_symbolic_denominator_scale_open_interval",
                "inverse_hyperbolic_root_atanh_exact_square_denominator_scale_open_interval",
                "inverse_hyperbolic_root_symbolic_numerator_scale_positive_gap",
                "inverse_hyperbolic_root_asinh_symbolic_denominator_scale_positive_gap",
                "inverse_trig_root_negative_argument",
                "trig_product_rule",
                "sqrt_chain_trig_log_presimplified_condition_dedupe",
                "variable_power_log_domain",
                "shifted_linear_base_variable_power_log_domain",
                "quadratic_base_variable_power_disconnected_domain",
                "abs_piecewise_required_condition",
                "abs_quadratic_factored_nondifferentiable_domain",
                "discontinuous_sign_polynomial_nonzero_domain",
                "discontinuous_sign_quadratic_factored_domain",
            },
        )
        self.assertEqual(supported_step_unchecked, set())
        self.assertEqual(
            SMOKE.count_by(cases, "domain_regime"),
            {
                "empty_derivative_domain": 4,
                "empty_lower_bound_domain": 1,
                "empty_open_interval_domain": 4,
                "empty_positive_argument_domain": 3,
                "empty_positive_exponent_domain": 1,
                "disconnected_positive_base_required": 1,
                "factored_discontinuity_points_required": 1,
                "factored_nondifferentiable_points_required": 1,
                "invalid_log_base_domain": 1,
                "interval_required": 1,
                "linear_poles_required": 6,
                "linear_wrapper_pole_required": 1,
                "negative_base_undefined": 1,
                "nonfinite_undefined": 1,
                "open_interval_required": 3,
                "positive_exponent_required": 1,
                "positive_dominates_nonnegative_source_trig_domain": 1,
                "positive_gap_with_symbolic_denominator_scale_deduped": 1,
                "positive_source_trig_domain": 2,
                "required_condition": 30,
                "shifted_positive_base_required": 1,
                "structurally_satisfied_lower_bound_domain": 1,
                "symbolic_radius_minimal_nonzero_parameter": 2,
                "trig_sine_pole_required": 1,
                "unconditional": 6,
                "unconditional_cosh_positive": 1,
                "unconditional_positive_quadratic": 3,
            },
        )
        self.assertGreaterEqual(len({case.family for case in cases}), 11)

    def test_run_matrix_accepts_expected_required_display_and_residual(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cas_cli = Path(temp_dir) / "cas_cli"
            cas_cli.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/sh
                    case "$*" in
                    *"diff(x^3, x)"*)
                    cat <<'OUT'
                    {"ok":true,"result":"3·x^2","warnings":[],"required_display":[],"steps":[{"substeps":[{"title":"Usar regla de la potencia"}]}]}
                    OUT
                    ;;
                    *"diff((x^2+1)^3, x)"*)
                    cat <<'OUT'
                    {"ok":true,"result":"6·x·(x^2 + 1)^2","warnings":[],"required_display":[],"steps":[{"substeps":[{"title":"Usar regla de la potencia con cadena"},{"title":"Identificar u y du"}]}]}
                    OUT
                    ;;
                    *"diff(sqrt(x), x)"*)
                    cat <<'OUT'
                    {"ok":true,"result":"1 / (2·sqrt(x))","warnings":[],"required_display":["x > 0"],"steps":[{"substeps":[{"title":"Usar regla de la potencia"}]}]}
                    OUT
                    ;;
                    *"diff(sign(x), x)"*)
                    cat <<'OUT'
                    {"ok":true,"result":"0","warnings":[],"required_display":["x ≠ 0"],"steps":[{"substeps":[{"title":"Usar derivada de sign(u) fuera de u = 0"}]}]}
                    OUT
                    ;;
                    *)
                    echo '{"ok":false,"result":"unexpected","warnings":[],"required_display":[],"steps":[]}'
                    ;;
                    esac
                    """
                ),
                encoding="utf-8",
            )
            cas_cli.chmod(cas_cli.stat().st_mode | stat.S_IXUSR)

            cases = SMOKE.build_cases(
                (
                    "polynomial_power_direct",
                    "polynomial_inner_chain_power",
                    "sqrt_variable_open_domain",
                    "discontinuous_sign_polynomial_nonzero_domain",
                )
            )
            matrix = SMOKE.run_matrix(cases, cas_cli=cas_cli, timeout_seconds=2.0)

        self.assertEqual(matrix["status"], "pass", matrix)
        self.assertEqual(matrix["total"], 4)
        self.assertEqual(matrix["status_counts"]["pass"], 4)
        self.assertEqual(matrix["supported_case_count"], 4)
        self.assertEqual(matrix["residual_case_count"], 0)
        self.assertEqual(matrix["warning_expected_case_count"], 0)
        self.assertEqual(matrix["blocked_hint_expected_case_count"], 0)
        self.assertEqual(matrix["required_display_case_count"], 2)
        self.assertEqual(matrix["required_display_counts"], {"x > 0": 1, "x ≠ 0": 1})
        self.assertEqual(matrix["step_checked_case_count"], 4)
        self.assertEqual(matrix["supported_step_unchecked_case_count"], 0)
        self.assertEqual(matrix["supported_step_unchecked_cases"], [])
        self.assertEqual(matrix["expected_step_substring_count"], 5)
        self.assertEqual(
            matrix["calculus_maturity_block_counts"],
            {
                "block2_real_domain_differentiation": 4,
            },
        )
        self.assertEqual(
            matrix["calculus_block_gate_counts"],
            {
                "didactic_trace_and_diff_policy": 2,
                "domain_conditions_and_diff_policy": 2,
            },
        )
        self.assertEqual(
            matrix["cases"][0]["calculus_maturity_block"],
            "block2_real_domain_differentiation",
        )
        self.assertEqual(
            matrix["cases"][0]["calculus_block_gate"],
            "didactic_trace_and_diff_policy",
        )
        self.assertEqual(
            matrix["runtime_by_calculus_maturity_block"][0]["group"],
            "block2_real_domain_differentiation",
        )
        self.assertEqual(
            {
                row["group"]
                for row in matrix["runtime_by_calculus_block_gate"]
            },
            {
                "didactic_trace_and_diff_policy",
                "domain_conditions_and_diff_policy",
            },
        )

    def test_run_matrix_reports_result_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cas_cli = Path(temp_dir) / "cas_cli"
            cas_cli.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/sh
                    cat <<'OUT'
                    {"ok":true,"result":"0","warnings":[],"required_display":[],"steps":[]}
                    OUT
                    """
                ),
                encoding="utf-8",
            )
            cas_cli.chmod(cas_cli.stat().st_mode | stat.S_IXUSR)

            cases = SMOKE.build_cases(("polynomial_power_direct",))
            matrix = SMOKE.run_matrix(cases, cas_cli=cas_cli, timeout_seconds=2.0)

        self.assertEqual(matrix["status"], "fail")
        self.assertEqual(matrix["status_counts"]["fail"], 1)
        self.assertEqual(matrix["issue_kind_counts"], {"result_mismatch": 1})
        self.assertEqual(matrix["problem_cases"][0]["name"], "polynomial_power_direct")

    def test_run_matrix_reports_missing_expected_step_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cas_cli = Path(temp_dir) / "cas_cli"
            cas_cli.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/sh
                    cat <<'OUT'
                    {"ok":true,"result":"6·x·(x^2 + 1)^2","warnings":[],"required_display":[],"steps":[{"rule":"Calcular la derivada"}]}
                    OUT
                    """
                ),
                encoding="utf-8",
            )
            cas_cli.chmod(cas_cli.stat().st_mode | stat.S_IXUSR)

            cases = SMOKE.build_cases(("polynomial_inner_chain_power",))
            matrix = SMOKE.run_matrix(cases, cas_cli=cas_cli, timeout_seconds=2.0)

        self.assertEqual(matrix["status"], "fail")
        self.assertEqual(matrix["status_counts"]["fail"], 1)
        self.assertEqual(matrix["issue_kind_counts"], {"step_trace_mismatch": 1})
        self.assertEqual(matrix["problem_cases"][0]["name"], "polynomial_inner_chain_power")

    def test_run_matrix_reports_fragile_stderr_even_when_result_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cas_cli = Path(temp_dir) / "cas_cli"
            cas_cli.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/sh
                    echo 'WARN simplify: depth_overflow' >&2
                    cat <<'OUT'
                    {"ok":true,"result":"3·x^2","warnings":[],"required_display":[],"steps":[{"substeps":[{"title":"Usar regla de la potencia"}]}]}
                    OUT
                    """
                ),
                encoding="utf-8",
            )
            cas_cli.chmod(cas_cli.stat().st_mode | stat.S_IXUSR)

            cases = SMOKE.build_cases(("polynomial_power_direct",))
            matrix = SMOKE.run_matrix(cases, cas_cli=cas_cli, timeout_seconds=2.0)

        self.assertEqual(matrix["status"], "fail")
        self.assertEqual(matrix["status_counts"]["fail"], 1)
        self.assertEqual(matrix["issue_kind_counts"], {"stderr_fragility": 1})
        self.assertEqual(matrix["problem_cases"][0]["name"], "polynomial_power_direct")
        self.assertIn("fragile substring", matrix["problem_cases"][0]["error"])


if __name__ == "__main__":
    unittest.main()
