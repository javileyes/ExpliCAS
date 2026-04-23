use cas_ast::{Constant, Context, Expr, ExprId};
use cas_parser::parse;
use cas_solver::runtime::Simplifier;
use cas_solver::runtime::SimplifyOptions;
use cas_solver::session_api::analysis::{
    evaluate_derive_command_lines_with_resolver, FullSimplifyDisplayMode,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq)]
struct DeriveCase {
    id: String,
    family: String,
    source: String,
    target: String,
    expected_status: String,
    expected_strategy: Option<String>,
}

fn load_derive_cases() -> Vec<DeriveCase> {
    include_str!("derive_pairs.csv")
        .lines()
        .skip(1)
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let parts = split_csv_line(line);
            assert_eq!(parts.len(), 6, "unexpected derive csv columns: {line}");
            let id = parts[0].trim().to_string();
            let family = parts[1].trim().to_string();
            let source = parts[2].trim().to_string();
            let target = parts[3].trim().to_string();
            let expected_status = parts[4].trim().to_string();
            let expected_strategy = {
                let trimmed = parts[5].trim();
                (!trimmed.is_empty()).then(|| trimmed.to_string())
            };
            DeriveCase {
                id,
                family,
                source,
                target,
                expected_status,
                expected_strategy,
            }
        })
        .collect()
}

fn split_csv_line(line: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;

    for ch in line.chars() {
        match ch {
            '"' => in_quotes = !in_quotes,
            ',' if !in_quotes => {
                parts.push(current.trim().to_string());
                current.clear();
            }
            _ => current.push(ch),
        }
    }

    parts.push(current.trim().to_string());
    parts
}

fn run_case(case: &DeriveCase) -> Vec<String> {
    let mut simplifier = Simplifier::with_default_rules();
    let input = format!("derive {}, {}", case.source, case.target);
    evaluate_derive_command_lines_with_resolver(
        &mut simplifier,
        &input,
        FullSimplifyDisplayMode::Normal,
        SimplifyOptions::default(),
        |_ctx, expr| Ok(expr),
    )
    .expect("derive should evaluate")
}

fn count_top_level_steps(lines: &[String]) -> usize {
    lines
        .iter()
        .filter(|line| {
            let trimmed = line.trim_start();
            let Some(prefix) = trimmed.split_whitespace().next() else {
                return false;
            };
            prefix.ends_with('.')
                && prefix[..prefix.len().saturating_sub(1)]
                    .chars()
                    .all(|ch| ch.is_ascii_digit())
        })
        .count()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SignatureMode {
    Conservative,
    Abstract,
}

#[derive(Debug, Default)]
struct SignatureState {
    variables: BTreeMap<String, usize>,
    functions: BTreeMap<(String, usize), usize>,
}

fn pair_shape_signature(case: &DeriveCase, mode: SignatureMode) -> String {
    let mut ctx = Context::new();
    let source = parse(&case.source, &mut ctx)
        .unwrap_or_else(|err| panic!("failed to parse derive source {}: {err}", case.id));
    let target = parse(&case.target, &mut ctx)
        .unwrap_or_else(|err| panic!("failed to parse derive target {}: {err}", case.id));
    let mut state = SignatureState::default();
    format!(
        "{} => {}",
        expr_shape_signature(&ctx, source, &mut state, mode),
        expr_shape_signature(&ctx, target, &mut state, mode)
    )
}

#[derive(Debug, Default)]
struct DeriveOutcomeStats {
    derived: usize,
    unsupported: usize,
    not_equivalent: usize,
    derived_step_total: usize,
    long_path_count: usize,
    unsupported_by_family: BTreeMap<String, usize>,
    derived_by_family: BTreeMap<String, usize>,
}

const LONG_PATH_THRESHOLD: usize = 4;

fn assert_case_matches_expected_outcome(case: &DeriveCase, stats: &mut DeriveOutcomeStats) {
    let lines = run_case(case);
    match case.expected_status.as_str() {
        "derived" => {
            stats.derived += 1;
            *stats
                .derived_by_family
                .entry(case.family.clone())
                .or_default() += 1;
            let expected_strategy = case
                .expected_strategy
                .as_ref()
                .expect("derived cases must declare strategy");
            assert!(
                lines
                    .iter()
                    .any(|line| line.starts_with("Strategy:") && line.contains(expected_strategy)),
                "case {} should use strategy {}; lines={:?}",
                case.id,
                expected_strategy,
                lines
            );
            assert!(
                lines.iter().any(|line| line.starts_with("Result:")),
                "case {} should emit Result line",
                case.id
            );
            let step_count = count_top_level_steps(&lines);
            stats.derived_step_total += step_count;
            if step_count > LONG_PATH_THRESHOLD {
                stats.long_path_count += 1;
            }
        }
        "equivalent_but_unsupported" => {
            stats.unsupported += 1;
            *stats
                .unsupported_by_family
                .entry(case.family.clone())
                .or_default() += 1;
            assert!(
                lines.iter().any(|line| {
                    line.contains(
                        "Equivalent, but the second expression is not a supported simplification target yet."
                    )
                }),
                "case {} should report equivalent-but-unsupported; lines={:?}",
                case.id,
                lines
            );
        }
        "not_equivalent" => {
            stats.not_equivalent += 1;
            assert!(
                lines.iter().any(|line| {
                    line.contains("Derive unavailable: the two expressions are not equivalent.")
                }),
                "case {} should report not-equivalent; lines={:?}",
                case.id,
                lines
            );
        }
        other => panic!("unexpected expected_status {other}"),
    }
}

fn evaluate_cases_follow_expected_outcomes<'a>(
    cases: impl IntoIterator<Item = &'a DeriveCase>,
) -> DeriveOutcomeStats {
    let mut stats = DeriveOutcomeStats::default();
    for case in cases {
        assert_case_matches_expected_outcome(case, &mut stats);
    }
    stats
}

fn derive_case_bucket(case: &DeriveCase) -> &'static str {
    match case.family.as_str() {
        "trig_expand" | "trig_contract" => "trig",
        "simplify" | "log_expand" | "log_contract" | "rationalize" | "radical_power" => {
            "simplify_log"
        }
        "expand" | "fraction_expand" | "fraction_combine" | "fraction_decompose"
        | "nested_fraction" => "expansion_fraction",
        "collect"
        | "conditional_factor"
        | "factor"
        | "finite_telescoping"
        | "integrate_prep"
        | "negative"
        | "polynomial_product"
        | "power_merge"
        | "solve_prep"
        | "telescoping_fraction" => "structural",
        other => panic!("unassigned derive family bucket: {other}"),
    }
}

fn cases_in_bucket<'a>(cases: &'a [DeriveCase], bucket: &str) -> Vec<&'a DeriveCase> {
    cases
        .iter()
        .filter(|case| derive_case_bucket(case) == bucket)
        .collect()
}

fn derive_case_perf_slice(case: &DeriveCase) -> &'static str {
    match case.family.as_str() {
        "trig_expand" | "trig_contract" => "trig",
        "simplify" | "log_expand" | "log_contract" | "rationalize" | "radical_power" => {
            "simplify_log"
        }
        "expand" => "expand",
        "fraction_expand" | "fraction_combine" | "fraction_decompose" | "nested_fraction" => {
            "fractional"
        }
        "collect" | "conditional_factor" | "factor" | "negative" => "factor_collect",
        "finite_telescoping" | "integrate_prep" | "solve_prep" | "telescoping_fraction" => {
            "prep_telescoping"
        }
        "polynomial_product" | "power_merge" => "poly_merge",
        other => panic!("unassigned derive family perf slice: {other}"),
    }
}

fn cases_in_perf_slice<'a>(cases: &'a [DeriveCase], slice: &str) -> Vec<&'a DeriveCase> {
    cases
        .iter()
        .filter(|case| derive_case_perf_slice(case) == slice)
        .collect()
}

fn find_case_by_id<'a>(cases: &'a [DeriveCase], id: &str) -> &'a DeriveCase {
    cases
        .iter()
        .find(|case| case.id == id)
        .unwrap_or_else(|| panic!("missing derive case id in corpus: {id}"))
}

fn cases_by_ids<'a>(cases: &'a [DeriveCase], ids: &[&str]) -> Vec<&'a DeriveCase> {
    ids.iter().map(|id| find_case_by_id(cases, id)).collect()
}

fn expr_shape_signature(
    ctx: &Context,
    expr: ExprId,
    state: &mut SignatureState,
    mode: SignatureMode,
) -> String {
    match ctx.get(expr) {
        Expr::Number(_) => "N".to_string(),
        Expr::Constant(constant) => match constant {
            Constant::Pi => "pi".to_string(),
            Constant::E => "e".to_string(),
            Constant::Infinity => "infinity".to_string(),
            Constant::Undefined => "undefined".to_string(),
            Constant::I => "i".to_string(),
            Constant::Phi => "phi".to_string(),
        },
        Expr::Variable(id) => {
            let name = ctx.sym_name(*id).to_string();
            let next = state.variables.len() + 1;
            let slot = *state.variables.entry(name).or_insert(next);
            format!("v{slot}")
        }
        Expr::Add(_, _) => {
            let mut terms = Vec::new();
            collect_add_terms(ctx, expr, &mut terms);
            let mut signatures = terms
                .into_iter()
                .map(|term| expr_shape_signature(ctx, term, state, mode))
                .collect::<Vec<_>>();
            signatures.sort();
            format!("add({})", signatures.join(","))
        }
        Expr::Mul(_, _) => {
            let mut factors = Vec::new();
            collect_mul_factors(ctx, expr, &mut factors);
            let mut signatures = factors
                .into_iter()
                .map(|factor| expr_shape_signature(ctx, factor, state, mode))
                .collect::<Vec<_>>();
            signatures.sort();
            format!("mul({})", signatures.join(","))
        }
        Expr::Sub(lhs, rhs) => format!(
            "sub({},{})",
            expr_shape_signature(ctx, *lhs, state, mode),
            expr_shape_signature(ctx, *rhs, state, mode)
        ),
        Expr::Div(lhs, rhs) => format!(
            "div({},{})",
            expr_shape_signature(ctx, *lhs, state, mode),
            expr_shape_signature(ctx, *rhs, state, mode)
        ),
        Expr::Pow(base, exponent) => format!(
            "pow({},{})",
            expr_shape_signature(ctx, *base, state, mode),
            expr_shape_signature(ctx, *exponent, state, mode)
        ),
        Expr::Neg(inner) => format!("neg({})", expr_shape_signature(ctx, *inner, state, mode)),
        Expr::Function(id, args) => {
            let fn_name = match mode {
                SignatureMode::Conservative => ctx.sym_name(*id).to_string(),
                SignatureMode::Abstract => {
                    let key = (ctx.sym_name(*id).to_string(), args.len());
                    let next = state.functions.len() + 1;
                    let slot = *state.functions.entry(key).or_insert(next);
                    format!("f{slot}/{}", args.len())
                }
            };
            let arg_signatures = args
                .iter()
                .map(|arg| expr_shape_signature(ctx, *arg, state, mode))
                .collect::<Vec<_>>();
            format!("{fn_name}({})", arg_signatures.join(","))
        }
        Expr::Matrix { rows, cols, data } => {
            let entries = data
                .iter()
                .map(|entry| expr_shape_signature(ctx, *entry, state, mode))
                .collect::<Vec<_>>();
            format!("matrix[{rows}x{cols}]({})", entries.join(","))
        }
        Expr::SessionRef(_) => "session_ref".to_string(),
        Expr::Hold(inner) => format!("hold({})", expr_shape_signature(ctx, *inner, state, mode)),
    }
}

fn collect_add_terms(ctx: &Context, expr: ExprId, out: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Add(lhs, rhs) => {
            collect_add_terms(ctx, *lhs, out);
            collect_add_terms(ctx, *rhs, out);
        }
        _ => out.push(expr),
    }
}

fn collect_mul_factors(ctx: &Context, expr: ExprId, out: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Mul(lhs, rhs) => {
            collect_mul_factors(ctx, *lhs, out);
            collect_mul_factors(ctx, *rhs, out);
        }
        _ => out.push(expr),
    }
}

fn shape_budget_violations(
    cases: &[DeriveCase],
    mode: SignatureMode,
    max_cluster_size: usize,
) -> Vec<String> {
    let mut clusters: BTreeMap<(String, String, String, String), Vec<String>> = BTreeMap::new();

    for case in cases
        .iter()
        .filter(|case| case.expected_status == "derived")
    {
        let signature = pair_shape_signature(case, mode);
        let strategy = case.expected_strategy.clone().unwrap_or_default();
        clusters
            .entry((
                case.family.clone(),
                case.expected_status.clone(),
                strategy,
                signature,
            ))
            .or_default()
            .push(case.id.clone());
    }

    clusters
        .into_iter()
        .filter_map(|((family, _status, strategy, signature), mut ids)| {
            if ids.len() <= max_cluster_size {
                return None;
            }
            ids.sort();
            Some(format!(
                "mode={mode:?} family={family} strategy={strategy} cluster_size={} ids={} signature={signature}",
                ids.len(),
                ids.join(",")
            ))
        })
        .collect()
}

#[test]
fn derive_pairs_corpus_is_unique_and_nonempty() {
    let cases = load_derive_cases();
    assert!(!cases.is_empty(), "derive corpus must not be empty");

    let mut ids = BTreeSet::new();
    for case in cases {
        assert!(
            ids.insert(case.id.clone()),
            "duplicate derive case id: {}",
            case.id
        );
        assert!(!case.source.is_empty(), "case {} missing source", case.id);
        assert!(!case.target.is_empty(), "case {} missing target", case.id);
        assert!(
            matches!(
                case.expected_status.as_str(),
                "derived" | "equivalent_but_unsupported" | "not_equivalent"
            ),
            "case {} has unsupported expected_status {}",
            case.id,
            case.expected_status
        );
    }
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Debug CI uses partitioned derive outcome slices; broad corpus summary remains for manual/release sweeps"
)]
fn derive_pairs_follow_expected_outcomes() {
    let cases = load_derive_cases();
    let stats = evaluate_cases_follow_expected_outcomes(&cases);

    let supported_equivalent_total = stats.derived + stats.unsupported;
    let derive_reachability_rate = if supported_equivalent_total == 0 {
        0.0
    } else {
        stats.derived as f64 / supported_equivalent_total as f64
    };
    let derive_mean_step_count = if stats.derived == 0 {
        0.0
    } else {
        stats.derived_step_total as f64 / stats.derived as f64
    };
    let derive_long_path_rate = if stats.derived == 0 {
        0.0
    } else {
        stats.long_path_count as f64 / stats.derived as f64
    };

    eprintln!(
        "derive corpus summary: derived={} unsupported={} not_equivalent={}",
        stats.derived, stats.unsupported, stats.not_equivalent
    );
    eprintln!(
        "derive stats: reachability_rate={:.3} supported_equiv_rate={:.3} mean_step_count={:.2} long_path_rate={:.3}",
        derive_reachability_rate,
        derive_reachability_rate,
        derive_mean_step_count,
        derive_long_path_rate
    );
    eprintln!("derive derived-by-family: {:?}", stats.derived_by_family);
    eprintln!(
        "derive unsupported-equivalent-by-family: {:?}",
        stats.unsupported_by_family
    );
}

#[test]
fn derive_pairs_partition_covers_corpus() {
    let cases = load_derive_cases();
    let mut assigned_ids = BTreeSet::new();
    let mut seen_buckets = BTreeSet::new();

    for case in &cases {
        seen_buckets.insert(derive_case_bucket(case));
        assert!(
            assigned_ids.insert(case.id.clone()),
            "derive partition assigned duplicate case id: {}",
            case.id
        );
    }

    assert_eq!(
        assigned_ids.len(),
        cases.len(),
        "derive partition must cover all cases"
    );
    assert_eq!(
        seen_buckets,
        BTreeSet::from(["trig", "simplify_log", "expansion_fraction", "structural",]),
        "derive partition buckets changed unexpectedly"
    );
}

#[test]
fn derive_pairs_perf_slices_cover_corpus() {
    let cases = load_derive_cases();
    let mut assigned_ids = BTreeSet::new();
    let mut seen_slices = BTreeSet::new();

    for case in &cases {
        seen_slices.insert(derive_case_perf_slice(case));
        assert!(
            assigned_ids.insert(case.id.clone()),
            "derive perf slice assigned duplicate case id: {}",
            case.id
        );
    }

    assert_eq!(
        assigned_ids.len(),
        cases.len(),
        "derive perf slices must cover all cases"
    );
    assert_eq!(
        seen_slices,
        BTreeSet::from([
            "trig",
            "simplify_log",
            "expand",
            "fractional",
            "factor_collect",
            "prep_telescoping",
            "poly_merge",
        ]),
        "derive perf slices changed unexpectedly"
    );
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Debug CI uses representative trig derive cases; broad trig sweep remains for manual/release runs"
)]
fn derive_pairs_follow_expected_outcomes_trig() {
    let cases = load_derive_cases();
    let bucket = cases_in_bucket(&cases, "trig");
    assert!(!bucket.is_empty(), "trig derive bucket must not be empty");
    let _ = evaluate_cases_follow_expected_outcomes(bucket);
}

#[test]
fn derive_pairs_follow_expected_outcomes_trig_representatives() {
    let cases = load_derive_cases();
    let reps = cases_by_ids(
        &cases,
        &[
            "expand_trig_product_to_sum_sin_sin",
            "contract_trig_phase_shift_sum_to_shifted_sine",
            "expand_trig_phase_shift_general_shifted_sine_to_sum",
            "contract_trig_tan_quotient_after_arg_simplify",
            "expand_trig_double_cos_as_two_cos_sq_minus_one",
            "contract_trig_half_angle_tangent",
            "expand_trig_sine_eighth_power_reduction",
            "contract_trig_triple_angle_cosine",
            "expand_trig_angle_diff_sine",
            "contract_trig_cos_diff_sin_diff_quotient",
        ],
    );
    let _ = evaluate_cases_follow_expected_outcomes(reps);
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Debug CI uses representative simplify/log derive cases; broad simplify/log sweep remains for manual/release runs"
)]
fn derive_pairs_follow_expected_outcomes_simplify_log() {
    let cases = load_derive_cases();
    let bucket = cases_in_bucket(&cases, "simplify_log");
    assert!(
        !bucket.is_empty(),
        "simplify_log derive bucket must not be empty"
    );
    let _ = evaluate_cases_follow_expected_outcomes(bucket);
}

#[test]
fn derive_pairs_follow_expected_outcomes_simplify_log_representatives() {
    let cases = load_derive_cases();
    let reps = cases_by_ids(
        &cases,
        &[
            "combine_like_terms",
            "hyperbolic_pythagorean_identity_with_passthrough",
            "inverse_tan_identity",
            "perfect_square_root_to_abs_with_passthrough",
            "expand_log_general_base_powered_two_denominator_factors_with_powered_denominator",
            "contract_general_base_logs_to_grouped_power_with_passthrough",
            "rationalize_then_cancel_to_zero",
            "radical_notable_quotient",
            "expand_odd_half_power_after_simplify_with_passthrough",
            "consecutive_factorial_ratio_gap_two",
            "sec_tan_pythagorean_to_one",
            "contract_exponential_power",
            "expand_trig_sine_cosine_square_product_reduction",
        ],
    );
    let _ = evaluate_cases_follow_expected_outcomes(reps);
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Debug CI uses narrower derive outcome slices for expansion and fraction families"
)]
fn derive_pairs_follow_expected_outcomes_expansion_fraction() {
    let cases = load_derive_cases();
    let bucket = cases_in_bucket(&cases, "expansion_fraction");
    assert!(
        !bucket.is_empty(),
        "expansion_fraction derive bucket must not be empty"
    );
    let _ = evaluate_cases_follow_expected_outcomes(bucket);
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Debug CI uses representative expand cases; broad expand sweep remains for manual/release runs"
)]
fn derive_pairs_follow_expected_outcomes_expand() {
    let cases = load_derive_cases();
    let slice = cases_in_perf_slice(&cases, "expand");
    assert!(
        !slice.is_empty(),
        "expand derive perf slice must not be empty"
    );
    let _ = evaluate_cases_follow_expected_outcomes(slice);
}

#[test]
fn derive_pairs_follow_expected_outcomes_expand_representatives() {
    let cases = load_derive_cases();
    let reps = cases_by_ids(
        &cases,
        &[
            "expand_symbolic_binomial",
            "expand_symbolic_trinomial_square",
            "expand_sophie_germain",
            "expand_hyperbolic_sinh_sum_to_product_exact",
            "expand_trig_product_to_sum_to_cosine_difference_polynomial",
            "expand_then_cancel_to_square",
        ],
    );
    let _ = evaluate_cases_follow_expected_outcomes(reps);
}

#[test]
fn derive_pairs_follow_expected_outcomes_fractional() {
    let cases = load_derive_cases();
    let slice = cases_in_perf_slice(&cases, "fractional");
    assert!(
        !slice.is_empty(),
        "fractional derive perf slice must not be empty"
    );
    let _ = evaluate_cases_follow_expected_outcomes(slice);
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Debug CI uses narrower derive outcome slices for structural families"
)]
fn derive_pairs_follow_expected_outcomes_structural() {
    let cases = load_derive_cases();
    let bucket = cases_in_bucket(&cases, "structural");
    assert!(
        !bucket.is_empty(),
        "structural derive bucket must not be empty"
    );
    let _ = evaluate_cases_follow_expected_outcomes(bucket);
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Debug CI uses representative factor/collect derive cases; broad factor/collect sweep remains for manual/release runs"
)]
fn derive_pairs_follow_expected_outcomes_factor_collect() {
    let cases = load_derive_cases();
    let slice = cases_in_perf_slice(&cases, "factor_collect");
    assert!(
        !slice.is_empty(),
        "factor_collect derive perf slice must not be empty"
    );
    let _ = evaluate_cases_follow_expected_outcomes(slice);
}

#[test]
fn derive_pairs_follow_expected_outcomes_factor_collect_representatives() {
    let cases = load_derive_cases();
    let reps = cases_by_ids(
        &cases,
        &[
            "collect_multiple_power_groups",
            "factor_out_with_division_quadratic",
            "factor_difference_squares",
            "factor_perfect_square_trinomial_symbolic",
            "factor_sophie_germain",
            "factor_symbolic_binomial_cube",
            "factor_symbolic_sixth_power_difference",
            "non_equivalent_mismatch",
        ],
    );
    let _ = evaluate_cases_follow_expected_outcomes(reps);
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Debug CI uses representative prep/telescoping cases; broad sweep remains for manual/release runs"
)]
fn derive_pairs_follow_expected_outcomes_prep_telescoping() {
    let cases = load_derive_cases();
    let slice = cases_in_perf_slice(&cases, "prep_telescoping");
    assert!(
        !slice.is_empty(),
        "prep_telescoping derive perf slice must not be empty"
    );
    let _ = evaluate_cases_follow_expected_outcomes(slice);
}

#[test]
fn derive_pairs_follow_expected_outcomes_prep_telescoping_representatives() {
    let cases = load_derive_cases();
    let reps = cases_by_ids(
        &cases,
        &[
            "finite_telescoping_sum_basic",
            "integrate_prep_morrie_basic",
            "integrate_prep_dirichlet_basic",
            "solve_prep_complete_square_symbolic_negative_linear_coeff",
            "split_telescoping_fraction_affine_symbolic_shift_gap",
            "combine_telescoping_fraction_symbolic_difference_squares_unfactored",
        ],
    );
    let _ = evaluate_cases_follow_expected_outcomes(reps);
}

#[test]
fn derive_pairs_follow_expected_outcomes_poly_merge() {
    let cases = load_derive_cases();
    let slice = cases_in_perf_slice(&cases, "poly_merge");
    assert!(
        !slice.is_empty(),
        "poly_merge derive perf slice must not be empty"
    );
    let _ = evaluate_cases_follow_expected_outcomes(slice);
}

#[test]
fn derive_pairs_shape_clusters_stay_within_budget() {
    let cases = load_derive_cases();
    let mut violations = Vec::new();
    violations.extend(shape_budget_violations(
        &cases,
        SignatureMode::Conservative,
        2,
    ));
    violations.extend(shape_budget_violations(&cases, SignatureMode::Abstract, 2));

    assert!(
        violations.is_empty(),
        "derive corpus inflation guardrail failed:\n{}",
        violations.join("\n")
    );
}
