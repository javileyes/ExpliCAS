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
fn derive_pairs_follow_expected_outcomes() {
    let cases = load_derive_cases();

    let mut derived = 0usize;
    let mut unsupported = 0usize;
    let mut not_equivalent = 0usize;
    let mut derived_step_total = 0usize;
    let mut long_path_count = 0usize;
    let mut unsupported_by_family: BTreeMap<String, usize> = BTreeMap::new();
    let mut derived_by_family: BTreeMap<String, usize> = BTreeMap::new();

    const LONG_PATH_THRESHOLD: usize = 4;

    for case in &cases {
        let lines = run_case(case);
        match case.expected_status.as_str() {
            "derived" => {
                derived += 1;
                *derived_by_family.entry(case.family.clone()).or_default() += 1;
                let expected_strategy = case
                    .expected_strategy
                    .as_ref()
                    .expect("derived cases must declare strategy");
                assert!(
                    lines
                        .iter()
                        .any(|line| line.starts_with("Strategy:")
                            && line.contains(expected_strategy)),
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
                derived_step_total += step_count;
                if step_count > LONG_PATH_THRESHOLD {
                    long_path_count += 1;
                }
            }
            "equivalent_but_unsupported" => {
                unsupported += 1;
                *unsupported_by_family
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
                not_equivalent += 1;
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

    let supported_equivalent_total = derived + unsupported;
    let derive_reachability_rate = if supported_equivalent_total == 0 {
        0.0
    } else {
        derived as f64 / supported_equivalent_total as f64
    };
    let derive_mean_step_count = if derived == 0 {
        0.0
    } else {
        derived_step_total as f64 / derived as f64
    };
    let derive_long_path_rate = if derived == 0 {
        0.0
    } else {
        long_path_count as f64 / derived as f64
    };

    eprintln!(
        "derive corpus summary: derived={} unsupported={} not_equivalent={}",
        derived, unsupported, not_equivalent
    );
    eprintln!(
        "derive stats: reachability_rate={:.3} supported_equiv_rate={:.3} mean_step_count={:.2} long_path_rate={:.3}",
        derive_reachability_rate,
        derive_reachability_rate,
        derive_mean_step_count,
        derive_long_path_rate
    );
    eprintln!("derive derived-by-family: {:?}", derived_by_family);
    eprintln!(
        "derive unsupported-equivalent-by-family: {:?}",
        unsupported_by_family
    );
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
