//! `metamorphic_simplification_tests`: familia `support`.
//!
//! Ver la cabecera de `metamorphic_simplification_tests.rs` para el contexto.

use super::*;

pub(super) fn find_test_data_file(filename: &str) -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let local = manifest_dir.join("tests").join(filename);
    if local.exists() {
        return local;
    }

    // Migration compatibility: data moved to cas_solver/tests.
    manifest_dir
        .parent()
        .map(|p| p.join("cas_solver").join("tests").join(filename))
        .unwrap_or(local)
}

/// Get metatest configuration from environment
///
/// Modes:
/// - Normal: 50 samples, depth=3 (default, ~1s per test)
/// - Stress: 500 samples, depth=5 (METATEST_STRESS=1, ~4s per test)
/// - Extreme: 1000 samples, depth=8 (METATEST_EXTREME=1, ~30s+ per test)
pub(super) fn metatest_config() -> MetatestConfig {
    let extreme = env::var("METATEST_EXTREME").ok().as_deref() == Some("1");
    let stress = env::var("METATEST_STRESS").ok().as_deref() == Some("1");

    let (samples, min_valid, depth, eval_samples) = if extreme {
        (1000, 500, 7, 1000) // depth=7 avoids numerical precision issues
    } else if stress {
        (500, 250, 5, 500)
    } else {
        (50, 20, 3, 200)
    };

    // Seed from env or default
    let seed = env::var("METATEST_SEED")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0xC0FFEE_u64);

    MetatestConfig {
        samples,
        min_valid,
        depth,
        seed,
        atol: 1e-9,
        rtol: 1e-9,
        sample_range: (-5.0, 5.0),
        eval_samples,
        near_singularity_threshold: 1e10, // Values > 10^10 are considered near-singularity
    }
}

/// Generate a stable shape signature from an expression.
/// Collapses literals to NUM, variables to SYM, and marks negative exponents.
/// Used to identify dominant patterns in numeric-only cases.
pub(super) fn expr_shape_signature(ctx: &Context, expr: ExprId) -> String {
    fn inner(ctx: &Context, expr: ExprId, depth: usize) -> String {
        if depth > 10 {
            return "...".to_string();
        }
        match ctx.get(expr) {
            Expr::Number(n) => {
                if n.is_integer() {
                    let val = n.numer().to_string().parse::<i64>().unwrap_or(0);
                    if val < 0 {
                        format!("INT_NEG({})", val)
                    } else {
                        "INT".to_string()
                    }
                } else {
                    "FRAC".to_string()
                }
            }
            Expr::Variable(_) => "SYM".to_string(),
            Expr::Constant(_) => "CONST".to_string(),
            Expr::Add(l, r) => {
                let mut parts = [inner(ctx, *l, depth + 1), inner(ctx, *r, depth + 1)];
                parts.sort(); // Canonical order for commutativity
                format!("Add({})", parts.join(","))
            }
            Expr::Sub(l, r) => {
                format!(
                    "Sub({},{})",
                    inner(ctx, *l, depth + 1),
                    inner(ctx, *r, depth + 1)
                )
            }
            Expr::Mul(l, r) => {
                let mut parts = [inner(ctx, *l, depth + 1), inner(ctx, *r, depth + 1)];
                parts.sort(); // Canonical order for commutativity
                format!("Mul({})", parts.join(","))
            }
            Expr::Div(l, r) => {
                format!(
                    "Div({},{})",
                    inner(ctx, *l, depth + 1),
                    inner(ctx, *r, depth + 1)
                )
            }
            Expr::Pow(base, exp) => {
                // Special handling for negative integer exponents
                let exp_sig = match ctx.get(*exp) {
                    Expr::Number(n) if n.is_integer() => {
                        let val = n.numer().to_string().parse::<i64>().unwrap_or(0);
                        if val < 0 {
                            format!("INT_NEG({})", val)
                        } else if val == 2 {
                            "2".to_string()
                        } else {
                            "INT".to_string()
                        }
                    }
                    Expr::Number(n) => {
                        // Check for 1/n fractions (roots)
                        if *n.numer() == 1.into() {
                            "1/N".to_string()
                        } else {
                            "FRAC".to_string()
                        }
                    }
                    _ => inner(ctx, *exp, depth + 1),
                };
                format!("Pow({},{})", inner(ctx, *base, depth + 1), exp_sig)
            }
            Expr::Neg(inner_expr) => {
                format!("Neg({})", inner(ctx, *inner_expr, depth + 1))
            }
            Expr::Function(name, args) => {
                let arg_sigs: Vec<_> = args.iter().map(|a| inner(ctx, *a, depth + 1)).collect();
                format!("{}({})", name, arg_sigs.join(","))
            }
            Expr::Matrix { .. } => "MAT".to_string(),
            Expr::SessionRef(_) => "REF".to_string(),
            Expr::Hold(held) => format!("Hold({})", inner(ctx, *held, depth + 1)),
        }
    }
    inner(ctx, expr, 0)
}

pub(super) fn fold_constants_safe(ctx: &mut Context, expr: ExprId) -> ExprId {
    let cfg = cas_solver::runtime::EvalConfig::default();
    let mut budget = cas_solver::runtime::Budget::preset_cli();
    let folded = cas_solver::api::fold_constants(
        ctx,
        expr,
        &cfg,
        cas_solver::api::ConstFoldMode::Safe,
        &mut budget,
    )
    .map(|r| r.expr)
    .unwrap_or(expr);
    cas_ast::hold::strip_all_holds(ctx, folded)
}

pub(super) fn prove_zero_from_curated_pair_corpus_text(lhs: &str, rhs: &str) -> bool {
    let lhs = normalize_pair_text(lhs);
    let rhs = normalize_pair_text(rhs);
    if curated_pair_corpus()
        .raw
        .contains(&(lhs.clone(), rhs.clone()))
    {
        return true;
    }

    let (Some(lhs_alpha), Some(rhs_alpha)) = (
        alpha_normalize_pair_text(&lhs),
        alpha_normalize_pair_text(&rhs),
    ) else {
        return false;
    };
    curated_pair_corpus()
        .alpha
        .contains(&(lhs_alpha, rhs_alpha))
}

pub(super) fn prove_zero_from_metamorphic_texts(
    simplifier: &mut Simplifier,
    lhs_text: &str,
    rhs_text: &str,
    lhs_simp: ExprId,
    rhs_simp: ExprId,
) -> bool {
    prove_zero_from_metamorphic_texts_with_flavor(
        simplifier,
        lhs_text,
        rhs_text,
        lhs_simp,
        rhs_simp,
        MetamorphicProofFlavor::Curated,
    )
}

pub(super) fn prove_zero_from_engine_texts_in_child_process(lhs: &str, rhs: &str) -> bool {
    let Ok(current_exe) = std::env::current_exe() else {
        return false;
    };

    let mut child = match std::process::Command::new(current_exe)
        // `super::`, not `crate::`: cas_engine's compatibility wrapper includes
        // this harness one module deeper, where `crate::runners` does not exist.
        .arg(super::runners::CHILD_RAW_PRESSURE_PROOF.filter())
        .arg("--ignored")
        .arg("--exact")
        .arg("--nocapture")
        .env(METATEST_CHILD_RAW_PROOF_LHS_ENV, lhs)
        .env(METATEST_CHILD_RAW_PROOF_RHS_ENV, rhs)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        Ok(child) => child,
        Err(_) => return false,
    };

    let timeout = std::time::Duration::from_millis(METATEST_CHILD_RAW_PROOF_TIMEOUT_MS);
    let start = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return status.success(),
            Ok(None) => {
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return false;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                return false;
            }
        }
    }
}

pub(super) fn normalize_metamorphic_text(text: &str) -> String {
    text.chars().filter(|c| !c.is_whitespace()).collect()
}

pub(super) fn known_domain_frontier_reason(lhs_text: &str, rhs_text: &str) -> Option<&'static str> {
    let lhs = normalize_metamorphic_text(lhs_text);
    let rhs = normalize_metamorphic_text(rhs_text);
    let pair_matches = |a: &str, b: &str| (lhs == a && rhs == b) || (lhs == b && rhs == a);

    if pair_matches("ln((1/(u-1)+1/(u+1))^2)", "2*ln((1/(u-1)+1/(u+1)))")
        || pair_matches("ln((-u)^2)", "2*ln((-u))")
        || pair_matches("ln((2*u)^2)", "2*ln((2*u))")
        || pair_matches("ln((1-u)^2)", "2*ln((1-u))")
    {
        return Some("log-square expansion changes domain");
    }
    if pair_matches(
        "((exp(x)-exp(-x))/2)*(sin(2*arcsin(u)))",
        "(sinh(x))*(2*u*sqrt(1-u^2))",
    ) || pair_matches(
        "(tanh(x))*(sin(2*arcsin(u)))",
        "((exp(x)-exp(-x))/(exp(x)+exp(-x)))*(2*u*sqrt(1-u^2))",
    ) || pair_matches(
        "(sin(2*arcsin(x)))*(abs(sin(u/2)))",
        "(2*x*sqrt(1-x^2))*(sqrt((1-cos(u))/2))",
    ) {
        return Some("inverse-trig branch introduces domain/branch sensitivity");
    }
    if pair_matches(
        "(cos(3*pi/8))*(sqrt(u)*sqrt(4*u))",
        "(sqrt(2-sqrt(2))/2)*(2*u)",
    ) || pair_matches(
        "(sin(2*arcsin(x)))*(sqrt(u)*sqrt(4*u))",
        "(2*x*sqrt(1-x^2))*(2*u)",
    ) || pair_matches(
        "(tanh(x))-(sqrt(u)*sqrt(4*u))",
        "((exp(x)-exp(-x))/(exp(x)+exp(-x)))-(2*u)",
    ) {
        return Some("sqrt product contraction changes sign/domain behavior");
    }

    None
}

pub(super) fn known_domain_frontier_reason_for_numeric_cause(
    cause: &str,
    lhs_text: &str,
    rhs_text: &str,
) -> Option<&'static str> {
    if cause != "domain-sensitive" {
        return None;
    }

    known_domain_frontier_reason(lhs_text, rhs_text)
}

pub(super) fn classify_numeric_equiv_for_vars(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    vars: &[String],
    filters: &[FilterSpec],
    config: &MetatestConfig,
) -> NumericCheckOutcome {
    match vars {
        [] => NumericCheckOutcome::Inconclusive("No free vars for numeric check".to_string()),
        [var] => classify_numeric_equiv_1var_relaxed_with(config, |retry_filter| {
            let effective = if retry_filter.is_none() {
                filters.first().unwrap_or(&FilterSpec::None)
            } else {
                retry_filter
            };
            check_numeric_equiv_1var_stats(ctx, a, b, var, config, effective)
        }),
        [var1, var2] => classify_numeric_equiv_2var_relaxed(
            ctx,
            a,
            b,
            var1,
            var2,
            config,
            filters.first().unwrap_or(&FilterSpec::None),
            filters.get(1).unwrap_or(&FilterSpec::None),
        ),
        _ => classify_numeric_equiv_nvar_relaxed(ctx, a, b, vars, filters, config),
    }
}

pub(super) fn numeric_only_cause_for_vars(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    vars: &[String],
    filters: &[FilterSpec],
    config: &MetatestConfig,
    residual_shape: &str,
) -> NumericOnlyCause {
    match vars {
        [var] => numeric_only_cause_for_1var(
            ctx,
            a,
            b,
            var,
            config,
            filters.first().unwrap_or(&FilterSpec::None),
            residual_shape,
        ),
        [var1, var2] => numeric_only_cause_for_2var(
            ctx,
            a,
            b,
            var1,
            var2,
            config,
            filters.first().unwrap_or(&FilterSpec::None),
            filters.get(1).unwrap_or(&FilterSpec::None),
            residual_shape,
        ),
        many => classify_numeric_only_cause(None, many.len(), residual_shape),
    }
}

pub(super) fn effective_combo_window(
    total_combos: usize,
    requested_start: Option<usize>,
    requested_cap: Option<usize>,
) -> (usize, usize) {
    let start = requested_start.unwrap_or(0).min(total_combos);
    let remaining = total_combos.saturating_sub(start);
    (start, effective_combo_cap(remaining, requested_cap))
}
