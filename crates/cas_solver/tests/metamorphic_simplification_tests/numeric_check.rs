//! `metamorphic_simplification_tests`: familia `numeric_check`.
//!
//! Ver la cabecera de `metamorphic_simplification_tests.rs` para el contexto.

use super::*;

fn numeric_sample_value(
    profile_order: &[NumericSampleProfile; 3],
    sample_idx: usize,
    var_idx: usize,
) -> f64 {
    let profile = profile_order[sample_idx % profile_order.len()];
    let round = sample_idx / profile_order.len();
    let (values, step, var_step): (&[f64], usize, usize) = match profile {
        NumericSampleProfile::Interior => (&NUMERIC_INTERIOR_VALUES, 7, 13),
        NumericSampleProfile::General => (&NUMERIC_GENERAL_VALUES, 7, 13),
        NumericSampleProfile::Positive => (&NUMERIC_POSITIVE_VALUES, 4, 11),
        NumericSampleProfile::Rational => (&NUMERIC_RATIONAL_VALUES, 5, 9),
    };
    let idx = (round * step + var_idx * var_step) % values.len();
    values[idx]
}

fn collect_numeric_sampling_features(
    ctx: &Context,
    expr: ExprId,
    features: &mut NumericSamplingFeatures,
) {
    match ctx.get(expr) {
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) => {
            collect_numeric_sampling_features(ctx, *a, features);
            collect_numeric_sampling_features(ctx, *b, features);
        }
        Expr::Div(a, b) => {
            features.rational_sensitive = true;
            collect_numeric_sampling_features(ctx, *a, features);
            collect_numeric_sampling_features(ctx, *b, features);
        }
        Expr::Pow(base, exp) => {
            if let Some(exp_q) = as_rational_const(ctx, *exp, 4) {
                if exp_q.is_negative() {
                    features.rational_sensitive = true;
                }
                if !exp_q.is_integer() {
                    features.positivity_sensitive = true;
                }
            } else if matches!(ctx.get(*exp), Expr::Div(_, _)) {
                features.positivity_sensitive = true;
            }
            collect_numeric_sampling_features(ctx, *base, features);
            collect_numeric_sampling_features(ctx, *exp, features);
        }
        Expr::Neg(a) | Expr::Hold(a) => {
            collect_numeric_sampling_features(ctx, *a, features);
        }
        Expr::Function(fn_id, args) => {
            if ctx.is_builtin(*fn_id, BuiltinFn::Ln)
                || ctx.is_builtin(*fn_id, BuiltinFn::Log)
                || ctx.is_builtin(*fn_id, BuiltinFn::Log2)
                || ctx.is_builtin(*fn_id, BuiltinFn::Log10)
                || ctx.is_builtin(*fn_id, BuiltinFn::Sqrt)
                || ctx.is_builtin(*fn_id, BuiltinFn::Cbrt)
                || ctx.is_builtin(*fn_id, BuiltinFn::Root)
            {
                features.positivity_sensitive = true;
            }
            if ctx.is_builtin(*fn_id, BuiltinFn::Asin)
                || ctx.is_builtin(*fn_id, BuiltinFn::Acos)
                || ctx.is_builtin(*fn_id, BuiltinFn::Arcsin)
                || ctx.is_builtin(*fn_id, BuiltinFn::Arccos)
            {
                features.bounded_inverse_trig = true;
            }
            for arg in args {
                collect_numeric_sampling_features(ctx, *arg, features);
            }
        }
        Expr::Matrix { data, .. } => {
            for d in data {
                collect_numeric_sampling_features(ctx, *d, features);
            }
        }
        Expr::Variable(_) | Expr::Number(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
    }
}

pub(super) fn choose_numeric_sample_profile_order_exprs(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
) -> Option<[NumericSampleProfile; 3]> {
    let mut features = NumericSamplingFeatures::default();
    collect_numeric_sampling_features(ctx, a, &mut features);
    collect_numeric_sampling_features(ctx, b, &mut features);

    if !(features.positivity_sensitive
        || features.bounded_inverse_trig
        || features.rational_sensitive)
    {
        return None;
    }

    Some(
        if features.positivity_sensitive && features.bounded_inverse_trig {
            [
                NumericSampleProfile::Positive,
                NumericSampleProfile::Interior,
                if features.rational_sensitive {
                    NumericSampleProfile::Rational
                } else {
                    NumericSampleProfile::General
                },
            ]
        } else if features.positivity_sensitive {
            [
                NumericSampleProfile::Positive,
                if features.rational_sensitive {
                    NumericSampleProfile::Rational
                } else {
                    NumericSampleProfile::General
                },
                NumericSampleProfile::Interior,
            ]
        } else if features.bounded_inverse_trig {
            [
                NumericSampleProfile::Interior,
                if features.rational_sensitive {
                    NumericSampleProfile::Rational
                } else {
                    NumericSampleProfile::General
                },
                NumericSampleProfile::Positive,
            ]
        } else if features.rational_sensitive {
            [
                NumericSampleProfile::Rational,
                NumericSampleProfile::General,
                NumericSampleProfile::Positive,
            ]
        } else {
            [
                NumericSampleProfile::General,
                NumericSampleProfile::Interior,
                NumericSampleProfile::Positive,
            ]
        },
    )
}

fn collect_numeric_denominator_guards(ctx: &Context, expr: ExprId, guards: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) => {
            collect_numeric_denominator_guards(ctx, *a, guards);
            collect_numeric_denominator_guards(ctx, *b, guards);
        }
        Expr::Div(a, b) => {
            guards.push(*b);
            collect_numeric_denominator_guards(ctx, *a, guards);
            collect_numeric_denominator_guards(ctx, *b, guards);
        }
        Expr::Pow(base, exp) => {
            if let Some(exp_q) = as_rational_const(ctx, *exp, 4) {
                if exp_q.is_negative() {
                    guards.push(*base);
                }
                collect_numeric_denominator_guards(ctx, *base, guards);
            } else {
                collect_numeric_denominator_guards(ctx, *base, guards);
                collect_numeric_denominator_guards(ctx, *exp, guards);
            }
        }
        Expr::Neg(a) | Expr::Hold(a) => {
            collect_numeric_denominator_guards(ctx, *a, guards);
        }
        Expr::Function(_, args) => {
            for arg in args {
                collect_numeric_denominator_guards(ctx, *arg, guards);
            }
        }
        Expr::Matrix { data, .. } => {
            for d in data {
                collect_numeric_denominator_guards(ctx, *d, guards);
            }
        }
        Expr::Variable(_) | Expr::Number(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
    }
}

fn collect_numeric_analytic_guards(
    ctx: &Context,
    expr: ExprId,
    guards: &mut Vec<NumericAnalyticGuard>,
) {
    match ctx.get(expr) {
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) | Expr::Pow(a, b) => {
            collect_numeric_analytic_guards(ctx, *a, guards);
            collect_numeric_analytic_guards(ctx, *b, guards);
        }
        Expr::Neg(a) | Expr::Hold(a) => {
            collect_numeric_analytic_guards(ctx, *a, guards);
        }
        Expr::Function(fn_id, args) => {
            if (ctx.is_builtin(*fn_id, BuiltinFn::Ln)
                || ctx.is_builtin(*fn_id, BuiltinFn::Log2)
                || ctx.is_builtin(*fn_id, BuiltinFn::Log10))
                && !args.is_empty()
            {
                guards.push(NumericAnalyticGuard {
                    expr: args[0],
                    kind: NumericAnalyticGuardKind::Positive,
                });
            }

            if ctx.is_builtin(*fn_id, BuiltinFn::Log) {
                if let Some(&base) = args.first() {
                    guards.push(NumericAnalyticGuard {
                        expr: base,
                        kind: NumericAnalyticGuardKind::Positive,
                    });
                    guards.push(NumericAnalyticGuard {
                        expr: base,
                        kind: NumericAnalyticGuardKind::NotOne,
                    });
                }
                if args.len() > 1 {
                    guards.push(NumericAnalyticGuard {
                        expr: args[1],
                        kind: NumericAnalyticGuardKind::Positive,
                    });
                }
            }

            if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) && !args.is_empty() {
                guards.push(NumericAnalyticGuard {
                    expr: args[0],
                    kind: NumericAnalyticGuardKind::NonNegative,
                });
            }

            if (ctx.is_builtin(*fn_id, BuiltinFn::Asin)
                || ctx.is_builtin(*fn_id, BuiltinFn::Acos)
                || ctx.is_builtin(*fn_id, BuiltinFn::Arcsin)
                || ctx.is_builtin(*fn_id, BuiltinFn::Arccos))
                && !args.is_empty()
            {
                guards.push(NumericAnalyticGuard {
                    expr: args[0],
                    kind: NumericAnalyticGuardKind::UnitInterval,
                });
            }

            for arg in args {
                collect_numeric_analytic_guards(ctx, *arg, guards);
            }
        }
        Expr::Matrix { data, .. } => {
            for d in data {
                collect_numeric_analytic_guards(ctx, *d, guards);
            }
        }
        Expr::Variable(_) | Expr::Number(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
    }
}

fn violates_numeric_analytic_guard(
    ctx: &Context,
    guard: NumericAnalyticGuard,
    var_map: &HashMap<String, f64>,
) -> bool {
    match eval_f64(ctx, guard.expr, var_map) {
        Some(v) if v.is_finite() => match guard.kind {
            NumericAnalyticGuardKind::Positive => v <= NUMERIC_DENOM_GUARD_ATOL,
            NumericAnalyticGuardKind::NonNegative => v < -NUMERIC_DENOM_GUARD_ATOL,
            NumericAnalyticGuardKind::NotOne => (v - 1.0).abs() <= NUMERIC_DENOM_GUARD_ATOL,
            NumericAnalyticGuardKind::UnitInterval => v.abs() > 1.0 + NUMERIC_DENOM_GUARD_ATOL,
        },
        _ => true,
    }
}

pub(super) fn collect_numeric_precheck_guards(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
) -> (Vec<ExprId>, Vec<NumericAnalyticGuard>) {
    let mut denom_guards = Vec::new();
    let mut analytic_guards = Vec::new();
    collect_numeric_denominator_guards(ctx, a, &mut denom_guards);
    collect_numeric_denominator_guards(ctx, b, &mut denom_guards);
    collect_numeric_analytic_guards(ctx, a, &mut analytic_guards);
    collect_numeric_analytic_guards(ctx, b, &mut analytic_guards);
    (denom_guards, analytic_guards)
}

pub(super) fn sample_violates_numeric_precheck_guards(
    ctx: &Context,
    denom_guards: &[ExprId],
    analytic_guards: &[NumericAnalyticGuard],
    var_map: &HashMap<String, f64>,
) -> bool {
    denom_guards
        .iter()
        .any(|guard| near_numeric_guard_zero(ctx, *guard, var_map))
        || analytic_guards
            .iter()
            .any(|guard| violates_numeric_analytic_guard(ctx, *guard, var_map))
}

/// Check if two expressions are numerically equivalent for 1 variable.
/// Returns Ok(valid_count) or Err(message).
pub(super) fn check_numeric_equiv_1var(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    var: &str,
    config: &MetatestConfig,
) -> Result<usize, String> {
    let stats = check_numeric_equiv_1var_stats(ctx, a, b, var, config, &FilterSpec::None);
    finalize_numeric_equiv_1var(stats, config)
}

/// Stats-returning version of check_numeric_equiv_1var for diagnostics
pub(super) fn check_numeric_equiv_1var_stats(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    var: &str,
    config: &MetatestConfig,
    filter_spec: &FilterSpec,
) -> NumericEquivStats {
    let (lo, hi) = config.sample_range;
    let mut stats = NumericEquivStats::default();
    let (denom_guards, analytic_guards) = collect_numeric_precheck_guards(ctx, a, b);
    let profile_order = choose_numeric_sample_profile_order_exprs(ctx, a, b);

    // Configure checked evaluator with near-pole detection
    let opts = EvalCheckedOptions {
        zero_abs_eps: 1e-12,
        zero_rel_eps: 1e-12,
        trig_pole_eps: 1e-9,
        max_depth: 200,
    };

    for i in 0..config.eval_samples {
        let x = if let Some(order) = profile_order {
            numeric_sample_value(&order, i, 0)
        } else {
            let t = (i as f64 + 0.5) / config.eval_samples as f64;
            lo + (hi - lo) * t
        };

        // Apply filter if specified
        if !filter_spec.accept(x) {
            stats.filtered_out += 1;
            continue;
        }

        let mut var_map = HashMap::new();
        var_map.insert(var.to_string(), x);

        if sample_violates_numeric_precheck_guards(ctx, &denom_guards, &analytic_guards, &var_map) {
            stats.domain_error += 1;
            continue;
        }

        let va = eval_f64_checked(ctx, a, &var_map, &opts);
        let vb = eval_f64_checked(ctx, b, &var_map, &opts);

        match (&va, &vb) {
            (Ok(va), Ok(vb)) => {
                let diff = (va - vb).abs();
                let scale = va.abs().max(vb.abs()).max(1.0);
                let allowed = config.atol + config.rtol * scale;

                if diff <= allowed {
                    stats.valid += 1;
                } else {
                    stats.record_mismatch(x, *va, *vb, var);
                }
            }
            (Err(EvalCheckedError::NearPole { .. }), Err(EvalCheckedError::NearPole { .. })) => {
                stats.near_pole += 1;
            }
            (Err(EvalCheckedError::Domain { .. }), Err(EvalCheckedError::Domain { .. })) => {
                stats.domain_error += 1;
            }
            (Ok(_), Err(_)) | (Err(_), Ok(_)) => {
                stats.asymmetric_invalid += 1;
            }
            _ => {
                stats.eval_failed += 1;
            }
        }
    }

    stats
}

/// Check if two expressions are numerically equivalent for 2 variables.
/// Returns Ok(valid_count) or Err(message).
#[allow(clippy::too_many_arguments)]
pub(super) fn check_numeric_equiv_2var(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    var1: &str,
    var2: &str,
    config: &MetatestConfig,
    filter1: &FilterSpec,
    filter2: &FilterSpec,
) -> Result<usize, String> {
    let stats = check_numeric_equiv_2var_stats(ctx, a, b, var1, var2, config, filter1, filter2);
    finalize_numeric_equiv_2var(stats, config)
}

#[allow(clippy::too_many_arguments)]
fn check_numeric_equiv_2var_stats(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    var1: &str,
    var2: &str,
    config: &MetatestConfig,
    filter1: &FilterSpec,
    filter2: &FilterSpec,
) -> NumericEquivStats {
    let (lo, hi) = config.sample_range;
    let mut stats = NumericEquivStats::default();
    let (denom_guards, analytic_guards) = collect_numeric_precheck_guards(ctx, a, b);
    let profile_order = choose_numeric_sample_profile_order_exprs(ctx, a, b);

    // Configure checked evaluator
    let opts = EvalCheckedOptions {
        zero_abs_eps: 1e-12,
        zero_rel_eps: 1e-12,
        trig_pole_eps: 1e-9,
        max_depth: 200,
    };

    // Use fewer samples for 2D grid to keep runtime reasonable
    let samples_per_dim = (config.eval_samples as f64).sqrt() as usize;

    for i in 0..samples_per_dim {
        for j in 0..samples_per_dim {
            let (x, y) = if let Some(order) = profile_order {
                (
                    numeric_sample_value(&order, i, 0),
                    numeric_sample_value(&order, j, 1),
                )
            } else {
                let t1 = (i as f64 + 0.5) / samples_per_dim as f64;
                let t2 = (j as f64 + 0.5) / samples_per_dim as f64;
                (lo + (hi - lo) * t1, lo + (hi - lo) * t2)
            };

            let mut var_map = HashMap::new();
            var_map.insert(var1.to_string(), x);
            var_map.insert(var2.to_string(), y);

            // Apply per-variable domain filters
            if !filter1.accept(x) || !filter2.accept(y) {
                stats.domain_error += 1;
                continue;
            }

            if sample_violates_numeric_precheck_guards(
                ctx,
                &denom_guards,
                &analytic_guards,
                &var_map,
            ) {
                stats.domain_error += 1;
                continue;
            }

            let va = eval_f64_checked(ctx, a, &var_map, &opts);
            let vb = eval_f64_checked(ctx, b, &var_map, &opts);

            match (&va, &vb) {
                (Ok(va), Ok(vb)) => {
                    let diff = (va - vb).abs();
                    let scale = va.abs().max(vb.abs()).max(1.0);
                    let allowed = config.atol + config.rtol * scale;

                    if diff <= allowed {
                        stats.valid += 1;
                    } else {
                        stats.record_mismatch_label(
                            format!("{var1}={x:.6}, {var2}={y:.6}"),
                            *va,
                            *vb,
                        );
                    }
                }
                // Symmetric failures
                (
                    Err(EvalCheckedError::NearPole { .. }),
                    Err(EvalCheckedError::NearPole { .. }),
                ) => {
                    stats.near_pole += 1;
                }
                (Err(EvalCheckedError::Domain { .. }), Err(EvalCheckedError::Domain { .. })) => {
                    stats.domain_error += 1;
                }
                // Asymmetric: one Ok, one Err
                (Ok(_), Err(_)) | (Err(_), Ok(_)) => {
                    stats.asymmetric_invalid += 1;
                }
                _ => {
                    stats.eval_failed += 1;
                }
            }
        }
    }

    stats
}

fn finalize_numeric_equiv_2var(
    stats: NumericEquivStats,
    config: &MetatestConfig,
) -> Result<usize, String> {
    // Lower threshold for 2D, adjusted for problematic samples
    let problematic =
        stats.near_pole + stats.domain_error + stats.eval_failed + stats.asymmetric_invalid;
    let total_samples = {
        let samples_per_dim = (config.eval_samples as f64).sqrt() as usize;
        samples_per_dim * samples_per_dim
    };
    let base_min_valid = config.min_valid / 4;
    let adjusted_min_valid = if problematic > total_samples / 4 {
        (total_samples - problematic) / 2
    } else {
        base_min_valid
    };

    if stats.valid < adjusted_min_valid {
        return Err(format!(
            "Too few valid samples: {} < {} (near_pole={}, domain_error={}, asymmetric={}, eval_failed={})",
            stats.valid,
            adjusted_min_valid,
            stats.near_pole,
            stats.domain_error,
            stats.asymmetric_invalid,
            stats.eval_failed
        ));
    }

    if !stats.mismatches.is_empty() {
        return Err(format!(
            "Numeric mismatches: {}",
            stats.mismatches.join("; ")
        ));
    }

    Ok(stats.valid)
}

/// Check if two expressions are numerically equivalent for 3+ variables.
/// Uses a deterministic low-discrepancy style sampling pattern instead of a full grid
/// to keep runtime bounded while still covering multivariate contextual identities.
fn check_numeric_equiv_nvar(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    vars: &[String],
    config: &MetatestConfig,
) -> Result<usize, String> {
    let (lo, hi) = config.sample_range;
    let mut valid = 0usize;
    let mut eval_failed = 0usize;
    let mut near_pole = 0usize;
    let mut domain_error = 0usize;
    let mut asymmetric_invalid = 0usize;

    let opts = EvalCheckedOptions {
        zero_abs_eps: 1e-12,
        zero_rel_eps: 1e-12,
        trig_pole_eps: 1e-9,
        max_depth: 200,
    };
    let (denom_guards, analytic_guards) = collect_numeric_precheck_guards(ctx, a, b);
    let profile_order = choose_numeric_sample_profile_order_exprs(ctx, a, b);

    // Golden-ratio increment for a simple deterministic low-discrepancy walk.
    const PHASE: f64 = 0.381_966_011_250_105_1;

    for i in 0..config.eval_samples {
        let base = (i as f64 + 0.5) / config.eval_samples as f64;
        let mut var_map = HashMap::new();

        for (idx, var) in vars.iter().enumerate() {
            let value = if let Some(order) = profile_order {
                numeric_sample_value(&order, i + idx, idx)
            } else {
                let t = (base + idx as f64 * PHASE).fract();
                lo + (hi - lo) * t
            };
            var_map.insert(var.clone(), value);
        }

        if sample_violates_numeric_precheck_guards(ctx, &denom_guards, &analytic_guards, &var_map) {
            domain_error += 1;
            continue;
        }

        let va = eval_f64_checked(ctx, a, &var_map, &opts);
        let vb = eval_f64_checked(ctx, b, &var_map, &opts);

        match (&va, &vb) {
            (Ok(va), Ok(vb)) => {
                valid += 1;

                let diff = (va - vb).abs();
                let scale = va.abs().max(vb.abs()).max(1.0);
                let allowed = config.atol + config.rtol * scale;

                if diff > allowed {
                    let bindings = vars
                        .iter()
                        .map(|v| format!("{v}={:.12}", var_map[v]))
                        .collect::<Vec<_>>()
                        .join(", ");
                    return Err(format!(
                        "Numeric mismatch at {}:\n  a={:.15}\n  b={:.15}\n  diff={:.3e} > allowed={:.3e}",
                        bindings, va, vb, diff, allowed
                    ));
                }
            }
            (Err(EvalCheckedError::NearPole { .. }), Err(EvalCheckedError::NearPole { .. })) => {
                near_pole += 1;
            }
            (Err(EvalCheckedError::Domain { .. }), Err(EvalCheckedError::Domain { .. })) => {
                domain_error += 1;
            }
            (Ok(_), Err(_)) | (Err(_), Ok(_)) => {
                asymmetric_invalid += 1;
            }
            _ => {
                eval_failed += 1;
            }
        }
    }

    let problematic = near_pole + domain_error + eval_failed + asymmetric_invalid;
    let adjusted_min_valid = if problematic > config.eval_samples / 4 {
        (config.eval_samples - problematic) / 2
    } else {
        config.min_valid / 2
    };

    if valid < adjusted_min_valid {
        return Err(format!(
            "Too few valid samples: {} < {} (near_pole={}, domain_error={}, asymmetric={}, eval_failed={})",
            valid, adjusted_min_valid, near_pole, domain_error, asymmetric_invalid, eval_failed
        ));
    }

    Ok(valid)
}

fn check_numeric_equiv_1var_with_fixed(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    var: &str,
    fixed_vars: &[(String, f64)],
    config: &MetatestConfig,
) -> Result<usize, String> {
    let stats = check_numeric_equiv_1var_with_fixed_stats(ctx, a, b, var, fixed_vars, config);
    finalize_numeric_equiv_1var(stats, config)
}

fn check_numeric_equiv_1var_with_fixed_stats(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    var: &str,
    fixed_vars: &[(String, f64)],
    config: &MetatestConfig,
) -> NumericEquivStats {
    check_numeric_equiv_1var_with_fixed_stats_filtered(
        ctx,
        a,
        b,
        var,
        fixed_vars,
        config,
        &FilterSpec::None,
    )
}

fn check_numeric_equiv_1var_with_fixed_stats_filtered(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    var: &str,
    fixed_vars: &[(String, f64)],
    config: &MetatestConfig,
    filter_spec: &FilterSpec,
) -> NumericEquivStats {
    let (lo, hi) = config.sample_range;
    let mut stats = NumericEquivStats::default();
    let (denom_guards, analytic_guards) = collect_numeric_precheck_guards(ctx, a, b);
    let profile_order = choose_numeric_sample_profile_order_exprs(ctx, a, b);

    let opts = EvalCheckedOptions {
        zero_abs_eps: 1e-12,
        zero_rel_eps: 1e-12,
        trig_pole_eps: 1e-9,
        max_depth: 200,
    };

    for i in 0..config.eval_samples {
        let x = if let Some(order) = profile_order {
            numeric_sample_value(&order, i, 0)
        } else {
            let t = (i as f64 + 0.5) / config.eval_samples as f64;
            lo + (hi - lo) * t
        };

        if !filter_spec.accept(x) {
            stats.filtered_out += 1;
            continue;
        }

        let mut var_map = HashMap::new();
        for (name, value) in fixed_vars {
            var_map.insert(name.clone(), *value);
        }
        var_map.insert(var.to_string(), x);

        if sample_violates_numeric_precheck_guards(ctx, &denom_guards, &analytic_guards, &var_map) {
            stats.domain_error += 1;
            continue;
        }

        let va = eval_f64_checked(ctx, a, &var_map, &opts);
        let vb = eval_f64_checked(ctx, b, &var_map, &opts);

        match (&va, &vb) {
            (Ok(va), Ok(vb)) => {
                let diff = (va - vb).abs();
                let scale = va.abs().max(vb.abs()).max(1.0);
                let allowed = config.atol + config.rtol * scale;

                if diff <= allowed {
                    stats.valid += 1;
                } else {
                    stats.record_mismatch(x, *va, *vb, var);
                }
            }
            (Err(EvalCheckedError::NearPole { .. }), Err(EvalCheckedError::NearPole { .. })) => {
                stats.near_pole += 1;
            }
            (Err(EvalCheckedError::Domain { .. }), Err(EvalCheckedError::Domain { .. })) => {
                stats.domain_error += 1;
            }
            (Ok(_), Err(_)) | (Err(_), Ok(_)) => {
                stats.asymmetric_invalid += 1;
            }
            _ => {
                stats.eval_failed += 1;
            }
        }
    }

    stats
}

fn check_numeric_equiv_2var_with_fixed(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    var1: &str,
    var2: &str,
    fixed_vars: &[(String, f64)],
    config: &MetatestConfig,
) -> Result<usize, String> {
    let stats =
        check_numeric_equiv_2var_with_fixed_stats(ctx, a, b, var1, var2, fixed_vars, config);
    finalize_numeric_equiv_2var(stats, config)
}

fn check_numeric_equiv_2var_with_fixed_stats(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    var1: &str,
    var2: &str,
    fixed_vars: &[(String, f64)],
    config: &MetatestConfig,
) -> NumericEquivStats {
    check_numeric_equiv_2var_with_fixed_stats_filtered(
        ctx,
        a,
        b,
        var1,
        var2,
        fixed_vars,
        config,
        &FilterSpec::None,
        &FilterSpec::None,
    )
}

#[allow(clippy::too_many_arguments)]
fn check_numeric_equiv_2var_with_fixed_stats_filtered(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    var1: &str,
    var2: &str,
    fixed_vars: &[(String, f64)],
    config: &MetatestConfig,
    filter1: &FilterSpec,
    filter2: &FilterSpec,
) -> NumericEquivStats {
    let (lo, hi) = config.sample_range;
    let mut stats = NumericEquivStats::default();
    let (denom_guards, analytic_guards) = collect_numeric_precheck_guards(ctx, a, b);
    let profile_order = choose_numeric_sample_profile_order_exprs(ctx, a, b);

    let opts = EvalCheckedOptions {
        zero_abs_eps: 1e-12,
        zero_rel_eps: 1e-12,
        trig_pole_eps: 1e-9,
        max_depth: 200,
    };

    let samples_per_dim = (config.eval_samples as f64).sqrt() as usize;

    for i in 0..samples_per_dim {
        for j in 0..samples_per_dim {
            let (x, y) = if let Some(order) = profile_order {
                (
                    numeric_sample_value(&order, i, 0),
                    numeric_sample_value(&order, j, 1),
                )
            } else {
                let t1 = (i as f64 + 0.5) / samples_per_dim as f64;
                let t2 = (j as f64 + 0.5) / samples_per_dim as f64;
                (lo + (hi - lo) * t1, lo + (hi - lo) * t2)
            };

            if !filter1.accept(x) || !filter2.accept(y) {
                stats.filtered_out += 1;
                continue;
            }

            let mut var_map = HashMap::new();
            for (name, value) in fixed_vars {
                var_map.insert(name.clone(), *value);
            }
            var_map.insert(var1.to_string(), x);
            var_map.insert(var2.to_string(), y);

            if sample_violates_numeric_precheck_guards(
                ctx,
                &denom_guards,
                &analytic_guards,
                &var_map,
            ) {
                stats.domain_error += 1;
                continue;
            }

            let va = eval_f64_checked(ctx, a, &var_map, &opts);
            let vb = eval_f64_checked(ctx, b, &var_map, &opts);

            match (&va, &vb) {
                (Ok(va), Ok(vb)) => {
                    let diff = (va - vb).abs();
                    let scale = va.abs().max(vb.abs()).max(1.0);
                    let allowed = config.atol + config.rtol * scale;

                    if diff <= allowed {
                        stats.valid += 1;
                    } else {
                        stats.record_mismatch_label(
                            format!("{var1}={x:.6}, {var2}={y:.6}"),
                            *va,
                            *vb,
                        );
                    }
                }
                (
                    Err(EvalCheckedError::NearPole { .. }),
                    Err(EvalCheckedError::NearPole { .. }),
                ) => {
                    stats.near_pole += 1;
                }
                (Err(EvalCheckedError::Domain { .. }), Err(EvalCheckedError::Domain { .. })) => {
                    stats.domain_error += 1;
                }
                (Ok(_), Err(_)) | (Err(_), Ok(_)) => {
                    stats.asymmetric_invalid += 1;
                }
                _ => {
                    stats.eval_failed += 1;
                }
            }
        }
    }

    stats
}

fn sample_nvar_slice_anchor(
    config: &MetatestConfig,
    idx: usize,
    seed: f64,
    profile_order: Option<&[NumericSampleProfile; 3]>,
) -> f64 {
    if let Some(order) = profile_order {
        let seed_slot = ((seed * 1024.0).abs() as usize) % 97;
        return numeric_sample_value(order, seed_slot + idx * 3, idx);
    }

    let (lo, hi) = config.sample_range;
    const PHASE: f64 = 0.381_966_011_250_105_1;
    let t = (seed + idx as f64 * PHASE).fract();
    lo + (hi - lo) * t
}

pub(super) fn sample_nvar_slice_anchor_filtered(
    config: &MetatestConfig,
    idx: usize,
    seed: f64,
    filter: &FilterSpec,
    profile_order: Option<&[NumericSampleProfile; 3]>,
) -> f64 {
    let base = sample_nvar_slice_anchor(config, idx, seed, profile_order);
    if filter.accept(base) || filter.is_none() {
        return base;
    }

    const OFFSETS: [f64; 8] = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875];

    for offset in OFFSETS {
        let candidate =
            sample_nvar_slice_anchor(config, idx, (seed + offset).fract(), profile_order);
        if filter.accept(candidate) {
            return candidate;
        }
    }

    base
}

pub(super) fn classify_numeric_equiv_nvar_relaxed(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    vars: &[String],
    filters: &[FilterSpec],
    config: &MetatestConfig,
) -> NumericCheckOutcome {
    let direct = check_numeric_equiv_nvar(ctx, a, b, vars, config);
    let direct_error = match direct {
        Ok(_) => return NumericCheckOutcome::Pass,
        Err(msg) => msg,
    };

    let direct_was_inconclusive = direct_error.starts_with("Too few valid samples:");
    let mut passed_slices = 0usize;
    let mut inconclusive_slices = 0usize;
    const SLICE_SEEDS: [f64; 2] = [0.173_205_080_756_887_73, 0.618_033_988_749_894_8];
    const MAX_PAIR_SLICES_PER_SEED: usize = 6;

    for seed in SLICE_SEEDS {
        let mut checked_pairs = 0usize;
        let anchors = build_nvar_slice_anchors(ctx, a, b, vars, filters, config, seed);

        for (idx, free_var) in vars.iter().enumerate() {
            let fixed = anchors
                .iter()
                .filter(|(name, _)| name != free_var)
                .cloned()
                .collect::<Vec<_>>();
            match classify_numeric_equiv_1var_with_fixed_relaxed(
                ctx,
                a,
                b,
                free_var,
                &fixed,
                config,
                filters.get(idx).unwrap_or(&FilterSpec::None),
            ) {
                NumericCheckOutcome::Pass => passed_slices += 1,
                NumericCheckOutcome::Inconclusive(_) => inconclusive_slices += 1,
                NumericCheckOutcome::Failed(msg) => {
                    return NumericCheckOutcome::Failed(format!(
                        "{} | slice(1d,{free_var}) failed: {}",
                        direct_error, msg
                    ));
                }
            }
        }

        'pair_slices: for (idx, var1) in vars.iter().enumerate() {
            for (offset, var2) in vars.iter().skip(idx + 1).enumerate() {
                if checked_pairs >= MAX_PAIR_SLICES_PER_SEED {
                    break 'pair_slices;
                }
                checked_pairs += 1;
                let idx2 = idx + 1 + offset;

                let fixed = anchors
                    .iter()
                    .filter(|(name, _)| name != var1 && name != var2)
                    .cloned()
                    .collect::<Vec<_>>();

                match classify_numeric_equiv_2var_with_fixed_relaxed(
                    ctx,
                    a,
                    b,
                    var1,
                    var2,
                    &fixed,
                    config,
                    filters.get(idx).unwrap_or(&FilterSpec::None),
                    filters.get(idx2).unwrap_or(&FilterSpec::None),
                ) {
                    NumericCheckOutcome::Pass => passed_slices += 1,
                    NumericCheckOutcome::Inconclusive(_) => inconclusive_slices += 1,
                    NumericCheckOutcome::Failed(msg) => {
                        return NumericCheckOutcome::Failed(format!(
                            "{} | slice(2d,{var1},{var2}) failed: {}",
                            direct_error, msg
                        ));
                    }
                }
            }
        }
    }

    if passed_slices > 0 {
        NumericCheckOutcome::Inconclusive(format!(
            "Direct n-var check failed but deterministic slices passed ({passed_slices} passed, {inconclusive_slices} inconclusive): {direct_error}"
        ))
    } else if direct_was_inconclusive || inconclusive_slices > 0 {
        NumericCheckOutcome::Inconclusive(format!(
            "Direct n-var check remained inconclusive ({inconclusive_slices} slices inconclusive): {direct_error}"
        ))
    } else {
        NumericCheckOutcome::Failed(direct_error)
    }
}

pub(super) fn finalize_numeric_equiv_1var(
    stats: NumericEquivStats,
    config: &MetatestConfig,
) -> Result<usize, String> {
    let problematic =
        stats.near_pole + stats.domain_error + stats.eval_failed + stats.asymmetric_invalid;
    let adjusted_min_valid = if problematic > config.eval_samples / 4 {
        (config.eval_samples - problematic) / 2
    } else {
        config.min_valid
    };

    if stats.valid < adjusted_min_valid {
        return Err(format!(
            "Too few valid samples: {} < {} (near_pole={}, domain_error={}, asymmetric={}, eval_failed={})",
            stats.valid,
            adjusted_min_valid,
            stats.near_pole,
            stats.domain_error,
            stats.asymmetric_invalid,
            stats.eval_failed
        ));
    }

    if !stats.mismatches.is_empty() {
        return Err(format!(
            "Numeric mismatches: {}",
            stats.mismatches.join("; ")
        ));
    }

    Ok(stats.valid)
}

fn classify_numeric_check(result: Result<usize, String>) -> NumericCheckOutcome {
    match result {
        Ok(_) => NumericCheckOutcome::Pass,
        Err(msg)
            if msg.starts_with("Too few valid samples:")
                || msg.starts_with("Unsupported contextual numeric arity:") =>
        {
            NumericCheckOutcome::Inconclusive(msg)
        }
        Err(msg) => NumericCheckOutcome::Failed(msg),
    }
}

pub(super) fn classify_numeric_check_with_stats(
    result: Result<usize, String>,
    stats: &NumericEquivStats,
) -> NumericCheckOutcome {
    match result {
        Ok(_) => NumericCheckOutcome::Pass,
        Err(msg)
            if msg.starts_with("Too few valid samples:")
                || msg.starts_with("Unsupported contextual numeric arity:") =>
        {
            NumericCheckOutcome::Inconclusive(msg)
        }
        Err(msg) => match classify_diagnostic(stats) {
            DiagCategory::BugSignal | DiagCategory::Ok => NumericCheckOutcome::Failed(msg),
            DiagCategory::ConfigError | DiagCategory::NeedsFilter | DiagCategory::Fragile => {
                NumericCheckOutcome::Inconclusive(format!(
                    "{}: {}",
                    classify_diagnostic(stats).name(),
                    msg
                ))
            }
        },
    }
}

fn numeric_retry_filters_1var() -> [FilterSpec; 4] {
    [
        FilterSpec::AwayFrom {
            centers: vec![0.0, 1.0, -1.0],
            eps: 0.1,
        },
        FilterSpec::AbsLtAndAway {
            limit: 0.9,
            centers: vec![0.0, 1.0, -1.0],
            eps: 0.1,
        },
        FilterSpec::Range {
            min: -0.8,
            max: 0.8,
        },
        FilterSpec::AbsLt { limit: 0.9 },
    ]
}

fn numeric_retry_filters_2var() -> [(FilterSpec, FilterSpec); 4] {
    [
        (
            FilterSpec::AwayFrom {
                centers: vec![0.0, 1.0, -1.0],
                eps: 0.1,
            },
            FilterSpec::AwayFrom {
                centers: vec![0.0, 1.0, -1.0],
                eps: 0.1,
            },
        ),
        (
            FilterSpec::AbsLtAndAway {
                limit: 0.9,
                centers: vec![0.0, 1.0, -1.0],
                eps: 0.1,
            },
            FilterSpec::AbsLtAndAway {
                limit: 0.9,
                centers: vec![0.0, 1.0, -1.0],
                eps: 0.1,
            },
        ),
        (
            FilterSpec::Range {
                min: -0.8,
                max: 0.8,
            },
            FilterSpec::Range {
                min: -0.8,
                max: 0.8,
            },
        ),
        (
            FilterSpec::AbsLt { limit: 0.9 },
            FilterSpec::AbsLt { limit: 0.9 },
        ),
    ]
}

fn should_retry_relaxed_numeric_1var(
    result: &Result<usize, String>,
    stats: &NumericEquivStats,
) -> bool {
    match result {
        Ok(_) => false,
        Err(msg) if msg.starts_with("Unsupported contextual numeric arity:") => false,
        Err(msg) if msg.starts_with("Too few valid samples:") => true,
        Err(_) => matches!(
            classify_diagnostic(stats),
            DiagCategory::NeedsFilter | DiagCategory::Fragile
        ),
    }
}

fn should_retry_relaxed_numeric_2var(
    result: &Result<usize, String>,
    stats: &NumericEquivStats,
) -> bool {
    match result {
        Ok(_) => false,
        Err(msg) if msg.starts_with("Unsupported contextual numeric arity:") => false,
        Err(msg) if msg.starts_with("Too few valid samples:") => true,
        Err(_) => matches!(
            classify_diagnostic(stats),
            DiagCategory::NeedsFilter | DiagCategory::Fragile
        ),
    }
}

pub(super) fn classify_numeric_equiv_1var_relaxed_with<F>(
    config: &MetatestConfig,
    mut run_stats: F,
) -> NumericCheckOutcome
where
    F: FnMut(&FilterSpec) -> NumericEquivStats,
{
    let direct_stats = run_stats(&FilterSpec::None);
    let direct_result = finalize_numeric_equiv_1var(direct_stats.clone(), config);
    let direct_outcome = classify_numeric_check_with_stats(direct_result.clone(), &direct_stats);

    if matches!(direct_outcome, NumericCheckOutcome::Pass) {
        return NumericCheckOutcome::Pass;
    }

    if !should_retry_relaxed_numeric_1var(&direct_result, &direct_stats) {
        return direct_outcome;
    }

    let mut retry_notes = Vec::new();
    for filter in numeric_retry_filters_1var() {
        let stats = run_stats(&filter);
        let result = finalize_numeric_equiv_1var(stats.clone(), config);
        match classify_numeric_check_with_stats(result, &stats) {
            NumericCheckOutcome::Pass => return NumericCheckOutcome::Pass,
            NumericCheckOutcome::Failed(msg) => {
                return NumericCheckOutcome::Failed(format!(
                    "after filter {} => {}",
                    filter.as_str(),
                    msg
                ));
            }
            NumericCheckOutcome::Inconclusive(msg) => {
                retry_notes.push(format!("{} => {}", filter.as_str(), msg));
            }
        }
    }

    let base_msg = match direct_outcome {
        NumericCheckOutcome::Inconclusive(msg) | NumericCheckOutcome::Failed(msg) => msg,
        NumericCheckOutcome::Pass => unreachable!("pass returns early"),
    };

    if retry_notes.is_empty() {
        NumericCheckOutcome::Inconclusive(base_msg)
    } else {
        NumericCheckOutcome::Inconclusive(format!(
            "{} [retry_filters: {}]",
            base_msg,
            retry_notes.join(" | ")
        ))
    }
}

fn classify_numeric_equiv_2var_relaxed_with<F>(
    config: &MetatestConfig,
    mut run_stats: F,
) -> NumericCheckOutcome
where
    F: FnMut(&FilterSpec, &FilterSpec) -> NumericEquivStats,
{
    let direct_stats = run_stats(&FilterSpec::None, &FilterSpec::None);
    let direct_result = finalize_numeric_equiv_2var(direct_stats.clone(), config);
    let direct_outcome = classify_numeric_check_with_stats(direct_result.clone(), &direct_stats);

    if matches!(direct_outcome, NumericCheckOutcome::Pass) {
        return NumericCheckOutcome::Pass;
    }

    if !should_retry_relaxed_numeric_2var(&direct_result, &direct_stats) {
        return direct_outcome;
    }

    let mut retry_notes = Vec::new();
    for (filter1, filter2) in numeric_retry_filters_2var() {
        let stats = run_stats(&filter1, &filter2);
        let result = finalize_numeric_equiv_2var(stats.clone(), config);
        match classify_numeric_check_with_stats(result, &stats) {
            NumericCheckOutcome::Pass => return NumericCheckOutcome::Pass,
            NumericCheckOutcome::Failed(msg) => {
                return NumericCheckOutcome::Failed(format!(
                    "after filters ({}, {}) => {}",
                    filter1.as_str(),
                    filter2.as_str(),
                    msg
                ));
            }
            NumericCheckOutcome::Inconclusive(msg) => {
                retry_notes.push(format!(
                    "({}, {}) => {}",
                    filter1.as_str(),
                    filter2.as_str(),
                    msg
                ));
            }
        }
    }

    let base_msg = match direct_outcome {
        NumericCheckOutcome::Inconclusive(msg) | NumericCheckOutcome::Failed(msg) => msg,
        NumericCheckOutcome::Pass => unreachable!("pass returns early"),
    };

    if retry_notes.is_empty() {
        NumericCheckOutcome::Inconclusive(base_msg)
    } else {
        NumericCheckOutcome::Inconclusive(format!(
            "{} [retry_filters: {}]",
            base_msg,
            retry_notes.join(" | ")
        ))
    }
}

#[allow(clippy::too_many_arguments)]
fn check_numeric_equiv_2var_with_fixed_stats_retry_filters(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    var1: &str,
    var2: &str,
    fixed_vars: &[(String, f64)],
    config: &MetatestConfig,
    base_filter1: &FilterSpec,
    base_filter2: &FilterSpec,
    retry_filter1: &FilterSpec,
    retry_filter2: &FilterSpec,
) -> NumericEquivStats {
    let (lo, hi) = config.sample_range;
    let mut stats = NumericEquivStats::default();
    let (denom_guards, analytic_guards) = collect_numeric_precheck_guards(ctx, a, b);
    let profile_order = choose_numeric_sample_profile_order_exprs(ctx, a, b);

    let opts = EvalCheckedOptions {
        zero_abs_eps: 1e-12,
        zero_rel_eps: 1e-12,
        trig_pole_eps: 1e-9,
        max_depth: 200,
    };

    let samples_per_dim = (config.eval_samples as f64).sqrt() as usize;

    for i in 0..samples_per_dim {
        for j in 0..samples_per_dim {
            let (x, y) = if let Some(order) = profile_order {
                (
                    numeric_sample_value(&order, i, 0),
                    numeric_sample_value(&order, j, 1),
                )
            } else {
                let t1 = (i as f64 + 0.5) / samples_per_dim as f64;
                let t2 = (j as f64 + 0.5) / samples_per_dim as f64;
                (lo + (hi - lo) * t1, lo + (hi - lo) * t2)
            };

            if !(base_filter1.accept(x)
                && retry_filter1.accept(x)
                && base_filter2.accept(y)
                && retry_filter2.accept(y))
            {
                stats.filtered_out += 1;
                continue;
            }

            let mut var_map = HashMap::new();
            for (name, value) in fixed_vars {
                var_map.insert(name.clone(), *value);
            }
            var_map.insert(var1.to_string(), x);
            var_map.insert(var2.to_string(), y);

            if sample_violates_numeric_precheck_guards(
                ctx,
                &denom_guards,
                &analytic_guards,
                &var_map,
            ) {
                stats.domain_error += 1;
                continue;
            }

            let va = eval_f64_checked(ctx, a, &var_map, &opts);
            let vb = eval_f64_checked(ctx, b, &var_map, &opts);

            match (&va, &vb) {
                (Ok(va), Ok(vb)) => {
                    let diff = (va - vb).abs();
                    let scale = va.abs().max(vb.abs()).max(1.0);
                    let allowed = config.atol + config.rtol * scale;

                    if diff <= allowed {
                        stats.valid += 1;
                    } else {
                        stats.record_mismatch_label(
                            format!("{var1}={x:.6}, {var2}={y:.6}"),
                            *va,
                            *vb,
                        );
                    }
                }
                (
                    Err(EvalCheckedError::NearPole { .. }),
                    Err(EvalCheckedError::NearPole { .. }),
                ) => {
                    stats.near_pole += 1;
                }
                (Err(EvalCheckedError::Domain { .. }), Err(EvalCheckedError::Domain { .. })) => {
                    stats.domain_error += 1;
                }
                (Ok(_), Err(_)) | (Err(_), Ok(_)) => {
                    stats.asymmetric_invalid += 1;
                }
                _ => {
                    stats.eval_failed += 1;
                }
            }
        }
    }

    stats
}

pub(super) fn classify_numeric_equiv_1var_relaxed(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    var: &str,
    config: &MetatestConfig,
) -> NumericCheckOutcome {
    classify_numeric_equiv_1var_relaxed_with(config, |filter_spec| {
        check_numeric_equiv_1var_stats(ctx, a, b, var, config, filter_spec)
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn classify_numeric_equiv_2var_relaxed(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    var1: &str,
    var2: &str,
    config: &MetatestConfig,
    filter1: &FilterSpec,
    filter2: &FilterSpec,
) -> NumericCheckOutcome {
    classify_numeric_equiv_2var_relaxed_with(config, |retry_filter1, retry_filter2| {
        if retry_filter1.is_none() && retry_filter2.is_none() {
            check_numeric_equiv_2var_stats(ctx, a, b, var1, var2, config, filter1, filter2)
        } else {
            check_numeric_equiv_2var_with_fixed_stats_retry_filters(
                ctx,
                a,
                b,
                var1,
                var2,
                &[],
                config,
                filter1,
                filter2,
                retry_filter1,
                retry_filter2,
            )
        }
    })
}

fn classify_numeric_equiv_1var_with_fixed_relaxed(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    var: &str,
    fixed_vars: &[(String, f64)],
    config: &MetatestConfig,
    filter: &FilterSpec,
) -> NumericCheckOutcome {
    classify_numeric_equiv_1var_relaxed_with(config, |filter_spec| {
        let effective_filter = if filter_spec.is_none() {
            filter
        } else {
            filter_spec
        };
        check_numeric_equiv_1var_with_fixed_stats_filtered(
            ctx,
            a,
            b,
            var,
            fixed_vars,
            config,
            effective_filter,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn classify_numeric_equiv_2var_with_fixed_relaxed(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    var1: &str,
    var2: &str,
    fixed_vars: &[(String, f64)],
    config: &MetatestConfig,
    filter1: &FilterSpec,
    filter2: &FilterSpec,
) -> NumericCheckOutcome {
    classify_numeric_equiv_2var_relaxed_with(config, |retry_filter1, retry_filter2| {
        if retry_filter1.is_none() && retry_filter2.is_none() {
            check_numeric_equiv_2var_with_fixed_stats_filtered(
                ctx, a, b, var1, var2, fixed_vars, config, filter1, filter2,
            )
        } else {
            check_numeric_equiv_2var_with_fixed_stats_retry_filters(
                ctx,
                a,
                b,
                var1,
                var2,
                fixed_vars,
                config,
                filter1,
                filter2,
                retry_filter1,
                retry_filter2,
            )
        }
    })
}

pub(super) fn classify_nf_first_add_sub_combo_in_child_process(
    lhs: &str,
    rhs: &str,
) -> NfFirstAddSubChildOutcome {
    let Ok(current_exe) = std::env::current_exe() else {
        return NfFirstAddSubChildOutcome::Timeout;
    };

    let outcome_path = std::env::temp_dir().join(format!(
        "metatest_nf_first_add_sub_{}_{}.txt",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));

    let mut child = match std::process::Command::new(current_exe)
        .arg("metatest_child_nf_first_add_sub_classify")
        .arg("--ignored")
        .arg("--exact")
        .arg("--nocapture")
        .env(METATEST_CHILD_NF_ADD_SUB_EXP_ENV, lhs)
        .env(METATEST_CHILD_NF_ADD_SUB_SIMP_ENV, rhs)
        .env(
            METATEST_CHILD_NF_ADD_SUB_OUTCOME_ENV,
            outcome_path.to_string_lossy().to_string(),
        )
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        Ok(child) => child,
        Err(_) => return NfFirstAddSubChildOutcome::Timeout,
    };

    let timeout = std::time::Duration::from_millis(METATEST_CHILD_NF_ADD_SUB_TIMEOUT_MS);
    let start = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                if !status.success() {
                    let _ = std::fs::remove_file(&outcome_path);
                    return NfFirstAddSubChildOutcome::Timeout;
                }
                let outcome = std::fs::read_to_string(&outcome_path)
                    .ok()
                    .map(|s| s.trim().to_string())
                    .unwrap_or_else(|| "inconclusive".to_string());
                let _ = std::fs::remove_file(&outcome_path);
                return match outcome.as_str() {
                    "nf" => NfFirstAddSubChildOutcome::Nf,
                    "proved" => NfFirstAddSubChildOutcome::Proved,
                    "inconclusive" => NfFirstAddSubChildOutcome::Inconclusive,
                    _ => NfFirstAddSubChildOutcome::Timeout,
                };
            }
            Ok(None) => {
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    let _ = std::fs::remove_file(&outcome_path);
                    return NfFirstAddSubChildOutcome::Timeout;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = std::fs::remove_file(&outcome_path);
                return NfFirstAddSubChildOutcome::Timeout;
            }
        }
    }
}

pub(super) fn classify_nf_first_mul_div_combo_in_child_process(
    lhs: &str,
    rhs: &str,
    vars: &[String],
    filters: &[FilterSpec],
    timeout: std::time::Duration,
) -> NfFirstMulDivChildOutcome {
    let Ok(current_exe) = std::env::current_exe() else {
        return NfFirstMulDivChildOutcome::Timeout;
    };

    let outcome_path = std::env::temp_dir().join(format!(
        "metatest_nf_first_mul_div_{}_{}.json",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));

    let mut child = match std::process::Command::new(current_exe)
        .arg("metatest_child_nf_first_mul_div_classify")
        .arg("--ignored")
        .arg("--exact")
        .arg("--nocapture")
        .env(METATEST_CHILD_NF_MUL_DIV_EXP_ENV, lhs)
        .env(METATEST_CHILD_NF_MUL_DIV_SIMP_ENV, rhs)
        .env(METATEST_CHILD_NF_MUL_DIV_VARS_ENV, encode_child_vars(vars))
        .env(
            METATEST_CHILD_NF_MUL_DIV_FILTERS_ENV,
            encode_child_filters(filters),
        )
        .env(
            METATEST_CHILD_NF_MUL_DIV_OUTCOME_ENV,
            outcome_path.to_string_lossy().to_string(),
        )
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        Ok(child) => child,
        Err(_) => return NfFirstMulDivChildOutcome::Timeout,
    };

    let start = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                if !status.success() {
                    let _ = std::fs::remove_file(&outcome_path);
                    return NfFirstMulDivChildOutcome::Timeout;
                }
                let outcome = std::fs::read_to_string(&outcome_path)
                    .ok()
                    .and_then(|raw| serde_json::from_str::<Value>(&raw).ok());
                let _ = std::fs::remove_file(&outcome_path);
                let Some(payload) = outcome else {
                    return NfFirstMulDivChildOutcome::Inconclusive {
                        reason: "missing_child_payload".to_string(),
                        cycles: 0,
                    };
                };
                let kind = payload
                    .get("kind")
                    .and_then(Value::as_str)
                    .unwrap_or("inconclusive");
                let cycles = payload.get("cycles").and_then(Value::as_u64).unwrap_or(0) as usize;
                return match kind {
                    "nf" => NfFirstMulDivChildOutcome::Nf { cycles },
                    "proved-q" => NfFirstMulDivChildOutcome::ProvedQuotient { cycles },
                    "proved-d" => NfFirstMulDivChildOutcome::ProvedDifference { cycles },
                    "numeric" => NfFirstMulDivChildOutcome::Numeric {
                        diff_str: payload
                            .get("diff_str")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        shape: payload
                            .get("shape")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        cause: payload
                            .get("cause")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        cycles,
                    },
                    "domain_frontier" => NfFirstMulDivChildOutcome::DomainFrontier {
                        reason: payload
                            .get("reason")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        shape: payload
                            .get("shape")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        cause: payload
                            .get("cause")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        cycles,
                    },
                    "inconclusive" => NfFirstMulDivChildOutcome::Inconclusive {
                        reason: payload
                            .get("reason")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        cycles,
                    },
                    "failed" => NfFirstMulDivChildOutcome::Failed { cycles },
                    "skip" => NfFirstMulDivChildOutcome::Skip,
                    _ => NfFirstMulDivChildOutcome::Timeout,
                };
            }
            Ok(None) => {
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    let _ = std::fs::remove_file(&outcome_path);
                    if prove_zero_from_engine_texts_in_child_process(lhs, rhs) {
                        return NfFirstMulDivChildOutcome::ProvedDifference { cycles: 0 };
                    }
                    return NfFirstMulDivChildOutcome::Timeout;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = std::fs::remove_file(&outcome_path);
                if prove_zero_from_engine_texts_in_child_process(lhs, rhs) {
                    return NfFirstMulDivChildOutcome::ProvedDifference { cycles: 0 };
                }
                return NfFirstMulDivChildOutcome::Timeout;
            }
        }
    }
}

#[test]
#[ignore]
fn metatest_child_nf_first_add_sub_classify() {
    let lhs =
        std::env::var(METATEST_CHILD_NF_ADD_SUB_EXP_ENV).expect("missing child add/sub lhs env");
    let rhs =
        std::env::var(METATEST_CHILD_NF_ADD_SUB_SIMP_ENV).expect("missing child add/sub rhs env");
    let outcome_path = std::env::var(METATEST_CHILD_NF_ADD_SUB_OUTCOME_ENV)
        .expect("missing child add/sub outcome env");

    let handle = std::thread::Builder::new()
        .stack_size(METATEST_DEEP_WORKER_STACK_SIZE_BYTES)
        .spawn(move || {
            let mut engine = Engine::new();
            let simplifier = &mut engine.simplifier;
            let outcome = match (
                parse(&lhs, &mut simplifier.context),
                parse(&rhs, &mut simplifier.context),
            ) {
                (Ok(lhs_parsed), Ok(rhs_parsed)) => {
                    let opts = cas_solver::runtime::SimplifyOptions::default();
                    let (mut lhs_simp, _, _) =
                        simplifier.simplify_with_stats(lhs_parsed, opts.clone());
                    let (mut rhs_simp, _, _) = simplifier.simplify_with_stats(rhs_parsed, opts);
                    lhs_simp = fold_constants_safe(&mut simplifier.context, lhs_simp);
                    rhs_simp = fold_constants_safe(&mut simplifier.context, rhs_simp);
                    if normal_forms_visibly_equal(&simplifier.context, lhs_simp, rhs_simp) {
                        "nf"
                    } else if prove_zero_from_engine_texts_child_hint(&lhs, &rhs) {
                        "proved"
                    } else {
                        "inconclusive"
                    }
                }
                _ => "inconclusive",
            };
            std::fs::write(&outcome_path, outcome).expect("write add/sub child outcome");
        })
        .expect("spawn nf-first add/sub child worker");
    handle
        .join()
        .expect("nf-first add/sub child worker panicked");
}

#[test]
#[ignore]
fn metatest_child_nf_first_mul_div_classify() {
    let lhs =
        std::env::var(METATEST_CHILD_NF_MUL_DIV_EXP_ENV).expect("missing child mul/div lhs env");
    let rhs =
        std::env::var(METATEST_CHILD_NF_MUL_DIV_SIMP_ENV).expect("missing child mul/div rhs env");
    let vars = decode_child_vars(
        &std::env::var(METATEST_CHILD_NF_MUL_DIV_VARS_ENV).expect("missing child mul/div vars env"),
    );
    let filters = decode_child_filters(
        &std::env::var(METATEST_CHILD_NF_MUL_DIV_FILTERS_ENV)
            .expect("missing child mul/div filters env"),
    );
    let outcome_path = std::env::var(METATEST_CHILD_NF_MUL_DIV_OUTCOME_ENV)
        .expect("missing child mul/div outcome env");

    let handle = std::thread::Builder::new()
        .stack_size(METATEST_DEEP_WORKER_STACK_SIZE_BYTES)
        .spawn(move || {
            let mut engine = Engine::new();
            let simplifier = &mut engine.simplifier;
            let payload = match (
                parse(&lhs, &mut simplifier.context),
                parse(&rhs, &mut simplifier.context),
            ) {
                (Ok(lhs_parsed), Ok(rhs_parsed)) => {
                    let opts = cas_solver::runtime::SimplifyOptions::default();
                    let (mut lhs_simp, _, lhs_stats) =
                        simplifier.simplify_with_stats(lhs_parsed, opts.clone());
                    let (mut rhs_simp, _, rhs_stats) =
                        simplifier.simplify_with_stats(rhs_parsed, opts);
                    let cycles = lhs_stats.cycle_events.len() + rhs_stats.cycle_events.len();
                    lhs_simp = fold_constants_safe(&mut simplifier.context, lhs_simp);
                    rhs_simp = fold_constants_safe(&mut simplifier.context, rhs_simp);
                    if normal_forms_visibly_equal(&simplifier.context, lhs_simp, rhs_simp) {
                        serde_json::json!({ "kind": "nf", "cycles": cycles })
                    } else if prove_zero_from_engine_texts_child_hint(&lhs, &rhs) {
                        serde_json::json!({ "kind": "proved-d", "cycles": cycles })
                    } else {
                        let mut proved_kind: Option<&str> = None;

                        {
                            let q_str = format!("({}) / ({})", lhs, rhs);
                            let mut sq = Simplifier::with_default_rules();
                            if let Ok(qp) = parse(&q_str, &mut sq.context) {
                                let (mut qr, _) = sq.simplify(qp);
                                qr = fold_constants_safe(&mut sq.context, qr);
                                let target = num_rational::BigRational::from_integer(1.into());
                                if matches!(sq.context.get(qr), cas_ast::Expr::Number(n) if *n == target) {
                                    proved_kind = Some("proved-q");
                                }
                            }
                        }

                        if proved_kind.is_none()
                            && prove_zero_from_metamorphic_texts(
                                simplifier,
                                &lhs,
                                &rhs,
                                lhs_simp,
                                rhs_simp,
                            )
                        {
                            proved_kind = Some("proved-d");
                        }

                        if let Some(kind) = proved_kind {
                            serde_json::json!({ "kind": kind, "cycles": cycles })
                        } else {
                            let config = metatest_config();
                            match classify_numeric_equiv_for_vars(
                                &simplifier.context,
                                lhs_simp,
                                rhs_simp,
                                &vars,
                                &filters,
                                &config,
                            ) {
                                NumericCheckOutcome::Pass => {
                                    let d_diag = simplifier
                                        .context
                                        .add(cas_ast::Expr::Sub(lhs_simp, rhs_simp));
                                    let (d_simp, _) = simplifier.simplify(d_diag);
                                    let diff_str = format!(
                                        "simplify(LHS-RHS) => {}",
                                        cas_formatter::LaTeXExpr {
                                            context: &simplifier.context,
                                            id: d_simp
                                        }
                                        .to_latex()
                                    );
                                    let shape =
                                        expr_shape_signature(&simplifier.context, d_simp);
                                    let cause = numeric_only_cause_for_vars(
                                        &simplifier.context,
                                        lhs_simp,
                                        rhs_simp,
                                        &vars,
                                        &filters,
                                        &config,
                                        &shape,
                                    )
                                    .label()
                                    .to_string();
                                    if let Some(reason) =
                                        known_domain_frontier_reason_for_numeric_cause(
                                            &cause, &lhs, &rhs,
                                        )
                                    {
                                        serde_json::json!({
                                            "kind": "domain_frontier",
                                            "reason": reason,
                                            "shape": shape,
                                            "cause": cause,
                                            "cycles": cycles
                                        })
                                    } else {
                                        serde_json::json!({
                                            "kind": "numeric",
                                            "diff_str": diff_str,
                                            "shape": shape,
                                            "cause": cause,
                                            "cycles": cycles
                                        })
                                    }
                                }
                                NumericCheckOutcome::Inconclusive(reason) => serde_json::json!({
                                    "kind": "inconclusive",
                                    "reason": reason,
                                    "cycles": cycles
                                }),
                                NumericCheckOutcome::Failed(_) => serde_json::json!({
                                    "kind": "failed",
                                    "cycles": cycles
                                }),
                            }
                        }
                    }
                }
                _ => serde_json::json!({ "kind": "skip", "cycles": 0 }),
            };
            std::fs::write(&outcome_path, payload.to_string()).expect("write mul/div child outcome");
        })
        .expect("spawn nf-first mul/div child worker");
    handle
        .join()
        .expect("nf-first mul/div child worker panicked");
}

/// Classify an identity into a diagnostic category
///
/// Precedence (highest to lowest):
/// 1. BugSignal: asymmetric_invalid > 0
/// 2. ConfigError: eval_failed_rate > 50% (likely unbound variable)
/// 3. NeedsFilter: domain_rate > 20%
/// 4. Fragile: pole_rate > 15%
/// 5. Ok: everything else
#[allow(dead_code)]
pub(super) fn classify_diagnostic(stats: &NumericEquivStats) -> DiagCategory {
    // Priority 1: BugSignal (asymmetric failures indicate potential engine bugs)
    if stats.asymmetric_invalid > 0 {
        return DiagCategory::BugSignal;
    }

    // Priority 2: ConfigError (high eval_failed usually means unbound variable)
    if stats.eval_failed_rate() > EVAL_FAILED_THRESHOLD {
        return DiagCategory::ConfigError;
    }

    // Priority 3: NeedsFilter (high domain_error means function called outside domain)
    if stats.domain_rate() > DOMAIN_ERROR_THRESHOLD {
        return DiagCategory::NeedsFilter;
    }

    // Priority 4: Fragile (high pole_rate means near singularities)
    if stats.pole_rate() > POLE_RATE_THRESHOLD {
        return DiagCategory::Fragile;
    }

    // Priority 5: Ok
    DiagCategory::Ok
}

pub(super) fn classify_numeric_only_cause(
    stats: Option<&NumericEquivStats>,
    free_var_count: usize,
    residual_shape: &str,
) -> NumericOnlyCause {
    if let Some(stats) = stats {
        match classify_diagnostic(stats) {
            DiagCategory::NeedsFilter => return NumericOnlyCause::DomainSensitive,
            DiagCategory::Fragile | DiagCategory::ConfigError | DiagCategory::BugSignal => {
                return NumericOnlyCause::SamplingWeak;
            }
            DiagCategory::Ok => {}
        }
    }

    if free_var_count >= 2 {
        return NumericOnlyCause::MultivarContext;
    }

    if shape_has_div(residual_shape) || shape_has_neg_exp(residual_shape) {
        return NumericOnlyCause::SymbolicResidual;
    }

    NumericOnlyCause::SymbolicResidual
}

pub(super) fn numeric_only_cause_for_1var(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    var: &str,
    config: &MetatestConfig,
    filter: &FilterSpec,
    residual_shape: &str,
) -> NumericOnlyCause {
    let stats = check_numeric_equiv_1var_stats(ctx, a, b, var, config, filter);
    classify_numeric_only_cause(Some(&stats), 1, residual_shape)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn numeric_only_cause_for_2var(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    var1: &str,
    var2: &str,
    config: &MetatestConfig,
    filter1: &FilterSpec,
    filter2: &FilterSpec,
    residual_shape: &str,
) -> NumericOnlyCause {
    let stats = check_numeric_equiv_2var_stats(ctx, a, b, var1, var2, config, filter1, filter2);
    classify_numeric_only_cause(Some(&stats), 2, residual_shape)
}

pub(super) fn print_numeric_only_cause_breakdown(counts: &HashMap<String, usize>) {
    if counts.is_empty() {
        return;
    }

    eprintln!("   🧭 Numeric-only by cause:");
    let mut sorted: Vec<_> = counts.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
    for (cause, count) in sorted {
        eprintln!("      - {}: {}", cause, count);
    }
}

/// Check numeric equivalence with BranchMode support
///
/// This is the unified branch-aware numeric equivalence checker.
/// - PrincipalStrict: direct comparison with atol/rtol
/// - ModuloPi: compare modulo π (for arctan identities)
/// - Modulo2Pi: compare modulo 2π (for trig identities)
/// - PrincipalWithFilter: direct comparison but requires filter (panics if None)
#[allow(dead_code)]
fn check_numeric_equiv_branch_1var<F>(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    var: &str,
    branch_mode: BranchMode,
    config: &MetatestConfig,
    filter: Option<F>,
) -> NumericEquivStats
where
    F: Fn(f64) -> bool,
{
    use std::f64::consts::PI;

    // PrincipalWithFilter requires a filter
    if branch_mode == BranchMode::PrincipalWithFilter && filter.is_none() {
        panic!("PrincipalWithFilter mode requires a non-None filter");
    }

    let (lo, hi) = config.sample_range;
    let mut stats = NumericEquivStats::default();

    let opts = EvalCheckedOptions {
        zero_abs_eps: 1e-12,
        zero_rel_eps: 1e-12,
        trig_pole_eps: 1e-9,
        max_depth: 200,
    };

    for i in 0..config.eval_samples {
        let t = (i as f64 + 0.5) / config.eval_samples as f64;
        let x = lo + (hi - lo) * t;

        // Apply optional filter
        if let Some(ref f) = filter {
            if !f(x) {
                stats.filtered_out += 1;
                continue;
            }
        }

        let mut var_map = HashMap::new();
        var_map.insert(var.to_string(), x);

        let va = eval_f64_checked(ctx, a, &var_map, &opts);
        let vb = eval_f64_checked(ctx, b, &var_map, &opts);

        match (&va, &vb) {
            (Ok(va), Ok(vb)) => {
                // Choose comparison method based on branch mode
                let is_equal = match branch_mode {
                    BranchMode::PrincipalStrict | BranchMode::PrincipalWithFilter => {
                        let diff = (va - vb).abs();
                        let scale = va.abs().max(vb.abs()).max(1.0);
                        let allowed = config.atol + config.rtol * scale;
                        diff <= allowed
                    }
                    BranchMode::ModuloPi => {
                        approx_eq_mod_period(*va, *vb, PI, config.atol, config.rtol)
                    }
                    BranchMode::Modulo2Pi => {
                        approx_eq_mod_period(*va, *vb, 2.0 * PI, config.atol, config.rtol)
                    }
                };

                if is_equal {
                    stats.valid += 1;
                } else {
                    stats.record_mismatch(x, *va, *vb, var);
                }
            }
            // Symmetric failures
            (Err(EvalCheckedError::NearPole { .. }), Err(EvalCheckedError::NearPole { .. })) => {
                stats.near_pole += 1;
            }
            (Err(EvalCheckedError::Domain { .. }), Err(EvalCheckedError::Domain { .. })) => {
                stats.domain_error += 1;
            }
            // Asymmetric: one Ok, one Err (suspicious)
            (Ok(_), Err(_)) | (Err(_), Ok(_)) => {
                stats.asymmetric_invalid += 1;
            }
            _ => {
                stats.eval_failed += 1;
            }
        }
    }

    stats
}

pub(super) fn should_promote_numeric_to_composed(
    op: CombineOp,
    pair_composed_ok: bool,
    cause: &str,
) -> bool {
    matches!(op, CombineOp::Add | CombineOp::Sub | CombineOp::Mul)
        && pair_composed_ok
        && matches!(cause, "multivar-context" | "sampling-weak")
}

#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate_add_sub_combo(
    combined_exp: &str,
    combined_simp: &str,
    combined_vars: &[String],
    combined_filters: &[FilterSpec],
    config: &MetatestConfig,
    verbose: bool,
    combo_timeout: std::time::Duration,
    op: CombineOp,
    pair_composed_ok: bool,
    shortcut_mode: MetatestShortcutMode,
) -> (String, String, String, String, usize) {
    if shortcut_mode.allows_pre_nf_proof_shortcuts()
        && prove_zero_from_additive_abs_square_passthrough_text(combined_exp, combined_simp)
    {
        return (
            "proved".to_string(),
            String::new(),
            String::new(),
            String::new(),
            0,
        );
    }

    let mut simplifier = Simplifier::with_default_rules();
    let exp_parsed = match parse(combined_exp, &mut simplifier.context) {
        Ok(e) => e,
        Err(_) => {
            return (
                "skip".to_string(),
                String::new(),
                String::new(),
                String::new(),
                0,
            );
        }
    };
    let simp_parsed = match parse(combined_simp, &mut simplifier.context) {
        Ok(e) => e,
        Err(_) => {
            return (
                "skip".to_string(),
                String::new(),
                String::new(),
                String::new(),
                0,
            );
        }
    };

    let combo_start = std::time::Instant::now();
    let mut inline_cycles: usize = 0;
    let (exp_simplified, simp_simplified) = {
        let opts = cas_solver::runtime::SimplifyOptions::default();
        let (mut e, _, stats_e) = simplifier.simplify_with_stats(exp_parsed, opts.clone());
        inline_cycles += stats_e.cycle_events.len();
        {
            let cfg = cas_solver::runtime::EvalConfig::default();
            let mut budget = cas_solver::runtime::Budget::preset_cli();
            if let Ok(r) = cas_solver::api::fold_constants(
                &mut simplifier.context,
                e,
                &cfg,
                cas_solver::api::ConstFoldMode::Safe,
                &mut budget,
            ) {
                e = r.expr;
            }
        }
        if combo_start.elapsed() > combo_timeout {
            return (
                "timeout".to_string(),
                String::new(),
                String::new(),
                String::new(),
                inline_cycles,
            );
        }
        let (mut s, _, stats_s) = simplifier.simplify_with_stats(simp_parsed, opts);
        inline_cycles += stats_s.cycle_events.len();
        {
            let cfg = cas_solver::runtime::EvalConfig::default();
            let mut budget = cas_solver::runtime::Budget::preset_cli();
            if let Ok(r) = cas_solver::api::fold_constants(
                &mut simplifier.context,
                s,
                &cfg,
                cas_solver::api::ConstFoldMode::Safe,
                &mut budget,
            ) {
                s = r.expr;
            }
        }
        (e, s)
    };
    if combo_start.elapsed() > combo_timeout {
        return (
            "timeout".to_string(),
            String::new(),
            String::new(),
            String::new(),
            inline_cycles,
        );
    }

    let nf_match =
        cas_solver::runtime::compare_expr(&simplifier.context, exp_simplified, simp_simplified)
            == std::cmp::Ordering::Equal;

    if nf_match {
        return (
            "nf".to_string(),
            String::new(),
            String::new(),
            String::new(),
            inline_cycles,
        );
    }

    let diff_simplified = {
        let diff_str = format!("({}) - ({})", combined_exp, combined_simp);
        let mut sd = Simplifier::with_default_rules();
        if let Ok(dp) = parse(&diff_str, &mut sd.context) {
            let (mut dr, _) = sd.simplify(dp);
            let cfg = cas_solver::runtime::EvalConfig::default();
            let mut budget = cas_solver::runtime::Budget::preset_cli();
            if let Ok(r) = cas_solver::api::fold_constants(
                &mut sd.context,
                dr,
                &cfg,
                cas_solver::api::ConstFoldMode::Safe,
                &mut budget,
            ) {
                dr = r.expr;
            }
            let zero = num_rational::BigRational::from_integer(0.into());
            if matches!(sd.context.get(dr), cas_ast::Expr::Number(n) if *n == zero) {
                return (
                    "proved".to_string(),
                    String::new(),
                    String::new(),
                    String::new(),
                    inline_cycles,
                );
            }
        }

        let d = simplifier
            .context
            .add(cas_ast::Expr::Sub(exp_simplified, simp_simplified));
        let (mut ds, _) = simplifier.simplify(d);
        {
            let cfg = cas_solver::runtime::EvalConfig::default();
            let mut budget = cas_solver::runtime::Budget::preset_cli();
            if let Ok(r) = cas_solver::api::fold_constants(
                &mut simplifier.context,
                ds,
                &cfg,
                cas_solver::api::ConstFoldMode::Safe,
                &mut budget,
            ) {
                ds = r.expr;
            }
        }
        let target_value = num_rational::BigRational::from_integer(0.into());
        if matches!(simplifier.context.get(ds), cas_ast::Expr::Number(n) if *n == target_value) {
            return (
                "proved".to_string(),
                String::new(),
                String::new(),
                String::new(),
                inline_cycles,
            );
        }
        ds
    };

    if shortcut_mode.allows_curated_shortcuts()
        && prove_zero_from_metamorphic_texts(
            &mut simplifier,
            combined_exp,
            combined_simp,
            exp_simplified,
            simp_simplified,
        )
    {
        return (
            "proved".to_string(),
            String::new(),
            String::new(),
            String::new(),
            inline_cycles,
        );
    }

    match classify_numeric_equiv_for_vars(
        &simplifier.context,
        exp_simplified,
        simp_simplified,
        combined_vars,
        combined_filters,
        config,
    ) {
        NumericCheckOutcome::Pass => {
            let diff_str = if verbose {
                format!(
                    "simplify(LHS-RHS) => {}",
                    cas_formatter::LaTeXExpr {
                        context: &simplifier.context,
                        id: diff_simplified
                    }
                    .to_latex()
                )
            } else {
                String::new()
            };
            let shape = if verbose {
                expr_shape_signature(&simplifier.context, diff_simplified)
            } else {
                String::new()
            };
            let cause = numeric_only_cause_for_vars(
                &simplifier.context,
                exp_simplified,
                simp_simplified,
                combined_vars,
                combined_filters,
                config,
                &shape,
            )
            .label()
            .to_string();
            if should_promote_numeric_to_composed(op, pair_composed_ok, &cause) {
                (
                    "proved-composed".to_string(),
                    String::new(),
                    String::new(),
                    String::new(),
                    inline_cycles,
                )
            } else if let Some(reason) =
                known_domain_frontier_reason_for_numeric_cause(&cause, combined_exp, combined_simp)
            {
                (
                    "domain_frontier".to_string(),
                    reason.to_string(),
                    shape,
                    cause,
                    inline_cycles,
                )
            } else {
                ("numeric".to_string(), diff_str, shape, cause, inline_cycles)
            }
        }
        NumericCheckOutcome::Inconclusive(reason) => {
            if pair_composed_ok {
                (
                    "proved-composed".to_string(),
                    String::new(),
                    String::new(),
                    String::new(),
                    inline_cycles,
                )
            } else {
                (
                    "inconclusive".to_string(),
                    reason,
                    String::new(),
                    String::new(),
                    inline_cycles,
                )
            }
        }
        NumericCheckOutcome::Failed(_) => {
            if pair_composed_ok {
                (
                    "proved-composed".to_string(),
                    String::new(),
                    String::new(),
                    String::new(),
                    inline_cycles,
                )
            } else {
                (
                    "failed".to_string(),
                    String::new(),
                    String::new(),
                    String::new(),
                    inline_cycles,
                )
            }
        }
    }
}

/// Quick numeric equivalence check (5 sample points)
pub(super) fn check_numeric_equiv_quick(ctx: &Context, a: ExprId, b: ExprId, var: &str) -> bool {
    let samples = [-2.0, -0.5, 0.5, 1.5, 3.0];
    let mut valid_checks = 0;
    let mut matching = 0;

    for x in samples {
        let mut vars = HashMap::new();
        vars.insert(var.to_string(), x);

        let va = eval_f64(ctx, a, &vars);
        let vb = eval_f64(ctx, b, &vars);

        match (va, vb) {
            (Some(a_val), Some(b_val)) if a_val.is_finite() && b_val.is_finite() => {
                valid_checks += 1;
                let diff = (a_val - b_val).abs();
                let rel = diff / a_val.abs().max(1e-10);
                if diff < 1e-8 || rel < 1e-8 {
                    matching += 1;
                }
            }
            _ => {} // Skip invalid samples
        }
    }

    // If no valid samples, return true (inconclusive, not a failure)
    // If at least 2 valid samples, require all to match
    if valid_checks < 2 {
        true // Inconclusive - skip this check
    } else {
        matching == valid_checks
    }
}

fn load_eval_path_behavior_contract_expressions() -> Vec<EvalPathBehaviorContractExpr> {
    let csv_path = find_test_data_file("eval_path_behavior_contract_expressions.csv");
    let content = std::fs::read_to_string(csv_path)
        .expect("Failed to read eval_path_behavior_contract_expressions.csv");

    let mut exprs = Vec::new();
    let mut current_family = String::from("Uncategorized");
    for (line_idx, line) in content.lines().enumerate() {
        let line_num = line_idx + 1;
        let line = line.trim();
        if line.starts_with('#') {
            let label = line.trim_start_matches('#').trim();
            if !label.is_empty()
                && !label.starts_with("Format")
                && !label.starts_with("Expressions")
            {
                current_family = label.to_string();
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.rsplitn(7, ',').collect();
        if parts.len() != 7 {
            panic!(
                "eval_path_behavior_contract_expressions.csv line {}: expected expr,value_domain,mode,complex_mode,const_fold_mode,match_kind,expected. Line: '{}'",
                line_num, line
            );
        }

        let expr = parts[6].trim().to_string();
        let value_domain = parse_value_domain_label(
            parts[5],
            "eval_path_behavior_contract_expressions.csv",
            line_num,
        );
        let mode = parse_domain_mode_label(
            parts[4],
            "eval_path_behavior_contract_expressions.csv",
            line_num,
        );
        let complex_mode = parse_complex_mode_label(
            parts[3],
            "eval_path_behavior_contract_expressions.csv",
            line_num,
        );
        let const_fold_mode = parse_const_fold_mode_label(
            parts[2],
            "eval_path_behavior_contract_expressions.csv",
            line_num,
        );
        let expectation = match parts[1].trim().to_lowercase().as_str() {
            "exact" => SemanticBehaviorExpectation::Exact(parts[0].trim().to_string()),
            "contains_all" => SemanticBehaviorExpectation::ContainsAll(
                parts[0]
                    .split(';')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect(),
            ),
            other => panic!(
                "eval_path_behavior_contract_expressions.csv line {}: invalid match_kind '{}'",
                line_num, other
            ),
        };

        exprs.push(EvalPathBehaviorContractExpr {
            expr,
            value_domain,
            mode,
            complex_mode,
            const_fold_mode,
            expectation,
            family: current_family.clone(),
        });
    }

    exprs
}

fn load_eval_path_axes_contract_expressions() -> Vec<EvalPathAxesContractExpr> {
    let csv_path = find_test_data_file("eval_path_axes_contract_expressions.csv");
    let content = std::fs::read_to_string(csv_path)
        .expect("Failed to read eval_path_axes_contract_expressions.csv");

    let mut exprs = Vec::new();
    let mut current_family = String::from("Uncategorized");
    for (line_idx, line) in content.lines().enumerate() {
        let line_num = line_idx + 1;
        let line = line.trim();
        if line.starts_with('#') {
            let label = line.trim_start_matches('#').trim();
            if !label.is_empty()
                && !label.starts_with("Format")
                && !label.starts_with("Expressions")
            {
                current_family = label.to_string();
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.rsplitn(7, ',').collect();
        if parts.len() != 7 {
            panic!(
                "eval_path_axes_contract_expressions.csv line {}: expected expr,value_domain,mode,complex_mode,const_fold_mode,expect_requires,expect_warning. Line: '{}'",
                line_num, line
            );
        }

        let expr = parts[6].trim().to_string();
        let value_domain = parse_value_domain_label(
            parts[5],
            "eval_path_axes_contract_expressions.csv",
            line_num,
        );
        let mode = parse_domain_mode_label(
            parts[4],
            "eval_path_axes_contract_expressions.csv",
            line_num,
        );
        let complex_mode = parse_complex_mode_label(
            parts[3],
            "eval_path_axes_contract_expressions.csv",
            line_num,
        );
        let const_fold_mode = parse_const_fold_mode_label(
            parts[2],
            "eval_path_axes_contract_expressions.csv",
            line_num,
        );
        let expect_requires = match parts[1].trim().to_lowercase().as_str() {
            "yes" | "true" => true,
            "no" | "false" => false,
            other => panic!(
                "eval_path_axes_contract_expressions.csv line {}: invalid expect_requires '{}'",
                line_num, other
            ),
        };
        let expect_warning = match parts[0].trim().to_lowercase().as_str() {
            "yes" | "true" => true,
            "no" | "false" => false,
            other => panic!(
                "eval_path_axes_contract_expressions.csv line {}: invalid expect_warning '{}'",
                line_num, other
            ),
        };

        exprs.push(EvalPathAxesContractExpr {
            expr,
            value_domain,
            mode,
            complex_mode,
            const_fold_mode,
            expect_requires,
            expect_warning,
            family: current_family.clone(),
        });
    }

    exprs
}

fn load_eval_path_inv_trig_axes_contract_expressions() -> Vec<EvalPathInvTrigAxesContractExpr> {
    let csv_path = find_test_data_file("eval_path_inv_trig_axes_contract_expressions.csv");
    let content = std::fs::read_to_string(csv_path)
        .expect("Failed to read eval_path_inv_trig_axes_contract_expressions.csv");

    let mut exprs = Vec::new();
    let mut current_family = String::from("Uncategorized");
    for (line_idx, line) in content.lines().enumerate() {
        let line_num = line_idx + 1;
        let line = line.trim();
        if line.starts_with('#') {
            let label = line.trim_start_matches('#').trim();
            if !label.is_empty()
                && !label.starts_with("Format")
                && !label.starts_with("Expressions")
            {
                current_family = label.to_string();
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.rsplitn(6, ',').collect();
        if parts.len() != 6 {
            panic!(
                "eval_path_inv_trig_axes_contract_expressions.csv line {}: expected expr,value_domain,mode,inv_trig,expect_requires,expect_warning. Line: '{}'",
                line_num, line
            );
        }

        let expr = parts[5].trim().to_string();
        let value_domain = parse_value_domain_label(
            parts[4],
            "eval_path_inv_trig_axes_contract_expressions.csv",
            line_num,
        );
        let mode = parse_domain_mode_label(
            parts[3],
            "eval_path_inv_trig_axes_contract_expressions.csv",
            line_num,
        );
        let inv_trig = parse_inv_trig_policy_label(
            parts[2],
            "eval_path_inv_trig_axes_contract_expressions.csv",
            line_num,
        );
        let expect_requires = match parts[1].trim().to_lowercase().as_str() {
            "yes" | "true" => true,
            "no" | "false" => false,
            other => panic!(
                "eval_path_inv_trig_axes_contract_expressions.csv line {}: invalid expect_requires '{}'",
                line_num, other
            ),
        };
        let expect_warning = match parts[0].trim().to_lowercase().as_str() {
            "yes" | "true" => true,
            "no" | "false" => false,
            other => panic!(
                "eval_path_inv_trig_axes_contract_expressions.csv line {}: invalid expect_warning '{}'",
                line_num, other
            ),
        };

        exprs.push(EvalPathInvTrigAxesContractExpr {
            expr,
            value_domain,
            mode,
            inv_trig,
            expect_requires,
            expect_warning,
            family: current_family.clone(),
        });
    }

    exprs
}

fn simplify_with_eval_path_behavior(
    input: &str,
    mode: cas_solver::runtime::DomainMode,
    value_domain: cas_solver::runtime::ValueDomain,
    complex_mode: cas_solver::runtime::ComplexMode,
    const_fold_mode: cas_solver::api::ConstFoldMode,
) -> Result<String, String> {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().shared.semantics.domain_mode = mode;
    state.options_mut().shared.semantics.value_domain = value_domain;
    state.options_mut().complex_mode = complex_mode;
    state.options_mut().const_fold = const_fold_mode;

    let parsed = parse(input, &mut engine.simplifier.context)
        .map_err(|e| format!("parse failed for '{}': {:?}", input, e))?;
    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine
        .eval(&mut state, req)
        .map_err(|e| format!("eval failed for '{}': {:?}", input, e))?;

    let result = match &output.result {
        EvalResult::Expr(e) => DisplayExpr {
            context: &engine.simplifier.context,
            id: *e,
        }
        .to_string(),
        other => {
            return Err(format!(
                "unexpected eval result for '{}': {:?}",
                input, other
            ));
        }
    };

    Ok(result)
}

fn simplify_with_eval_path_metadata(
    input: &str,
    mode: cas_solver::runtime::DomainMode,
    value_domain: cas_solver::runtime::ValueDomain,
    complex_mode: cas_solver::runtime::ComplexMode,
    const_fold_mode: cas_solver::api::ConstFoldMode,
) -> Result<SimplifyMetadata, String> {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().shared.semantics.domain_mode = mode;
    state.options_mut().shared.semantics.value_domain = value_domain;
    state.options_mut().complex_mode = complex_mode;
    state.options_mut().const_fold = const_fold_mode;

    let parsed = parse(input, &mut engine.simplifier.context)
        .map_err(|e| format!("parse failed for '{}': {:?}", input, e))?;
    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine
        .eval(&mut state, req)
        .map_err(|e| format!("eval failed for '{}': {:?}", input, e))?;

    let result = match &output.result {
        EvalResult::Expr(e) => DisplayExpr {
            context: &engine.simplifier.context,
            id: *e,
        }
        .to_string(),
        other => {
            return Err(format!(
                "unexpected eval result for '{}': {:?}",
                input, other
            ));
        }
    };

    let mut required: Vec<String> = output
        .required_conditions
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();
    required.sort();
    required.dedup();

    let mut warnings: Vec<String> = output
        .domain_warnings
        .iter()
        .map(|w| w.message.clone())
        .collect();
    warnings.sort();
    warnings.dedup();

    Ok(SimplifyMetadata {
        result,
        required,
        warnings,
    })
}

fn simplify_with_eval_path_metadata_and_inv_trig(
    input: &str,
    mode: cas_solver::runtime::DomainMode,
    value_domain: cas_solver::runtime::ValueDomain,
    inv_trig: cas_solver::runtime::InverseTrigPolicy,
) -> Result<SimplifyMetadata, String> {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().shared.semantics.domain_mode = mode;
    state.options_mut().shared.semantics.value_domain = value_domain;
    state.options_mut().shared.semantics.inv_trig = inv_trig;

    let parsed = parse(input, &mut engine.simplifier.context)
        .map_err(|e| format!("parse failed for '{}': {:?}", input, e))?;
    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine
        .eval(&mut state, req)
        .map_err(|e| format!("eval failed for '{}': {:?}", input, e))?;

    let result = match &output.result {
        EvalResult::Expr(e) => DisplayExpr {
            context: &engine.simplifier.context,
            id: *e,
        }
        .to_string(),
        other => {
            return Err(format!(
                "unexpected eval result for '{}': {:?}",
                input, other
            ));
        }
    };

    let mut required: Vec<String> = output
        .required_conditions
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();
    required.sort();
    required.dedup();

    let mut warnings: Vec<String> = output
        .domain_warnings
        .iter()
        .map(|w| w.message.clone())
        .collect();
    warnings.sort();
    warnings.dedup();

    Ok(SimplifyMetadata {
        result,
        required,
        warnings,
    })
}

pub(super) fn run_eval_path_behavior_contract_tests() -> EvalPathBehaviorContractMetrics {
    let cases = load_eval_path_behavior_contract_expressions();
    let verbose = std::env::var("METATEST_VERBOSE").is_ok();
    let mut metrics = EvalPathBehaviorContractMetrics {
        total: cases.len(),
        ..Default::default()
    };
    let mut failures: Vec<String> = Vec::new();
    let mut relaxed_examples: Vec<String> = Vec::new();

    eprintln!(
        "📊 Running eval-path behavior contracts: {} expressions from {} families",
        cases.len(),
        cases
            .iter()
            .map(|c| &c.family)
            .collect::<std::collections::HashSet<_>>()
            .len()
    );

    for case in &cases {
        let first = match simplify_with_eval_path_behavior(
            &case.expr,
            case.mode,
            case.value_domain,
            case.complex_mode,
            case.const_fold_mode,
        ) {
            Ok(v) => v,
            Err(err) => {
                metrics.parse_errors += 1;
                failures.push(format!(
                    "[{}|{}|{}|{}|{}] {} — {}",
                    case.family,
                    value_domain_label(case.value_domain),
                    domain_mode_label(case.mode),
                    complex_mode_label(case.complex_mode),
                    const_fold_mode_label(case.const_fold_mode),
                    case.expr,
                    err
                ));
                continue;
            }
        };

        let second = match simplify_with_eval_path_behavior(
            &first,
            case.mode,
            case.value_domain,
            case.complex_mode,
            case.const_fold_mode,
        ) {
            Ok(v) => v,
            Err(err) => {
                metrics.parse_errors += 1;
                failures.push(format!(
                    "[{}|{}|{}|{}|{}] {} -> '{}' reparsed failed: {}",
                    case.family,
                    value_domain_label(case.value_domain),
                    domain_mode_label(case.mode),
                    complex_mode_label(case.complex_mode),
                    const_fold_mode_label(case.const_fold_mode),
                    case.expr,
                    first,
                    err
                ));
                continue;
            }
        };

        let expected_ok = semantic_behavior_matches(&case.expectation, &first);
        let second_ok = semantic_behavior_matches(&case.expectation, &second);

        if !expected_ok {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}|{}|{}|{}] {} — expected {}, got '{}'",
                case.family,
                value_domain_label(case.value_domain),
                domain_mode_label(case.mode),
                complex_mode_label(case.complex_mode),
                const_fold_mode_label(case.const_fold_mode),
                case.expr,
                semantic_behavior_label(&case.expectation),
                first
            ));
            continue;
        }

        if !second_ok {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}|{}|{}|{}] {} — second eval simplify broke behavior: first='{}', second='{}', expected {}",
                case.family,
                value_domain_label(case.value_domain),
                domain_mode_label(case.mode),
                complex_mode_label(case.complex_mode),
                const_fold_mode_label(case.const_fold_mode),
                case.expr,
                first,
                second,
                semantic_behavior_label(&case.expectation)
            ));
            continue;
        }

        if first == second {
            metrics.exact_preserved += 1;
        } else {
            metrics.relaxed_preserved += 1;
            if verbose && relaxed_examples.len() < 10 {
                relaxed_examples.push(format!(
                    "[{}|{}|{}|{}|{}] {} — result '{}' -> '{}'",
                    case.family,
                    value_domain_label(case.value_domain),
                    domain_mode_label(case.mode),
                    complex_mode_label(case.complex_mode),
                    const_fold_mode_label(case.const_fold_mode),
                    case.expr,
                    first,
                    second
                ));
            }
        }
    }

    eprintln!(
        "✅ Eval-path behavior contracts: exact={} relaxed={} failed={} parse={}",
        metrics.exact_preserved, metrics.relaxed_preserved, metrics.failed, metrics.parse_errors
    );

    if verbose && !relaxed_examples.is_empty() {
        eprintln!("\nℹ️ eval-path behavior relaxed-preserved examples:");
        for example in relaxed_examples.iter().take(10) {
            eprintln!("  {}", example);
        }
    }

    if !failures.is_empty() {
        eprintln!("\n🚨 eval-path behavior failures:");
        for failure in failures.iter().take(10) {
            eprintln!("  {}", failure);
        }
    }

    metrics
}

pub(super) fn run_eval_path_axes_contract_tests() -> EvalPathAxesContractMetrics {
    let cases = load_eval_path_axes_contract_expressions();
    let verbose = std::env::var("METATEST_VERBOSE").is_ok();
    let mut metrics = EvalPathAxesContractMetrics {
        total: cases.len(),
        ..Default::default()
    };
    let mut failures: Vec<String> = Vec::new();
    let mut relaxed_examples: Vec<String> = Vec::new();

    eprintln!(
        "📊 Running eval-path axes contracts: {} expressions from {} families",
        cases.len(),
        cases
            .iter()
            .map(|c| &c.family)
            .collect::<std::collections::HashSet<_>>()
            .len()
    );

    for case in &cases {
        let first = match simplify_with_eval_path_metadata(
            &case.expr,
            case.mode,
            case.value_domain,
            case.complex_mode,
            case.const_fold_mode,
        ) {
            Ok(v) => v,
            Err(err) => {
                metrics.parse_errors += 1;
                failures.push(format!(
                    "[{}|{}|{}|{}|{}] {} — {}",
                    case.family,
                    value_domain_label(case.value_domain),
                    domain_mode_label(case.mode),
                    complex_mode_label(case.complex_mode),
                    const_fold_mode_label(case.const_fold_mode),
                    case.expr,
                    err
                ));
                continue;
            }
        };

        let second = match simplify_with_eval_path_metadata(
            &first.result,
            case.mode,
            case.value_domain,
            case.complex_mode,
            case.const_fold_mode,
        ) {
            Ok(v) => v,
            Err(err) => {
                metrics.parse_errors += 1;
                failures.push(format!(
                    "[{}|{}|{}|{}|{}] {} -> '{}' reparsed failed: {}",
                    case.family,
                    value_domain_label(case.value_domain),
                    domain_mode_label(case.mode),
                    complex_mode_label(case.complex_mode),
                    const_fold_mode_label(case.const_fold_mode),
                    case.expr,
                    first.result,
                    err
                ));
                continue;
            }
        };

        let mut case_failed = false;
        if case.expect_requires {
            if first.required.is_empty() {
                metrics.failed += 1;
                failures.push(format!(
                    "[{}|{}|{}|{}|{}] {} — expected requires, got none",
                    case.family,
                    value_domain_label(case.value_domain),
                    domain_mode_label(case.mode),
                    complex_mode_label(case.complex_mode),
                    const_fold_mode_label(case.const_fold_mode),
                    case.expr
                ));
                case_failed = true;
            } else {
                metrics.expected_requires_present += 1;
            }
        } else if first.required.is_empty() {
            metrics.expected_requires_absent += 1;
        } else {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}|{}|{}|{}] {} — unexpected requires: {:?}",
                case.family,
                value_domain_label(case.value_domain),
                domain_mode_label(case.mode),
                complex_mode_label(case.complex_mode),
                const_fold_mode_label(case.const_fold_mode),
                case.expr,
                first.required
            ));
            case_failed = true;
        }

        if case.expect_warning {
            if first.warnings.is_empty() {
                metrics.failed += 1;
                failures.push(format!(
                    "[{}|{}|{}|{}|{}] {} — expected warning, got none",
                    case.family,
                    value_domain_label(case.value_domain),
                    domain_mode_label(case.mode),
                    complex_mode_label(case.complex_mode),
                    const_fold_mode_label(case.const_fold_mode),
                    case.expr
                ));
                case_failed = true;
            } else {
                metrics.expected_warning_present += 1;
            }
        } else if first.warnings.is_empty() {
            metrics.expected_warning_absent += 1;
        } else {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}|{}|{}|{}] {} — unexpected warnings: {:?}",
                case.family,
                value_domain_label(case.value_domain),
                domain_mode_label(case.mode),
                complex_mode_label(case.complex_mode),
                const_fold_mode_label(case.const_fold_mode),
                case.expr,
                first.warnings
            ));
            case_failed = true;
        }

        let first_required: std::collections::HashSet<_> = first.required.iter().cloned().collect();
        let second_required: std::collections::HashSet<_> =
            second.required.iter().cloned().collect();
        let first_warnings: std::collections::HashSet<_> = first.warnings.iter().cloned().collect();
        let second_warnings: std::collections::HashSet<_> =
            second.warnings.iter().cloned().collect();

        let introduced_requires: Vec<_> = second_required
            .difference(&first_required)
            .cloned()
            .collect();
        let introduced_warnings: Vec<_> = second_warnings
            .difference(&first_warnings)
            .cloned()
            .collect();

        if !introduced_requires.is_empty() {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}|{}|{}|{}] {} — introduced requires: {:?} (first={:?}, second={:?})",
                case.family,
                value_domain_label(case.value_domain),
                domain_mode_label(case.mode),
                complex_mode_label(case.complex_mode),
                const_fold_mode_label(case.const_fold_mode),
                case.expr,
                introduced_requires,
                first.required,
                second.required
            ));
            case_failed = true;
        }

        if !introduced_warnings.is_empty() {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}|{}|{}|{}] {} — introduced warnings: {:?} (first={:?}, second={:?})",
                case.family,
                value_domain_label(case.value_domain),
                domain_mode_label(case.mode),
                complex_mode_label(case.complex_mode),
                const_fold_mode_label(case.const_fold_mode),
                case.expr,
                introduced_warnings,
                first.warnings,
                second.warnings
            ));
            case_failed = true;
        }

        if !case_failed {
            if first.required == second.required && first.warnings == second.warnings {
                metrics.exact_preserved += 1;
            } else {
                metrics.relaxed_preserved += 1;
                if verbose && relaxed_examples.len() < 10 {
                    relaxed_examples.push(format!(
                        "[{}|{}|{}|{}|{}] {} — requires {:?} -> {:?}, warnings {:?} -> {:?}",
                        case.family,
                        value_domain_label(case.value_domain),
                        domain_mode_label(case.mode),
                        complex_mode_label(case.complex_mode),
                        const_fold_mode_label(case.const_fold_mode),
                        case.expr,
                        first.required,
                        second.required,
                        first.warnings,
                        second.warnings
                    ));
                }
            }
        }
    }

    eprintln!(
        "✅ Eval-path axes contracts: exact={} relaxed={} requires_present={} requires_absent={} warning_present={} warning_absent={} failed={} parse={}",
        metrics.exact_preserved,
        metrics.relaxed_preserved,
        metrics.expected_requires_present,
        metrics.expected_requires_absent,
        metrics.expected_warning_present,
        metrics.expected_warning_absent,
        metrics.failed,
        metrics.parse_errors
    );

    if verbose && !relaxed_examples.is_empty() {
        eprintln!("\nℹ️ eval-path axes relaxed-preserved examples:");
        for example in relaxed_examples.iter().take(10) {
            eprintln!("  {}", example);
        }
    }

    if !failures.is_empty() {
        eprintln!("\n🚨 eval-path axes failures:");
        for failure in failures.iter().take(10) {
            eprintln!("  {}", failure);
        }
    }

    metrics
}

pub(super) fn run_eval_path_inv_trig_axes_contract_tests() -> EvalPathInvTrigAxesContractMetrics {
    let cases = load_eval_path_inv_trig_axes_contract_expressions();
    let verbose = std::env::var("METATEST_VERBOSE").is_ok();
    let mut metrics = EvalPathInvTrigAxesContractMetrics {
        total: cases.len(),
        ..Default::default()
    };
    let mut failures: Vec<String> = Vec::new();
    let mut relaxed_examples: Vec<String> = Vec::new();

    eprintln!(
        "📊 Running eval-path inv-trig axes contracts: {} expressions from {} families",
        cases.len(),
        cases
            .iter()
            .map(|c| &c.family)
            .collect::<std::collections::HashSet<_>>()
            .len()
    );

    for case in &cases {
        let first = match simplify_with_eval_path_metadata_and_inv_trig(
            &case.expr,
            case.mode,
            case.value_domain,
            case.inv_trig,
        ) {
            Ok(v) => v,
            Err(err) => {
                metrics.parse_errors += 1;
                failures.push(format!(
                    "[{}|{}|{}|{}] {} — {}",
                    case.family,
                    value_domain_label(case.value_domain),
                    domain_mode_label(case.mode),
                    inv_trig_policy_label(case.inv_trig),
                    case.expr,
                    err
                ));
                continue;
            }
        };

        let second = match simplify_with_eval_path_metadata_and_inv_trig(
            &first.result,
            case.mode,
            case.value_domain,
            case.inv_trig,
        ) {
            Ok(v) => v,
            Err(err) => {
                metrics.parse_errors += 1;
                failures.push(format!(
                    "[{}|{}|{}|{}] {} -> '{}' reparsed failed: {}",
                    case.family,
                    value_domain_label(case.value_domain),
                    domain_mode_label(case.mode),
                    inv_trig_policy_label(case.inv_trig),
                    case.expr,
                    first.result,
                    err
                ));
                continue;
            }
        };

        let mut case_failed = false;
        if case.expect_requires {
            if first.required.is_empty() {
                metrics.failed += 1;
                failures.push(format!(
                    "[{}|{}|{}|{}] {} — expected requires, got none",
                    case.family,
                    value_domain_label(case.value_domain),
                    domain_mode_label(case.mode),
                    inv_trig_policy_label(case.inv_trig),
                    case.expr
                ));
                case_failed = true;
            } else {
                metrics.expected_requires_present += 1;
            }
        } else if first.required.is_empty() {
            metrics.expected_requires_absent += 1;
        } else {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}|{}|{}] {} — unexpected requires: {:?}",
                case.family,
                value_domain_label(case.value_domain),
                domain_mode_label(case.mode),
                inv_trig_policy_label(case.inv_trig),
                case.expr,
                first.required
            ));
            case_failed = true;
        }

        if case.expect_warning {
            if first.warnings.is_empty() {
                metrics.failed += 1;
                failures.push(format!(
                    "[{}|{}|{}|{}] {} — expected warning, got none",
                    case.family,
                    value_domain_label(case.value_domain),
                    domain_mode_label(case.mode),
                    inv_trig_policy_label(case.inv_trig),
                    case.expr
                ));
                case_failed = true;
            } else {
                metrics.expected_warning_present += 1;
            }
        } else if first.warnings.is_empty() {
            metrics.expected_warning_absent += 1;
        } else {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}|{}|{}] {} — unexpected warnings: {:?}",
                case.family,
                value_domain_label(case.value_domain),
                domain_mode_label(case.mode),
                inv_trig_policy_label(case.inv_trig),
                case.expr,
                first.warnings
            ));
            case_failed = true;
        }

        let first_required: std::collections::HashSet<_> = first.required.iter().cloned().collect();
        let second_required: std::collections::HashSet<_> =
            second.required.iter().cloned().collect();
        let first_warnings: std::collections::HashSet<_> = first.warnings.iter().cloned().collect();
        let second_warnings: std::collections::HashSet<_> =
            second.warnings.iter().cloned().collect();

        let introduced_requires: Vec<_> = second_required
            .difference(&first_required)
            .cloned()
            .collect();
        let introduced_warnings: Vec<_> = second_warnings
            .difference(&first_warnings)
            .cloned()
            .collect();

        if !introduced_requires.is_empty() {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}|{}|{}] {} — introduced requires: {:?} (first={:?}, second={:?})",
                case.family,
                value_domain_label(case.value_domain),
                domain_mode_label(case.mode),
                inv_trig_policy_label(case.inv_trig),
                case.expr,
                introduced_requires,
                first.required,
                second.required
            ));
            case_failed = true;
        }

        if !introduced_warnings.is_empty() {
            metrics.failed += 1;
            failures.push(format!(
                "[{}|{}|{}|{}] {} — introduced warnings: {:?} (first={:?}, second={:?})",
                case.family,
                value_domain_label(case.value_domain),
                domain_mode_label(case.mode),
                inv_trig_policy_label(case.inv_trig),
                case.expr,
                introduced_warnings,
                first.warnings,
                second.warnings
            ));
            case_failed = true;
        }

        if !case_failed {
            if first.required == second.required && first.warnings == second.warnings {
                metrics.exact_preserved += 1;
            } else {
                metrics.relaxed_preserved += 1;
                if verbose && relaxed_examples.len() < 10 {
                    relaxed_examples.push(format!(
                        "[{}|{}|{}|{}] {} — requires {:?} -> {:?}, warnings {:?} -> {:?}",
                        case.family,
                        value_domain_label(case.value_domain),
                        domain_mode_label(case.mode),
                        inv_trig_policy_label(case.inv_trig),
                        case.expr,
                        first.required,
                        second.required,
                        first.warnings,
                        second.warnings
                    ));
                }
            }
        }
    }

    eprintln!(
        "✅ Eval-path inv-trig axes contracts: exact={} relaxed={} requires_present={} requires_absent={} warning_present={} warning_absent={} failed={} parse={}",
        metrics.exact_preserved,
        metrics.relaxed_preserved,
        metrics.expected_requires_present,
        metrics.expected_requires_absent,
        metrics.expected_warning_present,
        metrics.expected_warning_absent,
        metrics.failed,
        metrics.parse_errors
    );

    if verbose && !relaxed_examples.is_empty() {
        eprintln!("\nℹ️ eval-path inv-trig axes relaxed-preserved examples:");
        for example in relaxed_examples.iter().take(10) {
            eprintln!("  {}", example);
        }
    }

    if !failures.is_empty() {
        eprintln!("\n🚨 eval-path inv-trig axes failures:");
        for failure in failures.iter().take(10) {
            eprintln!("  {}", failure);
        }
    }

    metrics
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_simplify_eval_path_behavior_contracts -- --ignored --nocapture
fn metatest_simplify_eval_path_behavior_contracts() {
    let m = run_eval_path_behavior_contract_tests();
    assert_eq!(
        m.failed, 0,
        "{} eval-path behavior contracts failed",
        m.failed
    );
    assert_eq!(
        m.parse_errors, 0,
        "{} eval-path behavior contract expressions failed to parse/eval",
        m.parse_errors
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_simplify_eval_path_axes_contracts -- --ignored --nocapture
fn metatest_simplify_eval_path_axes_contracts() {
    let m = run_eval_path_axes_contract_tests();
    assert_eq!(m.failed, 0, "{} eval-path axes contracts failed", m.failed);
    assert_eq!(
        m.parse_errors, 0,
        "{} eval-path axes contract expressions failed to parse/eval",
        m.parse_errors
    );
}

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_simplification_tests metatest_simplify_eval_path_inv_trig_axes_contracts -- --ignored --nocapture
fn metatest_simplify_eval_path_inv_trig_axes_contracts() {
    let m = run_eval_path_inv_trig_axes_contract_tests();
    assert_eq!(
        m.failed, 0,
        "{} eval-path inv-trig axes contracts failed",
        m.failed
    );
    assert_eq!(
        m.parse_errors, 0,
        "{} eval-path inv-trig axes contract expressions failed to parse/eval",
        m.parse_errors
    );
}

#[test]
fn choose_numeric_sample_profile_order_prioritizes_positive_for_logs() {
    let mut ctx = Context::new();
    let lhs = parse("ln(x-1)", &mut ctx).expect("parse lhs");
    let rhs = parse("0", &mut ctx).expect("parse rhs");

    let order = choose_numeric_sample_profile_order_exprs(&ctx, lhs, rhs);
    assert_eq!(
        order,
        Some([
            NumericSampleProfile::Positive,
            NumericSampleProfile::General,
            NumericSampleProfile::Interior,
        ])
    );
}

#[test]
fn known_domain_frontier_requires_domain_sensitive_numeric_cause() {
    assert_eq!(
        known_domain_frontier_reason_for_numeric_cause(
            "domain-sensitive",
            "ln((2*u)^2)",
            "2*ln((2*u))"
        ),
        Some("log-square expansion changes domain")
    );
    assert_eq!(
        known_domain_frontier_reason_for_numeric_cause(
            "symbolic-residual",
            "ln((2*u)^2)",
            "2*ln((2*u))"
        ),
        None
    );
}

#[test]
fn known_domain_frontier_safe_runtime_breakdown_matches_expected_numeric_cause_counts() {
    let metrics = run_known_domain_frontier_safe_pair_tests();
    let pair_count = load_known_domain_frontier_safe_pairs().len();

    assert_eq!(metrics.failed, 0);
    assert_eq!(metrics.timeouts, 0);
    assert_eq!(metrics.inconclusive, 0);
    assert_eq!(
        metrics.nf_convergent + metrics.proved_symbolic(),
        pair_count
    );
    assert_eq!(metrics.numeric_only, 0);
    assert_eq!(metrics.numeric_only_cause_count("domain-sensitive"), 0);
    assert_eq!(metrics.numeric_only_causes.len(), 0);
}

#[test]
fn choose_numeric_sample_profile_order_prioritizes_interior_for_inverse_trig() {
    let mut ctx = Context::new();
    let lhs = parse("arcsin(x/2)", &mut ctx).expect("parse lhs");
    let rhs = parse("0", &mut ctx).expect("parse rhs");

    let order = choose_numeric_sample_profile_order_exprs(&ctx, lhs, rhs);
    assert_eq!(
        order,
        Some([
            NumericSampleProfile::Interior,
            NumericSampleProfile::Rational,
            NumericSampleProfile::Positive,
        ])
    );
}

#[test]
fn choose_numeric_sample_profile_order_prioritizes_rational_for_negative_power() {
    let mut ctx = Context::new();
    let lhs = parse("(x-1)^(-1/2)", &mut ctx).expect("parse lhs");
    let rhs = parse("0", &mut ctx).expect("parse rhs");

    let order = choose_numeric_sample_profile_order_exprs(&ctx, lhs, rhs);
    assert_eq!(
        order,
        Some([
            NumericSampleProfile::Positive,
            NumericSampleProfile::Rational,
            NumericSampleProfile::Interior,
        ])
    );
}

#[test]
fn collect_numeric_denominator_guards_finds_division_denominator() {
    let mut ctx = Context::new();
    let expr = parse("1/(x-1)", &mut ctx).expect("parse expr");
    let mut guards = Vec::new();
    collect_numeric_denominator_guards(&ctx, expr, &mut guards);
    assert_eq!(guards.len(), 1);

    let mut bad = HashMap::new();
    bad.insert("x".to_string(), 1.0);
    assert!(near_numeric_guard_zero(&ctx, guards[0], &bad));

    let mut good = HashMap::new();
    good.insert("x".to_string(), 3.0);
    assert!(!near_numeric_guard_zero(&ctx, guards[0], &good));
}

#[test]
fn collect_numeric_denominator_guards_finds_negative_power_base() {
    let mut ctx = Context::new();
    let expr = parse("(x-1)^(-1/2)", &mut ctx).expect("parse expr");
    let mut guards = Vec::new();
    collect_numeric_denominator_guards(&ctx, expr, &mut guards);
    assert_eq!(guards.len(), 1);

    let mut bad = HashMap::new();
    bad.insert("x".to_string(), 1.0);
    assert!(near_numeric_guard_zero(&ctx, guards[0], &bad));

    let mut good = HashMap::new();
    good.insert("x".to_string(), 3.0);
    assert!(!near_numeric_guard_zero(&ctx, guards[0], &good));
}

#[test]
fn collect_numeric_analytic_guards_finds_ln_argument_guard() {
    let mut ctx = Context::new();
    let expr = parse("ln(x-1)", &mut ctx).expect("parse expr");
    let mut guards = Vec::new();
    collect_numeric_analytic_guards(&ctx, expr, &mut guards);
    assert_eq!(guards.len(), 1);
    assert_eq!(guards[0].kind, NumericAnalyticGuardKind::Positive);

    let mut bad = HashMap::new();
    bad.insert("x".to_string(), 1.0);
    assert!(violates_numeric_analytic_guard(&ctx, guards[0], &bad));

    let mut good = HashMap::new();
    good.insert("x".to_string(), 3.0);
    assert!(!violates_numeric_analytic_guard(&ctx, guards[0], &good));
}

#[test]
fn collect_numeric_analytic_guards_finds_sqrt_nonnegative_guard() {
    let mut ctx = Context::new();
    let expr = parse("sqrt(x-1)", &mut ctx).expect("parse expr");
    let mut guards = Vec::new();
    collect_numeric_analytic_guards(&ctx, expr, &mut guards);
    assert_eq!(guards.len(), 1);
    assert_eq!(guards[0].kind, NumericAnalyticGuardKind::NonNegative);

    let mut bad = HashMap::new();
    bad.insert("x".to_string(), 0.0);
    assert!(violates_numeric_analytic_guard(&ctx, guards[0], &bad));

    let mut good = HashMap::new();
    good.insert("x".to_string(), 3.0);
    assert!(!violates_numeric_analytic_guard(&ctx, guards[0], &good));
}

#[test]
fn collect_numeric_analytic_guards_finds_inverse_trig_unit_interval_guard() {
    let mut ctx = Context::new();
    let expr = parse("arcsin(x/2)", &mut ctx).expect("parse expr");
    let mut guards = Vec::new();
    collect_numeric_analytic_guards(&ctx, expr, &mut guards);
    assert_eq!(guards.len(), 1);
    assert_eq!(guards[0].kind, NumericAnalyticGuardKind::UnitInterval);

    let mut bad = HashMap::new();
    bad.insert("x".to_string(), 3.0);
    assert!(violates_numeric_analytic_guard(&ctx, guards[0], &bad));

    let mut good = HashMap::new();
    good.insert("x".to_string(), 1.0);
    assert!(!violates_numeric_analytic_guard(&ctx, guards[0], &good));
}

#[test]
fn relaxed_numeric_classification_marks_fragile_stats_inconclusive() {
    let stats = NumericEquivStats {
        valid: 2,
        near_pole: 8,
        domain_error: 6,
        asymmetric_invalid: 0,
        eval_failed: 0,
        filtered_out: 0,
        mismatches: Vec::new(),
        max_abs_err: 0.0,
        max_rel_err: 0.0,
        worst_sample: None,
    };

    let outcome = classify_numeric_check_with_stats(
        Err("Too few valid samples: 2 < 10 (near_pole=8, domain_error=6, asymmetric=0, eval_failed=0)".to_string()),
        &stats,
    );

    assert!(matches!(outcome, NumericCheckOutcome::Inconclusive(_)));
}

#[test]
fn relaxed_numeric_classification_keeps_true_mismatches_failed() {
    let stats = NumericEquivStats {
        valid: 12,
        near_pole: 0,
        domain_error: 0,
        asymmetric_invalid: 0,
        eval_failed: 0,
        filtered_out: 0,
        mismatches: vec!["x=0.5 => 1 != 2".to_string()],
        max_abs_err: 1.0,
        max_rel_err: 1.0,
        worst_sample: None,
    };

    let outcome = classify_numeric_check_with_stats(
        Err("Numeric mismatches: x=0.5 => 1 != 2".to_string()),
        &stats,
    );

    assert!(matches!(outcome, NumericCheckOutcome::Failed(_)));
}

#[test]
fn relaxed_numeric_classification_with_fixed_retries_filtered_samples() {
    let mut ctx = Context::new();
    let lhs = parse("sec(x)^2 - tan(x)^2", &mut ctx).expect("parse lhs");
    let rhs = parse("1", &mut ctx).expect("parse rhs");

    let config = MetatestConfig {
        eval_samples: 24,
        min_valid: 8,
        sample_range: (-1.6, 1.6),
        ..metatest_config()
    };

    let outcome = classify_numeric_equiv_1var_with_fixed_relaxed(
        &ctx,
        lhs,
        rhs,
        "x",
        &[],
        &config,
        &FilterSpec::None,
    );

    assert!(
        matches!(
            outcome,
            NumericCheckOutcome::Pass | NumericCheckOutcome::Inconclusive(_)
        ),
        "expected relaxed fixed-var classification to avoid hard failure, got {outcome:?}"
    );
}

#[test]
fn relaxed_numeric_classification_2var_retries_sampling_weak_cases() {
    let config = metatest_config();
    let mut calls = 0usize;

    let outcome = classify_numeric_equiv_2var_relaxed_with(&config, |_filter1, _filter2| {
        calls += 1;
        if calls == 1 {
            NumericEquivStats {
                valid: 1,
                near_pole: 24,
                domain_error: 18,
                asymmetric_invalid: 0,
                eval_failed: 0,
                filtered_out: 0,
                mismatches: Vec::new(),
                max_abs_err: 0.0,
                max_rel_err: 0.0,
                worst_sample: None,
            }
        } else {
            NumericEquivStats {
                valid: 8,
                ..Default::default()
            }
        }
    });

    assert!(matches!(outcome, NumericCheckOutcome::Pass));
    assert!(calls > 1, "expected relaxed 2var classification to retry");
}
