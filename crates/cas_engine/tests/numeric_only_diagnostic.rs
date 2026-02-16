//! Diagnostic analysis of numeric-only cases in substitution metamorphic tests.
//!
//! This test instruments the substitution pipeline with rich diagnostics:
//! - Additive term multisets (before simplification)
//! - simplify(LHS-RHS) residual structure
//! - Root cause classification (ordering, distribution, strict-vs-generic)
//!
//! Run with:
//! ```text
//! cargo test --release -p cas_engine --test numeric_only_diagnostic \
//!     -- --ignored --nocapture 2>&1 | tee /tmp/numeric_only_diag.txt
//! ```

use cas_ast::{Context, DisplayExpr, Expr, ExprId};
use cas_engine::ordering::compare_expr;
use cas_engine::Simplifier;
use cas_parser::parse;
use std::cmp::Ordering;
use std::collections::HashMap;

// ─── CSV Loading (duplicated from metamorphic tests for isolation) ─────────

struct IdentityPair {
    exp: String,
    simp: String,
    vars: Vec<String>,
    family: String,
}

struct SubstitutionExpr {
    expr: String,
    var: String,
    label: String,
}

fn load_identities() -> Vec<IdentityPair> {
    let csv_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/substitution_identities.csv"
    );
    let content = std::fs::read_to_string(csv_path).expect("read substitution_identities.csv");
    let mut pairs = Vec::new();
    let mut family = String::from("Uncategorized");
    for line in content.lines() {
        let line = line.trim();
        if line.starts_with('#') {
            let label = line.trim_start_matches('#').trim();
            if !label.is_empty()
                && !label.starts_with("Format")
                && !label.starts_with("Each row")
                && !label.starts_with("Substitution-Based")
            {
                family = label.to_string();
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 3 {
            let vars: Vec<String> = parts[2]
                .trim()
                .split(';')
                .map(|s| s.trim().to_string())
                .collect();
            pairs.push(IdentityPair {
                exp: parts[0].trim().to_string(),
                simp: parts[1].trim().to_string(),
                vars,
                family: family.clone(),
            });
        }
    }
    pairs
}

fn load_substitutions() -> Vec<SubstitutionExpr> {
    let csv_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/substitution_expressions.csv"
    );
    let content = std::fs::read_to_string(csv_path).expect("read substitution_expressions.csv");
    let mut exprs = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 3 {
            exprs.push(SubstitutionExpr {
                expr: parts[0].trim().to_string(),
                var: parts[1].trim().to_string(),
                label: parts[2].trim().to_string(),
            });
        }
    }
    exprs
}

fn text_substitute(template: &str, var: &str, replacement: &str) -> String {
    let mut result = String::with_capacity(template.len() * 2);
    let chars: Vec<char> = template.chars().collect();
    let var_chars: Vec<char> = var.chars().collect();
    let var_len = var_chars.len();
    let mut i = 0;
    while i < chars.len() {
        if i + var_len <= chars.len() && chars[i..i + var_len] == var_chars[..] {
            let before_ok = i == 0 || {
                let c = chars[i - 1];
                !c.is_alphanumeric() && c != '_'
            };
            let after_ok = i + var_len >= chars.len() || {
                let c = chars[i + var_len];
                !c.is_alphanumeric() && c != '_'
            };
            if before_ok && after_ok {
                result.push('(');
                result.push_str(replacement);
                result.push(')');
                i += var_len;
                continue;
            }
        }
        result.push(chars[i]);
        i += 1;
    }
    result
}

// ─── Additive Term Collector ──────────────────────────────────────────────

/// Collect additive terms from an expression, returning (term, is_positive) pairs.
fn collect_add_terms(ctx: &Context, id: ExprId, positive: bool, out: &mut Vec<(ExprId, bool)>) {
    match ctx.get(id) {
        Expr::Add(l, r) => {
            collect_add_terms(ctx, *l, positive, out);
            collect_add_terms(ctx, *r, positive, out);
        }
        Expr::Sub(l, r) => {
            collect_add_terms(ctx, *l, positive, out);
            collect_add_terms(ctx, *r, !positive, out);
        }
        Expr::Neg(inner) => {
            collect_add_terms(ctx, *inner, !positive, out);
        }
        _ => {
            out.push((id, positive));
        }
    }
}

/// Format a term for display.
fn fmt_term(ctx: &Context, id: ExprId) -> String {
    format!("{}", DisplayExpr { context: ctx, id })
}

/// Get a compact fingerprint of a term's structure.
fn term_shape(ctx: &Context, id: ExprId) -> String {
    match ctx.get(id) {
        Expr::Number(_) => "NUM".to_string(),
        Expr::Variable(s) => ctx.sym_name(*s).to_string(),
        Expr::Constant(_) => "CONST".to_string(),
        Expr::Add(_, _) => "Add(…)".to_string(),
        Expr::Sub(_, _) => "Sub(…)".to_string(),
        Expr::Mul(l, r) => format!("Mul({},{})", term_shape(ctx, *l), term_shape(ctx, *r)),
        Expr::Div(l, r) => format!("Div({},{})", term_shape(ctx, *l), term_shape(ctx, *r)),
        Expr::Pow(b, e) => format!("Pow({},{})", term_shape(ctx, *b), term_shape(ctx, *e)),
        Expr::Neg(e) => format!("Neg({})", term_shape(ctx, *e)),
        Expr::Function(name, _) => format!("{}(…)", ctx.sym_name(*name)),
        Expr::Hold(e) => format!("Hold({})", term_shape(ctx, *e)),
        Expr::Matrix { .. } => "Mat".to_string(),
        Expr::SessionRef(_) => "Ref".to_string(),
    }
}

// ─── Root Cause Classification ────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum RootCause {
    /// Same multiset of terms, different ordering/grouping
    OrderingMismatch,
    /// Coefficients distributed differently (2(x+y) vs 2x+2y)
    DistributionMismatch,
    /// Factored vs expanded form (x²-1 vs (x-1)(x+1))
    FactoredVsExpanded,
    /// Requires knowledge the engine doesn't have (trig identities etc.)
    MissingIdentity(String),
    /// One side has fraction, other doesn't
    FractionMismatch,
    /// Involves absolute values
    AbsoluteValue,
    /// Other / unclassified
    Other(String),
}

impl std::fmt::Display for RootCause {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RootCause::OrderingMismatch => write!(f, "ordering/grouping mismatch"),
            RootCause::DistributionMismatch => {
                write!(f, "distribution mismatch (c*(a+b) vs ca+cb)")
            }
            RootCause::FactoredVsExpanded => write!(f, "factored vs expanded form"),
            RootCause::MissingIdentity(s) => write!(f, "missing identity: {}", s),
            RootCause::FractionMismatch => write!(f, "fraction representation mismatch"),
            RootCause::AbsoluteValue => write!(f, "absolute value handling"),
            RootCause::Other(s) => write!(f, "other: {}", s),
        }
    }
}

/// Classify the root cause of a numeric-only case by analyzing the residual.
fn classify_root_cause(
    ctx: &Context,
    lhs_simplified: ExprId,
    rhs_simplified: ExprId,
    diff_simplified: ExprId,
    lhs_str: &str,
    rhs_str: &str,
    family: &str,
) -> RootCause {
    // Collect additive terms from both simplified sides
    let mut lhs_terms = Vec::new();
    let mut rhs_terms = Vec::new();
    collect_add_terms(ctx, lhs_simplified, true, &mut lhs_terms);
    collect_add_terms(ctx, rhs_simplified, true, &mut rhs_terms);

    // Check if absolute value is involved
    let src = format!("{} {}", lhs_str, rhs_str);
    if src.contains('|') || src.contains("abs(") {
        return RootCause::AbsoluteValue;
    }

    // Check for fraction mismatch
    let has_div = |id: ExprId| -> bool {
        let s = format!("{}", DisplayExpr { context: ctx, id });
        s.contains('/')
    };
    let lhs_has_div = has_div(lhs_simplified);
    let rhs_has_div = has_div(rhs_simplified);
    if lhs_has_div != rhs_has_div {
        return RootCause::FractionMismatch;
    }

    // Check for same term count — might be ordering mismatch
    if lhs_terms.len() == rhs_terms.len() {
        // Compare term-by-term to check for ordering
        let mut lhs_sorted: Vec<String> = lhs_terms
            .iter()
            .map(|(t, p)| {
                let s = fmt_term(ctx, *t);
                if *p {
                    s
                } else {
                    format!("-{}", s)
                }
            })
            .collect();
        let mut rhs_sorted: Vec<String> = rhs_terms
            .iter()
            .map(|(t, p)| {
                let s = fmt_term(ctx, *t);
                if *p {
                    s
                } else {
                    format!("-{}", s)
                }
            })
            .collect();
        lhs_sorted.sort();
        rhs_sorted.sort();

        if lhs_sorted == rhs_sorted {
            return RootCause::OrderingMismatch;
        }
    }

    // Check for distribution: one side has more terms (expanded) than the other
    let diff_count = (lhs_terms.len() as i32 - rhs_terms.len() as i32).unsigned_abs();
    if diff_count >= 1 {
        // Check if the diff has obvious algebraic structure
        let diff_display = format!(
            "{}",
            DisplayExpr {
                context: ctx,
                id: diff_simplified
            }
        );

        // If diff contains multiplied sums, it's likely distribution
        if diff_display.contains("*(") || diff_display.contains(")*") {
            return RootCause::DistributionMismatch;
        }

        // If one side has Pow(Add,n) and other doesn't, it's factored vs expanded
        let has_pow_add = |terms: &[(ExprId, bool)]| -> bool {
            terms.iter().any(|(t, _)| {
                matches!(ctx.get(*t), Expr::Pow(b, _) if matches!(ctx.get(*b), Expr::Add(_, _) | Expr::Sub(_, _)))
            })
        };
        if has_pow_add(&lhs_terms) != has_pow_add(&rhs_terms) {
            return RootCause::FactoredVsExpanded;
        }
    }

    // Check for trig/hyperbolic identity mismatch
    let diff_display = format!(
        "{}",
        DisplayExpr {
            context: ctx,
            id: diff_simplified
        }
    );
    if diff_display.contains("sin") || diff_display.contains("cos") || diff_display.contains("tan")
    {
        if family.contains("Trig") || family.contains("Angle") || family.contains("Pythagorean") {
            return RootCause::MissingIdentity("trig".to_string());
        }
    }
    if diff_display.contains("sinh")
        || diff_display.contains("cosh")
        || diff_display.contains("tanh")
    {
        return RootCause::MissingIdentity("hyperbolic".to_string());
    }
    if diff_display.contains("exp") {
        return RootCause::MissingIdentity("exponential".to_string());
    }
    if diff_display.contains("ln") || diff_display.contains("log") {
        return RootCause::MissingIdentity("logarithmic".to_string());
    }

    // Fallback: check the residual shape
    let node_count = cas_ast::traversal::count_all_nodes(ctx, diff_simplified);
    RootCause::Other(format!("{} nodes, {} diff-terms", node_count, {
        let mut dt = Vec::new();
        collect_add_terms(ctx, diff_simplified, true, &mut dt);
        dt.len()
    }))
}

// ─── Numeric Equivalence Check ────────────────────────────────────────────

fn check_numeric(ctx: &Context, a: ExprId, b: ExprId, var: &str) -> bool {
    let samples: Vec<f64> = (-50..=50).map(|i| (i as f64) * 0.2).collect();
    let mut valid = 0;
    let mut matching = 0;
    for &x in &samples {
        let mut vars = HashMap::new();
        vars.insert(var.to_string(), x);
        let va = eval_f64(ctx, a, &vars);
        let vb = eval_f64(ctx, b, &vars);
        match (va, vb) {
            (Some(a), Some(b)) if a.is_finite() && b.is_finite() => {
                valid += 1;
                let diff = (a - b).abs();
                let rel = diff / a.abs().max(1e-10);
                if diff < 1e-6 || rel < 1e-6 {
                    matching += 1;
                }
            }
            _ => {}
        }
    }
    valid >= 2 && matching == valid
}

fn eval_f64(ctx: &Context, id: ExprId, vars: &HashMap<String, f64>) -> Option<f64> {
    match ctx.get(id) {
        Expr::Number(n) => {
            use num_traits::ToPrimitive;
            n.to_f64()
        }
        Expr::Variable(s) => vars.get(ctx.sym_name(*s)).copied(),
        Expr::Constant(c) => Some(match c {
            cas_ast::Constant::Pi => std::f64::consts::PI,
            cas_ast::Constant::E => std::f64::consts::E,
            cas_ast::Constant::Phi => (1.0 + 5.0_f64.sqrt()) / 2.0,
            cas_ast::Constant::Infinity => f64::INFINITY,
            _ => return None,
        }),
        Expr::Add(l, r) => {
            let a = eval_f64(ctx, *l, vars)?;
            let b = eval_f64(ctx, *r, vars)?;
            Some(a + b)
        }
        Expr::Sub(l, r) => {
            let a = eval_f64(ctx, *l, vars)?;
            let b = eval_f64(ctx, *r, vars)?;
            Some(a - b)
        }
        Expr::Mul(l, r) => {
            let a = eval_f64(ctx, *l, vars)?;
            let b = eval_f64(ctx, *r, vars)?;
            Some(a * b)
        }
        Expr::Div(l, r) => {
            let a = eval_f64(ctx, *l, vars)?;
            let b = eval_f64(ctx, *r, vars)?;
            if b.abs() < 1e-15 {
                return None;
            }
            Some(a / b)
        }
        Expr::Pow(b, e) => {
            let base = eval_f64(ctx, *b, vars)?;
            let exp = eval_f64(ctx, *e, vars)?;
            Some(base.powf(exp))
        }
        Expr::Neg(e) => Some(-eval_f64(ctx, *e, vars)?),
        Expr::Function(name, args) => {
            let name_str = ctx.sym_name(*name);
            let a = if !args.is_empty() {
                eval_f64(ctx, args[0], vars)?
            } else {
                return None;
            };
            match name_str {
                "sin" => Some(a.sin()),
                "cos" => Some(a.cos()),
                "tan" => Some(a.tan()),
                "exp" => Some(a.exp()),
                "ln" | "log" => {
                    if a <= 0.0 {
                        return None;
                    }
                    Some(a.ln())
                }
                "sqrt" => {
                    if a < 0.0 {
                        return None;
                    }
                    Some(a.sqrt())
                }
                "abs" => Some(a.abs()),
                "arcsin" | "asin" => {
                    if a.abs() > 1.0 {
                        return None;
                    }
                    Some(a.asin())
                }
                "arccos" | "acos" => {
                    if a.abs() > 1.0 {
                        return None;
                    }
                    Some(a.acos())
                }
                "arctan" | "atan" => Some(a.atan()),
                "sinh" => Some(a.sinh()),
                "cosh" => Some(a.cosh()),
                "tanh" => Some(a.tanh()),
                "sec" => {
                    let c = a.cos();
                    if c.abs() < 1e-15 {
                        return None;
                    }
                    Some(1.0 / c)
                }
                "csc" => {
                    let s = a.sin();
                    if s.abs() < 1e-15 {
                        return None;
                    }
                    Some(1.0 / s)
                }
                _ => None,
            }
        }
        Expr::Hold(e) => eval_f64(ctx, *e, vars),
        _ => None,
    }
}

// ─── Main Diagnostic Test ─────────────────────────────────────────────────

#[derive(Clone)]
struct DiagCase {
    lhs_str: String,
    rhs_str: String,
    family: String,
    sub_label: String,
    identity_exp: String,
    identity_simp: String,
    lhs_display: String,
    rhs_display: String,
    residual_display: String,
    lhs_term_count: usize,
    rhs_term_count: usize,
    residual_nodes: usize,
    root_cause: RootCause,
}

#[test]
#[ignore]
fn numeric_only_root_cause_analysis() {
    let identities = load_identities();
    let substitutions = load_substitutions();

    eprintln!("🔍 Numeric-Only Root Cause Diagnostic");
    eprintln!(
        "   {} identities × {} substitutions = {} combos",
        identities.len(),
        substitutions.len(),
        identities.len() * substitutions.len()
    );

    let mut cases: Vec<DiagCase> = Vec::new();
    let mut total = 0usize;
    let mut nf_conv = 0usize;
    let mut proved = 0usize;
    let mut numeric = 0usize;
    let timeout = std::time::Duration::from_secs(5);

    for identity in &identities {
        let id_var = &identity.vars[0];

        for sub in &substitutions {
            let lhs_str = text_substitute(&identity.exp, id_var, &sub.expr);
            let rhs_str = text_substitute(&identity.simp, id_var, &sub.expr);
            let free_var = sub.var.clone();
            total += 1;

            let lhs_clone = lhs_str.clone();
            let rhs_clone = rhs_str.clone();
            let free_var_clone = free_var.clone();
            let family_clone = identity.family.clone();
            let sub_label_clone = sub.label.clone();
            let id_exp = identity.exp.clone();
            let id_simp = identity.simp.clone();

            let (tx, rx) = std::sync::mpsc::channel();
            let _handle = std::thread::Builder::new()
                .stack_size(8 * 1024 * 1024)
                .spawn(move || {
                    let mut simp = Simplifier::with_default_rules();
                    let exp_parsed = match parse(&lhs_clone, &mut simp.context) {
                        Ok(e) => e,
                        Err(_) => {
                            let _ = tx.send(None);
                            return;
                        }
                    };
                    let simp_parsed = match parse(&rhs_clone, &mut simp.context) {
                        Ok(e) => e,
                        Err(_) => {
                            let _ = tx.send(None);
                            return;
                        }
                    };

                    let (e, _) = simp.simplify(exp_parsed);
                    let (s, _) = simp.simplify(simp_parsed);

                    // Post-process: fold_constants
                    let mut e_final = e;
                    let mut s_final = s;
                    {
                        let cfg = cas_engine::semantics::EvalConfig::default();
                        let mut budget = cas_engine::budget::Budget::preset_cli();
                        if let Ok(r) = cas_engine::const_fold::fold_constants(
                            &mut simp.context,
                            e_final,
                            &cfg,
                            cas_engine::const_fold::ConstFoldMode::Safe,
                            &mut budget,
                        ) {
                            e_final = r.expr;
                        }
                        if let Ok(r) = cas_engine::const_fold::fold_constants(
                            &mut simp.context,
                            s_final,
                            &cfg,
                            cas_engine::const_fold::ConstFoldMode::Safe,
                            &mut budget,
                        ) {
                            s_final = r.expr;
                        }
                    }

                    // Check 1: NF convergence
                    let nf_match = compare_expr(&simp.context, e_final, s_final) == Ordering::Equal;
                    if nf_match {
                        let _ = tx.send(Some(("nf".to_string(), None)));
                        return;
                    }

                    // Check 2: Proved symbolic — simplify(LHS - RHS) == 0
                    let d = simp.context.add(Expr::Sub(e_final, s_final));
                    let (mut diff, _) = simp.simplify(d);
                    {
                        let cfg = cas_engine::semantics::EvalConfig::default();
                        let mut budget = cas_engine::budget::Budget::preset_cli();
                        if let Ok(r) = cas_engine::const_fold::fold_constants(
                            &mut simp.context,
                            diff,
                            &cfg,
                            cas_engine::const_fold::ConstFoldMode::Safe,
                            &mut budget,
                        ) {
                            diff = r.expr;
                        }
                    }
                    let zero = num_rational::BigRational::from_integer(0.into());
                    if matches!(simp.context.get(diff), Expr::Number(n) if *n == zero) {
                        let _ = tx.send(Some(("proved".to_string(), None)));
                        return;
                    }

                    // Check 3: Numeric
                    if check_numeric(&simp.context, e_final, s_final, &free_var_clone) {
                        // Diagnostic data
                        let lhs_display = format!(
                            "{}",
                            DisplayExpr {
                                context: &simp.context,
                                id: e_final
                            }
                        );
                        let rhs_display = format!(
                            "{}",
                            DisplayExpr {
                                context: &simp.context,
                                id: s_final
                            }
                        );
                        let residual_display = format!(
                            "{}",
                            DisplayExpr {
                                context: &simp.context,
                                id: diff
                            }
                        );

                        let mut lhs_terms = Vec::new();
                        let mut rhs_terms = Vec::new();
                        collect_add_terms(&simp.context, e_final, true, &mut lhs_terms);
                        collect_add_terms(&simp.context, s_final, true, &mut rhs_terms);

                        let residual_nodes =
                            cas_ast::traversal::count_all_nodes(&simp.context, diff);

                        let root_cause = classify_root_cause(
                            &simp.context,
                            e_final,
                            s_final,
                            diff,
                            &lhs_clone,
                            &rhs_clone,
                            &family_clone,
                        );

                        let diag = DiagCase {
                            lhs_str: lhs_clone,
                            rhs_str: rhs_clone,
                            family: family_clone,
                            sub_label: sub_label_clone,
                            identity_exp: id_exp,
                            identity_simp: id_simp,
                            lhs_display,
                            rhs_display,
                            residual_display,
                            lhs_term_count: lhs_terms.len(),
                            rhs_term_count: rhs_terms.len(),
                            residual_nodes,
                            root_cause,
                        };
                        let _ = tx.send(Some(("numeric".to_string(), Some(diag))));
                    } else {
                        let _ = tx.send(Some(("failed".to_string(), None)));
                    }
                });

            match rx.recv_timeout(timeout) {
                Ok(Some((kind, diag))) => match kind.as_str() {
                    "nf" => nf_conv += 1,
                    "proved" => proved += 1,
                    "numeric" => {
                        numeric += 1;
                        if let Some(d) = diag {
                            cases.push(d);
                        }
                    }
                    _ => {}
                },
                Ok(None) => {} // parse error
                Err(_) => {}   // timeout
            }
        }
    }

    // ─── Summary ──────────────────────────────────────────────────────────
    eprintln!("\n╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  SUBSTITUTION DIAGNOSTIC SUMMARY                           ║");
    eprintln!("╠══════════════════════════════════════════════════════════════╣");
    eprintln!(
        "║  Total combos:     {:>5}                                   ║",
        total
    );
    eprintln!(
        "║  NF-convergent:    {:>5} ({:.1}%)                          ║",
        nf_conv,
        nf_conv as f64 / total as f64 * 100.0
    );
    eprintln!(
        "║  Proved-symbolic:  {:>5} ({:.1}%)                          ║",
        proved,
        proved as f64 / total as f64 * 100.0
    );
    eprintln!(
        "║  Numeric-only:     {:>5} ({:.1}%)                          ║",
        numeric,
        numeric as f64 / total as f64 * 100.0
    );
    eprintln!("╚══════════════════════════════════════════════════════════════╝");

    // ─── Root Cause Clustering ────────────────────────────────────────────
    let mut by_cause: HashMap<String, Vec<&DiagCase>> = HashMap::new();
    for case in &cases {
        by_cause
            .entry(format!("{}", case.root_cause))
            .or_default()
            .push(case);
    }

    let mut cause_counts: Vec<(String, usize)> =
        by_cause.iter().map(|(k, v)| (k.clone(), v.len())).collect();
    cause_counts.sort_by(|a, b| b.1.cmp(&a.1));

    eprintln!("\n┌──────────────────────────────────────────────────────────────┐");
    eprintln!(
        "│  ROOT CAUSE CLUSTERS (numeric-only: {} cases)             │",
        numeric
    );
    eprintln!("├────────────────────────────────────────────┬───────┬───────┤");
    eprintln!("│ Root Cause                                 │ Count │   %   │");
    eprintln!("├────────────────────────────────────────────┼───────┼───────┤");
    for (cause, count) in &cause_counts {
        let pct = *count as f64 / numeric.max(1) as f64 * 100.0;
        let short_cause = if cause.len() > 42 {
            format!("{}…", &cause[..41])
        } else {
            cause.clone()
        };
        eprintln!("│ {:42} │ {:>5} │ {:>4.1}% │", short_cause, count, pct);
    }
    eprintln!("└────────────────────────────────────────────┴───────┴───────┘");

    // ─── Detailed Examples per Cluster ─────────────────────────────────────
    for (cause, count) in &cause_counts {
        let examples = &by_cause[cause];
        eprintln!("\n══ {} ({} cases) ══", cause, count);

        // Group by identity
        let mut by_identity: HashMap<String, Vec<&&DiagCase>> = HashMap::new();
        for ex in examples {
            let key = format!("{} ≡ {}", ex.identity_exp, ex.identity_simp);
            by_identity.entry(key).or_default().push(ex);
        }

        let mut id_counts: Vec<(String, usize)> = by_identity
            .iter()
            .map(|(k, v)| (k.clone(), v.len()))
            .collect();
        id_counts.sort_by(|a, b| b.1.cmp(&a.1));

        for (id_key, id_count) in id_counts.iter().take(5) {
            let ex = by_identity[id_key][0];
            eprintln!("  Identity: {} [{} subs fail]", id_key, id_count);
            eprintln!("    Family: {}", ex.family);
            eprintln!("    Example sub: x → {}", ex.sub_label);
            eprintln!("    simplify(LHS) = {}", truncate(&ex.lhs_display, 80));
            eprintln!("    simplify(RHS) = {}", truncate(&ex.rhs_display, 80));
            eprintln!(
                "    LHS terms: {}, RHS terms: {}",
                ex.lhs_term_count, ex.rhs_term_count
            );
            eprintln!(
                "    residual ({} nodes) = {}",
                ex.residual_nodes,
                truncate(&ex.residual_display, 80)
            );
            eprintln!();
        }
        if id_counts.len() > 5 {
            eprintln!("  … and {} more identities", id_counts.len() - 5);
        }
    }

    // ─── Identity × Substitution Heatmap ──────────────────────────────────
    eprintln!("\n┌──────────────────────────────────────────────────────────────┐");
    eprintln!("│  IDENTITY FAMILY × SUBSTITUTION LABEL HEATMAP              │");
    eprintln!("└──────────────────────────────────────────────────────────────┘");

    let mut family_sub: HashMap<(String, String), usize> = HashMap::new();
    for case in &cases {
        *family_sub
            .entry((case.family.clone(), case.sub_label.clone()))
            .or_default() += 1;
    }

    // Collect unique families and labels
    let mut families: Vec<String> = cases
        .iter()
        .map(|c| c.family.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    families.sort();
    let mut sub_labels: Vec<String> = cases
        .iter()
        .map(|c| c.sub_label.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    sub_labels.sort();

    // Print header
    eprint!("{:>30} │", "");
    for label in &sub_labels {
        eprint!(" {:>6} │", &label[..label.len().min(6)]);
    }
    eprintln!();
    eprintln!(
        "{:─>30}─┼{}",
        "",
        sub_labels.iter().map(|_| "────────┼").collect::<String>()
    );

    for family in &families {
        let mut row_total = 0;
        let mut cells = Vec::new();
        for label in &sub_labels {
            let count = family_sub
                .get(&(family.clone(), label.clone()))
                .copied()
                .unwrap_or(0);
            row_total += count;
            cells.push(count);
        }
        if row_total > 0 {
            let short_fam = if family.len() > 28 {
                format!("{}…", &family[..27])
            } else {
                family.clone()
            };
            eprint!("{:>30} │", short_fam);
            for &c in &cells {
                if c > 0 {
                    eprint!(" {:>6} │", c);
                } else {
                    eprint!("      · │");
                }
            }
            eprintln!(" = {}", row_total);
        }
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max - 1])
    }
}
