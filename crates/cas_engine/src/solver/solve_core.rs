//! Core solve dispatch pipeline.
//!
//! Contains the main `solve`, `solve_with_options`, and `solve_with_display_steps`
//! entry points, plus the rational-exponent pre-check.

use cas_ast::{Expr, ExprId, SolutionSet};
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::isolation::contains_var;
use crate::solver::strategies::{
    CollectTermsStrategy, IsolationStrategy, QuadraticStrategy, RationalExponentStrategy,
    RationalRootsStrategy, SubstitutionStrategy, UnwrapStrategy,
};
use crate::solver::strategy::SolverStrategy;

use super::utilities::{
    extract_denominators_with_var, is_symbolic_expr, verify_solution, wrap_with_domain_guards,
};
use super::{
    clear_current_domain_env, set_current_domain_env, step_cleanup, DepthGuard, DisplaySolveSteps,
    SolveDomainEnv, SolveStep, SolverOptions, MAX_SOLVE_DEPTH, SOLVE_DEPTH,
};

// ---------------------------------------------------------------------------
// Pre-solve exponent normalization
// ---------------------------------------------------------------------------
// The parser represents `x^(5/6)` as `Pow(x, Div(Number(5), Number(6)))`,
// but `add_exp` (used by ProductPowerRule when merging x^(1/2)*x^(1/3))
// produces `Pow(x, Number(5/6))` as a single BigRational node.
//
// These are *structurally different* AST representations of the same value,
// so `Sub(Pow(x, Number(5/6)), Pow(x, Div(5,6)))` does NOT cancel during
// simplification, causing cycle detection to fire when strategies revisit
// the equation.
//
// Fix: recursively normalize `Pow(base, Div(Number(p), Number(q)))` →
// `Pow(base, Number(p/q))` on both equation sides before computing the diff.

/// Normalize Pow exponents from Div(Number,Number) to Number(rational).
/// Returns `Some(new_expr)` if any normalization happened, `None` otherwise.
fn normalize_pow_exponents(ctx: &mut cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    let result = normalize_pow_rec(ctx, expr);
    if result != expr {
        Some(result)
    } else {
        None
    }
}

/// Recursive worker for Pow exponent normalization.
fn normalize_pow_rec(ctx: &mut cas_ast::Context, id: ExprId) -> ExprId {
    match ctx.get(id).clone() {
        Expr::Pow(base, exp) => {
            let base_rec = normalize_pow_rec(ctx, base);
            let exp_rec = normalize_pow_rec(ctx, exp);

            // Normalize Div(Number(p), Number(q)) → Number(p/q)
            let exp_final = if let Expr::Div(num, den) = ctx.get(exp_rec).clone() {
                if let (Expr::Number(p), Expr::Number(q)) = (ctx.get(num), ctx.get(den)) {
                    let (p, q) = (p.clone(), q.clone());
                    use num_traits::Zero;
                    if !q.is_zero() {
                        ctx.add(Expr::Number(&p / &q))
                    } else {
                        exp_rec
                    }
                } else {
                    exp_rec
                }
            } else {
                exp_rec
            };

            // Also fold nested Pow: Pow(Pow(base, k), r) → Pow(base, k*r)
            // when both exponents are numeric rationals.
            if let Expr::Pow(inner_base, inner_exp) = ctx.get(base_rec).clone() {
                if let (Expr::Number(k), Expr::Number(r)) = (ctx.get(inner_exp), ctx.get(exp_final))
                {
                    let product = k * r;
                    let product_exp = ctx.add(Expr::Number(product));
                    return ctx.add(Expr::Pow(inner_base, product_exp));
                }
            }

            if base_rec != base || exp_final != exp {
                ctx.add(Expr::Pow(base_rec, exp_final))
            } else {
                id
            }
        }

        // Recurse into other node types
        Expr::Add(l, r) => {
            let nl = normalize_pow_rec(ctx, l);
            let nr = normalize_pow_rec(ctx, r);
            if nl != l || nr != r {
                ctx.add(Expr::Add(nl, nr))
            } else {
                id
            }
        }
        Expr::Sub(l, r) => {
            let nl = normalize_pow_rec(ctx, l);
            let nr = normalize_pow_rec(ctx, r);
            if nl != l || nr != r {
                ctx.add(Expr::Sub(nl, nr))
            } else {
                id
            }
        }
        Expr::Mul(l, r) => {
            let nl = normalize_pow_rec(ctx, l);
            let nr = normalize_pow_rec(ctx, r);
            if nl != l || nr != r {
                ctx.add(Expr::Mul(nl, nr))
            } else {
                id
            }
        }
        Expr::Div(l, r) => {
            let nl = normalize_pow_rec(ctx, l);
            let nr = normalize_pow_rec(ctx, r);
            if nl != l || nr != r {
                ctx.add(Expr::Div(nl, nr))
            } else {
                id
            }
        }
        Expr::Neg(inner) => {
            let ni = normalize_pow_rec(ctx, inner);
            if ni != inner {
                ctx.add(Expr::Neg(ni))
            } else {
                id
            }
        }

        // Atoms, functions, etc. — no change
        _ => id,
    }
}

// ---------------------------------------------------------------------------
// Cancel common additive terms across equation sides
// ---------------------------------------------------------------------------
// After normalization, both LHS and RHS may share additive terms (e.g. x^(5/6))
// that the simplifier can't cancel in Sub(Add(A, B), B) form.
// This helper collects all additive terms from each side, removes matching
// pairs using structural comparison, and rebuilds the sides.

/// Collect all top-level additive terms from an expression.
/// `a + b + c` → vec![a, b, c] (positive terms)
/// Neg/Sub terms are also collected with a negative marker.
fn collect_additive_terms(
    ctx: &cas_ast::Context,
    id: ExprId,
    positive: bool,
    out: &mut Vec<(ExprId, bool)>,
) {
    match ctx.get(id) {
        Expr::Add(l, r) => {
            collect_additive_terms(ctx, *l, positive, out);
            collect_additive_terms(ctx, *r, positive, out);
        }
        Expr::Sub(l, r) => {
            collect_additive_terms(ctx, *l, positive, out);
            collect_additive_terms(ctx, *r, !positive, out);
        }
        Expr::Neg(inner) => {
            collect_additive_terms(ctx, *inner, !positive, out);
        }
        _ => {
            out.push((id, positive));
        }
    }
}

/// Rebuild an expression from additive terms.
fn rebuild_from_terms(ctx: &mut cas_ast::Context, terms: &[(ExprId, bool)]) -> ExprId {
    if terms.is_empty() {
        return ctx.num(0);
    }
    let mut acc = if terms[0].1 {
        terms[0].0
    } else {
        ctx.add(Expr::Neg(terms[0].0))
    };
    for &(term, positive) in &terms[1..] {
        if positive {
            acc = ctx.add(Expr::Add(acc, term));
        } else {
            acc = ctx.add(Expr::Sub(acc, term));
        }
    }
    acc
}

/// Cancel common additive terms between LHS and RHS using structural comparison.
/// Returns `Some((new_lhs, new_rhs))` if any terms were cancelled, `None` otherwise.
fn cancel_common_additive_terms(
    ctx: &mut cas_ast::Context,
    lhs: ExprId,
    rhs: ExprId,
) -> Option<(ExprId, ExprId)> {
    let mut lhs_terms: Vec<(ExprId, bool)> = Vec::new();
    let mut rhs_terms: Vec<(ExprId, bool)> = Vec::new();
    collect_additive_terms(ctx, lhs, true, &mut lhs_terms);
    collect_additive_terms(ctx, rhs, true, &mut rhs_terms);

    // For each RHS term, try to find and remove a matching LHS term
    let mut rhs_used = vec![false; rhs_terms.len()];
    let mut lhs_used = vec![false; lhs_terms.len()];
    let mut cancelled = 0usize;

    for (ri, (rt, rp)) in rhs_terms.iter().enumerate() {
        for (li, (lt, lp)) in lhs_terms.iter().enumerate() {
            if lhs_used[li] {
                continue;
            }
            // Both must have the same sign and the same structure
            if lp == rp
                && cas_ast::ordering::compare_expr(ctx, *lt, *rt) == std::cmp::Ordering::Equal
            {
                lhs_used[li] = true;
                rhs_used[ri] = true;
                cancelled += 1;
                break;
            }
        }
    }

    if cancelled == 0 {
        return None;
    }

    // Rebuild both sides without cancelled terms
    let new_lhs_terms: Vec<(ExprId, bool)> = lhs_terms
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !lhs_used[*i])
        .map(|(_, t)| t)
        .collect();
    let new_rhs_terms: Vec<(ExprId, bool)> = rhs_terms
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !rhs_used[*i])
        .map(|(_, t)| t)
        .collect();

    let new_lhs = rebuild_from_terms(ctx, &new_lhs_terms);
    let new_rhs = rebuild_from_terms(ctx, &new_rhs_terms);
    Some((new_lhs, new_rhs))
}

// ---------------------------------------------------------------------------
// Cycle detection: per-call-stack fingerprint set
// ---------------------------------------------------------------------------

thread_local! {
    /// Set of equation fingerprints seen in the current top-level solve call.
    /// Prevents infinite loops where strategies rewrite an equation into an
    /// equivalent form that would be solved again.
    static SOLVE_SEEN: std::cell::RefCell<HashSet<u64>> =
        std::cell::RefCell::new(HashSet::new());
}

/// Compute a deterministic structural hash of an AST subtree.
/// Used to fingerprint equations for cycle detection.
fn expr_fingerprint(ctx: &cas_ast::Context, id: ExprId, h: &mut impl Hasher) {
    let node = ctx.get(id);
    // Discriminant first for type safety
    std::mem::discriminant(node).hash(h);
    match node {
        Expr::Number(n) => n.hash(h),
        Expr::Variable(s) => ctx.sym_name(*s).hash(h),
        Expr::Constant(c) => std::mem::discriminant(c).hash(h),
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            expr_fingerprint(ctx, *l, h);
            expr_fingerprint(ctx, *r, h);
        }
        Expr::Pow(b, e) => {
            expr_fingerprint(ctx, *b, h);
            expr_fingerprint(ctx, *e, h);
        }
        Expr::Neg(e) | Expr::Hold(e) => expr_fingerprint(ctx, *e, h),
        Expr::Function(name, args) => {
            ctx.sym_name(*name).hash(h);
            for a in args {
                expr_fingerprint(ctx, *a, h);
            }
        }
        Expr::Matrix { rows, cols, data } => {
            rows.hash(h);
            cols.hash(h);
            for d in data {
                expr_fingerprint(ctx, *d, h);
            }
        }
        Expr::SessionRef(s) => s.hash(h),
    }
}

/// Compute a u64 fingerprint for (var, simplified_lhs, simplified_rhs).
fn equation_fingerprint(ctx: &cas_ast::Context, lhs: ExprId, rhs: ExprId, var: &str) -> u64 {
    let mut hasher = std::hash::DefaultHasher::new();
    var.hash(&mut hasher);
    expr_fingerprint(ctx, lhs, &mut hasher);
    // Separator to avoid hash collisions between different (lhs, rhs) splits
    0xFFu8.hash(&mut hasher);
    expr_fingerprint(ctx, rhs, &mut hasher);
    hasher.finish()
}

/// RAII guard that removes a fingerprint from SOLVE_SEEN on drop.
struct CycleGuard {
    fp: u64,
}

impl Drop for CycleGuard {
    fn drop(&mut self) {
        SOLVE_SEEN.with(|s| {
            s.borrow_mut().remove(&self.fp);
        });
    }
}

/// Solve with default options (for backward compatibility with tests).
/// Uses RealOnly domain and Generic mode.
pub fn solve(
    eq: &cas_ast::Equation,
    var: &str,
    simplifier: &mut Simplifier,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    solve_with_options(eq, var, simplifier, SolverOptions::default())
}

/// V2.9.8: Solve with type-enforced display-ready steps.
///
/// This is the PREFERRED entry point for display-facing code (REPL, timeline, JSON API).
/// Returns `DisplaySolveSteps` which enforces that all renderers consume post-processed
/// steps, eliminating bifurcation between text/timeline outputs at compile time.
///
/// The cleanup is applied automatically based on `opts.detailed_steps`:
/// - `true` → 5 atomic sub-steps for Normal/Verbose verbosity
/// - `false` → 3 compact steps for Succinct verbosity
pub fn solve_with_display_steps(
    eq: &cas_ast::Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
) -> Result<(SolutionSet, DisplaySolveSteps), CasError> {
    let (solution_set, raw_steps) = solve_with_options(eq, var, simplifier, opts)?;

    // Apply didactic cleanup using opts.detailed_steps
    let cleaned =
        step_cleanup::cleanup_solve_steps(&mut simplifier.context, raw_steps, opts.detailed_steps);

    Ok((solution_set, DisplaySolveSteps(cleaned)))
}

/// Internal: Solve an equation with explicit semantic options.
///
/// **For display-facing code**, use [`solve_with_display_steps`] instead.
/// This function returns raw steps that need cleanup before display.
///
/// `opts` contains ValueDomain and DomainMode which control:
/// - Whether log operations are valid (RealOnly requires positive arguments)
/// - Whether to emit assumptions or reject operations
pub(crate) fn solve_with_options(
    eq: &cas_ast::Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    // Check and increment recursion depth
    let current_depth = SOLVE_DEPTH.with(|d| {
        let mut depth = d.borrow_mut();
        *depth += 1;
        *depth
    });

    // Create guard to decrement on exit
    let _guard = DepthGuard;

    if current_depth > MAX_SOLVE_DEPTH {
        return Err(CasError::SolverError(
            "Maximum solver recursion depth exceeded. The equation may be too complex.".to_string(),
        ));
    }

    // 1. Check if variable exists in equation
    if !contains_var(&simplifier.context, eq.lhs, var)
        && !contains_var(&simplifier.context, eq.rhs, var)
    {
        return Err(CasError::VariableNotFound(var.to_string()));
    }

    // V2.1 Issue #10: Extract denominators containing the variable BEFORE simplification
    // These will be used to create NonZero guards in the final result
    let mut domain_exclusions: std::collections::HashSet<ExprId> = std::collections::HashSet::new();
    domain_exclusions.extend(extract_denominators_with_var(
        &simplifier.context,
        eq.lhs,
        var,
    ));
    domain_exclusions.extend(extract_denominators_with_var(
        &simplifier.context,
        eq.rhs,
        var,
    ));
    let domain_exclusions: Vec<ExprId> = domain_exclusions.into_iter().collect();

    // V2.2+: Build SolveDomainEnv with global required conditions
    // This is the "semantic ground" for all solver decisions
    let mut domain_env = SolveDomainEnv::new();
    {
        use crate::implicit_domain::{derive_requires_from_equation, infer_implicit_domain};

        // 1. Infer implicit domain from both sides
        let lhs_domain = infer_implicit_domain(&simplifier.context, eq.lhs, opts.value_domain);
        let rhs_domain = infer_implicit_domain(&simplifier.context, eq.rhs, opts.value_domain);
        domain_env.required.extend(&lhs_domain);
        domain_env.required.extend(&rhs_domain);

        // 2. Derive additional requires from equation equality
        // e.g., 2^x = sqrt(y) → since 2^x > 0, we have sqrt(y) > 0, so y > 0
        let derived = derive_requires_from_equation(
            &simplifier.context,
            eq.lhs,
            eq.rhs,
            &domain_env.required,
            opts.value_domain,
        );
        for cond in derived {
            domain_env.required.conditions_mut().insert(cond);
        }
    }

    // V2.2+: Set domain_env in TLS for strategies to access via get_current_domain_env()
    set_current_domain_env(domain_env);

    // RAII guard to clear env on exit (handles all return paths)
    struct EnvGuard;
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            clear_current_domain_env();
        }
    }
    let _env_guard = EnvGuard;

    // EARLY CHECK: Handle rational exponent equations BEFORE simplification
    // This prevents x^(3/2) from being simplified to |x|*sqrt(x) which causes loops
    if eq.op == cas_ast::RelOp::Eq {
        if let Some(result) = try_solve_rational_exponent(eq, var, simplifier) {
            // Wrap result with domain guards if needed
            return wrap_with_domain_guards(result, &domain_exclusions, simplifier);
        }
    }

    // 2. Simplify both sides BEFORE applying strategies
    // This is crucial for equations like "1/3*x + 1/2*x = 5"
    // which need to be simplified to "5/6*x = 5" before isolation
    let mut simplified_eq = eq.clone();

    // Simplify LHS if it contains the variable
    if contains_var(&simplifier.context, eq.lhs, var) {
        // SolveSafety: use prepass to avoid conditional rules corrupting solution set
        let sim_lhs = simplifier.simplify_for_solve(eq.lhs);
        simplified_eq.lhs = sim_lhs;

        // After simplification, try to recompose a^x/b^x -> (a/b)^x
        // This fixes cases where (a/b)^x was expanded to a^x/b^x during simplify,
        // which would leave 'x' on both sides after isolation attempts.
        if let Some(recomposed) =
            super::isolation::try_recompose_pow_quotient(&mut simplifier.context, sim_lhs)
        {
            simplified_eq.lhs = recomposed;
        }
    }

    // Simplify RHS if it contains the variable
    if contains_var(&simplifier.context, eq.rhs, var) {
        // SolveSafety: use prepass to avoid conditional rules corrupting solution set
        let sim_rhs = simplifier.simplify_for_solve(eq.rhs);
        simplified_eq.rhs = sim_rhs;

        // Also try recomposition on RHS
        if let Some(recomposed) =
            super::isolation::try_recompose_pow_quotient(&mut simplifier.context, sim_rhs)
        {
            simplified_eq.rhs = recomposed;
        }
    }

    // PRE-SOLVE RADICAL CANONICALIZATION: merge mixed-radical forms
    // on each equation side so that strategies receive a canonical
    // representation. Applied to LHS and RHS **before** computing the diff.
    // This prevents cycle detection triggers where the solver oscillates
    // between equivalent radical representations (e.g. x*sqrt(x) ↔ x^(3/2)).
    if let Some(canon_lhs) = normalize_pow_exponents(&mut simplifier.context, simplified_eq.lhs) {
        let re = simplifier.simplify_for_solve(canon_lhs);
        simplified_eq.lhs = re;
    }
    if let Some(canon_rhs) = normalize_pow_exponents(&mut simplifier.context, simplified_eq.rhs) {
        let re = simplifier.simplify_for_solve(canon_rhs);
        simplified_eq.rhs = re;
    }

    // CANCEL COMMON ADDITIVE TERMS: if both sides share structurally-identical
    // additive terms (e.g. x^(5/6) on both sides from identity noise), strip
    // them before computing the diff. This is needed because the simplifier's
    // Sub(Add(A, B), B) path doesn't have deep like-term collection.
    if let Some((new_lhs, new_rhs)) = cancel_common_additive_terms(
        &mut simplifier.context,
        simplified_eq.lhs,
        simplified_eq.rhs,
    ) {
        simplified_eq.lhs = simplifier.simplify_for_solve(new_lhs);
        simplified_eq.rhs = simplifier.simplify_for_solve(new_rhs);
    }

    // CRITICAL: After simplification, check for identities and contradictions
    // Do this by moving everything to one side: LHS - RHS
    let difference = simplifier
        .context
        .add(cas_ast::Expr::Sub(simplified_eq.lhs, simplified_eq.rhs));
    // SolveSafety: use prepass for identity/contradiction check
    let mut diff_simplified = simplifier.simplify_for_solve(difference);

    // PRE-SOLVE CANCELLATION: if diff still contains the variable, try
    // expand + simplify to cancel identity noise. This handles cases like
    // exp(x) + (x+1)(x+2) - (x² + 3x + 2) - 1 → exp(x) - 1
    // where the simplifier couldn't cancel poly(x) across the subtraction
    // boundary due to different AST shapes.
    // Also handles radical identity noise like x^(5/6) + x^2 + 1 - x^(5/6)
    // where expand() is a no-op but a second simplification pass catches
    // the cancellation.
    // Guard: only rebuild the equation when expansion achieves a significant
    // (>25%) node reduction OR eliminates the variable entirely.
    if contains_var(&simplifier.context, diff_simplified, var) {
        let expanded_diff = crate::expand::expand(&mut simplifier.context, diff_simplified);
        let re_simplified = simplifier.simplify_for_solve(expanded_diff);
        let old_nodes = cas_ast::traversal::count_all_nodes(&simplifier.context, diff_simplified);
        let new_nodes = cas_ast::traversal::count_all_nodes(&simplifier.context, re_simplified);
        // Use re-simplified form if:
        // 1. Variable was eliminated (identity/contradiction), OR
        // 2. Significant (>25%) node reduction (real simplification, not cosmetic)
        let var_eliminated = !contains_var(&simplifier.context, re_simplified, var);
        let significant_reduction = old_nodes > 4 && new_nodes * 4 < old_nodes * 3;
        if var_eliminated || significant_reduction {
            diff_simplified = re_simplified;
            let zero = simplifier.context.num(0);
            simplified_eq.lhs = diff_simplified;
            simplified_eq.rhs = zero;
        }
    }

    // Check if the difference has NO variable
    if !contains_var(&simplifier.context, diff_simplified, var) {
        // Variable disappeared - this is either an identity, contradiction, or parameter-dependent
        use cas_ast::Expr;
        match simplifier.context.get(diff_simplified) {
            Expr::Number(n) => {
                use num_traits::Zero;
                if n.is_zero() {
                    // 0 = 0: Identity, all real numbers
                    return Ok((SolutionSet::AllReals, vec![]));
                } else {
                    // c = 0 where c ≠ 0: Contradiction, no solution
                    return Ok((SolutionSet::Empty, vec![]));
                }
            }
            _ => {
                // Variable was eliminated during simplification (e.g., x/x = 1)
                // The equation is now a constraint on OTHER variables.
                // Example: (x*y)/x = 0 simplifies to y = 0
                // Solution: x can be any value (AllReals) when the constraint holds,
                // EXCEPT values that make denominators zero.
                let steps = if simplifier.collect_steps() {
                    vec![SolveStep {
                        description: format!(
                            "Variable '{}' canceled during simplification. Solution depends on constraint: {} = 0",
                            var,
                            cas_ast::DisplayExpr { context: &simplifier.context, id: diff_simplified }
                        ),
                        equation_after: cas_ast::Equation {
                            lhs: diff_simplified,
                            rhs: simplifier.context.add(Expr::Number(num_rational::BigRational::from_integer(0.into()))),
                            op: cas_ast::RelOp::Eq,
                        },
                        importance: crate::step::ImportanceLevel::Medium, substeps: vec![],
                    }]
                } else {
                    vec![]
                };

                // V2.1 Issue #10: Apply domain guards if any denominators contained the variable
                return wrap_with_domain_guards(
                    Ok((SolutionSet::AllReals, steps)),
                    &domain_exclusions,
                    simplifier,
                );
            }
        }
    }

    // CYCLE DETECTION: compute fingerprint from the simplified equation and check for repetition.
    // This catches loops where strategies rewrite equations into equivalent forms.
    // We fingerprint (var, simplified_lhs, simplified_rhs) — not the diff — to avoid
    // false positives when CollectTermsStrategy moves terms between sides.
    let fp = equation_fingerprint(
        &simplifier.context,
        simplified_eq.lhs,
        simplified_eq.rhs,
        var,
    );
    let is_cycle = SOLVE_SEEN.with(|s| !s.borrow_mut().insert(fp));
    if is_cycle {
        return Err(CasError::SolverError(
            "Cycle detected: equation revisited after rewriting (equivalent form loop)".to_string(),
        ));
    }
    let _cycle_guard = CycleGuard { fp };

    // 3. Define strategies
    // In a real app, these might be configured in Simplifier or passed in.
    let strategies: Vec<Box<dyn SolverStrategy>> = vec![
        Box::new(RationalExponentStrategy), // Must run BEFORE UnwrapStrategy to avoid loops
        Box::new(SubstitutionStrategy),
        Box::new(UnwrapStrategy),
        Box::new(QuadraticStrategy),
        Box::new(RationalRootsStrategy), // Degree ≥ 3 with numeric coefficients
        Box::new(CollectTermsStrategy),  // Must run before IsolationStrategy
        Box::new(IsolationStrategy),
    ];

    // 4. Try strategies on the simplified equation
    for strategy in strategies {
        if let Some(res) = strategy.apply(&simplified_eq, var, simplifier, &opts) {
            match res {
                Ok((result, steps)) => {
                    // Verify solutions if Discrete
                    if let SolutionSet::Discrete(sols) = result {
                        if !strategy.should_verify() {
                            return Ok((SolutionSet::Discrete(sols), steps));
                        }
                        let mut valid_sols = Vec::new();
                        for sol in sols {
                            // Skip verification for symbolic solutions (containing functions/variables)
                            // These can't be verified by substitution. Examples: ln(c/d)/ln(a/b)
                            // Only verify pure numeric solutions to catch extraneous roots.
                            if is_symbolic_expr(&simplifier.context, sol) {
                                // Trust symbolic solutions - can't verify
                                valid_sols.push(sol);
                            } else {
                                // CRITICAL: Verify against ORIGINAL equation, not simplified
                                // This ensures we reject solutions that cause division by zero
                                // in the original equation, even if they work in the simplified form
                                if verify_solution(eq, var, sol, simplifier) {
                                    valid_sols.push(sol);
                                }
                            }
                        }
                        return Ok((SolutionSet::Discrete(valid_sols), steps));
                    }
                    return Ok((result, steps));
                }
                Err(e) => {
                    return Err(e);
                }
            }
        }
    }

    Err(CasError::SolverError(
        "No strategy could solve this equation.".to_string(),
    ))
}

/// Try to solve equations with rational exponents like x^(3/2) = 8
/// by converting to x^3 = 64 (raising both sides to power q)
fn try_solve_rational_exponent(
    eq: &cas_ast::Equation,
    var: &str,
    simplifier: &mut Simplifier,
) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
    use super::strategies::match_rational_power;
    use cas_ast::Expr;

    // Check if RHS contains the variable (we only handle simple cases)
    if contains_var(&simplifier.context, eq.rhs, var) {
        return None;
    }

    // Try to match x^(p/q) on LHS
    let (base, p, q) = match_rational_power(&simplifier.context, eq.lhs, var)?;

    let mut steps = Vec::new();

    // Build new equation: base^p = rhs^q
    let p_expr = simplifier.context.num(p);
    let q_expr = simplifier.context.num(q);

    let new_lhs = simplifier.context.add(Expr::Pow(base, p_expr));
    let new_rhs = simplifier.context.add(Expr::Pow(eq.rhs, q_expr));

    // Simplify RHS (no variable, safe to simplify)
    let (sim_rhs, _) = simplifier.simplify(new_rhs);

    // DON'T simplify LHS yet - it might contain x^p which we want to solve

    let new_eq = cas_ast::Equation {
        lhs: new_lhs,
        rhs: sim_rhs,
        op: cas_ast::RelOp::Eq,
    };

    if simplifier.collect_steps() {
        steps.push(SolveStep {
            description: format!(
                "Raise both sides to power {} to eliminate rational exponent",
                q
            ),
            equation_after: new_eq.clone(),
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        });
    }

    // Recursively solve (this will go through the full solve pipeline)
    match solve(&new_eq, var, simplifier) {
        Ok((set, mut sub_steps)) => {
            steps.append(&mut sub_steps);

            // Verify solutions against original equation (handles extraneous roots)
            if let SolutionSet::Discrete(sols) = set {
                let mut valid_sols = Vec::new();
                for sol in sols {
                    if verify_solution(eq, var, sol, simplifier) {
                        valid_sols.push(sol);
                    }
                }
                Some(Ok((SolutionSet::Discrete(valid_sols), steps)))
            } else {
                Some(Ok((set, steps)))
            }
        }
        Err(e) => Some(Err(e)),
    }
}
