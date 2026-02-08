//! Polynomial GCD structural rule.
//!
//! Implements `poly_gcd(a, b)` which finds the structural GCD of two expressions
//! by collecting multiplicative factors and intersecting them.
//!
//! Example:
//! ```text
//! poly_gcd((1+x)^3 * (2+y), (1+x)^2 * (3+z)) = (1+x)^2
//! poly_gcd(a*g, b*g) = g
//! ```
//!
//! This allows Mathematica/Symbolica-style polynomial GCD without expanding.

use crate::engine::Simplifier;
use crate::gcd_zippel_modp::ZippelPreset;
use crate::options::EvalOptions;
use crate::phase::PhaseMask;
use crate::rule::{Rewrite, Rule};
use crate::rules::algebra::gcd_exact::{gcd_exact, GcdExactBudget, GcdExactLayer};
use cas_ast::{Context, DisplayExpr, Expr, ExprId};
use num_traits::{One, ToPrimitive};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Pre-evaluate an expression to resolve specific function wrappers.
///
/// SAFETY:
/// - Only evaluates `expand()` (explicit user intent) and `__hold` (internal wrapper)
/// - Uses StepsMode::Off and ExpandPolicy::Never to avoid recursive work
/// - Does NOT evaluate `factor()` or `simplify()` (too expensive for GCD path)
/// - Avoids recursion: won't trigger poly_gcd from within poly_gcd
fn pre_evaluate_for_gcd(ctx: &mut Context, expr: ExprId) -> ExprId {
    use crate::options::StepsMode;
    use crate::phase::ExpandPolicy;

    // Only process specific wrappers that need evaluation
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        let fn_id = *fn_id;
        let args = args.clone();

        // Check for expand() via builtin
        if matches!(ctx.builtin_of(fn_id), Some(cas_ast::BuiltinFn::Expand)) {
            // expand() is explicitly requested by user - evaluate it
            let opts = EvalOptions {
                steps_mode: StepsMode::Off,       // No step tracking
                expand_policy: ExpandPolicy::Off, // Don't auto-expand other things
                ..Default::default()
            };
            let mut simplifier = Simplifier::with_profile(&opts);
            simplifier.set_steps_mode(StepsMode::Off);

            // Transfer context
            std::mem::swap(&mut simplifier.context, ctx);
            let (result, _) = simplifier.expand(expr); // Use expand() specifically
                                                       // Transfer back
            std::mem::swap(&mut simplifier.context, ctx);
            return result;
        }

        // __hold is an internal wrapper - unwrap it using canonical helper
        if ctx.is_builtin(fn_id, cas_ast::BuiltinFn::Hold) && !args.is_empty() {
            return args[0]; // Just unwrap, don't recurse
        }

        // factor() and simplify() are TOO EXPENSIVE for GCD path
        // Leave them as-is and let the converter handle or fail gracefully
        let name = ctx.sym_name(fn_id);
        if name == "factor" || name == "simplify" {
            // Don't pre-evaluate these - they could be O(expensive)
            // The GCD will fall back to structural if conversion fails
            return expr;
        }
    }
    expr
}

// =============================================================================
// GCD Mode enum for unified API
// =============================================================================

/// Mode for poly_gcd computation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GcdMode {
    /// Structural GCD (HoldAll, no expansion) - default
    Structural,
    /// Auto-select: structural → exact → modp
    Auto,
    /// Force exact GCD over ℚ[x]
    Exact,
    /// Force modular GCD over 𝔽p[x]
    Modp,
}

/// Goal/context for GCD computation - determines allowed methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GcdGoal {
    /// User explicitly asked for GCD (full pipeline allowed including modp)
    UserPolyGcd,
    /// Simplifier canceling fractions (safe methods only: Structural → Exact)
    /// Modp is BLOCKED for soundness - fraction cancellation must be deterministic
    CancelFraction,
}

/// Parse GcdMode from expression (Variable token)
pub fn parse_gcd_mode(ctx: &Context, expr: ExprId) -> GcdMode {
    if let Expr::Variable(sym_id) = ctx.get(expr) {
        let s = ctx.sym_name(*sym_id);
        match s.to_lowercase().as_str() {
            "auto" => GcdMode::Auto,
            "exact" | "rational" | "algebraic" | "q" => GcdMode::Exact,
            "modp" | "mod_p" | "fast" | "zippel" => GcdMode::Modp,
            _ => GcdMode::Structural, // Unknown = structural
        }
    } else {
        GcdMode::Structural
    }
}

/// Parse modp options (preset symbol and/or main_var int) from remaining args
fn parse_modp_options(ctx: &Context, args: &[ExprId]) -> (Option<ZippelPreset>, Option<usize>) {
    let mut preset: Option<ZippelPreset> = None;
    let mut main_var: Option<usize> = None;

    for &arg in args {
        // Try as integer (main_var)
        if let Expr::Number(n) = ctx.get(arg) {
            if n.is_integer() {
                if let Some(v) = n.to_integer().to_usize() {
                    if v <= 64 {
                        main_var = Some(v);
                        continue;
                    }
                }
            }
        }
        // Try as symbol (preset)
        if let Expr::Variable(sym_id) = ctx.get(arg) {
            let s = ctx.sym_name(*sym_id);
            if let Some(p) = ZippelPreset::parse(s) {
                preset = Some(p);
            }
        }
    }

    (preset, main_var)
}

// =============================================================================
// AC-Canonical Key for expression comparison
// =============================================================================

/// A hash-based key for AC (associative-commutative) comparison of expressions.
/// Two expressions with the same ExprKey are considered structurally equivalent
/// even if they have different parenthesization or term ordering.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExprKey(u64);

/// Compute AC-canonical key for an expression.
/// This flattens Add and Mul chains, sorts children by their keys, and produces
/// a stable hash that's independent of parenthesization and term order.
fn expr_key_ac(ctx: &Context, expr: ExprId) -> ExprKey {
    let mut hasher = DefaultHasher::new();
    expr_key_hash(ctx, expr, &mut hasher);
    ExprKey(hasher.finish())
}

/// Core hashing logic for AC-canonical key
fn expr_key_hash<H: Hasher>(ctx: &Context, expr: ExprId, hasher: &mut H) {
    match ctx.get(expr) {
        Expr::Add(_, _) => {
            // Flatten and sort by key
            let mut children = Vec::new();
            flatten_add(ctx, expr, &mut children);
            let mut keys: Vec<ExprKey> = children.iter().map(|&e| expr_key_ac(ctx, e)).collect();
            keys.sort();

            "Add".hash(hasher);
            keys.len().hash(hasher);
            for key in keys {
                key.hash(hasher);
            }
        }
        Expr::Mul(_, _) => {
            // Flatten and sort by key
            let mut children = Vec::new();
            flatten_mul(ctx, expr, &mut children);
            let mut keys: Vec<ExprKey> = children.iter().map(|&e| expr_key_ac(ctx, e)).collect();
            keys.sort();

            "Mul".hash(hasher);
            keys.len().hash(hasher);
            for key in keys {
                key.hash(hasher);
            }
        }
        Expr::Pow(base, exp) => {
            "Pow".hash(hasher);
            expr_key_ac(ctx, *base).hash(hasher);
            expr_key_ac(ctx, *exp).hash(hasher);
        }
        Expr::Neg(inner) => {
            "Neg".hash(hasher);
            expr_key_ac(ctx, *inner).hash(hasher);
        }
        Expr::Sub(a, b) => {
            // Treat as Add(a, Neg(b)) for AC equivalence
            "Add".hash(hasher);
            2usize.hash(hasher);
            let mut keys = vec![expr_key_ac(ctx, *a), expr_key_neg(ctx, *b)];
            keys.sort();
            for key in keys {
                key.hash(hasher);
            }
        }
        Expr::Div(a, b) => {
            "Div".hash(hasher);
            expr_key_ac(ctx, *a).hash(hasher);
            expr_key_ac(ctx, *b).hash(hasher);
        }
        Expr::Number(n) => {
            "Number".hash(hasher);
            n.numer().hash(hasher);
            n.denom().hash(hasher);
        }
        Expr::Variable(name) => {
            "Variable".hash(hasher);
            name.hash(hasher);
        }
        Expr::Constant(c) => {
            "Constant".hash(hasher);
            format!("{:?}", c).hash(hasher);
        }
        Expr::Function(name, args) => {
            "Function".hash(hasher);
            name.hash(hasher);
            args.len().hash(hasher);
            for arg in args {
                expr_key_ac(ctx, *arg).hash(hasher);
            }
        }
        Expr::Matrix { rows, cols, data } => {
            "Matrix".hash(hasher);
            rows.hash(hasher);
            cols.hash(hasher);
            for d in data {
                expr_key_ac(ctx, *d).hash(hasher);
            }
        }
        Expr::SessionRef(id) => {
            "SessionRef".hash(hasher);
            id.hash(hasher);
        }
        Expr::Hold(inner) => {
            "Hold".hash(hasher);
            expr_key_ac(ctx, *inner).hash(hasher);
        }
    }
}

/// Hash for Neg(expr) - used when treating Sub as Add(a, Neg(b))
fn expr_key_neg(ctx: &Context, expr: ExprId) -> ExprKey {
    let mut hasher = DefaultHasher::new();
    "Neg".hash(&mut hasher);
    expr_key_ac(ctx, expr).hash(&mut hasher);
    ExprKey(hasher.finish())
}

/// Flatten Add chain (associative)
///
/// Uses canonical AddView from nary.rs for shape-independence and __hold transparency.
/// (See ARCHITECTURE.md "Canonical Utilities Registry")
fn flatten_add(ctx: &Context, expr: ExprId, out: &mut Vec<ExprId>) {
    // Note: add_terms_no_sign ignores signs - for AC-key hashing this is fine
    // since we only care about term multiset equality
    out.extend(crate::nary::add_terms_no_sign(ctx, expr));
}

/// Flatten Mul chain (associative)
///
/// Uses canonical MulView from nary.rs for shape-independence and __hold transparency.
/// (See ARCHITECTURE.md "Canonical Utilities Registry")
fn flatten_mul(ctx: &Context, expr: ExprId, out: &mut Vec<ExprId>) {
    out.extend(crate::nary::mul_factors(ctx, expr));
}

/// Check if two expressions are AC-equivalent
fn expr_equal_ac(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    expr_key_ac(ctx, a) == expr_key_ac(ctx, b)
}

// =============================================================================
// __hold transparency helper
// =============================================================================

/// Strip __hold() wrapper(s) from an expression. __hold is an internal barrier
/// that should be transparent for structural operations like poly_gcd.
/// Uses canonical implementation from cas_ast::hold
fn strip_hold(ctx: &Context, mut expr: ExprId) -> ExprId {
    loop {
        let unwrapped = cas_ast::hold::unwrap_hold(ctx, expr);
        if unwrapped == expr {
            return expr;
        }
        expr = unwrapped;
    }
}

/// Collect multiplicative factors with integer exponents from an expression.
/// - Mul(...) is flattened
/// - Pow(base, k) with integer k becomes (base, k)
/// - Everything else becomes (expr, 1)
/// - __hold wrappers are stripped transparently
fn collect_mul_factors(ctx: &Context, expr: ExprId) -> Vec<(ExprId, i64)> {
    // Strip __hold wrapper first
    let expr = strip_hold(ctx, expr);
    let mut factors = Vec::new();
    collect_mul_factors_rec(ctx, expr, 1, &mut factors);
    factors
}

fn collect_mul_factors_rec(
    ctx: &Context,
    expr: ExprId,
    mult: i64,
    factors: &mut Vec<(ExprId, i64)>,
) {
    match ctx.get(expr) {
        Expr::Mul(left, right) => {
            collect_mul_factors_rec(ctx, *left, mult, factors);
            collect_mul_factors_rec(ctx, *right, mult, factors);
        }
        Expr::Pow(base, exp) => {
            if let Some(k) = get_integer_exp(ctx, *exp) {
                if k > 0 {
                    factors.push((*base, mult * k));
                } else {
                    // Negative exponents: treat whole as factor
                    factors.push((expr, mult));
                }
            } else {
                factors.push((expr, mult));
            }
        }
        _ => {
            factors.push((expr, mult));
        }
    }
}

/// Extract integer from exponent expression
fn get_integer_exp(ctx: &Context, exp: ExprId) -> Option<i64> {
    match ctx.get(exp) {
        Expr::Number(n) => {
            if n.is_integer() {
                n.to_integer().try_into().ok()
            } else {
                None
            }
        }
        Expr::Neg(inner) => get_integer_exp(ctx, *inner).map(|k| -k),
        _ => None,
    }
}

/// Build a product from factors with exponents.
///
/// Uses canonical `MulBuilder` (right-fold with exponents).
/// (See ARCHITECTURE.md "Canonical Utilities Registry")
fn build_mul_from_factors(ctx: &mut Context, factors: &[(ExprId, i64)]) -> ExprId {
    use cas_ast::views::MulBuilder;

    let mut builder = MulBuilder::new_simple();
    for &(base, exp) in factors {
        if exp > 0 {
            builder.push_pow(base, exp);
        }
        // Negative exponents shouldn't appear in GCD factors
    }
    builder.build(ctx)
}

// =============================================================================
// Structural GCD computation
// =============================================================================

/// Compute structural GCD by intersecting factor lists.
/// Returns the GCD expression (or 1 if no common factors).
///
/// Uses AC-canonical key for factor matching - handles different parenthesization
/// and term ordering by flattening Add/Mul chains and sorting by canonical hash.
fn poly_gcd_structural(ctx: &mut Context, a: ExprId, b: ExprId) -> ExprId {
    let a_factors = collect_mul_factors(ctx, a);
    let b_factors = collect_mul_factors(ctx, b);

    // Find common factors by AC-canonical key comparison
    let mut gcd_factors: Vec<(ExprId, i64)> = Vec::new();
    let mut used_b: Vec<bool> = vec![false; b_factors.len()];

    for (a_base, a_exp) in &a_factors {
        // Find matching factor in b_factors (by AC-canonical equality)
        for (j, (b_base, b_exp)) in b_factors.iter().enumerate() {
            if !used_b[j] && expr_equal_ac(ctx, *a_base, *b_base) {
                // Common factor found: take min exponent
                let min_exp = (*a_exp).min(*b_exp);
                if min_exp > 0 {
                    gcd_factors.push((*a_base, min_exp));
                }
                used_b[j] = true;
                break;
            }
        }
    }

    build_mul_from_factors(ctx, &gcd_factors)
}

// =============================================================================
// Shallow GCD for fraction cancellation (stack-safe)
// =============================================================================

/// Shallow GCD for fraction cancellation - designed to be called from SimplifyFractionRule.
///
/// Unlike the full `compute_poly_gcd_unified`, this function:
/// - Does NOT recurse deeply (O(1) stack depth)
/// - Does NOT call `pre_evaluate_for_gcd` (no simplifier invocations)
/// - Only does structural matching at top level
/// - Handles power bases: (x+y)^10 / (x+y)^9 → GCD = (x+y)^9
///
/// Returns (gcd, description) where gcd=1 means "no common factor found".
pub fn gcd_shallow_for_fraction(ctx: &mut Context, num: ExprId, den: ExprId) -> (ExprId, String) {
    // Strip __hold wrappers first (shallow, no recursion)
    let num = strip_hold(ctx, num);
    let den = strip_hold(ctx, den);

    // 1. Check for identical expressions → GCD = itself
    if num == den {
        return (num, "gcd(a, a) = a".to_string());
    }

    // 2. Extract power base/exponent from num and den
    let (num_base, num_exp) = extract_power_base_exp(ctx, num);
    let (den_base, den_exp) = extract_power_base_exp(ctx, den);

    // 3. If bases are structurally equal (shallow comparison), GCD is base^min(exp)
    // Use shallow structural equality that checks 1 level without deep recursion
    if expr_equal_shallow(ctx, num_base, den_base) {
        let min_exp = num_exp.min(den_exp);
        if min_exp > 0 {
            let gcd = if min_exp == 1 {
                num_base
            } else {
                let exp_expr = ctx.num(min_exp);
                ctx.add(Expr::Pow(num_base, exp_expr))
            };
            return (
                gcd,
                format!(
                    "Common power base: min({}, {}) = {}",
                    num_exp, den_exp, min_exp
                ),
            );
        }
    }

    // 4. No common factor found (no recursive calls to avoid stack overflow)
    // The univariate/multivar paths in SimplifyFractionRule handle more complex cases
    (ctx.num(1), "No common factor (shallow)".to_string())
}

/// Shallow structural equality: compares expressions at 1-2 levels depth.
/// Returns true if structurally equivalent without deep recursion.
/// This is safe for stack-constrained contexts.
fn expr_equal_shallow(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    // Fast path: identical ExprId
    if a == b {
        return true;
    }

    match (ctx.get(a), ctx.get(b)) {
        // Both numbers: compare values
        (Expr::Number(n1), Expr::Number(n2)) => n1 == n2,

        // Both variables: compare names
        (Expr::Variable(v1), Expr::Variable(v2)) => v1 == v2,

        // Both Add: compare children by ExprId
        (Expr::Add(l1, r1), Expr::Add(l2, r2)) => (l1 == l2 && r1 == r2) || (l1 == r2 && r1 == l2),

        // Both Mul: compare children by ExprId
        (Expr::Mul(l1, r1), Expr::Mul(l2, r2)) => (l1 == l2 && r1 == r2) || (l1 == r2 && r1 == l2),

        // Both Sub: compare children by ExprId (order matters)
        (Expr::Sub(l1, r1), Expr::Sub(l2, r2)) => l1 == l2 && r1 == r2,

        // Both Div: compare children by ExprId (order matters)
        (Expr::Div(n1, d1), Expr::Div(n2, d2)) => n1 == n2 && d1 == d2,

        // Both Pow: compare children by ExprId (order matters)
        (Expr::Pow(b1, e1), Expr::Pow(b2, e2)) => b1 == b2 && e1 == e2,

        // Both Neg: compare inner
        (Expr::Neg(i1), Expr::Neg(i2)) => i1 == i2,

        // Different types: not equal
        _ => false,
    }
}

/// Extract base and integer exponent from a Pow expression.
/// For non-Pow expressions, returns (expr, 1).
fn extract_power_base_exp(ctx: &Context, expr: ExprId) -> (ExprId, i64) {
    let expr = strip_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Pow(base, exp) => {
            if let Some(k) = get_integer_exp(ctx, *exp) {
                if k > 0 {
                    return (*base, k);
                }
            }
            // Non-integer or negative exponent - treat whole as base^1
            (expr, 1)
        }
        _ => (expr, 1),
    }
}

// =============================================================================
// Unified GCD dispatcher
// =============================================================================

/// Compute GCD using specified mode, returning (result, description).
///
/// The `goal` parameter determines allowed methods:
/// - `UserPolyGcd`: Full pipeline (Structural → Exact → Modp)
/// - `CancelFraction`: Safe methods only (Structural → Exact), modp BLOCKED
pub fn compute_poly_gcd_unified(
    ctx: &mut Context,
    a: ExprId,
    b: ExprId,
    goal: GcdGoal,
    mode: GcdMode,
    modp_preset: Option<ZippelPreset>,
    modp_main_var: Option<usize>,
) -> (ExprId, String) {
    match mode {
        GcdMode::Structural => {
            let gcd = poly_gcd_structural(ctx, a, b);
            let desc = format!(
                "poly_gcd({}, {})",
                DisplayExpr {
                    context: ctx,
                    id: a
                },
                DisplayExpr {
                    context: ctx,
                    id: b
                }
            );
            (gcd, desc)
        }

        GcdMode::Exact => {
            // Pre-evaluate arguments to handle expand(), factor(), etc.
            let eval_a = pre_evaluate_for_gcd(ctx, a);
            let eval_b = pre_evaluate_for_gcd(ctx, b);
            let budget = GcdExactBudget::default();
            let result = gcd_exact(ctx, eval_a, eval_b, &budget);
            let desc = format!(
                "poly_gcd({}, {}, exact) [{}]",
                DisplayExpr {
                    context: ctx,
                    id: a
                },
                DisplayExpr {
                    context: ctx,
                    id: b
                },
                format!("{:?}", result.layer_used).to_lowercase()
            );
            (result.gcd, desc)
        }

        GcdMode::Modp => {
            // V2.14.35: Block modp for CancelFraction goal (soundness)
            if goal == GcdGoal::CancelFraction {
                // Return gcd=1 (no cancellation) - this is safe
                let one = ctx.num(1);
                return (
                    one,
                    "poly_gcd(..., modp) [blocked for soundness]".to_string(),
                );
            }

            // Pre-evaluate arguments to handle expand(), factor(), etc.
            let eval_a = pre_evaluate_for_gcd(ctx, a);
            let eval_b = pre_evaluate_for_gcd(ctx, b);
            // Call modp through gcd_modp module
            use crate::rules::algebra::gcd_modp::{compute_gcd_modp_with_options, DEFAULT_PRIME};
            let preset = modp_preset.unwrap_or(ZippelPreset::Aggressive);
            match compute_gcd_modp_with_options(
                ctx,
                eval_a,
                eval_b,
                DEFAULT_PRIME,
                modp_main_var,
                Some(preset),
            ) {
                Ok(result) => {
                    let desc = format!(
                        "poly_gcd({}, {}, modp) [{:?}]",
                        DisplayExpr {
                            context: ctx,
                            id: a
                        },
                        DisplayExpr {
                            context: ctx,
                            id: b
                        },
                        preset
                    );
                    (result, desc)
                }
                Err(_e) => {
                    // V2.14.35: Remove eprintln - just return gcd=1 on error
                    let one = ctx.num(1);
                    (one, "poly_gcd(..., modp) [error]".to_string())
                }
            }
        }

        GcdMode::Auto => {
            // Try structural first
            let structural_gcd = poly_gcd_structural(ctx, a, b);

            // Check if structural found something (not just 1)
            let is_one = matches!(ctx.get(structural_gcd), Expr::Number(n) if n.is_one());

            if !is_one {
                // Structural found a non-trivial GCD
                let desc = format!(
                    "poly_gcd({}, {}, auto) [structural]",
                    DisplayExpr {
                        context: ctx,
                        id: a
                    },
                    DisplayExpr {
                        context: ctx,
                        id: b
                    }
                );
                return (structural_gcd, desc);
            }

            // Try exact if within budget - pre-evaluate arguments first
            let eval_a = pre_evaluate_for_gcd(ctx, a);
            let eval_b = pre_evaluate_for_gcd(ctx, b);
            let budget = GcdExactBudget::default();
            let exact_result = gcd_exact(ctx, eval_a, eval_b, &budget);

            if exact_result.layer_used != GcdExactLayer::BudgetExceeded {
                let desc = format!(
                    "poly_gcd({}, {}, auto) [exact:{:?}]",
                    DisplayExpr {
                        context: ctx,
                        id: a
                    },
                    DisplayExpr {
                        context: ctx,
                        id: b
                    },
                    exact_result.layer_used
                );
                return (exact_result.gcd, desc);
            }

            // V2.14.35: Block modp fallback for CancelFraction goal (soundness)
            if goal == GcdGoal::CancelFraction {
                // Return gcd=1 (no cancellation) - safe, may miss some simplifications
                let one = ctx.num(1);
                return (
                    one,
                    "poly_gcd(..., auto) [exact exceeded budget, modp blocked for soundness]"
                        .to_string(),
                );
            }

            // Fallback to modp (already have eval_a, eval_b)
            use crate::rules::algebra::gcd_modp::{compute_gcd_modp_with_options, DEFAULT_PRIME};
            let preset = modp_preset.unwrap_or(ZippelPreset::Aggressive);
            match compute_gcd_modp_with_options(
                ctx,
                eval_a,
                eval_b,
                DEFAULT_PRIME,
                modp_main_var,
                Some(preset),
            ) {
                Ok(result) => {
                    let desc = format!(
                        "poly_gcd({}, {}, auto) [modp:{:?} - probabilistic]",
                        DisplayExpr {
                            context: ctx,
                            id: a
                        },
                        DisplayExpr {
                            context: ctx,
                            id: b
                        },
                        preset
                    );
                    (result, desc)
                }
                Err(_e) => {
                    // V2.14.35: Remove eprintln - just return gcd=1 on error
                    let one = ctx.num(1);
                    (one, "poly_gcd(..., auto) [modp error]".to_string())
                }
            }
        }
    }
}

// =============================================================================
// REPL function rule
// =============================================================================

/// Rule for poly_gcd(a, b) function.
/// Computes structural GCD of two polynomial expressions.
pub struct PolyGcdRule;

impl Rule for PolyGcdRule {
    fn name(&self) -> &str {
        "Polynomial GCD"
    }

    fn allowed_phases(&self) -> PhaseMask {
        PhaseMask::CORE | PhaseMask::TRANSFORM
    }

    fn priority(&self) -> i32 {
        200 // High priority to evaluate early
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Function"])
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        let fn_expr = ctx.get(expr).clone();

        if let Expr::Function(fn_id, args) = fn_expr {
            let name = ctx.sym_name(fn_id);
            // Match poly_gcd, pgcd with 2-4 arguments
            let is_poly_gcd = name == "poly_gcd" || name == "pgcd";

            if is_poly_gcd && args.len() >= 2 && args.len() <= 4 {
                let a = args[0];
                let b = args[1];

                // Parse mode from 3rd argument (or default to Structural)
                let mode = if args.len() >= 3 {
                    parse_gcd_mode(ctx, args[2])
                } else {
                    GcdMode::Structural
                };

                // Parse modp options from remaining args
                let (modp_preset, modp_main_var) = if args.len() >= 4 {
                    parse_modp_options(ctx, &args[3..])
                } else if args.len() == 3 && mode == GcdMode::Modp {
                    // No extra args for modp, use defaults
                    (None, None)
                } else {
                    (None, None)
                };

                let (result, description) = compute_poly_gcd_unified(
                    ctx,
                    a,
                    b,
                    GcdGoal::UserPolyGcd,
                    mode,
                    modp_preset,
                    modp_main_var,
                );

                // Wrap result in __hold() to prevent further simplification
                let held_gcd = cas_ast::hold::wrap_hold(ctx, result);

                return Some(Rewrite::simple(held_gcd, description));
            }
        }

        None
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    fn setup_ctx() -> Context {
        Context::new()
    }

    #[test]
    fn test_poly_gcd_simple_common_factor() {
        let mut ctx = setup_ctx();

        // x+1
        let x = ctx.var("x");
        let one = ctx.num(1);
        let x_plus_1 = ctx.add(Expr::Add(x, one));

        // y+2
        let y = ctx.var("y");
        let two = ctx.num(2);
        let y_plus_2 = ctx.add(Expr::Add(y, two));

        // (x+1) * (y+2)
        let a = ctx.add(Expr::Mul(x_plus_1, y_plus_2));

        // z+3
        let z = ctx.var("z");
        let three = ctx.num(3);
        let z_plus_3 = ctx.add(Expr::Add(z, three));

        // (x+1) * (z+3)
        let b = ctx.add(Expr::Mul(x_plus_1, z_plus_3));

        // GCD should be (x+1)
        let gcd = poly_gcd_structural(&mut ctx, a, b);

        // Verify it's x+1
        assert_eq!(gcd, x_plus_1);
    }

    #[test]
    fn test_poly_gcd_with_powers() {
        let mut ctx = setup_ctx();

        // x+1
        let x = ctx.var("x");
        let one = ctx.num(1);
        let x_plus_1 = ctx.add(Expr::Add(x, one));

        // (x+1)^3
        let three = ctx.num(3);
        let pow3 = ctx.add(Expr::Pow(x_plus_1, three));

        // (x+1)^2
        let two = ctx.num(2);
        let pow2 = ctx.add(Expr::Pow(x_plus_1, two));

        // GCD((x+1)^3, (x+1)^2) = (x+1)^2
        let gcd = poly_gcd_structural(&mut ctx, pow3, pow2);

        // Should be (x+1)^2
        let gcd_str = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: gcd
            }
        );
        assert!(gcd_str.contains("x") && gcd_str.contains("1"));
    }

    #[test]
    fn test_poly_gcd_no_common() {
        let mut ctx = setup_ctx();

        // x
        let x = ctx.var("x");
        // y
        let y = ctx.var("y");

        // GCD(x, y) = 1 (no structural common factor)
        let gcd = poly_gcd_structural(&mut ctx, x, y);

        // Should be 1
        if let Expr::Number(n) = ctx.get(gcd) {
            assert_eq!(*n, num_rational::BigRational::from_integer(BigInt::from(1)));
        } else {
            panic!("Expected number 1");
        }
    }
}
