use crate::build::mul2_raw;
use crate::nary::{AddView, Sign};
// Telescoping Strategy for Dirichlet Kernel and similar identities
//
// This module implements a step-by-step proof strategy for telescoping sums
// like the Dirichlet kernel identity:
//   1 + 2*cos(x) + 2*cos(2x) - sin(5x/2)/sin(x/2) = 0
//
// The strategy:
// 1. Multiply by sin(x/2) to clear denominators
// 2. Apply product-to-sum: 2*cos(kx)*sin(x/2) = sin((k+½)x) - sin((k-½)x)
// 3. Observe telescoping cancellation

use cas_ast::{Context, DisplayExpr, Expr, ExprId};
use num_traits::{One, Zero};

/// Helper: Build a 2-factor product (no normalization).

/// Result of telescoping analysis
pub struct TelescopingResult {
    pub success: bool,
    pub steps: Vec<TelescopingStep>,
    pub final_result: Option<ExprId>,
}

/// A step in the telescoping proof
pub struct TelescopingStep {
    pub description: String,
    pub before: ExprId,
    pub after: ExprId,
}

impl TelescopingResult {
    pub fn format(&self, ctx: &Context) -> String {
        let mut output = String::new();

        output.push_str("\n═══════════════════════════════════════════════════════════════\n");
        output.push_str("                TELESCOPING PROOF\n");
        output.push_str("═══════════════════════════════════════════════════════════════\n\n");

        for (i, step) in self.steps.iter().enumerate() {
            let before_display = DisplayExpr {
                context: ctx,
                id: step.before,
            };
            let after_display = DisplayExpr {
                context: ctx,
                id: step.after,
            };

            output.push_str(&format!("Step {}: {}\n", i + 1, step.description));
            output.push_str(&format!("  Before: {}\n", before_display));
            output.push_str(&format!("  After:  {}\n\n", after_display));
        }

        if self.success {
            output.push_str("✓ PROVED: Expression equals 0 by telescoping cancellation\n");
        } else {
            output.push_str("✗ Could not complete telescoping proof\n");
            if let Some(result) = self.final_result {
                let result_display = DisplayExpr {
                    context: ctx,
                    id: result,
                };
                output.push_str(&format!("  Final form: {}\n", result_display));
            }
        }

        output.push_str("═══════════════════════════════════════════════════════════════\n");

        output
    }
}

/// Attempt to prove an identity using telescoping strategy
pub fn telescope(ctx: &mut Context, expr: ExprId) -> TelescopingResult {
    let mut steps = Vec::new();

    // ========================================================================
    // STEP 0: Try TrigSummationStrategy - detect Dirichlet kernel pattern
    // ========================================================================
    if let Some(result) = try_dirichlet_kernel_identity(ctx, expr) {
        let zero = ctx.num(0);
        return TelescopingResult {
            success: true,
            steps: vec![TelescopingStep {
                description: format!(
                    "Dirichlet Kernel Identity: 1 + 2Σcos(kx) = sin((n+½)x)/sin(x/2) for n={}",
                    result.n
                ),
                before: expr,
                after: zero,
            }],
            final_result: Some(zero),
        };
    }

    // Step 1: Check if expression has the form A - B/C where we can multiply by C
    let multiplier = find_denominator_for_clearing(ctx, expr);

    if let Some(mult_expr) = multiplier {
        // Create description string before mutating ctx
        let mult_description = {
            let mult_display = DisplayExpr {
                context: ctx,
                id: mult_expr,
            };
            format!("Multiply by {} to clear denominator", mult_display)
        };

        // Multiply entire expression by the denominator
        let multiplied = mul2_raw(ctx, expr, mult_expr);

        steps.push(TelescopingStep {
            description: mult_description,
            before: expr,
            after: multiplied,
        });

        // Step 2: Expand and simplify using our simplifier
        let mut simplifier = crate::Simplifier::with_default_rules();
        simplifier.context = ctx.clone();

        let (simplified, _) = simplifier.simplify(multiplied);
        *ctx = simplifier.context;

        steps.push(TelescopingStep {
            description: "Expand and apply product-to-sum identities".to_string(),
            before: multiplied,
            after: simplified,
        });

        // Step 3: Check if result is zero
        let is_zero = match ctx.get(simplified) {
            Expr::Number(n) => n.is_zero(),
            _ => false,
        };

        if is_zero {
            return TelescopingResult {
                success: true,
                steps,
                final_result: Some(simplified),
            };
        }

        // Try additional simplification passes
        let mut simplifier2 = crate::Simplifier::with_default_rules();
        simplifier2.context = ctx.clone();
        let (final_result, _) = simplifier2.simplify(simplified);
        *ctx = simplifier2.context;

        let final_is_zero = match ctx.get(final_result) {
            Expr::Number(n) => n.is_zero(),
            _ => false,
        };

        if final_is_zero {
            steps.push(TelescopingStep {
                description: "Telescoping cancellation (all terms cancel)".to_string(),
                before: simplified,
                after: final_result,
            });

            return TelescopingResult {
                success: true,
                steps,
                final_result: Some(final_result),
            };
        }

        return TelescopingResult {
            success: false,
            steps,
            final_result: Some(final_result),
        };
    }

    // No suitable structure found - try direct simplification
    let mut simplifier = crate::Simplifier::with_default_rules();
    simplifier.context = ctx.clone();
    let (result, _) = simplifier.simplify(expr);
    *ctx = simplifier.context;

    let is_zero = match ctx.get(result) {
        Expr::Number(n) => n.is_zero(),
        _ => false,
    };

    TelescopingResult {
        success: is_zero,
        steps: vec![TelescopingStep {
            description: "Direct simplification".to_string(),
            before: expr,
            after: result,
        }],
        final_result: Some(result),
    }
}

/// Find a suitable multiplier to clear denominators (looks for sin(x/2) patterns)
/// Uses AddView for shape-independent traversal of sum terms.
fn find_denominator_for_clearing(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    // Use AddView to traverse all terms regardless of tree shape
    let view = AddView::from_expr(ctx, expr);

    // Look for any term that has a denominator
    for &(term, _sign) in &view.terms {
        if let Some(denom) = extract_denominator(ctx, term) {
            return Some(denom);
        }
    }

    None
}

// nary-lint: allow-binary (structural, not n-ary sum traversal)
fn extract_denominator(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Div(_, denom) => Some(*denom),
        Expr::Neg(inner) => extract_denominator(ctx, *inner),
        Expr::Mul(l, r) => extract_denominator(ctx, *l).or_else(|| extract_denominator(ctx, *r)),
        _ => None,
    }
}

// =============================================================================
// TRIG SUMMATION STRATEGY: Dirichlet Kernel Pattern Detection
// =============================================================================
// Detects: 1 + 2*cos(x) + 2*cos(2x) + ... + 2*cos(nx) - sin((n+½)x)/sin(x/2) = 0

/// Result of Dirichlet kernel detection
pub struct DirichletKernelResult {
    pub n: usize,         // The n in the sum (highest cosine multiple)
    pub base_var: ExprId, // The base variable x
}

/// Try to detect Dirichlet kernel identity pattern (public interface for orchestrator)
pub fn try_dirichlet_kernel_identity_pub(
    ctx: &Context,
    expr: ExprId,
) -> Option<DirichletKernelResult> {
    try_dirichlet_kernel_identity(ctx, expr)
}

/// Try to detect Dirichlet kernel identity pattern
fn try_dirichlet_kernel_identity(ctx: &Context, expr: ExprId) -> Option<DirichletKernelResult> {
    // Use AddView for shape-independent term collection
    let view = AddView::from_expr(ctx, expr);

    // Look for the pattern components:
    // 1. A constant 1
    // 2. Terms of form 2*cos(k*x) for k = 1, 2, ..., n
    // 3. Negative term -sin((n+1/2)*x)/sin(x/2)

    let mut has_one = false;
    let mut cosine_multiples: Vec<(usize, ExprId)> = Vec::new(); // (k, base_var)
    let mut sin_ratio: Option<(ExprId, ExprId)> = None; // (numerator arg, denominator arg)
    let mut sin_ratio_is_negative = false;

    for &(term, sign) in &view.terms {
        let is_positive = sign == Sign::Pos;
        let term_data = ctx.get(term).clone();

        // Check for constant 1
        if let Expr::Number(n) = &term_data {
            if n.is_one() && is_positive {
                has_one = true;
                continue;
            }
        }

        // Check for 2*cos(k*x)
        if let Some((k, base)) = extract_cosine_multiple(ctx, term) {
            if is_positive {
                cosine_multiples.push((k, base));
            }
            continue;
        }

        // Check for sin(a)/sin(b) ratio
        if let Some((num_arg, den_arg)) = extract_sin_ratio(ctx, term) {
            sin_ratio = Some((num_arg, den_arg));
            sin_ratio_is_negative = !is_positive; // Should be subtracted (negative)
        }
    }

    // Verify pattern: need 1, consecutive cosine multiples, and matching sin ratio
    if !has_one || cosine_multiples.is_empty() {
        return None;
    }

    // Sort cosines by their multiple
    cosine_multiples.sort_by_key(|(k, _)| *k);

    // Check if we have 1, 2, 3, ..., n
    let n = cosine_multiples.len();
    for (i, (k, _)) in cosine_multiples.iter().enumerate() {
        if *k != i + 1 {
            return None; // Not consecutive starting from 1
        }
    }

    // Get base variable from first cosine term
    let base_var = cosine_multiples[0].1;

    // Verify sin ratio matches expected form: sin((n+1/2)*x)/sin(x/2)
    if let Some((num_arg, den_arg)) = sin_ratio {
        if sin_ratio_is_negative {
            // Check denominator is sin(x/2)
            if is_half_angle(ctx, den_arg, base_var) {
                // Check numerator is sin((n+1/2)*x)
                if is_half_integer_multiple(ctx, num_arg, base_var, n) {
                    return Some(DirichletKernelResult { n, base_var });
                }
            }
        }
    }

    None
}

// NOTE: flatten_add_sub removed - replaced by AddView::from_expr() for shape-independence

// nary-lint: allow-binary (structural pattern match for 2*cos(k*x))
/// Extract (k, base_var) from 2*cos(k*x) pattern
fn extract_cosine_multiple(ctx: &Context, expr: ExprId) -> Option<(usize, ExprId)> {
    // Pattern: 2 * cos(k * x) or cos(...) * 2
    if let Expr::Mul(l, r) = ctx.get(expr) {
        // Check if one side is 2
        let (two_side, other_side) = if is_number(ctx, *l, 2) {
            (Some(*l), *r)
        } else if is_number(ctx, *r, 2) {
            (Some(*r), *l)
        } else {
            return None;
        };

        if two_side.is_some() {
            // Other side should be cos(k*x)
            if let Expr::Function(name, args) = ctx.get(other_side) {
                if name == "cos" && args.len() == 1 {
                    return extract_multiple_of_var(ctx, args[0]);
                }
            }
        }
    }
    None
}

/// Extract sin(a)/sin(b) pattern, returning (a, b)
fn extract_sin_ratio(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    if let Expr::Div(num, den) = ctx.get(expr) {
        if let Expr::Function(num_name, num_args) = ctx.get(*num) {
            if num_name == "sin" && num_args.len() == 1 {
                if let Expr::Function(den_name, den_args) = ctx.get(*den) {
                    if den_name == "sin" && den_args.len() == 1 {
                        return Some((num_args[0], den_args[0]));
                    }
                }
            }
        }
    }
    None
}

// nary-lint: allow-binary (structural pattern match for half-angle)
/// Check if expr equals x/2 where x is base_var
fn is_half_angle(ctx: &Context, expr: ExprId, base_var: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Div(num, den) => *num == base_var && is_number(ctx, *den, 2),
        Expr::Mul(l, r) => {
            // Check for (1/2)*x or x*(1/2)
            (is_half(ctx, *l) && *r == base_var) || (is_half(ctx, *r) && *l == base_var)
        }
        _ => false,
    }
}

// nary-lint: allow-binary (structural pattern match for half-integer multiples)
/// Check if expr equals (n+1/2)*x = (2n+1)*x/2
fn is_half_integer_multiple(ctx: &Context, expr: ExprId, base_var: ExprId, n: usize) -> bool {
    let expected_num = 2 * n + 1; // (n+1/2) = (2n+1)/2

    match ctx.get(expr) {
        Expr::Div(num, den) => {
            // Check denominator is 2
            if !is_number(ctx, *den, 2) {
                return false;
            }
            // Check numerator is (2n+1)*x
            if let Expr::Mul(l, r) = ctx.get(*num) {
                (is_number(ctx, *l, expected_num as i32) && *r == base_var)
                    || (is_number(ctx, *r, expected_num as i32) && *l == base_var)
            } else {
                false
            }
        }
        Expr::Mul(l, r) => {
            // Check for ((2n+1)/2)*x pattern
            let half_mult = num_rational::BigRational::new((expected_num as i64).into(), 2.into());
            if let Expr::Number(n) = ctx.get(*l) {
                if *n == half_mult && *r == base_var {
                    return true;
                }
            }
            if let Expr::Number(n) = ctx.get(*r) {
                if *n == half_mult && *l == base_var {
                    return true;
                }
            }
            false
        }
        _ => false,
    }
}

// nary-lint: allow-binary (structural pattern match for k*x extraction)
/// Extract (multiple, base_var) from k*x expression
fn extract_multiple_of_var(ctx: &Context, expr: ExprId) -> Option<(usize, ExprId)> {
    match ctx.get(expr) {
        Expr::Mul(l, r) => {
            // k * x
            if let Expr::Number(n) = ctx.get(*l) {
                if let Some(k) = n.to_integer().to_u64_digits().1.first() {
                    if n.is_integer() && n.numer() > &0.into() {
                        return Some((*k as usize, *r));
                    }
                }
            }
            if let Expr::Number(n) = ctx.get(*r) {
                if let Some(k) = n.to_integer().to_u64_digits().1.first() {
                    if n.is_integer() && n.numer() > &0.into() {
                        return Some((*k as usize, *l));
                    }
                }
            }
            None
        }
        // Just x means k=1
        Expr::Variable(_) => Some((1, expr)),
        _ => None,
    }
}

fn is_number(ctx: &Context, expr: ExprId, expected: i32) -> bool {
    if let Expr::Number(n) = ctx.get(expr) {
        *n == num_rational::BigRational::from_integer(expected.into())
    } else {
        false
    }
}

fn is_half(ctx: &Context, expr: ExprId) -> bool {
    if let Expr::Number(n) = ctx.get(expr) {
        *n == num_rational::BigRational::new(1.into(), 2.into())
    } else {
        false
    }
}
