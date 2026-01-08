//! Didactic Step Enhancement Layer
//!
//! This module provides visualization-layer enrichment of engine steps
//! without modifying the core engine. It post-processes steps to add
//! instructive detail for human learners.
//!
//! # Architecture
//! - Pure post-processing: never modifies engine behavior
//! - Optional: can be enabled/disabled via verbosity
//! - Extensible: easy to add new enrichers
//!
//! # Example
//! ```ignore
//! let enriched = didactic::enrich_steps(&ctx, original_expr, steps);
//! for step in enriched {
//!     println!("{}", step.base_step.description);
//!     for sub in &step.sub_steps {
//!         println!("    → {}", sub.description);
//!     }
//! }
//! ```

use crate::step::Step;
use cas_ast::{Context, Expr, ExprId};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{Signed, Zero};

/// An enriched step with optional sub-steps for didactic explanation
#[derive(Debug, Clone)]
pub struct EnrichedStep {
    /// The original step from the engine
    pub base_step: Step,
    /// Synthetic sub-steps that explain hidden operations
    pub sub_steps: Vec<SubStep>,
}

/// A synthetic sub-step that explains a hidden operation
#[derive(Debug, Clone)]
pub struct SubStep {
    /// Human-readable description of the operation
    pub description: String,
    /// LaTeX representation of expression before
    pub before_latex: String,
    /// LaTeX representation of expression after
    pub after_latex: String,
}

/// Enrich a list of steps with didactic sub-steps
///
/// This is the main entry point for the didactic layer.
/// It analyzes each step and adds explanatory sub-steps where helpful.
pub fn enrich_steps(ctx: &Context, original_expr: ExprId, steps: Vec<Step>) -> Vec<EnrichedStep> {
    let mut enriched = Vec::with_capacity(steps.len());

    // Check original expression for fraction sums (before any simplification)
    let all_fraction_sums = find_all_fraction_sums(ctx, original_expr);

    // Keep only the sum with the most fractions (ignore partial subsums)
    // AND deduplicate identical fraction sums
    let unique_fraction_sums: Vec<_> = if all_fraction_sums.is_empty() {
        Vec::new()
    } else {
        let max_fractions = all_fraction_sums
            .iter()
            .map(|s| s.fractions.len())
            .max()
            .unwrap_or(0);
        let mut seen = std::collections::HashSet::new();
        all_fraction_sums
            .into_iter()
            .filter(|info| info.fractions.len() == max_fractions)
            .filter(|info| {
                // Deduplicate by result value
                let key = format!("{}", info.result);
                seen.insert(key)
            })
            .collect()
    };

    for (step_idx, step) in steps.iter().enumerate() {
        let mut sub_steps = Vec::new();

        // Attach fraction sum sub-steps to EVERY step
        // The CLI will track and show them only once on the first VISIBLE step
        // This ensures sub-steps appear even if early steps are filtered out
        if !unique_fraction_sums.is_empty() {
            for info in &unique_fraction_sums {
                sub_steps.extend(generate_fraction_sum_substeps(info));
            }
        }

        // Also check for fraction sums in exponent (between steps)
        if let Some(fraction_info) = detect_exponent_fraction_change(ctx, &steps, step_idx) {
            // Avoid duplicates
            if !unique_fraction_sums
                .iter()
                .any(|o| o.fractions == fraction_info.fractions)
            {
                sub_steps.extend(generate_fraction_sum_substeps(&fraction_info));
            }
        }

        // Add factorization sub-steps for fraction GCD simplification
        if step.description.starts_with("Simplified fraction by GCD") {
            sub_steps.extend(generate_gcd_factorization_substeps(ctx, step));
        }

        // Add sub-steps for nested fraction simplification
        // Match by rule_name pattern (more stable than description string)
        let is_nested_fraction = step.rule_name.to_lowercase().contains("complex fraction")
            || step.rule_name.to_lowercase().contains("nested fraction")
            || step.description.to_lowercase().contains("nested fraction");
        if is_nested_fraction {
            sub_steps.extend(generate_nested_fraction_substeps(ctx, step));
        }

        // Add sub-steps for rationalization (generalized and product)
        if step.description.contains("Rationalize") || step.rule_name.contains("Rationalize") {
            sub_steps.extend(generate_rationalization_substeps(ctx, step));
        }

        // Add sub-steps for polynomial identity normalization (PolyZero airbag)
        if step.poly_proof.is_some() {
            sub_steps.extend(generate_polynomial_identity_substeps(ctx, step));
        }

        enriched.push(EnrichedStep {
            base_step: step.clone(),
            sub_steps,
        });
    }

    enriched
}

/// Get didactic sub-steps for an expression when there are no simplification steps
///
/// This is useful when fraction sums are computed during parsing/canonicalization
/// and there are no engine steps to attach the explanation to.
///
/// Example: `x^(1/3 + 1/6)` becomes `x^(1/2)` without any steps,
/// but we want to show how 1/3 + 1/6 = 1/2.
pub fn get_standalone_substeps(ctx: &Context, original_expr: ExprId) -> Vec<SubStep> {
    let all_fraction_sums = find_all_fraction_sums(ctx, original_expr);

    if all_fraction_sums.is_empty() {
        return Vec::new();
    }

    // Keep only the sum with the most fractions (ignore partial subsums)
    let max_fractions = all_fraction_sums
        .iter()
        .map(|s| s.fractions.len())
        .max()
        .unwrap_or(0);
    let mut seen = std::collections::HashSet::new();
    let unique_fraction_sums: Vec<_> = all_fraction_sums
        .into_iter()
        .filter(|info| info.fractions.len() == max_fractions)
        .filter(|info| {
            let key = format!("{}", info.result);
            seen.insert(key)
        })
        .collect();

    let mut sub_steps = Vec::new();
    for info in &unique_fraction_sums {
        sub_steps.extend(generate_fraction_sum_substeps(info));
    }
    sub_steps
}

/// Find all fraction sums in an expression tree
fn find_all_fraction_sums(ctx: &Context, expr: ExprId) -> Vec<FractionSumInfo> {
    let mut results = Vec::new();
    find_all_fraction_sums_recursive(ctx, expr, &mut results);
    results
}

fn find_all_fraction_sums_recursive(
    ctx: &Context,
    expr: ExprId,
    results: &mut Vec<FractionSumInfo>,
) {
    // Check if this is an Add chain of fractions
    if let Some(info) = find_fraction_sum_in_expr(ctx, expr) {
        results.push(info);
    }

    // Recurse into children
    match ctx.get(expr) {
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            find_all_fraction_sums_recursive(ctx, *l, results);
            find_all_fraction_sums_recursive(ctx, *r, results);
        }
        Expr::Pow(b, e) => {
            find_all_fraction_sums_recursive(ctx, *b, results);
            find_all_fraction_sums_recursive(ctx, *e, results);
        }
        Expr::Neg(e) => find_all_fraction_sums_recursive(ctx, *e, results),
        Expr::Function(_, args) => {
            for arg in args {
                find_all_fraction_sums_recursive(ctx, *arg, results);
            }
        }
        _ => {}
    }
}

/// Information about a fraction sum that was computed
#[derive(Debug)]
struct FractionSumInfo {
    /// The fractions that were summed
    fractions: Vec<BigRational>,
    /// The result of the sum
    result: BigRational,
}

/// Detect if between this step and the previous one, an exponent changed
/// due to fraction arithmetic
fn detect_exponent_fraction_change(
    ctx: &Context,
    steps: &[Step],
    step_idx: usize,
) -> Option<FractionSumInfo> {
    let current_step = &steps[step_idx];

    // We need to check if there's a fraction sum in the GLOBAL expression
    // that will be simplified silently (not its own step)

    // For "Add Inverse" rule - this is when x^(a) - x^(a) = 0, meaning
    // the fractional exponent sum was already computed silently
    if current_step.rule_name.contains("Inverse") || current_step.rule_name.contains("Power") {
        // Check the global_after of previous step if available,
        // or use global_after of current step
        let global_expr = if step_idx > 0 {
            steps[step_idx - 1]
                .global_after
                .unwrap_or(steps[step_idx - 1].after)
        } else {
            current_step.global_after.unwrap_or(current_step.before)
        };

        // Search recursively for fraction sums
        if let Some(info) = find_fraction_sum_in_expr(ctx, global_expr) {
            return Some(info);
        }
    }

    None
}

/// Search an expression tree for Add chains of fractions
fn find_fraction_sum_in_expr(ctx: &Context, expr: ExprId) -> Option<FractionSumInfo> {
    match ctx.get(expr) {
        Expr::Add(_, _) => {
            // Collect all terms in the Add chain
            let mut terms = Vec::new();
            collect_add_terms(ctx, expr, &mut terms);

            // Check if all terms are fractions (Number or Div(Number,Number))
            let mut fractions = Vec::new();
            for term in &terms {
                if let Some(frac) = try_as_fraction(ctx, *term) {
                    fractions.push(frac);
                } else {
                    return None; // Not all terms are fractions
                }
            }

            if fractions.len() >= 2 {
                // Only consider it a fraction sum if at least one fraction has denominator != 1
                // This filters out integer sums like 1 + 4 which aren't really "fraction sums"
                let has_actual_fraction = fractions.iter().any(|f| !f.denom().is_one());
                if !has_actual_fraction {
                    return None;
                }
                let result: BigRational = fractions.iter().cloned().sum();
                return Some(FractionSumInfo { fractions, result });
            }
            None
        }
        Expr::Pow(_, e) => {
            // Check the exponent for fraction sums
            find_fraction_sum_in_expr(ctx, *e)
        }
        Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Sub(l, r) => {
            // Check both sides
            find_fraction_sum_in_expr(ctx, *l).or_else(|| find_fraction_sum_in_expr(ctx, *r))
        }
        Expr::Neg(e) => find_fraction_sum_in_expr(ctx, *e),
        Expr::Function(_, args) => {
            for arg in args {
                if let Some(info) = find_fraction_sum_in_expr(ctx, *arg) {
                    return Some(info);
                }
            }
            None
        }
        _ => None,
    }
}

/// Try to interpret an expression as a fraction (BigRational)
/// Handles both Number(n) and Div(Number, Number) patterns
fn try_as_fraction(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    match ctx.get(expr) {
        Expr::Number(n) => Some(n.clone()),
        Expr::Div(numer, denom) => {
            // Check if both numerator and denominator are numbers
            if let (Expr::Number(n), Expr::Number(d)) = (ctx.get(*numer), ctx.get(*denom)) {
                // Convert to BigRational: n/d
                if !d.is_zero() {
                    // n and d are already BigRational, compute n/d
                    return Some(n / d);
                }
            }
            None
        }
        _ => None,
    }
}

/// Collect all terms from an Add chain
fn collect_add_terms(ctx: &Context, expr: ExprId, terms: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Add(l, r) => {
            collect_add_terms(ctx, *l, terms);
            collect_add_terms(ctx, *r, terms);
        }
        _ => terms.push(expr),
    }
}

/// Generate sub-steps explaining how fractions were summed
fn generate_fraction_sum_substeps(info: &FractionSumInfo) -> Vec<SubStep> {
    let mut sub_steps = Vec::new();

    if info.fractions.len() < 2 {
        return sub_steps;
    }

    // Step 1: Show the original sum
    let original_sum: Vec<String> = info.fractions.iter().map(format_fraction).collect();

    // Step 2: Find common denominator
    let lcm = info
        .fractions
        .iter()
        .fold(BigInt::from(1), |acc, f| lcm_bigint(&acc, f.denom()));

    // Step 3: Show conversion to common denominator
    let converted: Vec<String> = info
        .fractions
        .iter()
        .map(|f| {
            let multiplier = &lcm / f.denom();
            let new_numer = f.numer() * &multiplier;
            format!("\\frac{{{}}}{{{}}}", new_numer, lcm)
        })
        .collect();

    // Only add sub-steps if there's actual conversion needed
    let needs_conversion = info.fractions.iter().any(|f| f.denom() != &lcm);

    if needs_conversion {
        sub_steps.push(SubStep {
            description: format!("Find common denominator: {}", lcm),
            before_latex: original_sum.join(" + "),
            after_latex: converted.join(" + "),
        });
    }

    // Step 4: Show the result
    sub_steps.push(SubStep {
        description: "Sum the fractions".to_string(),
        before_latex: if needs_conversion {
            converted.join(" + ")
        } else {
            original_sum.join(" + ")
        },
        after_latex: format_fraction(&info.result),
    });

    sub_steps
}

/// Generate sub-steps explaining polynomial factorization and GCD cancellation
/// For example: (x² - 4) / (2 + x) shows:
///   1. Factor numerator: x² - 4 → (x-2)(x+2)
///   2. Cancel common factor: (x-2)(x+2) / (x+2) → x-2
fn generate_gcd_factorization_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    use cas_ast::DisplayExpr;

    let mut sub_steps = Vec::new();

    // Extract GCD from description: "Simplified fraction by GCD: <gcd>"
    let gcd_start = "Simplified fraction by GCD: ";
    if !step.description.starts_with(gcd_start) {
        return sub_steps;
    }
    let gcd_str = &step.description[gcd_start.len()..];

    // Get the before expression (which should be a Div)
    let before_expr = step.before;
    if let Expr::Div(num, den) = ctx.get(before_expr) {
        let num_str = format!(
            "{}",
            DisplayExpr {
                context: ctx,
                id: *num
            }
        );
        let den_str = format!(
            "{}",
            DisplayExpr {
                context: ctx,
                id: *den
            }
        );
        let after_str = format!(
            "{}",
            DisplayExpr {
                context: ctx,
                id: step.after
            }
        );

        // Check if the step's "before_local" shows factored form in Rule display
        // If available, use it to show the factorization step
        if let Some(local_before) = step.before_local {
            let local_before_str = format!(
                "{}",
                DisplayExpr {
                    context: ctx,
                    id: local_before
                }
            );

            // Sub-step 1: Factor the expression
            // If the rule display shows factored form like "(x - 2) * (2 + x) / (2 + x)"
            // we can extract the factorization
            if local_before_str.contains(gcd_str) && local_before_str.contains('*') {
                sub_steps.push(SubStep {
                    description: format!("Factor: {} contains factor {}", num_str, gcd_str),
                    before_latex: num_str.clone(),
                    after_latex: local_before_str
                        .split('/')
                        .next()
                        .unwrap_or(&local_before_str)
                        .trim()
                        .to_string(),
                });
            }
        }

        // Sub-step 2: Cancel common factor
        // Wrap in parentheses if they contain operators
        let needs_parens_num =
            num_str.contains('+') || num_str.contains('-') || num_str.contains(' ');
        let needs_parens_den =
            den_str.contains('+') || den_str.contains('-') || den_str.contains(' ');
        let before_formatted = format!(
            "{} / {}",
            if needs_parens_num {
                format!("({})", num_str)
            } else {
                num_str.clone()
            },
            if needs_parens_den {
                format!("({})", den_str)
            } else {
                den_str.clone()
            }
        );
        sub_steps.push(SubStep {
            description: format!("Cancel common factor: {}", gcd_str),
            before_latex: before_formatted,
            after_latex: after_str,
        });
    }

    sub_steps
}

/// Format a BigRational as a LaTeX fraction or integer
fn format_fraction(r: &BigRational) -> String {
    if r.denom().is_one() {
        format!("{}", r.numer())
    } else {
        format!("\\frac{{{}}}{{{}}}", r.numer(), r.denom())
    }
}

/// Compute LCM of two BigInts
fn lcm_bigint(a: &BigInt, b: &BigInt) -> BigInt {
    if a.is_zero() || b.is_zero() {
        BigInt::zero()
    } else {
        (a * b).abs() / gcd_bigint(a, b)
    }
}

/// Compute GCD of two BigInts using Euclidean algorithm
fn gcd_bigint(a: &BigInt, b: &BigInt) -> BigInt {
    let mut a = a.abs();
    let mut b = b.abs();
    while !b.is_zero() {
        let temp = b.clone();
        b = &a % &b;
        a = temp;
    }
    a
}

// Trait for is_one check on BigInt
trait IsOne {
    fn is_one(&self) -> bool;
}

impl IsOne for BigInt {
    fn is_one(&self) -> bool {
        *self == BigInt::from(1)
    }
}

/// Pattern classification for nested fractions
#[derive(Debug)]
enum NestedFractionPattern {
    /// P1: 1/(a + 1/b) → b/(a·b + 1)
    OneOverSumWithUnitFraction,
    /// P2: 1/(a + b/c) → c/(a·c + b)
    OneOverSumWithFraction,
    /// P3: A/(B + C/D) → A·D/(B·D + C)
    FractionOverSumWithFraction,
    /// P4: (A + 1/B)/C → (A·B + 1)/(B·C)
    SumWithFractionOverScalar,
    /// Fallback for complex patterns
    General,
}

/// Check if expression contains a division (nested fraction)
fn contains_div(ctx: &Context, id: ExprId) -> bool {
    match ctx.get(id) {
        Expr::Div(_, _) => true,
        Expr::Add(l, r) | Expr::Sub(l, r) => contains_div(ctx, *l) || contains_div(ctx, *r),
        Expr::Mul(l, r) => contains_div(ctx, *l) || contains_div(ctx, *r),
        Expr::Pow(b, e) => {
            // Check for negative exponent (b^(-1) = 1/b)
            if let Expr::Neg(_) = ctx.get(*e) {
                return true;
            }
            if let Expr::Number(n) = ctx.get(*e) {
                if n.is_negative() {
                    return true;
                }
            }
            contains_div(ctx, *b) || contains_div(ctx, *e)
        }
        Expr::Neg(inner) => contains_div(ctx, *inner),
        _ => false,
    }
}

/// Classify a nested fraction expression and return the pattern and extracted components
fn classify_nested_fraction(ctx: &Context, expr: ExprId) -> Option<NestedFractionPattern> {
    // Helper to check if expression is 1
    let is_one = |id: ExprId| -> bool {
        matches!(ctx.get(id), Expr::Number(n) if n.is_integer() && *n.numer() == BigInt::from(1))
    };

    // Helper to extract a fraction (1/x or a/b) from Add terms
    let find_fraction_in_add = |id: ExprId| -> Option<ExprId> {
        match ctx.get(id) {
            Expr::Add(l, r) => {
                if matches!(ctx.get(*l), Expr::Div(_, _)) {
                    Some(*l)
                } else if matches!(ctx.get(*r), Expr::Div(_, _)) {
                    Some(*r)
                } else {
                    None
                }
            }
            _ => None,
        }
    };

    if let Expr::Div(num, den) = ctx.get(expr) {
        // P1/P2/P3: Something/(... + .../...)
        if let Some(inner_frac) = find_fraction_in_add(*den) {
            if is_one(*num) {
                // P1 or P2: 1/(a + ?/?)
                if let Expr::Div(n, _) = ctx.get(inner_frac) {
                    if is_one(*n) {
                        return Some(NestedFractionPattern::OneOverSumWithUnitFraction);
                    }
                }
                return Some(NestedFractionPattern::OneOverSumWithFraction);
            } else {
                // P3: A/(B + C/D)
                return Some(NestedFractionPattern::FractionOverSumWithFraction);
            }
        }

        // P4: (A + 1/B)/C - numerator contains fraction
        if contains_div(ctx, *num) && !contains_div(ctx, *den) {
            return Some(NestedFractionPattern::SumWithFractionOverScalar);
        }

        // General nested: denominator has nested structure
        if contains_div(ctx, *den) {
            return Some(NestedFractionPattern::General);
        }
    }

    None
}

/// Extract the combined fraction string from an Add expression containing a fraction.
/// For example: 1 + 1/x → "(1·x + 1) / x" = "(x + 1) / x"
fn extract_combined_fraction_str(ctx: &Context, add_expr: ExprId) -> String {
    use cas_ast::DisplayExpr;

    // Find the fraction term and non-fraction term
    if let Expr::Add(l, r) = ctx.get(add_expr) {
        let (frac_id, other_id) = if matches!(ctx.get(*l), Expr::Div(_, _)) {
            (*l, *r)
        } else if matches!(ctx.get(*r), Expr::Div(_, _)) {
            (*r, *l)
        } else {
            // No fraction found, return generic
            return "(combinado)".to_string();
        };

        // Extract numerator and denominator of the fraction
        if let Expr::Div(frac_num, frac_den) = ctx.get(frac_id) {
            let frac_num_str = format!(
                "{}",
                DisplayExpr {
                    context: ctx,
                    id: *frac_num
                }
            );
            let frac_den_str = format!(
                "{}",
                DisplayExpr {
                    context: ctx,
                    id: *frac_den
                }
            );
            let other_str = format!(
                "{}",
                DisplayExpr {
                    context: ctx,
                    id: other_id
                }
            );

            // Build the combined expression: (other*den + num) / den
            // or simplified for 1/x case: (other·x + 1) / x
            // Add parentheses around denominator if it contains operators
            let needs_parens = frac_den_str.contains('+')
                || frac_den_str.contains('-')
                || frac_den_str.contains(' ');
            let den_display = if needs_parens {
                format!("({})", frac_den_str)
            } else {
                frac_den_str.clone()
            };
            return format!(
                "({} · {} + {}) / {}",
                other_str, den_display, frac_num_str, den_display
            );
        }
    }

    "(combinado)".to_string()
}

/// Generate sub-steps explaining nested fraction simplification
/// For example: 1/(1 + 1/x) shows:
///   1. Combine terms in denominator: 1 + 1/x → (x+1)/x
///   2. Invert the fraction: 1/((x+1)/x) → x/(x+1)
fn generate_nested_fraction_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    use cas_ast::DisplayExpr;

    let mut sub_steps = Vec::new();

    // Get the before expression (which should be a nested Div)
    let before_expr = step.before;
    let after_expr = step.after;

    // Classify the pattern
    let pattern = match classify_nested_fraction(ctx, before_expr) {
        Some(p) => p,
        None => return sub_steps, // Not a nested fraction pattern we handle
    };

    let before_str = format!(
        "{}",
        DisplayExpr {
            context: ctx,
            id: before_expr
        }
    );
    let after_str = format!(
        "{}",
        DisplayExpr {
            context: ctx,
            id: after_expr
        }
    );

    // Generate pattern-specific sub-steps
    match pattern {
        NestedFractionPattern::OneOverSumWithUnitFraction
        | NestedFractionPattern::OneOverSumWithFraction => {
            // P1/P2: 1/(a + b/c) → c/(a·c + b)
            // Extract denominator for display
            if let Expr::Div(_, den) = ctx.get(before_expr) {
                let den_str = format!(
                    "{}",
                    DisplayExpr {
                        context: ctx,
                        id: *den
                    }
                );

                // Try to extract inner fraction to show real intermediate
                // For 1/(a + b/c), the inner fraction is b/c, and combined = (a*c + b)/c
                let intermediate_str = extract_combined_fraction_str(ctx, *den);

                // Sub-step 1: Common denominator in the denominator
                sub_steps.push(SubStep {
                    description: "Combinar términos del denominador (denominador común)"
                        .to_string(),
                    before_latex: den_str.clone(),
                    after_latex: intermediate_str,
                });

                // Sub-step 2: Invert the fraction
                sub_steps.push(SubStep {
                    description: "Invertir la fracción: 1/(a/b) = b/a".to_string(),
                    before_latex: format!("1 / ({})", den_str),
                    after_latex: after_str,
                });
            }
        }

        NestedFractionPattern::FractionOverSumWithFraction => {
            // P3: A/(B + C/D) → A·D/(B·D + C)
            if let Expr::Div(num, den) = ctx.get(before_expr) {
                let num_str = format!(
                    "{}",
                    DisplayExpr {
                        context: ctx,
                        id: *num
                    }
                );
                let den_str = format!(
                    "{}",
                    DisplayExpr {
                        context: ctx,
                        id: *den
                    }
                );

                // Try to extract inner fraction to show real intermediate
                let intermediate_str = extract_combined_fraction_str(ctx, *den);

                // Sub-step 1: Common denominator in the denominator
                sub_steps.push(SubStep {
                    description: "Combinar términos del denominador (denominador común)"
                        .to_string(),
                    before_latex: den_str,
                    after_latex: intermediate_str,
                });

                // Sub-step 2: Multiply numerator by D and simplify
                sub_steps.push(SubStep {
                    description: format!("Multiplicar {} por el denominador interno", num_str),
                    before_latex: before_str,
                    after_latex: after_str,
                });
            }
        }

        NestedFractionPattern::SumWithFractionOverScalar => {
            // P4: (A + 1/B)/C → (A·B + 1)/(B·C)
            if let Expr::Div(num, den) = ctx.get(before_expr) {
                let num_str = format!(
                    "{}",
                    DisplayExpr {
                        context: ctx,
                        id: *num
                    }
                );
                let den_str = format!(
                    "{}",
                    DisplayExpr {
                        context: ctx,
                        id: *den
                    }
                );

                // Sub-step 1: Combine the numerator
                sub_steps.push(SubStep {
                    description: "Combinar términos del numerador (denominador común)".to_string(),
                    before_latex: num_str,
                    after_latex: "(numerador combinado) / B".to_string(),
                });

                // Sub-step 2: Divide by C (multiply denominators)
                sub_steps.push(SubStep {
                    description: format!("Dividir por {}: multiplicar denominadores", den_str),
                    before_latex: before_str,
                    after_latex: after_str,
                });
            }
        }

        NestedFractionPattern::General => {
            // Fallback: general algorithm explanation
            sub_steps.push(SubStep {
                description: "Multiplicar numerador y denominador por los denominadores internos"
                    .to_string(),
                before_latex: before_str.clone(),
                after_latex: "(expresión expandida)".to_string(),
            });

            sub_steps.push(SubStep {
                description: "Simplificar la fracción resultante".to_string(),
                before_latex: "(expresión expandida)".to_string(),
                after_latex: after_str,
            });
        }
    }

    sub_steps
}

/// Generate sub-steps explaining rationalization process
/// Uses LaTeXExprWithHints for proper sqrt notation rendering
fn generate_rationalization_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    use cas_ast::display_context::DisplayContext;
    use cas_ast::LaTeXExprWithHints;

    let mut sub_steps = Vec::new();

    // Build display hints for sqrt notation
    let hints = DisplayContext::with_root_index(2);

    // Helper to convert expression to LaTeX with hints
    let to_latex = |id: ExprId| -> String {
        LaTeXExprWithHints {
            context: ctx,
            id,
            hints: &hints,
        }
        .to_latex()
    };

    // Helper to collect additive terms from an expression
    fn collect_add_terms(ctx: &Context, expr: ExprId) -> Vec<ExprId> {
        let mut terms = Vec::new();
        collect_add_terms_recursive(ctx, expr, &mut terms);
        terms
    }

    fn collect_add_terms_recursive(ctx: &Context, expr: ExprId, terms: &mut Vec<ExprId>) {
        match ctx.get(expr) {
            Expr::Add(l, r) => {
                collect_add_terms_recursive(ctx, *l, terms);
                collect_add_terms_recursive(ctx, *r, terms);
            }
            _ => terms.push(expr),
        }
    }

    // Extract before/after expressions
    let before = step.before;
    let after = step.after;

    // Check if it's a generalized rationalization (3+ terms)
    if step.description.contains("group") {
        // Generalized rationalization: a/(x + y + z) -> a(x+y-z)/[(x+y)² - z²]
        if let Expr::Div(num, den) = ctx.get(before) {
            let num_latex = to_latex(*num);
            let den_latex = to_latex(*den);

            // Collect terms from denominator
            let den_terms = collect_add_terms(ctx, *den);

            if den_terms.len() >= 3 {
                // Group first n-1 terms as "group", last term as "c"
                let group_terms: Vec<String> = den_terms[..den_terms.len() - 1]
                    .iter()
                    .map(|t| to_latex(*t))
                    .collect();
                let last_term = to_latex(den_terms[den_terms.len() - 1]);

                let group_str = group_terms.join(" + ");

                // Sub-step 1: Show the original fraction and grouping
                sub_steps.push(SubStep {
                    description: "Agrupar términos del denominador".to_string(),
                    before_latex: format!("\\frac{{{}}}{{{}}}", num_latex, den_latex),
                    after_latex: if group_terms.len() > 1 {
                        format!("\\frac{{{}}}{{({}) + {}}}", num_latex, group_str, last_term)
                    } else {
                        format!("\\frac{{{}}}{{{} + {}}}", num_latex, group_str, last_term)
                    },
                });

                // Sub-step 2: Identify conjugate with specific terms
                let conjugate = if group_terms.len() > 1 {
                    format!("({}) - {}", group_str, last_term)
                } else {
                    format!("{} - {}", group_str, last_term)
                };

                sub_steps.push(SubStep {
                    description: "Multiplicar por el conjugado".to_string(),
                    before_latex: if group_terms.len() > 1 {
                        format!("({}) + {}", group_str, last_term)
                    } else {
                        format!("{} + {}", group_str, last_term)
                    },
                    after_latex: conjugate.clone(),
                });

                // Sub-step 3: Apply difference of squares with specific terms
                if let Expr::Div(_new_num, new_den) = ctx.get(after) {
                    let after_den_latex = to_latex(*new_den);
                    sub_steps.push(SubStep {
                        description: "Diferencia de cuadrados".to_string(),
                        before_latex: if group_terms.len() > 1 {
                            format!("({})^2 - ({})^2", group_str, last_term)
                        } else {
                            format!("{}^2 - {}^2", group_str, last_term)
                        },
                        after_latex: after_den_latex,
                    });
                }
            }
        }
    } else if step.description.contains("product") {
        // Product rationalization: a/(b·√c) -> a·√c/(b·c)
        if let Expr::Div(num, den) = ctx.get(before) {
            let num_latex = to_latex(*num);
            let den_latex = to_latex(*den);

            // Sub-step 1: Identify the radical in denominator
            sub_steps.push(SubStep {
                description: "Denominador con producto de radical".to_string(),
                before_latex: format!("\\frac{{{}}}{{{}}}", num_latex, den_latex),
                after_latex: "\\frac{a}{k \\cdot \\sqrt{n}}".to_string(),
            });

            // Sub-step 2: Multiply by √n/√n
            if let Expr::Div(new_num, new_den) = ctx.get(after) {
                let after_num_latex = to_latex(*new_num);
                let after_den_latex = to_latex(*new_den);
                sub_steps.push(SubStep {
                    description: "Multiplicar por \\sqrt{n}/\\sqrt{n}".to_string(),
                    before_latex: format!(
                        "\\frac{{{} \\cdot \\sqrt{{n}}}}{{{} \\cdot \\sqrt{{n}}}}",
                        num_latex, den_latex
                    ),
                    after_latex: format!("\\frac{{{}}}{{{}}}", after_num_latex, after_den_latex),
                });
            }
        }
    } else {
        // Binary rationalization (difference of squares with 2 terms)
        if let Expr::Div(num, den) = ctx.get(before) {
            let num_latex = to_latex(*num);
            let den_latex = to_latex(*den);

            sub_steps.push(SubStep {
                description: "Denominador binomial con radical".to_string(),
                before_latex: format!("\\frac{{{}}}{{{}}}", num_latex, den_latex),
                after_latex: "\\text{Conjugado: } a + b \\to a - b".to_string(),
            });

            if let Expr::Div(new_num, new_den) = ctx.get(after) {
                let after_num_latex = to_latex(*new_num);
                let after_den_latex = to_latex(*new_den);
                sub_steps.push(SubStep {
                    description: "(a+b)(a-b) = a² - b²".to_string(),
                    before_latex: format!(
                        "\\frac{{({}) \\cdot (a-b)}}{{({}) \\cdot (a-b)}}",
                        num_latex, den_latex
                    ),
                    after_latex: format!("\\frac{{{}}}{{{}}}", after_num_latex, after_den_latex),
                });
            }
        }
    }

    sub_steps
}

/// Generate sub-steps explaining polynomial identity normalization (PolyZero airbag)
///
/// When a polynomial identity like `(a+b)^2 - a^2 - 2ab - b^2` is detected to equal 0,
/// this generates explanatory sub-steps:
///   1. "Convert to polynomial normal form" - shows the expanded/normalized polynomial or stats
///   2. "All coefficients cancel → 0" - explains the cancellation
///
/// The proof data attached to the step contains:
/// - monomials: count of monomials in the polynomial (0 for identity = 0)
/// - degree: maximum degree of the polynomial
/// - vars: list of variable names
/// - normal_form_expr: the normalized expression if small enough to display
fn generate_polynomial_identity_substeps(ctx: &Context, step: &crate::step::Step) -> Vec<SubStep> {
    use cas_ast::DisplayExpr;

    let mut sub_steps = Vec::new();

    // Get the proof data (caller should have checked is_some())
    let proof = match &step.poly_proof {
        Some(p) => p,
        None => return sub_steps,
    };

    // Sub-step 1: Convert to polynomial normal form
    // Show the normalized expression if available, otherwise show stats
    let normal_form_description = if let Some(expr_id) = proof.normal_form_expr {
        // Expression is small enough to display
        format!(
            "{}",
            DisplayExpr {
                context: ctx,
                id: expr_id
            }
        )
    } else {
        // Expression is too large, show stats instead
        let vars_str = if proof.vars.is_empty() {
            "constante".to_string()
        } else {
            proof.vars.join(", ")
        };
        format!(
            "{} monomios, grado {}, vars: {}",
            proof.monomials, proof.degree, vars_str
        )
    };

    sub_steps.push(SubStep {
        description: "Convertir a forma normal polinómica".to_string(),
        before_latex: format!(
            "{}",
            DisplayExpr {
                context: ctx,
                id: step.before
            }
        ),
        after_latex: normal_form_description,
    });

    // Sub-step 2: Cancel like terms → 0
    sub_steps.push(SubStep {
        description: "Cancelar términos semejantes".to_string(),
        before_latex: "todos los coeficientes".to_string(),
        after_latex: "0".to_string(),
    });

    sub_steps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_fraction() {
        let half = BigRational::new(BigInt::from(1), BigInt::from(2));
        assert_eq!(format_fraction(&half), "\\frac{1}{2}");

        let three = BigRational::from_integer(BigInt::from(3));
        assert_eq!(format_fraction(&three), "3");
    }

    #[test]
    fn test_gcd_lcm() {
        let a = BigInt::from(12);
        let b = BigInt::from(8);
        assert_eq!(gcd_bigint(&a, &b), BigInt::from(4));
        assert_eq!(lcm_bigint(&a, &b), BigInt::from(24));
    }

    #[test]
    fn test_fraction_sum_substeps() {
        let fractions = vec![
            BigRational::new(BigInt::from(1), BigInt::from(24)),
            BigRational::new(BigInt::from(1), BigInt::from(2)),
            BigRational::new(BigInt::from(1), BigInt::from(6)),
        ];
        let result: BigRational = fractions.iter().cloned().sum();

        let info = FractionSumInfo {
            fractions,
            result: result.clone(),
        };

        let substeps = generate_fraction_sum_substeps(&info);
        assert!(!substeps.is_empty());

        // Result should be 17/24
        assert_eq!(result, BigRational::new(BigInt::from(17), BigInt::from(24)));
    }

    #[test]
    fn test_nested_fraction_pattern_classification_p1() {
        // P1: 1/(1 + 1/x) - unit fraction in denominator
        let mut ctx = Context::new();
        let x = ctx.add(Expr::Variable("x".to_string()));
        let one = ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(1))));
        let one_over_x = ctx.add(Expr::Div(one, x));
        let denom = ctx.add(Expr::Add(one, one_over_x));
        let expr = ctx.add(Expr::Div(one, denom));

        let pattern = classify_nested_fraction(&ctx, expr);
        assert!(matches!(
            pattern,
            Some(NestedFractionPattern::OneOverSumWithUnitFraction)
        ));
    }

    #[test]
    fn test_nested_fraction_pattern_classification_p3() {
        // P3: 2/(1 + 1/x) - non-unit numerator
        let mut ctx = Context::new();
        let x = ctx.add(Expr::Variable("x".to_string()));
        let one = ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(1))));
        let two = ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(2))));
        let one_over_x = ctx.add(Expr::Div(one, x));
        let denom = ctx.add(Expr::Add(one, one_over_x));
        let expr = ctx.add(Expr::Div(two, denom));

        let pattern = classify_nested_fraction(&ctx, expr);
        assert!(matches!(
            pattern,
            Some(NestedFractionPattern::FractionOverSumWithFraction)
        ));
    }

    #[test]
    fn test_extract_combined_fraction_simple() {
        // 1 + 1/x → "(1 · x + 1) / x"
        let mut ctx = Context::new();
        let x = ctx.add(Expr::Variable("x".to_string()));
        let one = ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(1))));
        let one_over_x = ctx.add(Expr::Div(one, x));
        let add_expr = ctx.add(Expr::Add(one, one_over_x));

        let result = extract_combined_fraction_str(&ctx, add_expr);
        assert!(
            result.contains("x"),
            "Should contain denominator 'x': {}",
            result
        );
        assert!(
            result.contains("1"),
            "Should contain numerator '1': {}",
            result
        );
    }

    #[test]
    fn test_extract_combined_fraction_complex_denominator() {
        // 1 + x/(x+1) → "(1 · (x + 1) + x) / (x + 1)"
        let mut ctx = Context::new();
        let x = ctx.add(Expr::Variable("x".to_string()));
        let one = ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(1))));
        let x_plus_1 = ctx.add(Expr::Add(x, one));
        let x_over_xplus1 = ctx.add(Expr::Div(x, x_plus_1));
        let add_expr = ctx.add(Expr::Add(one, x_over_xplus1));

        let result = extract_combined_fraction_str(&ctx, add_expr);
        // Should have parentheses around denominator
        assert!(
            result.contains("(1 + x)") || result.contains("(x + 1)"),
            "Should have parentheses around complex denominator: {}",
            result
        );
    }

    #[test]
    fn test_contains_div_simple() {
        let mut ctx = Context::new();
        let x = ctx.add(Expr::Variable("x".to_string()));
        let one = ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(1))));

        // x does not contain div
        assert!(!contains_div(&ctx, x));

        // 1/x contains div
        let div = ctx.add(Expr::Div(one, x));
        assert!(contains_div(&ctx, div));

        // 1 + 1/x contains div
        let add = ctx.add(Expr::Add(one, div));
        assert!(contains_div(&ctx, add));
    }
}
