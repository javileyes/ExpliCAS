use crate::step::Step;
use cas_ast::{Context, Expr, ExprId};
use num_bigint::BigInt;
use num_rational::BigRational;

use super::{IsOne, SubStep};

/// Information about a fraction sum that was computed
#[derive(Debug)]
pub(crate) struct FractionSumInfo {
    /// The fractions that were summed
    pub fractions: Vec<BigRational>,
    /// The result of the sum
    pub result: BigRational,
}

/// Find all fraction sums in an expression tree
pub(crate) fn find_all_fraction_sums(ctx: &Context, expr: ExprId) -> Vec<FractionSumInfo> {
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
        Expr::Neg(e) | Expr::Hold(e) => find_all_fraction_sums_recursive(ctx, *e, results),
        Expr::Function(_, args) => {
            for arg in args {
                find_all_fraction_sums_recursive(ctx, *arg, results);
            }
        }
        Expr::Matrix { data, .. } => {
            for elem in data {
                find_all_fraction_sums_recursive(ctx, *elem, results);
            }
        }
        // Leaves
        Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
    }
}

/// Detect if between this step and the previous one, an exponent changed
/// due to fraction arithmetic
pub(crate) fn detect_exponent_fraction_change(
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
            super::collect_add_terms(ctx, expr, &mut terms);

            // Check if all terms are fractions (Number or Div(Number,Number))
            let mut fractions = Vec::new();
            for term in &terms {
                if let Some(frac) = super::try_as_fraction(ctx, *term) {
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
        Expr::Neg(e) | Expr::Hold(e) => find_fraction_sum_in_expr(ctx, *e),
        Expr::Function(_, args) => {
            for arg in args {
                if let Some(info) = find_fraction_sum_in_expr(ctx, *arg) {
                    return Some(info);
                }
            }
            None
        }
        Expr::Matrix { data, .. } => {
            for elem in data {
                if let Some(info) = find_fraction_sum_in_expr(ctx, *elem) {
                    return Some(info);
                }
            }
            None
        }
        // Leaves
        Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => None,
    }
}

/// Generate sub-steps explaining how fractions were summed
pub(crate) fn generate_fraction_sum_substeps(info: &FractionSumInfo) -> Vec<SubStep> {
    let mut sub_steps = Vec::new();

    if info.fractions.len() < 2 {
        return sub_steps;
    }

    // Step 1: Show the original sum
    let original_sum: Vec<String> = info.fractions.iter().map(super::format_fraction).collect();

    // Step 2: Find common denominator
    let lcm = info
        .fractions
        .iter()
        .fold(BigInt::from(1), |acc, f| super::lcm_bigint(&acc, f.denom()));

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
            before_expr: original_sum.join(" + "),
            after_expr: converted.join(" + "),
            before_latex: None,
            after_latex: None,
        });
    }

    // Step 4: Show the result
    sub_steps.push(SubStep {
        description: "Sum the fractions".to_string(),
        before_expr: if needs_conversion {
            converted.join(" + ")
        } else {
            original_sum.join(" + ")
        },
        after_expr: super::format_fraction(&info.result),
        before_latex: None,
        after_latex: None,
    });

    sub_steps
}

/// Generate sub-steps explaining polynomial factorization and GCD cancellation
/// For example: (x² - 4) / (2 + x) shows:
///   1. Factor numerator: x² - 4 → (x-2)(x+2)
///   2. Cancel common factor: (x-2)(x+2) / (x+2) → x-2
pub(crate) fn generate_gcd_factorization_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
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
        if let Some(local_before) = step.before_local() {
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
                    before_expr: num_str.clone(),
                    after_expr: local_before_str
                        .split('/')
                        .next()
                        .unwrap_or(&local_before_str)
                        .trim()
                        .to_string(),
                    before_latex: None,
                    after_latex: None,
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
            before_expr: before_formatted,
            after_expr: after_str,
            before_latex: None,
            after_latex: None,
        });
    }

    sub_steps
}
