use crate::build::mul2_raw;
use cas_ast::{Context, Expr, ExprId};
use num_traits::Zero;

/// A step in a telescoping proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TelescopingProofStep {
    pub description: String,
    pub before: ExprId,
    pub after: ExprId,
}

/// Result of telescoping analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TelescopingProofResult {
    pub success: bool,
    pub steps: Vec<TelescopingProofStep>,
    pub final_result: Option<ExprId>,
}

/// Attempt to prove an identity using telescoping strategy.
///
/// The caller provides a simplification callback to keep this algorithm
/// independent from engine/runtime crates.
pub fn telescope_with_simplify<F>(
    ctx: &mut Context,
    expr: ExprId,
    mut simplify_expr: F,
    mut render_expr: impl FnMut(&Context, ExprId) -> String,
) -> TelescopingProofResult
where
    F: FnMut(&mut Context, ExprId) -> ExprId,
{
    let mut steps = Vec::new();

    // Step 0: detect Dirichlet kernel identity directly.
    if let Some(result) = crate::telescoping_dirichlet::try_dirichlet_kernel_identity(ctx, expr) {
        let zero = ctx.num(0);
        return TelescopingProofResult {
            success: true,
            steps: vec![TelescopingProofStep {
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

    // Step 1: if possible, clear denominator and simplify.
    let multiplier = crate::telescoping_support::find_denominator_for_clearing(ctx, expr);
    if let Some(mult_expr) = multiplier {
        let mult_description = format!(
            "Multiply by {} to clear denominator",
            render_expr(ctx, mult_expr)
        );

        let multiplied = mul2_raw(ctx, expr, mult_expr);
        steps.push(TelescopingProofStep {
            description: mult_description,
            before: expr,
            after: multiplied,
        });

        let simplified = simplify_expr(ctx, multiplied);
        steps.push(TelescopingProofStep {
            description: "Expand and apply product-to-sum identities".to_string(),
            before: multiplied,
            after: simplified,
        });

        if is_zero_number(ctx, simplified) {
            return TelescopingProofResult {
                success: true,
                steps,
                final_result: Some(simplified),
            };
        }

        let final_result = simplify_expr(ctx, simplified);
        if is_zero_number(ctx, final_result) {
            steps.push(TelescopingProofStep {
                description: "Telescoping cancellation (all terms cancel)".to_string(),
                before: simplified,
                after: final_result,
            });
            return TelescopingProofResult {
                success: true,
                steps,
                final_result: Some(final_result),
            };
        }

        return TelescopingProofResult {
            success: false,
            steps,
            final_result: Some(final_result),
        };
    }

    // Fallback: direct simplification.
    let result = simplify_expr(ctx, expr);
    TelescopingProofResult {
        success: is_zero_number(ctx, result),
        steps: vec![TelescopingProofStep {
            description: "Direct simplification".to_string(),
            before: expr,
            after: result,
        }],
        final_result: Some(result),
    }
}

fn is_zero_number(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(n) => n.is_zero(),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::telescope_with_simplify;

    #[test]
    fn telescope_with_simplify_detects_dirichlet_identity() {
        let mut ctx = cas_ast::Context::new();
        let expr = cas_parser::parse("1 + 2*cos(x) + 2*cos(2*x) - sin(5*x/2)/sin(x/2)", &mut ctx)
            .expect("parse");
        let out = telescope_with_simplify(
            &mut ctx,
            expr,
            |_ctx, id| id,
            |_ctx, id| format!("E{}", id.index()),
        );
        assert!(out.success);
        assert!(out
            .steps
            .first()
            .is_some_and(|s| s.description.contains("Dirichlet Kernel Identity")));
    }

    #[test]
    fn telescope_with_simplify_fallback_returns_single_step() {
        let mut ctx = cas_ast::Context::new();
        let expr = cas_parser::parse("x + 1", &mut ctx).expect("parse");
        let out = telescope_with_simplify(
            &mut ctx,
            expr,
            |_ctx, id| id,
            |_ctx, id| format!("E{}", id.index()),
        );
        assert_eq!(out.steps.len(), 1);
        assert_eq!(out.steps[0].description, "Direct simplification");
        assert!(!out.success);
    }
}
