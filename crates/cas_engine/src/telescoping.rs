use crate::build::mul2_raw;
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

use cas_ast::{Context, Expr, ExprId};
use cas_formatter::DisplayExpr;
use num_traits::Zero;

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
    if let Some(result) = cas_math::telescoping_dirichlet::try_dirichlet_kernel_identity(ctx, expr)
    {
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
    let multiplier = cas_math::telescoping_support::find_denominator_for_clearing(ctx, expr);

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

pub use cas_math::telescoping_dirichlet::DirichletKernelResult;

/// Try to detect Dirichlet kernel identity pattern (public interface for orchestrator).
pub fn try_dirichlet_kernel_identity_pub(
    ctx: &Context,
    expr: ExprId,
) -> Option<DirichletKernelResult> {
    cas_math::telescoping_dirichlet::try_dirichlet_kernel_identity(ctx, expr)
}
