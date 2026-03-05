//! Telescoping strategy facade.
//!
//! Keeps the existing `cas_solver` API while owning the implementation
//! locally during migration.

use cas_ast::{Context, ExprId};
use cas_formatter::DisplayExpr;

/// Result of telescoping analysis.
pub struct TelescopingResult {
    pub success: bool,
    pub steps: Vec<TelescopingStep>,
    pub final_result: Option<ExprId>,
}

/// A step in the telescoping proof.
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

/// Attempt to prove an identity using telescoping strategy.
pub fn telescope(ctx: &mut Context, expr: ExprId) -> TelescopingResult {
    let proof = cas_math::telescoping_proof::telescope_with_simplify(
        ctx,
        expr,
        |source_ctx, source_expr| {
            let mut simplifier = crate::Simplifier::with_default_rules();
            simplifier.context = source_ctx.clone();
            let (result, _) = simplifier.simplify(source_expr);
            *source_ctx = simplifier.context;
            result
        },
        |source_ctx, id| {
            format!(
                "{}",
                DisplayExpr {
                    context: source_ctx,
                    id
                }
            )
        },
    );

    TelescopingResult {
        success: proof.success,
        steps: proof
            .steps
            .into_iter()
            .map(|step| TelescopingStep {
                description: step.description,
                before: step.before,
                after: step.after,
            })
            .collect(),
        final_result: proof.final_result,
    }
}
