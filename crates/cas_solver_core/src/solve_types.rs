//! Generic solver data models shared across runtime crates.

use cas_ast::{Context, Equation};

/// Diagnostics collected during solve operation.
#[derive(Debug, Clone, Default)]
pub struct SolveDiagnostics<Required, Assumed, AssumedRecord, Scope> {
    /// Conditions required by the equation structure (domain minimum).
    pub required: Vec<Required>,
    /// Assumptions made during solving (policy decisions).
    pub assumed: Vec<Assumed>,
    /// Deduplicated assumption summary records for external output.
    pub assumed_records: Vec<AssumedRecord>,
    /// Output scopes for context-aware display transforms.
    pub output_scopes: Vec<Scope>,
}

/// Educational sub-step for solver derivations (e.g., completing the square).
#[derive(Debug, Clone)]
pub struct SolveSubStep<Equation, Importance> {
    pub description: String,
    pub equation_after: Equation,
    pub importance: Importance,
}

impl<Equation, Importance> SolveSubStep<Equation, Importance> {
    /// Build one sub-step with explicit importance.
    pub fn new(
        description: impl Into<String>,
        equation_after: Equation,
        importance: Importance,
    ) -> Self {
        Self {
            description: description.into(),
            equation_after,
            importance,
        }
    }
}

/// Primary solver step with optional educational sub-steps.
#[derive(Debug, Clone)]
pub struct SolveStep<Equation, Importance, SubStep> {
    pub description: String,
    pub equation_after: Equation,
    pub importance: Importance,
    pub substeps: Vec<SubStep>,
}

impl<Equation, Importance, SubStep> SolveStep<Equation, Importance, SubStep> {
    /// Build one step with explicit importance.
    pub fn new(
        description: impl Into<String>,
        equation_after: Equation,
        importance: Importance,
    ) -> Self {
        Self {
            description: description.into(),
            equation_after,
            importance,
            substeps: vec![],
        }
    }

    /// Attach substeps (builder pattern).
    pub fn with_substeps(mut self, substeps: Vec<SubStep>) -> Self {
        self.substeps = substeps;
        self
    }
}

/// Cleanup + wrap solver steps for display consumption.
///
/// This canonicalizes step narration through the shared cleanup pipeline and
/// returns a display-enforced container.
pub fn cleanup_display_solve_steps<Importance, SubStep>(
    ctx: &mut Context,
    steps: Vec<SolveStep<Equation, Importance, SubStep>>,
    detailed: bool,
    var: &str,
) -> crate::display_steps::DisplaySteps<SolveStep<Equation, Importance, SubStep>>
where
    Importance: Clone,
    SubStep: Clone,
{
    crate::step_cleanup::cleanup_into_display_steps_by(
        ctx,
        steps,
        detailed,
        var,
        |step: &SolveStep<Equation, Importance, SubStep>| crate::step_cleanup::CleanupStep {
            description: step.description.clone(),
            equation_after: step.equation_after.clone(),
        },
        |template: SolveStep<Equation, Importance, SubStep>, payload| SolveStep::<
            Equation,
            Importance,
            SubStep,
        > {
            description: payload.description,
            equation_after: payload.equation_after,
            importance: template.importance,
            substeps: template.substeps,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::{SolveDiagnostics, SolveStep, SolveSubStep};
    use cas_ast::{Context, RelOp};

    #[test]
    fn diagnostics_default_is_empty() {
        let diag: SolveDiagnostics<u8, u16, u32, u64> = SolveDiagnostics::default();
        assert!(diag.required.is_empty());
        assert!(diag.assumed.is_empty());
        assert!(diag.assumed_records.is_empty());
        assert!(diag.output_scopes.is_empty());
    }

    #[test]
    fn step_and_substep_builders_preserve_payload() {
        let sub = SolveSubStep::new("inner", 11u32, 1u8);
        let step = SolveStep::new("outer", 22u32, 2u8).with_substeps(vec![sub]);

        assert_eq!(step.description, "outer");
        assert_eq!(step.equation_after, 22u32);
        assert_eq!(step.importance, 2u8);
        assert_eq!(step.substeps.len(), 1);
        assert_eq!(step.substeps[0].description, "inner");
        assert_eq!(step.substeps[0].equation_after, 11u32);
        assert_eq!(step.substeps[0].importance, 1u8);
    }

    #[test]
    fn cleanup_display_solve_steps_preserves_step_metadata() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let zero = ctx.num(0);
        let one = ctx.num(1);

        let steps = vec![
            SolveStep::new(
                "step 1",
                cas_ast::Equation {
                    lhs: x,
                    rhs: zero,
                    op: RelOp::Eq,
                },
                2u8,
            )
            .with_substeps(vec![SolveSubStep::new(
                "inner 1",
                cas_ast::Equation {
                    lhs: x,
                    rhs: zero,
                    op: RelOp::Eq,
                },
                1u8,
            )]),
            SolveStep::new(
                "step 2",
                cas_ast::Equation {
                    lhs: x,
                    rhs: one,
                    op: RelOp::Eq,
                },
                3u8,
            ),
        ];

        let out = super::cleanup_display_solve_steps(&mut ctx, steps, true, "x");
        assert_eq!(out.len(), 2);
        let cleaned = out.into_inner();
        assert_eq!(cleaned[0].importance, 2u8);
        assert_eq!(cleaned[0].substeps.len(), 1);
        assert_eq!(cleaned[1].importance, 3u8);
    }
}
