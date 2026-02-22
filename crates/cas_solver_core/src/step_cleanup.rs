use crate::sign_normalize::{cleanup_step_description, normalize_expr_signs};
use cas_ast::{Context, Equation};
use cas_math::expr_predicates::is_zero_expr as is_zero;

/// Generic step payload for cleanup operations.
#[derive(Debug, Clone)]
pub struct CleanupStep {
    pub description: String,
    pub equation_after: Equation,
}

#[derive(Clone)]
struct CleanupEnvelope<T> {
    step: T,
    payload: CleanupStep,
}

/// Remove consecutive steps where extracted equations are identical.
pub fn remove_duplicate_equations_by<T, FEq>(steps: Vec<T>, mut eq_of: FEq) -> Vec<T>
where
    T: Clone,
    FEq: FnMut(&T) -> &Equation,
{
    if steps.len() < 2 {
        return steps;
    }

    let mut result = Vec::with_capacity(steps.len());
    result.push(steps[0].clone());

    for i in 1..steps.len() {
        let prev = &steps[i - 1];
        let curr = &steps[i];
        let prev_eq = eq_of(prev);
        let curr_eq = eq_of(curr);

        if prev_eq.lhs == curr_eq.lhs && prev_eq.rhs == curr_eq.rhs {
            continue;
        }

        result.push(curr.clone());
    }

    result
}

/// Remove consecutive duplicate equations for `CleanupStep`.
pub fn remove_duplicate_equations(steps: Vec<CleanupStep>) -> Vec<CleanupStep> {
    remove_duplicate_equations_by(steps, |s| &s.equation_after)
}

/// Remove redundant "normalize to zero then immediately undo" step pairs.
pub fn remove_redundant_steps_by<T, FDesc, FEq>(
    ctx: &Context,
    steps: Vec<T>,
    mut desc_of: FDesc,
    mut eq_of: FEq,
) -> Vec<T>
where
    T: Clone,
    FDesc: FnMut(&T) -> &str,
    FEq: FnMut(&T) -> &Equation,
{
    if steps.len() < 2 {
        return steps;
    }

    let mut result = Vec::with_capacity(steps.len());
    let mut i = 0;

    while i < steps.len() {
        let current = &steps[i];

        if i + 1 < steps.len() {
            let next = &steps[i + 1];
            if is_step_normalize_to_zero(ctx, current, &mut eq_of)
                && is_step_undo_normalization(ctx, current, next, &mut desc_of, &mut eq_of)
            {
                i += 1;
                continue;
            }
        }

        result.push(current.clone());
        i += 1;
    }

    result
}

/// Remove redundant steps for `CleanupStep`.
pub fn remove_redundant_steps(ctx: &Context, steps: Vec<CleanupStep>) -> Vec<CleanupStep> {
    remove_redundant_steps_by(
        ctx,
        steps,
        |s| s.description.as_str(),
        |s| &s.equation_after,
    )
}

/// End-to-end cleanup pipeline over arbitrary step payloads.
///
/// This applies:
/// 1. Redundant-step removal
/// 2. Log-linear didactic rewrite
/// 3. Sign normalization + description cleanup
/// 4. Duplicate-equation removal
///
/// The caller controls how to extract/rebuild payloads from custom step types.
pub fn cleanup_steps_by<T, FExtract, FRebuild>(
    ctx: &mut Context,
    steps: Vec<T>,
    detailed: bool,
    var: &str,
    mut extract: FExtract,
    mut rebuild: FRebuild,
) -> Vec<T>
where
    T: Clone,
    FExtract: FnMut(&T) -> CleanupStep,
    FRebuild: FnMut(T, CleanupStep) -> T,
{
    if steps.is_empty() {
        return steps;
    }

    let envelopes: Vec<CleanupEnvelope<T>> = steps
        .into_iter()
        .map(|step| CleanupEnvelope {
            payload: extract(&step),
            step,
        })
        .collect();

    let filtered = remove_redundant_steps_by(
        ctx,
        envelopes,
        |s| s.payload.description.as_str(),
        |s| &s.payload.equation_after,
    );

    let narrated = crate::log_linear_narration::rewrite_log_linear_steps_by(
        ctx,
        filtered,
        detailed,
        var,
        |s| s.payload.description.as_str(),
        |s| &s.payload.equation_after,
        |template, payload| CleanupEnvelope {
            step: template.step.clone(),
            payload: CleanupStep {
                description: payload.description,
                equation_after: payload.equation_after,
            },
        },
    );

    let normalized: Vec<CleanupEnvelope<T>> = narrated
        .into_iter()
        .map(|mut s| {
            s.payload.equation_after = Equation {
                lhs: normalize_expr_signs(ctx, s.payload.equation_after.lhs),
                rhs: normalize_expr_signs(ctx, s.payload.equation_after.rhs),
                op: s.payload.equation_after.op,
            };
            s.payload.description = cleanup_step_description(&s.payload.description);
            s
        })
        .collect();

    let deduped = remove_duplicate_equations_by(normalized, |s| &s.payload.equation_after);

    deduped
        .into_iter()
        .map(|s| rebuild(s.step, s.payload))
        .collect()
}

/// End-to-end cleanup for plain `CleanupStep` payloads.
pub fn cleanup_steps(
    ctx: &mut Context,
    steps: Vec<CleanupStep>,
    detailed: bool,
    var: &str,
) -> Vec<CleanupStep> {
    cleanup_steps_by(
        ctx,
        steps,
        detailed,
        var,
        |s| s.clone(),
        |_step, payload| payload,
    )
}

fn is_step_normalize_to_zero<T, FEq>(ctx: &Context, step: &T, eq_of: &mut FEq) -> bool
where
    FEq: FnMut(&T) -> &Equation,
{
    is_zero(ctx, eq_of(step).rhs)
}

fn is_step_undo_normalization<T, FDesc, FEq>(
    ctx: &Context,
    prev: &T,
    curr: &T,
    desc_of: &mut FDesc,
    eq_of: &mut FEq,
) -> bool
where
    FDesc: FnMut(&T) -> &str,
    FEq: FnMut(&T) -> &Equation,
{
    let prev_rhs_zero = is_zero(ctx, eq_of(prev).rhs);
    let curr_rhs_zero = is_zero(ctx, eq_of(curr).rhs);

    if prev_rhs_zero && !curr_rhs_zero {
        let prev_desc = desc_of(prev);
        let curr_desc = desc_of(curr);
        if prev_desc.contains("Subtract")
            && curr_desc.contains("Subtract")
            && (curr_desc.contains("-(") || curr_desc.contains("- -("))
        {
            return true;
        }

        if curr_desc == "Move terms to one side" {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::{Expr, RelOp};

    fn mk_step(desc: &str, lhs: cas_ast::ExprId, rhs: cas_ast::ExprId) -> CleanupStep {
        CleanupStep {
            description: desc.to_string(),
            equation_after: Equation {
                lhs,
                rhs,
                op: RelOp::Eq,
            },
        }
    }

    #[test]
    fn remove_duplicate_equations_drops_consecutive_duplicates() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let zero = ctx.num(0);
        let one = ctx.num(1);

        let steps = vec![
            mk_step("a", x, zero),
            mk_step("b", x, zero),
            mk_step("c", x, one),
        ];

        let out = remove_duplicate_equations(steps);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].description, "a");
        assert_eq!(out[1].description, "c");
    }

    #[test]
    fn remove_redundant_steps_skips_zero_normalization_when_immediately_undone() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let zero = ctx.num(0);
        let rhs = ctx.add(Expr::Sub(y, x));

        let steps = vec![
            mk_step("Subtract y from both sides", x, zero),
            mk_step("Subtract -(x) from both sides", x, rhs),
        ];

        let out = remove_redundant_steps(&ctx, steps);
        assert_eq!(out.len(), 1);
        assert!(out[0].description.contains("Subtract -("));
    }

    #[test]
    fn remove_duplicate_equations_by_preserves_payload_type() {
        #[derive(Clone)]
        struct RichStep {
            step: CleanupStep,
            payload: usize,
        }

        let mut ctx = Context::new();
        let x = ctx.var("x");
        let zero = ctx.num(0);
        let one = ctx.num(1);
        let steps = vec![
            RichStep {
                step: mk_step("a", x, zero),
                payload: 1,
            },
            RichStep {
                step: mk_step("b", x, zero),
                payload: 2,
            },
            RichStep {
                step: mk_step("c", x, one),
                payload: 3,
            },
        ];

        let out = remove_duplicate_equations_by(steps, |s| &s.step.equation_after);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].payload, 1);
        assert_eq!(out[1].payload, 3);
    }

    #[test]
    fn cleanup_steps_by_preserves_metadata() {
        #[derive(Clone)]
        struct RichStep {
            description: String,
            equation_after: Equation,
            marker: usize,
        }

        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let z = ctx.var("z");
        let eq = Equation {
            lhs: x,
            rhs: y,
            op: RelOp::Eq,
        };

        let steps = vec![
            RichStep {
                description: crate::log_linear_narration::TAKE_LOG_BOTH_SIDES_STEP.to_string(),
                equation_after: eq.clone(),
                marker: 7,
            },
            RichStep {
                description: "Collect terms in x".to_string(),
                equation_after: Equation {
                    lhs: x,
                    rhs: z,
                    op: RelOp::Eq,
                },
                marker: 9,
            },
        ];

        let out = cleanup_steps_by(
            &mut ctx,
            steps,
            false,
            "x",
            |s| CleanupStep {
                description: s.description.clone(),
                equation_after: s.equation_after.clone(),
            },
            |template, payload| RichStep {
                description: payload.description,
                equation_after: payload.equation_after,
                marker: template.marker,
            },
        );

        assert_eq!(out.len(), 2);
        assert_eq!(out[0].marker, 7);
        assert_eq!(out[1].marker, 9);
        assert_eq!(out[1].description, "Collect and factor x terms");
    }

    #[test]
    fn cleanup_steps_normalizes_description() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let neg_x = ctx.add(Expr::Neg(x));
        let zero = ctx.num(0);
        let step = CleanupStep {
            description: "Subtract -(x) from both sides".to_string(),
            equation_after: Equation {
                lhs: x,
                rhs: ctx.add(Expr::Sub(zero, neg_x)),
                op: RelOp::Eq,
            },
        };

        let out = cleanup_steps(&mut ctx, vec![step], false, "x");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].description, "Move terms to one side");
    }
}
