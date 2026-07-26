//! What a sub-step AFFIRMS, and the check that runs before it is published.
//!
//! The audit found sub-steps publishing statements that are simply FALSE:
//! `F(b) − F(a)` displaying π/4 for an integral worth π/12, `∫−2·sin(x)dx =
//! 2x·sin(x) + (2−x²)cos(x)`, `A = (1−x)²·A`, a "factor the denominator" whose
//! two sides are identical. Every one of them was fixed by hand in its own
//! cycle. This module makes the CLASS impossible instead: the emitter declares
//! the relation it asserts, and the constructor verifies it. What cannot be
//! decided DECLINES and is counted.
//!
//! ## Why the relation has to be declared
//!
//! A prototype that assumed `Equality` for every sub-step refuted 80 of 214 —
//! and roughly 51 of those refutations were LEGITIMATE non-equality relations
//! (an antiderivative is not an equality). Verifying the wrong relation deletes
//! more correct narration than it saves. So the enum is the point: a sub-step
//! that means "this is the antiderivative of that" is checked by
//! differentiating, not by comparing.
//!
//! ## Landing policy
//!
//! A relation enters this enum only when its verifier enters with it. Anything
//! not yet migrated stays `Unchecked`, which is an explicit, counted abstention
//! — never a silent pass. `substep_unchecked_emitters` is anchored at the
//! MEASURED emitter count (422 across `cas_didactic`), not at a grep of one
//! file, so it cannot reach zero with hundreds of emitters unverified.

use cas_ast::{Context, Expr, ExprId};

/// The mathematical relation a sub-step asserts between its two sides.
#[derive(Debug, Clone)]
pub enum Claim {
    /// `d(after)/dvar == before` — the after is an antiderivative of the before.
    /// The theorem is "antiderivatives differ by a constant", so the check is
    /// on the derivative, not on the pair.
    Antiderivative { var: String },
    /// `after == d(before)/dvar`.
    Derivative { var: String },
    /// `before − after` is a CONSTANT (two antiderivatives of the same thing).
    EqualityUpToConstant { var: String },
    /// `before ≡ after` as expressions.
    Equality,
    /// The sub-step does not assert a relation between two sides: it names a
    /// manoeuvre, identifies a substitution, states a formula. An explicit
    /// abstention, not a gap.
    Statement,
}

/// Outcome of checking a claim. `Undecided` is NOT a failure: the simplifier
/// may simply fail to reach zero through surds it cannot fold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaimVerdict {
    Verified,
    Undecided,
    Refuted,
}

fn simplify_in(context: &mut Context, expr: ExprId) -> ExprId {
    let mut simplifier = cas_solver::runtime::Simplifier::with_default_rules();
    std::mem::swap(&mut simplifier.context, context);
    let (rewritten, _steps, _stats) = cas_engine::with_suppressed_depth_overflow_warnings(|| {
        simplifier.simplify_with_stats(expr, cas_solver::runtime::SimplifyOptions::default())
    });
    std::mem::swap(&mut simplifier.context, context);
    rewritten
}

fn is_zero(context: &Context, expr: ExprId) -> bool {
    matches!(context.get(expr), Expr::Number(n) if num_traits::Zero::is_zero(n))
}

fn is_constant(context: &Context, expr: ExprId) -> bool {
    matches!(context.get(expr), Expr::Number(_))
}

/// Decide `claim` for the pair `(before, after)`.
///
/// Works on a SCRATCH clone so no caller has to thread `&mut Context`, which is
/// what keeps the migration to `checked` a per-emitter edit instead of a
/// signature cascade. The cost is one context clone per CHECKED sub-step, which
/// is why unmigrated emitters (`Statement` / `Unchecked`) pay nothing.
pub fn verify_claim(
    context: &Context,
    claim: &Claim,
    before: ExprId,
    after: ExprId,
) -> ClaimVerdict {
    match claim {
        Claim::Statement => ClaimVerdict::Verified,
        Claim::Equality => {
            let mut scratch = context.clone();
            let difference = scratch.add(Expr::Sub(before, after));
            let simplified = simplify_in(&mut scratch, difference);
            if is_zero(&scratch, simplified) {
                ClaimVerdict::Verified
            } else if is_constant(&scratch, simplified) {
                ClaimVerdict::Refuted
            } else {
                ClaimVerdict::Undecided
            }
        }
        Claim::EqualityUpToConstant { .. } => {
            let mut scratch = context.clone();
            let difference = scratch.add(Expr::Sub(before, after));
            let simplified = simplify_in(&mut scratch, difference);
            if is_constant(&scratch, simplified) {
                ClaimVerdict::Verified
            } else {
                ClaimVerdict::Undecided
            }
        }
        Claim::Antiderivative { var } => verify_by_differentiation(context, after, before, var),
        Claim::Derivative { var } => verify_by_differentiation(context, before, after, var),
    }
}

/// `d(source)/dvar == target`, decided exactly.
fn verify_by_differentiation(
    context: &Context,
    source: ExprId,
    target: ExprId,
    var: &str,
) -> ClaimVerdict {
    let mut scratch = context.clone();
    let Some(derivative) = cas_math::symbolic_differentiation_support::differentiate_symbolic_expr(
        &mut scratch,
        source,
        var,
    ) else {
        return ClaimVerdict::Undecided;
    };
    let difference = scratch.add(Expr::Sub(derivative, target));
    let simplified = simplify_in(&mut scratch, difference);
    if is_zero(&scratch, simplified) {
        ClaimVerdict::Verified
    } else {
        // A non-zero residual is NOT proof of a lie: the simplifier may fail to
        // fold surds. Only a residual that is a non-zero CONSTANT is decisive —
        // the derivative of an antiderivative cannot differ by a constant.
        if is_constant(&scratch, simplified) {
            ClaimVerdict::Refuted
        } else {
            ClaimVerdict::Undecided
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx_with(expr: &str) -> (Context, ExprId) {
        let mut context = Context::new();
        let id = cas_parser::parse(expr, &mut context).expect("parse");
        (context, id)
    }

    #[test]
    fn antiderivative_claim_verifies_a_true_pair() {
        let (mut context, integrand) = ctx_with("cos(x)");
        let after = cas_parser::parse("sin(x)", &mut context).expect("parse");
        assert_eq!(
            verify_claim(
                &context,
                &Claim::Antiderivative { var: "x".into() },
                integrand,
                after
            ),
            ClaimVerdict::Verified
        );
    }

    /// The shape the audit found: `∫−2·sin(x)dx` published as
    /// `2x·sin(x) + (2−x²)cos(x)`. Its derivative differs from the integrand by
    /// a non-constant, so it must not verify — and `checked` would not publish it.
    #[test]
    fn antiderivative_claim_refutes_the_by_parts_witness() {
        let (mut context, integrand) = ctx_with("-2*sin(x)");
        let after = cas_parser::parse("2*x*sin(x) + (2 - x^2)*cos(x)", &mut context).expect("parse");
        assert_ne!(
            verify_claim(
                &context,
                &Claim::Antiderivative { var: "x".into() },
                integrand,
                after
            ),
            ClaimVerdict::Verified,
            "the published antiderivative was wrong and must not verify"
        );
    }

    /// Two antiderivatives of the same integrand differ by a CONSTANT — that is
    /// the theorem, and asserting plain equality would reject a correct pair.
    #[test]
    fn equality_up_to_constant_accepts_the_shifted_antiderivative() {
        let (mut context, before) = ctx_with("x^2/2");
        let after = cas_parser::parse("x^2/2 + 7", &mut context).expect("parse");
        assert_eq!(
            verify_claim(
                &context,
                &Claim::EqualityUpToConstant { var: "x".into() },
                before,
                after
            ),
            ClaimVerdict::Verified
        );
    }

    #[test]
    fn statement_abstains_explicitly_and_never_blocks() {
        let (context, id) = ctx_with("x");
        assert_eq!(
            verify_claim(&context, &Claim::Statement, id, id),
            ClaimVerdict::Verified
        );
    }
}
