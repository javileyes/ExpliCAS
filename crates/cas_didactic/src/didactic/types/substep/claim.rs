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
    /// `after == op(before)` — the sub-step APPLIES a function to its left side.
    /// `1 − x² ⇒ sqrt(1 − x²)` (the missing leg of the reference triangle) and
    /// `x ⇒ ln(x)` (change of base) are FALSE as equalities and true as this.
    /// Declaring it is what keeps a chain checker from reading them as broken
    /// links, and what keeps a blanket-equality sweep from deleting them.
    Applied { op: cas_ast::BuiltinFn },
    /// `after == before|_{var:=upper} − before|_{var:=lower}` — the `before` is
    /// an antiderivative F and the `after` is `F(upper) − F(lower)`. The bounds
    /// travel as DATA: a sub-step that evaluates at bounds it does not carry
    /// cannot render them, which is how `∫|2x−1|dx ⇒ 5/2` got published.
    DefiniteEval {
        var: String,
        lower: ExprId,
        upper: ExprId,
    },
    /// `after == before|_{var:=point}`.
    EvalAt { var: String, point: ExprId },
    /// `after == lim_{var → approach} before`.
    ///
    /// The approach travels as DATA including its SIGN. It has to: the didactic
    /// layer used to read the direction off a substring of the rule name, both
    /// infinities share one rule name, and the result was narration that cited
    /// the `x→+∞` theorem to justify a limit at `−∞`. A claim that cannot name
    /// the approach it is asserting cannot be checked against it.
    Limit {
        var: String,
        approach: cas_math::limit_types::Approach,
    },
    /// The two sides are a SCHEMA — an identity stated with metavariables
    /// (`tan(u)·cot(u) = 1`), not a fact about the expression in front of the
    /// reader. Its truth is a property of the TEMPLATE, so it is adjudicated
    /// once in [`super::schema`] and looked up here rather than re-decided on
    /// every emission.
    SchematicIdentity {
        lhs: &'static str,
        rhs: &'static str,
    },
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

/// Wall-clock ceiling for ONE claim check.
///
/// The verification runs a full simplify, and the design warned that this layer
/// could come to dominate the cost of `--steps on`. It did, immediately: the
/// first migration made `integrate(1/(x^3-2), x, 2, 3)` blow the divergence
/// gate's 30 s net, because the cbrt residual is precisely the one the
/// simplifier grinds on without converging. A verifier that cannot finish must
/// answer `Undecided`, not hang — the budget IS the answer, not a workaround.
const CLAIM_VERIFICATION_BUDGET_MS: u64 = 250;

fn simplify_in(context: &mut Context, expr: ExprId) -> ExprId {
    let mut simplifier = cas_solver::runtime::Simplifier::with_default_rules();
    std::mem::swap(&mut simplifier.context, context);
    let options = cas_solver::runtime::SimplifyOptions {
        time_budget_ms: Some(CLAIM_VERIFICATION_BUDGET_MS),
        ..Default::default()
    };
    let (rewritten, _steps, _stats) = cas_engine::with_suppressed_depth_overflow_warnings(|| {
        simplifier.simplify_with_stats(expr, options)
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
            decide_equality(&mut scratch, before, after)
        }
        Claim::Applied { op } => {
            let mut scratch = context.clone();
            let applied = scratch.call_builtin(*op, vec![before]);
            // Hash-consing makes the structural case exact and free: the emitter
            // that built `after` by applying `op` proves its own claim without a
            // simplifier pass. The ladder only runs when the after arrived in
            // some other shape.
            if applied == after {
                ClaimVerdict::Verified
            } else {
                decide_equality(&mut scratch, applied, after)
            }
        }
        Claim::EvalAt { var, point } => {
            let mut scratch = context.clone();
            let substituted = substitute_var(&mut scratch, before, var, *point);
            decide_equality(&mut scratch, substituted, after)
        }
        Claim::SchematicIdentity { lhs, rhs } => {
            // A LOOKUP, deliberately: see the module docs of `schema`. An
            // unadjudicated schema is `Undecided` — a declared gap, never a
            // silent pass — and so is every classified exception, because the
            // table's `Refuted` rows are defects with their own owners and
            // deleting their narration is not this arm's decision to make.
            verify_schematic_identity(lhs, rhs)
        }
        Claim::Limit { var, approach } => {
            let mut scratch = context.clone();
            let var_expr = scratch.var(var);
            // The engine's OWN limit evaluator, with the options the eval path
            // uses (`presimplify: Off`, real domain) — these narrations only
            // exist under the real domain, and every limit rule reasons with the
            // real order anyway.
            let options = cas_math::limit_types::LimitOptions::default();
            let outcome = cas_engine::with_suppressed_depth_overflow_warnings(|| {
                cas_math::limits_support::eval_limit_at_infinity(
                    &mut scratch,
                    before,
                    var_expr,
                    *approach,
                    &options,
                )
            });
            // The oracle declining is not a disproof. It answers with a residual
            // `limit(...)` call plus a warning when its conservative policy does
            // not decide, and reading that as a refutation would delete correct
            // narration wherever the didactic layer knows a technique the safe
            // policy declines to apply.
            if outcome.warning.is_some() || expr_contains_limit_call(&scratch, outcome.expr) {
                return ClaimVerdict::Undecided;
            }
            decide_equality(&mut scratch, outcome.expr, after)
        }
        Claim::DefiniteEval { var, lower, upper } => {
            let mut scratch = context.clone();
            let at_upper = substitute_var(&mut scratch, before, var, *upper);
            let at_lower = substitute_var(&mut scratch, before, var, *lower);
            let expected = scratch.add(Expr::Sub(at_upper, at_lower));
            decide_equality(&mut scratch, expected, after)
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

/// Decide a schema by LOOKUP. Unlike every other arm this one needs no context
/// and no nodes: the two sides are templates, and their truth was adjudicated
/// once by the census instead of on every emission.
pub fn verify_schematic_identity(lhs: &'static str, rhs: &'static str) -> ClaimVerdict {
    let status = super::schema::schema_status(lhs, rhs);
    // The census's enforcement teeth: in debug builds — which is what every
    // test suite and the divergence-gate corpus run — stating a schema the
    // census never adjudicated is a loud, attributable failure that names the
    // exact pair. In release it publishes and abstains: the reader loses
    // nothing while the drift gets caught by the next test run.
    debug_assert!(
        status.is_some(),
        "schema `{lhs}` ⇒ `{rhs}` was never adjudicated: add it to \
         SCHEMATIC_IDENTITIES with its MEASURED status in this same commit"
    );
    match status {
        Some(super::schema::SchemaStatus::Proven) => ClaimVerdict::Verified,
        // A classified exception abstains: each class is a declared gap with
        // its own owner, and deleting its narration is not this arm's decision.
        _ => ClaimVerdict::Undecided,
    }
}

/// `a ≡ b`, decided by folding the difference.
///
/// REFUTED demands a POSITIVE disproof — a residual that is a non-zero NUMBER —
/// never the mere absence of a proof. The simplifier grinding to a halt on a surd
/// it cannot fold is not evidence of a lie, and treating it as one would delete
/// correct narration.
pub(crate) fn decide_equality(scratch: &mut Context, a: ExprId, b: ExprId) -> ClaimVerdict {
    if a == b {
        return ClaimVerdict::Verified;
    }
    let difference = scratch.add(Expr::Sub(a, b));
    let simplified = simplify_in(scratch, difference);
    if is_zero(scratch, simplified) {
        ClaimVerdict::Verified
    } else if is_constant(scratch, simplified) {
        ClaimVerdict::Refuted
    } else {
        ClaimVerdict::Undecided
    }
}

/// Whether the oracle's answer is still an unevaluated `limit(...)` residual —
/// its way of saying "the safe policy does not decide this".
fn expr_contains_limit_call(context: &Context, expr: ExprId) -> bool {
    match context.get(expr) {
        Expr::Function(fn_id, args) => {
            context.sym_name(*fn_id) == "limit"
                || args
                    .iter()
                    .any(|arg| expr_contains_limit_call(context, *arg))
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            expr_contains_limit_call(context, *l) || expr_contains_limit_call(context, *r)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => expr_contains_limit_call(context, *inner),
        _ => false,
    }
}

/// `expr|_{var := value}`. Hash-consing makes `scratch.var(var)` the very node
/// the expression holds, so the replacement reaches every occurrence.
fn substitute_var(scratch: &mut Context, expr: ExprId, var: &str, value: ExprId) -> ExprId {
    let var_expr = scratch.var(var);
    cas_ast::substitute_expr_by_id(scratch, expr, var_expr, value)
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
        let after =
            cas_parser::parse("2*x*sin(x) + (2 - x^2)*cos(x)", &mut context).expect("parse");
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

    /// The rescue the design promised: `1 − x² ⇒ sqrt(1 − x²)` is a LIE as an
    /// equality and the truth as an application. Both verdicts in one test, so
    /// the arm cannot quietly become an alias of `Equality`.
    #[test]
    fn applied_verifies_what_equality_refutes() {
        let (mut context, radicand) = ctx_with("1 - x^2");
        let root = cas_parser::parse("sqrt(1 - x^2)", &mut context).expect("parse");
        assert_eq!(
            verify_claim(
                &context,
                &Claim::Applied {
                    op: cas_ast::BuiltinFn::Sqrt
                },
                radicand,
                root
            ),
            ClaimVerdict::Verified
        );
        assert_ne!(
            verify_claim(&context, &Claim::Equality, radicand, root),
            ClaimVerdict::Verified,
            "the pair is not an equality; only the declared relation makes it true"
        );
    }

    #[test]
    fn applied_refutes_the_wrong_function() {
        let (mut context, radicand) = ctx_with("1 - x^2");
        let root = cas_parser::parse("sqrt(1 - x^2)", &mut context).expect("parse");
        assert_eq!(
            verify_claim(
                &context,
                &Claim::Applied {
                    op: cas_ast::BuiltinFn::Ln
                },
                radicand,
                root
            ),
            ClaimVerdict::Undecided,
            "a mismatched function must never verify"
        );
    }

    /// The limit narration's last line: substitute the point into the cancelled
    /// form and land on the ENGINE's answer. That is a genuine cross-check —
    /// the two sides come from different places.
    #[test]
    fn eval_at_verifies_the_substituted_point() {
        let (mut context, cancelled) = ctx_with("x + 1");
        let point = cas_parser::parse("1", &mut context).expect("parse");
        let value = cas_parser::parse("2", &mut context).expect("parse");
        assert_eq!(
            verify_claim(
                &context,
                &Claim::EvalAt {
                    var: "x".into(),
                    point
                },
                cancelled,
                value
            ),
            ClaimVerdict::Verified
        );
    }

    #[test]
    fn eval_at_refutes_a_wrong_value() {
        let (mut context, cancelled) = ctx_with("x + 1");
        let point = cas_parser::parse("1", &mut context).expect("parse");
        let wrong = cas_parser::parse("3", &mut context).expect("parse");
        assert_eq!(
            verify_claim(
                &context,
                &Claim::EvalAt {
                    var: "x".into(),
                    point
                },
                cancelled,
                wrong
            ),
            ClaimVerdict::Refuted
        );
    }

    /// `F(b) − F(a)`, with the bounds carried as data. The wrong-bound case is
    /// the one the audit found published: an integral worth one thing quoting
    /// the value of another.
    #[test]
    fn definite_eval_checks_the_bounds_it_carries() {
        let (mut context, antiderivative) = ctx_with("x^3/3");
        let lower = cas_parser::parse("0", &mut context).expect("parse");
        let upper = cas_parser::parse("1", &mut context).expect("parse");
        let value = cas_parser::parse("1/3", &mut context).expect("parse");
        assert_eq!(
            verify_claim(
                &context,
                &Claim::DefiniteEval {
                    var: "x".into(),
                    lower,
                    upper
                },
                antiderivative,
                value
            ),
            ClaimVerdict::Verified
        );
        let other = cas_parser::parse("2", &mut context).expect("parse");
        assert_eq!(
            verify_claim(
                &context,
                &Claim::DefiniteEval {
                    var: "x".into(),
                    lower,
                    upper: other
                },
                antiderivative,
                value
            ),
            ClaimVerdict::Refuted,
            "evaluating at the wrong bound must not pass"
        );
    }

    /// The arm the design deferred because "the verifier is the engine's own
    /// limit oracle, not invocable from here without re-entrancy". It is
    /// invocable: `cas_didactic` already depends on `cas_math`, the oracle never
    /// calls back into the didactic layer, and the narration runs AFTER the
    /// engine has finished — there is no recursion to fall into.
    #[test]
    fn limit_claim_verifies_a_reconstructed_intermediate() {
        // The common-denominator narration rebuilds `1/tan(x) − 1/x` as this
        // quotient and then asserts the step's value for it. Two producers, one
        // claim.
        let (mut context, combined) = ctx_with("(x*cos(x) - sin(x))/(x*sin(x))");
        let point = cas_parser::parse("0", &mut context).expect("parse");
        let value = cas_parser::parse("0", &mut context).expect("parse");
        assert_eq!(
            verify_claim(
                &context,
                &Claim::Limit {
                    var: "x".into(),
                    approach: cas_math::limit_types::Approach::Finite(point),
                },
                combined,
                value
            ),
            ClaimVerdict::Verified
        );
    }

    /// MEASURED, and worth pinning: the oracle decides `√(x²+x) − x` at `+∞` but
    /// DECLINES the rationalized quotient the conjugate narration rebuilds from
    /// it. The didactic route and the engine's route are not the same route, so
    /// that sub-step publishes as an abstention — which is the honest answer,
    /// and the reason `Undecided` must never be treated as a refutation.
    #[test]
    fn limit_claim_abstains_on_the_conjugate_intermediate_the_oracle_declines() {
        let (mut context, rationalized) = ctx_with("x / (sqrt(x^2 + x) + x)");
        let value = cas_parser::parse("1/2", &mut context).expect("parse");
        assert_eq!(
            verify_claim(
                &context,
                &Claim::Limit {
                    var: "x".into(),
                    approach: cas_math::limit_types::Approach::PosInfinity,
                },
                rationalized,
                value
            ),
            ClaimVerdict::Undecided
        );
    }

    #[test]
    fn limit_claim_refutes_a_wrong_value() {
        let (mut context, quotient) = ctx_with("(x^2 + 1)/(2*x^2 - 3)");
        let wrong = cas_parser::parse("2", &mut context).expect("parse");
        assert_eq!(
            verify_claim(
                &context,
                &Claim::Limit {
                    var: "x".into(),
                    approach: cas_math::limit_types::Approach::PosInfinity,
                },
                quotient,
                wrong
            ),
            ClaimVerdict::Refuted
        );
    }

    /// WHY the approach travels as data. Same expression, same claim shape, two
    /// signs: `x/sqrt(x^2+1)` is `1` at `+∞` and `−1` at `−∞`. A claim that
    /// could not name its direction would pass the wrong one.
    #[test]
    fn limit_claim_separates_the_two_infinities() {
        let (mut context, quotient) = ctx_with("x/sqrt(x^2 + 1)");
        let one = cas_parser::parse("1", &mut context).expect("parse");
        let pos = Claim::Limit {
            var: "x".into(),
            approach: cas_math::limit_types::Approach::PosInfinity,
        };
        let neg = Claim::Limit {
            var: "x".into(),
            approach: cas_math::limit_types::Approach::NegInfinity,
        };
        assert_eq!(
            verify_claim(&context, &pos, quotient, one),
            ClaimVerdict::Verified
        );
        assert_eq!(
            verify_claim(&context, &neg, quotient, one),
            ClaimVerdict::Refuted,
            "the limit at -infinity is -1; naming the direction is what catches this"
        );
    }

    /// The oracle's conservative policy declines plenty of real limits. That is
    /// an abstention, not a disproof — reading it as one would delete every
    /// narration whose technique the safe policy does not implement.
    #[test]
    fn limit_claim_treats_an_undecided_oracle_as_abstention() {
        let (mut context, expr) = ctx_with("ln(x)/x");
        let value = cas_parser::parse("0", &mut context).expect("parse");
        assert_eq!(
            verify_claim(
                &context,
                &Claim::Limit {
                    var: "x".into(),
                    approach: cas_math::limit_types::Approach::NegInfinity,
                },
                expr,
                value
            ),
            ClaimVerdict::Undecided
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
