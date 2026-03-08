use cas_ast::{Context, ExprId};

/// Unified interface for querying domain facts.
///
/// Outer crates choose their decision payload via `Decision`.
pub trait DomainOracle {
    type Decision;

    /// Query the strength of evidence for a predicate.
    fn query(
        &self,
        pred: &crate::domain_facts_model::Predicate,
    ) -> crate::domain_facts_model::FactStrength;

    /// Decide whether a transformation requiring this predicate is allowed.
    fn allows(&self, pred: &crate::domain_facts_model::Predicate) -> Self::Decision;
}

/// Hint gate kind used to route predicate checks into the right domain gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HintGateKind {
    Definability,
    Analytic,
}

/// Planned data for hint-aware domain gate evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredicateHintPlan {
    pub gate: HintGateKind,
    pub proof: crate::domain_proof::Proof,
    pub key: crate::assumption_model::AssumptionKey,
    pub expr: ExprId,
}

/// Query predicate strength using injected prover callbacks.
///
/// The callback contract is:
/// - `prove_nonzero(expr)` answers `expr != 0`
/// - `prove_positive(expr)` answers `expr > 0`
/// - `prove_nonnegative(expr)` answers `expr >= 0`
///
/// `Predicate::Defined(_)` currently maps to `Unknown`.
pub fn query_predicate_strength_with_provers<FNonZero, FPositive, FNonNegative>(
    pred: &crate::domain_facts_model::Predicate,
    mut prove_nonzero: FNonZero,
    mut prove_positive: FPositive,
    mut prove_nonnegative: FNonNegative,
) -> crate::domain_facts_model::FactStrength
where
    FNonZero: FnMut(ExprId) -> crate::domain_proof::Proof,
    FPositive: FnMut(ExprId) -> crate::domain_proof::Proof,
    FNonNegative: FnMut(ExprId) -> crate::domain_proof::Proof,
{
    use crate::domain_facts_model::{proof_to_strength, FactStrength, Predicate};

    match pred {
        Predicate::NonZero(expr) => proof_to_strength(prove_nonzero(*expr)),
        Predicate::Positive(expr) => proof_to_strength(prove_positive(*expr)),
        Predicate::NonNegative(expr) => proof_to_strength(prove_nonnegative(*expr)),
        Predicate::Defined(_) => FactStrength::Unknown,
    }
}

/// Resolve one predicate to a final cancel decision using prover callbacks.
///
/// This combines `query_predicate_strength_with_provers` and domain policy.
pub fn allows_with_provers<FNonZero, FPositive, FNonNegative>(
    mode: crate::domain_mode::DomainMode,
    pred: &crate::domain_facts_model::Predicate,
    prove_nonzero: FNonZero,
    prove_positive: FPositive,
    prove_nonnegative: FNonNegative,
) -> crate::domain_cancel_decision::CancelDecision
where
    FNonZero: FnMut(ExprId) -> crate::domain_proof::Proof,
    FPositive: FnMut(ExprId) -> crate::domain_proof::Proof,
    FNonNegative: FnMut(ExprId) -> crate::domain_proof::Proof,
{
    let strength = query_predicate_strength_with_provers(
        pred,
        prove_nonzero,
        prove_positive,
        prove_nonnegative,
    );
    crate::domain_gate::decide(mode, pred, strength)
}

/// Build a hint-routing plan for one predicate using prover callbacks.
///
/// This keeps predicate-to-key/proof routing centralized and reusable between
/// solver/engine layers.
pub fn build_predicate_hint_plan_with_provers<FNonZero, FPositive, FNonNegative>(
    ctx: &Context,
    pred: &crate::domain_facts_model::Predicate,
    mut prove_nonzero: FNonZero,
    mut prove_positive: FPositive,
    mut prove_nonnegative: FNonNegative,
) -> PredicateHintPlan
where
    FNonZero: FnMut(ExprId) -> crate::domain_proof::Proof,
    FPositive: FnMut(ExprId) -> crate::domain_proof::Proof,
    FNonNegative: FnMut(ExprId) -> crate::domain_proof::Proof,
{
    use crate::assumption_model::{expr_fingerprint, AssumptionKey};
    use crate::domain_facts_model::Predicate;
    use crate::domain_proof::Proof;

    let expr = pred.expr();
    match pred {
        Predicate::NonZero(_) => PredicateHintPlan {
            gate: HintGateKind::Definability,
            proof: prove_nonzero(expr),
            key: AssumptionKey::nonzero_key(ctx, expr),
            expr,
        },
        Predicate::Positive(_) => PredicateHintPlan {
            gate: HintGateKind::Analytic,
            proof: prove_positive(expr),
            key: AssumptionKey::positive_key(ctx, expr),
            expr,
        },
        Predicate::NonNegative(_) => PredicateHintPlan {
            gate: HintGateKind::Analytic,
            proof: prove_nonnegative(expr),
            key: AssumptionKey::nonnegative_key(ctx, expr),
            expr,
        },
        Predicate::Defined(_) => PredicateHintPlan {
            gate: HintGateKind::Definability,
            proof: Proof::Unknown,
            key: AssumptionKey::Defined {
                expr_fingerprint: expr_fingerprint(ctx, expr),
            },
            expr,
        },
    }
}

/// Resolve a predicate through hint-aware gates using injected provers and gate handlers.
///
/// This centralizes routing logic so outer crates only provide:
/// - prover callbacks
/// - the two domain gates (definability / analytic)
#[allow(clippy::too_many_arguments)]
pub fn resolve_with_hint_plan_and_provers<
    FNonZero,
    FPositive,
    FNonNegative,
    FDefinability,
    FAnalytic,
    TDecision,
>(
    ctx: &Context,
    pred: &crate::domain_facts_model::Predicate,
    rule: &'static str,
    prove_nonzero: FNonZero,
    prove_positive: FPositive,
    prove_nonnegative: FNonNegative,
    mut on_definability: FDefinability,
    mut on_analytic: FAnalytic,
) -> TDecision
where
    FNonZero: FnMut(ExprId) -> crate::domain_proof::Proof,
    FPositive: FnMut(ExprId) -> crate::domain_proof::Proof,
    FNonNegative: FnMut(ExprId) -> crate::domain_proof::Proof,
    FDefinability: FnMut(
        crate::domain_proof::Proof,
        crate::assumption_model::AssumptionKey,
        ExprId,
        &'static str,
    ) -> TDecision,
    FAnalytic: FnMut(
        crate::domain_proof::Proof,
        crate::assumption_model::AssumptionKey,
        ExprId,
        &'static str,
    ) -> TDecision,
{
    let plan = build_predicate_hint_plan_with_provers(
        ctx,
        pred,
        prove_nonzero,
        prove_positive,
        prove_nonnegative,
    );

    match plan.gate {
        HintGateKind::Definability => on_definability(plan.proof, plan.key, plan.expr, rule),
        HintGateKind::Analytic => on_analytic(plan.proof, plan.key, plan.expr, rule),
    }
}

/// Canonical hint-aware `allows()` using only prover callbacks.
///
/// This helper centralizes the full routing:
/// predicate -> proof/key plan -> definability/analytic gate.
pub fn allows_with_hint_using_provers<FNonZero, FPositive, FNonNegative>(
    ctx: &Context,
    mode: crate::domain_mode::DomainMode,
    pred: &crate::domain_facts_model::Predicate,
    rule: &'static str,
    prove_nonzero: FNonZero,
    prove_positive: FPositive,
    prove_nonnegative: FNonNegative,
) -> crate::domain_cancel_decision::CancelDecision
where
    FNonZero: FnMut(ExprId) -> crate::domain_proof::Proof,
    FPositive: FnMut(ExprId) -> crate::domain_proof::Proof,
    FNonNegative: FnMut(ExprId) -> crate::domain_proof::Proof,
{
    resolve_with_hint_plan_and_provers(
        ctx,
        pred,
        rule,
        prove_nonzero,
        prove_positive,
        prove_nonnegative,
        |proof, key, expr, local_rule| {
            crate::domain_gate::can_cancel_factor_with_hint(mode, proof, key, expr, local_rule)
        },
        |proof, key, expr, local_rule| {
            crate::domain_gate::can_apply_analytic_with_hint(mode, proof, key, expr, local_rule)
        },
    )
}

#[cfg(test)]
mod tests {
    use super::{
        allows_with_hint_using_provers, allows_with_provers,
        build_predicate_hint_plan_with_provers, query_predicate_strength_with_provers,
        resolve_with_hint_plan_and_provers, HintGateKind,
    };
    use crate::assumption_model::AssumptionKey;
    use crate::domain_facts_model::{FactStrength, Predicate};
    use crate::domain_mode::DomainMode;
    use crate::domain_proof::Proof;

    #[test]
    fn query_predicate_strength_routes_to_expected_prover() {
        let mut ctx = cas_ast::Context::default();
        let x = ctx.var("x");

        let nz = query_predicate_strength_with_provers(
            &Predicate::NonZero(x),
            |_expr| Proof::Proven,
            |_expr| Proof::Unknown,
            |_expr| Proof::Unknown,
        );
        assert_eq!(nz, FactStrength::Proven);

        let pos = query_predicate_strength_with_provers(
            &Predicate::Positive(x),
            |_expr| Proof::Unknown,
            |_expr| Proof::Disproven,
            |_expr| Proof::Unknown,
        );
        assert_eq!(pos, FactStrength::Disproven);

        let nonneg = query_predicate_strength_with_provers(
            &Predicate::NonNegative(x),
            |_expr| Proof::Unknown,
            |_expr| Proof::Unknown,
            |_expr| Proof::Unknown,
        );
        assert_eq!(nonneg, FactStrength::Unknown);

        let defined = query_predicate_strength_with_provers(
            &Predicate::Defined(x),
            |_expr| Proof::Proven,
            |_expr| Proof::Proven,
            |_expr| Proof::Proven,
        );
        assert_eq!(defined, FactStrength::Unknown);
    }

    #[test]
    fn hint_plan_maps_predicate_to_gate_key_and_proof() {
        let mut ctx = cas_ast::Context::default();
        let x = ctx.var("x");

        let nonzero = build_predicate_hint_plan_with_provers(
            &ctx,
            &Predicate::NonZero(x),
            |_expr| Proof::Unknown,
            |_expr| Proof::Proven,
            |_expr| Proof::Proven,
        );
        assert_eq!(nonzero.gate, HintGateKind::Definability);
        assert_eq!(nonzero.proof, Proof::Unknown);
        assert!(matches!(nonzero.key, AssumptionKey::NonZero { .. }));
        assert_eq!(nonzero.expr, x);

        let positive = build_predicate_hint_plan_with_provers(
            &ctx,
            &Predicate::Positive(x),
            |_expr| Proof::Unknown,
            |_expr| Proof::Proven,
            |_expr| Proof::Unknown,
        );
        assert_eq!(positive.gate, HintGateKind::Analytic);
        assert_eq!(positive.proof, Proof::Proven);
        assert!(matches!(positive.key, AssumptionKey::Positive { .. }));

        let nonnegative = build_predicate_hint_plan_with_provers(
            &ctx,
            &Predicate::NonNegative(x),
            |_expr| Proof::Unknown,
            |_expr| Proof::Unknown,
            |_expr| Proof::Disproven,
        );
        assert_eq!(nonnegative.gate, HintGateKind::Analytic);
        assert_eq!(nonnegative.proof, Proof::Disproven);
        assert!(matches!(nonnegative.key, AssumptionKey::NonNegative { .. }));

        let defined = build_predicate_hint_plan_with_provers(
            &ctx,
            &Predicate::Defined(x),
            |_expr| Proof::Proven,
            |_expr| Proof::Proven,
            |_expr| Proof::Proven,
        );
        assert_eq!(defined.gate, HintGateKind::Definability);
        assert_eq!(defined.proof, Proof::Unknown);
        assert!(matches!(defined.key, AssumptionKey::Defined { .. }));
    }

    #[test]
    fn resolve_with_hint_plan_routes_to_expected_gate_handler() {
        let mut ctx = cas_ast::Context::default();
        let x = ctx.var("x");

        let def = resolve_with_hint_plan_and_provers(
            &ctx,
            &Predicate::NonZero(x),
            "RuleA",
            |_expr| Proof::Unknown,
            |_expr| Proof::Proven,
            |_expr| Proof::Proven,
            |_proof, _key, _expr, rule| format!("def:{rule}"),
            |_proof, _key, _expr, rule| format!("ana:{rule}"),
        );
        assert_eq!(def, "def:RuleA");

        let ana = resolve_with_hint_plan_and_provers(
            &ctx,
            &Predicate::Positive(x),
            "RuleB",
            |_expr| Proof::Unknown,
            |_expr| Proof::Unknown,
            |_expr| Proof::Proven,
            |_proof, _key, _expr, rule| format!("def:{rule}"),
            |_proof, _key, _expr, rule| format!("ana:{rule}"),
        );
        assert_eq!(ana, "ana:RuleB");
    }

    #[test]
    fn allows_with_hint_using_provers_matches_mode_policy() {
        let mut ctx = cas_ast::Context::default();
        let x = ctx.var("x");

        let strict_block = allows_with_hint_using_provers(
            &ctx,
            DomainMode::Strict,
            &Predicate::NonZero(x),
            "RuleA",
            |_expr| Proof::Unknown,
            |_expr| Proof::Unknown,
            |_expr| Proof::Unknown,
        );
        assert!(!strict_block.allow);
        assert!(strict_block.blocked_hint.is_some());

        let generic_allow = allows_with_hint_using_provers(
            &ctx,
            DomainMode::Generic,
            &Predicate::NonZero(x),
            "RuleA",
            |_expr| Proof::Unknown,
            |_expr| Proof::Unknown,
            |_expr| Proof::Unknown,
        );
        assert!(generic_allow.allow);
        assert_eq!(generic_allow.assumed_keys.len(), 1);
    }

    #[test]
    fn allows_with_provers_maps_to_domain_policy() {
        let mut ctx = cas_ast::Context::default();
        let x = ctx.var("x");

        let strict_nonzero_unknown = allows_with_provers(
            DomainMode::Strict,
            &Predicate::NonZero(x),
            |_expr| Proof::Unknown,
            |_expr| Proof::Unknown,
            |_expr| Proof::Unknown,
        );
        assert!(!strict_nonzero_unknown.allow);

        let generic_nonzero_unknown = allows_with_provers(
            DomainMode::Generic,
            &Predicate::NonZero(x),
            |_expr| Proof::Unknown,
            |_expr| Proof::Unknown,
            |_expr| Proof::Unknown,
        );
        assert!(generic_nonzero_unknown.allow);
        assert!(generic_nonzero_unknown.assumption.is_some());
    }
}
