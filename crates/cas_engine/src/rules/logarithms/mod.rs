//! Logarithm rules: evaluation, properties, inverse composition, and auto-expansion.
//!
//! This module is split into submodules:
//! - `properties`: Domain-aware expansion, contraction, abs/power rules, chain product
//! - `inverse`: Inverse composition rules (exp↔log), auto-expand log

mod inverse;
mod properties;

pub use inverse::{
    AutoExpandLogRule, ExponentialLogRule, LogExpInverseRule, LogInversePowerRule,
    LogPowerBaseRule, SplitLogExponentsRule,
};
pub(crate) use properties::expand_logs_with_assumptions;
pub use properties::{
    LogAbsPowerRule, LogAbsSimplifyRule, LogChainProductRule, LogEvenPowerWithChainedAbsRule,
    LogExpansionRule,
};

use crate::define_rule;
use crate::rule::Rewrite;
use cas_math::logarithm_inverse_support::{
    try_rewrite_evaluate_log_expr, try_rewrite_ln_e_div_expr, try_rewrite_ln_e_product_expr,
    try_rewrite_log_perfect_square_expr,
};

define_rule!(EvaluateLogRule, "Evaluate Logarithms", |ctx, expr| {
    let planned = try_rewrite_evaluate_log_expr(ctx, expr)?;
    let mut rewrite = Rewrite::new(planned.rewritten).desc(planned.desc);
    if let Some(positive_subject) = planned.assume_positive_base {
        rewrite = rewrite.assume(crate::AssumptionEvent::positive_assumed(
            ctx,
            positive_subject,
        ));
    }
    Some(rewrite)
});

// =============================================================================
// LnEProductRule: ln(e*x) → 1 + ln(x)
// =============================================================================
// This is a SAFE, targeted expansion because ln(e) = 1 is a known constant.
// Unlike general LogExpansionRule, this doesn't risk explosion because it only
// extracts the known constant `e` factor.
//
// This enables residuals like `2*ln(e*u) - 2 - 2*ln(u)` to simplify:
// → 2*(1 + ln(u)) - 2 - 2*ln(u) → 2 + 2*ln(u) - 2 - 2*ln(u) → 0
// =============================================================================
define_rule!(LnEProductRule, "Factor e from ln Product", |ctx, expr| {
    let rewrite = try_rewrite_ln_e_product_expr(ctx, expr)?;
    Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
});

// =============================================================================
// LnEDivRule: ln(x/e) → ln(x) - 1, ln(e/x) → 1 - ln(x)
// =============================================================================
// Companion to LnEProductRule. These are SAFE expansions because ln(e) = 1.
// This enables residuals like `2*ln(u/e) - 2*(ln(u)-1)` to simplify to 0.
// =============================================================================
define_rule!(LnEDivRule, "Factor e from ln Quotient", |ctx, expr| {
    let rewrite = try_rewrite_ln_e_div_expr(ctx, expr)?;
    Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
});

/// LogContractionRule: Contracts sums/differences of logs into single logs.
/// - ln(a) + ln(b) → ln(a*b)
/// - ln(a) - ln(b) → ln(a/b)
/// - log(b, x) + log(b, y) → log(b, x*y)  (same base required)
/// - log(b, x) - log(b, y) → log(b, x/y)
///
/// This rule REDUCES node count and is a valid simplification.
/// Unlike LogExpansionRule, this is registered by default.
pub struct LogContractionRule;

impl crate::rule::Rule for LogContractionRule {
    fn name(&self) -> &str {
        "Log Contraction"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        use crate::semantics::NormalFormGoal;

        // GATE: Don't contract logs when goal is ExpandedLog
        // This prevents undoing the effect of expand_log command
        if parent_ctx.goal() == NormalFormGoal::ExpandedLog {
            return None;
        }

        // GATE: Don't contract logs when in auto-expand mode
        // This prevents cycle with AutoExpandLogRule (expand→contract→expand→...)
        if parent_ctx.is_auto_expand() || parent_ctx.in_auto_expand_context() {
            return None;
        }

        let rewrite =
            cas_math::logarithm_inverse_support::try_rewrite_log_contraction_expr(ctx, expr)?;
        Some(crate::rule::Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::ADD_SUB)
    }
}

// =============================================================================
// LogPerfectSquareRule: ln(f²) → 2·ln(|f|)
// =============================================================================
// Detects perfect-square arguments of log functions and factorizes them.
//
// Sub-detectors:
//   1. Polynomial trinomial: ln(4u²+12u+9) → 2·ln(|2u+3|) via discriminant=0
//   2. Trinomial via AST: ln(u⁴+2u²+1) → 2·ln(|u²+1|)
//   3. Even power: ln(u⁴) → 2·ln(|u²|)
//   4. Monomial: ln(4u²) → 2·ln(|2u|)
//   5. Div of squares: ln(u²/v²) → 2·ln(|u/v|)
//
// Domain: always wraps factor in |·| (safe); downstream |x|→x handles removal.
// =============================================================================
define_rule!(
    LogPerfectSquareRule,
    "Factor Perfect Square in Logarithm",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        let rewrite = try_rewrite_log_perfect_square_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

pub fn register(simplifier: &mut crate::Simplifier) {
    // V2.14.45: LogPowerBaseRule MUST come BEFORE LogEvenPowerRule
    // Otherwise log(x^2, x^6) gets expanded to 6·log(x², |x|) before simplifying to 3
    simplifier.add_rule(Box::new(LogPowerBaseRule)); // log(a^m, a^n) → n/m

    // LogAbsPowerRule: ln(|u|^n) → n·ln(|u|) - highest priority (15)
    // MUST come BEFORE AbsSquareRule which would turn |u|^2 → u^2 and lose the abs
    simplifier.add_rule(Box::new(LogAbsPowerRule));

    // V2.14.20: LogEvenPowerWithChainedAbsRule handles ln(x^even) with ChainedRewrite
    // Has higher priority (10) than EvaluateLogRule (0) so matches first
    simplifier.add_rule(Box::new(LogEvenPowerWithChainedAbsRule));

    // LogPerfectSquareRule: ln(A² ± 2AB + B²) → 2·ln(|A ± B|)
    // Factor perfect-square trinomials inside log arguments (e.g. u⁴+2u²+1 → (u²+1)²)
    simplifier.add_rule(Box::new(LogPerfectSquareRule));

    simplifier.add_rule(Box::new(EvaluateLogRule));

    // LnEProductRule: ln(e*x) → 1 + ln(x) - safe targeted expansion
    // This enables residuals like `2*ln(e*u) - 2 - 2*ln(u)` to simplify to 0
    simplifier.add_rule(Box::new(LnEProductRule));
    // LnEDivRule: ln(x/e) → ln(x) - 1, ln(e/x) → 1 - ln(x)
    // Companion to LnEProductRule for quotient cases
    simplifier.add_rule(Box::new(LnEDivRule));

    // NOTE: LogExpansionRule removed from auto-registration.
    // Log expansion increases node count (ln(xy) → ln(x) + ln(y)) and is not always desirable.
    // Use the `expand_log` command for explicit expansion.
    // simplifier.add_rule(Box::new(LogExpansionRule));

    // LogAbsSimplifyRule: ln(|x|) → ln(x) when x > 0
    // Must be BEFORE LogContractionRule to catch `ln(|x|) - ln(x)` before it becomes `ln(|x|/x)`
    simplifier.add_rule(Box::new(LogAbsSimplifyRule));

    // LogChainProductRule: log(b,a)*log(a,c) → log(b,c) (telescoping)
    // Must be BEFORE LogContractionRule
    simplifier.add_rule(Box::new(LogChainProductRule));

    // LogContractionRule DOES reduce node count (ln(a)+ln(b) → ln(ab)) - valid simplification
    simplifier.add_rule(Box::new(LogContractionRule));

    simplifier.add_rule(Box::new(ExponentialLogRule));
    simplifier.add_rule(Box::new(SplitLogExponentsRule));
    simplifier.add_rule(Box::new(LogInversePowerRule));
    simplifier.add_rule(Box::new(LogExpInverseRule));

    // AutoExpandLogRule: auto-expand log products/quotients when log_expand_policy=Auto
    simplifier.add_rule(Box::new(AutoExpandLogRule));
}
