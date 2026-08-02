//! Value-domain axis for INTEGRATION routes — the single place where the
//! validity of logarithmic antiderivatives is declared (plan de desacoplo
//! D4, 2026-08-02).
//!
//! ## The semantic contract
//!
//! Under `ValueDomain::RealOnly`, integration is real-variable calculus: the
//! antiderivative of `u'/u` is `ln(|u|)` (valid on every interval avoiding
//! the poles of `u`).
//!
//! Under `ComplexEnabled`, a bare symbol may hold a complex value (see
//! `complex_support`): `ln(|u|)` is NOT a complex antiderivative (`|·|` is
//! not analytic), while `ln(u)` with the principal branch is the standard
//! formal antiderivative — exactly what the rest of the engine already does
//! under this axis (`solve(x²=−1) → {i, −i}`, `sqrt(x²)` stays un-collapsed)
//! and what sympy emits (`integrate(1/x) = log(x)`). The partial-fraction
//! backend already made this exact distinction locally for complex roots
//! (`is_real_root` gate); this module is that decision promoted to a single
//! declared chokepoint.
//!
//! ## The vehicle
//!
//! The integration entry points of the ENGINE arm the ambient axis via
//! [`arm`] (RAII guard, save/restore). The default is `RealOnly`-behaviour,
//! so every un-armed path — internal probes, tests calling `cas_math`
//! directly, sibling commands — stays byte-identical to the pre-D4 engine.
//! Emitters inside integration routes call [`ln_antiderivative_arg`] instead
//! of hand-building `Abs`; nothing else about the routes changes.

use std::cell::Cell;

use cas_ast::{BuiltinFn, Context, ExprId};

thread_local! {
    static COMPLEX_LOG_PRIMITIVES: Cell<bool> = const { Cell::new(false) };
}

/// RAII guard restoring the previous axis value on drop.
pub struct IntegrationValueDomainGuard {
    previous: bool,
}

impl Drop for IntegrationValueDomainGuard {
    fn drop(&mut self) {
        COMPLEX_LOG_PRIMITIVES.with(|cell| cell.set(self.previous));
    }
}

/// Arm the integration value-domain axis for the current thread.
///
/// `complex_enabled = false` is the neutral default: it makes the guard a
/// no-op, so callers may arm unconditionally with the ambient value.
#[must_use]
pub fn arm(complex_enabled: bool) -> IntegrationValueDomainGuard {
    let previous = COMPLEX_LOG_PRIMITIVES.with(|cell| cell.replace(complex_enabled));
    IntegrationValueDomainGuard { previous }
}

/// True when the ambient integration axis admits complex symbol values.
pub fn complex_log_primitives_enabled() -> bool {
    COMPLEX_LOG_PRIMITIVES.with(Cell::get)
}

/// THE chokepoint: the argument a logarithmic antiderivative wraps.
///
/// Real axis → `|u|` (interval-wise real antiderivative). Complex axis →
/// `u` untouched (principal-branch antiderivative). Every integration route
/// that emits `ln(<this>)` must build the argument here; absolute values
/// that are part of the antiderivative VALUE itself (`∫|x| = x·|x|/2`) are
/// not logarithm arguments and must NOT use this.
pub fn ln_antiderivative_arg(ctx: &mut Context, u: ExprId) -> ExprId {
    if complex_log_primitives_enabled() {
        u
    } else {
        ctx.call_builtin(BuiltinFn::Abs, vec![u])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_real_only_abs() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let arg = ln_antiderivative_arg(&mut ctx, x);
        assert!(matches!(
            ctx.get(arg),
            cas_ast::Expr::Function(f, _) if ctx.builtin_of(*f) == Some(BuiltinFn::Abs)
        ));
    }

    #[test]
    fn armed_complex_emits_plain_and_restores() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        {
            let _guard = arm(true);
            assert!(complex_log_primitives_enabled());
            let arg = ln_antiderivative_arg(&mut ctx, x);
            assert_eq!(arg, x);
            {
                let _inner = arm(false);
                assert!(!complex_log_primitives_enabled());
            }
            assert!(complex_log_primitives_enabled());
        }
        assert!(!complex_log_primitives_enabled());
    }
}
