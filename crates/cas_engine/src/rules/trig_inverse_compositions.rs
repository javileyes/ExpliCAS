//! Legacy shim for inverse-trig bridge identities (deprecated).
//!
//! All composition rules have been migrated to `inv_trig_n_angle.rs`:
//! - **Arctan** n=1..10: `NAngleAtanRule` (Weierstrass recurrence)
//! - **Arccos** n=1..10: `NAngleAcosRule` (Chebyshev T_n/U_n recurrence)
//! - **Arcsin** n=1..10: `NAngleAsinRule` (sin/cos recurrence)
//!
//! This module is retained as a no-op shim to preserve `mod.rs` and `use` stability.
//! It can be deleted in a future cleanup ticket.

/// No-op: all rules now live in `inv_trig_n_angle`.
pub fn register(_simplifier: &mut crate::Simplifier) {
    // Intentionally empty — see inv_trig_n_angle::register()
}
