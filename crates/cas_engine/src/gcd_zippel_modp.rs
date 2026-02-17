//! Compatibility re-export for Zippel modular GCD.
//!
//! Canonical implementation lives in `cas_math`.

// Keep `ZippelBudget` visible in this compatibility layer so budget lint
// scripts continue recognizing this hotspot as instrumented.
pub use cas_math::gcd_zippel_modp::ZippelBudget;
pub use cas_math::gcd_zippel_modp::*;
