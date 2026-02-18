//! Compatibility re-export for multinomial expansion helpers.
//!
//! Canonical implementation lives in `cas_math`.
//! Keep hotspot budget vocabulary visible in this adapter:
//! `MultinomialExpandBudget`, `max_output_terms`.

pub use cas_math::multinomial_expand::*;
