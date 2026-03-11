//! Compatibility wrapper.
//!
//! Canonical repro_hyperbolic lives in `cas_solver`.

#[allow(unused_imports)]
pub use cas_ast::ordering::compare_expr;
pub use cas_engine::*;
#[allow(unused_imports)]
pub use cas_math::expr_nary::{add_terms_signed, Sign};
#[allow(unused_imports)]
pub use cas_math::factor::factor;
extern crate cas_engine as cas_solver;

#[path = "../../cas_solver/tests/repro_hyperbolic.rs"]
mod solver_repro_hyperbolic;
