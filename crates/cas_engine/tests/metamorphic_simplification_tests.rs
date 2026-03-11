// Compatibility wrapper: migrated to `cas_solver`.
#[allow(unused_imports)]
pub use cas_ast::ordering::compare_expr;
pub use cas_engine::*;
#[allow(unused_imports)]
pub use cas_math::expr_nary::{add_terms_signed, Sign};
#[allow(unused_imports)]
pub use cas_math::factor::factor;
extern crate cas_engine as cas_solver;

#[path = "../../cas_solver/tests/metamorphic_simplification_tests.rs"]
mod metamorphic_simplification_tests;
