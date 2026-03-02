// Compatibility wrapper: migrated to `cas_solver`.
pub use cas_engine::*;
extern crate self as cas_solver;

#[path = "../../cas_solver/tests/dyadic_cos_product_tests.rs"]
mod dyadic_cos_product_tests;
