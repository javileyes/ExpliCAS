// Compatibility wrapper: migrated to `cas_solver`.
extern crate cas_engine as cas_solver;

#[path = "../../cas_solver/tests/repro_pythagorean_overflow.rs"]
mod repro_pythagorean_overflow;
