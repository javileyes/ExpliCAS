// Backward-compatible wrapper:
// keeps `-p cas_engine --test metamorphic_equation_tests ...` working
// while canonical ownership of this metatest lives in `cas_solver/tests`.

pub use cas_engine::*;
extern crate self as cas_solver;

#[path = "../../cas_solver/tests/metamorphic_equation_tests.rs"]
mod solver_metamorphic_equation_tests;
