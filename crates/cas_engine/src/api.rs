//! V2.1 Issue #4: Stable Public API
//!
//! This module re-exports the stable public types for external integrators.
//! These types form the contractual API and should maintain backward compatibility.
//!
//! # Stability Guarantee
//!
//! Types in this module are considered stable. Changes to their structure
//! will be documented in CHANGELOG and follow semantic versioning.
//!
//! # Usage
//!
//! ```rust,ignore
//! use cas_engine::api::{SolveResult, SolutionSet, Case, ConditionPredicate, ConditionSet};
//! use cas_engine::api::SolveBudget;
//! ```
//!
//! # Example: Checking Solution Type
//!
//! ```rust,ignore
//! use cas_engine::api::SolutionSet;
//!
//! fn process_solution(sol: &SolutionSet) {
//!     match sol {
//!         SolutionSet::Discrete(solutions) => {
//!             println!("Found {} discrete solutions", solutions.len());
//!         }
//!         SolutionSet::AllReals => {
//!             println!("Solution is all real numbers");
//!         }
//!         SolutionSet::Empty => {
//!             println!("No solution exists");
//!         }
//!         SolutionSet::Conditional(cases) => {
//!             println!("Conditional solution with {} cases", cases.len());
//!             for case in cases {
//!                 // Access case.when (ConditionSet) and case.then (SolveResult)
//!             }
//!         }
//!         _ => {}
//!     }
//! }
//! ```

// =============================================================================
// Stable Types from cas_ast::domain
// =============================================================================

/// Complete result of a solve operation.
/// Contains solutions and optional residual for partially solved equations.
pub use cas_ast::SolveResult;

/// Set of solutions: Discrete, Continuous, Conditional, etc.
pub use cas_ast::SolutionSet;

/// A single case in a conditional solution (guard + result).
pub use cas_ast::Case;

/// A condition predicate (NonZero, Positive, EqZero, etc.).
pub use cas_ast::ConditionPredicate;

/// A conjunction of condition predicates.
pub use cas_ast::ConditionSet;

/// Interval representation for continuous solutions.
pub use cas_ast::Interval;

/// Bound type for intervals (Open or Closed).
pub use cas_ast::BoundType;

// =============================================================================
// Stable Types from cas_engine::solver
// =============================================================================

/// Budget for conditional branching in solver.
/// Controls how many branches can be created (anti-explosion).
pub use crate::solver::SolveBudget;

/// Options for solver operations including domain and budget.
pub use crate::solver::SolverOptions;

// =============================================================================
// Stable Entrypoints
// =============================================================================

/// Solve an equation with default options.
///
/// # Example
/// ```rust,ignore
/// use cas_engine::api::solve;
/// use cas_engine::Engine;
///
/// let mut engine = Engine::new();
/// // ... set up equation ...
/// // let (solution_set, steps) = solve(&equation, "x", &mut engine.simplifier)?;
/// ```
pub use crate::solver::solve;

/// Solve an equation with explicit options, returning display-ready steps.
///
/// V2.9.8: Type-safe API - returns `DisplaySolveSteps` which guarantees
/// cleanup has been applied. Replaces the old `solve_with_options` for
/// display-facing code.
pub use crate::solver::solve_with_display_steps;

/// Display-ready solve steps (post-cleanup).
/// Wrapper type that enforces step processing has been applied.
pub use crate::solver::DisplaySolveSteps;

// =============================================================================
// Display Traits (Human-readable output)
// =============================================================================

/// Display wrapper for expressions with context.
/// Used for human-readable output of expressions.
pub use cas_formatter::DisplayExpr;

/// LaTeX wrapper for expressions with context.
/// Used for mathematical typesetting output.
pub use cas_formatter::LaTeXExpr;
