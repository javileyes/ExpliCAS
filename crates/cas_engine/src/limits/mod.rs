//! Limit framework for symbolic limit computation.
//!
//! # V1 Scope
//! - Limits to ±∞ for polynomials, rationals, and simple powers
//! - Conservative policy: never invent results
//! - Residual `limit(expr, var, approach)` for unresolved limits
//!
//! # Example
//! ```ignore
//! use cas_engine::limits::{limit, Approach, LimitOptions};
//!
//! let result = limit(ctx, expr, var, Approach::PosInfinity, &opts, &mut budget)?;
//! ```

mod engine;
mod helpers;
mod presimplify;
mod rules;
mod types;

pub use engine::limit;
pub use types::{Approach, LimitOptions, LimitResult, PreSimplifyMode};
