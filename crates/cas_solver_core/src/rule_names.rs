//! Canonical rule-name vocabulary shared across crates.
//!
//! Several rule names are a cross-crate contract: `cas_engine` (and the
//! calculus actions in it) *emit* them as `Step::rule_name`, while
//! `cas_solver_core::step_rules`, `cas_didactic`, and `cas_solver` *match* on
//! them by string to classify or preserve steps. Historically each producer
//! and matcher spelled the literal itself, so a rename risked a silent RED far
//! from the edit (the string flows in both directions across five crates).
//!
//! These `pub const`s are the single source of truth. Producers and matchers in
//! library code reference them; a rename is now one edit. Test assertions and
//! the Python smoke harness deliberately keep the bare string literals — they
//! are independent wire-contract anchors that must fail if a const value drifts.
//!
//! The values are byte-for-byte identical to the previous literals, so the
//! rendered wire output (and the scorecard huella) is unchanged.

/// `x^a · x^b → x^(a+b)` exponent aggregation.
pub const RULE_SUM_EXPONENTS: &str = "Sum Exponents";

/// Evaluate a numeric power to its closed value.
pub const RULE_EVALUATE_NUMERIC_POWER: &str = "Evaluate Numeric Power";

/// Expand a product/quotient so a fraction cancels.
pub const RULE_EXPAND_TO_CANCEL_FRACTION: &str = "Expand to Cancel Fraction";

/// Expand `log|a·b|` / `log|a/b|` into a sum/difference of logs.
pub const RULE_EXPAND_LOG_ABS_MUL_DIV: &str = "Expand Log Abs Mul/Div";

/// Cancel an exact additive pair `a + (-a) → 0`.
pub const RULE_CANCEL_EXACT_ADDITIVE_PAIRS: &str = "Cancel Exact Additive Pairs";

/// `|a| → a` under a proven-positive assumption.
pub const RULE_ABS_UNDER_POSITIVITY: &str = "Abs Under Positivity";

/// `|a| → a` under a proven-non-negative assumption.
pub const RULE_ABS_UNDER_NON_NEGATIVITY: &str = "Abs Under Non-Negativity";

/// Preserve an unresolved derivative as a residual step.
pub const RULE_CONSERVAR_DERIVADA_RESIDUAL: &str = "Conservar derivada residual";

/// Preserve an unresolved integral as a residual step.
pub const RULE_CONSERVAR_INTEGRAL_RESIDUAL: &str = "Conservar integral residual";

/// Preserve an unresolved limit as a residual step.
pub const RULE_CONSERVAR_LIMITE_RESIDUAL: &str = "Conservar límite residual";
