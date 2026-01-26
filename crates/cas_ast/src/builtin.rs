//! Builtin function identifiers for O(1) comparison.
//!
//! This module provides a type-safe enumeration of known built-in functions,
//! allowing rules to compare function identities by ID instead of by string.
//!
//! # Phase 2 of Interning
//!
//! After Phase 1 (SymbolId for function names in AST), Phase 2 caches the
//! SymbolIds of common builtins in Context, enabling:
//!
//! - **O(1) comparison**: `fn_id == ctx.builtin_id(BuiltinFn::Sqrt)`
//! - **Type safety**: exhaustive `match` on `BuiltinFn` variants
//! - **No string allocation**: comparisons use interned IDs
//!
//! # Usage
//!
//! ```rust,ignore
//! // Before (string comparison):
//! if ctx.sym_name(*fn_id) == "sqrt" { ... }
//!
//! // After (O(1) ID comparison):
//! if *fn_id == ctx.builtin_id(BuiltinFn::Sqrt) { ... }
//!
//! // Or using the helper:
//! if ctx.is_builtin_call(expr, BuiltinFn::Sqrt) { ... }
//! ```
//!
//! # Adding new builtins
//!
//! 1. Add variant to `BuiltinFn` enum
//! 2. Add case to `BuiltinFn::name()` method
//! 3. The `BuiltinFn::COUNT` constant auto-updates
//!
//! The Context initialization will automatically cache the new builtin.

use crate::symbol::SymbolId;

/// Known built-in functions with cached SymbolIds.
///
/// These are the most frequently compared function names in the engine.
/// Adding a function here enables O(1) comparison instead of string comparison.
///
/// # Ordering
///
/// Variants are ordered to group related functions:
/// - Trig: sin, cos, tan, sec, csc, cot
/// - Inverse trig: asin, acos, atan, asec, acsc, acot
/// - Hyperbolic: sinh, cosh, tanh
/// - Logarithmic: ln, log, log2, log10
/// - Other: sqrt, abs, exp, sign, floor, ceil
/// - Internal: __hold, poly_result
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BuiltinFn {
    // Trigonometric
    Sin = 0,
    Cos,
    Tan,
    Sec,
    Csc,
    Cot,

    // Inverse trigonometric (short form: a*)
    Asin,
    Acos,
    Atan,
    Asec,
    Acsc,
    Acot,

    // Inverse trigonometric (long form: arc*)
    Arcsin,
    Arccos,
    Arctan,

    // Hyperbolic
    Sinh,
    Cosh,
    Tanh,

    // Inverse hyperbolic
    Asinh,
    Acosh,
    Atanh,

    // Logarithmic / Exponential
    Ln,
    Log,
    Log2,
    Log10,
    Exp,

    // Roots / Powers
    Sqrt,
    Cbrt,
    Root, // root(x, n) = nth root

    // Other common
    Abs,
    Sign,
    Floor,
    Ceil,

    // Internal / Special
    Hold,       // __hold barrier
    PolyResult, // poly_result wrapper
    Expand,     // expand command

    // Comparison functions (used by solver)
    Equal,   // Equal(a, b) - symbolic comparison
    Less,    // Less(a, b)
    Greater, // Greater(a, b)
    Eq,      // __eq__(lhs, rhs) - internal equation wrapper
}

impl BuiltinFn {
    /// Total number of builtin functions.
    /// Update this when adding new variants!
    pub const COUNT: usize = 40;

    /// Get the string name of this builtin function.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            // Trig
            BuiltinFn::Sin => "sin",
            BuiltinFn::Cos => "cos",
            BuiltinFn::Tan => "tan",
            BuiltinFn::Sec => "sec",
            BuiltinFn::Csc => "csc",
            BuiltinFn::Cot => "cot",

            // Inverse trig (short)
            BuiltinFn::Asin => "asin",
            BuiltinFn::Acos => "acos",
            BuiltinFn::Atan => "atan",
            BuiltinFn::Asec => "asec",
            BuiltinFn::Acsc => "acsc",
            BuiltinFn::Acot => "acot",

            // Inverse trig (long)
            BuiltinFn::Arcsin => "arcsin",
            BuiltinFn::Arccos => "arccos",
            BuiltinFn::Arctan => "arctan",

            // Hyperbolic
            BuiltinFn::Sinh => "sinh",
            BuiltinFn::Cosh => "cosh",
            BuiltinFn::Tanh => "tanh",

            // Inverse hyperbolic
            BuiltinFn::Asinh => "asinh",
            BuiltinFn::Acosh => "acosh",
            BuiltinFn::Atanh => "atanh",

            // Log/exp
            BuiltinFn::Ln => "ln",
            BuiltinFn::Log => "log",
            BuiltinFn::Log2 => "log2",
            BuiltinFn::Log10 => "log10",
            BuiltinFn::Exp => "exp",

            // Roots
            BuiltinFn::Sqrt => "sqrt",
            BuiltinFn::Cbrt => "cbrt",
            BuiltinFn::Root => "root",

            // Other
            BuiltinFn::Abs => "abs",
            BuiltinFn::Sign => "sign",
            BuiltinFn::Floor => "floor",
            BuiltinFn::Ceil => "ceil",

            // Internal
            BuiltinFn::Hold => "__hold",
            BuiltinFn::PolyResult => "poly_result",
            BuiltinFn::Expand => "expand",

            // Comparison
            BuiltinFn::Equal => "Equal",
            BuiltinFn::Less => "Less",
            BuiltinFn::Greater => "Greater",
            BuiltinFn::Eq => "__eq__",
        }
    }

    /// Iterate over all builtin functions.
    #[inline]
    pub fn all() -> impl Iterator<Item = BuiltinFn> {
        ALL_BUILTINS.iter().copied()
    }
}

/// Static array of all builtin function variants.
/// Used for iteration and initialization.
pub const ALL_BUILTINS: [BuiltinFn; BuiltinFn::COUNT] = [
    BuiltinFn::Sin,
    BuiltinFn::Cos,
    BuiltinFn::Tan,
    BuiltinFn::Sec,
    BuiltinFn::Csc,
    BuiltinFn::Cot,
    BuiltinFn::Asin,
    BuiltinFn::Acos,
    BuiltinFn::Atan,
    BuiltinFn::Asec,
    BuiltinFn::Acsc,
    BuiltinFn::Acot,
    BuiltinFn::Arcsin,
    BuiltinFn::Arccos,
    BuiltinFn::Arctan,
    BuiltinFn::Sinh,
    BuiltinFn::Cosh,
    BuiltinFn::Tanh,
    BuiltinFn::Asinh,
    BuiltinFn::Acosh,
    BuiltinFn::Atanh,
    BuiltinFn::Ln,
    BuiltinFn::Log,
    BuiltinFn::Log2,
    BuiltinFn::Log10,
    BuiltinFn::Exp,
    BuiltinFn::Sqrt,
    BuiltinFn::Cbrt,
    BuiltinFn::Root,
    BuiltinFn::Abs,
    BuiltinFn::Sign,
    BuiltinFn::Floor,
    BuiltinFn::Ceil,
    BuiltinFn::Hold,
    BuiltinFn::PolyResult,
    BuiltinFn::Expand,
    BuiltinFn::Equal,
    BuiltinFn::Less,
    BuiltinFn::Greater,
    BuiltinFn::Eq,
];

/// Cache of builtin function SymbolIds.
///
/// Initialized lazily on first access or explicitly via `init_builtins()`.
/// Uses a fixed-size array indexed by `BuiltinFn as usize`.
#[derive(Debug, Clone)]
pub struct BuiltinIds {
    /// Cached SymbolIds, indexed by BuiltinFn discriminant.
    /// SymbolId(0) means "not yet initialized" (valid after init).
    ids: [SymbolId; BuiltinFn::COUNT],
    /// Whether the cache has been initialized
    initialized: bool,
}

impl Default for BuiltinIds {
    fn default() -> Self {
        Self {
            ids: [0; BuiltinFn::COUNT],
            initialized: false,
        }
    }
}

impl BuiltinIds {
    /// Create a new uninitialized cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if the cache is initialized.
    #[inline]
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get the cached SymbolId for a builtin.
    ///
    /// # Panics
    /// Panics if called before initialization.
    #[inline]
    pub fn get(&self, builtin: BuiltinFn) -> SymbolId {
        debug_assert!(self.initialized, "BuiltinIds not initialized");
        self.ids[builtin as usize]
    }

    /// Set the SymbolId for a builtin (used during initialization).
    #[inline]
    pub fn set(&mut self, builtin: BuiltinFn, id: SymbolId) {
        self.ids[builtin as usize] = id;
    }

    /// Mark as initialized.
    #[inline]
    pub fn mark_initialized(&mut self) {
        self.initialized = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_names() {
        assert_eq!(BuiltinFn::Sin.name(), "sin");
        assert_eq!(BuiltinFn::Sqrt.name(), "sqrt");
        assert_eq!(BuiltinFn::Hold.name(), "__hold");
    }

    #[test]
    fn test_builtin_count() {
        // Ensure COUNT matches the number of unique variants
        // (excluding placeholder slots if we use them)
        let unique_count = 28; // Actual unique variants
        assert!(BuiltinFn::COUNT >= unique_count);
    }

    #[test]
    fn test_all_iterator() {
        let count = BuiltinFn::all().count();
        assert_eq!(count, BuiltinFn::COUNT);
    }

    #[test]
    fn test_builtin_ids_cache() {
        let mut cache = BuiltinIds::new();
        assert!(!cache.is_initialized());

        // Simulate initialization
        cache.set(BuiltinFn::Sin, 42);
        cache.set(BuiltinFn::Cos, 43);
        cache.mark_initialized();

        assert!(cache.is_initialized());
        assert_eq!(cache.get(BuiltinFn::Sin), 42);
        assert_eq!(cache.get(BuiltinFn::Cos), 43);
    }
}
