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
    Arcsec,
    Arccsc,
    Arccot,

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
    Equal,        // Equal(a, b) - symbolic comparison
    Less,         // Less(a, b)
    Greater,      // Greater(a, b)
    LessEqual,    // LessEqual(a, b)
    GreaterEqual, // GreaterEqual(a, b)
    NotEqual,     // NotEqual(a, b)
    Eq,           // __eq__(lhs, rhs) - internal equation wrapper

    // Complex (Fase 2): evaluated only under ComplexEnabled; in RealOnly they
    // stay symbolic (honest residual), never error.
    Re,        // Re(z) - real part
    Im,        // Im(z) - imaginary part
    Conjugate, // conjugate(z) - complex conjugate
    Arg,       // arg(z) - principal argument in (-pi, pi]
}

impl BuiltinFn {
    /// Total number of builtin functions.
    /// Update this when adding new variants!
    pub const COUNT: usize = 50;

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
            BuiltinFn::Arcsec => "arcsec",
            BuiltinFn::Arccsc => "arccsc",
            BuiltinFn::Arccot => "arccot",

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
            BuiltinFn::LessEqual => "LessEqual",
            BuiltinFn::GreaterEqual => "GreaterEqual",
            BuiltinFn::NotEqual => "NotEqual",
            BuiltinFn::Eq => "__eq__",

            // Complex
            BuiltinFn::Re => "Re",
            BuiltinFn::Im => "Im",
            BuiltinFn::Conjugate => "conjugate",
            BuiltinFn::Arg => "arg",
        }
    }

    /// Resolve a builtin from its canonical textual name.
    #[inline]
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "sin" => Some(BuiltinFn::Sin),
            "cos" => Some(BuiltinFn::Cos),
            "tan" => Some(BuiltinFn::Tan),
            "sec" => Some(BuiltinFn::Sec),
            "csc" => Some(BuiltinFn::Csc),
            "cot" => Some(BuiltinFn::Cot),
            "asin" => Some(BuiltinFn::Asin),
            "acos" => Some(BuiltinFn::Acos),
            "atan" => Some(BuiltinFn::Atan),
            "asec" => Some(BuiltinFn::Asec),
            "acsc" => Some(BuiltinFn::Acsc),
            "acot" => Some(BuiltinFn::Acot),
            "arcsin" => Some(BuiltinFn::Arcsin),
            "arccos" => Some(BuiltinFn::Arccos),
            "arctan" => Some(BuiltinFn::Arctan),
            "arcsec" => Some(BuiltinFn::Arcsec),
            "arccsc" => Some(BuiltinFn::Arccsc),
            "arccot" => Some(BuiltinFn::Arccot),
            // Spanish/European trig spellings — the standard notation in Spanish-language maths
            // (`sen` = seno, `tg` = tangente, `cotg`/`ctg` = cotangente, `cosec` = cosecante),
            // aliased to the existing variants. This engine narrates in Spanish, so a student who
            // writes `sen(x)` or `tg(x)` should get the function, not "función no definida".
            "sen" => Some(BuiltinFn::Sin),
            "tg" => Some(BuiltinFn::Tan),
            "cotg" => Some(BuiltinFn::Cot),
            "ctg" => Some(BuiltinFn::Cot),
            "cosec" => Some(BuiltinFn::Csc),
            "arcsen" => Some(BuiltinFn::Arcsin),
            "arctg" => Some(BuiltinFn::Arctan),
            "sinh" => Some(BuiltinFn::Sinh),
            "cosh" => Some(BuiltinFn::Cosh),
            "tanh" => Some(BuiltinFn::Tanh),
            "asinh" => Some(BuiltinFn::Asinh),
            "acosh" => Some(BuiltinFn::Acosh),
            "atanh" => Some(BuiltinFn::Atanh),
            // `arc*` input spellings of the inverse hyperbolics, mirroring the trig `arcsin`
            // aliases: the engine already implements `asinh`/`acosh`/`atanh` fully (and emits
            // them from integration), so a user who writes `arcsinh(x)` — by analogy with the
            // accepted `arcsin(x)` — should get the same function, not "función no definida".
            "arcsinh" => Some(BuiltinFn::Asinh),
            "arccosh" => Some(BuiltinFn::Acosh),
            "arctanh" => Some(BuiltinFn::Atanh),
            "ln" => Some(BuiltinFn::Ln),
            "log" => Some(BuiltinFn::Log),
            "log2" => Some(BuiltinFn::Log2),
            "log10" => Some(BuiltinFn::Log10),
            "exp" => Some(BuiltinFn::Exp),
            "sqrt" => Some(BuiltinFn::Sqrt),
            "cbrt" => Some(BuiltinFn::Cbrt),
            "root" => Some(BuiltinFn::Root),
            "abs" => Some(BuiltinFn::Abs),
            "sign" => Some(BuiltinFn::Sign),
            "floor" => Some(BuiltinFn::Floor),
            "ceil" => Some(BuiltinFn::Ceil),
            "__hold" => Some(BuiltinFn::Hold),
            "poly_result" => Some(BuiltinFn::PolyResult),
            "expand" => Some(BuiltinFn::Expand),
            "Equal" => Some(BuiltinFn::Equal),
            "Less" => Some(BuiltinFn::Less),
            "Greater" => Some(BuiltinFn::Greater),
            "LessEqual" => Some(BuiltinFn::LessEqual),
            "GreaterEqual" => Some(BuiltinFn::GreaterEqual),
            "NotEqual" => Some(BuiltinFn::NotEqual),
            "__eq__" => Some(BuiltinFn::Eq),
            // Complex (canonical + lowercase/short aliases)
            "Re" => Some(BuiltinFn::Re),
            "re" => Some(BuiltinFn::Re),
            "Im" => Some(BuiltinFn::Im),
            "im" => Some(BuiltinFn::Im),
            "conjugate" => Some(BuiltinFn::Conjugate),
            "conj" => Some(BuiltinFn::Conjugate),
            "arg" => Some(BuiltinFn::Arg),
            "Arg" => Some(BuiltinFn::Arg),
            _ => None,
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
    BuiltinFn::Arcsec,
    BuiltinFn::Arccsc,
    BuiltinFn::Arccot,
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
    BuiltinFn::LessEqual,
    BuiltinFn::GreaterEqual,
    BuiltinFn::NotEqual,
    BuiltinFn::Eq,
    BuiltinFn::Re,
    BuiltinFn::Im,
    BuiltinFn::Conjugate,
    BuiltinFn::Arg,
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

    /// Create an initialized cache for the standard identity layout used by
    /// `Context::new()`, where builtin `SymbolId`s match their enum indices.
    pub fn initialized_identity() -> Self {
        let mut ids = [0; BuiltinFn::COUNT];
        let mut idx = 0;
        while idx < BuiltinFn::COUNT {
            ids[idx] = idx;
            idx += 1;
        }

        Self {
            ids,
            initialized: true,
        }
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

    /// Reverse lookup: get the BuiltinFn for a SymbolId.
    ///
    /// In the standard `Context::new()` layout this is O(1) because builtins are
    /// interned first, in enum order, so their `SymbolId` matches the builtin
    /// index. A tiny fallback scan keeps the helper correct for any unusual
    /// non-identity layout.
    #[inline]
    pub fn lookup(&self, id: SymbolId) -> Option<BuiltinFn> {
        debug_assert!(self.initialized, "BuiltinIds not initialized");
        if id < BuiltinFn::COUNT && self.ids[id] == id {
            return Some(ALL_BUILTINS[id]);
        }

        self.ids
            .iter()
            .position(|&cached_id| cached_id == id)
            .map(|idx| ALL_BUILTINS[idx])
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
    fn spanish_european_trig_spellings_alias_the_standard_variants() {
        // `sen`/`tg`/`cotg`/`ctg`/`cosec` and `arcsen`/`arctg` are the standard Spanish/European
        // spellings, aliased to the existing trig variants.
        assert_eq!(BuiltinFn::from_name("sen"), Some(BuiltinFn::Sin));
        assert_eq!(BuiltinFn::from_name("tg"), Some(BuiltinFn::Tan));
        assert_eq!(BuiltinFn::from_name("cotg"), Some(BuiltinFn::Cot));
        assert_eq!(BuiltinFn::from_name("ctg"), Some(BuiltinFn::Cot));
        assert_eq!(BuiltinFn::from_name("cosec"), Some(BuiltinFn::Csc));
        assert_eq!(BuiltinFn::from_name("arcsen"), Some(BuiltinFn::Arcsin));
        assert_eq!(BuiltinFn::from_name("arctg"), Some(BuiltinFn::Arctan));
    }

    #[test]
    fn arc_prefixed_inverse_hyperbolics_alias_the_a_spelling() {
        // `arcsinh`/`arccosh`/`arctanh` are accepted input spellings of the existing inverse
        // hyperbolics (mirroring the `arcsin` trig alias), resolving to the same variants.
        assert_eq!(BuiltinFn::from_name("arcsinh"), Some(BuiltinFn::Asinh));
        assert_eq!(BuiltinFn::from_name("arccosh"), Some(BuiltinFn::Acosh));
        assert_eq!(BuiltinFn::from_name("arctanh"), Some(BuiltinFn::Atanh));
        // The `a*` spellings still resolve to the same variants.
        assert_eq!(BuiltinFn::from_name("asinh"), Some(BuiltinFn::Asinh));
        assert_eq!(BuiltinFn::from_name("atanh"), Some(BuiltinFn::Atanh));
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

    #[test]
    fn test_builtin_ids_lookup_non_identity_fallback() {
        let mut cache = BuiltinIds::new();
        cache.set(BuiltinFn::Sin, 42);
        cache.set(BuiltinFn::Cos, 43);
        cache.mark_initialized();

        assert_eq!(cache.lookup(42), Some(BuiltinFn::Sin));
        assert_eq!(cache.lookup(43), Some(BuiltinFn::Cos));
        assert_eq!(cache.lookup(999), None);
    }

    #[test]
    fn test_builtin_ids_initialized_identity() {
        let cache = BuiltinIds::initialized_identity();
        assert!(cache.is_initialized());
        assert_eq!(cache.get(BuiltinFn::Sin), BuiltinFn::Sin as SymbolId);
        assert_eq!(
            cache.lookup(BuiltinFn::Sqrt as SymbolId),
            Some(BuiltinFn::Sqrt)
        );
    }
}
