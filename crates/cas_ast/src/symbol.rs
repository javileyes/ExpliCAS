//! Symbol interning for variable names.
//!
//! Provides fast comparison and reduced memory usage for repeated variable names.
//! All variable names are stored once and referenced by SymbolId.

use crate::builtin::{BuiltinFn, ALL_BUILTINS};
use rustc_hash::FxHashMap;
use smol_str::SmolStr;
/// Unique identifier for an interned symbol (variable name).
///
/// Using usize for direct Vec indexing without casts.
pub type SymbolId = usize;

/// Symbol table for interning variable names.
///
/// # Design
/// - `strings`: canonical storage for non-builtin symbols.
/// - `lookup`: reverse lookup for non-builtin symbols.
///
/// # Thread Safety
/// Not thread-safe. Intended for single-threaded use within Context.
#[derive(Debug, Clone, Default)]
pub struct SymbolTable {
    /// Canonical storage for non-builtin symbols.
    strings: Vec<SmolStr>,
    /// Reverse lookup for non-builtin symbols.
    lookup: FxHashMap<SmolStr, SymbolId>,
}

impl SymbolTable {
    /// Create a new empty symbol table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a symbol table with enough space for the expected symbol count.
    pub fn with_capacity(capacity: usize) -> Self {
        let mut lookup = FxHashMap::default();
        lookup.reserve(capacity);
        Self {
            strings: Vec::with_capacity(capacity),
            lookup,
        }
    }

    /// Intern a string, returning its SymbolId.
    ///
    /// If the string is already interned, returns the existing id.
    /// Otherwise, stores it and returns a new id.
    pub fn intern(&mut self, s: &str) -> SymbolId {
        if let Some(builtin) = BuiltinFn::from_name(s) {
            return builtin as SymbolId;
        }

        // Fast path: already interned
        if let Some(&id) = self.lookup.get(s) {
            return id;
        }

        // Slow path: intern new string
        let id = BuiltinFn::COUNT + self.strings.len();
        let owned = SmolStr::new(s);
        self.strings.push(owned.clone());
        self.lookup.insert(owned, id);
        id
    }

    /// Resolve a SymbolId back to its string.
    ///
    /// # Panics
    /// Panics if id is invalid (out of bounds).
    #[inline]
    pub fn resolve(&self, id: SymbolId) -> &str {
        if id < BuiltinFn::COUNT {
            return ALL_BUILTINS[id].name();
        }

        self.strings[id - BuiltinFn::COUNT].as_str()
    }

    /// Get id for a string if it exists, without interning.
    ///
    /// Useful for read-only comparisons.
    #[inline]
    pub fn get_id(&self, s: &str) -> Option<SymbolId> {
        BuiltinFn::from_name(s)
            .map(|builtin| builtin as SymbolId)
            .or_else(|| self.lookup.get(s).copied())
    }

    /// Number of interned symbols.
    #[inline]
    pub fn len(&self) -> usize {
        BuiltinFn::COUNT + self.strings.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intern_roundtrip() {
        let mut table = SymbolTable::new();
        let id = table.intern("x");
        assert_eq!(table.resolve(id), "x");
    }

    #[test]
    fn test_intern_deduplication() {
        let mut table = SymbolTable::new();
        let id1 = table.intern("x");
        let id2 = table.intern("x");
        assert_eq!(id1, id2);
        assert_eq!(table.len(), BuiltinFn::COUNT + 1);
    }

    #[test]
    fn test_different_symbols() {
        let mut table = SymbolTable::new();
        let x = table.intern("x");
        let y = table.intern("y");
        assert_ne!(x, y);
        assert_eq!(table.len(), BuiltinFn::COUNT + 2);
    }

    #[test]
    fn test_get_id_existing() {
        let mut table = SymbolTable::new();
        let id = table.intern("x");
        assert_eq!(table.get_id("x"), Some(id));
    }

    #[test]
    fn test_get_id_missing() {
        let table = SymbolTable::new();
        assert_eq!(table.get_id("x"), None);
    }

    #[test]
    fn test_builtin_ids_are_logical_prefix() {
        let mut table = SymbolTable::new();
        assert_eq!(table.get_id("sin"), Some(BuiltinFn::Sin as SymbolId));
        assert_eq!(table.get_id("cos"), Some(BuiltinFn::Cos as SymbolId));

        let x = table.intern("x");
        assert_eq!(x, BuiltinFn::COUNT);
    }

    #[test]
    fn test_len_counts_builtin_prefix() {
        let mut table = SymbolTable::new();
        assert_eq!(table.len(), BuiltinFn::COUNT);
        table.intern("x");
        assert_eq!(table.len(), BuiltinFn::COUNT + 1);
    }

    #[test]
    fn test_unicode_symbols() {
        let mut table = SymbolTable::new();
        let alpha = table.intern("α");
        let beta = table.intern("β");
        assert_ne!(alpha, beta);
        assert_eq!(table.resolve(alpha), "α");
        assert_eq!(table.resolve(beta), "β");
    }
}
