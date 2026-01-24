//! Symbol interning for variable names.
//!
//! Provides fast comparison and reduced memory usage for repeated variable names.
//! All variable names are stored once and referenced by SymbolId.

use std::collections::HashMap;

/// Unique identifier for an interned symbol (variable name).
///
/// Using usize for direct Vec indexing without casts.
pub type SymbolId = usize;

/// Symbol table for interning variable names.
///
/// # Design
/// - `strings`: canonical storage, indexed by SymbolId
/// - `lookup`: reverse map for O(1) intern check
///
/// # Thread Safety
/// Not thread-safe. Intended for single-threaded use within Context.
#[derive(Debug, Clone, Default)]
pub struct SymbolTable {
    /// Canonical string storage (SymbolId = index)
    strings: Vec<String>,
    /// Reverse lookup: string → SymbolId
    lookup: HashMap<String, SymbolId>,
}

impl SymbolTable {
    /// Create a new empty symbol table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern a string, returning its SymbolId.
    ///
    /// If the string is already interned, returns the existing id.
    /// Otherwise, stores it and returns a new id.
    pub fn intern(&mut self, s: &str) -> SymbolId {
        // Fast path: already interned
        if let Some(&id) = self.lookup.get(s) {
            return id;
        }

        // Slow path: intern new string
        let id = self.strings.len();
        let owned = s.to_string();
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
        &self.strings[id]
    }

    /// Get id for a string if it exists, without interning.
    ///
    /// Useful for read-only comparisons.
    #[inline]
    pub fn get_id(&self, s: &str) -> Option<SymbolId> {
        self.lookup.get(s).copied()
    }

    /// Number of interned symbols.
    #[inline]
    pub fn len(&self) -> usize {
        self.strings.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
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
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn test_different_symbols() {
        let mut table = SymbolTable::new();
        let x = table.intern("x");
        let y = table.intern("y");
        assert_ne!(x, y);
        assert_eq!(table.len(), 2);
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
    fn test_unicode_symbols() {
        let mut table = SymbolTable::new();
        let alpha = table.intern("α");
        let beta = table.intern("β");
        assert_ne!(alpha, beta);
        assert_eq!(table.resolve(alpha), "α");
        assert_eq!(table.resolve(beta), "β");
    }
}
