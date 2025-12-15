//! Session storage for expressions and equations with auto-incrementing IDs.
//!
//! Provides a "notebook-style" storage where each input gets a unique `#id`
//! that can be referenced in subsequent commands.

use cas_ast::ExprId;

/// Unique identifier for a session entry
pub type EntryId = u64;

/// Type of entry stored in the session
#[derive(Debug, Clone)]
pub enum EntryKind {
    /// A single expression
    Expr(ExprId),
    /// An equation (lhs = rhs)
    Eq { lhs: ExprId, rhs: ExprId },
}

/// A stored entry in the session
#[derive(Debug, Clone)]
pub struct Entry {
    /// Unique ID (auto-incrementing, never reused)
    pub id: EntryId,
    /// The stored expression or equation
    pub kind: EntryKind,
    /// Original raw text input (for display)
    pub raw_text: String,
}

impl Entry {
    /// Check if this entry is an expression
    pub fn is_expr(&self) -> bool {
        matches!(self.kind, EntryKind::Expr(_))
    }

    /// Check if this entry is an equation
    pub fn is_eq(&self) -> bool {
        matches!(self.kind, EntryKind::Eq { .. })
    }

    /// Get the type as a string for display
    pub fn type_str(&self) -> &'static str {
        match self.kind {
            EntryKind::Expr(_) => "Expr",
            EntryKind::Eq { .. } => "Eq",
        }
    }
}

/// Storage for session entries with auto-incrementing IDs
#[derive(Debug, Clone)]
pub struct SessionStore {
    next_id: EntryId,
    entries: Vec<Entry>,
}

impl Default for SessionStore {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionStore {
    /// Create a new empty session store
    pub fn new() -> Self {
        Self {
            next_id: 1, // Start at 1 for human-friendly IDs
            entries: Vec::new(),
        }
    }

    /// Store a new entry and return its ID
    pub fn push(&mut self, kind: EntryKind, raw_text: String) -> EntryId {
        let id = self.next_id;
        self.next_id += 1;
        self.entries.push(Entry { id, kind, raw_text });
        id
    }

    /// Get an entry by ID
    pub fn get(&self, id: EntryId) -> Option<&Entry> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// Remove entries by IDs (IDs are never reused)
    pub fn remove(&mut self, ids: &[EntryId]) {
        self.entries.retain(|e| !ids.contains(&e.id));
    }

    /// Clear all entries (IDs are still never reused)
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Get all entries
    pub fn list(&self) -> &[Entry] {
        &self.entries
    }

    /// Check if an entry exists
    pub fn contains(&self, id: EntryId) -> bool {
        self.entries.iter().any(|e| e.id == id)
    }

    /// Get the number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the store is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the next ID that will be assigned (for preview)
    pub fn next_id(&self) -> EntryId {
        self.next_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_and_get() {
        let mut store = SessionStore::new();

        // Create a dummy ExprId (in real usage this comes from Context)
        let expr_id = ExprId::from_raw(0);

        let id1 = store.push(EntryKind::Expr(expr_id), "x + 1".to_string());
        let id2 = store.push(EntryKind::Expr(expr_id), "x^2".to_string());

        assert_eq!(id1, 1);
        assert_eq!(id2, 2);

        let entry1 = store.get(1).unwrap();
        assert_eq!(entry1.raw_text, "x + 1");
        assert!(entry1.is_expr());

        let entry2 = store.get(2).unwrap();
        assert_eq!(entry2.raw_text, "x^2");
    }

    #[test]
    fn test_ids_not_reused_after_delete() {
        let mut store = SessionStore::new();
        let expr_id = ExprId::from_raw(0);

        let id1 = store.push(EntryKind::Expr(expr_id), "a".to_string());
        let id2 = store.push(EntryKind::Expr(expr_id), "b".to_string());

        // Delete id1
        store.remove(&[id1]);

        // Next ID should be 3, not 1
        let id3 = store.push(EntryKind::Expr(expr_id), "c".to_string());
        assert_eq!(id3, 3);
        assert!(!store.contains(id1));
        assert!(store.contains(id2));
        assert!(store.contains(id3));
    }

    #[test]
    fn test_remove_multiple() {
        let mut store = SessionStore::new();
        let expr_id = ExprId::from_raw(0);

        store.push(EntryKind::Expr(expr_id), "a".to_string());
        store.push(EntryKind::Expr(expr_id), "b".to_string());
        store.push(EntryKind::Expr(expr_id), "c".to_string());

        store.remove(&[1, 3]);
        assert_eq!(store.len(), 1);
        assert!(store.contains(2));
    }

    #[test]
    fn test_clear() {
        let mut store = SessionStore::new();
        let expr_id = ExprId::from_raw(0);

        store.push(EntryKind::Expr(expr_id), "a".to_string());
        store.push(EntryKind::Expr(expr_id), "b".to_string());

        store.clear();
        assert!(store.is_empty());

        // Next ID should still be 3
        let id = store.push(EntryKind::Expr(expr_id), "c".to_string());
        assert_eq!(id, 3);
    }

    #[test]
    fn test_equation_entry() {
        let mut store = SessionStore::new();
        let lhs = ExprId::from_raw(0);
        let rhs = ExprId::from_raw(1);

        let id = store.push(EntryKind::Eq { lhs, rhs }, "x + 1 = 5".to_string());

        let entry = store.get(id).unwrap();
        assert!(entry.is_eq());
        assert_eq!(entry.type_str(), "Eq");
    }
}
