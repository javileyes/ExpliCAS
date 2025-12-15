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

// =============================================================================
// Session Reference Resolution
// =============================================================================

use std::collections::{HashMap, HashSet};

/// Error during session reference resolution
#[derive(Debug, Clone, PartialEq)]
pub enum ResolveError {
    /// Reference to non-existent entry
    NotFound(EntryId),
    /// Circular reference detected (e.g., #3 contains #3, or #3 -> #4 -> #3)
    CircularReference(EntryId),
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolveError::NotFound(id) => write!(f, "Session reference #{} not found", id),
            ResolveError::CircularReference(id) => {
                write!(f, "Circular reference detected involving #{}", id)
            }
        }
    }
}

impl std::error::Error for ResolveError {}

/// Resolve all `Expr::SessionRef` in an expression tree.
///
/// - For expression entries: replaces `#id` with the stored ExprId
/// - For equation entries: replaces `#id` with `(lhs - rhs)` (residue form)
///
/// Uses memoization to avoid re-resolving the same reference.
/// Detects circular references and returns an error.
pub fn resolve_session_refs(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    store: &SessionStore,
) -> Result<ExprId, ResolveError> {
    let mut cache: HashMap<EntryId, ExprId> = HashMap::new();
    let mut visiting: HashSet<EntryId> = HashSet::new();
    resolve_recursive(ctx, expr, store, &mut cache, &mut visiting)
}

fn resolve_recursive(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    store: &SessionStore,
    cache: &mut HashMap<EntryId, ExprId>,
    visiting: &mut HashSet<EntryId>,
) -> Result<ExprId, ResolveError> {
    use cas_ast::Expr;

    let node = ctx.get(expr).clone();

    match node {
        Expr::SessionRef(id) => resolve_session_id(ctx, id, store, cache, visiting),
        Expr::Variable(ref name) => {
            if name.starts_with('#') && name.len() > 1 && name[1..].chars().all(char::is_numeric) {
                if let Ok(id) = name[1..].parse::<u64>() {
                    return resolve_session_id(ctx, id, store, cache, visiting);
                }
            }
            Ok(expr)
        }

        // Binary operators - recurse into children
        Expr::Add(l, r) => {
            let new_l = resolve_recursive(ctx, l, store, cache, visiting)?;
            let new_r = resolve_recursive(ctx, r, store, cache, visiting)?;
            if new_l == l && new_r == r {
                Ok(expr)
            } else {
                Ok(ctx.add(Expr::Add(new_l, new_r)))
            }
        }
        Expr::Sub(l, r) => {
            let new_l = resolve_recursive(ctx, l, store, cache, visiting)?;
            let new_r = resolve_recursive(ctx, r, store, cache, visiting)?;
            if new_l == l && new_r == r {
                Ok(expr)
            } else {
                Ok(ctx.add(Expr::Sub(new_l, new_r)))
            }
        }
        Expr::Mul(l, r) => {
            let new_l = resolve_recursive(ctx, l, store, cache, visiting)?;
            let new_r = resolve_recursive(ctx, r, store, cache, visiting)?;
            if new_l == l && new_r == r {
                Ok(expr)
            } else {
                Ok(ctx.add(Expr::Mul(new_l, new_r)))
            }
        }
        Expr::Div(l, r) => {
            let new_l = resolve_recursive(ctx, l, store, cache, visiting)?;
            let new_r = resolve_recursive(ctx, r, store, cache, visiting)?;
            if new_l == l && new_r == r {
                Ok(expr)
            } else {
                Ok(ctx.add(Expr::Div(new_l, new_r)))
            }
        }
        Expr::Pow(b, e) => {
            let new_b = resolve_recursive(ctx, b, store, cache, visiting)?;
            let new_e = resolve_recursive(ctx, e, store, cache, visiting)?;
            if new_b == b && new_e == e {
                Ok(expr)
            } else {
                Ok(ctx.add(Expr::Pow(new_b, new_e)))
            }
        }

        // Unary
        Expr::Neg(e) => {
            let new_e = resolve_recursive(ctx, e, store, cache, visiting)?;
            if new_e == e {
                Ok(expr)
            } else {
                Ok(ctx.add(Expr::Neg(new_e)))
            }
        }

        // Function - recurse into args
        Expr::Function(name, args) => {
            let mut changed = false;
            let mut new_args = Vec::with_capacity(args.len());
            for arg in &args {
                let new_arg = resolve_recursive(ctx, *arg, store, cache, visiting)?;
                if new_arg != *arg {
                    changed = true;
                }
                new_args.push(new_arg);
            }
            if changed {
                Ok(ctx.add(Expr::Function(name, new_args)))
            } else {
                Ok(expr)
            }
        }

        // Matrix - recurse into elements
        Expr::Matrix { rows, cols, data } => {
            let mut changed = false;
            let mut new_data = Vec::with_capacity(data.len());
            for elem in &data {
                let new_elem = resolve_recursive(ctx, *elem, store, cache, visiting)?;
                if new_elem != *elem {
                    changed = true;
                }
                new_data.push(new_elem);
            }
            if changed {
                Ok(ctx.add(Expr::Matrix {
                    rows,
                    cols,
                    data: new_data,
                }))
            } else {
                Ok(expr)
            }
        }

        // Leaf nodes - no change needed
        Expr::Number(_) | Expr::Constant(_) => Ok(expr),
    }
}

fn resolve_session_id(
    ctx: &mut cas_ast::Context,
    id: EntryId,
    store: &SessionStore,
    cache: &mut HashMap<EntryId, ExprId>,
    visiting: &mut HashSet<EntryId>,
) -> Result<ExprId, ResolveError> {
    use cas_ast::Expr;

    // Check cache first
    if let Some(&resolved) = cache.get(&id) {
        return Ok(resolved);
    }

    // Cycle detection
    if visiting.contains(&id) {
        return Err(ResolveError::CircularReference(id));
    }

    // Get entry from store
    let entry = store.get(id).ok_or(ResolveError::NotFound(id))?;

    // Mark as visiting for cycle detection
    visiting.insert(id);

    // Get the expression to substitute
    let substitution = match &entry.kind {
        EntryKind::Expr(stored_expr) => {
            // Recursively resolve the stored expression (it may contain #refs too)
            resolve_recursive(ctx, *stored_expr, store, cache, visiting)?
        }
        EntryKind::Eq { lhs, rhs } => {
            // For equations used as expressions, use residue form: (lhs - rhs)
            let resolved_lhs = resolve_recursive(ctx, *lhs, store, cache, visiting)?;
            let resolved_rhs = resolve_recursive(ctx, *rhs, store, cache, visiting)?;
            ctx.add(Expr::Sub(resolved_lhs, resolved_rhs))
        }
    };

    // Done visiting
    visiting.remove(&id);

    // Cache the result
    cache.insert(id, substitution);

    Ok(substitution)
}

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

    // ========== resolve_session_refs Tests ==========

    #[test]
    fn test_resolve_simple_ref() {
        use cas_ast::{Context, DisplayExpr, Expr};

        let mut ctx = Context::new();
        let mut store = SessionStore::new();

        // Store x + 1 as #1
        let x = ctx.var("x");
        let one = ctx.num(1);
        let expr1 = ctx.add(Expr::Add(x, one));
        store.push(EntryKind::Expr(expr1), "x + 1".to_string());

        // Create #1 * 2
        let ref1 = ctx.add(Expr::SessionRef(1));
        let two = ctx.num(2);
        let input = ctx.add(Expr::Mul(ref1, two));

        // Resolve
        let resolved = resolve_session_refs(&mut ctx, input, &store).unwrap();

        // Check using DisplayExpr - should contain (x + 1) and 2
        let display = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: resolved
            }
        );
        // Resolved should not contain "#" anymore
        assert!(
            !display.contains('#'),
            "Resolved should not contain session refs: {}",
            display
        );
        // Should contain x and 2
        assert!(display.contains('x'), "Should contain x: {}", display);
        assert!(display.contains('2'), "Should contain 2: {}", display);
        // Should be a multiplication
        assert!(
            display.contains('*'),
            "Should contain multiplication: {}",
            display
        );
    }

    #[test]
    fn test_resolve_not_found() {
        use cas_ast::{Context, Expr};

        let mut ctx = Context::new();
        let store = SessionStore::new();

        // Reference to non-existent #99
        let ref99 = ctx.add(Expr::SessionRef(99));

        let result = resolve_session_refs(&mut ctx, ref99, &store);
        assert!(matches!(result, Err(ResolveError::NotFound(99))));
    }

    #[test]
    fn test_resolve_equation_as_residue() {
        use cas_ast::{Context, Expr};

        let mut ctx = Context::new();
        let mut store = SessionStore::new();

        // Store equation: x + 1 = 5 as #1
        let x = ctx.var("x");
        let one = ctx.num(1);
        let five = ctx.num(5);
        let lhs = ctx.add(Expr::Add(x, one));
        store.push(EntryKind::Eq { lhs, rhs: five }, "x + 1 = 5".to_string());

        // Create just #1
        let ref1 = ctx.add(Expr::SessionRef(1));

        // Resolve - should get (x + 1) - 5
        let resolved = resolve_session_refs(&mut ctx, ref1, &store).unwrap();

        // Should be Sub
        if let Expr::Sub(l, r) = ctx.get(resolved) {
            // Left should be (x + 1)
            assert!(matches!(ctx.get(*l), Expr::Add(_, _)));
            // Right should be 5
            if let Expr::Number(n) = ctx.get(*r) {
                assert_eq!(n.to_integer(), 5.into());
            } else {
                panic!("Expected Number(5)");
            }
        } else {
            panic!("Expected Sub for equation residue");
        }
    }

    #[test]
    fn test_resolve_no_refs() {
        use cas_ast::{Context, Expr};

        let mut ctx = Context::new();
        let store = SessionStore::new();

        // Expression without refs: x + 1
        let x = ctx.var("x");
        let one = ctx.num(1);
        let input = ctx.add(Expr::Add(x, one));

        // Should return same expression
        let resolved = resolve_session_refs(&mut ctx, input, &store).unwrap();
        assert_eq!(resolved, input);
    }

    #[test]
    fn test_resolve_chained_refs() {
        use cas_ast::{Context, DisplayExpr, Expr};

        let mut ctx = Context::new();
        let mut store = SessionStore::new();

        // #1 = x
        let x = ctx.var("x");
        store.push(EntryKind::Expr(x), "x".to_string());

        // #2 = #1 + 1 (references #1)
        let ref1 = ctx.add(Expr::SessionRef(1));
        let one = ctx.num(1);
        let expr2 = ctx.add(Expr::Add(ref1, one));
        store.push(EntryKind::Expr(expr2), "#1 + 1".to_string());

        // Input: #2 * 2
        let ref2 = ctx.add(Expr::SessionRef(2));
        let two = ctx.num(2);
        let input = ctx.add(Expr::Mul(ref2, two));

        // Resolve - should get (x + 1) * 2
        let resolved = resolve_session_refs(&mut ctx, input, &store).unwrap();

        // Check using DisplayExpr
        let display = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: resolved
            }
        );
        // Should not contain any # refs
        assert!(
            !display.contains('#'),
            "Resolved should not contain session refs: {}",
            display
        );
        // Should contain x, 1, 2 and be a multiplication
        assert!(display.contains('x'), "Should contain x: {}", display);
        assert!(display.contains('2'), "Should contain 2: {}", display);
        assert!(
            display.contains('*'),
            "Should contain multiplication: {}",
            display
        );
    }
}
