//! Session Snapshot Persistence (V2.15.36)
//!
//! Enables persistent sessions across CLI invocations via binary snapshots.
//! The snapshot contains the complete Context (arena) and SessionStore,
//! allowing `#N` references and cached results to survive process restarts.

use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::{Entry, EntryKind, SessionStore, SimplifiedCache, SimplifyCacheKey};
use cas_session_core::types::CacheConfig;

/// Complete session state for persistence.
/// Contains header for compatibility checking, plus Context and SessionStore.
#[derive(Debug, Serialize, Deserialize)]
pub struct SessionSnapshot {
    pub header: SnapshotHeader,
    pub context: ContextSnapshot,
    pub session: SessionStoreSnapshot,
}

/// Header for version checking and cache invalidation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SnapshotHeader {
    /// Magic bytes for file identification
    pub magic: [u8; 8],
    /// Format version (increment on breaking changes)
    pub version: u32,
    /// Cache key for invalidation (domain mode + ruleset version)
    pub cache_key: SimplifyCacheKey,
}

impl SnapshotHeader {
    pub const MAGIC: [u8; 8] = *b"EXPLICAS";
    pub const VERSION: u32 = 1;

    pub fn new(cache_key: SimplifyCacheKey) -> Self {
        Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            cache_key,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.magic == Self::MAGIC && self.version == Self::VERSION
    }
}

/// Serializable Context representation.
/// Since cas_ast::Context doesn't have serde, we serialize the nodes array manually.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextSnapshot {
    /// The expression nodes in arena order (ExprId.index() = position)
    pub nodes: Vec<ExprNodeSnapshot>,
}

/// Serializable Expr variant - mirrors cas_ast::Expr
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExprNodeSnapshot {
    Number {
        num: String,
        den: String,
    }, // BigRational as strings for portability
    Constant(ConstantSnapshot),
    Variable(String),
    Add(u32, u32), // ExprId indices
    Sub(u32, u32),
    Mul(u32, u32),
    Div(u32, u32),
    Pow(u32, u32),
    Neg(u32),
    Function(String, Vec<u32>),
    Matrix {
        rows: usize,
        cols: usize,
        data: Vec<u32>,
    },
    SessionRef(u64),
    Hold(u32), // ExprId index for inner expression
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstantSnapshot {
    Pi,
    E,
    Infinity,
    Undefined,
    I,
    Phi,
}

/// Serializable SessionStore representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStoreSnapshot {
    pub next_id: u64,
    pub entries: Vec<EntrySnapshot>,
    pub cache_order: Vec<u64>,
    pub cache_config: CacheConfigSnapshot,
    pub cached_steps_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntrySnapshot {
    pub id: u64,
    pub raw_text: String,
    pub kind: EntryKindSnapshot,
    pub simplified: Option<SimplifiedCacheSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntryKindSnapshot {
    Expr(u32), // ExprId index
    Eq { lhs: u32, rhs: u32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimplifiedCacheSnapshot {
    pub key: SimplifyCacheKey,
    pub expr: u32, // ExprId index
                   // Note: We don't persist steps (light cache by design for snapshots)
                   // requires are also omitted for simplicity - they'll be recalculated if needed
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfigSnapshot {
    pub max_cached_entries: usize,
    pub max_cached_steps: usize,
    pub light_cache_threshold: Option<usize>,
}

// ============================================================================
// Conversion: Context -> ContextSnapshot
// ============================================================================

impl ContextSnapshot {
    pub fn from_context(ctx: &cas_ast::Context) -> Self {
        use cas_ast::Expr;

        let nodes = ctx
            .nodes
            .iter()
            .map(|expr| match expr {
                Expr::Number(r) => ExprNodeSnapshot::Number {
                    num: r.numer().to_string(),
                    den: r.denom().to_string(),
                },
                Expr::Constant(c) => ExprNodeSnapshot::Constant(ConstantSnapshot::from(c)),
                Expr::Variable(sym_id) => {
                    ExprNodeSnapshot::Variable(ctx.sym_name(*sym_id).to_string())
                }
                Expr::Add(l, r) => ExprNodeSnapshot::Add(l.index() as u32, r.index() as u32),
                Expr::Sub(l, r) => ExprNodeSnapshot::Sub(l.index() as u32, r.index() as u32),
                Expr::Mul(l, r) => ExprNodeSnapshot::Mul(l.index() as u32, r.index() as u32),
                Expr::Div(l, r) => ExprNodeSnapshot::Div(l.index() as u32, r.index() as u32),
                Expr::Pow(l, r) => ExprNodeSnapshot::Pow(l.index() as u32, r.index() as u32),
                Expr::Neg(inner) => ExprNodeSnapshot::Neg(inner.index() as u32),
                Expr::Function(fn_id, args) => ExprNodeSnapshot::Function(
                    ctx.sym_name(*fn_id).to_string(),
                    args.iter().map(|a| a.index() as u32).collect(),
                ),
                Expr::Matrix { rows, cols, data } => ExprNodeSnapshot::Matrix {
                    rows: *rows,
                    cols: *cols,
                    data: data.iter().map(|d| d.index() as u32).collect(),
                },
                Expr::SessionRef(id) => ExprNodeSnapshot::SessionRef(*id),
                Expr::Hold(inner) => ExprNodeSnapshot::Hold(inner.index() as u32),
            })
            .collect();

        Self { nodes }
    }

    pub fn into_context(self) -> cas_ast::Context {
        use cas_ast::{Context, Expr, ExprId};
        use num_bigint::BigInt;
        use num_rational::BigRational;

        let mut ctx = Context::new();

        // Reconstruct nodes in the same order to preserve ExprId stability
        for node in self.nodes {
            let expr = match node {
                ExprNodeSnapshot::Number { num, den } => {
                    let n: BigInt = num.parse().unwrap_or_default();
                    let d: BigInt = den.parse().unwrap_or_else(|_| BigInt::from(1));
                    Expr::Number(BigRational::new(n, d))
                }
                ExprNodeSnapshot::Constant(c) => Expr::Constant(c.into()),
                ExprNodeSnapshot::Variable(s) => Expr::Variable(ctx.intern_symbol(&s)),
                ExprNodeSnapshot::Add(l, r) => Expr::Add(ExprId::from_raw(l), ExprId::from_raw(r)),
                ExprNodeSnapshot::Sub(l, r) => Expr::Sub(ExprId::from_raw(l), ExprId::from_raw(r)),
                ExprNodeSnapshot::Mul(l, r) => Expr::Mul(ExprId::from_raw(l), ExprId::from_raw(r)),
                ExprNodeSnapshot::Div(l, r) => Expr::Div(ExprId::from_raw(l), ExprId::from_raw(r)),
                ExprNodeSnapshot::Pow(l, r) => Expr::Pow(ExprId::from_raw(l), ExprId::from_raw(r)),
                ExprNodeSnapshot::Neg(inner) => Expr::Neg(ExprId::from_raw(inner)),
                ExprNodeSnapshot::Function(name, args) => {
                    let fn_id = ctx.intern_symbol(&name);
                    Expr::Function(fn_id, args.into_iter().map(ExprId::from_raw).collect())
                }
                ExprNodeSnapshot::Matrix { rows, cols, data } => Expr::Matrix {
                    rows,
                    cols,
                    data: data.into_iter().map(ExprId::from_raw).collect(),
                },
                ExprNodeSnapshot::SessionRef(id) => Expr::SessionRef(id),
                ExprNodeSnapshot::Hold(inner) => Expr::Hold(ExprId::from_raw(inner)),
            };
            // Use add_raw to preserve exact structure without re-canonicalization
            ctx.nodes.push(expr);
        }

        ctx
    }
}

impl From<&cas_ast::Constant> for ConstantSnapshot {
    fn from(c: &cas_ast::Constant) -> Self {
        use cas_ast::Constant;
        match c {
            Constant::Pi => ConstantSnapshot::Pi,
            Constant::E => ConstantSnapshot::E,
            Constant::Infinity => ConstantSnapshot::Infinity,
            Constant::Undefined => ConstantSnapshot::Undefined,
            Constant::I => ConstantSnapshot::I,
            Constant::Phi => ConstantSnapshot::Phi,
        }
    }
}

impl From<ConstantSnapshot> for cas_ast::Constant {
    fn from(c: ConstantSnapshot) -> Self {
        use cas_ast::Constant;
        match c {
            ConstantSnapshot::Pi => Constant::Pi,
            ConstantSnapshot::E => Constant::E,
            ConstantSnapshot::Infinity => Constant::Infinity,
            ConstantSnapshot::Undefined => Constant::Undefined,
            ConstantSnapshot::I => Constant::I,
            ConstantSnapshot::Phi => Constant::Phi,
        }
    }
}

// ============================================================================
// Conversion: SessionStore <-> SessionStoreSnapshot
// ============================================================================

impl SessionStoreSnapshot {
    pub fn from_store(store: &SessionStore) -> Self {
        let entries = store
            .entries()
            .map(|e| EntrySnapshot {
                id: e.id,
                raw_text: e.raw_text.clone(),
                kind: match &e.kind {
                    EntryKind::Expr(id) => EntryKindSnapshot::Expr(id.index() as u32),
                    EntryKind::Eq { lhs, rhs } => EntryKindSnapshot::Eq {
                        lhs: lhs.index() as u32,
                        rhs: rhs.index() as u32,
                    },
                },
                simplified: e.simplified.as_ref().map(|s| SimplifiedCacheSnapshot {
                    key: s.key.clone(),
                    expr: s.expr.index() as u32,
                    // Light cache: don't persist steps
                }),
            })
            .collect();

        let (_cached_entries, cached_steps) = store.cache_stats();

        Self {
            next_id: store.next_id(),
            entries,
            cache_order: store.cache_order().iter().copied().collect(),
            cache_config: CacheConfigSnapshot {
                max_cached_entries: store.cache_config().max_cached_entries,
                max_cached_steps: store.cache_config().max_cached_steps,
                light_cache_threshold: store.cache_config().light_cache_threshold,
            },
            cached_steps_count: cached_steps,
        }
    }

    pub fn into_store(self) -> SessionStore {
        use cas_ast::ExprId;

        let config = CacheConfig {
            max_cached_entries: self.cache_config.max_cached_entries,
            max_cached_steps: self.cache_config.max_cached_steps,
            light_cache_threshold: self.cache_config.light_cache_threshold,
        };

        let mut store = SessionStore::with_cache_config(config);

        for entry in self.entries {
            let kind = match entry.kind {
                EntryKindSnapshot::Expr(id) => EntryKind::Expr(ExprId::from_raw(id)),
                EntryKindSnapshot::Eq { lhs, rhs } => EntryKind::Eq {
                    lhs: ExprId::from_raw(lhs),
                    rhs: ExprId::from_raw(rhs),
                },
            };

            let simplified = entry.simplified.map(|s| SimplifiedCache {
                key: s.key,
                expr: ExprId::from_raw(s.expr),
                requires: Vec::new(), // Recalculated on use if needed
                steps: None,          // Light cache: no steps persisted
            });

            store.restore_entry(Entry {
                id: entry.id,
                kind,
                raw_text: entry.raw_text,
                diagnostics: Default::default(),
                simplified,
            });
        }

        // Restore LRU order
        store.restore_cache_order(self.cache_order);

        store
    }
}

// ============================================================================
// SessionSnapshot: Main API
// ============================================================================

impl SessionSnapshot {
    pub fn new(
        context: &cas_ast::Context,
        session: &SessionStore,
        cache_key: SimplifyCacheKey,
    ) -> Self {
        Self {
            header: SnapshotHeader::new(cache_key),
            context: ContextSnapshot::from_context(context),
            session: SessionStoreSnapshot::from_store(session),
        }
    }

    pub fn is_compatible(&self, key: &SimplifyCacheKey) -> bool {
        self.header.is_valid() && &self.header.cache_key == key
    }

    pub fn load(path: &Path) -> Result<Self, SnapshotError> {
        let bytes = fs::read(path)?;
        let snap: SessionSnapshot = bincode::deserialize(&bytes)?;
        Ok(snap)
    }

    /// Atomic save: write to temp file then rename.
    pub fn save_atomic(&self, path: &Path) -> Result<(), SnapshotError> {
        let tmp = tmp_path(path);
        let bytes = bincode::serialize(self)?;
        fs::write(&tmp, bytes)?;
        fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Extract Context and SessionStore from snapshot.
    pub fn into_parts(self) -> (cas_ast::Context, SessionStore) {
        (self.context.into_context(), self.session.into_store())
    }
}

fn tmp_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".tmp");
    path.with_file_name(name)
}

// ============================================================================
// Error type
// ============================================================================

#[derive(Debug)]
pub enum SnapshotError {
    Io(io::Error),
    Bincode(Box<bincode::ErrorKind>),
}

impl std::fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SnapshotError::Io(e) => write!(f, "IO error: {}", e),
            SnapshotError::Bincode(e) => write!(f, "Serialization error: {}", e),
        }
    }
}

impl std::error::Error for SnapshotError {}

impl From<io::Error> for SnapshotError {
    fn from(e: io::Error) -> Self {
        SnapshotError::Io(e)
    }
}

impl From<Box<bincode::ErrorKind>> for SnapshotError {
    fn from(e: Box<bincode::ErrorKind>) -> Self {
        SnapshotError::Bincode(e)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_context_snapshot_roundtrip() {
        let mut ctx = cas_ast::Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let _expr = ctx.add(cas_ast::Expr::Mul(x, two));

        let snapshot = ContextSnapshot::from_context(&ctx);
        let restored = snapshot.into_context();

        // Nodes should match
        assert_eq!(ctx.nodes.len(), restored.nodes.len());
    }

    #[test]
    fn test_session_snapshot_save_load() {
        use cas_engine::domain::DomainMode;

        let dir = tempdir().unwrap();
        let path = dir.path().join("test.session");

        // Create a context with some expressions
        let mut ctx = cas_ast::Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let expr = ctx.add(cas_ast::Expr::Add(x, one));

        // Create a session store with an entry
        let mut store = SessionStore::new();
        store.push(crate::EntryKind::Expr(expr), "x + 1".to_string());

        let key = SimplifyCacheKey {
            domain: DomainMode::Generic,
            ruleset_rev: 1,
        };

        // Save
        let snapshot = SessionSnapshot::new(&ctx, &store, key.clone());
        snapshot.save_atomic(&path).unwrap();

        // Load
        let loaded = SessionSnapshot::load(&path).unwrap();
        assert!(loaded.is_compatible(&key));

        // Verify
        let (restored_ctx, restored_store) = loaded.into_parts();
        assert_eq!(ctx.nodes.len(), restored_ctx.nodes.len());
        assert_eq!(store.len(), restored_store.len());
    }
}
