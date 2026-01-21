//! PolyStore: Opaque polynomial storage for fast mod-p operations.
//!
//! Stores polynomials as `MultiPolyModP` without materializing AST.
//! Used by `poly_mul_modp()` and `expand()` to avoid exponential AST growth.
//!
//! # Thread-Local Store
//!
//! For evaluation contexts where passing SessionState is not possible,
//! a thread-local store is provided. Use `with_thread_local_store()` to
//! run code with access to the store.

use crate::multipoly_modp::MultiPolyModP;
use std::cell::RefCell;

/// Opaque polynomial identifier
pub type PolyId = u32;

/// Metadata about a stored polynomial (without the full data)
#[derive(Debug, Clone)]
pub struct PolyMeta {
    /// Prime modulus used
    pub modulus: u64,
    /// Number of terms in the polynomial
    pub n_terms: usize,
    /// Number of variables
    pub n_vars: usize,
    /// Maximum total degree
    pub max_total_degree: u32,
    /// Variable names (for reconstruction)
    pub var_names: Vec<String>,
}

/// Storage for opaque polynomials (mod p)
#[derive(Debug, Default)]
pub struct PolyStore {
    polys: Vec<(PolyMeta, MultiPolyModP)>,
}

impl PolyStore {
    /// Create empty store
    pub fn new() -> Self {
        Self { polys: Vec::new() }
    }

    /// Insert a polynomial and return its ID
    pub fn insert(&mut self, meta: PolyMeta, poly: MultiPolyModP) -> PolyId {
        let id = self.polys.len() as PolyId;
        self.polys.push((meta, poly));
        id
    }

    /// Get polynomial by ID
    pub fn get(&self, id: PolyId) -> Option<(&PolyMeta, &MultiPolyModP)> {
        self.polys.get(id as usize).map(|(m, p)| (m, p))
    }

    /// Get metadata only
    pub fn meta(&self, id: PolyId) -> Option<&PolyMeta> {
        self.polys.get(id as usize).map(|(m, _)| m)
    }

    /// Number of stored polynomials
    pub fn len(&self) -> usize {
        self.polys.len()
    }

    /// Is store empty?
    pub fn is_empty(&self) -> bool {
        self.polys.is_empty()
    }

    /// Clear all stored polynomials
    pub fn clear(&mut self) {
        self.polys.clear();
    }

    /// Add two polynomials, returning new ID
    /// Uses VarTable unification to handle different variable orderings
    pub fn add(&mut self, a: PolyId, b: PolyId) -> Option<PolyId> {
        use crate::poly_modp_conv::VarTable;

        let (meta_a, poly_a) = self.get(a)?;
        let (meta_b, poly_b) = self.get(b)?;

        // Must have same modulus
        if meta_a.modulus != meta_b.modulus {
            return None;
        }

        // Fast path: same variable order
        if meta_a.var_names == meta_b.var_names {
            let result = poly_a.add(poly_b);
            let meta = PolyMeta {
                modulus: meta_a.modulus,
                n_terms: result.num_terms(),
                n_vars: meta_a.n_vars.max(meta_b.n_vars),
                max_total_degree: meta_a.max_total_degree.max(meta_b.max_total_degree),
                var_names: meta_a.var_names.clone(),
            };
            return Some(self.insert(meta, result));
        }

        // Phase 3: Use VarTable unification for different orderings
        let mut vt_a = VarTable::new();
        for name in &meta_a.var_names {
            vt_a.get_or_insert(name);
        }
        let mut vt_b = VarTable::new();
        for name in &meta_b.var_names {
            vt_b.get_or_insert(name);
        }

        let (unified, remap_a, remap_b) = vt_a.unify(&vt_b)?;

        // Remap both polynomials to unified variable space
        let poly_a_remapped = poly_a.remap(&remap_a, unified.len());
        let poly_b_remapped = poly_b.remap(&remap_b, unified.len());

        let result = poly_a_remapped.add(&poly_b_remapped);
        let meta = PolyMeta {
            modulus: meta_a.modulus,
            n_terms: result.num_terms(),
            n_vars: unified.len(),
            max_total_degree: meta_a.max_total_degree.max(meta_b.max_total_degree),
            var_names: unified.names().to_vec(),
        };

        Some(self.insert(meta, result))
    }

    /// Subtract two polynomials, returning new ID
    /// Uses VarTable unification to handle different variable orderings
    pub fn sub(&mut self, a: PolyId, b: PolyId) -> Option<PolyId> {
        use crate::poly_modp_conv::VarTable;

        let (meta_a, poly_a) = self.get(a)?;
        let (meta_b, poly_b) = self.get(b)?;

        if meta_a.modulus != meta_b.modulus {
            return None;
        }

        // Fast path: same variable order
        if meta_a.var_names == meta_b.var_names {
            let result = poly_a.sub(poly_b);
            let meta = PolyMeta {
                modulus: meta_a.modulus,
                n_terms: result.num_terms(),
                n_vars: meta_a.n_vars.max(meta_b.n_vars),
                max_total_degree: meta_a.max_total_degree.max(meta_b.max_total_degree),
                var_names: meta_a.var_names.clone(),
            };
            return Some(self.insert(meta, result));
        }

        // Phase 3: Use VarTable unification
        let mut vt_a = VarTable::new();
        for name in &meta_a.var_names {
            vt_a.get_or_insert(name);
        }
        let mut vt_b = VarTable::new();
        for name in &meta_b.var_names {
            vt_b.get_or_insert(name);
        }

        let (unified, remap_a, remap_b) = vt_a.unify(&vt_b)?;

        let poly_a_remapped = poly_a.remap(&remap_a, unified.len());
        let poly_b_remapped = poly_b.remap(&remap_b, unified.len());

        let result = poly_a_remapped.sub(&poly_b_remapped);
        let meta = PolyMeta {
            modulus: meta_a.modulus,
            n_terms: result.num_terms(),
            n_vars: unified.len(),
            max_total_degree: meta_a.max_total_degree.max(meta_b.max_total_degree),
            var_names: unified.names().to_vec(),
        };

        Some(self.insert(meta, result))
    }

    /// Multiply two polynomials, returning new ID
    /// Uses VarTable unification to handle different variable orderings
    pub fn mul(&mut self, a: PolyId, b: PolyId) -> Option<PolyId> {
        use crate::poly_modp_conv::VarTable;

        let (meta_a, poly_a) = self.get(a)?;
        let (meta_b, poly_b) = self.get(b)?;

        if meta_a.modulus != meta_b.modulus {
            return None;
        }

        // Fast path: same variable order
        if meta_a.var_names == meta_b.var_names {
            let result = poly_a.mul(poly_b);
            let meta = PolyMeta {
                modulus: meta_a.modulus,
                n_terms: result.num_terms(),
                n_vars: meta_a.n_vars.max(meta_b.n_vars),
                max_total_degree: meta_a.max_total_degree + meta_b.max_total_degree,
                var_names: meta_a.var_names.clone(),
            };
            return Some(self.insert(meta, result));
        }

        // Phase 3: Use VarTable unification
        let mut vt_a = VarTable::new();
        for name in &meta_a.var_names {
            vt_a.get_or_insert(name);
        }
        let mut vt_b = VarTable::new();
        for name in &meta_b.var_names {
            vt_b.get_or_insert(name);
        }

        let (unified, remap_a, remap_b) = vt_a.unify(&vt_b)?;

        let poly_a_remapped = poly_a.remap(&remap_a, unified.len());
        let poly_b_remapped = poly_b.remap(&remap_b, unified.len());

        let result = poly_a_remapped.mul(&poly_b_remapped);
        let meta = PolyMeta {
            modulus: meta_a.modulus,
            n_terms: result.num_terms(),
            n_vars: unified.len(),
            max_total_degree: meta_a.max_total_degree + meta_b.max_total_degree,
            var_names: unified.names().to_vec(),
        };

        Some(self.insert(meta, result))
    }

    /// Negate a polynomial, returning new ID
    pub fn neg(&mut self, a: PolyId) -> Option<PolyId> {
        let (meta_a, poly_a) = self.get(a)?;

        let result = poly_a.neg();
        let meta = meta_a.clone();

        Some(self.insert(meta, result))
    }

    /// Raise polynomial to power, returning new ID
    pub fn pow(&mut self, a: PolyId, n: u32) -> Option<PolyId> {
        let (meta_a, poly_a) = self.get(a)?;

        let result = poly_a.pow(n);
        let meta = PolyMeta {
            modulus: meta_a.modulus,
            n_terms: result.num_terms(),
            n_vars: meta_a.n_vars,
            max_total_degree: meta_a.max_total_degree * n,
            var_names: meta_a.var_names.clone(),
        };

        Some(self.insert(meta, result))
    }
}

/// Maximum terms to store in a single polynomial.
/// Above this, poly_mul_modp() aborts to prevent memory explosion.
pub const POLY_MAX_STORE_TERMS: usize = 10_000_000;

// =============================================================================
// Thread-local PolyStore for evaluation contexts
// =============================================================================

thread_local! {
    static THREAD_POLY_STORE: RefCell<PolyStore> = RefCell::new(PolyStore::new());
}

/// Execute a function with access to the thread-local PolyStore.
/// The store is cleared before each evaluation to prevent state leakage.
pub fn with_thread_local_store<F, R>(f: F) -> R
where
    F: FnOnce(&mut PolyStore) -> R,
{
    THREAD_POLY_STORE.with(|store| {
        let mut store = store.borrow_mut();
        f(&mut store)
    })
}

/// Clear the thread-local store (call at start of evaluation)
pub fn clear_thread_local_store() {
    THREAD_POLY_STORE.with(|store| {
        store.borrow_mut().clear();
    });
}

/// Insert a polynomial into thread-local store
pub fn thread_local_insert(meta: PolyMeta, poly: MultiPolyModP) -> PolyId {
    THREAD_POLY_STORE.with(|store| store.borrow_mut().insert(meta, poly))
}

/// Get metadata from thread-local store
pub fn thread_local_meta(id: PolyId) -> Option<PolyMeta> {
    THREAD_POLY_STORE.with(|store| store.borrow().meta(id).cloned())
}

/// Add two polys in thread-local store
pub fn thread_local_add(a: PolyId, b: PolyId) -> Option<PolyId> {
    THREAD_POLY_STORE.with(|store| store.borrow_mut().add(a, b))
}

/// Subtract two polys in thread-local store
pub fn thread_local_sub(a: PolyId, b: PolyId) -> Option<PolyId> {
    THREAD_POLY_STORE.with(|store| store.borrow_mut().sub(a, b))
}

/// Multiply two polys in thread-local store
pub fn thread_local_mul(a: PolyId, b: PolyId) -> Option<PolyId> {
    THREAD_POLY_STORE.with(|store| store.borrow_mut().mul(a, b))
}

/// Negate poly in thread-local store
pub fn thread_local_neg(a: PolyId) -> Option<PolyId> {
    THREAD_POLY_STORE.with(|store| store.borrow_mut().neg(a))
}

/// Pow poly in thread-local store
pub fn thread_local_pow(a: PolyId, n: u32) -> Option<PolyId> {
    THREAD_POLY_STORE.with(|store| store.borrow_mut().pow(a, n))
}

/// Get polynomial from thread-local store for materialization
pub fn thread_local_get_for_materialize(id: PolyId) -> Option<(PolyMeta, MultiPolyModP)> {
    THREAD_POLY_STORE.with(|store| store.borrow().get(id).map(|(m, p)| (m.clone(), p.clone())))
}

/// Render a polynomial directly to string without AST construction.
/// Used for auto-display of poly_result expressions.
/// Returns None if id is invalid.
pub fn render_poly_result(id: PolyId, max_terms: usize) -> Option<String> {
    use crate::mono::Mono;
    use std::fmt::Write;

    let (meta, poly) = thread_local_get_for_materialize(id)?;

    if poly.terms.is_empty() {
        return Some("0".to_string());
    }

    // Sort terms by graded lex order
    let mut sorted_terms: Vec<&(Mono, u64)> = poly.terms.iter().collect();
    sorted_terms.sort_by(|a, b| {
        let deg_cmp = b.0.total_degree().cmp(&a.0.total_degree());
        if deg_cmp != std::cmp::Ordering::Equal {
            return deg_cmp;
        }
        b.0.cmp(&a.0)
    });

    let total_terms = sorted_terms.len();
    let show_terms = total_terms.min(max_terms);
    let truncated = total_terms > max_terms;

    let mut result = String::with_capacity(show_terms * 25);

    for (i, (mono, coeff)) in sorted_terms.iter().take(show_terms).enumerate() {
        let is_constant = mono.total_degree() == 0;

        if i == 0 {
            if is_constant {
                write!(result, "{}", coeff).unwrap();
            } else if *coeff == 1 {
                // coefficient 1 is implicit
            } else {
                write!(result, "{}", coeff).unwrap();
            }
        } else if is_constant || *coeff != 1 {
            write!(result, " + {}", coeff).unwrap();
        } else {
            result.push_str(" + ");
        }

        // Format monomial
        let mut first_var = i == 0 && *coeff == 1 && !is_constant;
        for (var_idx, &exp) in mono.0.iter().enumerate() {
            if exp > 0 && var_idx < meta.var_names.len() {
                if !first_var {
                    result.push('·');
                }
                first_var = false;
                result.push_str(&meta.var_names[var_idx]);
                if exp > 1 {
                    write!(result, "^{}", exp).unwrap();
                }
            }
        }
    }

    if truncated {
        let remaining = total_terms - max_terms;
        write!(result, " + ... (+{} more terms)", remaining).unwrap();
    }

    Some(result)
}

/// Check if an expression is a poly_result and render it directly.
/// Returns None if not a poly_result or if rendering fails.
/// Default max_terms is 100000 for auto-display (shows full polynomial for large products).
pub fn try_render_poly_result(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> Option<String> {
    use cas_ast::Expr;

    if let Expr::Function(name, args) = ctx.get(expr) {
        if name == "poly_result" && args.len() == 1 {
            if let Expr::Number(n) = ctx.get(args[0]) {
                if let Ok(id) = n.to_integer().try_into() {
                    let id: PolyId = id;
                    // Default limit for auto-display - 100k terms covers large products
                    return render_poly_result(id, 100_000);
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poly_store_basic() {
        let mut store = PolyStore::new();
        assert!(store.is_empty());

        let p = MultiPolyModP::constant(42, 101, 2);
        let meta = PolyMeta {
            modulus: 101,
            n_terms: 1,
            n_vars: 2,
            max_total_degree: 0,
            var_names: vec!["x".to_string(), "y".to_string()],
        };

        let id = store.insert(meta, p);
        assert_eq!(id, 0);
        assert_eq!(store.len(), 1);

        let (m, poly) = store.get(id).unwrap();
        assert_eq!(m.modulus, 101);
        assert_eq!(poly.num_terms(), 1);
    }

    #[test]
    fn test_thread_local_store() {
        clear_thread_local_store();

        let p = MultiPolyModP::constant(42, 101, 2);
        let meta = PolyMeta {
            modulus: 101,
            n_terms: 1,
            n_vars: 2,
            max_total_degree: 0,
            var_names: vec!["x".to_string(), "y".to_string()],
        };

        let id = thread_local_insert(meta, p);
        assert_eq!(id, 0);

        let m = thread_local_meta(id).unwrap();
        assert_eq!(m.modulus, 101);
    }
}
