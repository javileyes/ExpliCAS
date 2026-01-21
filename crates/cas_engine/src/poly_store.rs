//! PolyStore: Opaque polynomial storage for fast mod-p operations.
//!
//! Stores polynomials as `MultiPolyModP` without materializing AST.
//! Used by `poly_mul_modp()` to avoid exponential AST growth.

use crate::multipoly_modp::MultiPolyModP;

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
}

/// Maximum terms to store in a single polynomial.
/// Above this, poly_mul_modp() aborts to prevent memory explosion.
pub const POLY_MAX_STORE_TERMS: usize = 10_000_000;

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
}
