//! Nombres sintéticos frescos que ESQUIVAN los ya presentes en un árbol.
//!
//! Clase L15 del ledger de saneamiento: los generadores de temporales
//! (`__opq{N}`, `__polyzero_*`…) que no comprueban colisiones fusionan átomos
//! cuando una ronda anidada hereda nombres del nivel exterior — el gate
//! "exacto" colapsa entonces no-constantes a constantes (wrong answer 7/3).
//! Este módulo es el implementador ÚNICO del patrón; antes había tres copias
//! con la misma semántica (engine/polynomial_identity, math/div_expand_cancel,
//! math/poly_compare) y la deriva entre copias fue precisamente la causa del
//! P0. El cuarto asignador (`verification_algebraic::fresh_atom_name`) tiene
//! semántica PROPIA de stride y queda fuera a propósito.

use crate::{Context, ExprId};
use std::collections::HashSet;

/// Conjunto de nombres a esquivar: las variables presentes en `roots`.
pub fn taken_variable_names(ctx: &Context, roots: &[ExprId]) -> HashSet<String> {
    let mut taken = HashSet::new();
    for &root in roots {
        taken.extend(crate::collect_variables(ctx, root));
    }
    taken
}

/// Asigna `{prefix}{N}` con el menor `N ≥ preferred` libre en `taken`, lo
/// marca como usado y lo devuelve. Monótono si se reutiliza el mismo `taken`:
/// dos llamadas nunca devuelven el mismo nombre.
pub fn alloc_indexed_name(taken: &mut HashSet<String>, prefix: &str, preferred: usize) -> String {
    let mut idx = preferred;
    loop {
        let candidate = format!("{prefix}{idx}");
        if taken.insert(candidate.clone()) {
            return candidate;
        }
        idx += 1;
    }
}

/// Base segura para esquemas `prefix{base + i}` con `i` monótono creciente:
/// el primer índice ESTRICTAMENTE por encima de todo `{prefix}{N}` presente.
///
/// Distinción que importa (y que una migración descuidada rompió): con huecos
/// en `taken` (p.ej. solo `{prefix}1`), [`alloc_indexed_name`] devolvería 0 y
/// un esquema `base + i` colisionaría en i=1. Para asignación UNO-A-UNO usa
/// `alloc_indexed_name`; para un OFFSET de lote usa esta.
pub fn fresh_suffix_base(taken: &HashSet<String>, prefix: &str) -> usize {
    taken
        .iter()
        .filter_map(|n| n.strip_prefix(prefix).and_then(|s| s.parse::<usize>().ok()))
        .max()
        .map_or(0, |m| m + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_skips_taken_and_never_repeats() {
        let mut taken: HashSet<String> = ["__opq0".to_string(), "__opq2".to_string()]
            .into_iter()
            .collect();
        assert_eq!(alloc_indexed_name(&mut taken, "__opq", 0), "__opq1");
        assert_eq!(alloc_indexed_name(&mut taken, "__opq", 0), "__opq3");
        assert_eq!(alloc_indexed_name(&mut taken, "__opq", 5), "__opq5");
    }

    #[test]
    fn suffix_base_clears_gaps() {
        let taken: HashSet<String> = ["__polyzero_atom_1".to_string()].into_iter().collect();
        // alloc daría 0 (primer libre) — un esquema base+i chocaría en i=1;
        // la base de lote debe saltar POR ENCIMA del hueco.
        assert_eq!(fresh_suffix_base(&taken, "__polyzero_atom_"), 2);
        assert_eq!(fresh_suffix_base(&taken, "__otra_"), 0);
    }
}
