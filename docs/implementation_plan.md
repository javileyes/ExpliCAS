> [!NOTE]
> **Archived Plan (Dec 2025)** — This implementation plan documents a proposed algebra.rs split that was deferred.
> File paths (e.g., `timeline.rs`) may reference pre-refactor locations.
> See [ARCHITECTURE.md](ARCHITECTURE.md) for current project structure.

# Fase 3: División de Archivos Grandes

Dividir [algebra.rs](file:///Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/algebra.rs) (2649 líneas) en módulos más pequeños y manejables.

## Análisis del Contenido Actual

El archivo [algebra.rs](file:///Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/algebra.rs) contiene:

| Categoría | Líneas Aprox | Reglas |
|-----------|--------------|--------|
| Simplificación de Fracciones | ~900 | `SimplifyFractionRule`, `SimplifyMulDivRule`, [are_denominators_opposite](file:///Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/algebra.rs#907-1018) |
| Distribución y Expansión | ~200 | `ExpandRule`, `ConservativeExpandRule`, `DistributeRule` |
| Factorización | ~150 | `FactorRule` |
| Helpers | ~100 | [gcd_rational](file:///Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/algebra.rs#107-117), [smart_mul](file:///Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/algebra.rs#241-251), [distribute](file:///Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/algebra.rs#252-300), [collect_denominators](file:///Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/algebra.rs#322-347), etc. |
| Canonicalización | ~800 | Reglas de ordenamiento, asociatividad |
| Tests | ~500 | Tests unitarios |
| Resto | ~200 | [register](file:///Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/trig_canonicalization.rs#674-687), imports |

---

## Propuesta de División

### Nueva Estructura

```
rules/
├── algebra/
│   ├── mod.rs           # Re-exports y register
│   ├── fractions.rs     # Reglas de fracciones
│   ├── distribution.rs  # Expand, Distribute
│   ├── factoring.rs     # Factor
│   └── helpers.rs       # Helpers locales de algebra
└── algebra.rs           # Mantener por compatibilidad (re-export)
```

> [!IMPORTANT]
> Esta refactorización es significativa. Se preservará la API pública.

---

## Cambios Propuestos

### [NEW] `rules/algebra/mod.rs`
- Re-exports públicos de todos los submódulos
- Función [register()](file:///Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/trig_canonicalization.rs#674-687) que delega a submódulos

### [NEW] `rules/algebra/fractions.rs`
- `SimplifyFractionRule`
- `SimplifyMulDivRule`
- [gcd_rational()](file:///Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/algebra.rs#107-117)
- [are_denominators_opposite()](file:///Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/algebra.rs#907-1018)
- Tests de fracciones

### [NEW] `rules/algebra/distribution.rs`  
- `ExpandRule`
- `ConservativeExpandRule`
- `DistributeRule`
- [distribute()](file:///Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/algebra.rs#252-300) helper
- Tests de distribución

### [NEW] `rules/algebra/factoring.rs`
- `FactorRule`
- Tests de factorización

### [NEW] `rules/algebra/helpers.rs`
- [count_nodes_of_type()](file:///Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/algebra.rs#213-237)
- [smart_mul()](file:///Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/algebra.rs#241-251)
- [get_quotient()](file:///Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/algebra.rs#324-344)
- [collect_denominators()](file:///Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/algebra.rs#322-347)
- [collect_variables()](file:///Users/javiergimenezmoya/developer/math/crates/cas_engine/src/engine.rs#408-416)

### [MODIFY] [rules/algebra.rs](file:///Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/algebra.rs)
- Simplificado a un re-export de `rules/algebra/mod.rs`

---

## Alternativa Simple

Si la refactorización completa es muy arriesgada, una alternativa mínima:

1. Mover solo los tests a `rules/algebra_tests.rs`
2. Extraer helpers a `rules/algebra_helpers.rs`

Esto reduciría ~600 líneas sin cambiar la estructura de reglas.

---

## Verificación

- Compilar con `cargo build -p cas_engine`
- Ejecutar tests: `cargo test -p cas_engine`
- Verificar que todas las reglas siguen registradas

---

## Decisión Requerida

1. **Opción A**: División completa en módulos (más trabajo, mejor organización)
2. **Opción B**: Extracción mínima de tests y helpers (~600 líneas menos)
3. **Opción C**: Posponer división de [algebra.rs](file:///Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/algebra.rs), enfocar en [trigonometry.rs](file:///Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/trigonometry.rs) o [timeline.rs](file:///Users/javiergimenezmoya/developer/math/crates/cas_engine/src/timeline.rs)

¿Cuál prefieres?
