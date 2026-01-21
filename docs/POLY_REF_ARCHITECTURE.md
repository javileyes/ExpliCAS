# Poly-Ref Architecture: Opaque Polynomial Domain

## Status: DESIGN PHASE

Este documento define la arquitectura para **representación opaca de polinomios** (`poly_ref`) como solución definitiva al problema de explosión AST en expansiones grandes.

---

## 1. Problema Raíz

### Situación Actual (Problemática)

```
expand(A^7) + expand(B^7)
    │              │
    ▼              ▼
__hold(3432)   __hold(3432)   ← ASTs materializados
    │              │
    └──────┬───────┘
           ▼
   Add(__hold, __hold)
           │
           ▼
   Simplifier procesa 6864 términos → 59 segundos
```

### Situación Objetivo

```
expand(A^7) + expand(B^7)
    │              │
    ▼              ▼
poly_ref(1)     poly_ref(2)    ← Referencias opacas
    │              │
    └──────┬───────┘
           ▼
    poly_lower_pass
           │
           ▼
      poly_ref(3)              ← Suma interna en PolyStore
           │
           ▼
   Simplifier ve solo 1 átomo → milisegundos
```

---

## 2. Arquitectura de Dos Mundos

### Mundo AST
- `Expr` binario: `Add/Mul/Pow/...`
- Donde vive el simplificador, reglas, display
- Coste O(n) traversal por operación

### Mundo Polynomial
- `PolyRepr`: HashMap `Monomial → Coeff`
- Operaciones O(n_términos) para add, O(n·m) para mul
- Sin traversal recursivo
- Vive en `PolyStore`

### Frontera
```
┌─────────────────────────────────────────────────────────────────┐
│                        MUNDO AST                                │
│   Expr::Add, Expr::Mul, Expr::Pow, ...                         │
│   poly_ref(id) ← ÁTOMO OPACO                                   │
└─────────────────────────────────────────────────────────────────┘
                    ▲                          │
                    │ poly_to_expr(id)         │ expr_to_poly(expr)
                    │ (materializa)            │ (convierte)
                    │                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MUNDO POLYNOMIAL                            │
│   PolyStore { polys: Vec<(PolyMeta, PolyRepr)> }               │
│   Operaciones: add, mul, pow, gcd                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Estructuras de Datos Core

### `PolyId`
```rust
pub type PolyId = u32;
```

### `VarTable`
```rust
/// Mapea símbolos a índices de variable (canónico y estable)
#[derive(Debug, Clone, Default)]
pub struct VarTable {
    /// Nombres de variables en orden canónico
    pub names: Vec<String>,
    /// Lookup: nombre → índice
    index: FxHashMap<String, u8>,
}

impl VarTable {
    /// Obtiene o asigna índice para una variable
    pub fn get_or_insert(&mut self, name: &str) -> u8;
    
    /// Unifica dos VarTables, devuelve (unión, remap_self, remap_other)
    pub fn unify(&self, other: &VarTable) -> (VarTable, Vec<u8>, Vec<u8>);
}
```

### `PolyRepr`
```rust
pub enum PolyRepr {
    /// Coeficientes exactos (i128 con fallback a BigRational)
    Exact(MultiPolyExact),
    /// Módulo p (rápido, pero pierde precisión para materialización)
    ModP(MultiPolyModP),
}
```

### `PolyMeta`
```rust
#[derive(Debug, Clone)]
pub struct PolyMeta {
    pub n_terms: usize,
    pub n_vars: usize,
    pub max_total_degree: u32,
    pub var_table: VarTable,
    pub repr_kind: PolyReprKind,  // Exact | ModP
    pub modulus: Option<u64>,     // Si ModP
    pub materializable: bool,     // true si se puede convertir a Expr exacto
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolyReprKind {
    Exact,
    ModP,
}
```

### `PolyStore`
```rust
#[derive(Debug, Default)]
pub struct PolyStore {
    polys: Vec<(PolyMeta, PolyRepr)>,
}

impl PolyStore {
    pub fn new() -> Self;
    pub fn insert(&mut self, meta: PolyMeta, repr: PolyRepr) -> PolyId;
    pub fn get(&self, id: PolyId) -> Option<(&PolyMeta, &PolyRepr)>;
    pub fn meta(&self, id: PolyId) -> Option<&PolyMeta>;
    
    // Operaciones que producen nuevos polinomios
    pub fn add(&mut self, a: PolyId, b: PolyId) -> PolyId;
    pub fn sub(&mut self, a: PolyId, b: PolyId) -> PolyId;
    pub fn mul(&mut self, a: PolyId, b: PolyId) -> PolyId;
    pub fn pow(&mut self, a: PolyId, n: u32) -> PolyId;
    pub fn neg(&mut self, a: PolyId) -> PolyId;
    
    // Limpieza
    pub fn clear(&mut self);
    pub fn len(&self) -> usize;
}
```

---

## 4. Flujo de Evaluación

### 4.1 `expand()` con Umbral

```rust
// En ExpandRule o eager_eval_expand
if let Some(est) = estimate_expand_terms(ctx, arg) {
    if est > POLY_REF_THRESHOLD {  // e.g., 500
        // Convertir a poly, almacenar, devolver ref
        let poly = expr_to_poly(ctx, arg, &budget)?;
        let id = session.poly_store.insert(meta, poly);
        return Some(make_poly_ref(ctx, id));
    }
}
// else: expansión AST normal
```

### 4.2 `poly_lower_pass()` (Antes del Pipeline)

Este pase **colapsa** operaciones entre `poly_ref`:

```rust
pub fn poly_lower_pass(
    ctx: &mut Context,
    store: &mut PolyStore,
    expr: ExprId,
) -> ExprId {
    match ctx.get(expr).clone() {
        // poly_ref(a) + poly_ref(b) → poly_ref(c)
        Expr::Add(l, r) => {
            let nl = poly_lower_pass(ctx, store, l);
            let nr = poly_lower_pass(ctx, store, r);
            
            if let (Some(id_l), Some(id_r)) = (extract_poly_ref(ctx, nl),
                                                extract_poly_ref(ctx, nr)) {
                let new_id = store.add(id_l, id_r);
                return make_poly_ref(ctx, new_id);
            }
            // Si solo uno es poly_ref y el otro es poly-like, convertir y sumar
            // ...
            ctx.add(Expr::Add(nl, nr))
        }
        // Análogo para Sub, Mul, Pow
        _ => /* recursión estándar */
    }
}
```

### 4.3 Pipeline Completo

```
1. Parse
2. eager_eval_expand_calls  ← expand() grandes → poly_ref
3. eager_eval_poly_gcd_calls
4. poly_lower_pass          ← combina poly_ref + poly_ref
5. Simplifier Pipeline      ← ve poly_ref como átomo
6. Output (strip_holds, etc.)
```

---

## 5. Conversiones

### `expr_to_poly()`

```rust
/// Convierte expresión poly-like a PolyRepr
/// Acepta: Add, Sub, Mul, Neg, Pow(int), Number, Variable
pub fn expr_to_poly_exact(
    ctx: &Context,
    expr: ExprId,
    budget: &PolyBudget,
    vars: &mut VarTable,
) -> Result<MultiPolyExact, PolyConvError>
```

### `poly_to_expr()`

```rust
/// Materializa polinomio a Expr (bajo demanda)
pub fn poly_to_expr(
    ctx: &mut Context,
    store: &PolyStore,
    id: PolyId,
    max_terms: usize,
) -> Result<ExprId, PolyMaterializeError>

#[derive(Debug)]
pub enum PolyMaterializeError {
    NotFound(PolyId),
    TooManyTerms { actual: usize, limit: usize },
    NotMaterializable,  // ModP sin exacto disponible
}
```

---

## 6. API de Usuario

### Funciones REPL

| Función | Descripción |
|---------|-------------|
| `expand(expr)` | Devuelve AST o `poly_ref(id)` según tamaño |
| `poly_stats(poly_ref(id))` | Muestra metadata: términos, grado, vars |
| `poly_to_expr(poly_ref(id))` | Materializa (con límite configurable) |
| `poly_mul_modp(a, b [,p])` | Multiplicación mod-p, devuelve `poly_ref` |

### Output JSON

```json
{
  "kind": "poly_ref",
  "id": 12,
  "meta": {
    "terms": 3432,
    "degree": 7,
    "vars": ["x1", "x2", "x3", "x4", "x5", "x6", "x7"],
    "repr": "exact",
    "materializable": true
  }
}
```

### Display

```
poly_ref(12)  # Compacto
⟦poly#12: 3432 terms, deg 7⟧  # Descriptivo (opcional)
```

---

## 7. Exact vs ModP

### Estrategia Híbrida (Recomendada)

| Tamaño | Representación | Materializable |
|--------|----------------|----------------|
| < 500 términos | AST directo | N/A |
| 500 - 100k términos | `Exact` | ✅ |
| > 100k términos | `ModP` | ❌ (solo operaciones) |

### `MultiPolyExact`

```rust
/// Polinomio con coeficientes exactos
pub struct MultiPolyExact {
    terms: FxHashMap<Monomial, BigRational>,
    var_table: VarTable,
}
```

Para coeficientes que caben en i128, se usa i128; si overflow → BigRational.

---

## 8. Persistencia (Snapshot)

Al serializar `SessionSnapshot`:

```rust
struct SessionSnapshot {
    store: SessionStore,
    env: Environment,
    // NUEVO:
    poly_store: PolyStoreSnapshot,
}

struct PolyStoreSnapshot {
    polys: Vec<(PolyMeta, PolyReprSnapshot)>,
}

enum PolyReprSnapshot {
    Exact(Vec<(Vec<u16>, String)>),  // (exponentes, coef as string)
    ModP(Vec<(Vec<u16>, u64)>, u64), // (exponentes, coef), modulus
}
```

---

## 9. Invariantes Críticos

1. **`poly_ref` es átomo**: AddView/MulView NUNCA atraviesan
2. **VarTable unificación**: Al operar dos polys, unificar VarTables primero
3. **ID estabilidad**: IDs dentro de una sesión son estables; entre sesiones se regeneran al cargar snapshot
4. **Materialización opcional**: No fallar si no se puede materializar; informar al usuario

---

## 10. Impacto en Rendimiento

| Caso | Antes | Después |
|------|-------|---------|
| `expand(P^7)` | 0.1s | 0.1s (sin cambio) |
| `expand(P^7) + expand(Q^7)` | 59s | **~0.1s** |
| `expand(A^7 * B^7)` (11.8M términos) | timeout | **~3s** (como poly_ref) |

---

## 11. Plan de Implementación

### Fase 1: Infraestructura (2-3 PRs)
- [ ] `MultiPolyExact` con coefs exactos
- [ ] `PolyStore` con operaciones básicas (add, mul, pow)
- [ ] `VarTable::unify` para combinar polys
- [ ] `poly_ref(id)` AST wrapper
- [ ] Tests: roundtrip expr → poly → expr

### Fase 2: Integración expand (1 PR)
- [ ] `expand()` devuelve `poly_ref` cuando > umbral
- [ ] `poly_stats()` regla
- [ ] `poly_to_expr()` regla con límite

### Fase 3: poly_lower_pass (1 PR)
- [ ] Pase que combina `poly_ref + poly_ref`
- [ ] Integración en orchestrator antes del pipeline

### Fase 4: Pulido (1 PR)
- [ ] JSON envelope con metadata
- [ ] Display descriptivo
- [ ] Snapshot serialización
- [ ] Documentación usuario

---

## Referencias

- [POLY_EXPAND_PERFORMANCE.md](POLY_EXPAND_PERFORMANCE.md) - Análisis del problema
- [FAST_EXPAND.md](FAST_EXPAND.md) - Multinomial expansion algorithm
- [ZIPPEL_GCD.md](ZIPPEL_GCD.md) - GCD mod-p infrastructure
