# Poly-Ref Architecture: Opaque Polynomial Domain

## Status: ✅ IMPLEMENTED (Phase 1-3 Complete)

Este documento define la arquitectura para **representación opaca de polinomios** (`poly_result`) implementada en ExpliCAS.

---

## 1. Problema Resuelto

### Situación Anterior (Problemática)

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
   Simplifier procesa 6864 términos → 59 segundos ❌
```

### Situación Actual (Implementada)

```
expand(A^7) + expand(B^7)
    │              │
    ▼              ▼
poly_result(0)  poly_result(1)  ← Referencias opacas
    │              │
    └──────┬───────┘
           ▼
    poly_lower_pass (VarTable unify + remap)
           │
           ▼
      poly_result(2)              ← Suma interna en PolyStore
           │
           ▼
   Simplifier ve solo 1 átomo → 0.22 segundos ✅
```

**Speedup: 270x (59s → 0.22s)**

---

## 2. Arquitectura de Dos Mundos

### Mundo AST
- `Expr` binario: `Add/Mul/Pow/...`
- Donde vive el simplificador, reglas, display
- Coste O(n) traversal por operación

### Mundo Polynomial
- `MultiPolyModP`: HashMap `Monomial → Coeff`
- Operaciones O(n_términos) para add, O(n·m) para mul
- Sin traversal recursivo
- Vive en `PolyStore` (thread-local)

### Frontera
```
┌─────────────────────────────────────────────────────────────────┐
│                        MUNDO AST                                │
│   Expr::Add, Expr::Mul, Expr::Pow, ...                         │
│   poly_result(id) ← ÁTOMO OPACO                                │
└─────────────────────────────────────────────────────────────────┘
                    ▲                          │
                    │ poly_to_expr(id)         │ expand() interno
                    │ (materializa)            │ (convierte)
                    │                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MUNDO POLYNOMIAL                            │
│   Thread-Local PolyStore { polys: Vec<(PolyMeta, MultiPolyModP)>│
│   VarTable unification + monomial remapping                     │
│   Operaciones: add, sub, mul, pow, neg                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. API de Usuario

### Funciones REPL

| Función | Descripción | Ejemplo |
|---------|-------------|---------|
| `expand(expr)` | Devuelve AST o `poly_result(id)` según tamaño | `expand((1+x+y)^7)` |
| `poly_stats(poly_result(id))` | Muestra `poly_info(id, terms, vars, repr)` | `poly_stats(poly_result(0))` |
| `poly_to_expr(poly_result(id) [, limit])` | Materializa a AST (lento, crea nodos) | `poly_to_expr(poly_result(0), 50000)` |
| `poly_print(poly_result(id) [, limit])` | **Impresión directa sin AST (rápido)** | `poly_print(poly_result(0), 50)` |
| `poly_latex(poly_result(id) [, limit])` | Formato LaTeX sin AST | `poly_latex(poly_result(0), 10)` |

### Comparativa de Tiempos (3432 términos)

| Función | Tiempo | Método |
|---------|--------|--------|
| `poly_to_expr(p)` | ~30s | Crea AST completo |
| `poly_print(p)` | **0.26s** | String directo, grlex sort |
| `poly_latex(p)` | **0.26s** | LaTeX string directo |

### Ejemplo Completo

```
> a := (1 + 3*x1 + 5*x2 + 7*x3 + 9*x4 + 11*x5 + 13*x6 + 15*x7)^7 - 1
> b := (1 - 3*x1 - 5*x2 - 7*x3 + 9*x4 - 11*x5 - 13*x6 + 15*x7)^7 + 1

> expand(a) + expand(b)
poly_result(2)

> poly_stats(poly_result(2))
poly_info(2, 1716, 7, modp)

> poly_to_expr(poly_result(2))
# Materializa los 1716 términos como AST
```

### Umbral de Materialización

| Términos Estimados | Comportamiento |
|--------------------| --------------- |
| ≤ 1000 | Materializa AST directamente |
| > 1000 | Devuelve `poly_result(id)` opaco |

---

## 4. Flujo de Evaluación

```
┌─────────────────────────────────────────────────────────────────┐
│                         Orchestrator                             │
├─────────────────────────────────────────────────────────────────┤
│  1. clear_thread_local_store()    ← Limpia store por evaluación │
│  2. eager_eval_expand_calls()     ← expand() grandes → poly_result│
│  3. eager_eval_poly_gcd_calls()                                  │
│  4. poly_lower_pass()             ← Combina poly_results con    │
│                                      VarTable unify + remap     │
│  5. Simplifier Pipeline           ← Ve poly_result como átomo   │
│  6. Output (strip_holds, etc.)                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Estructuras de Datos

### `PolyId`
```rust
pub type PolyId = usize;
```

### `PolyMeta`
```rust
pub struct PolyMeta {
    pub modulus: u64,
    pub n_terms: usize,
    pub n_vars: usize,
    pub max_total_degree: u32,
    pub var_names: Vec<String>,
}
```

### `VarTable` (con unificación Phase 3)
```rust
impl VarTable {
    /// Obtiene o asigna índice para una variable
    pub fn get_or_insert(&mut self, name: &str) -> Option<usize>;
    
    /// Obtiene índice de variable existente
    pub fn get_index(&self, name: &str) -> Option<usize>;
    
    /// Unifica dos VarTables → (unified, remap_a, remap_b)
    /// Variables ordenadas lexicográficamente para orden canónico
    pub fn unify(&self, other: &VarTable) -> Option<(VarTable, Vec<usize>, Vec<usize>)>;
}
```

### `MultiPolyModP::remap()`
```rust
impl MultiPolyModP {
    /// Remapea índices de variable según vector de remapeo
    /// Usado para alinear monomios al espacio de variables unificado
    pub fn remap(&self, remap: &[usize], new_num_vars: usize) -> Self;
}
```

### `PolyStore`
```rust
impl PolyStore {
    pub fn insert(&mut self, meta: PolyMeta, poly: MultiPolyModP) -> PolyId;
    pub fn get(&self, id: PolyId) -> Option<(&PolyMeta, &MultiPolyModP)>;
    
    // Operaciones con VarTable unification automática
    pub fn add(&mut self, a: PolyId, b: PolyId) -> Option<PolyId>;
    pub fn sub(&mut self, a: PolyId, b: PolyId) -> Option<PolyId>;
    pub fn mul(&mut self, a: PolyId, b: PolyId) -> Option<PolyId>;
    pub fn neg(&mut self, a: PolyId) -> Option<PolyId>;
    pub fn pow(&mut self, a: PolyId, n: u32) -> Option<PolyId>;
    
    pub fn clear(&mut self);
}
```

---

## 6. Invariantes Críticos

1. **`poly_result` es átomo**: AddView/MulView NUNCA atraviesan estos nodos
2. **VarTable unificación automática**: Al operar dos polys, se unifican VarTables y remapean monomios
3. **Thread-local store**: Cada evaluación tiene su propio PolyStore limpio
4. **Materialización controlada**: `poly_to_expr()` con límite opcional

---

## 7. Rendimiento Medido

| Caso | Antes | Después | Speedup |
|------|-------|---------|---------|
| `expand(P^7)` | 0.1s | 0.1s | - |
| `expand(P^7) + expand(Q^7)` | 59s | **0.22s** | **270x** |
| Combinación con VarTable diff | N/A (fallaba) | **0.22s** | ✅ |

---

## 8. Estado de Implementación

### Fase 1: Surgical Fix ✅
- [x] `poly_result` y `poly_ref` son atómicos en AddView/MulView
- [x] Previene traversal O(n²)

### Fase 2: Thread-Local PolyStore ✅
- [x] `PolyStore` con operaciones add/sub/mul/neg/pow
- [x] `poly_stats()` para metadata sin materializar
- [x] `poly_to_expr()` para materialización controlada
- [x] Integración en orchestrator (clear antes de cada eval)

### Fase 2.5: Hardening ✅
- [x] Formato `poly_info(id, terms, vars, repr)` claro
- [x] Warnings para combinaciones no realizadas

### Fase 3: VarTable Unification ✅
- [x] `VarTable::unify()` con ordenación canónica lexicográfica
- [x] `MultiPolyModP::remap()` para permutación de exponentes
- [x] Operaciones PolyStore usan unificación automática
- [x] Fast path cuando var_names ya coinciden

---

## 9. Archivos Clave

| Archivo | Contenido |
|---------|-----------|
| `poly_store.rs` | PolyStore, thread-local ops, PolyMeta |
| `poly_modp_conv.rs` | VarTable con unify(), expr↔poly conversion |
| `multipoly_modp.rs` | MultiPolyModP con remap() |
| `poly_lowering.rs` | Pre-pass que combina poly_results |
| `rules/algebra/poly_stats.rs` | poly_stats() y poly_to_expr() rules |
| `nary.rs` | is_poly_ref() para atomicidad |
| `expand.rs` | expand_to_poly_ref_or_hold() |
| `orchestrator.rs` | Integración del pipeline |

---

## Referencias

- [POLY_EXPAND_PERFORMANCE.md](POLY_EXPAND_PERFORMANCE.md) - Análisis del problema original
- [FAST_EXPAND.md](FAST_EXPAND.md) - Multinomial expansion algorithm
- [ZIPPEL_GCD.md](ZIPPEL_GCD.md) - GCD mod-p infrastructure
