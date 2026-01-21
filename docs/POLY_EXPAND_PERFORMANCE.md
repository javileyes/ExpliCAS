# Polynomial Expansion Performance Analysis

## Executive Summary

La expansión de polinomios grandes presenta un cuello de botella de rendimiento significativo. Este documento analiza el problema, identifica las causas raíz, y propone soluciones a corto y largo plazo.

| Escenario | Tiempo Actual | Tiempo Objetivo |
|-----------|---------------|-----------------|
| `expand(P^7)` solo | 0.1s ✅ | 0.1s |
| `expand(P^7) + expand(Q^7)` | **59s ❌** | <1s |
| `poly_mul_modp(P^7+Q^7, 1)` | 0.015s ✅ | - |

---

## 1. Arquitectura Actual

### 1.1 Flujo de Evaluación

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT                                    │
│   expand((1+x+y)^7) + expand((1-x+y)^7)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PRE-PASS: Eager Eval                           │
│                                                                 │
│   1. Detecta expand((1+x+y)^7) → usa expand_modp_safe()        │
│      - Convierte a MultiPolyModP (rápido: O(output_terms))     │
│      - Reconstruye AST con multipoly_modp_to_expr()            │
│      - Devuelve: __hold(expanded_poly_1)                       │
│                                                                 │
│   2. Detecta expand((1-x+y)^7) → usa expand_modp_safe()        │
│      - Mismo proceso → __hold(expanded_poly_2)                 │
│                                                                 │
│   Resultado: Add(__hold(poly_1), __hold(poly_2))               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  SIMPLIFIER PIPELINE                            │
│                                                                 │
│   Core → Transform → Rationalize → PostCleanup                  │
│                                                                 │
│   El simplificador recibe: Add(__hold(~1000 términos),          │
│                                __hold(~1000 términos))          │
│                                                                 │
│   PROBLEMA: El simplificador aún procesa los children          │
│   aunque estén en __hold, causando O(n²) operaciones            │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 El Mecanismo `__hold`

`__hold(expr)` es un wrapper interno que debería:

1. ✅ Bloquear reglas expansivas (autoexpand, distribute)
2. ✅ Ser transparente a AddView/MulView (para cancelación)
3. ❌ **Problema**: NO bloquea el traversal del simplificador

```rust
// cas_ast/src/hold.rs - Contrato
// 1. __hold blocks: autoexpand, distribute, factor-undo rules
// 2. __hold is transparent to: AddView, MulView, basic arithmetic  ← PROBLEMA
// 3. __hold MUST be stripped before user-facing output
```

El punto 2 causa que cuando tienes `__hold(A) + __hold(B)`, el AddView
"ve a través" de los holds para extraer términos, y esto dispara un
traversal completo de ambos subárboles.

---

## 2. Análisis del Cuello de Botella

### 2.1 Mediciones

```bash
# Caso 1: Un solo expand (RÁPIDO)
expand((1+3*x1+...+15*x7)^7)
# Tiempo: 0.10s
# Output: ~98KB

# Caso 2: Dos expands sumados (LENTO)  
expand(A^7) + expand(B^7)
# Tiempo: 59s (590x más lento)
# Output: ~197KB

# Caso 3: Equivalente en mod-p (INSTANTÁNEO)
poly_mul_modp(A^7, 1) + poly_mul_modp(B^7, 1)
# Tiempo: 0.015s
# Output: 2·poly_result(3432, 7, 7, p)
```

### 2.2 Dónde se Gasta el Tiempo

| Fase | Tiempo (caso 2) | Descripción |
|------|-----------------|-------------|
| Parse | ~1ms | Negligible |
| Eager Eval (expand #1) | ~50ms | expr_to_poly + multipoly_to_expr |
| Eager Eval (expand #2) | ~50ms | expr_to_poly + multipoly_to_expr |
| **Simplifier Pipeline** | **~58.9s** | Procesa Add(hold, hold) |
| Output | ~5ms | Serialización |

El **99% del tiempo** se gasta en el simplificador post-eager-eval.

### 2.3 Por Qué el Simplificador es Lento

Cuando el simplificador ve `Add(__hold(poly_1), __hold(poly_2))`:

1. **AddView extrae términos**: Llama a `unwrap_hold()` en cada child
2. **Flatten recursivo**: Recorre TODOS los ~3432 términos de cada poly
3. **Comparaciones O(n²)**: Busca términos que cancelan entre poly_1 y poly_2
4. **Ciclos múltiples**: El pipeline tiene 4 fases, cada una re-procesa

```
Término count: 3432 + 3432 = 6864 términos
Comparaciones potenciales: O(6864²) = ~47 millones
```

---

## 3. Causa Raíz

El contrato de `__hold` tiene una **contradicción inherente**:

> "__hold is transparent to AddView/MulView for cancellation"

Esta transparencia es necesaria para casos como:
```
__hold(x + 1) - 1  →  x  (cancela el 1)
```

Pero causa problemas catastróficos cuando:
```
__hold(3432 términos) + __hold(3432 términos)
```

El simplificador no sabe que estos polinomios son "opacos" y trata de
encontrar cancelaciones término a término.

---

## 4. Soluciones Propuestas

### 4.1 Solución Superficial: `poly_expand()` (Corto plazo)

Crear una función que expanda **toda** una expresión en mod-p sin
materializar ASTs intermedios:

```
poly_expand(a^7 + b^7)  →  __hold(resultado_combinado)
```

**Pros:**
- Implementación simple (~50 líneas)
- No rompe nada existente
- Rendimiento óptimo para el caso de uso

**Contras:**
- El usuario debe aprender una nueva función
- No resuelve el problema arquitectónico

**Implementación:**
```rust
// expand.rs
pub fn poly_expand_combined(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    // 1. Convertir TODA la expresión a MultiPolyModP (sin expandir a AST)
    let poly = expr_to_poly_modp(ctx, expr, p, &budget, &mut vars).ok()?;
    
    // 2. Solo al final, convertir a AST
    let expanded = multipoly_modp_to_expr(ctx, &poly, &vars);
    
    // 3. Devolver con __hold
    Some(ctx.add(Expr::Function("__hold".to_string(), vec![expanded])))
}
```

### 4.2 Solución Media: Fusión de Holds en Eager Eval

Modificar `eager_eval_expand_recursive` para detectar patrones
`Add/Sub(__hold(poly), __hold(poly))` y fusionarlos en mod-p:

```rust
fn eager_eval_expand_recursive(...) -> ExprId {
    // Primero, procesar children recursivamente
    let result = match ctx.get(expr).clone() {
        Expr::Add(l, r) => {
            let nl = eager_eval_expand_recursive(ctx, l, steps);
            let nr = eager_eval_expand_recursive(ctx, r, steps);
            
            // NUEVO: Si ambos son __hold(poly), fusionar en mod-p
            if let (Some(poly_l), Some(poly_r)) = (extract_hold_poly(ctx, nl), 
                                                    extract_hold_poly(ctx, nr)) {
                let combined = poly_l.add(&poly_r);
                let expr = multipoly_modp_to_expr(ctx, &combined, &vars);
                return ctx.add(Expr::Function("__hold".to_string(), vec![expr]));
            }
            
            ctx.add(Expr::Add(nl, nr))
        }
        // ... resto igual
    };
    result
}
```

**Pros:**
- Transparente al usuario (expand(a)+expand(b) es rápido automáticamente)
- No cambia la API

**Contras:**
- Complejidad moderada
- Requiere mantener VarTable consistente entre conversiones

### 4.3 Solución Profunda: __hold Opaco (Largo plazo)

Rediseñar el contrato de `__hold` con dos niveles:

1. **`__hold(expr)`**: Actual, transparente al AddView
2. **`__opaque(expr)`**: Nuevo, completamente opaco al simplificador

```rust
// Nuevo en cas_ast/src/hold.rs
pub fn is_opaque(ctx: &Context, id: ExprId) -> bool {
    matches!(ctx.get(id), Expr::Function(name, _) if name == "__opaque")
}

// En AddView/MulView:
fn collect_terms(...) {
    if is_opaque(ctx, id) {
        // NO atravesar, tratar como átomo
        return vec![(id, Sign::Pos)];
    }
    // ... lógica actual
}
```

Luego, `expand_modp_safe` devolvería `__opaque(...)` en lugar de `__hold(...)`.

**Pros:**
- Solución arquitectónicamente correcta
- Previene todos los problemas de traversal
- Extensible a otros casos (factor, gcd, etc.)

**Contras:**
- Cambio invasivo (afecta muchos módulos)
- Riesgo de regresiones
- Requiere auditoría de todos los lugares que usan __hold

### 4.4 Solución Alternativa: Lazy AST Construction

En lugar de materializar el AST en `multipoly_modp_to_expr`, devolver
un `poly_ref(id)` opaco que solo se materializa cuando el usuario
lo solicita explícitamente:

```
expand((1+x)^10)  →  poly_ref(42)  # Instantáneo
poly_to_expr(poly_ref(42))  →  1 + 10x + 45x² + ...  # Bajo demanda
```

Esto ya está parcialmente implementado con `PolyStore` y `poly_mul_modp`.

---

## 5. Recomendación

### Fase 1 (Inmediata): Documentación + Workaround

1. Documentar que `expand(a) + expand(b)` es lento
2. Recomendar `poly_mul_modp(a + b, 1)` como alternativa rápida

### Fase 2 (Corto plazo, 1-2 días): `poly_expand()`

Implementar función dedicada:
```
poly_expand(a^7 + b^7)  →  resultado expandido en un solo paso
```

### Fase 3 (Medio plazo, 1 semana): Fusión en Eager Eval

Modificar eager eval para detectar y fusionar `__hold(poly) + __hold(poly)`
automáticamente.

### Fase 4 (Largo plazo, revisión arquitectónica): __opaque

Considerar para la próxima versión mayor del motor.

---

## 6. Apéndice: Código Relevante

### Ubicación de archivos clave:

| Archivo | Propósito |
|---------|-----------|
| `expand.rs::eager_eval_expand_calls` | Pre-procesamiento de expand() |
| `expand.rs::expand_modp_safe` | Expansión vía mod-p |
| `gcd_modp.rs::multipoly_modp_to_expr` | Reconstrucción AST desde mod-p |
| `cas_ast/src/hold.rs` | Contrato de __hold |
| `nary.rs::add_terms_signed` | AddView que atraviesa holds |
| `orchestrator.rs::simplify_pipeline` | Pipeline principal |

### Constantes configurables:

```rust
// expand.rs
pub const EAGER_EXPAND_MODP_THRESHOLD: u64 = 500;

// distribution.rs
pub const EXPAND_MAX_MATERIALIZE_TERMS: u64 = 200_000;
pub const EXPAND_MODP_THRESHOLD: u64 = 1_000;

// poly_store.rs
pub const POLY_MAX_STORE_TERMS: usize = 10_000_000;
```

---

## 7. Conclusión

El problema fundamental es que el diseño actual de `__hold` intenta
ser simultáneamente:

1. **Protector**: Evitar reglas expansivas
2. **Transparente**: Permitir cancelaciones simples

Estos objetivos son incompatibles cuando el contenido protegido tiene
miles de términos. La solución correcta es una separación explícita
entre "protección contra reglas" y "opacidad al traversal".

Mientras tanto, las soluciones a corto plazo (`poly_expand`, fusión
en eager eval) pueden proporcionar alivio significativo sin cambios
arquitectónicos mayores.
