## (Fase 1: Best-so-far)

### Objetivo

Evitar que `simplify()` devuelva una expresión **peor** que la de entrada cuando, durante el proceso, se aplican reglas expansivas (distribute, common denominator, racionalizaciones, etc.) que **no llegan a “cerrarse”** con cancelaciones posteriores.

### Idea

En cada iteración del loop de simplificación se calcula un `Score` basado principalmente en vuestra `Complexity`. Se mantiene un registro del **mejor estado visto** hasta el momento:

* `best_expr`
* `best_steps` (traza)
* `best_requires`
* `best_score`

Al final, **se devuelve el best**, no necesariamente el último estado alcanzado por el loop greedy.

### Propiedad clave

Si inicializamos `best = input`, entonces:

* Si el engine no encuentra nada mejor, devuelve el input (**no empeora nunca**).
* Si encuentra mejoras parciales y luego “se lía” expandiendo, el resultado final vuelve al mejor punto (“rollback”).

### Alcance de la Fase 1

* **No** hay backtracking real ni beam search todavía.
* **No** se cambia el conjunto de reglas ni su orden.
* Solo se añaden:

  1. un `Score` comparable,
  2. un `BestSoFarTracker`,
  3. un pequeño hook “después de cada rewrite exitosa”.

### Criterios de aceptación

1. `Complexity(result) <= Complexity(input)` siempre (si `best` arranca en input).
2. Para casos como el que mostraste (distribute/common denominator que empeoran), el resultado final se queda en la mejor forma encontrada (a menudo el input si no hubo mejora real).
3. No rompe CI / metatests. (De hecho suele bajar “numeric-only” porque evita finales peores.)

### Trade-offs (asumidos y aceptables en Fase 1)

* Podéis “perder” una forma **más canónica** que tenga igual complejidad que otra: se resuelve con un tie-breaker determinista (ver abajo).
* Copiar `Vec<Step>` al actualizar el best puede costar algo, pero suele ser bajo (y se puede optimizar en Fase 2 con trazas persistentes).

---

## 2) Bocetos de implementación (Rust)

### 2.1. `score.rs` (Score basado en vuestra Complexity)

```rust
// crates/cas_engine/src/simplify/score.rs

use crate::expr::Expr;
use crate::requires::RequiresSet;
use crate::complexity::Complexity;

#[derive(Clone, Debug)]
pub struct Score {
    pub complexity: i64,   // principal
    pub nodes: i64,        // tie-breaker
    pub requires: i64,     // tie-breaker (opcional)
}

impl Score {
    #[inline]
    pub fn total_key(&self) -> (i64, i64, i64) {
        // orden lexicográfico: menor es mejor
        (self.complexity, self.nodes, self.requires)
    }
}

pub fn score_expr(expr: &Expr, req: &RequiresSet) -> Score {
    let c = Complexity::measure(expr).value() as i64; // adapta a tu API
    let n = expr.node_count() as i64;                 // si no existe, implementa simple walk
    let r = req.len() as i64;
    Score { complexity: c, nodes: n, requires: r }
}
```

> Nota: `node_count()` puede ser un helper no intrusivo (un visitor que cuente nodos).

---

### 2.2. `best_so_far.rs` (tracker mínimo)

```rust
// crates/cas_engine/src/simplify/best_so_far.rs

use crate::expr::Expr;
use crate::requires::RequiresSet;
use crate::steps::Step;

use super::score::{Score, score_expr};

#[derive(Clone)]
pub struct BestSoFar {
    best_expr: Expr,
    best_score: Score,
    best_steps: Vec<Step>,
    best_requires: RequiresSet,
}

impl BestSoFar {
    pub fn new(initial_expr: &Expr, initial_steps: &[Step], initial_req: &RequiresSet) -> Self {
        let s = score_expr(initial_expr, initial_req);
        Self {
            best_expr: initial_expr.clone(),
            best_score: s,
            best_steps: initial_steps.to_vec(),
            best_requires: initial_req.clone(),
        }
    }

    #[inline]
    pub fn consider(&mut self, expr: &Expr, steps: &[Step], req: &RequiresSet) {
        let s = score_expr(expr, req);
        if s.total_key() < self.best_score.total_key() {
            self.best_expr = expr.clone();
            self.best_score = s;
            self.best_steps = steps.to_vec();     // fase 1: simple clone
            self.best_requires = req.clone();
        }
    }

    pub fn into_parts(self) -> (Expr, Vec<Step>, RequiresSet) {
        (self.best_expr, self.best_steps, self.best_requires)
    }
}
```

---

### 2.3. Hook en el loop actual (cambio mínimo)

Imaginemos que ahora tenéis algo como:

* `simplify(expr) -> SimplifyResult { expr, steps, requires }`
* un loop que aplica reglas hasta fixpoint / budget.

**El único cambio:** crear `BestSoFar` al inicio y llamar `best.consider(...)` tras cada rewrite exitosa.

```rust
// crates/cas_engine/src/simplify/mod.rs (o donde esté vuestro simplify)

use crate::expr::Expr;
use crate::requires::RequiresSet;
use crate::steps::Step;
use crate::rules::RuleRegistry;

mod best_so_far;
mod score;

use best_so_far::BestSoFar;

pub struct SimplifyResult {
    pub expr: Expr,
    pub steps: Vec<Step>,
    pub requires: RequiresSet,
}

pub fn simplify(expr: Expr, reg: &RuleRegistry, ctx: &mut SimplifyContext) -> SimplifyResult {
    let mut cur_expr = expr;
    let mut steps: Vec<Step> = vec![];
    let mut req = RequiresSet::default();

    // 1) best arranca en el estado inicial (garantía no empeorar)
    let mut best = BestSoFar::new(&cur_expr, &steps, &req);

    // 2) loop actual (sin cambiar reglas)
    let mut changed = true;
    let mut budget = ctx.budget_steps;

    while changed && budget > 0 {
        changed = false;

        for (rule, meta) in reg.iter_all() {
            if budget == 0 { break; }

            // tu API real puede variar:
            if let Some(apply_out) = rule.try_apply(&cur_expr, ctx) {
                // apply_out: (new_expr, step, req_delta) o similar
                let (new_expr, step, req_delta) = apply_out.into_parts();

                // aplicar al estado actual
                cur_expr = new_expr;
                steps.push(step);
                req.extend(req_delta);

                // ✅ hook best-so-far
                best.consider(&cur_expr, &steps, &req);

                changed = true;
                budget -= 1;

                // si vuestro loop reinicia iteración tras una aplicación:
                // break;
            }
        }
    }

    // 3) devolver best, no cur
    let (best_expr, best_steps, best_req) = best.into_parts();
    SimplifyResult { expr: best_expr, steps: best_steps, requires: best_req }
}
```

**Esto por sí solo** ya evita el caso que muestras: si la secuencia `Distribute -> CommonDen` sube complejidad y no consigue cerrar, el best se queda en el último punto “bueno” (o en el input si nunca mejoró).

---

## 3) Tests rápidos (regresión del comportamiento “no empeorar”)

### 3.1. Test: `complexity(result) <= complexity(input)`

```rust
// crates/cas_engine/tests/best_so_far_tests.rs

use cas_engine::{parse::parse_expr, simplify::simplify, complexity::Complexity};

#[test]
fn simplify_never_worsens_by_complexity() {
    let e = parse_expr("(x^4 - 16) + (sqrt(u)/sqrt(b)) - (((x^2+4)*(x+2)*(x-2)) + (sqrt(u/b)))").unwrap();

    let before = Complexity::measure(&e).value();
    let r = simplify(e.clone(), /*reg*/ &default_registry(), /*ctx*/ &mut default_ctx());

    let after = Complexity::measure(&r.expr).value();
    assert!(after <= before, "after={after} before={before} expr={}", r.expr);
}
```

### 3.2. Test: “si no mejora, devuelve el input”

```rust
#[test]
fn simplify_returns_input_if_no_improvement_found() {
    let e = parse_expr("x^4 - 16 + sqrt(u)/sqrt(b) - ((x^2+4)*(x+2)*(x-2) + sqrt(u/b))").unwrap();
    let r = simplify(e.clone(), &default_registry(), &mut default_ctx());

    // Si realmente no hay reglas que cierren el asunto, best se queda en input:
    // Ojo: si alguna regla sí mejora, este test no aplica. Úsalo con una expresión conocida.
    assert_eq!(r.expr, e);
}
```

> En la práctica, el test 3.1 es el más robusto.

---

## 4) Micro-mejora opcional (sin fase 2): tie-break “más canónico”

Si te preocupa “misma Complexity pero forma más fea/larga”, en `Score` ya metimos `nodes` y `requires`. Podéis añadir un tie-break adicional:

* penalización por `Div` anidados
* penalización por `Neg` internos
* o un `canonical_rank(expr)` muy barato.

Esto no toca reglas: solo el `score_expr`.

---

## 5) Checklist de integración (muy corto)

1. Añadir `simplify/score.rs` + `simplify/best_so_far.rs`.
2. Insertar `best.consider(...)` en el loop actual tras cada rewrite.
3. Al final, devolver best en vez del estado final.
4. Añadir test “no empeorar”.

---

Si me pegas el fichero donde está vuestro `simplify()`/loop real (aunque sea parcialmente), te adapto estos bocetos a **vuestros nombres exactos** (`Rule::apply`, `Step`, `RequiresSet`, `Complexity`) para que sea casi “copiar/pegar”.





Perfecto — aquí va **la reescritura del plan “Best-so-far”** pero ajustado para que, **siempre dentro de un margen controlado**, el simplificador *prefiera* expresiones “mejor racionalizadas” (p. ej. menos √ en denominadores / menos divisiones anidadas), aunque cuesten unos pocos nodos más.

---

## Best-So-Far Guard con Preferencia por Racionalización (con margen)

### Objetivo

Evitar que el simplificador devuelva una expresión peor *porque abrió* (distribute / common denominator / rationalize) y no cerró con cancelaciones… **pero permitiendo** que, dentro de un **presupuesto de “empeoramiento”** pequeño, el resultado final pueda ser **más “limpio”** (racionalizado) aunque tenga *algo* más de tamaño.

Ejemplo típico: aceptar una forma con +6 nodos si elimina `sqrt(...)` en denominadores o reduce `Div` anidadas.

---

## Principio de Seguridad (Hard Guard)

Se introduce un **budget** (margen máximo) relativo al input:

* `nodes(candidate) <= nodes(input) + max_extra_nodes`
* (opcional) `complexity(candidate) <= complexity(input) + max_extra_complexity`

Si el candidato **excede el budget**, se descarta siempre.

Esto mantiene el sistema estable (no se descontrola el tamaño) y te permite “premiar” racionalización sin abrir la puerta al caos.

---

## Ranking (Soft Preference) dentro del margen

Una vez filtrados candidatos *admisibles*, el “mejor” se decide por una **score lexicográfica** que prioriza racionalización:

1. **Penalización por irracionalidad en denominadores** (primario)

* `sqrt_in_den_count` (menor es mejor)
* `root_in_den_count` (si lo queréis generalizar a `Pow(_, 1/n)`)

2. **Divisiones anidadas / fracciones feas**

* `nested_div_count` (menor es mejor)
* `has_add_in_den` (bool como tie-breaker)

3. Tamaño / coste

* `nodes` (menor es mejor)
* `complexity` (menor es mejor)

✅ Resultado: **se favorece racionalizar** “hasta cierto margen”, sin permitir explosiones.

---

## Cambios propuestos (Fase 1: solo Best-so-far, sin beam)

### [NEW] `best_so_far.rs`

```rust
pub struct BestSoFarBudget {
    pub max_extra_nodes: usize,        // p.ej. 8 o 12
    pub max_extra_complexity: i64,      // opcional
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Score {
    // --- Preferencias (primario) ---
    pub sqrt_in_den: u16,
    pub nested_div: u16,
    pub add_in_den: bool,

    // --- Coste (secundario) ---
    pub nodes: usize,
    pub complexity: i64,
}

pub fn score_expr(expr: ExprId, ctx: &Context) -> Score {
    Score {
        sqrt_in_den: count_sqrt_in_den(expr, ctx),
        nested_div: count_nested_div(expr, ctx),
        add_in_den: has_add_in_den(expr, ctx),
        nodes: node_count(expr, ctx),
        complexity: complexity(expr, ctx), // vuestra métrica existente
    }
}

pub struct BestSoFar {
    baseline_score: Score,
    budget: BestSoFarBudget,

    best_expr: ExprId,
    best_score: Score,
    best_steps: Vec<Step>,
}

impl BestSoFar {
    pub fn new(input: ExprId, steps: &[Step], ctx: &Context, budget: BestSoFarBudget) -> Self {
        let s = score_expr(input, ctx);
        Self {
            baseline_score: s,
            budget,
            best_expr: input,
            best_score: s,
            best_steps: steps.to_vec(),
        }
    }

    fn admissible(&self, cand: Score) -> bool {
        if cand.nodes > self.baseline_score.nodes + self.budget.max_extra_nodes {
            return false;
        }
        if self.budget.max_extra_complexity != 0
            && cand.complexity > self.baseline_score.complexity + self.budget.max_extra_complexity
        {
            return false;
        }
        true
    }

    pub fn consider(&mut self, cand_expr: ExprId, all_steps: &[Step], ctx: &Context) {
        let cand_score = score_expr(cand_expr, ctx);
        if !self.admissible(cand_score) {
            return;
        }
        if cand_score < self.best_score {
            self.best_expr = cand_expr;
            self.best_score = cand_score;
            self.best_steps = all_steps.to_vec();
        }
    }

    pub fn into_parts(self) -> (ExprId, Vec<Step>) {
        (self.best_expr, self.best_steps)
    }
}
```

**Importante:** implementad `Ord/PartialOrd` para `Score` con el orden:

```rust
// menor es mejor
(sqrt_in_den, nested_div, add_in_den, nodes, complexity)
```

---

### [MODIFY] `orchestrator.rs`

En `simplify_pipeline()`:

* Inicializas `BestSoFar` con `budget`.
* Al final de cada fase llamas `best.consider(current, &steps, ctx)`.
* **Devuelves siempre** el `best` al final (sin “if final”).

```rust
let budget = BestSoFarBudget {
    max_extra_nodes: ctx.settings.best_so_far_max_extra_nodes, // o env var
    max_extra_complexity: 0, // opcional
};

let mut best = BestSoFar::new(expr, &steps, ctx, budget);

phase_core(...);        best.consider(current, &steps, ctx);
phase_transform(...);   best.consider(current, &steps, ctx);
phase_rationalize(...); best.consider(current, &steps, ctx);
phase_cleanup(...);     best.consider(current, &steps, ctx);

let (out_expr, out_steps) = best.into_parts();
return (out_expr, out_steps, stats);
```

---

### [MODIFY] configuración (muy recomendado)

Permitid configurar el margen por env var o settings:

* `BEST_SO_FAR_MAX_EXTRA_NODES=12` (por defecto 0 u 8)
* (opcional) `BEST_SO_FAR_MAX_EXTRA_COMPLEXITY=…`

---

## Verificación / Tests

### 1) Test unitario “no explota”

* `nodes(out) <= nodes(in) + max_extra_nodes` (siempre)

### 2) Test de racionalización “merece la pena”

Caso como el tuyo:

* Input: mezcla de `sqrt(u)/sqrt(b)` + estructuras que tienden a empeorar
* Esperado: si se racionaliza y queda dentro del margen, **se prefiere** la forma racionalizada incluso si sube unos nodos.

### 3) Metamórficos

* Debe mantener (o mejorar) numeric-only.
* Y además deberíais ver menos residuos con `sqrt(...)` en denominadores.

---

## Por qué este enfoque es bueno (y seguro)

* No toca reglas reductoras.
* Evita “me fui por un camino malo y devolví basura”.
* Permite “calidad algebraica” (racionalización) **sin** beam search.
* El margen es explícito y controlado.



---

## 6) Implementación Final (V2.15.26) 

> **Estado: IMPLEMENTADO Y FUNCIONANDO** 

### 6.1 Archivos creados

| Archivo | Propósito |
|---------|-----------|
| `crates/cas_engine/src/best_so_far.rs` | Tracker `BestSoFar` + `Score` + helpers iterativos |
| Modificación en `engine.rs` | Integración en `simplify_pipeline()` |

### 6.2 Score implementado

```rust
pub struct Score {
    pub sqrt_in_den: u16,   // √ en denominadores (menor = mejor)
    pub nested_div: u16,    // Div anidados (menor = mejor)
    pub add_in_den: bool,   // Add en denominador (false = mejor)
    pub nodes: usize,       // Conteo total de nodos (menor = mejor)
}
```

Comparación lexicográfica: `sqrt_in_den > nested_div > add_in_den > nodes`

### 6.3 Parámetros finales

- **Threshold de rollback**: **+12 nodos** (permite expansiones binomiales grado-3)
- **Baseline**: Se establece **después de Phase 1 (Core)**, no en el input raw
- **Helpers**: Todos usan **traversal iterativo** con stack para cumplir Recursion Guard Policy

### 6.4 Hazards resueltos

| Hazard | Síntoma | Solución |
|--------|---------|----------|
| Canonicalization Rollback | Revertía `arcsin` → `asin` | Threshold +12 nodos |
| Command Function Baseline | `factor(x²-9)` bloqueado | Post-Core baseline |
| Functional Reduction | `arcsec(x) → arccos(1/x)` revertido | Threshold +12 nodos |
| Expansion Policy Conflict | `(x+1)³` no se expandía | Threshold +12 nodos |

### 6.5 Ejemplo de uso

```
Input:  (x^4 - 16) + (sqrt(u)/sqrt(b)) - (((x^2+4)*(x+2)*(x-2)) + (sqrt(u/b)))

Sin BSF Guard:  Explota con distribute + common denominator sin cerrar
Con BSF Guard:  Retorna la mejor forma encontrada (usualmente el input si no mejora)

Garantía: nodes(result) <= nodes(post_core_baseline) + 12
```

