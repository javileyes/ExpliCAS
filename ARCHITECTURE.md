# Arquitectura del Motor CAS (ExpliCAS)

## Índice
1. [Visión General](#visión-general)
   - [Objetivos de Diseño](#objetivos-de-diseño)
   - [Principios Arquitectónicos](#principios-arquitectónicos)
2. [Componentes Principales](#componentes-principales)
   - [2.0. `cas_ast` - Abstract Syntax Tree](#20-cas_ast---abstract-syntax-tree)
   - [2.1. `cas_ast` - Automatic Canonical Ordering (conmutativo-only) ★★★](#21-cas_ast---automatic-canonical-ordering-conmutativo-only-)
   - [2.2. Normalización Canónica y Views (C2) ★★★](#22-normalización-canónica-y-views-c2-)
   - [2.5. `cas_engine` - Pattern Detection Infrastructure ★★★](#25-cas_engine---pattern-detection-infrastructure-)
   - [2.6. N-ary Pattern Matching (Add/Mul Flattening) ★★](#26-n-ary-pattern-matching-addmul-flattening-)
   - [2.7. Patrón de Negación Generalizada ★★★](#27-patrón-de-negación-generalizada-)
   - [2.8. `cas_engine` - Semantic Equality + nf_score (aceptación de rewrites) ★★★](#28-cas_engine---semantic-equality--nf_score-aceptación-de-rewrites-)
   - [2. `cas_parser` - Parser de Expresiones](#2-cas_parser---parser-de-expresiones)
3. [Optimizaciones (Phase 1 y Phase 2)](#optimizaciones-phase-1-y-phase-2)
   - [Phase 1: Performance Optimization](#phase-1-performance-optimization)
   - [Phase 2: Debug Tools](#phase-2-debug-tools)
   - [Phase 3: Safety Nets (Correctness Hardening)](#phase-3-safety-nets-correctness-hardening)
     - [1) CI Lint: prohibir `ctx.add(Expr::Mul...)` en `src/`](#1-ci-lint-prohibir-ctxaddexprmul-en-src)
     - [2) Guard tests de canonicidad y trazas](#2-guard-tests-de-canonicidad-y-trazas)
   - [CLI Commands](#cli-commands)
   - [Performance Summary](#performance-summary)
4. [Métricas de Complejidad](#métricas-de-complejidad)
5. [Arquitectura del Sistema de Reglas](#arquitectura-del-sistema-de-reglas)
   - [Categorización de Reglas](#categorización-de-reglas)
   - [Diseño de Reglas Complejas](#diseño-de-reglas-complejas)
6. [Orquestación y Estrategias](#orquestación-y-estrategias)
   - [Problema: Explosión Combinatoria](#problema-explosión-combinatoria)
   - [Solución: Orchestrator + Strategies](#solución-orchestrator--strategies)
   - [Step Optimization](#step-optimization)
7. [Flujo de Datos](#flujo-de-datos)
   - [Diagrama de Flujo Completo](#diagrama-de-flujo-completo)
   - [Ejemplo Concreto: `simplify 2x + 3x`](#ejemplo-concreto-simplify-2x--3x)
8. [Puntos de Extensión](#puntos-de-extensión)
   - [1. Añadir Nueva Regla](#1-añadir-nueva-regla)
   - [2. Añadir Nueva Función Matemática](#2-añadir-nueva-función-matemática)
   - [3. Añadir Estrategia de Simplificación](#3-añadir-estrategia-de-simplificación)
   - [4. Añadir Comando CLI](#4-añadir-comando-cli)
9. [Optimizaciones](#optimizaciones)
   - [1. Expression Interning](#1-expression-interning)
   - [2. Caché de Simplificación](#2-caché-de-simplificación)
   - [3. Reglas Específicas por Tipo](#3-reglas-específicas-por-tipo)
   - [4. Early Return en Reglas](#4-early-return-en-reglas)
10. [Conclusión](#conclusión)
11. [8. Stress Testing Infrastructure ★ (Added 2025-12)](#8-stress-testing-infrastructure--added-2025-12)
12. [Matrix Operations](#matrix-operations)
13. [Number Theory Implementation](#number-theory-implementation)
14. [Configuración del Repo](#configuración-del-repo)
15. [7. Optimizaciones](#7-optimizaciones)
---

## Visión General

ExpliCAS es un **Sistema de Álgebra Computacional (CAS)** educativo diseñado en Rust con énfasis en proporcionar explicaciones paso a paso de las transformaciones matemáticas. La arquitectura está construida siguiendo principios de modularidad y separación de responsabilidades.

### Objetivos de Diseño

- **Modularidad**: Cada componente tiene responsabilidades claras y bien definidas
- **Extensibilidad**: Fácil adición de nuevas reglas y funcionalidades
- **Trazabilidad**: Registro detallado de cada paso de simplificación
- **Rendimiento**: Uso de estructuras de datos eficientes (interning, caching)
- **Educación**: Presentación clara de las transformaciones matemáticas

### Principios Arquitectónicos

1. **Separación AST/Engine**: La representación de expresiones está desacoplada de las reglas de transformación
2. **Rule-Based System**: Las simplificaciones se implementan como reglas independientes y componibles
3. **Immutabilidad**: Las expresiones son inmutables (usando expression interning)
4. **Bottom-Up Simplification**: Simplificación recursiva desde las hojas hacia la raíz

---

## Componentes Principales

### 1. `cas_ast` - Abstract Syntax Tree

**Responsabilidad**: Definir la representación interna de expresiones matemáticas.

#### Estructura de Datos

```rust
pub enum Expr {
    Number(BigRational),         // Números racionales arbitrarios
    Variable(String),            // Variables simbólicas (x, y, z)
    Constant(Constant),          // Constantes (pi, e, infinity)
    Add(ExprId, ExprId),        // Suma
    Sub(ExprId, ExprId),        // Resta
    Mul(ExprId, ExprId),        // Multiplicación
    Div(ExprId, ExprId),        // División
    Pow(ExprId, ExprId),        // Potencia
    Neg(ExprId),                // Negación
    Function(String, Vec<ExprId>) // Funciones (sin, cos, ln, etc.)
}
```

#### Context - Expression Interning

El `Context` implementa **expression interning** para:
- Evitar duplicación de expresiones idénticas en memoria
- Permitir comparaciones rápidas por identidad (`ExprId` es un índice)
- Facilitar el caché de resultados

```rust
pub struct Context {
    exprs: Vec<Expr>,           // Pool central de expresiones
    map: HashMap<Expr, ExprId>, // Deduplicación
}
```

**Ventajas**:
- Memoria: `O(n)` en lugar de `O(n²)` para subexpresiones compartidas
- Comparación: `O(1)` en lugar de `O(n)` para igualdad estructural

#### Domain - Representación de Soluciones

```rust
pub enum SolutionSet {
    Empty,                           // ∅
    Point(BigRational),             // {x}
    Interval(Interval),             // (a, b), [a, b], etc.
    Union(Vec<SolutionSet>),        // A ∪ B
    Intersection(Vec<SolutionSet>), // A ∩ B
}
```

Permite representar soluciones de ecuaciones e inecuaciones de forma precisa.

#### Visitor Pattern

Implementa el patrón Visitor para recorrer y transformar expresiones:

```rust
pub trait Visitor {
    fn visit_expr(&mut self, ctx: &Context, id: ExprId);
}

pub trait Transformer {
    fn transform_expr(&mut self, ctx: &mut Context, id: ExprId) -> ExprId;
}
```

**Uso**: Recolección de variables, cálculo de profundidad, validación, etc.

---

### 2.1. `cas_ast` - Automatic Canonical Ordering (conmutativo-only) ★★★

**CRÍTICO**: garantiza una representación estructural **determinista** para operaciones conmutativas sin imponer una “normalización global” cara.

#### El Problema: Múltiples Representaciones Estructuralmente Distintas

Sin un criterio canónico, expresiones matemáticamente equivalentes aparecen con múltiples ASTs:

- `2 + x` → `Add(2, x)` o `Add(x, 2)`
- `x*y` → `Mul(x, y)` o `Mul(y, x)`
- `a - b` → puede existir como `Sub(a,b)` o como azúcar sintáctico `Add(a, Neg(b))`

Esto complica:
- pattern matching (múltiples patrones para lo mismo)
- control de ciclos / terminación
- trazas (Before/After) inconsistentes

#### La Solución: Canonicalización local en `Context::add()`

`Context::add()` aplica **ordenación determinista** de operandos para `Add` y `Mul` (cuando es seguro), usando `compare_expr` (orden estructural determinista, sin hash).

**Importante**: la canonicalización en `add()` es **local** (swap), no flattening/ordenación de toda la cadena.

```rust
// crates/cas_ast/src/expression.rs (resumen)
match expr {
    Expr::Add(l, r) => {
        if compare_expr(self, l, r) == Ordering::Greater { Expr::Add(r, l) }
        else { Expr::Add(l, r) }
    }
    Expr::Mul(l, r) => {
        // CRITICAL: Matrix multiplication is NOT commutative
        let l_is_matrix = matches!(self.get(l), Expr::Matrix { .. });
        let r_is_matrix = matches!(self.get(r), Expr::Matrix { .. });

        if l_is_matrix && r_is_matrix {
            Expr::Mul(l, r) // no swap: A*B ≠ B*A
        } else if compare_expr(self, l, r) == Ordering::Greater {
            Expr::Mul(r, l) // swap seguro (escalares / mixto)
        } else {
            Expr::Mul(l, r)
        }
    }
    other => other,
}
```

#### `Context::add_raw()` - Construcción sin canonicalizar (para snapshots/trace)

Además de `add()`, existe `add_raw()` para **preservar estructura** sin swaps ni otras canonicalizaciones.

Usos principales:
- **Tracer / steps**: reconstrucción de subárboles por “path” debe ser estable; usar `add()` podía reordenar y corromper `Before/After`.
- **Pruebas y herramientas** que necesitan construir exactamente el árbol esperado.

> Regla práctica: el motor “normal” construye expresiones con builders (`mul2_raw`, `mul2_simpl`, `MulBuilder`), no con `ctx.add(Expr::Mul...)` directo.

#### Separación de responsabilidades (muy importante)

- **Canonical Ordering** (estructura canónica local): `compare_expr` (sin hash)
- **Semantic Equality** (prueba de igualdad): `SemanticEqualityChecker::are_equal` (sin hash; “true” = probado, “false” = no probado)
- **Cycle detection** (heurística rápida): `semantic_hash` (con hash, scope limitado)

---
### 2.2. Normalización Canónica y Views (C2) ★★★

Esta parte recoge el cambio arquitectónico más importante de la conversación: pasar de “múltiples formas internas” (Sub/Div vs Add+Neg / Mul+Pow(-1)) a un enfoque **híbrido**:

- El AST interno puede contener azúcar (`Sub`, `Div`) por compatibilidad / parser / reglas existentes.
- El *pattern matching* y la construcción de expresiones se hace sobre **views normalizadas** y **builders deterministas**.
- El display (y LaTeX) puede “embellecer” representaciones canónicas (por ejemplo `a * b^(-1)` → `a/b`).

#### Problema original: representación interna inconsistente

Ejemplo típico:
- división podía aparecer como `Div(n,d)` o como `Mul(n, Pow(d,-1))`
- resta podía aparecer como `Sub(a,b)` o `Add(a, Neg(b))`

Esto obligaba a reglas a reconocer demasiados casos (`Div`, `Pow(-1)`, `Mul(..., Pow(...,-1))`, `Neg(Div(...))`, etc.).

#### Decisión C2: “dos views + normalizador canónico”

En lugar de una normalización interna total (refactor masivo), se adoptó una estrategia tipo *NormalizedView*:

> **Idea**: exponer una “vista canónica” solo para matching / análisis, manteniendo el AST original para compatibilidad y display.

En el código, esta idea se materializa con **views** en `crates/cas_ast/src/views.rs`:

- **`MulView` / `MulChain`**: lineariza factores de `Mul` independientemente de asociatividad (y puede preservar orden).
- **`MulParts`**: extrae signo + factores como `(base, exponente_entero)`; útil para productos conmutativos.
- **`RationalFnView` / `FractionParts`**: descompone expresiones racionales en numerador/denominador (sin depender de `Div` como forma única).
- **`MulBuilder`**: construye productos de forma determinista y segura (right-fold), con modos que controlan flattening.

#### `MulBuilder` y terminación (lecciones aprendidas)

`MulBuilder` incluye modos explícitos:

- `new_simple()` (**para recursión**): no aplana, solo añade factores tal cual.
- `FlattenMul`: aplana `Mul` anidados.
- `MulPowInt`: además combina potencias con exponente entero (sin tocar `Div`).

Reglas de seguridad:
- **Nunca** usar `push_expr` agresivo dentro de funciones recursivas (expansión, distribución, etc.).
- **Nunca** aplanar `Div` en un builder: para fracciones se usa `RationalFnView`.

#### API de construcción unificada (cas_engine)

Para eliminar `ctx.add(Expr::Mul(..))` dispersos y hacer el comportamiento explícito:

- `mul2_raw(ctx,a,b)` / `mul_many_raw(ctx, &[...])`: construcción estable (sin simplificar)
- `mul2_simpl(ctx,a,b)` / `mul_many_simpl(...)`: construcción + simplificación local (`1*x → x`, etc.)

Además existe un lint de CI que impide reintroducir `ctx.add(Expr::Mul..)` en `src/`.

#### Display separado del AST

Con C2, el AST interno puede quedar en forma canónica (ej. `Mul(a, Pow(b,-1))`) y el display:
- lo presenta como `a/b`
- y presenta `Add(a, Neg(b))` como `a - b`

Esto mantiene:
- reglas más simples (una sola forma canónica “preferida”)
- trazas legibles para el usuario

---
### 2.5. `cas_engine` - Pattern Detection Infrastructure ★★★

**CRÍTICO**: Sistema agregado en 2025-12 después de 10+ horas de implementación y debugging.

#### Motivación

**Problema Fundamental**: El sistema de simplificación bottom-up pierde contexto.

**Ejemplo del Problema**:
```
Input: sec²(x) - tan²(x)

Bottom-Up Processing:
1. Simplifica tan²(x) primero
2. TanToSinCosRule: tan(x) → sin(x)/cos(x)
3. Se pierde la oportunidad de aplicar sec²-tan²=1
4. Result: Expresión compleja en sin/cos
```

**Solución**: **Pre-análisis de patrones antes de la simplificación**.

---

#### Arquitectura del Sistema

```
┌─────────────────────────────────────────┐
│  1. Pattern Detection (Pre-Analysis)    │
│     - Scan AST before simplification    │
│     - Mark protected expressions        │
│     - O(n) traversal, one time cost     │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  2. Pattern Marks (Data Structure)      │
│     - HashSet<ExprId>                   │
│     - O(1) lookups                      │
│     - Marks base trig functions         │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  3. Data Flow (Orchestrator → Rules)    │
│     - Thread through transformers       │
│     - Via ParentContext                 │
│     - No global state                   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  4. Guards & Direct Rules                │
│     - Guards: Skip premature conversion │
│     - Direct: Apply identity directly   │
│     - Context-aware decisions           │
└─────────────────────────────────────────┘
```

---

#### Componente 1: PatternMarks (`pattern_marks.rs`)

**Responsabilidad**: Lightweight data structure para marcar expresiones protegidas.

```rust
pub struct PatternMarks {
    protected: HashSet<ExprId>,
}

impl PatternMarks {
    pub fn new() -> Self {
        Self { protected: HashSet::new() }
    }
    
    pub fn mark_protected(&mut self, expr: ExprId) {
        self.protected.insert(expr);
    }
    
    pub fn is_pythagorean_protected(&self, expr: ExprId) -> bool {
        self.protected.contains(&expr)
    }
}
```

**Características**:
- O(1) lookups
- ~8 bytes por expresión marcada
- Clone-friendly (para threading)
- No lifetime issues

---

#### Componente 2: PatternScanner (`pattern_scanner.rs`)

**Responsabilidad**: Detectar patrones Pythagorean antes de simplificación.

```rust
pub fn scan_and_mark_patterns(
    ctx: &Context,
    expr_id: ExprId,
    marks: &mut PatternMarks
) {
    // Recursive depth-first traversal
    match ctx.get(expr_id) {
        Expr::Add(left, right) => {
            // Check for sec²-tan² or csc²-cot² pattern
            if is_pythagorean_difference(ctx, *left, *right) {
                // Mark base expressions (tan(x), sec(x), etc.)
                marks.mark_protected(extract_base(*left));
                marks.mark_protected(extract_base(*right));
            }
            
            // Recurse to children
            scan_and_mark_patterns(ctx, *left, marks);
            scan_and_mark_patterns(ctx, *right, marks);
        }
        
        // ... other cases ...
    }
}
```

**Patrones Detectados**:
1. **sec²(x) - tan²(x)**: Marca `sec(x)` y `tan(x)`
2. **csc²(x) - cot²(x)**: Marca `csc(x)` y `cot(x)`

**Complejidad**: O(n) donde n = tamaño del AST, ejecutado **una única vez**.

---

#### Componente 3: ParentContext (`parent_context.rs`)

**Responsabilidad**: Pasar contexto de padres a hijos durante transformación.

```rust
pub struct ParentContext {
    ancestors: Vec<ExprId>,           // From closest to furthest
    pattern_marks: Option<PatternMarks>, // Pre-scanned marks
}

impl ParentContext {
    pub fn root() -> Self {
        Self { ancestors: Vec::new(), pattern_marks: None }
    }
    
    pub fn with_marks(marks: PatternMarks) -> Self {
        Self { ancestors: Vec::new(), pattern_marks: Some(marks) }
    }
    
    pub fn extend(&self, parent_id: ExprId) -> Self {
        let mut new_ancestors = self.ancestors.clone();
        new_ancestors.push(parent_id); // Add to end
        Self {
            ancestors: new_ancestors,
            pattern_marks: self.pattern_marks.clone(),
        }
    }
    
    pub fn pattern_marks(&self) -> Option<&PatternMarks> {
        self.pattern_marks.as_ref()
    }
    
    pub fn immediate_parent(&self) -> Option<ExprId> {
        self.ancestors.last().copied() // Most recent is last
    }
}
```

**Uso**: Threading de pattern_marks desde Orchestrator hasta Rules.

---

#### Componente 4: Pattern Detection Helpers (`pattern_detection.rs`)

**Responsabilidad**: Funciones auxiliares para detectar patrones específicos.

```rust
/// Check if expression is sec²(x)
pub fn is_sec_squared(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    if let Expr::Pow(base, exp) = ctx.get(expr) {
        if is_constant_two(ctx, *exp) {
            if let Expr::Function(name, args) = ctx.get(*base) {
                if name == "sec" && args.len() == 1 {
                    return Some(args[0]); // Return argument of sec(x)
                }
            }
        }
    }
    None
}

/// Similarly for tan², csc², cot²
pub fn is_tan_squared(ctx: &Context, expr: ExprId) -> Option<ExprId> { ... }
pub fn is_csc_squared(ctx: &Context, expr: ExprId) -> Option<ExprId> { ... }
pub fn is_cot_squared(ctx: &Context, expr: ExprId) -> Option<ExprId> { ... }
```

**Tests**: 17/17 passing para detección de patrones.

---

#### Data Flow Completo

```
User Input: sec²(x) - tan²(x)
    ↓
┌────────────────────────────────────────┐
│ Orchestrator::simplify()               │
│   1. pattern_marks = PatternScanner    │
│      ::scan_and_mark_patterns(expr)    │
│      → Marks: {tan(x), sec(x)}         │
└──────────┬─────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────┐
│ Simplifier::apply_rules_loop(expr,     │
│                             &pattern_marks)
│   → Passes marks to transformer        │
└──────────┬─────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────┐
│ LocalSimplificationTransformer         │
│   initial_parent_ctx:                  │
│     ParentContext::with_marks(marks)   │
└──────────┬─────────────────────────────┘
           │
           ▼ (for each node)
┌────────────────────────────────────────┐
│ Rule::apply(ctx, expr, parent_ctx)     │
│                                        │
│ ┌──────────────────────────────────┐  │
│ │ TanToSinCosRule (GUARD):         │  │
│ │   if parent_ctx.pattern_marks()  │  │
│ │      .is_pythagorean_protected() │  │
│ │      { return None; } // SKIP!   │  │
│ └──────────────────────────────────┘  │
│                                        │
│ ┌──────────────────────────────────┐  │
│ │ SecTanPythagoreanRule (DIRECT):  │  │
│ │   sec²(x) - tan²(x) → 1          │  │
│ │   Matches Add(sec², Neg(tan²))   │  │
│ │   Returns: Rewrite { 1 }         │  │
│ └──────────────────────────────────┘  │
└────────────────────────────────────────┘
```

---

#### Cambios en el Motor de Simplificación

**engine.rs** - Modificaciones clave:

```rust
// ANTES: apply_rules_loop no recibía pattern_marks
pub fn apply_rules_loop(&mut self, expr: ExprId) -> (ExprId, Vec<Step>)

// AHORA: Recibe y utiliza pattern_marks
pub fn apply_rules_loop(
    &mut self,
    expr: ExprId,
    pattern_marks: &PatternMarks  // ← NUEVO parámetro
) -> (ExprId, Vec<Step>) {
    let initial_parent_ctx = ParentContext::with_marks(pattern_marks.clone());
    
    let transformer = LocalSimplificationTransformer {
        // ...
        initial_parent_ctx,  // ← NUEVO campo
    };
    
    // ...
}
```

**LocalSimplificationTransformer** - Nuevo campo:

```rust
struct LocalSimplificationTransformer<'a> {
    context: &'a mut Context,
    rules: &'a HashMap<String, Vec<Rc<dyn Rule>>>,
    global_rules: &'a Vec<Rc<dyn Rule>>,
    cache: HashMap<ExprId, ExprId>,
    steps: Vec<Step>,
    current_path: Vec<PathStep>,
    initial_parent_ctx: ParentContext,  // ← NUEVO: Pattern marks threading
}
```

**apply_rules** - Uso del ParentContext:

```rust
fn apply_rules(
    &mut self,
    expr: ExprId,
    parent_ctx: &ParentContext,  // ← Recibe context
) -> ExprId {
    // Pass parent_ctx to each rule
    for rule in specific_rules {
        if let Some(rewrite) = rule.apply(&mut self.context, expr, parent_ctx) {
            // ... record step ...
            return self.apply_rules(rewrite.new_expr, parent_ctx);
        }
    }
    expr
}
```

---

#### Implementación de Reglas con Guards

**TanToSinCosRule** - Conversión manual de macro a implementación:

```rust
pub struct TanToSinCosRule;

impl Rule for TanToSinCosRule {
    fn name(&self) -> &str {
        "Tan to Sin/Cos"
    }
    
    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        parent_ctx: &ParentContext,  // ← Recibe context
    ) -> Option<Rewrite> {
        // GUARD: Check if protected by pattern detection
        if let Some(marks) = parent_ctx.pattern_marks() {
            if marks.is_pythagorean_protected(expr) {
                return None; // ← SKIP conversion!
            }
        }
        
        // Normal logic: tan(x) → sin(x)/cos(x)
        if let Expr::Function(name, args) = ctx.get(expr) {
            if name == "tan" && args.len() == 1 {
                let sin_expr = ctx.add(Expr::Function("sin".to_string(), args.clone()));
                let cos_expr = ctx.add(Expr::Function("cos".to_string(), args.clone()));
                let div_expr = ctx.add(Expr::Div(sin_expr, cos_expr));
                
                return Some(Rewrite {
                    new_expr: div_expr,
                    description: "tan(x) → sin(x)/cos(x)".to_string(),
                });
            }
        }
        None
    }
}
```

---

#### Implementación de Reglas Directas

**SecTanPythagoreanRule** - Identidad directa sec²-tan²=1:

**CRÍTICO - AST Normalization Insight**:
```
CAS normaliza: a - b → Add(a, Neg(b))  NO Sub(a, b)!

Por tanto, sec²(x) - tan²(x) en el AST es:
  Add(Pow(sec(x), 2), Neg(Pow(tan(x), 2)))
  
NO es:
  Sub(Pow(sec(x), 2), Pow(tan(x), 2))  ← Este NUNCA existe!
```

**Implementación correcta**:

```rust
define_rule!(
    SecTanPythagoreanRule,
    "Secant-Tangent Pythagorean Identity",
    |ctx, expr| {
        use crate::pattern_detection::{is_sec_squared, is_tan_squared};
        
        let expr_data = ctx.get(expr).clone();
        
        // Pattern: sec²(x) - tan²(x) = 1
        // CRITICAL: Matches Add(sec², Neg(tan²))
        if let Expr::Add(left, right) = expr_data {
            // Try both orderings
            for (pos, neg) in [(left, right), (right, left)] {
                if let Expr::Neg(neg_inner) = ctx.get(neg) {
                    if let (Some(sec_arg), Some(tan_arg)) =
                        (is_sec_squared(ctx, pos), is_tan_squared(ctx, *neg_inner))
                    {
                        // Check arguments match
                        if compare_expr(ctx, sec_arg, tan_arg) == Ordering::Equal {
                            return Some(Rewrite {
                                new_expr: ctx.num(1),
                                description: "sec²(x) - tan²(x) = 1".to_string(),
                            });
                        }
                    }
                }
            }
        }
        None
    }
);
```

**CscCotPythagoreanRule** - Misma estructura para csc²-cot²=1.

---

#### Registro de Reglas

**trigonometry.rs** - Función `register`:

```rust
pub fn register(simplifier: &mut Simplifier) {
    // ... existing rules ...
    
    // NEW: Pythagorean identity rules
    simplifier.add_rule(Box::new(SecTanPythagoreanRule));
    simplifier.add_rule(Box::new(CscCotPythagoreanRule));
    
    // NOTE: TanToSinCosRule now has guard, manual impl
    simplifier.add_rule(Box::new(TanToSinCosRule));
}
```

---

#### Tests y Verificación

**Cobertura**:
- `pattern_scanner.rs`: 17/17 tests passing
- `pythagorean_variants_test.rs`: 3/3 tests passing
- `debug_sec_tan.rs`: 3/3 tests passing
- Full test suite: 102/102 passing

**Ejemplos de test**:

```rust
#[test]
fn test_sec_tan_equals_one() {
    let expr = parse("sec(x)^2 - tan(x)^2");
    let (result, _) = simplifier.simplify(expr);
    assert_eq!(display(result), "1");
}

#[test]
fn test_csc_cot_equals_one() {
    let expr = parse("csc(x)^2 - cot(x)^2");
    let (result, _) = simplifier.simplify(expr);
    assert_eq!(display(result), "1");
}
```

---

#### Métricas de Implementación

- **Tiempo de desarrollo**: 10+ horas
- **Líneas de código**: ~750
  - Pattern infrastructure: ~400
  - Engine modifications: ~200
  - Rules + tests: ~150
- **Archivos nuevos**: 3 (`pattern_marks`, `pattern_scanner`, tests)
- **Archivos modificados**: 7
- **Performance overhead**: O(n) one-time scan, O(1) lookups
- **Memory overhead**: ~8 bytes × número de expresiones protegidas

---

#### Lessons Learned - AST Normalization

**CRÍTICO**: El descubrimiento más importante de esta implementación.

1. **Subtraction is Sugar**:
   ```
   Parser input:     a - b
   AST representation: Add(a, Neg(b))
   ```

2. **Pattern Matching Implications**:
   ```rust
   // ❌ NUNCA funciona
   if let Expr::Sub(left, right) = ctx.get(expr) { ... }
   
   // ✅ SIEMPRE usar
   if let Expr::Add(left, right) = ctx.get(expr) {
       if let Expr::Neg(neg_inner) = ctx.get(right) { ... }
   }
   ```

3. **Why This Matters**:
   - Simplifica el motor: solo un variant para suma
   - Canonical forms más fáciles
   - Pero requiere understanding explícito de normalización

4. **Other Normalizations** (to be documented):
   - `a/b` podría ser `Div(a, b)` o `Mul(a, Pow(b, -1))` dependiendo de fase
   - `sqrt(x)` normalizado a `Pow(x, Rational(1,2))`

**Tiempo invertido en descubrir esto**: ~9 horas de las 10 horas totales.

---

#### Extensibilidad

Para agregar nuevos patrones protegidos:

1. **Añadir detección en `pattern_scanner.rs`**:
   ```rust
   fn scan_and_mark_patterns(...) {
       match ctx.get(expr_id) {
           // ... existing patterns ...
           
           // NEW pattern
           Expr::YourPattern(...) => {
               if matches_your_condition(...) {
                   marks.mark_protected(base_expr);
               }
           }
       }
   }
   ```

2. **Añadir helper en `pattern_detection.rs`**:
   ```rust
   pub fn is_your_pattern(ctx: &Context, expr: ExprId) -> Option<ExprId> {
       // Detection logic
   }
   ```

3. **Añadir regla directa opcional**:
   ```rust
   define_rule!(YourPatternRule, "Description", |ctx, expr| {
       if matches_your_pattern(ctx, expr) {
           return Some(Rewrite { ... });
       }
       None
   });
   ```

4. **Añadir guard en rule existente**:
   ```rust
   if let Some(marks) = parent_ctx.pattern_marks() {
       if marks.is_your_pattern_protected(expr) {
           return None;
       }
   }
    ```

---

### 2.6. N-ary Pattern Matching (Add/Mul Flattening) ★★

**Implementado**: 2025-12-08  
**Motivación**: Permitir que reglas encuentren patrones entre términos NO adyacentes en sumas/productos.

#### El Problema: Canonicalización Separa Patrones

**Ejemplo**: `atan(2) + atan(1/2) - π/2`

```
Parser Output: atan(2) + atan(1/2) - π/2

Canonicalización:
  → atan(1/2) + atan(2) + (-1)*(1/2)*π
  → Add(Add(atan(1/2), atan(2)), Mul(-1, Mul(1/2, π)))
  
Problema: InverseTrigAtanRule busca: atan(x) + atan(1/x)
  Pero solo ve pares binarios en Add:
    - Add(atan(1/2), atan(2)) ← ¡SÍ!, pero...
    - Adición externa: Add(pair, otros_términos)
    - La regla no se activa porque hay un tercer término
```

#### La Solución: Aplanar Árbol Add → Lista de Términos

```rust
// Helper: Aplana árbol Add recursivamente
fn collect_add_terms_flat(ctx: &Context, expr_id: ExprId) -> Vec<ExprId> {
    let mut terms = Vec::new();
    collect_add_terms_recursive(ctx, expr_id, &mut terms);
    terms
}

fn collect_add_terms_recursive(ctx: &Context, expr_id: ExprId, terms: &mut Vec<ExprId>) {
    match ctx.get(expr_id) {
        Expr::Add(l, r) => {
            collect_add_terms_recursive(ctx, *l, terms);  // Recursión izquierda
            collect_add_terms_recursive(ctx, *r, terms);  // Recursión derecha
        }
        _ => terms.push(expr_id),  // Hoja: agregar término
    }
}
```

**Ejemplo**: `((a + b) + c) - d` → `[a, b, c, Neg(d)]`

#### Implementación en InverseTrigAtanRule

```rust
define_rule!(InverseTrigAtanRule, "Inverse Tan Relations", Some(vec!["Add"]), |ctx, expr| {
    // 1. Aplanar todos los términos
    let terms = collect_add_terms_flat(ctx, expr);
    if terms.len() < 2 { return None; }
    
    // 2. Buscar pares reciprocales en TODOS los términos (O(n²))
    for i in 0..terms.len() {
        for j in (i+1)..terms.len() {
            if is_atan(i) && is_atan(j) && are_reciprocals(arg[i], arg[j]) {
                // 3. Reconstruir suma sin términos i, j
                let remaining = build_sum_without(&terms, i, j);
                // 4. Agregar π/2
                return Some(combine(remaining, pi_half));
            }
        }
    }
    None
});
```

#### Caso de Uso: Test 48

```
Input: atan(2) + atan(1/2) - π/2

Paso 1: terms = [atan(1/2), atan(2), (-1)*(1/2)*π]
Paso 2: i=0,j=1 → are_reciprocals(1/2, 2)? SÍ ✓
Paso 3: remaining = [(-1)*(1/2)*π]
        result = (-1)*(1/2)*π + π/2
Paso 4: Iteración 2 (multi-pass) → 0 ✓
```

#### Performance

- **Complejidad**: O(t²) donde t = número de términos
- **Típico**: t=3-5 → 3-10 comparaciones → <1ms
- **Aceptable** hasta ~20 términos

#### Archivos: `inverse_trig.rs`, Tests: `test_48`

---

### 2.7. Patrón de Negación Generalizada ★★★

**Implementado**: 2025-12-08  
**Motivación**: Reglas de pares (f+g=V) automáticamente manejan versiones negadas (-f-g=-V).

#### El Patrón Matemático

Si una regla detecta: `f(x) + g(x) = V`  
Entonces también debe detectar: `-f(x) - g(x) = -V`

**Ejemplo**:
- `atan(x) + atan(1/x) = π/2` → Ya implementado ✓
- `-atan(x) - atan(1/x) = -π/2` → ¿Duplicar código? ❌

#### Problema: Duplicación de Código

**Antes** (sin generalización):
```rust
define_rule!(InverseTrigAtanRule, ..., |ctx, expr| {
    // Case 1: Positive pair
    if is_atan(i) && is_atan(j) && are_reciprocals(arg[i], arg[j]) {
        return Some(Rewrite { new_expr: pi_half, ... });  // ~15 líneas
    }
    
    // Case 2: Negated pair (CÓDIGO DUPLICAD O!)
    if let (Neg(inner_i), Neg(inner_j)) = (term_i, term_j) {
        // Extraer inner atan...
        if is_atan(i) && is_atan(j) && are_reciprocals(arg[i], arg[j]) {
            return Some(Rewrite { new_expr: neg_pi_half, ... });  // ~15 líneas IDÉNTICAS
        }
    }
});
```

**Problema**: Lógica duplicada, bugs duplicados, mantenimiento duplicado.

#### Solución: Helper `check_pair_with_negation()`

**Archivo**: `crates/cas_engine/src/rules/inverse_trig.rs`

```rust
/// Helper genérico para reglas de pares con negación automática
/// 
/// Patrón: Si f(x) + g(x) = V, entonces -f(x) - g(x) = -V
fn check_pair_with_negation<F>(
    ctx: &mut Context,
    term_i_data: Expr,      // Primer término (owned para evitar borrow issues)
    term_j_data: Expr,      // Segundo término
    terms: &[ExprId],       // Todos los términos de la suma
    i: usize, j: usize,     // Índices a eliminar
    check_fn: F,            // Función de chequeo específica de la regla
) -> Option<Rewrite>
where
    F: Fn(&mut Context, &Expr, &Expr) -> Option<(ExprId, String)>
{
    //  Case 1: Pares positivos
    if let Some((result, desc)) = check_fn(ctx, &term_i_data, &term_j_data) {
        let remaining = build_sum_without(ctx, terms, i, j);
        let final_result = combine_with_term(ctx, remaining, result);
        return Some(Rewrite { new_expr: final_result, description: desc });
    }

    // Case 2: Pares negados (AUTOMÁTICO!)
    if let (Expr::Neg(inner_i), Expr::Neg(inner_j)) = (&term_i_data, &term_j_data) {
        let inner_i_data = ctx.get(*inner_i).clone();
        let inner_j_data = ctx.get(*inner_j).clone();

        if let Some((result, desc)) = check_fn(ctx, &inner_i_data, &inner_j_data) {
            // Negar el resultado
            let neg_result = ctx.add(Expr::Neg(result));
            let remaining = build_sum_without(ctx, terms, i, j);
            let final_result = combine_with_term(ctx, remaining, neg_result);

            return Some(Rewrite {
                new_expr: final_result,
                description: format!("-[{}]", desc),  // Descripción ajustada
            });
        }
    }

    None
}
```

#### Uso: Reglas Refactorizadas

**InverseTrigAtanRule** (atan reciprocal):
```rust
define_rule!(InverseTrigAtanRule, ..., |ctx, expr| {
    let terms = collect_add_terms_flat(ctx, expr);
    
    for i in 0..terms.len() {
        for j in (i + 1)..terms.len() {
            // Usa el helper - maneja pos/neg automáticamente!
            if let Some(rewrite) = check_pair_with_negation(
                ctx, term_i, term_j, &terms, i, j,
                |ctx, expr_i, expr_j| {
                    // Solo escribe la lógica UNA vez para caso positivo
                    if is_atan(i) && is_atan(j) && are_reciprocals(arg_i, arg_j) {
                        return Some((pi_half, "arctan(x) + arctan(1/x) = π/2".to_string()));
                    }
                    None
                }
            ) {
                return Some(rewrite);
            }
        }
    }
    None
});
```

**InverseTrigSumRule** (asin + acos):
```rust
// Mismo patrón - reutilización del helper
check_pair_with_negation(ctx, term_i, term_j, &terms, i, j, |ctx, i, j| {
    if is_asin(i) && is_acos(j) && args_equal {
        return Some((pi_half, "arcsin(x) + arccos(x) = π/2".to_string()));
    }
    None
})
```

#### Beneficios

| Aspecto | Antes | Después |
|---------|-------|---------|
| Líneas de código | ~80 por regla | ~35 por regla (-56%) |
| Duplicación | 100% | 0% ✓ |
| Bugs | 2× (pos + neg) | 1× (compartido) |
| Mantenimiento | 2 lugares | 1 lugar |
| Reglas usando patrón | 0 | 2 (atan, asin/acos) |

#### Casos de Uso

**Positivo** (ya funcionaba):
```
atan(2) + atan(1/2) → π/2
asin(x) + acos(x) → π/2
```

**Negado** (NUEVO con helper):
```
-atan(2) - atan(1/2) → -π/2  ✓ NUEVO!
-asin(x) - acos(x) → -π/2     ✓ NUEVO!
```

**Con variables**:
```
-atan(1/2) - atan(2) + x → x - π/2  ✓ ¡Beats Sympy!
-asin(x) - acos(x) + y → y - π/2     ✓ NUEVO!
```

#### Extensibilidad

**Listo para aplicar a**:
- **Pythagorean**: `sin²(x) + cos²(x) = 1` → `-sin²(x) - cos²(x) = -1`
- **Hyperbolic**: `cosh²(x) - sinh²(x) = 1` → ...
- **Cualquier regla de pares futura**

#### Tests

- `negated_atan_pairs_tests.rs`: 5 tests
- `generalized_negation_tests.rs`: 30 tests (atan)
- `asin_acos_negation_tests.rs`: 13 tests (asin/acos)

**Total**: 48 tests verificando el patrón de negación ✓

---

### 2.8. `cas_engine` - Semantic Equality + nf_score (aceptación de rewrites) ★★★

Este bloque resolvió un bug crítico: algunas reglas de canonicalización quedaban “bloqueadas” cuando el wrapper descartaba rewrites semánticamente iguales (por ejemplo `Div(1,2)` vs `Number(1/2)`), rompiendo cadenas como `EvaluatePowerRule`.

#### Semántica de `are_equal` (comentario clave)

- `checker.are_equal(a,b) == true` significa **“igualdad demostrada”**
- `false` significa **“no puedo probarlo”**, no “son distintos”

Esto evita asumir “diferencia” cuando la comparación es incompleta.

#### Wrapper: `apply_rule_with_semantic_check()`

Política actual:

1) Si `are_equal(before, after)` es **false** → aceptar rewrite (puede ser cambio real).
2) Si es **true** → aceptar **solo** si mejora forma normal (`nf_score_after_is_better`).
3) Excepción: reglas didácticas pueden forzar step aunque sea equivalente (para explicación).

#### `nf_score v2`: calidad de forma normal (terminación sin perder canonicidad)

La métrica prioriza:
1) menos `Div` y `Sub` (objetivo C2)
2) menos nodos totales
3) *tie-breaker* opcional: menos “inversiones” en cadenas `Mul` (determinismo sin reordenar agresivo)

Para evitar coste/overflow:
- `nf_score_base` hace un **single-pass traversal**.
- `mul_unsorted_adjacent` se calcula **solo si hay empate** (lazy).

#### Numeric evaluation depth-limited (anti stack overflow)

`SemanticEqualityChecker` usa evaluación numérica con profundidad limitada (p.ej. 50) para evitar overflow en expresiones profundas (proptest).

---

**Resultado**: se aceptan rewrites canónicos (ej. racionales constantes) sin abrir loops infinitos.

---
### 2. `cas_parser` - Parser de Expresiones

**Responsabilidad**: Convertir texto en AST.

#### Tecnología

Utiliza la crate `pest` con una gramática PEG (Parsing Expression Grammar).
---

## Optimizaciones (Phase 1 y Phase 2)

### Phase 1: Performance Optimization

#### **Conditional Multi-Pass** ★

**Problema**: El loop multi-pass incondicional causaba **regresiones masivas**:
- `sum_fractions_10`: +99.9% slower  
- `solve_quadratic`: +50.7% slower
- `diff_nested_trig_exp`: +34.4% slower

**Solución**: Multi-pass **solo cuando es necesario**.

**Cascade Triggers**:
```rust
let cascade_triggers = HashSet::from([
    "Rationalize Denominator",   // Crea nuevas oportunidades
    "Add fractions",             // Puede crear términos combinables
    "Pull Constant From Fraction",
    // ... (10 reglas totales)
]);
```

**Algoritmo**:
```rust
// First pass (always)
let (first_pass, first_steps) = simplifier.apply_rules_loop(current);

// Fast path: No changes at all
if first_pass == current {
    return;  // ~95% de casos
}

// Check if cascade triggers fired
let needs_multi_pass = first_steps.iter()
    .any(|step| cascade_triggers.contains(step.rule_name));

if needs_multi_pass {
    // SLOW PATH: Multi-pass loop (solo ~5% de casos)
    loop {
        let (simplified, steps) = simplifier.apply_rules_loop(current);
        if simplified == current { break; }  // Early exit (O(1))
        
        if cycle_detector.check(current).is_some() { break; }
        
        current = simplified;
        if pass_count >= 5 { break; }  // Safety
    }
}
```

**Resultados**:
| Benchmark | Antes | Después | Mejora |
|-----------|-------|---------|--------|
| `sum_fractions_10` | +99.9% | **-46.8%** | 146.7% faster |
| `solve_quadratic` | +50.7% | **-30.4%** | 81.1% faster |
| `diff_nested_trig_exp` | +34.4% | **-21.2%** | 55.6% faster |
| `integrate_trig_product` | +65.9% | **-36.0%** | 101.9% faster |

#### **Cycle Detection**

Previene loops infinitos:
```rust
struct CycleDetector {
    recent: VecDeque<ExprId>,
    max_len: usize,
}

impl CycleDetector {
    fn check(&mut self, expr: ExprId) -> Option<usize> {
        if let Some(pos) = self.recent.iter().position(|&e| e == expr) {
            return Some(self.recent.len() - pos);  // Cycle length
        }
        self.recent.push_back(expr);
        if self.recent.len() > self.max_len {
            self.recent.pop_front();
        }
        None
    }
}
```

#### **Early Exit Optimization**

```rust
// O(1) check antes de expensive compare_expr
if simplified == current {
    break;  // No changes
}
```

**Impact**: Evita ~1000 comparaciones estructurales por simplificación.

---

### Phase 2: Debug Tools

Herramientas de debugging con **zero runtime overhead** cuando no están en uso.

#### **1. Rule Profiler**

Track rule hit frequency para performance analysis.

**Implementation**:
```rust
pub struct RuleProfiler {
    stats: HashMap<String, RuleStats>,
    enabled: bool,  // Default: false
}

pub struct RuleStats {
    pub hit_count: AtomicUsize,  // Thread-safe
}

impl RuleProfiler {
    pub fn record(&mut self, rule_name: &str) {
        if !self.enabled { return; }  // Zero overhead
        
        let stats = self.stats.entry(rule_name.to_string()).or_default();
        stats.hit_count.fetch_add(1, Ordering::Relaxed);  // <1ns
    }
}
```

**Usage**:
```bash
> profile enable
> (x+1)^2
> profile
Rule Profiling Report
─────────────────────────────────────────────
Rule                                      Hits
─────────────────────────────────────────────
Binomial Expansion                           1
Combine Like Terms                           2
─────────────────────────────────────────────
TOTAL                                        3
```

**Overhead**:
- Disabled (default): **0ns**
- Enabled: **<1ns** per rule (atomic increment)

#### **2. AST Visualizer**

Export expression trees to Graphviz DOT format.

**Implementation**:
```rust
pub struct AstVisualizer<'a> {
    context: &'a Context,
    visited: HashSet<ExprId>,
}

impl<'a> AstVisualizer<'a> {
    pub fn to_dot(&mut self, expr: ExprId) -> String {
        // Generate DOT format with color-coded nodes
        // - Numbers: Light blue
        // - Variables: Light green
        // - Operators: Orange/pink gradients
    }
}
```

**Usage**:
```bash
> visualize (x+1)*(x-1)
AST exported to ast.dot
```

**Features**:
- Color-coded nodes
- Auto-layout with Graphviz
- Export to SVG/PNG/PDF
- Zero runtime overhead (export only)

#### **3. Timeline HTML - Visualización Inteligente de Pasos**

Sistema de visualización interactiva con filtrado inteligente y renderizado contextual.

**Implementation**:
```rust
pub struct TimelineHtml<'a> {
    context: &'a mut Context,
    steps: &'a [Step],
    original_expr: ExprId,
    verbosity: VerbosityLevel,
}

pub enum VerbosityLevel {
    Low,      // Solo pasos High importance
    Normal,   // High + Medium importance
    Verbose,  // Todos los pasos
}
```

**Características Principales**:

##### 1. **Filtrado Inteligente de Pasos** ★

Dos niveles de filtrado para reducir ruido visual:

###### a) Filtrado Global de Estado
```rust
// En strategies.rs
pub fn filter_non_productive_steps(
    ctx: &mut Context,
    original: ExprId,
    steps: Vec<Step>
) -> Vec<Step> {
    // Filtra pasos donde el estado global no cambia semánticamente
    // Ejemplo: `x^(1/3 * 2)` → `x^(2/3)` es cambio de notación, no semántico
}
```

**Problema resuelto**: Reglas como "Multiply exponents" pueden aplicarse a subexpresiones sin cambiar el estado global visible.

**Ejemplo**:
```
x^(√x)² → estados globales
─────────────────────────────
1. x^(√x)²              [ORIGINAL]
2. x^(√x)²              [Multiply exponents en subexpresión → FILTRADO]  
3. x^(x^(1/2·2))        [Power of Power → MOSTRADO]
4. x^x                  [Simplify exponent → MOSTRADO]
```

###### b) Filtrado por Importancia

**CLI Step Verbosity Modes** (comando `steps <mode>`):

| Modo | Filtrado | Display |
|------|----------|---------|
| `none` | Sin pasos | Solo resultado final |
| `succinct` | Medium+ | Compacto (1 línea/paso: expresión global) |
| `normal` | Medium+ | Detallado (regla + local → global) |
| `verbose` | Todos | Detallado incluyendo triviales |

**Nota**: `succinct` y `normal` filtran los mismos pasos (Medium+), pero con diferente presentación.

**Niveles de Importancia** (Single Source of Truth en `step.rs`):

```rust
// En step.rs - Step::importance()
pub enum ImportanceLevel {
    Trivial = 0,  // Add Zero, Mul One, no-ops (before == after)
    Low = 1,      // Collect, Canonicalize, Sort, Evaluate, Identity
    Medium = 2,   // Transformaciones algebraicas estándar
    High = 3,     // Factor, Expand, Integrate, Differentiate
}

impl Step {
    pub fn importance(&self) -> ImportanceLevel {
        // No-ops siempre son triviales
        if self.before == self.after {
            return ImportanceLevel::Trivial;
        }
        
        if self.rule_name.contains("Add Zero")
            || self.rule_name.contains("Mul By One") { ... }
            return ImportanceLevel::Trivial;
        }
        
        if self.rule_name.contains("Collect")
            || self.rule_name.contains("Canonicalize")
            || self.rule_name.contains("Sort") {
            return ImportanceLevel::Low;
        }
        
        if self.rule_name.contains("Factor")
            || self.rule_name.contains("Expand")
            || self.rule_name.contains("Integrate") {
            return ImportanceLevel::High;
        }
        
        ImportanceLevel::Medium  // Default
    }
}
```

**Impacto**: 
- Sin filtrado: 47 pasos
- Con filtrado: 13 pasos meaningful
- Reducción: **72% menos pasos** sin pérdida de información

##### 2. **Layout Compacto** ★

Eliminación del estado "After" redundante:

**Antes**:
```html
<div class="math-expr before">
    <strong>Before (Global):</strong> \[x^2 - 1\]
</div>
<div class="math-expr after">
    <strong>After (Global):</strong> \[(x-1)(x+1)\]
</div>
```

**Después**:
```html
<div class="math-expr">
    <strong>Expression:</strong> \[x^2 - 1\]
</div>
<div class="rule-description">
    <div class="rule-name">\(\text{Factor Polynomial}\)</div>
    <div class="local-change">
        \[x^2 - 1 \rightarrow (x-1)(x+1)\]
    </div>
</div>
```

**Beneficios**:
- 50% menos espacio vertical por paso
- Foco en la transformación (regla + cambio local)
- El "After" se ve implícitamente en el siguiente paso

##### 3. **Renderizado LaTeX Contextual** ★

Dos modos de renderizado según el contexto:

###### a) `LaTeXExpr` (Standard)
```rust
// Para expresiones globales (lectura general)
impl LaTeXExpr {
    pub fn to_latex(&self) -> String {
        // Convierte exponentes fraccionarios a raíces
        // x^(1/2) → \sqrt{x}
        // x^(2/3) → \sqrt[3]{x^2}
    }
}
```

###### b) `LatexNoRoots` (Preserve Exponents)
```rust
// Para reglas de exponentes (claridad pedagógica)
impl LatexNoRoots {
    pub fn to_latex(&self) -> String {
        // Preserva notación de exponentes
        // x^(1/2) → x^{\frac{1}{2}}
        // (x^a)^b → x^{a \cdot b}
    }
}
```

###### Decisión Contextual
```rust
// En timeline.rs
let should_preserve_exponents = step.rule_name.contains("Multiply exponents")
    || step.rule_name.contains("Power of a Power");

let (local_before, local_after) = if should_preserve_exponents {
    (LatexNoRoots::to_latex(before), LatexNoRoots::to_latex(after))
} else {
    (LaTeXExpr::to_latex(before), LaTeXExpr::to_latex(after))
};
```

**Ejemplo**:

| Regla | Renderizado | Razón |
|-------|-------------|-------|
| Multiply exponents | `{x^{1/2}}^2 → x^{1/2·2}` | Clarifica operación en exponentes |
| Rationalize Denominator | `\frac{1}{1+\sqrt{x}}` | Raíces son más legibles |
| Combine Like Terms | `\sqrt{x} + \sqrt{x} → 2\sqrt{x}` | Raíces son estándar |

**Beneficio**: Claridad pedagógica sin sacrificar legibilidad.

###### c) `DisplayContext` - Preservación de Notación de Raíces ★

El sistema `DisplayContext` preserva la notación original de raíces (`√x`) cuando internamente se canonicalizan a potencias fraccionarias (`x^(1/2)`).

**Arquitectura**:

```
┌─────────────────────────────────────────────────────────────┐
│                    Engine Core (sin cambios)                │
│  Rules → Rewrites → Steps                                   │
└────────────────────────────┬────────────────────────────────┘
                             │ steps
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    Display Layer                            │
│  build_display_context(ctx, expr, steps) → DisplayContext   │
│  DisplayExprWithHints renders √x instead of x^(1/2)         │
└─────────────────────────────────────────────────────────────┘
```

**Tipos Base** (`cas_ast/display_context.rs`):

```rust
pub enum DisplayHint {
    AsRoot { index: u32 },  // x^(1/n) → √x
}

pub struct DisplayContext {
    hints: HashMap<ExprId, DisplayHint>,
}
```

**Builder** (`cas_engine/display_context.rs`):

```rust
pub fn build_display_context(
    ctx: &Context,
    original_expr: ExprId,
    steps: &[Step],
) -> DisplayContext {
    let mut display_ctx = DisplayContext::new();
    
    // 1. Escanear expresión original para sqrt() functions
    scan_for_sqrt_hints(ctx, original_expr, &mut display_ctx);
    
    // 2. Capturar hints de steps "Canonicalize Roots"
    for step in steps {
        if step.rule_name.contains("Canonicalize Roots") {
            // step.before = sqrt(x), step.after = x^(1/2)
            display_ctx.insert(step.after, DisplayHint::AsRoot { index });
        }
    }
    display_ctx
}
```

**Rendering** (`DisplayExprWithHints`):

```rust
impl DisplayExprWithHints {
    fn fmt_internal(&self, f: &mut fmt::Formatter, id: ExprId) -> fmt::Result {
        // 1. Detectar sqrt() functions directamente → renderizar como √
        if let Expr::Function("sqrt", args) = self.context.get(id) {
            return write!(f, "√({})", args[0]);
        }
        
        // 2. Consultar hints para Pow que vino de sqrt
        if let Some(DisplayHint::AsRoot { index }) = self.hints.get(id) {
            if let Expr::Pow(base, _) = self.context.get(id) {
                return write!(f, "{}√({})", index, base);
            }
        }
        
        // 3. Formato regular...
    }
}
```

**Flujo Completo**:

| Input | Interno | Display |
|-------|---------|---------|
| `sqrt(x)` function | `Expr::Function("sqrt", [x])` | `√(x)` - detección directa |
| `x^(1/2)` de canonicalización | `Expr::Pow(x, 1/2)` | `√(x)` - vía hints |
| `x^(17/24)` general | `Expr::Pow(x, 17/24)` | `x^(17/24)` - no es raíz |

**Display sugar (v6)**:

El display ahora embellece representaciones canónicas sin tocar el AST:
- `Add(a, Neg(b))` → `a - b` (y términos negativos se ordenan de forma legible)
- `Mul(a, Pow(b, -1))` → `a/b` (con paréntesis conservadores)
- `Pow(x, -1)` → `1/x`

Esto permite preferir internamente C2 (`Add+Neg`, `Mul+Pow(-1)`) sin degradar la UX.
**Beneficios**:
- ✅ **Zero overhead** cuando steps están desactivados
- ✅ **Desacoplado** del engine - reglas no conocen DisplayContext
- ✅ **Extensible** - añadir nuevos hints (ej. `x^(-1)` → `1/x`) es trivial

##### 3b. Sistema de SubSteps Didácticos ★

El sistema de **SubSteps** enriquece los pasos de simplificación con explicaciones detalladas de cálculos ocultos (ej. sumas de fracciones en exponentes).

**Problema**:
```
Input: x^(1/3 + 1/6)
Result: x^(1/2)

Sin SubSteps: El usuario no ve cómo 1/3 + 1/6 = 1/2
```

**Solución** (`cas_engine/didactic.rs`):

```rust
pub struct SubStep {
    pub description: String,    // "Find common denominator: 6"
    pub before_latex: String,   // "1/3 + 1/6"
    pub after_latex: String,    // "2/6 + 1/6"
}

pub struct EnrichedStep {
    pub original: Step,
    pub sub_steps: Vec<SubStep>,
}

pub fn enrich_steps(
    ctx: &Context,
    original_expr: ExprId,
    steps: Vec<Step>,
) -> Vec<EnrichedStep> {
    // Buscar sumas de fracciones en exponentes
    let fraction_sums = find_all_fraction_sums(ctx, original_expr);
    
    // Generar sub-steps con explicaciones detalladas
    for sum in fraction_sums {
        sub_steps.push(SubStep {
            description: format!("Find common denominator: {}", lcm),
            before_latex: format_fractions(&sum.fractions),
            after_latex: format_fractions_with_common_denom(&sum),
        });
        sub_steps.push(SubStep {
            description: "Sum the fractions",
            before_latex: format_fractions_with_common_denom(&sum),
            after_latex: format_result(&sum),
        });
    }
    
    // Adjuntar sub-steps a cada paso (CLI muestra solo una vez)
    steps.map(|s| EnrichedStep { original: s, sub_steps: sub_steps.clone() })
}
```

**Función para Expresiones Sin Pasos**:

```rust
// Para casos como x^(1/3 + 1/6) que se evalúan en parsing
pub fn get_standalone_substeps(ctx: &Context, expr: ExprId) -> Vec<SubStep> {
    find_all_fraction_sums(ctx, expr)
        .into_iter()
        .flat_map(generate_fraction_sum_substeps)
        .collect()
}
```

**Ejemplo de Output**:

```
> x^(1/3 + 1/6) * x^(1/4 + 1/12)
Steps:
1. Combine base and power  [Product of Powers]
      → Find common denominator: 6
        (1/3) + (1/6) → (2/6) + (1/6)
      → Sum the fractions
        (2/6) + (1/6) → (3/6) = (1/2)
      → Find common denominator: 12
        (1/4) + (1/12) → (3/12) + (1/12)
      → Sum the fractions
        (3/12) + (1/12) → (4/12) = (1/3)
   Local: x^(1/2) * x^(1/3) -> x^(5/6)
   Global: x^(5/6)
Result: x^(5/6)
```

**Deduplicación Inteligente**:

```rust
// Evitar sub-steps duplicados o subconjuntos
fn deduplicate_fraction_sums(sums: Vec<FractionInfo>) -> Vec<FractionInfo> {
    // Mantener solo los más completos
    // ej. si tenemos (1/2 + 1/3) y (1/2 + 1/3 + 1/4),
    // solo mostrar (1/2 + 1/3 + 1/4)
}
```

**Integración CLI**:

```rust
// repl.rs - Mostrar sub-steps solo una vez (primer paso visible)
let mut sub_steps_shown = false;
for step in steps {
    if should_show_step(step, verbosity) && !sub_steps_shown {
        for sub in enriched_step.sub_steps {
            println!("      → {}", sub.description);
            println!("        {} → {}", sub.before_latex, sub.after_latex);
        }
        sub_steps_shown = true;
    }
}
```

**Beneficios**:
- ✅ **Didáctico**: Explica cálculos "ocultos" en parsing
- ✅ **Contextual**: Solo aparece cuando hay operaciones relevantes
- ✅ **Sin duplicados**: Lógica de deduplicación inteligente
- ✅ **Modular**: Funciona con o sin pasos de engine

##### 4. **Generación HTML**

```rust
impl TimelineHtml {
    pub fn to_html(&mut self) -> String {
        let filtered = self.filter_by_importance();
        
        for step in filtered {
            // Reconstruir estado global BEFORE
            let global_before = self.reconstruct_global(prev_global, &step.path, step.before);
            
            // Renderizado contextual
            let local_change = self.render_local_change(&step);
            
            // HTML con MathJax
            html.push_str(&format!(r#"
                <div class="step">
                    <div class="step-number">{}</div>
                    <div class="step-content">
                        <h3>{}</h3>
                        <div class="math-expr">
                            \(\textbf{{Expression:}}\)
                            \[{}\]
                        </div>
                        <div class="rule-description">
                            <div class="rule-name">\(\text{{{}}}\)</div>
                            <div class="local-change">
                                \[{}\]
                            </div>
                        </div>
                    </div>
                </div>
            "#, step_num, category, global_before, rule_name, local_change));
        }
        
        html
    }
}
```

**Usage**:
```bash
> timeline 1/(sqrt(x)+1)+1/(sqrt(x)-1)-(2*sqrt(x))/(x-1)
Timeline exported to timeline.html
```

**Generated HTML Features**:
- **MathJax 3**: Renderizado matemático profesional
- **Responsive Design**: Adaptativo a diferentes pantallas
- **Purple Gradient Background**: Estética profesional
- **Color-Coded Sections**: 
  - Expression: Orange border
  - Rule: Purple dashed border on lavender
  - Final Result: Green highlight
- **Step Numbers**: Círculos con gradiente purple
- **Animations**: Fade-in suave al cargar

**Métricas de Efectividad**:
```
Expresión compleja: 1/(sqrt(x)+1)+1/(sqrt(x)-1)-(2*sqrt(x))/(x-1)
─────────────────────────────────────────────────────────────────
Pasos sin filtrado:          47 pasos
Pasos con filtrado global:   28 pasos  (-40%)
Pasos con filtrado completo: 13 pasos  (-72%)
Resultado:                   0         (correcto)
```

**Lista de Reglas con Exponentes Preservados**:
```rust
// Ampliar esta lista según necesidad pedagógica
const PRESERVE_EXPONENT_RULES: &[&str] = &[
    "Multiply exponents",
    "Power of a Power",
    // Agregar más aquí según se identifiquen
];
```

#### **5. Sistema de Debug Parametrizable (tracing)** ★

Sistema profesional de debug logging con **zero overhead** cuando está desactivado.

**Implementación**:
```rust
// canonical_forms.rs, engine.rs, etc.
use tracing::debug;

pub fn is_canonical_form(ctx: &Context, expr: ExprId) -> bool {
    debug!("Checking if canonical: {:?}", ctx.get(expr));
    // ... lógica
}
```

**Ventajas**:
- ✅ **Zero overhead** en producción (compilación optimizada elimina el código)
- ✅ **Control granular** por módulo
- ✅ **No contamina** benchmarks ni tests
- ✅ **Estándar Rust** - compatible con herramientas de observabilidad

**Usage**:
```bash
# Sin debug (default)
cargo test
cargo bench

# Con debug de módulos específicos
RUST_LOG=cas_engine::canonical_forms=debug cargo test

# Muy verbose
RUST_LOG=cas_engine=trace ./target/release/cas_cli
```

**Documentación Completa**: Ver [DEBUG_SYSTEM.md](../DEBUG_SYSTEM.md) para guía detallada de uso.


#### **4. Sistema de Detección de Formas Canónicas** ★

Sistema modularizado para prevenir expansiones innecesarias de expresiones matemáticas elegantes.

##### Problema Resuelto

**Ciclo "Expand-Then-Factor"**: Expresiones ya factorizadas se expandían innecesariamente, creando trazas largas y confusas.

**Ejemplo del Problema**:
```
Input: ((x+1)*(x-1))^2

❌ SIN protección (30 pasos):
1. Distribute → ((-1+x)*1 + (-1+x)*x)^2
2. Binomial Expansion → ...
3-28. (expansión masiva, cleanup, combinar términos)
29. Factor back → (x-1)^2 * (x+1)^2
30. Sort → (x-1)^2 * (x+1)^2

✅ CON protección (1 paso):
1. Factor Polynomial → (x-1)^2 * (x+1)^2
Result: (x-1)^2 * (x+1)^2
```

**Impacto**: **-97% pasos** para expresiones elegantes, manteniendo expansiones educativas útiles.

##### Arquitectura Modularizada

```
┌─────────────────────────────────────┐
│   canonical_forms.rs (ÚNICA FUENTE) │
│                                     │
│  pub fn is_canonical_form() {       │
│      // Toda la lógica aquí         │
│      match expr {                   │
│        Pow(base, exp) => ...        │
│        Mul(l, r) => ...             │
│      }                              │
│  }                                  │
└──────────────┬──────────────────────┘
               │
       ┌───────┴────────┬──────────┬──────────┐
       │                │          │          │
       ▼                ▼          ▼          ▼
  ┌─────────┐   ┌──────────┐ ┌────────┐ ┌────────┐
  │expand.rs│   │algebra.rs│ │poly.rs │ │exp.rs  │
  │         │   │          │ │        │ │        │
  │ Modo    │   │Distribute│ │Binomial│ │Power   │
  │ Normal  │   │  Rule    │ │Expand  │ │Product │
  └─────────┘   └──────────┘ └────────┘ └────────┘
      ↓              ↓            ↓          ↓
   Protege      Protege      Protege    Protege
   expand()     aggressive   aggressive aggressive
                  mode         mode       mode
```

**Beneficios de Diseño**:
- ✅ **DRY**: 0% duplicación de lógica
- ✅ **Consistencia**: Modo normal y agresivo usan la misma función
- ✅ **Extensibilidad**: Agregar patrón → automáticamente aplica a todos los modos
- ✅ **Testabilidad**: Tests centralizados validan todos los escenarios

##### Patrones Canónicos Detectados

###### 1. **Conjugados Elevados**
```rust
// Patrón: ((a+b)*(a-b))^n
is_canonical_form(ctx, ((x+1)*(x-1))^2) → true
```

**Ejemplos protegidos**:
- `((x+1)*(x-1))^2` ✅
- `((x+y)*(x-y))^3` ✅
- `((2x+1)*(2x-1))^5` ✅

###### 2. **Conjugados Sin Potencia**
```rust
// Patrón: (a+b)*(a-b)
is_canonical_form(ctx, (x+y)*(x-y)) → true
```

**Razón**: Ya está en forma de diferencia de cuadrados factorizada.

###### 3. **Productos de Factores Lineales**
```rust
// Patrón: (ax+b)*(cx+d) elevado
is_canonical_form(ctx, ((x+1)*(x+2))^2) → true
```

**Casos NO protegidos** (expansión educativa útil):
- `(x+1)^2` → expande a `x^2+2x+1` ✅
- `(x+1)*(x+2)` → expande a `x^2+3x+2` ✅

##### Implementación

###### Módulo `canonical_forms.rs`

```rust
/// Detecta formas canónicas que no deben expandirse
pub fn is_canonical_form(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        // Caso 1: Producto factorizado elevado
        Expr::Pow(base, exp) => {
            is_product_of_factors(ctx, *base) && is_small_positive_integer(ctx, *exp)
        }
        
        // Caso 2: Conjugados sin elevar
        Expr::Mul(l, r) => {
            is_conjugate(ctx, *l, *r)
        }
        
        _ => false
    }
}

/// Detecta conjugados: (A+B) y (A-B)
fn is_conjugate(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    match (ctx.get(a), ctx.get(b)) {
        (Expr::Add(a1, a2), Expr::Sub(b1, b2)) | 
        (Expr::Sub(b1, b2), Expr::Add(a1, a2)) => {
            // Verificar si A coincide y B es opuesto
            compare_expr(ctx, *a1, *b1) == Ordering::Equal &&
            compare_expr(ctx, *a2, *b2) == Ordering::Equal
        }
        _ => false
    }
}
```

###### Integración en Reglas

**1. `expand.rs` - Punto de Entrada Principal**
```rust
pub fn expand(ctx: &mut Context, expr: ExprId) -> ExprId {
    // CRITICAL: Skip expansion for canonical forms
    if crate::canonical_forms::is_canonical_form(ctx, expr) {
        return expr;
    }
    
    // ... resto de expansión
}
```

**2. `algebra.rs` - DistributeRule**
```rust
define_rule!(DistributeRule, "Distributive Property", |ctx, expr| {
    // Skip canonical forms - even in aggressive mode
    if crate::canonical_forms::is_canonical_form(ctx, expr) {
        return None;
    }
    
    // ... distribución
});
```

**3. `polynomial.rs` - BinomialExpansionRule**
```rust
define_rule!(BinomialExpansionRule, "Binomial Expansion", |ctx, expr| {
    if crate::canonical_forms::is_canonical_form(ctx, expr) {
        return None;
    }
    
    // ... expansión binomial
});
```

**4. `exponents.rs` - PowerProductRule**
```rust
define_rule!(PowerProductRule, "Power of Product", |ctx, expr| {
    if crate::canonical_forms::is_canonical_form(ctx, expr) {
        return None;
    }
    
    // ... distribución de potencias
});
```

##### Modos de Operación

###### Modo Normal
```bash
> ((x+1)*(x-1))^2
Steps:
Result: (x - 1)^2 * (x + 1)^2
# 0 pasos ✅ - Protegido por expand()
```

###### Modo Agresivo
```bash
> simplify ((x+1)*(x-1))^2
Steps (Aggressive Mode):
Result: (x - 1)^2 * (x + 1)^2
# 0 pasos ✅ - Protegido por DistributeRule
```

**Diferencia clave**: Modo agresivo usa `Simplifier::with_default_rules()` que incluye `DistributeRule`. Sin el check en `DistributeRule`, el modo agresivo bypasseaba la protección.

##### Resultados

| Expresión | Modo | Antes | Después | Mejora |
|-----------|------|-------|---------|--------|
| `((x+1)*(x-1))^2` | Normal | 30 | 1 | **-97%** |
| `((x+1)*(x-1))^2` | Agresivo | 20 | 0 | **-100%** |
| `(x+y)*(x-y)` | Normal | 7 | 0 | **-100%** |
| `(x+y)*(x-y)` | Agresivo | 7 | 0 | **-100%** |
| `(x+1)*(x+2)` | Agresivo | 7 | 7 | Correcto ✅ |
| `(x+1)^2` | Normal | 3 | 3 | Correcto ✅ |

##### Tests

```rust
#[test]
fn test_canonical_difference_of_squares_squared() {
    // ((x+1)*(x-1))^2 should be canonical
    assert!(is_canonical_form(&ctx, expr));
}

#[test]
fn test_conjugate_without_power() {
    // (x+y)*(x-y) should be canonical
    assert!(is_canonical_form(&ctx, expr));
}

#[test]
fn test_simple_binomial_not_canonical() {
    // (x+1)^2 should NOT be canonical (educational expansion)
    assert!(!is_canonical_form(&ctx, expr));
}
```

**Cobertura**: 5 tests, 100% passing

##### Extensibilidad

Para agregar nuevos patrones canónicos:

```rust
// Solo modificar canonical_forms.rs
pub fn is_canonical_form(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        // Patrones existentes...
        
        // NUEVO: Agregar tu patrón aquí
        Expr::YourPattern(...) => {
            your_detection_logic(ctx, ...)
        }
        
        _ => false
    }
}
```

**Propagación Automática**:
- ✅ Modo normal
- ✅ Modo agresivo  
- ✅ Todas las reglas de expansión
- ✅ Tests (si usan `is_canonical_form`)

##### Métricas

- **Archivos modificados**: 8
- **Líneas de código**: ~250 (incl. tests)
- **Duplicación**: 0%
- **Llamadas a `is_canonical_form`**: 4 (todas al mismo código)
- **Tests**: 5 unitarios
- **Reducción promedio de pasos**: -90%

---

### Phase 3: Safety Nets (Correctness Hardening)

Esta fase agrupa “redes de seguridad” que aseguran que los refactors de canonicalización/builders/views no reintroducen bugs sutiles.

#### 1) CI Lint: prohibir `ctx.add(Expr::Mul...)` en `src/`

Script: `scripts/lint_no_raw_mul.sh`

- Busca usos directos de `ctx.add(Expr::Mul` en `crates/**/src`
- Obliga a usar `mul2_raw/mul2_simpl` o `MulBuilder`
- Evita reintroducción accidental de patrones frágiles

#### 2) Guard tests de canonicidad y trazas

- `4^(1/2) → 2` (tanto con exponente `Div(1,2)` como `Number(1/2)`)
- tests de integridad del tracer: `step[i].after` debe ser equivalente a `step[i+1].before`
- tests de matrices: `A*B` no se reordena por canonicalización
### CLI Commands

| Command | Description | Phase |
|---------|-------------|-------|
| `profile [cmd]` | Rule profiler (enable/disable/clear) | Phase 2 |
| `visualize <expr>` | Export AST to Graphviz DOT | Phase 2 |
| `dot <expr>` | Alias for visualize | Phase 2 |
| `timeline <expr>` | Export steps to HTML | Phase 2 |

**Autocomplete Support**:
- `profile <TAB>` → `enable`, `disable`, `clear`
- `help <TAB>` → All commands including new ones
- `help profile` → Detailed profiler help

---

### Performance Summary

#### Benchmark Results (Phase 1)

```
                         BEFORE      AFTER       Δ
sum_fractions_10         740µs       387µs      -46.8%
solve_quadratic          47µs        33µs       -30.4%
diff_nested_trig_exp     60µs        47µs       -21.2%
integrate_trig_product   29µs        18µs       -36.0%
expand_binomial          270µs       254µs      -6.1%
```

**Key Insight**: Conditional multi-pass recovered performance baseline **and exceeded it**.

#### Overhead Analysis (Phase 2)

```
Tool              Disabled    Enabled
─────────────────────────────────────
Profiler          0ns        <1ns/rule
AST Visualizer    0ns         N/A (export only)
Timeline HTML     0ns         N/ (export only)
```

**Zero Impact**: Debug tools add no overhead when not in use.

---

## Métricas de Complejidad

- **Tamaño del AST**: ~500 líneas
- **Parser**: ~200 líneas (PEG grammar)
- **Engine**: ~3250 líneas
  - Rules: ~1800 líneas
  - Orchestrator: ~300 líneas
  - Pattern Detection ★: ~400 líneas (NEW 2025-12)
  - Core simplification: ~750 líneas
- **CLI**: ~1200 líneas
- **Debug Tools**: ~560 líneas (Phase 2)
- **Total proyecto**: ~5750 líneas (+750 from pattern detection)

**Número de reglas**: ~75 reglas activas (incluyendo 2 nuevas Pythagorean directas)

#### Gramática Soportada

- **Operadores**: `+`, `-`, `*`, `/`, `^`, `!` (factorial)
- **Funciones**: `sin`, `cos`, `tan`, `ln`, `log`, `sqrt`, `abs`, etc.
- **Agrupación**: Paréntesis `()`
- **Números**: Enteros, decimales, fracciones (`1/2`)
- **Variables**: Letras (`x`, `y`, `alpha`, etc.)
- **Constantes**: `pi`, `e`, `infinity`

#### Flujo de Parsing

```
Input String  →  [Pest Parser]  →  Pairs  →  [AST Builder]  →  ExprId
```

**Ejemplo**:
```text
"x^2 + 2*x + 1"
  ↓
Add(Add(Pow(Var("x"), Num(2)), Mul(Num(2), Var("x"))), Num(1))
```

---

### 3. `cas_engine` - Motor de Simplificación

El componente más complejo del sistema. Implementa el sistema de reglas y la orquestación.

#### 3.1. Rule - Sistema de Reglas

##### Trait `Rule`

**ACTUALIZADO 2025-12**: Agregado parámetro `ParentContext` para pattern detection.

```rust
pub trait Rule {
    fn name(&self) -> &str;
    fn target_types(&self) -> Option<Vec<&'static str>>;
    
    // NEW SIGNATURE: Added parent_ctx parameter
    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        parent_ctx: &ParentContext  // ← NEW: Context from parent
    ) -> Option<Rewrite>;
}

pub struct Rewrite {
    pub new_expr: ExprId,
    pub description: String,
}
```

**Cambio Critical**:
- `parent_ctx` permite a las reglas acceder a:
  - `pattern_marks`: Expresiones protegidas por pattern detection
  - `ancestors`: Chain de expresiones padre (si se necesita)
  
**Backward Compatibility**: Reglas que no necesitan context pueden ignorar el parámetro.

##### Tipos de Reglas

Las reglas se organizan por categorías:

| Categoría | Archivo | Ejemplos |
|-----------|---------|----------|
| **Aritmética** | `arithmetic.rs` | `AddZeroRule`, `MulOneRule`, `CombineConstantsRule` |
| **Canonicalización** | `canonicalization.rs` | `CanonicalizeAddRule`, `CanonicalizeMulRule` |
| **Exponentes** | `exponents.rs` | `ProductPowerRule`, `EvaluatePowerRule` |
| **Álgebra** | `algebra.rs` | `CombineLikeTermsRule`, `DistributeRule`, `FactorRule` |
| **Trigonometría** | `trigonometry.rs` | `PythagoreanIdentityRule`, `SecTanPythagoreanRule` ★, `CscCotPythagoreanRule` ★, `DoubleAngleRule`, `TanToSinCosRule` (con guard) ★ |
| **Logaritmos** | `logarithms.rs` | `LogProductRule`, `LogPowerRule` |
| **Polinomios** | `polynomial.rs` | `ExpandRule`, `FactorRule` |
| **Cálculo** | `calculus.rs` | `IntegrateRule`, `DiffRule` |
| **Teoría de Números** | `number_theory.rs` | `GcdRule`, `FactorizeRule` |

##### Macro `define_rule!`

Simplifica la creación de reglas:

```rust
define_rule!(
    AddZeroRule,
    "Add Zero",
    |ctx, expr| {
        if let Expr::Add(lhs, rhs) = ctx.get(expr) {
            if matches!(ctx.get(*rhs), Expr::Number(n) if n.is_zero()) {
                return Some(Rewrite {
                    new_expr: *lhs,
                    description: "x + 0 = x".to_string(),
                });
            }
        }
        None
    }
);
```

#### 3.2. Simplifier - Motor Principal

```rust
pub struct Simplifier {
    pub context: Context,
    rules: HashMap<String, Vec<Rc<dyn Rule>>>, // Reglas específicas por tipo
    global_rules: Vec<Rc<dyn Rule>>,           // Reglas globales
    disabled_rules: HashSet<String>,           // Reglas deshabilitadas
    pub enable_polynomial_strategy: bool,
}
```

##### Registro de Reglas

Las reglas se registran por tipo de expresión (`Add`, `Mul`, etc.) o globalmente:

```rust
impl Simplifier {
    pub fn add_rule(&mut self, rule: Box<dyn Rule>) {
        if let Some(targets) = rule.target_types() {
            // Regla específica por tipo
            for target in targets {
                self.rules.entry(target).or_default().push(rule.clone());
            }
        } else {
            // Regla global (se aplica a todo)
            self.global_rules.push(rule);
        }
    }
}
```

**Ventaja**: Las reglas específicas solo se evalúan para expresiones relevantes.

##### Algoritmo de Simplificación

```rust
fn transform_expr_recursive(&mut self, id: ExprId) -> ExprId {
    // 1. Simplificar hijos (bottom-up)
    let simplified_children = simplify_children(id);
    
    // 2. Aplicar reglas específicas
    for rule in specific_rules {
        if let Some(rewrite) = rule.apply(ctx, simplified_children) {
            return transform_expr_recursive(rewrite.new_expr); // Recursión
        }
    }
    
    // 3. Aplicar reglas globales
    for rule in global_rules {
        if let Some(rewrite) = rule.apply(ctx, simplified_children) {
            return transform_expr_recursive(rewrite.new_expr);
        }
    }
    
    simplified_children // No hay cambios
}
```

**Complejidad**: 
- Caso promedio: `O(n * r)` donde `n` es el tamaño del AST y `r` el número de reglas aplicables
- Caso peor: `O(n² * r)` si hay muchas re-simplificaciones

##### Caché

```rust
cache: HashMap<ExprId, ExprId> // Evita re-simplificar subexpresiones
```

Reduce complejidad en casos con subexpresiones compartidas.

#### 3.3. Orchestrator - Orquestación de Simplificación

```rust
pub struct Orchestrator {
    pub max_iterations: usize,              // Límite de seguridad (default: 10)
    pub enable_polynomial_strategy: bool,
}
```

**Responsabilidad**: Coordinar múltiples estrategias de simplificación y gestionar el flujo multi-pass.

##### Pipeline de Simplificación

```
Input Expression
    ↓
1. Initial Collection (Normalize)
    ↓
2. Multi-Pass Rule Application ← ★ NUEVO: Iteración hasta convergencia
    ↓
3. Polynomial Strategy (si aplica)
    ↓
4. Final Collection (Ensure canonical form)
    ↓
5. Step Optimization
    ↓
Result + Steps
```

##### **Multi-Pass Simplification** ★

**Problema Resuelto**: Transformaciones como `RationalizeDenominatorRule` crean nuevas oportunidades de simplificación que no eran visibles en el primer pase.

**Ejemplo**:
```
1/(sqrt(x)+1) + 1/(sqrt(x)-1) - 2*sqrt(x)/(x-1)
  ↓ [Pass 1: Rationalize]
(1-sqrt(x))/(1-x) + (-1-sqrt(x))/(1-x) - 2*sqrt(x)/(x-1)
  ↓ [Pass 2: Add fractions with same denominator]  ← Sin multi-pass, esto no pasaba
-2*sqrt(x)/(1-x) - 2*sqrt(x)/(x-1)
  ↓ [Pass 3: Recognize opposite denominators]
0
```

**Implementación**:
```rust
pub fn simplify(&self, expr: ExprId, simplifier: &mut Simplifier) -> (ExprId, Vec<Step>) {
    // ... Initial collection ...
    
    // Multi-Pass Rule Application
    let max_passes = 5;
    let mut pass_count = 0;
    
    loop {
        let (simplified, rule_steps) = simplifier.apply_rules_loop(current);
        
        // Check if anything changed
        let changed = simplified != current || 
                     compare_expr(&simplifier.context, simplified, current) != Ordering::Equal;
        
        if changed {
            steps.extend(rule_steps);
            current = simplified;
            pass_count += 1;
            
            if pass_count >= max_passes {
                break; // Safety: prevent infinite loops
            }
        } else {
            break; // Converged
        }
    }
    
    // ... Polynomial strategy, final collection ...
}
```

**Características**:
- **Convergencia automática**: Itera hasta que no hay cambios
- **Safeguard**: Límite máximo de 5 iteraciones
- **Performance**: Casos simples terminan en 1 iteración
- **Casos complejos**: Típicamente 2-3 iteraciones

**Métricas Observadas**:
| Expresión | Iteraciones | Resultado |
|-----------|-------------|-----------|
| `1/(x-1) + 1/(1-x)` | 1 | `0` |
| `2/(x-1) + 3/(1-x)` | 1 | `-1/(x-1)` |
| `1/(sqrt(x)+1) + 1/(sqrt(x)-1)` | 2 | `2*sqrt(x)/(x-1)` |
| **Bridge case completo** | 3 | `0` ✓ |

##### Estrategias Implementadas

1. **Polynomial Strategy** (`strategies::polynomial_strategy`):
   - Detecta expresiones polinómicas
   - Aplica `collect` para agrupar términos
   - Evita expansiones innecesarias según heurísticas:
     - Skip si exponentes > 6 (evita explosión)
     - Skip si > 4 divisiones (no es polinomio simple)
     - Skip si no tiene Add/Sub (no hay términos que combinar)

2. **Global Simplification Loop**:
   - Aplica reglas hasta alcanzar un punto fijo
   - Detecta ciclos infinitos (límite de iteraciones)

**Flujo**:
```
expr → [Multi-Pass Loop] → [Polynomial Strategy] → [Final Collection] → result
       (hasta convergencia)   (si es polinomio)      (normalizar)
```

#### 3.4. Step - Trazabilidad

```rust
pub struct Step {
    pub description: String,
    pub rule_name: String,
    pub before: ExprId,
    pub after: ExprId,
    pub path: Vec<PathStep>, // Ubicación en el árbol
}
```

**`PathStep`** describe la ubicación de la transformación:
```rust
pub enum PathStep {
    Left,       // Hijo izquierdo de Add/Sub/Mul/Div/Pow
    Right,      // Hijo derecho
    Inner,      // Dentro de Neg
    Base,       // Base de Pow
    Exponent,   // Exponente de Pow
    Arg(usize), // Argumento de función
}
```

**Uso**: Permite mostrar *dónde* se aplicó cada regla en la expresión.

---

### 4. `cas_cli` - Interfaz de Línea de Comandos

**Responsabilidad**: Interacción con el usuario.

#### Estructura

```rust
pub struct Repl {
    simplifier: Simplifier,
    verbosity: Verbosity,
    config: CasConfig,
}

pub enum Verbosity {
    None,    // Sin pasos
    Low,     // Solo cambios globales (sin reglas "ruidosas")
    Normal,  // Cambios locales + globales filtrados
    Verbose, // Todos los pasos
}
```

#### Comandos Soportados

| Comando | Descripción |
|---------|-------------|
| `simplify <expr>` | Simplifica la expresión |
| `eval <expr>` | Igual que `simplify` |
| `solve <eq>, <var>` | Resuelve ecuación/inecuación |
| `subst <expr>, <var>=<val>` | Sustituye variable por valor |
| `equiv <expr1>, <expr2>` | Verifica equivalencia |
| `factor <expr>` | Factoriza polinomio |
| `expand <expr>` | Expande expresión |
| `steps <level>` | Cambia verbosidad (`normal`, `low`, `verbose`, `none`) |
| `config <cmd>` | Gestión de configuración |
| `help` | Muestra ayuda |

#### Autocompletado

Implementado con `rustyline`:
- Comandos de nivel 1: `simplify`, `solve`, etc.
- Comandos de nivel 2: `steps normal`, `config list`, etc.
- Funciones matemáticas: `sin`, `cos`, `sqrt`, etc.

#### Configuración Persistente

```toml
# config.toml
[rules]
distribute = true
trig_double_angle = true
root_denesting = true
# ... etc
```

Permite habilitar/deshabilitar reglas sin recompilar.

---

## Arquitectura del Sistema de Reglas

### Categorización de Reglas

#### 1. Reglas de Identidad
**Simplificación trivial sin cambio semántico**:
- `x + 0 = x` (`AddZeroRule`)
- `x * 1 = x` (`MulOneRule`)
- `x * 0 = 0` (`MulZeroRule`)
- `x^1 = x` (`IdentityPowerRule`)

#### 2. Reglas de Canonicalización
**Normalización de forma**:
- Ordenar términos: `b + a → a + b` (orden lexicográfico)
- Aplanar asociativas: `(a + b) + c → a + (b + c)`
- Convertir `sqrt(x)` a `x^(1/2)`

**Importancia**: Facilita pattern matching y comparación.

#### 3. Reglas Aritméticas
**Evaluación de constantes**:
- `CombineConstantsRule`: `2 + 3 → 5`
- `EvaluatePowerRule`: `2^3 → 8`
- División: `6 / 2 → 3`

#### 4. Reglas Algebraicas
**Transformaciones simbólicas**:
- `CombineLikeTermsRule`: `2x + 3x → 5x`
- `DistributeRule`: `a(b + c) → ab + ac`
- `FactorRule`: `x^2 - 1 → (x-1)(x+1)`
- `SimplifyFractionRule`: `(x^2 - 1)/(x - 1) → x + 1`

##### **AddFractionsRule - Suma de Fracciones Mejorada** ★

**Problema**: Sumar fracciones con denominadores algebraicos complejos.

**Casos Soportados**:

1. **Denominadores Opuestos Estructurales**:
   - `(a - b)` vs `(b - a)` → Se detectan como opuestos
   - `(-a + b)` vs `(a - b)` → Maneja formas con `Neg`
   - `(Number(-n) + x)` vs `(Number(n) - x)` → Números con signos opuestos
   - `Add(Number(-n), x)` vs `Add(Number(n), Neg(x))` → Formas mixtas de Add

2. **Denominadores Iguales**:
   - Se detectan mediante `compare_expr` estructural
   - Bypass de complejidad cuando denominadores son iguales

**Algoritmo**:
```rust
// 1. Detectar relación entre denominadores
let (n2, d2, opposite_denom, same_denom) = {
    // Verificar si son exactamente iguales
    if compare_expr(ctx, d1, d2) == Ordering::Equal {
        (n2, d1, false, true)  // Mismo denominador
    }
    // Verificar si son opuestos
    else if are_denominators_opposite(ctx, d1, d2) {
        (ctx.add(Expr::Neg(n2)), d1, true, false)  // Negar n2
    }
    else {
        (n2, d2, false, false)  // Diferentes
    }
};

// 2. Calcular común denominador (LCM)
let (common_den, mult1, mult2) = calculate_lcm(d1, d2);

// 3. Construir nueva expresión
let new_expr = (n1*mult1 + n2*mult2) / common_den;

// 4. Aplicar regla si:
//    - Denominadores opuestos/iguales (siempre beneficioso), O
//    - Complejidad no aumenta, O  
//    - Simplifica y complejidad < 1.5x
if opposite_denom || same_denom || 
   new_complexity <= old_complexity ||
   (simplifies && new_complexity < old_complexity * 1.5) {
    return Some(Rewrite { new_expr, ... });
}
```

**Función `are_denominators_opposite`**:
```rust
fn are_denominators_opposite(ctx: &Context, e1: ExprId, e2: ExprId) -> bool {
    match (ctx.get(e1), ctx.get(e2)) {
        // Caso 1: (a - b) vs (b - a)
        (Expr::Sub(l1, r1), Expr::Sub(l2, r2)) => {
            compare_expr(ctx, *l1, *r2) == Equal &&
            compare_expr(ctx, *r1, *l2) == Equal
        }
        
        // Caso 2: (-a + b) vs (a - b)
        (Expr::Add(l1, r1), Expr::Sub(l2, r2)) => {
            if let Expr::Neg(neg_l1) = ctx.get(*l1) {
                compare_expr(ctx, *neg_l1, *l2) == Equal &&
                compare_expr(ctx, *r1, *r2) == Equal
            } else { false }
        }
        
        // Caso 3: Add(Number(-n), x) vs Add(Number(n), Neg(x))
        (Expr::Add(l1, r1), Expr::Add(l2, r2)) => {
            if let (Expr::Number(n1), Expr::Number(n2)) = (ctx.get(*l1), ctx.get(*l2)) {
                if let Expr::Neg(neg_r2) = ctx.get(*r2) {
                    n1 == &(-n2.clone()) && 
                    compare_expr(ctx, *r1, *neg_r2) == Equal
                } else { false }
            } else { false }
        }
        
        _ => false
    }
}
```

**Ejemplo Complejo - "El Puente Conjugado"**:
```
Input: 1/(sqrt(x)+1) + 1/(sqrt(x)-1) - 2*sqrt(x)/(x-1)

Pass 1 [Rationalize denominators]:
  → (1-sqrt(x))/(1-x) + (-1-sqrt(x))/(1-x) - 2*sqrt(x)/(x-1)

Pass 2 [Detect same denominators (1-x)]:
  → -2*sqrt(x)/(1-x) - 2*sqrt(x)/(x-1)
     ^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^
         same_denom=true     opposite_denom=true

Pass 3 [Combine and recognize opposites]:
  → 0
```

**Métricas de Éxito**:
- ✅ Casos polinomiales simples: 100% éxito
- ✅ Casos con raíces (1 nivel): 100% éxito  
- ✅ Casos complejos (nested, bridge): 100% éxito

##### Optimización: Bypass de Complejidad

**Motivación**: Combinar fracciones con denominadores iguales u opuestos es **siempre matemáticamente beneficioso**, incluso si aumenta temporalmente la complejidad del AST.

**Implementación**:
```rust
// ANTES: Solo aplicaba si complejidad no aumentaba
if new_complexity <= old_complexity { apply(); }

// AHORA: Bypass si denominadores especiales
if opposite_denom || same_denom || new_complexity <= old_complexity {
    apply(); // ← SIEMPRE se aplica si denominadores son relevantes
}
```

**Justificación**: Una fracción como `a/d + b/d` debe SIEMPRE simplificarse a `(a+b)/d`, incluso si `(a+b)` es más complejo que mantener separado.

#### 5. Reglas Especializadas
**Dominios específicos**:
- Trigonometría: Identidades pitagóricas, ángulos dobles
- Logaritmos: Propiedades de productos/cocientes
- Raíces: Simplificación de radicales, denesting

### Diseño de Reglas Complejas

#### Ejemplo: `CombineLikeTermsRule`

**Objetivo**: `2x + 3x → 5x`

**Algoritmo**:
1. Identificar términos de suma: `a₁ + a₂ + ... + aₙ`
2. Agrupar por "término base":
   - `2*x` → base: `x`, coef: `2`
   - `3*x` → base: `x`, coef: `3`
3. Sumar coeficientes: `(2+3) = 5`
4. Reconstruir: `5*x`

**Implementación**:
```rust
// Agrupar términos por base
let mut groups: HashMap<ExprId, Vec<ExprId>> = HashMap::new();
for term in terms {
    let (coef, base) = extract_coefficient(term);
    groups.entry(base).or_default().push(coef);
}

// Combinar coeficientes
for (base, coefs) in groups {
    let sum_coef = add_all(coefs);
    result.push(Mul(sum_coef, base));
}
```

#### Ejemplo: `EvaluatePowerRule` (con simplificación parcial de raíces)

**Objetivo**: `sqrt(8) → 2*sqrt(2)`

**Algoritmo**:
1. Detectar exponente fraccionario: `8^(1/2)`
2. Factorizar base: `8 = 2^3`
3. Extraer factores perfectos:
   - `2^3 = 2^(2*1) * 2^1`
   - `(2^2)^(1/2) * 2^(1/2)`
4. Simplificar: `2 * 2^(1/2)`

**Implementación** (versión simplificada):
```rust
fn extract_root_factor(n: &BigInt, k: u32) -> (BigInt, BigInt) {
    // Factorización por división de prueba
    for prime in [2, 3, 5, 7, ...] {
        let mut exp = 0;
        while n % prime == 0 {
            exp += 1;
            n /= prime;
        }
        let out_exp = exp / k;
        let in_exp = exp % k;
        outside *= prime.pow(out_exp);
        inside *= prime.pow(in_exp);
    }
    (outside, inside)
}
```

---

## Orquestación y Estrategias

### Problema: Explosión Combinatoria

Al aplicar reglas libremente, el número de estados intermedios puede crecer exponencialmente.

**Ejemplo**:
```text
(x + 1)(x + 2)
  ├─ Distribuir primera: x(x+2) + 1(x+2)
  │   └─ Distribuir de nuevo: x² + 2x + x + 2
  │       └─ Combinar: x² + 3x + 2
  └─ Distribuir segunda: (x+1)x + (x+1)2
      └─ ... (camino alternativo)
```

### Solución: Orchestrator + Strategies

#### Polynomial Strategy

Detecta patrones polinómicos y aplica secuencia óptima:

```rust
pub fn polynomial_strategy(expr: ExprId, simplifier: &mut Simplifier) -> ExprId {
    if is_polynomial(expr) {
        // 1. Expandir si es necesario
        let expanded = apply_expand_if_needed(expr);
        
        // 2. Canonicalizar (ordenar términos)
        let canonical = canonicalize(expanded);
        
        // 3. Combinar términos semejantes
        let combined = apply_combine_like_terms(canonical);
        
        // 4. Factorizar si es posible
        let factored = try_factor(combined);
        
        return factored;
    }
    expr
}
```

**Ventaja**: Evita caminos redundantes, mejora rendimiento.

### Step Optimization

Filtra pasos "ruidosos" para presentación:

```rust
fn should_show_step(step: &Step, verbosity: Verbosity) -> bool {
    match verbosity {
        Verbosity::Verbose => true,
        Verbosity::Normal | Verbosity::Low => {
            // Ocultar canonicalización, identidades triviales
            !step.rule_name.starts_with("Canonicalize") &&
            step.rule_name != "Add Zero" &&
            step.rule_name != "Multiply by One"
        }
        Verbosity::None => false,
    }
}
```
#### Tracer / Steps: reconstrucción estable con `add_raw`

Los steps no se construyen “re-canonicalizando” el AST, sino **reconstruyendo** el árbol en un `Context` usando un *path* al subárbol reescrito.

Bug encontrado (v5→v6):
- `reconstruct_at_path` usaba `context.add()` y, por tanto, podía **swapear** operandos en `Add/Mul`
- esto corrompía el `After` (cambios de signo / orden) y hacía que el step siguiente no encajase con el anterior

Fix:
- el reconstructor (`reconstruct_recursive`) usa **`context.add_raw(...)`** en todas las reconstrucciones.
- el motor puede canonicalizar internamente; la traza muestra la estructura exacta del rewrite aplicado.

Esto desacopla “canonicidad interna” de “presentación de steps”.

---

## Flujo de Datos

### Diagrama de Flujo Completo

```
┌─────────────┐
│ Usuario CLI │
└──────┬──────┘
       │ "sec²(x) - tan²(x)"
       ▼
┌─────────────────┐
│  cas_parser     │ ←Gramática PEG (pest)
│  parse(input)   │
└────────┬────────┘
         │ ExprId
         ▼
┌─────────────────────────┐
│  cas_engine             │
│  Simplifier::simplify() │
└────────┬────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│  Orchestrator::simplify()                │
│  ┌────────────────────────────────────┐  │
│  │ 1. PRE-ANALYSIS ★ (NEW)            │  │
│  │    PatternScanner::scan_and_mark() │  │
│  │    → Creates PatternMarks          │  │
│  │    → O(n) one-time scan            │  │
│  └────────────────────────────────────┘  │
│  ┌────────────────────────────────────┐  │
│  │ 2. MULTI-PASS SIMPLIFICATION       │  │
│  │    Simplifier::apply_rules_loop(   │  │
│  │        expr, &pattern_marks)       │  │
│  │    → Passes marks to transformer   │  │
│  └────────────────────────────────────┘  │
│  ┌────────────────────────────────────┐  │
│  │ 3. POLYNOMIAL STRATEGY (optional)  │  │
│  └────────────────────────────────────┘  │
│  ┌────────────────────────────────────┐  │
│  │ 4. FINAL COLLECTION               │  │
│  └────────────────────────────────────┘  │
└────────┬─────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  LocalSimplificationTransformer      │
│  - initial_parent_ctx: ParentContext │
│    (with pattern_marks)              │
│  - Bottom-up recursion               │
│  - Apply specific rules              │
│  - Apply global rules                │
│  - Record steps                      │
└────────┬─────────────────────────────┘
         │
         │ (iteración hasta punto fijo)
         │
         ▼
┌─────────────────┐
│  Simplified     │
│  ExprId + Steps │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Display        │
│  - Format expr  │
│  - Show steps   │
└─────────────────┘
```

### Ejemplo Concreto: `simplify 2x + 3x`

```
Input: "2x + 3x"
  ↓ [Parser]
Add(Mul(Num(2), Var("x")), Mul(Num(3), Var("x")))
  ↓ [Simplifier]
  ↓ [Bottom-up: no cambios en hojas]
  ↓ [Apply CombineLikeTermsRule]
    - Detecta: base = Var("x"), coefs = [2, 3]
    - Suma: 2 + 3 = 5
    - Construye: Mul(Num(5), Var("x"))
  ↓ [Resultado]
Mul(Num(5), Var("x"))
  ↓ [Display]
"5 * x"
  
Steps:
1. Combine Like Terms [Global Combine Like Terms]
   Local: 2 * x + 3 * x → 5 * x
   Global: 5 * x
```

---

## Puntos de Extensión

### 1. Añadir Nueva Regla

**Paso 1**: Implementar la regla

```rust
// En crates/cas_engine/src/rules/mi_modulo.rs
define_rule!(
    MiNuevaRegla,
    "Mi Descripción",
    |ctx, expr| {
        // Lógica de transformación
        if let Expr::... = ctx.get(expr) {
            return Some(Rewrite {
                new_expr: ...,
                description: "...".to_string(),
            });
        }
        None
    }
);
```

**Paso 2**: Registrar en el simplificador

```rust
// En crates/cas_engine/src/rules/mod.rs
pub fn register(simplifier: &mut Simplifier) {
    simplifier.add_rule(Box::new(MiNuevaRegla));
}
```

### 2. Añadir Nueva Función Matemática

**Paso 1**: Actualizar gramática del parser

```pest
// En crates/cas_parser/grammars/math.pest
function = {
    // ...
    | ("mi_func" ~ "(" ~ expr ~ ")")
}
```

**Paso 2**: Crear regla de evaluación

```rust
define_rule!(
    EvaluateMiFuncRule,
    "Evaluate mi_func",
    |ctx, expr| {
        if let Expr::Function(name, args) = ctx.get(expr) {
            if name == "mi_func" && args.len() == 1 {
                // Lógica de evaluación
                return Some(Rewrite { ... });
            }
        }
        None
    }
);
```

### 3. Añadir Estrategia de Simplificación

```rust
// En crates/cas_engine/src/strategies.rs
pub fn mi_estrategia(expr: ExprId, simp: &mut Simplifier) -> ExprId {
    // Detectar patrón
    if matches_pattern(expr) {
        // Aplicar secuencia de transformaciones
        let paso1 = aplicar_regla_A(expr);
        let paso2 = aplicar_regla_B(paso1);
        return paso2;
    }
    expr
}
```

**Uso en Orchestrator**:
```rust
// En orchestrator.rs
if self.enable_mi_estrategia {
    expr = strategies::mi_estrategia(expr, simplifier);
}
```

### 4. Añadir Comando CLI

```rust
// En crates/cas_cli/src/repl.rs
fn handle_command(&mut self, line: &str) {
    let parts: Vec<&str> = line.split_whitespace().collect();
    match parts[0] {
        "mi_comando" => {
            // Lógica del comando
            self.handle_mi_comando(&parts[1..]);
        }
        // ...
    }
}
```

---

## Optimizaciones

### 1. Expression Interning

**Problema**: Subexpresiones duplicadas consumen memoria.

**Solución**: Pool central de expresiones con deduplicación.

```rust
// Antes
let expr1 = Expr::Add(x, y);
let expr2 = Expr::Add(x, y); // Duplicado en memoria

// Después (con interning)
let id1 = ctx.add(Expr::Add(x, y));
let id2 = ctx.add(Expr::Add(x, y)); // id1 == id2 (misma referencia)
```

**Ahorro**: En expresiones grandes con subexpresiones repetidas, reduce uso de memoria hasta 50%.

### 2. Caché de Simplificación

**Problema**: Re-simplificar subexpresiones ya procesadas.

**Solución**: `cache: HashMap<ExprId, ExprId>`

```rust
if let Some(&cached) = self.cache.get(&id) {
    return cached; // O(1)
}
// Simplificar...
self.cache.insert(id, result);
```

**Ganancia**: En expresiones con alta compartición, reduce tiempo hasta 10x.

### 3. Reglas Específicas por Tipo

**Problema**: Evaluar todas las reglas para cada expresión.

**Solución**: Indexar reglas por tipo de expresión (`Add`, `Mul`, etc.).

```rust
// Solo evalúa reglas relevantes para Mul
if let Some(rules) = self.rules.get("Mul") {
    for rule in rules {
        // ...
    }
}
```

**Ganancia**: Reduce número de evaluaciones de reglas entre 5-10x.

### 4. Early Return en Reglas

**Patrón**: Retornar `None` lo antes posible si la regla no aplica.

```rust
define_rule!(
    MiRegla,
    "...",
    |ctx, expr| {
        // Early return
        if !matches!(ctx.get(expr), Expr::Add(_, _)) {
            return None; // ← Evita trabajo innecesario
        }
        // Lógica compleja...
    }
);
```

---

## Conclusión

### Fortalezas del Diseño

1. **Modularidad**: Fácil añadir/quitar reglas
2. **Claridad**: Separación clara entre AST, Engine, y UI
3. **Trazabilidad**: Registro detallado de transformaciones
4. **Rendimiento**: Optimizaciones (interning, caché, indexación)
5. **Extensibilidad**: Múltiples puntos de extensión bien definidos

### Áreas de Mejora Futura

1. ~~**Multi-Pass Orchestration**~~: ✅ **COMPLETADO** - Implementado sistema de iteración hasta convergencia
2. ~~**Fraction Simplification**~~: ✅ **COMPLETADO** - Detecta denominares opuestos/iguales con bypass de complejidad
3. ~~**Pattern Detection & Context-Aware Rules**~~: ✅ **COMPLETADO (2025-12)** - Sistema completo de pre-analysis, PatternMarks, ParentContext threading, y reglas Pythagorean
4. **Paralelización**: Simplificar subexpresiones independientes en paralelo
5. **Heurísticas Avanzadas**: ML para predecir mejor secuencia de reglas
6. **Verificación Formal**: Probar corrección de reglas con SMT solvers
7. **Rendimiento**: Compilación JIT de expresiones frecuentes
8. **UI Gráfica**: Visualización de árbol de simplificación
9. **Pruebas de Equivalencia**: Mejorar `equiv` command con más estrategias
10. **Extensión de Pattern Detection**: Más familias de identidades protegidas (ej: sum-to-product trig, logaritmos)
11. **Refactorizar Simplificador a Iterativo**: Convertir la recursión a un stack explícito para eliminar stack overflows (ver sección Stress Testing)

---

## 8. Stress Testing Infrastructure ★ (Added 2025-12)

### Propósito

Sistema para identificar cuellos de botella en la orquestación de reglas y prevenir stack overflows.

### Perfiles de Complejidad

| Perfil   | Depth | Size | Items | Uso |
|----------|-------|------|-------|-----|
| SAFE     | 2     | 8    | 4     | CI/CD, nunca desborda |
| NORMAL   | 3     | 15   | 6     | Desarrollo |
| STRESS   | 4     | 20   | 10    | Detecta problemas |
| EXTREME  | 5     | 30   | 15    | Debugging profundo |

### Herramientas

**Ubicación**: `crates/cas_engine/tests/`

| Archivo | Propósito |
|---------|-----------|
| `strategies/mod.rs` | Perfiles (SAFE/NORMAL/STRESS/EXTREME) y generadores |
| `stress_test.rs` | Tests con detección de overflow y profiler |
| `property_tests.rs` | Tests de propiedades (idempotencia, etc.) |

### Uso

```bash
# Ejecutar stress test
STRESS_PROFILE=STRESS RUST_MIN_STACK=16777216 cargo test --package cas_engine --test stress_test -- --nocapture
```

### Resumen de Test

Al terminar, se muestra un resumen completo:

```
═══════════════════════════════════════════════════════════════════
                    STRESS TEST SUMMARY
═══════════════════════════════════════════════════════════════════
Profile: STRESS (depth=4, size=20, items=10)

📊 TEST RESULTS:
   Passed:       20 / 20 (100.0%)
   ⚠️  Overflows:    2              ← Solo si hay overflows

📈 RULE STATISTICS:
   Total simplifications:     20
   Total rule applications:   47
   Average rules/expression:  2.4

   Top 10 Rules:
      Combine Constants                          15 ( 31.9%)
      Canonicalize Addition                      12 ( 25.5%)
      ...

🔥 MOST EXPENSIVE EXPRESSION:
   Steps: 23
   Expr:  sin(x)^8 - cos(x)^8

⚠️  STACK OVERFLOW EXPRESSIONS (2):    ← Solo si hay overflows
   1. (sin(x)^2 + cos(x)^2)^10 / ((tan(x) + cot(x))^5)
   💡 TIP: Copy to test_stress_single() for debugging.
═══════════════════════════════════════════════════════════════════
```

### Workflow de Debugging

1. Ejecutar stress test → identificar expresiones con overflow
2. Copiar expresión problemática a `test_stress_single()`
3. Ejecutar con `RUST_LOG=debug` para ver paso a paso

```bash
RUST_LOG=debug cargo test test_stress_single -- --nocapture
```

**Ver MAINTENANCE.md sección 6** para documentación completa y ejemplos de código.

### Referencias Útiles

- **Código fuente**: `/crates/cas_engine/src/`
- **Design Decision**: Para polinomios multivariables, el sistema conservadoramente retorna GCD=1 en lugar de fallar, ya que la implementación actual solo soporta polinomios univariados.
- **Tests**: `/crates/cas_engine/tests/`
- **Documentación**: `README.md`, `MAINTENANCE.md`
- **Ejemplos**: `/crates/cas_cli/` (comandos interactivos)

---

*Documento generado para ExpliCAS v0.1.0*

---

## Matrix Operations

### Overview
El módulo de matrices proporciona operaciones básicas de álgebra lineal con soporte para matrices de cualquier tamaño. La implementación está diseñada para integrarse seamlessly con el sistema de reglas del simplificador.

### Architecture

**Files:**
- `crates/cas_engine/src/matrix.rs` - Core matrix implementation
- `crates/cas_engine/src/rules/matrix_ops.rs` - Simplification rules
- `crates/cas_cli/src/repl.rs` - CLI command handlers

### Matrix Representation

Las matrices se representan en el AST como:

```rust
Expr::Matrix {
    rows: usize,
    cols: usize,
    data: Vec<ExprId>,  // Row-major order
}
```

**Row-major order**: Los elementos se almacenan fila por fila:
- Matrix `[[a, b], [c, d]]` → `data = [a, b, c, d]`

### Core Operations

#### 1. **Determinant** (`matrix.rs`)

Implementado para matrices hasta 3×3 usando expansión directa:

```rust
pub fn determinant(ctx: &mut Context, matrix: ExprId) -> Option<ExprId>
```

**Casos soportados:**
- **1×1**: Directamente el elemento único
- **2×2**: `ad - bc` formula
- **3×3**: Sarrus rule / cofactor expansion

**Limitaciones**: No soporta matrices >3×3 (requerirían expansión por cofactores recursiva o LU decomposition).

#### 2. **Transpose** (`matrix.rs`)

Intercambia filas y columnas:

```rust
pub fn transpose(ctx: &mut Context, matrix: ExprId) -> Option<ExprId>
```

**Complejidad**: O(rows × cols)

#### 3. **Trace** (`matrix.rs`)

Suma de elementos diagonales (solo matrices cuadradas):

```rust
pub fn trace(ctx: &mut Context, matrix: ExprId) -> Option<ExprId>
```

### Simplification Rules

#### MatrixFunctionRule (`rules/matrix_ops.rs`)

Evaluates matrix functions when applied to matrix expressions:

```rust
pub struct MatrixFunctionRule;

impl Rule for MatrixFunctionRule {
    fn apply(&self, ctx: &mut Context, expr: ExprId) -> Option<Rewrite>
}
```

**Supported functions:**
- `det(M)` or `determinant(M)`
- `transpose(M)` or `T(M)`  
- `trace(M)` or `tr(M)`

**Function aliases** allow for both verbose and concise notation.

### CLI Integration

**Commands** (in `repl.rs`):
- `det <matrix>` - Compute determinant
- `transpose <matrix>` - Transpose matrix
- `trace <matrix>` - Compute trace

**Help System**: Organized in "Matrix Operations" category with specific help for each command.

**Autocomplete**: All three commands included in the autocomplete suggestions.

### Example Usage

```rust
// Parse matrix
let matrix = parse("[[1, 2], [3, 4]]", &mut ctx)?;

// Compute determinant
let det_expr = ctx.add(Expr::Function("det".to_string(), vec![matrix]));
let (result, _) = simplifier.simplify(det_expr);
// Result: -2

// Transpose
let t_expr = ctx.add(Expr::Function("transpose".to_string(), vec![matrix]));
let (result, _) = simplifier.simplify(t_expr);
// Result: [[1, 3], [2, 4]]
```

### Future Enhancements

Potential additions to matrix support:
- **Matrix inverse** (requires determinant ≠ 0 and cofactor matrix)
- **Matrix multiplication** (already supported via `MatrixMultiplyRule`)
- **Eigenvalues/eigenvectors** (requires characteristic polynomial)
- **LU/QR decomposition** (for larger determinants)
- **Rank** calculation

---

*Documento generado para ExpliCAS v0.1.0*

---

## Number Theory Implementation

### GCD (Greatest Common Divisor)

El sistema implementa el **Algoritmo de Euclides** para calcular el MCD de enteros y polinomios univariados.

#### Implementación

**Archivo**: `crates/cas_engine/src/rules/number_theory.rs`

```rust
pub struct GcdResult {
    pub value: Option<ExprId>,
    pub steps: Vec<String>,  // Educational explanations
}

pub fn compute_gcd(ctx: &mut Context, a: ExprId, b: ExprId, explain: bool) -> GcdResult
```

#### Casos Soportados

##### 1. **GCD Entero** ✅
Algoritmo de Euclides clásico para enteros.

```rust
gcd(48, 18) → 6
```

**Algoritmo**:
```
GCD(48, 18):
  48 = 18 × 2 + 12
  18 = 12 × 1 + 6
  12 = 6 × 2 + 0
  → GCD = 6
```

##### 2. **GCD Polinómico Univariado** ✅
Algoritmo de Euclides para polinomios en una variable.

```rust
gcd(2*x^2 + 7*x + 3, 2*x^2 + 5*x + 2) → 2*x + 1
```

**Algoritmo**:
- Division de polinomios con resto
- Reducción de grado iterativa
- **Normalización**: Devuelve polinomio **primitivo** (coeficientes enteros con GCD=1)

**Nota**: Versiones anteriores devolvían polinomios **mónicos** (coeficiente principal = 1), pero esto fue cambiado para devolver resultados con coeficientes enteros cuando es posible.

##### 3. **GCD Multivariable** ⚠️ LIMITADO

**Status**: No implementado. Devuelve conservadoramente GCD=1.

**Razón**: El GCD de polinomios multivariables requiere algoritmos significativamente más complejos:

**Algoritmos Necesarios**:
1. **Euclides Recursivo**: Tratar una variable como principal, otras como coeficientes
2. **Algoritmos Modulares**: Evaluación en puntos, interpolación
3. **Algoritmo de Brown**: Usando subresultantes

**Ejemplo del Problema**:
```rust
// Caso multivariable
gcd(x*y + y^2, x^2 + x*y) 
→ Actualmente devuelve: 1 (conservador)
→ Debería devolver: x + y

// Mensaje educativo
"Detectados polinomios multivariables."
"LIMITACIÓN: El GCD de polinomios multivariables no está implementado."
"Devolviendo GCD = 1 (conservador, no simplifica)."
```

**Decisión de Diseño**:
- Devolver GCD=1 es **matemáticamente correcto** (1 siempre divide a ambos)
- Es **conservador**: Peor caso, no simplificamos fracciones
- **No produce resultados incorrectos**
- Alternativa (rechazada): Devolver `None` o panic - más disruptivo

**Detección**:
```rust
let vars = collect_variables(ctx, a);
let vars_b = collect_variables(ctx, b);

if vars.len() > 1 || vars_b.len() > 1 || (vars != vars_b) {
    // Multivariable case
    return GcdResult {
        value: Some(ctx.num(1)),
        steps: vec!["LIMITACIÓN: ..."],
    };
}
```

#### Modo Educativo (`explain` Command)

El comando `explain` proporciona trazas educativas paso a paso en español.

**Usage**:
```text
> explain gcd(48, 18)
```

**Output**:
```
Algoritmo de Euclides para enteros:
Calculamos GCD(48, 18)
Dividimos 48 entre 18: Cociente = 2, Resto = 12
   → Como el resto es 12, el nuevo problema es GCD(18, 12)
Dividimos 18 entre 12: Cociente = 1, Resto = 6
   → Como el resto es 6, el nuevo problema es GCD(12, 6)
Dividimos 12 entre 6: Cociente = 2, Resto = 0
   → El resto es 0. ¡Hemos terminado!
El Máximo Común Divisor es: 6
```

**Implementación**:
- `verbose_integer_gcd()`: Trace de Euclides para enteros
- `verbose_poly_gcd()`: Trace de Euclides polinómico con detalles de grados y normalización

#### Futuras Mejoras

**GCD Multivariable**:
1. **Corto Plazo**: Factor común separable por variable
   ```rust
   gcd(6*x*y, 9*x*y^2) → 3*x*y  // GCD numérico × GCD_x × GCD_y
   ```
2. **Medio Plazo**: Euclides recursivo
3. **Largo Plazo**: Algoritmos modulares (Brown)

**Otras Extensiones**:
- LCM educativo (`explain lcm(...)`)
- Factorización educativa (`explain factors(...)`)
- Bezout coefficients (`extended_gcd(...)`)

---

*Documento generado para ExpliCAS v0.1.0*  
*Última actualización: 2025-12-08 - Agregado sistema Pattern Detection Infrastructure*

---

## Configuración del Repo

---

## 7. Optimizaciones

Para lograr un rendimiento aceptable a pesar de la semántica de "copia por valor" (simulada) y el ordenamiento canónico automático, el sistema implementa dos optimizaciones arquitectónicas profundas inspiradas en compiladores modernos.

### 7.1. Expression Interning (La Base del Rendimiento)

El sistema ya no almacena expresiones en un árbol simple, sino en un **Grafo Acíclico Dirigido (DAG)** mediante *Expression Interning*.

**Funcionamiento**:
1. El `Context` mantiene un `HashMap<u64, ExprId>` que mapea el hash estructural de una expresión a su ID existente.
2. Al llamar a `Context::add()`, calculamos el hash de la expresión canónica.
3. Si el hash existe y la expresión es estructuralmente idéntica, devolvemos el `ExprId` existente en lugar de crear un nodo nuevo.

**Beneficios**:
*   **Deduplicación Masiva**: Expresiones comunes como `x` o `1` se almacenan una sola vez en memoria.
*   **Comparaciones O(1)**: Para muchas operaciones, la igualdad `a == b` se reduce a comparar dos enteros `u32`.
*   **Rendimiento**: Mejora del 15-20% en operaciones intensivas como expansión de polinomios.

### 7.2. Compact ExprId (NaN-Boxing para Índices)

El identificador `ExprId` no es un simple índice `u32`. Utilizamos una técnica de "packing" para codificar información de tipo directamente en el identificador.

**Estructura del `u32`**:
*   **Bits 0-28**: Índice en el vector de nodos (`Context.nodes`). Capacidad para ~500 millones de nodos.
*   **Bits 29-31**: **Tag Estructural** (3 bits).

**Tags**:
*   `000` (0): **Number**
*   `001` (1): **Atom** (Variables, Constantes)
*   `010` (2): **Unary** (Neg)
*   `011` (3): **Binary** (Add, Mul, Pow, etc.)
*   `100` (4): **N-ary** (Function, Matrix)

**Beneficios**:
*   **Check de Tipo sin Pointer Chasing**: Podemos saber si una expresión es un número o un átomo simplemente mirando los bits del `ExprId`, sin acceder a la memoria del `Context`.
*   **Localidad de Caché**: Reduce la presión sobre la caché de CPU en funciones críticas como `compare_expr`.

---

## 8. Auto Rationalization System (Added 2025-12)

### Niveles de Racionalización Automática

El sistema implementa racionalización automática de denominadores con surds en múltiples niveles:

| Level | Descripción | Ejemplo | Auto |
|-------|-------------|---------|------|
| **0** | Single surd | `1/√n → √n/n` | ✅ |
| **1** | Binomial | `1/(a+b√n) → conjugate` | ✅ |
| **1.5** | Binomial in product | `1/(k*(a+b√n))` | ✅ |
| **2** | Multi-surd | `1/((1+√2)*(1+√3))` | Manual (`rationalize` cmd) |

### Tipos Centrales (`rationalize_policy.rs`)

```rust
enum AutoRationalizeLevel { Off, Level0, Level1, Level15 }

struct RationalizePolicy {
    auto_level: AutoRationalizeLevel,  // Default: Level15
    max_den_nodes: usize,              // Default: 30
    max_growth: usize,                 // Default: 20
    allow_same_surd_product: bool,     // Default: true
}

enum RationalizeReason {
    MultiSurdBlocked,
    BudgetExceeded,
    MaxGrowthExceeded,
    KNotSurdFree,
    NoBinomialFound,
    // ...
}
```

### Guards Implementados

1. **Multi-surd guard**: `count_distinct_numeric_surds(den) > 1` → block
2. **Budget guard**: `count_nodes(den) > 30` → block
3. **K surd-free check**: Non-binomial factors must be surd-free
4. **Phase flag**: `did_rationalize` prevents expansion after rationalization

### Tracing / Debugging

Activar trazas de racionalización:

```bash
RUST_LOG=rationalize=debug cargo run -p cas_cli
```

Logs de ejemplo:
```
DEBUG rationalize: auto rationalize rejected reason=MultiSurdBlocked surds=2
DEBUG rationalize: auto rationalize rejected reason=BudgetExceeded nodes=45 max=30
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `simplify <expr>` | Auto-rationalize (levels 0, 1, 1.5) |
| `rationalize <expr>` | Full multi-surd rationalization (level 2) |

### Tests

- **Semantic tests**: Verification via numeric substitution (`simplify(x) ≈ original`)
- **Policy tests**: Unit tests for `RationalizePolicy` behavior
- **Guard tests**: Verify multi-surd blocking, budget limits

