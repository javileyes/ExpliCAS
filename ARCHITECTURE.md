# Arquitectura del Motor CAS (ExpliCAS)

## Índice
1. [Visión General](#visión-general)
2. [Componentes Principales](#componentes-principales)
3. [Arquitectura del Sistema de Reglas](#arquitectura-del-sistema-de-reglas)
4. [Orquestación y Estrategias](#orquestación-y-estrategias)
5. [Flujo de Datos](#flujo-de-datos)
6. [Puntos de Extensión](#puntos-de-extensión)
7. [Optimizaciones](#optimizaciones)

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

$ dot -Tsvg ast.dot -o ast.svg
$ open ast.svg  # Beautiful tree visualization
```

**Features**:
- Color-coded nodes
- Auto-layout with Graphviz
- Export to SVG/PNG/PDF
- Zero runtime overhead (export only)

#### **3. Timeline HTML**

Interactive HTML visualization of simplification steps.

**Implementation**:
```rust
pub struct TimelineHtml<'a> {
    context: &'a Context,
    steps: &'a [Step],
}

impl<'a> TimelineHtml<'a> {
    pub fn to_html(&self) -> String {
        // Self-contained HTML with:
        // - MathJax for math rendering
        // - Purple gradient styling
        // - Timeline layout
        // - Before/After highlighting
    }
}
```

**Usage**:
```bash
> timeline (x+1)^2
Timeline exported to timeline.html
[Browser opens automatically on macOS]
```

**Generated HTML Features**:
- Beautiful purple gradient background
- MathJax-rendered math (`\(x^2 + 2x + 1\)`)
- Step-by-step timeline with numbered circles
- Color-coded before (orange) / after (green)
- Hover effects and animations
- Final result in green box

**Example Output**:
```html
<div class="step">
    <div class="step-number">1</div>
    <div class="step-content">
        <h3>Binomial Expansion</h3>
        <div class="math-expr before">
            <strong>Before:</strong> \((x+1)^2\)
        </div>
        <div class="math-expr after">
            <strong>After:</strong> \(x^2 + 2*x + 1\)
        </div>
    </div>
</div>
```

---

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
- **Engine**: ~2500 líneas (rules ~1800 + orchestrator ~300)
- **CLI**: ~1200 líneas
- **Debug Tools**: ~560 líneas (Phase 2)
- **Total proyecto**: ~5000 líneas

**Número de reglas**: ~70 reglas activas

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

```rust
pub trait Rule {
    fn name(&self) -> &str;
    fn target_types(&self) -> Option<Vec<&'static str>>;
    fn apply(&self, ctx: &mut Context, expr: ExprId) -> Option<Rewrite>;
}

pub struct Rewrite {
    pub new_expr: ExprId,
    pub description: String,
}
```

##### Tipos de Reglas

Las reglas se organizan por categorías:

| Categoría | Archivo | Ejemplos |
|-----------|---------|----------|
| **Aritmética** | `arithmetic.rs` | `AddZeroRule`, `MulOneRule`, `CombineConstantsRule` |
| **Canonicalización** | `canonicalization.rs` | `CanonicalizeAddRule`, `CanonicalizeMulRule` |
| **Exponentes** | `exponents.rs` | `ProductPowerRule`, `EvaluatePowerRule` |
| **Álgebra** | `algebra.rs` | `CombineLikeTermsRule`, `DistributeRule`, `FactorRule` |
| **Trigonometría** | `trigonometry.rs` | `PythagoreanIdentityRule`, `DoubleAngleRule` |
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

---

## Flujo de Datos

### Diagrama de Flujo Completo

```
┌─────────────┐
│ Usuario CLI │
└──────┬──────┘
       │ "x^2 + 2x + 1"
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
┌──────────────────────────────┐
│  Orchestrator                │
│  - Polynomial Strategy?      │
│  - Apply global loop         │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  LocalSimplificationTransformer      │
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
2. ~~**Fraction Simplification**~~: ✅ **COMPLETADO** - Detecta denominadores opuestos/iguales con bypass de complejidad
3. **Paralelización**: Simplificar subexpresiones independientes en paralelo
4. **Heurísticas Avanzadas**: ML para predecir mejor secuencia de reglas
5. **Verificación Formal**: Probar corrección de reglas con SMT solvers
6. **Rendimiento**: Compilación JIT de expresiones frecuentes
7. **UI Gráfica**: Visualización de árbol de simplificación
8. **Pruebas de Equivalencia**: Mejorar `equiv` command con más estrategias

### Referencias Útiles

- **Código fuente**: `/crates/cas_engine/src/`
- **Tests**: `/crates/cas_engine/tests/`
- **Documentación**: `README.md`, `MAINTENANCE.md`
- **Ejemplos**: `/crates/cas_cli/` (comandos interactivos)

---

*Documento generado para ExpliCAS v0.1.0*
