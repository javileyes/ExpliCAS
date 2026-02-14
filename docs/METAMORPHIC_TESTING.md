# Metamorphic Equivalence Testing

> Motor de mejora continua para el engine CAS basado en tests de identidades matemáticas.

## Introducción

El sistema de **Metamorphic Equivalence Testing** es la herramienta principal para:

1. **Validar** que el engine simplifica correctamente expresiones matemáticas
2. **Detectar** debilidades en las reglas de simplificación (identidades que no pasan simbólicamente)
3. **Identificar** bugs reales mediante detección de asimetrías numéricas
4. **Medir** la cobertura de simplificación del engine

---

## Arquitectura del Sistema

```
┌──────────────────────────────────────────────────────────────┐
│                    identity_pairs.csv                        │
│  (~400 identidades: algebra, trig, log, rationales, etc.)   │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                   load_identity_pairs()                      │
│  Soporta: 4-col legacy | 7-col extended                     │
└──────────────────────────┬───────────────────────────────────┘
                           │
           ┌───────────────┴───────────────┐
           ▼                               ▼
┌─────────────────────┐         ┌────────────────────────────┐
│  Symbolic Check     │         │    Numeric Check           │
│  simplify(L) == R   │         │  eval_f64_checked(L, R)    │
│  (engine-level)     │         │  (fallback validation)     │
└─────────┬───────────┘         └─────────────┬──────────────┘
          │                                   │
          ▼                                   ▼
┌─────────────────────────────────────────────────────────────┐
│                    NumericEquivStats                         │
│  valid | near_pole | domain_error | asymmetric_invalid      │
│  max_abs_err | worst_sample | is_fragile()                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Formato CSV de Identidades

### Legacy (4 columnas)
```csv
# exp,simp,vars,domain_mode
sin(x)^2+cos(x)^2,1,x,g
```

### Extended (7 columnas)
```csv
# exp,simp,vars,domain_mode,bucket,branch_mode,filter
2*arctan(x),arctan(2*x/(1-x^2)),x,a,branch_sensitive,modulo_pi,abs_lt(0.9)
```

### Campos

| Campo | Valores | Descripción |
|-------|---------|-------------|
| `exp` | expresión | Left-hand side (forma expandida) |
| `simp` | expresión | Right-hand side (forma simplificada) |
| `vars` | `x` o `x;y` | Variables separadas por `;` |
| `domain_mode` | `g`/`a` | Generic o Assume (DomainMode) |
| `bucket` | ver abajo | Clasificación de la identidad |
| `branch_mode` | ver abajo | Modo de comparación numérica |
| `filter` | spec | Filtro de muestreo |

---

## Sistema de Buckets

Los buckets clasifican identidades por su "tipo de verdad":

### `unconditional`
- Identidades algebraicas/trigonométricas puras
- **min_valid: 70%** de muestras
- `asymmetric_invalid > 0` → **FAIL**
- Ejemplo: `sin(x)^2 + cos(x)^2 = 1`

### `conditional_requires`
- Requieren condiciones de dominio (`x ≠ 0`, `cos(x) ≠ 0`)
- **min_valid: 50%** de muestras
- El evaluador checked detecta NearPole/Domain automáticamente
- Ejemplo: `tan(x) = sin(x)/cos(x)`

### `branch_sensitive`
- Involucran arctan/arcsin/log/pow con bases negativas
- **min_valid: 35%** de muestras
- `asymmetric_invalid` solo es warning
- Ejemplo: `2*arctan(x) = arctan(2x/(1-x²))`

---

## Modos de Comparación (BranchMode)

### `principal_strict`
Comparación directa con atol/rtol:
```rust
|L - R| <= atol + rtol * max(|L|, |R|, 1.0)
```

### `modulo_pi`
Para identidades de arctan (difieren por kπ):
```rust
circular_dist(L, R, π) <= tolerance
```

### `modulo_2pi`
Para identidades trigonométricas generales:
```rust
circular_dist(L, R, 2π) <= tolerance
```

### `principal_with_filter`
Como `principal_strict` pero **requiere** filter no vacío. Panic si filter = None.

---

## Evaluador Checked (`eval_f64_checked`)

### Errores Detectados

| Error | Causa | Tratamiento |
|-------|-------|-------------|
| `NearPole { op, denom, threshold }` | Denominador ≈ 0 | sample inválido |
| `DivisionByZero { op }` | Denominador = 0 | sample inválido |
| `Domain { function, arg }` | log(≤0), sqrt(<0) | sample inválido |
| `NonFinite` | NaN o Inf | sample inválido |
| `DepthExceeded` | Recursión excesiva | sample inválido |

### Opciones

```rust
EvalCheckedOptions {
    zero_abs_eps: 1e-12,   // Para divisiones
    zero_rel_eps: 1e-12,   // Escala con numerador
    trig_pole_eps: 1e-9,   // Mayor para trig (FP errors en π/2)
    max_depth: 200,
}
```

---

## Filtros de Muestreo (FilterSpec)

El sistema soporta filtros compilados en runtime desde el CSV (sin closures, determinista).

### Sintaxis CSV

```csv
# Sin filtro (campo vacío o no especificado)
sin(x)^2+cos(x)^2,1,x,g,unconditional,principal_strict,

# |x| < 0.9
...,abs_lt(0.9)

# Evitar singularidades (π/2, -π/2)
...,away_from(1.5707963;-1.5707963;eps=0.01)

# Combinado: |x| < 0.95 AND away from 1.0, -1.0
...,abs_lt_and_away(0.95;1.0;-1.0;eps=0.1)

# Filtros de dominio (NEW)
...,gt(0.0)      # x > 0 (para ln)
...,ge(0.0)      # x >= 0 (para sqrt)
...,lt(1.0)      # x < 1
...,le(1.0)      # x <= 1
...,range(0.1;3.0)  # 0.1 <= x <= 3.0
```

### FilterSpec Enum (Runtime)

```rust
enum FilterSpec {
    None,                                               // Sin filtro
    AbsLt { limit: f64 },                               // |x| < limit
    AwayFrom { centers: Vec<f64>, eps: f64 },           // |x - c| > eps
    AbsLtAndAway { limit: f64, centers: Vec<f64>, eps: f64 },
    // Filtros de dominio (V2.15.2)
    Gt { limit: f64 },    // x > limit (ln, log)
    Ge { limit: f64 },    // x >= limit (sqrt)
    Lt { limit: f64 },    // x < limit
    Le { limit: f64 },    // x <= limit
    Range { min: f64, max: f64 },  // min <= x <= max
}

impl FilterSpec {
    fn accept(&self, x: f64) -> bool { ... }
}
```

### Filtros por Función Matemática

| Función | Filter Recomendado | Razón |
|---------|-------------------|-------|
| `ln(x)`, `log(x)` | `gt(0.0)` | Dominio x > 0 |
| `sqrt(x)` | `ge(0.0)` | Dominio x >= 0 |
| `1/x` | `away_from(0;eps=0.01)` | Polo en x=0 |
| `tan(x)` | `away_from(1.57;-1.57;eps=0.01)` | Polos en ±π/2 |

---

## Métricas y Diagnósticos

### NumericEquivStats

```rust
struct NumericEquivStats {
    valid: usize,              // Samples que pasaron
    near_pole: usize,          // Ambos L y R tienen polo
    domain_error: usize,       // Ambos L y R tienen error de dominio
    asymmetric_invalid: usize, // L ok, R err (o viceversa) - SOSPECHOSO
    eval_failed: usize,        // Otros fallos
    filtered_out: usize,       // Rechazados por filtro
    mismatches: Vec<String>,   // Top 5 discrepancias
    max_abs_err: f64,          // Mayor error absoluto
    max_rel_err: f64,          // Mayor error relativo
    worst_sample: (x, a, b),   // Punto con mayor error
}

impl NumericEquivStats {
    fn invalid_rate(&self) -> f64; // (near_pole + domain_error + eval_failed) / total
    fn is_fragile(&self) -> bool;  // invalid_rate > 30%
}
```

### Indicadores Clave

| Métrica | Significado | Acción |
|---------|-------------|--------|
| `asymmetric_invalid > 0` | Bug probable en engine | Investigar |
| `is_fragile()` | >30% near_pole/domain | Revisar muestreo |
| `mismatches.len() > 0` | Fallo numérico real | Verificar identidad |

---

## Clasificación de Diagnósticos (DiagCategory)

Sistema de clasificación por prioridad para identificar el tipo de problema.

### Categorías

```rust
enum DiagCategory {
    BugSignal,    // 🐛 asymmetric_invalid > 0
    ConfigError,  // ⚙️ eval_failed_rate > 50%
    NeedsFilter,  // 🔧 domain_rate > 20%
    Fragile,      // ⚠️ pole_rate > 15%
    Ok,           // ✅ Todo bien
}
```

Aquí va el significado “operativo” de cada categoría, tal y como las estáis usando en el diagnóstico metamórfico (numérico + chequeos de asimetría), con ejemplos típicos y qué acción sugiere.

## 🐛 BugSignal

**Qué significa:** hay una señal fuerte de **bug del engine o del evaluador**, porque el fallo es **asimétrico**:

* L evalúa “OK” y R da error (NearPole/Domain/NonFinite/Unsupported…), **o al revés**, en un porcentaje no trivial, **con el mismo muestreo**.

**Por qué es serio:** una identidad correcta no debería producir “válido solo en un lado” si ambos lados representan la misma función en su dominio. La asimetría suele indicar:

* simplificación no sound que introduce/borra restricciones,
* evaluador que evalúa formas equivalentes de manera distinta (p. ej. reordenación que cambia estabilidad numérica),
* reglas que transforman a una forma con polos/dominio diferente sin añadir requires.

**Ejemplo típico:**

* L = `sqrt(x^2)` simplifica a `x` en generic (bug), R = `|x|`. Para x<0: L eval OK (da negativo), R eval OK (positivo). Aquí no hay error, pero hay **mismatch**.
  Más BugSignal típico:
* L = `ln(x^2)` (si el engine lo convierte mal) vs R = `2*ln(x)`; para x<0 una puede dar Domain y otra no → asimetría.

**Acción recomendada:** investigar reglas/evaluación. No se arregla con filtros “bonitos”.

---

## ⚙️ ConfigError

**Qué significa:** el test falla por **configuración**, no por fragilidad matemática.
Casos típicos:

* variable o constante **no evaluable** (`phi` antes de soportarla, símbolos no ligados),
* función marcada como `Unsupported` en el evaluador,
* faltan bindings para variables,
* modo/branch_mode incompatible con la identidad.

**Ejemplo típico:**

* `phi^2 ≡ phi + 1` cuando `phi` no está implementado en parser/evaluator → 100% `UnboundVariable`.

**Acción recomendada:** implementar constante/función, o ajustar el harness (binds, soportes). No es un bug algebraico.

---

## 🔧 NeedsFilter

**Qué significa:** la identidad es correcta **pero el muestreo aleatorio entra demasiado a menudo en regiones fuera del dominio** (o regiones donde la identidad requiere condiciones), y eso dispara muchos `DomainError`/`NearPole` *simétricos* (en ambos lados), o demasiadas muestras inválidas para decidir.

Diferencia clave con BugSignal:

* aquí la invalidez suele ser **simétrica**: ambos lados fallan por dominio/polo a la vez (o casi).

**Ejemplos típicos:**

* identidades con `ln(x)` → necesitas `gt(0)`
* `sqrt(x)` → necesitas `ge(0)`
* identidades con `1/x` → necesitas `away_from(0)`
* `tan(x)` → necesitas `away_from(pi/2 + k*pi)` si muestreáis en rango amplio

**Acción recomendada:** añadir `filter_spec` (gt/ge/range/away_from) o cambiar el rango de muestreo.
No implica que el motor esté mal; implica que el test está muestreando “demasiado agresivo” para esa identidad.

---

## ⚠️ Fragile

**Qué significa:** el test es matemáticamente válido y pasa en muchas muestras, pero es **numéricamente inestable** con el muestreo actual: produce un `invalid_rate` alto por **cercanía a singularidades** o problemas de floating-point, aun sin asimetría.

Suele ocurrir cuando:

* hay cancelaciones fuertes,
* hay denominadores que pueden hacerse pequeños,
* trig cerca de polos,
* expresiones que crecen muy rápido.

**Ejemplo típico:**

* `tan(x) ≡ sin(x)/cos(x)` cerca de `cos(x)=0`: ambos lados pueden dar NearPole/Inf; no es bug, pero es frágil.

**Acción recomendada:**

* endurecer filtros (away_from más estricto),
* subir eps de polos para trig,
* o tratarla como identidad “frágil” en el informe (permitir warning/umbral mayor).
  No es “NeedsFilter” si ya tienes filtro razonable y aun así hay inestabilidad notable: es fragilidad inherente a evaluación con floats.

---

## ✅ Ok

**Qué significa:** pasa y está “saludable”:

* suficientes muestras válidas (`valid >= min_valid(bucket)`),
* `invalid_rate` dentro de umbrales,
* `asymmetric_invalid = 0`,
* mismatches numéricos dentro de tolerancia (o 0).

**Acción recomendada:** nada; se puede usar como baseline/regresión.

---

### Resumen mental rápido

* **BugSignal** = “huele a bug”: *asimetría*.
* **ConfigError** = “no se puede evaluar / falta soporte”.
* **NeedsFilter** = “falta restringir dominio/rango”.
* **Fragile** = “dominio ok pero evaluación float es delicada”.
* **Ok** = “todo bien”.


### Precedencia

1. **BugSignal**: `asymmetric_invalid > 0` → Bug potencial en engine
2. **ConfigError**: `eval_failed > 50%` → Variable no ligada o unsupported
3. **NeedsFilter**: `domain_error > 20%` → Función fuera de dominio (ln/sqrt)
4. **Fragile**: `near_pole > 15%` → Cerca de singularidades
5. **Ok**: Todo dentro de umbrales

### Métricas por Categoría

```rust
impl NumericEquivStats {
    fn domain_rate(&self) -> f64;      // domain_error / total
    fn pole_rate(&self) -> f64;        // near_pole / total  
    fn eval_failed_rate(&self) -> f64; // eval_failed / total
}
```

### Output Diagnóstico (`METATEST_DIAG=1`)

```
METATEST_DIAG=1 cargo test --package cas_engine --test metamorphic_simplification_tests -- metatest_individual --ignored --nocapture 2>&1

📊 Diagnostic Classification (METATEST_DIAG=1):
   Summary: ✅ Ok=97 | 🐛 BugSignal=0 | ⚙️ ConfigError=0 | 🔧 NeedsFilter=0 | ⚠️ Fragile=0
```

---

## Políticas de CI (FragilityLevel)

### Niveles de Fragilidad

```rust
enum FragilityLevel {
    Ok,      // Dentro de umbrales normales
    Warning, // Elevado pero aceptable
    Fail,    // Debe fallar CI
}
```

### Umbrales por Bucket

| Bucket | Warning | Fail |
|--------|---------|------|
| `Unconditional` | ≥10% invalid | ≥25% invalid |
| `ConditionalRequires` | ≥30% invalid | ≥50% invalid |
| `BranchSensitive` | ≥40% invalid | ≥60% invalid |

### Reglas CI

1. **`asymmetric_invalid > 0`** → **FAIL** (todos los buckets)
   - Indica cambio de dominio asimétrico o bug en evaluador
   
2. **`FragilityLevel::Fail`** → **FAIL**
   - Demasiados samples inválidos para el bucket

3. **`FragilityLevel::Warning`** → **WARNING** (log, no fail)
   - Identidad frágil pero dentro de tolerancia

---

## Ejecución de Tests

### Test Individual (diagnóstico)

```bash
# Modo genérico (default)
cargo test --package cas_engine --test metamorphic_simplification_tests \
    -- metatest_individual --ignored --nocapture

# Modo assume
METATEST_MODE=assume cargo test ...

# Migración: bucket legacy = unconditional
METATEST_LEGACY_BUCKET=unconditional cargo test ...
```

### Test de Combinaciones

Los tests de combinaciones generan miles de expresiones compuestas a partir del CSV de identidades,
combinando pares con distintas operaciones (Add, Sub, Mul, Div).

#### Muestreo Estratificado (Stratified Sampling)

El sistema de selección de pares usa **muestreo estratificado por familias** para garantizar
cobertura diversa con un número manejable de pares:

1. **Fase 1**: Selecciona 1 representante por familia CSV (~134 familias) usando LCG RNG determinista
2. **Fase 2**: Rellena los slots restantes (`max_pairs - num_families`) desde pares no seleccionados
3. **Shuffle final**: Las selecciones se barajan para randomizar el orden de combinaciones

**Seed configurable**: La semilla del LCG se controla con `METATEST_SEED=<u64>` (default `0xC0FFEE`).
Distintas semillas seleccionan distintos pares, permitiendo exploración multi-seed para descubrir
edge cases. Ejemplo: `METATEST_SEED=42 cargo test ...`

**Modo legacy**: Con `METATEST_NOSHUFFLE=1` se usa el enfoque anterior de ventana contígua
(combinado con `METATEST_START_OFFSET=N` para desplazar la ventana).

#### Tests Disponibles

| Test | Op | Pares | Familias | Combos (≈) | Modo |
|------|-----|-------|----------|------------|------|
| `metatest_csv_combinations_small` | **Add** | 30 | ~30 | ~435 | CI (no-ignore) |
| `metatest_csv_combinations_add` | **Add** | 150 | ~134 | ~11,175 | `--ignored` |
| `metatest_csv_combinations_sub` | **Sub** | 150 | ~134 | ~11,175 | `--ignored` |
| `metatest_csv_combinations_mul` | **Mul** | 150 | ~134 | ~11,175 | `--ignored` |
| `metatest_csv_combinations_div` | **Div** | 50 | ~50 | ~1,225 | `--ignored` |
| `metatest_csv_combinations_full` | **Add** | 100 | ~100 | ~4,950+triples | `--ignored` |
| `metatest_benchmark_all_ops` | **All** | — | — | ~34k | `--ignored` |
| `metatest_unified_benchmark` | **All+Sub** | — | — | ~12k | `--ignored` |

**Nota sobre Div:** Usa solo 50 pares porque las limitaciones del CAS con divisores polinómicos de
alto grado causan fallos de simplificación de fracciones. Incluye un safety guard que salta identidades
cuyo divisor evalúa cerca de cero.

#### Comandos

```bash
# CI (Add, 30 pares estratificados, ~435 combinaciones dobles)
cargo test -p cas_engine --test metamorphic_simplification_tests \
    metatest_csv_combinations_small -- --nocapture 2>&1

# Add completo (150 pares estratificados)
cargo test -p cas_engine --test metamorphic_simplification_tests \
    metatest_csv_combinations_add -- --nocapture --ignored 2>&1

# Subtraction (150 pares estratificados)
cargo test -p cas_engine --test metamorphic_simplification_tests \
    metatest_csv_combinations_sub -- --nocapture --ignored 2>&1

# Multiplication (150 pares estratificados, 2s per-combo timeout)
cargo test --release -p cas_engine --test metamorphic_simplification_tests \
    metatest_csv_combinations_mul -- --nocapture --ignored 2>&1

# Division (50 pares estratificados, divisor safety guard)
cargo test --release -p cas_engine --test metamorphic_simplification_tests \
    metatest_csv_combinations_div -- --nocapture --ignored 2>&1

# Add legacy (100 pares + triples)
cargo test -p cas_engine --test metamorphic_simplification_tests \
    metatest_csv_combinations_full -- --nocapture --ignored 2>&1
```

#### Benchmark Unificado (`metatest_benchmark_all_ops`)

Test diagnóstico que ejecuta las 4 operaciones y muestra una tabla comparativa de
regresión/mejora. **No aserta sobre fallos** — solo imprime métricas.

```bash
cargo test --release -p cas_engine --test metamorphic_simplification_tests \
    metatest_benchmark_all_ops -- --nocapture --ignored 2>&1
```

Output de ejemplo:

```
╔═══════════════════════════════════════════════════════════════════════════════════╗
║                     METAMORPHIC BENCHMARK RESULTS                                ║
╠═════╤════════╤══════════╤══════════════╤═════════════════╤══════════════╤══════════╣
║ Op  │ Pairs  │ Families │ NF-convergent│ Proved-sym (Q+D)│ Numeric-only │ Failed   ║
╠═════╪════════╪══════════╪══════════════╪═════════════════╪══════════════╪══════════╣
║ add │   150  │     134  │   5797  67.0% │ 2788+0    32.2% │     61   0.7% │      0   ║
║ sub │   150  │     134  │   6082  70.3% │ 2532+0    29.3% │     32   0.4% │      0   ║
║ mul │   150  │     134  │   5860  68.3% │ 2033+375  28.1% │    316   3.7% │      0   ║
║ div │    50  │      50  │    489  59.1% │  282+34   38.2% │     22   2.7% │      0   ║
╠═════╪════════╪══════════╪══════════════╪═════════════════╪══════════════╪══════════╣
║ ALL │        │          │  18228  68.3% │      8044 30.1% │    431   1.6% │      0   ║
╚═════╧════════╧══════════╧══════════════╧═════════════════╧══════════════╧══════════╝
```

**Lectura de la columna `Proved-sym (Q+D)`:**
- **Q** (quotient) = el motor simplifica `A/B → 1` nativamente (para Mul/Div)
  o `A−B → 0` nativamente (para Add/Sub).
- **D** (difference fallback) = el motor **NO** puede simplificar `A/B → 1`, pero SÍ
  `A−B → 0`. Señal de **debilidad del motor** para simplificación de cocientes.

Para Add/Sub, D siempre es 0 (la diferencia ES la verificación nativa).
Para Mul/Div, D > 0 indica identidades que el motor no puede cancelar en forma de cociente.

Uso típico: comparar métricas antes/después de añadir una regla de simplificación.
La columna D indica el número de casos que mejorarían si se mejorara la simplificación de cocientes.

#### Benchmark Unificado Completo (`metatest_unified_benchmark`)

Test que combina **combinaciones (Add/Sub/Mul/Div) + sustituciones** en una sola ejecución
con tabla unificada. Usa pair counts reducidos para un runtime de ~7 minutos:

| Suite | Configuración |
|-------|---------------|
| `+add` | 30 pares estratificados |
| `−sub` | 30 pares estratificados |
| `×mul` | 150 pares estratificados |
| `÷div` | 50 pares estratificados |
| `⇄sub` | 75 identidades × 20 sustituciones |

```bash
cargo test --release -p cas_engine --test metamorphic_simplification_tests \
    metatest_unified_benchmark -- --ignored --nocapture
```

Output (seed 12648430, Feb 2026):

```
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║              UNIFIED METAMORPHIC REGRESSION BENCHMARK (seed 12648430  )                                    ║
╠═══════╤════════╤══════════════╤══════════════╤══════════════╤════════╤═══════╤════════╤════════════════════╣
║ Suite │ Combos │ NF-convergent│ Proved-sym   │ Numeric-only │ Failed │  T/O  │ Cycles │ Skip/Parse-err     ║
╠═══════╪════════╪══════════════╪══════════════╪══════════════╪════════╪═══════╪════════╪════════════════════╣
║ add   │    351 │   195  55.6% │   156  44.4% │     0   0.0% │      0 │     0 │      0 │      0             ║
║ sub   │    351 │   210  59.8% │   141  40.2% │     0   0.0% │      0 │     0 │      0 │      0             ║
║ mul   │   9045 │  6361  70.4% │  2526  27.9% │   151   1.7% │      0 │     7 │    120 │      0             ║
║ div   │    793 │   463  58.5% │   321  40.6% │     7   0.9% │      0 │     2 │     18 │      0             ║
║ ⇄sub  │   1500 │  1107  73.8% │   328  21.9% │    65   4.3% │      0 │     0 │     64 │      0             ║
╠═══════╪════════╪══════════════╪══════════════╪══════════════╪════════╪═══════╪════════╪════════════════════╣
║ TOTAL │  12040 │  8336  69.3% │  3472  28.9% │   223   1.9% │      0 │     9 │    202 │      0             ║
╚═══════╧════════╧══════════════╧══════════════╧══════════════╧════════╧═══════╧════════╧════════════════════╝
```

> [!TIP]
> `metatest_unified_benchmark` es el test recomendado para validar cambios antes de merge.
> Ejecuta ~12k combos en ~7 min y cubre las 5 dimensiones de testing metamórfico.
> `metatest_benchmark_all_ops` sigue disponible para ejecuciones más exhaustivas (150 pares/op, ~34k combos).

#### Modo Verbose

Para ver el **informe detallado con clasificación por niveles**:

```bash
METATEST_VERBOSE=1 cargo test --release -p cas_engine \
    --test metamorphic_simplification_tests metatest_csv_combinations_mul \
    -- --nocapture --ignored 2>&1
```

**Variables de entorno para combinaciones:**

| Variable | Default | Descripción |
|----------|---------|-------------|
| `METATEST_VERBOSE` | (desactivado) | Activa informe detallado con ejemplos y clasificadores |
| `METATEST_MAX_EXAMPLES` | `10` | Número máximo de ejemplos a mostrar por categoría |
| `METATEST_SEED` | `0xC0FFEE` | Semilla para el LCG del muestreo estratificado |
| `METATEST_NOSHUFFLE` | (desactivado) | Modo legacy: ventana contígua en vez de estratificado |
| `METATEST_START_OFFSET` | `0` | Offset para ventana legacy (solo con `METATEST_NOSHUFFLE=1`) |

#### Clasificación de Combinaciones (4 niveles)

Cada combinación `(identity_i ⊕ identity_j)` se clasifica en:

| Nivel | Emoji | Significado |
|-------|-------|-------------|
| **NF-convergent** | 📐 | **Equivalencia simbólica pura** — `simplify(LHS) == simplify(RHS)` estructuralmente idénticos |
| **Proved-quotient** | 🔢 Q | **Equivalencia nativa** — `simplify(LHS/RHS) == 1` (Mul/Div) o `simplify(LHS-RHS) == 0` (Add/Sub) |
| **Proved-difference** | 🔢 D | **Fallback** — `simplify(LHS-RHS) == 0` cuando el cociente no simplifica. **Señal de debilidad del motor** |
| **Numeric-only** | 🌡️ | **Equivalencia numérica** — solo pasa por muestreo numérico, no hay prueba simbólica |
| **Failed** | ❌ | **Error** — falla incluso la equivalencia numérica |

> [!NOTE]
> Para **Add/Sub**, el check nativo ya usa diferencia, así que D siempre es 0.
> Para **Mul/Div**, D > 0 indica combinaciones donde el motor no puede cancelar el cociente
> `A/B → 1`, pero sí puede demostrar `A−B → 0`. El número D es un indicador directo de
> mejoras posibles en la simplificación de cocientes.

#### Robustez: `catch_unwind`

El path inline de Add/Sub está protegido con `std::panic::catch_unwind` para capturar panics
latentes (p.ej. `num-rational` con denominador cero en combinaciones específicas). Los panics
se clasifican como skips, no como fallos.

#### Output Ejemplo

```
📊 Running CSV combination tests [mul] with 150 pairs from 134 families (seed 12648430, offset 0, stratified)
✅ Double combinations [mul]: 8584 passed, 0 failed, 0 skipped (timeout)
   📐 NF-convergent: 5860 | 🔢 Proved-symbolic: 2408 (quotient: 2033, diff: 375) | 🌡️ Numeric-only: 316
```

#### Secciones del Informe Verbose

Con `METATEST_VERBOSE=1` se muestran **4 secciones adicionales**:

**1. 🔢 NF-mismatch examples** — Proved-symbolic pero con formas normales diferentes:
```
🔢 NF-mismatch examples (proved symbolic but different normal forms):
    1. LHS: (sin(x)^2 + cos(x)^2) + ((u^2+1)*(u+1)*(u-1))
       RHS: (1) + ((u^2+1)*(u+1)*(u-1))
       (simplifies: 1 + (x^2+1)*(x+1)*(x-1))
```

**2. 🌡️ Numeric-only examples** — Con el residuo `simplify(LHS-RHS)` en LaTeX:
```
🌡️ Numeric-only examples (no symbolic proof found):
    1. LHS: (tan(x)^2 + 1) + (tan(2*u))
       RHS: (sec(x)^2) + (2*tan(u)/(1-tan(u)^2))
       simplify(LHS-RHS): \frac{...}{...}
```

**3. 📊 Family classifier** — Agrupación de casos numeric-only por familia matemática:
```
📊 Numeric-only grouped by family:
   ── tan (without sec/csc) (15 cases) ──
   ── sec/csc (Pythagorean: tan²+1=sec², 1+cot²=csc²) (9 cases) ──
```

Familias detectadas: `sec/csc`, `tan`, `cot`, `half/double angle`, `ln/log`, `exp`, `sqrt/roots`, `abs`, `arc*`, `other`.

**4. 📈 Top-N Shape Analysis** — Patrones dominantes en los residuos:
```
📈 Top-N Shape Analysis (residual patterns):
    1.   8.3% (  2) Div(Add(Mul(...),Mul(...)),...)  [NEG_EXP] [DIV]
    2.   4.2% (  1) Div(Add(Add(...),...),...) [NEG_EXP] [DIV]
```

Marcadores: `[NEG_EXP]` = exponentes negativos, `[DIV]` = divisiones. Apuntan a reglas de simplificación faltantes.

**Interpretación:** Las combinaciones numeric-only indican que el simplificador produce resultados
diferentes pero matemáticamente equivalentes. Esto es normal y **no es un error** — lo importante
es que `Failed = 0`. Los clasificadores ayudan a **priorizar qué reglas de simplificación añadir**.

### Baselines de Combinaciones (Feb 2026, Seed 42)

Resultados de referencia con muestreo estratificado, difference fallback, y Q/D split:

| Op | Pairs | Families | NF-conv | Proved (Q+D) | Numeric | Failed |
|----|-------|----------|---------|--------------|---------|--------|
| Add | 150 | 134 | 5797 | 2788+0 | 61 | 0 |
| Sub | 150 | 134 | 6082 | 2532+0 | 32 | 0 |
| Mul | 150 | 134 | 5860 | 2033+375 | 316 | 0 |
| Div | 50 | 50 | 489 | 282+34 | 22 | 0 |

> [!IMPORTANT]
> Los 375 Mul y 34 Div en la columna D son **debilidades del motor**: el engine
> no puede simplificar `A/B → 1` pero sí `A−B → 0`. Mejorar la simplificación
> de cocientes (trig normalization, polynomial cancellation, ln expansion)
> reduciría estos números.

### Baseline Unificado (Feb 2026, Seed 12648430)

Resultados del benchmark unificado (`metatest_unified_benchmark`) con pair counts reducidos:

| Suite | Combos | NF-conv | NF% | Proved | Proved% | Numeric | Num% | Failed | Timeout |
|-------|--------|---------|-----|--------|---------|---------|------|--------|---------|
| +add | 351 | 192 | 54.7 | 159 | 45.3 | 0 | 0.0 | 0 | 0 |
| −sub | 351 | 207 | 59.0 | 144 | 41.0 | 0 | 0.0 | 0 | 0 |
| ×mul | 9045 | 6244 | 69.6 | 2438 | 27.2 | 289 | 3.2 | 0 | 74 |
| ÷div | 793 | 452 | 57.2 | 320 | 40.5 | 18 | 2.3 | 0 | 3 |
| ⇄sub | 1500 | 981 | 65.4 | 293 | 19.5 | 226 | 15.1 | 0 | 0 |
| **TOTAL** | **12040** | **8076** | **67.5** | **3354** | **28.0** | **533** | **4.5** | **0** | **77** |

> [!NOTE]
> Runtime: ~7 min (release mode). La suite ⇄sub tiene el mayor % de numeric-only (15.1%),
> lo que indica oportunidades de mejora en la simplificación de expresiones con sustituciones compuestas.

### Qué Significan

**Individual:**
- **Symbolic**: Engine produjo la forma canónica esperada
- **Numeric-only**: Equivalentes numéricamente, pero el engine aún no simplifica a la misma forma
- **Failed**: Ni simbólico ni numérico equivalentes (bug o identidad incorrecta)
- **Skipped**: Identidad requiere modo `assume` y test corre en `generic`

**Combinaciones:**
- **NF-convergent**: Ambos lados simplifican a la misma expresión exacta (ideal)
- **Proved-quotient (Q)**: El motor simplifica `A/B → 1` o `A−B → 0` nativamente
- **Proved-difference (D)**: Solo `simplify(A−B) == 0` funciona, no el cociente (debilidad del motor)
- **Numeric-only**: Solo equivalencia numérica — oportunidad de mejora del engine

### Mejorar el Engine

1. **Aumentar Symbolic %**: Añadir reglas de simplificación
2. **Reducir Numeric-only**: Analizar familias y shapes para priorizar reglas
3. **Reducir Failed**: Verificar identidad matemática o corregir regla
4. **Investigar asymmetric_invalid**: Señal de bug en evaluación

---

## Agregar Nuevas Identidades

### Proceso

1. Añadir línea a `identity_pairs.csv`
2. Ejecutar test para verificar
3. Si falla simbólicamente pero pasa numéricamente → oportunidad de mejora del engine
4. Si falla numéricamente → verificar matemáticamente la identidad

### Buenas Prácticas

- Usar `unconditional` solo para identidades realmente universales
- Añadir filtros para identidades con singularidades conocidas
- Documentar identidades branch-sensitive con comentarios

---

## Variables de Entorno

| Variable | Valores | Default | Descripción |
|----------|---------|---------|-------------|
| `METATEST_MODE` | `generic`/`assume` | `generic` | DomainMode del engine |
| `METATEST_STRESS` | `0`/`1` | `0` | Más samples, mayor depth |
| `METATEST_DIAG` | `0`/`1` | `0` | Habilita diagnóstico detallado (individual) |
| `METATEST_LEGACY_BUCKET` | `unconditional`/`conditional_requires` | `conditional_requires` | Bucket para CSV 4-col |
| `METATEST_SNAPSHOT` | `0`/`1` | `0` | Compara resultados vs baseline |
| `METATEST_UPDATE_BASELINE` | `0`/`1` | `0` | Regenera archivo baseline |
| `METATEST_VERBOSE` | `0`/`1` | `0` | Informe detallado: ejemplos, familias, shapes |
| `METATEST_MAX_EXAMPLES` | número | `10` | Máximos ejemplos a mostrar por categoría |
| `METATEST_SEED` | `u64` | `0xC0FFEE` | Semilla para LCG del muestreo estratificado |
| `METATEST_NOSHUFFLE` | `0`/`1` | `0` | Modo legacy: ventana contígua en vez de estratificado |
| `METATEST_START_OFFSET` | número | `0` | Offset para ventana legacy (solo con `METATEST_NOSHUFFLE=1`) |

---

## Sistema de Baseline JSONL (Regresión Tracking)

El sistema de baseline permite detectar regresiones en la calidad del engine entre commits.

### Archivo Baseline

```
crates/cas_engine/tests/baselines/metatest_baseline.jsonl
```

**Primera línea**: Header de configuración con `cfg_hash`:

```json
{"_type":"config","cfg_hash":"b1e48281af9a6844","samples":200,"min_valid":180,"atol":1e-8,"rtol":1e-8,"range":[-10,10]}
```

**Líneas siguientes**: Snapshot por identidad:

```json
{"id":"c81215fe481d1332","exp":"tan(x)^2 + 1","simp":"sec(x)^2","category":"Ok","valid":200,"filtered_out":0,"near_pole":0,"domain_error":0,"eval_failed":0,"asymmetric":0,"mismatches":0,"total":200}
```

### Comandos

```bash
# Generar/actualizar baseline (después de cambios confirmados)
METATEST_DIAG=1 METATEST_UPDATE_BASELINE=1 cargo test --package cas_engine \
    --test metamorphic_simplification_tests -- metatest_individual --ignored --nocapture

# Comparar vs baseline (en CI o antes de PR)
METATEST_DIAG=1 METATEST_SNAPSHOT=1 cargo test --package cas_engine \
    --test metamorphic_simplification_tests -- metatest_individual --ignored --nocapture
```

### Output de Diagnóstico

```
🔍 Filter Coverage: 12/97 snapshot (12.4%) | 12/419 total loaded (2.9%)
   Top-5 by filtered_rate (potential 'cheating' filters):
    1. [  50%] valid=100/200 gt(0) → exp(ln(x))
    2. [  50%] valid=100/200 ge(0) → 1/(sqrt(x)+1)
    ...

📊 Baseline Comparison (METATEST_SNAPSHOT=1):
   Current: 97 | Baseline: 97 | Regressions: 0 | New: 0 | Missing: 0
```

### Validación de Configuración

Si los parámetros de test cambian (samples, tolerancias, rango), el sistema detecta el mismatch:

```
⚠️  Config mismatch detected!
   Baseline cfg_hash: b1e48281af9a6844
   Current cfg_hash:  XXXX...
   Run with METATEST_UPDATE_BASELINE=1 to regenerate.
→ panic!("Baseline/config mismatch")
```

Esto evita falsos positivos/negativos por cambios de configuración.

### Detección de Regresión

El sistema falla CI si ocurre cualquiera de:

| Regla | Condición | Significado |
|-------|-----------|-------------|
| Category worsens | `Ok → Fragile/NeedsFilter/ConfigError/BugSignal` | Identidad empeoró |
| Asymmetric appears | `asymmetric: 0 → >0` | Bug potencial introducido |
| Invalid rate spike | `+5% absoluto` | Más fallos de evaluación |
| Filter rate spike | `+20% absoluto` | Filtro se volvió más restrictivo |
| Mismatches appear | `0 → >0` | Discrepancias numéricas nuevas |
| Config mismatch | `cfg_hash` diferente | Parámetros de test cambiaron |

### Ranking de Categorías

```
Ok < Fragile < NeedsFilter < ConfigError < BugSignal
```

Una transición hacia la derecha es regresión; hacia la izquierda es mejora.

### Flujo de Trabajo

1. **Desarrollo local**: Hacer cambios al engine
2. **Verificar**: `METATEST_SNAPSHOT=1` para comparar vs baseline
3. **Si hay regresiones**: Investigar y corregir
4. **Si config mismatch**: Decidir si actualizar baseline conscientemente
5. **Si todo Ok**: `METATEST_UPDATE_BASELINE=1` para actualizar
6. **Commit**: Incluir cambios al baseline en el PR (o añadir a `.gitignore` si es local)

---

## Shuffle Canonicalization Test

Verifica que `simplify(E) == simplify(shuffle(E))` para detectar bugs de canonicalización orden-dependiente.

### Dual Check

| Check | Propósito | Resultado esperado |
|-------|-----------|-------------------|
| **Semántico** | `simplify(E) ≡ simplify(shuffle(E))` numéricamente | **0 failures** (bug si falla) |
| **Estructural** | `simplify(E) == simplify(shuffle(E))` exacto | Métrica (ideal: 0) |

### Comandos

```bash
# Modo métrica (no bloquea, reporta)
METATEST_SHUFFLE=1 cargo test --package cas_engine \
    --test metamorphic_simplification_tests -- metatest_shuffle --ignored --nocapture

# Modo estricto (falla si hay structural diffs)
METATEST_SHUFFLE=1 METATEST_STRICT_CANON=1 cargo test --package cas_engine \
    --test metamorphic_simplification_tests -- metatest_shuffle --ignored --nocapture
```

### Output

```
🔀 Shuffle Canonicalization Test
   Mode: METRIC (report only)
📊 Shuffle Results:
   Tested: 778 expressions
   Semantic failures: 0 (MUST be 0)
   Structural diffs: 164 (canonicalization gaps)
✅ Semantic checks passed. 164 structural diffs (non-blocking).
```

### Variables de Entorno

| Variable | Valor | Descripción |
|----------|-------|-------------|
| `METATEST_SHUFFLE` | `1` | Activa el test de shuffle |
| `METATEST_STRICT_CANON` | `1` | Falla CI si hay structural diffs |

---

## MetaTransform Test

Verifica que identidades se mantienen bajo transformaciones: `A(T(x)) ≡ B(T(x))`.

### Transforms Disponibles

| Transform | Descripción | Uso |
|-----------|-------------|-----|
| `scale:k` | x → k·x | Detecta errores de paridad, trig odd/even |
| `shift:k` | x → x+k | Desplaza dominio, puede acercarse a polos |
| `square` | x → x² | Cambia dominio fuerte (x≥0) |

### Comandos

```bash
# Defaults: scale(2), scale(-1)
METATEST_TRANSFORMS_DEFAULT=1 cargo test --package cas_engine \
    --test metamorphic_simplification_tests -- metatest_transform --ignored --nocapture

# Custom transforms
METATEST_TRANSFORMS=scale:2,shift:1,square cargo test --package cas_engine \
    --test metamorphic_simplification_tests -- metatest_transform --ignored --nocapture
```

### Output

```
🔄 MetaTransform Test
   Transforms: ["scale(2)", "scale(-1)"]
📊 Transform Results:
   Total tests: 778
   Passed: 775 (99.6%)
   Skipped (bucket gate): 0
   Semantic failures: 3
```

### Variables de Entorno

| Variable | Valor | Descripción |
|----------|-------|-------------|
| `METATEST_TRANSFORMS` | `scale:2,shift:1` | Lista de transforms |
| `METATEST_TRANSFORMS_DEFAULT` | `1` | Usa defaults (scale:2, scale:-1) |
| `METATEST_TRANSFORM_MIN_VALID_FACTOR` | `0.6` | Factor para min_valid |

### Bucket Gating

- **Unconditional/ConditionalRequires**: Todos los transforms
- **BranchSensitive**: Solo `scale(2)` (evita cruces de rama)

## Substitution-Based Metamorphic Tests

Verifica que las identidades se mantienen cuando la variable se reemplaza por sub-expresiones arbitrarias:
`A(S(u)) ≡ B(S(u))` para cada par de identidad `(A,B)` y cada sustitución `S`.

### Arquitectura

```
┌──────────────────────────────┐     ┌──────────────────────────────┐
│  substitution_identities.csv │     │ substitution_expressions.csv │
│  (~75 pares: trig, log,     │     │  (~20 sustituciones: trig,   │
│   algebra, radical, etc.)    │     │   poly, exp, rational, etc.) │
└──────────────┬───────────────┘     └──────────────┬───────────────┘
               │                                    │
               └────────────┬───────────────────────┘
                            ▼
               ┌────────────────────────┐
               │  Producto cartesiano   │
               │  75 × 20 = 1500       │
               │  combinaciones         │
               └────────────┬───────────┘
                            ▼
               ┌────────────────────────┐
               │  3-tier verification:  │
               │  NF → Symbolic → Num  │
               └────────────────────────┘
```

### CSV: substitution_identities.csv

```csv
# Format: exp,simp,var,mode
sin(2*x),2*sin(x)*cos(x),x,g
ln(x^2),2*ln(x),x,g
(x+1)^2,x^2 + 2*x + 1,x,g
```

Familias incluidas: Weierstrass, Log/Exp, Double/Triple/Half Angle, Pythagorean (extendido),
Power Reduction, Binomial, Power Rules, Difference of Squares/Cubes, Fraction Simplification,
Quotient, Negation, Cofunction, Reciprocal Trig, Sum/Product-to-Product/Sum, Log Rules,
Exponential, Algebraic, Rational, Sqrt/Radical, Shift/Phase, Even/Odd Powers.

### CSV: substitution_expressions.csv

```csv
# Format: expr,var,label
sin(u),u,trig
u^2 + 1,u,poly
exp(u),u,exp_log
u/(u + 1),u,rational
```

Clases de sustitución: `trig`, `inv_trig`, `poly`, `exp_log`, `composed`, `rational`, `simple`.

### Comandos

```bash
# Test de sustitución completo
cargo test --release -p cas_engine \
    --test metamorphic_simplification_tests metatest_csv_substitution \
    -- --ignored --nocapture

# Con tabla cross-product (familia × clase de sustitución)
METATEST_TABLE=1 cargo test --release -p cas_engine \
    --test metamorphic_simplification_tests metatest_csv_substitution \
    -- --ignored --nocapture

# Con ejemplos verbose de numeric-only
METATEST_VERBOSE=1 cargo test --release -p cas_engine \
    --test metamorphic_simplification_tests metatest_csv_substitution \
    -- --ignored --nocapture
```

### Cross-Product Table (METATEST_TABLE=1)

Con la variable `METATEST_TABLE=1`, el test imprime una tabla de cobertura:

```
╔═══════════════════════╤═══════╤═══════╤══════╤═══════╤═══════╤══════╗
║ Identity Family       │ trig  │ poly  │ e/ln │ comp  │ ratio │ simp ║
╠═══════════════════════╪═══════╪═══════╪══════╪═══════╪═══════╪══════╣
║ Pythagorean           │ 2/0/0 │ 5/0/0 │ 2/0/0│ 4/0/0 │ 2/0/0 │ 3/0 ║
║ Double Angle          │ 1/1/0 │ 4/0/1 │ ...  │ ...   │ ...   │ ... ║
╚═══════════════════════╧═══════╧═══════╧══════╧═══════╧═══════╧══════╝
Legend: NF/Proved/Numeric  (Failed shown as ❌)
```

Filas = familias de identidades, Columnas = clases de sustitución.
Cada celda muestra `NF-convergent / Proved-symbolic / Numeric-only`.

---

## Round-Trip Metamorphic Tests

Verifica las propiedades de ida y vuelta de las transformaciones del engine:

### Chain 1: `simplify(expand(x)) ≡ simplify(x)` (idempotencia)

Propiedad: expandir una expresión y luego simplificarla debe dar el mismo resultado
que simplificarla directamente.

### Chain 2: `expand(factor(x)) ≡ x` (round-trip)

Propiedad: factorizar una expresión y luego expandirla debe devolver la expresión original
(como polinomio equivalente).

### Expresiones de Test

53 expresiones curadas en 4 familias:

| Familia | Ejemplos | Count |
|---------|----------|-------|
| Polynomial | `x^2 - 1`, `x^3 + 8`, `x^4 - 5*x^2 + 4` | 25 |
| Product | `(x+1)*(x-1)`, `(a+b)^3`, `x*(x+1)*(x+2)` | 15 |
| Trig | `sin(x)^2 + cos(x)^2`, `sin(2*x)` | 8 |
| Mixed | `(x+1)^2 - (x-1)^2`, `(x-1)*(x^2+x+1)` | 5 |

### 3-Tier Verification

Misma filosofía que los tests de combinaciones:

1. **NF-convergent** (📐): `compare_expr(simplify(LHS), simplify(RHS)) == Equal`
2. **Proved-symbolic** (🔢): `simplify(LHS - RHS) == 0`
3. **Numeric-only** (🌡️): `eval_f64_checked` en puntos de muestreo

### Comandos

```bash
# Ejecutar todos los chains
cargo test --release -p cas_engine --test round_trip_tests \
    -- --ignored --nocapture

# Con detalle por expresión
ROUNDTRIP_VERBOSE=1 cargo test --release -p cas_engine --test round_trip_tests \
    -- --ignored --nocapture

# Solo Chain 1 (expand→simplify)
cargo test --release -p cas_engine --test round_trip_tests \
    roundtrip_expand_simplify -- --ignored --nocapture

# Solo Chain 2 (factor→expand)
cargo test --release -p cas_engine --test round_trip_tests \
    roundtrip_factor_expand -- --ignored --nocapture
```

### Variables de Entorno

| Variable | Default | Descripción |
|----------|---------|-------------|
| `ROUNDTRIP_VERBOSE` | (desactivado) | Muestra LaTeX de cada paso de transformación |

### Baseline (Feb 2026)

| Chain | NF-conv | Proved | Numeric | Skipped | Failed |
|-------|---------|--------|---------|---------|--------|
| expand→simplify | 45 | 5 | 3 | 0 | 0 |
| factor→expand | 28 | 8 | 2 | 7 | 0 |

> [!NOTE]
> **Skipped** en Chain 2 indica expresiones donde `factor()` no encontró factorización
> (devolvió la misma expresión). Esto es normal para expresiones irreducibles o multivariant.

---

## Archivo de Referencia

```
crates/cas_engine/tests/
├── identity_pairs.csv                   # Base de identidades (~400)
├── substitution_identities.csv          # Identidades para sustitución (~110)
├── substitution_expressions.csv         # Sub-expresiones de sustitución (~34)
├── metamorphic_simplification_tests.rs  # Implementación principal
├── round_trip_tests.rs                  # Tests de ida y vuelta
├── baselines/metatest_baseline.jsonl    # Baseline de regresión
└── metatest.log                         # Historial de ejecuciones
```

