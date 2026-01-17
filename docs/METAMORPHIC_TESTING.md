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
│  (351+ identidades: algebra, trig, log, rationales, etc.)   │
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

```bash
# Pequeño (CI)
cargo test metatest_csv_combinations_small

# Completo
cargo test metatest_csv_combinations_full --ignored
```

---

## Interpretación de Resultados

### Salida Típica

```
📊 Individual Identity Results:
   Total tested: 351
   ✅ Symbolic: 245 (69%)
   ❌ Failed: 0
   ⏭️  Skipped: 18
```

### Qué Significan

- **Symbolic**: Engine produjo la forma canónica esperada
- **Failed**: Ni simbólico ni numérico equivalentes (bug o identidad incorrecta)
- **Skipped**: Identidad requiere modo `assume` y test corre en `generic`

### Mejorar el Engine

1. **Aumentar Symbolic %**: Añadir reglas de simplificación
2. **Reducir Failed**: Verificar identidad matemática o corregir regla
3. **Investigar asymmetric_invalid**: Señal de bug en evaluación

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
| `METATEST_DIAG` | `0`/`1` | `0` | Habilita diagnóstico detallado |
| `METATEST_LEGACY_BUCKET` | `unconditional`/`conditional_requires` | `conditional_requires` | Bucket para CSV 4-col |
| `METATEST_SNAPSHOT` | `0`/`1` | `0` | Compara resultados vs baseline |
| `METATEST_UPDATE_BASELINE` | `0`/`1` | `0` | Regenera archivo baseline |
---

## Sistema de Baseline JSONL (Regresión Tracking)

El sistema de baseline permite detectar regresiones en la calidad del engine entre commits.

### Archivo Baseline

```
crates/cas_engine/tests/baselines/metatest_baseline.jsonl
```

Cada línea es un JSON con el snapshot de una identidad:

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

### Output de Comparación

```
📊 Baseline Comparison (METATEST_SNAPSHOT=1):
   Current: 97 | Baseline: 97 | Regressions: 0 | New: 0 | Missing: 0
```

### Detección de Regresión

El sistema falla CI si ocurre cualquiera de:

| Regla | Condición | Significado |
|-------|-----------|-------------|
| Category worsens | `Ok → Fragile/NeedsFilter/ConfigError/BugSignal` | Identidad empeoró |
| Asymmetric appears | `asymmetric: 0 → >0` | Bug potencial introducido |
| Invalid rate spike | `+5% absoluto` | Más fallos de evaluación |
| Filter rate spike | `+20% absoluto` | Filtro se volvió más restrictivo |
| Mismatches appear | `0 → >0` | Discrepancias numéricas nuevas |

### Ranking de Categorías

```
Ok < Fragile < NeedsFilter < ConfigError < BugSignal
```

Una transición hacia la derecha es regresión; hacia la izquierda es mejora.

### Flujo de Trabajo

1. **Desarrollo local**: Hacer cambios al engine
2. **Verificar**: `METATEST_SNAPSHOT=1` para comparar vs baseline
3. **Si hay regresiones**: Investigar y corregir
4. **Si todo Ok**: `METATEST_UPDATE_BASELINE=1` para actualizar
5. **Commit**: Incluir cambios al baseline en el PR

---

## Identidades de Regresión (Soundness Guards)

Identidades "idempotentes" que garantizan que reglas peligrosas no se apliquen incorrectamente:

```csv
# abs() no debe eliminarse de trig sin proof de signo
abs(sin(x)),abs(sin(x)),x,g
abs(cos(x)),abs(cos(x)),x,g
abs(sin(x/2)),abs(sin(x/2)),x,g
abs(cos(x/2)),abs(cos(x/2)),x,g
```

Si algún refactor futuro añade `abs(u) → u` incorrecto, CI fallará.

---

## Guía de Migración Legacy → 7-col

### Criterios para Migrar

1. **asymmetric_invalid > 0** → Investigar bug primero
2. **invalid_rate alto** → Añadir `filter` apropiado
3. **Identidades de ramas** → `branch_mode=ModuloPi/Modulo2Pi`

### Filtros Comunes

| Situación | Filter |
|-----------|--------|
| `ln(x)`, `log(x)` | `gt(0.0)` |
| `sqrt(x)` | `ge(0.0)` |
| Polos en x=0 | `away_from(0.0;eps=0.05)` |
| Polos en ±π/2 | `away_from(1.5707963;-1.5707963;eps=0.01)` |
| arctan con división | `abs_lt(0.9)` |
| Rango específico | `range(0.1;3.0)` |
| Combinado | `abs_lt_and_away(0.95;1.0;-1.0;eps=0.1)` |

---

## Archivo de Referencia

```
crates/cas_engine/tests/
├── identity_pairs.csv              # Base de identidades
├── metamorphic_simplification_tests.rs  # Implementación
└── metatest.log                    # Historial de ejecuciones
```
