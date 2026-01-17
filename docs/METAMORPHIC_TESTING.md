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
```

### FilterSpec Enum (Runtime)

```rust
enum FilterSpec {
    None,                                               // Sin filtro
    AbsLt { limit: f64 },                               // |x| < limit
    AwayFrom { centers: Vec<f64>, eps: f64 },           // |x - c| > eps
    AbsLtAndAway { limit: f64, centers: Vec<f64>, eps: f64 },
}

impl FilterSpec {
    fn accept(&self, x: f64) -> bool { ... }
}
```

### Uso en Tests

```rust
// Durante muestreo numérico:
if !pair.filter_spec.accept(x) {
    stats.filtered_out += 1;
    continue;
}
```

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
| `METATEST_LEGACY_BUCKET` | `unconditional`/`conditional_requires` | `conditional_requires` | Bucket para CSV 4-col |

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
| Polos en x=0 | `away_from(0.0;eps=0.05)` |
| Polos en ±π/2 | `away_from(1.5707963;-1.5707963;eps=0.01)` |
| arctan con división | `abs_lt(0.9)` |
| Combinado | `abs_lt_and_away(0.95;1.0;-1.0;eps=0.1)` |

---

## Archivo de Referencia

```
crates/cas_engine/tests/
├── identity_pairs.csv              # Base de identidades
├── metamorphic_simplification_tests.rs  # Implementación
└── metatest.log                    # Historial de ejecuciones
```
