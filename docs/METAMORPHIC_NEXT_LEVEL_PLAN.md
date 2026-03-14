# Metamorphic Next-Level Plan

> Plan de trabajo para llevar la batería metamórfica de "buena y útil" a "más completa, explicativa y resistente a falsos negativos".

## Objetivo

Esta fase no busca cambiar primero el motor, sino:

1. Aumentar la cobertura metamórfica útil.
2. Hacer que los resultados del harness sean más interpretables.
3. Separar mejor:
   - fallos reales del motor
   - límites de prueba simbólica
   - límites del checker numérico
4. Convertir patrones repetidos de `numeric-only` en regresiones curadas y mantenibles.

El foco está en:
- [/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/metamorphic_simplification_tests.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/metamorphic_simplification_tests.rs)
- [/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/metamorphic_equation_tests.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/metamorphic_equation_tests.rs)
- [/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/identity_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/identity_pairs.csv)
- [/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/substitution_expressions.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/substitution_expressions.csv)
- [/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/contextual_pairs.csv](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/contextual_pairs.csv)
- [/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/numeric_only_diagnostic.rs](/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/numeric_only_diagnostic.rs)

## Estado Actual

La batería metamórfica ya tiene varias fortalezas:

- Existe corpus base de identidades.
- Existe combinación automática de identidades.
- Existe fallback numérico útil.
- Existe ya una suite contextual.
- El harness distingue al menos:
  - `NF-convergent`
  - `proved-symbolic`
  - `numeric-only`
  - `mismatch`
  - `incomplete`
- El caso `proved-composed` ya no enmascara el fallback numérico: solo debe contar como recurso final, no como sustituto de la prueba del motor.

Las debilidades observadas hoy son:

- `numeric-only` mezcla causas distintas.
- Las familias difíciles siguen entrando por el mismo carril que identidades simples.
- El checker numérico sigue siendo demasiado genérico para familias con polos, dominio o cancelaciones fuertes.
- Algunas expresiones difíciles solo aparecen como combinaciones globales y no como regresiones curadas.
- Falta más coverage de contratos, no solo de equivalencia algebraica.

## Principio Rector

No meter más complejidad indiscriminada en el corpus global.

Las identidades "difíciles" deben probarse:

- en suites contextuales curadas
- con muestreo apropiado
- con clasificación separada

No queremos:

- inflar `numeric-only` artificialmente
- provocar falsos `failed`
- ni volver opaca la lectura del harness

## Problemas Recurrentes Detectados en Logs

### 1. Re-composición racional

Familias repetidas:

- `1/x + 1/(x+1) <-> (2*x+1)/(x*(x+1))`
- `1/(x-1) + 1/(x+1) <-> 2*x/(x^2-1)`

Síntomas:

- equivalencia correcta
- residual simbólico grande
- fallback numérico sí valida

### 2. Trig en contexto

Familias repetidas:

- `sec(x)^2 - tan(x)^2 <-> 1`
- `sin(u) + sin(3*u) <-> 2*sin(2*u)*cos(u)`
- `tanh(2*u) <-> 2*tanh(u)/(1+tanh(u)^2)`

Síntomas:

- la identidad base es conocida
- incrustada en sumas grandes no siempre cierra simbólicamente

### 3. Factorización / expansión polinómica

Familias repetidas:

- `(u+1)^2 <-> u^2 + 2*u + 1`
- `(u-1)^5 <-> u^5 - 5*u^4 + 10*u^3 - 10*u^2 + 5*u - 1`
- `u^6 - 1 <-> (u^2+u+1)(u^2-u+1)(u+1)(u-1)`
- `u^3 + u^2 + u + 1 <-> (u+1)(u^2+1)`

Síntomas:

- el motor no siempre conecta bien forma expandida y factorizada cuando están dentro de otra identidad mayor

### 4. Multivariante compuesta

Familia repetida:

- `(x^2 + y^2)(a^2 + b^2) <-> (x*a + y*b)^2 + (x*b - y*a)^2`

Síntomas:

- en aislamiento es razonable
- en contexto aditivo con otro polinomio/factorización queda más frágil

### 5. Exp/logistic

Familia repetida:

- `exp(u)/(exp(u)+1) + 1/(exp(u)+1) <-> 1`

Síntomas:

- el cierre simbólico no siempre se consigue dentro de expresiones grandes
- el checker numérico suele salvarlo

## Plan por Fases

## Fase 1. Clasificación más fina del Harness

Estado:
- Parcialmente implementada.
- El harness de simplificación ya distingue `numeric-only` por causa en logs verbose:
  - `domain-sensitive`
  - `sampling-weak`
  - `multivar-context`
  - `symbolic-residual`
- El harness de ecuaciones ya no trata `NumericVerifyResult::Inconclusive` como si fuera validación numérica correcta:
  - Strategy 1 lo expone como `NumericInconclusive`
  - Strategy 3 (equation pairs) lo expone como `NumericInconclusive`
  - Strategy 2 lo baja a `Incomplete(NumericInconclusive)`
- Esto ya está conectado en:
  - combinaciones (`add/sub/mul/div`)
  - `substitution`
  - suites de pares contextuales/residuales
- Sigue pendiente refinar la heurística para separar mejor `sampling-weak` de casos puramente de dominio dentro del harness de ecuaciones, no solo distinguir `inconclusive` como bucket propio.

### Objetivo

Dejar de tratar `numeric-only` como un bucket único.

### Cambios propuestos

Introducir subcategorías o razones explícitas:

- `numeric_only_domain_sensitive`
- `numeric_only_sampling_weak`
- `numeric_only_symbolic_residual`
- `numeric_only_multivar_context`
- `proved_composed_fallback`

### Resultado esperado

Cuando una suite falle o quede en `numeric-only`, sabremos por qué.

### Criterio de salida

El resumen verbose debe poder agrupar `numeric-only` por causa, no solo por familia.

Estado actual del criterio:
- Cumplido para el harness de simplificación.

## Fase 2. Suites Contextuales Temáticas

Estado:
- Parcialmente implementada.
- Ya existen suites temáticas nuevas:
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/contextual_rational_pairs.csv`
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/contextual_trig_pairs.csv`
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/contextual_polynomial_pairs.csv`
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/contextual_radical_pairs.csv`
- Ambas están conectadas al harness de simplificación y tienen runners dedicados en:
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/metamorphic_simplification_tests.rs`
- La suite contextual general se mantiene como umbrella suite; las nuevas temáticas sirven para aislar mejor familias recurrentes sin meterlas todas en el corpus global.

### Objetivo

Sacar del corpus global los contextos difíciles y hacerlos explícitos.

### Nuevos CSV propuestos

- `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/contextual_rational_pairs.csv`
- `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/contextual_trig_pairs.csv`
- `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/contextual_polynomial_pairs.csv`
- `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/contextual_radical_pairs.csv`

### Regla

`substitution_expressions.csv` debe contener solo sustituciones "seguras" y genéricas.

Ejemplos de expresiones que deben vivir preferentemente en suites contextuales:

- `sec(u)^2 - tan(u)^2`
- `(u - 1)^5`
- `u^6 - 1`
- `sin(u) + sin(3*u)`
- `exp(u)/(exp(u)+1) + 1/(exp(u)+1)`

### Resultado esperado

Más cobertura útil sin degradar artificialmente la suite global.

### Criterio de salida

Cada familia recurrente del log debe tener al menos un test contextual dedicado.

Estado actual del criterio:
- Cumplido para las familias racionales, trig, polinómicas y radicales más repetidas.
- Siguiente paso natural: decidir si parte de `contextual_pairs.csv` debe migrarse a estas suites temáticas o mantenerse como umbrella suite curada.

## Fase 3. Mejor Oracle Numérico

Estado:
- Parcialmente implementada.
- El harness de simplificación ya tiene retries relajados no solo para `1var`, sino también para:
  - `2var`
  - `2var` con variables fijadas (`*_with_fixed`)
- Esto reduce falsos fallos/inconclusivos en slices multivariantes y mejora especialmente el camino de fallback de `nvar`, que depende de slices `1d/2d`.
- El harness de ecuaciones ya usa un sampler determinista con tres perfiles:
  - `interior` para dominios acotados (`arcsin`, `acos`)
  - `general` para racionales/polinomios
  - `positive` para logs/raíces
- Strategy 2 también aprovecha mejor hints de dominio ya presentes en `identity_pairs.csv`
  (por ejemplo `ge(0.0)`) y un heurístico pequeño para identidades con `arcsin/arccos`,
  para reclasificar `numeric inconclusive` como `domain-changed` cuando proceda.
- El runner de combinaciones `add/sub` ya usa:
  - todas las variables libres reales de cada identidad combinada
  - la prueba simbólica rica `prove_zero_from_metamorphic_texts(...)` antes del fallback numérico
- Resultado observable: el slice de referencia `METATEST_START_OFFSET=300` para `metatest_csv_combinations_add` pasó de `Numeric-only: 24` a `Numeric-only: 0`.
- Sigue pendiente una mejora más fuerte y explícita de muestreo domain-aware por familia, en vez de reutilizar solo filtros genéricos.

### Objetivo

Reducir falsos `failed` o `numeric-only` débiles cuando el problema es muestreo.

### Mejoras propuestas

#### 3.1. Muestreo domain-aware

Para familias racionales:

- evitar puntos demasiado cercanos a polos
- evitar muestras donde el denominador combinado se acerque a cero en cualquiera de las ramas

Para radicales/log:

- evitar muestras pegadas al borde de dominio
- distinguir mejor entre "punto inválido esperable" y "asimetría real"

#### 3.2. Muestreo temático por familia

- racional/polinómico
- trig periódica
- exp/log
- radicales

#### 3.3. Sampling multivariante explícito

Si la identidad usa `x;y` o más variables:

- evitar degradarla artificialmente al mismo muestreo univariante
- usar tuples de muestras reproducibles

#### 3.4. Exact checks cuando el caso lo permita

Para algunas familias:

- racionales simples
- polinomios pequeños

podemos usar un chequeo más estructural/exacto antes del floating-point.

### Criterio de salida

Menos `numeric-only` explicables por sampling débil y menos `failed` por polos.

Estado actual del criterio:
- Parcialmente cumplido a nivel de infraestructura del harness.
- Falta aún medir y, si hace falta, ajustar familias concretas (`rational`, `radical`, `exp/logistic`) con filtros específicos por dominio.

## Fase 4. Contratos, no solo equivalencia

Estado:
- Parcialmente implementada.
- Ya existe una suite explícita de contrato de idempotencia en:
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/idempotence_expressions.csv`
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/metamorphic_simplification_tests.rs`
- Ya existe también una primera suite explícita de preservación de `required_conditions` en:
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/requires_contract_expressions.csv`
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/metamorphic_simplification_tests.rs`
- Ya existe una primera extensión mode-aware de `required_conditions` en:
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/requires_mode_contract_expressions.csv`
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/metamorphic_simplification_tests.rs`
- Ya existe una primera suite cruzada de `DomainMode × ValueDomain` en:
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/semantic_axes_contract_expressions.csv`
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/metamorphic_simplification_tests.rs`
- Ya existe una primera suite explícita de trazabilidad de `assumption_events` en:
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/assumption_trace_contract_expressions.csv`
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/metamorphic_simplification_tests.rs`
- Ya existe además un runner agregado de Fase 4 en:
  - `metatest_simplify_phase4_contract_suites`
  - dentro de `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/metamorphic_simplification_tests.rs`
- Ya existe una primera suite explícita de contrato de `warnings` en:
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/warnings_contract_expressions.csv`
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/metamorphic_simplification_tests.rs`
- Ya existe una primera suite explícita de `transparency signals` en:
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/transparency_signal_contract_expressions.csv`
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/metamorphic_simplification_tests.rs`
- Ya existe una primera suite explícita de `branch transparency` en:
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/branch_transparency_contract_expressions.csv`
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/metamorphic_simplification_tests.rs`
- Ya existe una primera suite explícita de `semantic behavior` en:
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/semantic_behavior_contract_expressions.csv`
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/metamorphic_simplification_tests.rs`
- El contrato actual de `required_conditions` es deliberadamente de no-endurecimiento:
  - la segunda simplificación no puede introducir `requires` ni warnings nuevos
  - sí puede podar metadata redundante o ligada al historial de la primera simplificación
- La extensión mode-aware confirma además una semántica importante del sistema:
  - `Strict` no significa necesariamente “sin requires”
  - en varias formas sensibles, el motor conserva metadata explícita de dominio aunque el rewrite quede bloqueado o sea conservador
- La suite cruzada por ejes ha dejado otra semántica explícita:
  - `ln(exp(x))` conserva `e^x > 0` como `required_condition` en todos los ejes actualmente cubiertos
  - eso incluye `RealOnly` y `ComplexEnabled`, no solo `Strict`
- La misma suite ya incluye al menos un positivo real de `warning_present`:
  - `log(b, b^x)` en `RealOnly + Assume`
  - además de warning, conserva `b - 1 ≠ 0` como `required_condition`
- La matriz por ejes ya cubre también un caso multivariante útil de producto logarítmico:
  - `ln(a^2 * b^3) - 2*ln(a) - 3*ln(b)` en `RealOnly + Generic/Assume`
  - preserva `required_conditions` y no introduce `warnings`
- La suite de trazabilidad cubre ya dos familias estructuradas que no pasan bien por `domain_warnings`:
  - `nonzero`, `positive`, `defined` y `nonnegative` en `DomainMode::Assume`
  - `principal_range` en `InverseTrigPolicy::PrincipalValue`
- Hallazgo de canal para la Fase 4:
  - no toda la semántica de "warning" llega por `output.domain_warnings`
  - familias como `mul_zero` y `0/risky` hoy viven principalmente en `step.assumption_events()`
  - por eso no deben mezclarse sin más en `warnings_contract_expressions.csv`
  - ese hueco ya queda cubierto por `transparency_signal_contract_expressions.csv`, que unifica:
    - `domain_warnings`
    - `assumption_events` de kinds `RequiresIntroduced` / `DerivedFromRequires`
- Investigación cerrada en esta tanda:
  - `ComplexPrincipalBranch` existe en el modelo de supuestos, pero no aparece hoy como emisor vivo en código de producción
  - por tanto no se ha forzado una suite metamórfica artificial para ese kind
  - la ampliación útil de `assumption_trace` se ha hecho sobre kinds con emisor vivo real
- El contrato actual de `warnings` usa la misma idea de no-endurecimiento:
  - la primera simplificación debe respetar si el caso espera warning o no
  - la segunda simplificación no puede introducir warnings nuevos
  - sí puede podar warnings ligados al paso de simplificación original
- El contrato de `transparency signals` extiende esa misma idea al canal combinado de transparencia:
  - acepta señal por `domain_warnings`
  - o por `assumption_events` cuando esa información no se promueve al canal de warnings
  - la segunda simplificación no puede introducir señales nuevas por ninguno de los dos canales
- La suite de `branch transparency` acota esa idea al caso educativo de `InverseTrigPolicy::PrincipalValue`:
  - solo cuenta señales visibles de tipo `BranchChoice`
  - verifica que las composiciones `inv_trig∘trig` sí surfacen esa decisión
  - y que las composiciones seguras `trig∘inv_trig` no quedan contaminadas con señales de branch
- La suite de `semantic behavior` fija además el resultado esperado bajo `ValueDomain × DomainMode`:
  - cuándo `ln(exp(x))` debe simplificar o preservarse
  - cuándo `exp(ln(x))` debe simplificar o preservarse
  - cuándo `log(b, b^x)` debe bloquear o simplificar
  - y algunos casos compuestos/multivariantes (`2 * x`, `0`)
  - exige además que una segunda simplificación no rompa ese comportamiento esperado
- La suite distingue:
  - estabilidad exacta (`simplify(simplify(E)) == simplify(E)`)
  - estabilidad simbólica
  - estabilidad solo numérica
  - inconclusivos/fallos reales
- Sigue pendiente ampliar estos contratos a más familias y encontrar/curar casos con warnings positivos reales en `ComplexEnabled` vía `domain_warnings`, no solo por `assumption_events`.
- Ya existe también una primera suite explícita de contratos curados para Strategy 2 de ecuaciones en:
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/equation_transform_contract_cases.csv`
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/metamorphic_equation_tests.rs`
- Esa suite fija dos clases de comportamiento del harness de ecuaciones:
  - transformaciones seguras que deben quedar en `ok`
  - transformaciones con contracción real de dominio que deben reclasificarse como `domain-changed`, no como `mismatch`
- Existe además una suite pequeña de preservación de `SolutionSet` coarse kind en ecuaciones:
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/equation_solution_kind_contract_cases.csv`
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/metamorphic_equation_tests.rs`
- Esa suite comprueba que una identidad tier-0 inofensiva no rompa contratos ya fijados del solver:
  - `discrete`
  - `empty`
  - `allreals`
  - `conditional`
  - `residual`
- incluyendo ya familias solver no triviales como `symbolic_linear`, `linear_collect`,
    `reciprocal`, el camino `wildcard residual` con RHS simbólico y el split de modo
    en `nested product-zero`
- La ampliación actual fija explícitamente esa semántica de modo:
  - `(a^x - b) * (x - 1) = 0` en `Assume + Real` debe seguir resolviendo como `discrete`
  - la misma ecuación en `Generic + Real` debe seguir quedando en `residual`
  - una identidad tier-0 inocua no puede borrar esa diferencia
- También quedan absorbidos ya varios prechecks básicos de exponenciales:
  - `1^x = 1` debe seguir en `allreals`
  - `1^x = 5` debe seguir en `empty`
  - `2^x = -5` debe seguir en `empty`
- Nota de borde ya observada:
  - `2^x = 0` sí queda cubierto por tests dedicados del solver como `Empty`
  - pero al inyectar una identidad tier-0 sin simplificar (`sec(t)^2 - tan(t)^2 = 1`),
    el solve path puede reclasificar la ecuación transformada como `Conditional`
    por el parámetro extra de la identidad
  - por ahora no se fija como contrato estricto de `solution_kind`; conviene tratarlo
    como edge de simplificación previa, no como invariante robusta del harness
- Existe también una primera suite explícita de preservación de `required_conditions` en ecuaciones:
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/equation_required_contract_cases.csv`
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/metamorphic_equation_tests.rs`
- Esa suite fija que una identidad tier-0 inofensiva no borre ni invente firmas estables de `required_conditions`
  en casos como `2^x = y`, `2^x = sqrt(y)` y ecuaciones triviales sin requisitos.
- La ampliación actual añade además familias `conditional` reales del solver:
  - `(x-1)/(x+1) = y`
  - `A = P + P*r*t`
  - `1/R = 1/R1 + 1/R2`
- Hallazgo fijado por la suite:
  - esas rutas sí pueden tener un suelo mínimo estable de `required_conditions`
    ligado a denominadores de entrada (`nonzero(x + 1)`, `nonzero(R)`, `nonzero(R1)`,
    `nonzero(R2)`), aunque no surfacen warnings ni transparencia educativa.
  - La variante `log-linear` con testigo de dominio en el RHS (`2^x = sqrt(y)`)
    sí queda bien fijada en `required/assumed/warnings/transparency`, pero no en
    `solution_kind`: la ecuación transformada puede reclasificarse a `Conditional`
    por el parámetro extra de la identidad tier-0.
- Existe además una suite explícita de preservación de señales de asunción del solver:
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/equation_assumption_contract_cases.csv`
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/metamorphic_equation_tests.rs`
- Esa suite fija que una identidad tier-0 inofensiva no borre firmas estables de
  `diagnostics.assumed` en caminos exponenciales que sí introducen suposiciones
  reales del solver, como `positive(base)` y `positive(rhs)` en modo `Assume`.
- La ampliación actual añade también casos de ausencia correcta en familias
  `conditional` del solver (`symbolic_linear`, `linear_collect`, `reciprocal`,
  `log-linear` con `sqrt(y)`),
  para fijar que esas rutas no empiecen a emitir `diagnostics.assumed` espurios.
- Además, la rama `wildcard residual` (`(-2)^x = y` en `Assume + Wildcard`) ya
  queda fijada como ausencia correcta en este canal: no introduce
  `diagnostics.assumed` estructurados aunque sí surfacing transparencia educativa.
- Para evitar falsos rojos por ruido de renderizado (`y + 0`, etc.), el harness
  normaliza el target de cada señal con el propio simplificador antes de comparar.
- Existe además una suite explícita de preservación de `assumed_records` agregados:
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/equation_assumption_record_contract_cases.csv`
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/metamorphic_equation_tests.rs`
- Esa suite fija que una identidad tier-0 inofensiva no rompa el resumen agregado
  que consumen UI/wire del solver (`positive(y)`, `positive(b)`, etc.), no solo
  el stream detallado de `diagnostics.assumed`.
- También cubre ya ausencia correcta de `assumed_records` en familias
  `conditional` donde el solver no debería inventar metadata agregada nueva,
  incluida la variante `2^x = sqrt(y)`.
- La misma idea ya cubre también `wildcard residual`: no debe inventar
  `assumed_records` agregados solo por caer en residual.
- Existe además una suite explícita de `warning preservation` del solver en el
  camino visible de render de `solve`:
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/equation_warning_contract_cases.csv`
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/metamorphic_equation_tests.rs`
- Como el solve path hoy no llena `domain_warnings`, esta suite fija el canal real
  que ve el usuario: la línea `⚠ Assumptions:` derivada de `solver_assumptions`
  bajo `AssumptionReporting::Summary`, y su ausencia correcta con `Off`.
- Ya cubre positivos reales tanto de `positive(rhs)` como de `positive(base)` en
  rutas exponenciales (`2^x = y`, `b^x = 5`, `b^x = y`).
- La ampliación actual fija además ausencia correcta de esta cabecera visible en
  familias `conditional` limpias (`symbolic_linear`, `linear_collect`, `reciprocal`,
  `2^x = sqrt(y)`).
- También fija ya ausencia correcta de `⚠ Assumptions:` en `wildcard residual`:
  hoy la transparencia de esa ruta entra por steps (`complex/preset`), no por la
  cabecera agregada de asunciones.
- Existe además una suite explícita para el canal detallado de asunciones del solver:
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/equation_assumption_section_contract_cases.csv`
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/metamorphic_equation_tests.rs`
- Esa suite fija la sección `ℹ️ Assumptions used:` en render de `solve` con
  `debug_mode=true` y `AssumptionReporting::Trace`, verificando presencia o
  ausencia de bullets estables sin depender del orden fino de pasos.
- Ya cubre positivos reales tanto de `rhs` como de `base` en esa sección:
  `y > 0`, `b > 0` y el caso combinado `b > 0; y > 0`.
- La ampliación actual añade también casos de ausencia correcta de esa sección en
  familias `conditional` limpias, para que un tier-0 inocuo no active trazas de
  asunción donde no debería.
- La variante `2^x = sqrt(y)` no se fija aquí como invariante robusta:
  bajo la ecuación transformada, el solve path puede surfacing un bullet ligado
  al testigo `sqrt(y)` aunque la ecuación original no lo haga. Se mantiene
  cubierta en `required/assumed/warnings/transparency`, pero no en
  `assumption_section`.
- Esa ausencia correcta cubre ya también `wildcard residual`: el solver puede
  surfacing transparencia educativa sin poblar la sección detallada
  `ℹ️ Assumptions used:`.
- Ya hay además una familia residual real distinta cubierta aquí:
  - `(a^x - b) * (x - 1) = 0` en `Generic + Real`
  - no llena `required/assumed/warnings` estructurados
  - pero sí deja un bullet estable `a > 0` en `trace`, ligado al intento de
    tomar logaritmos sobre la rama exponencial
  - esa diferencia ya queda fijada como contrato, en vez de asumir ausencia total
- Existe además una suite explícita de transparencia educativa del solver:
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/equation_transparency_contract_cases.csv`
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/metamorphic_equation_tests.rs`
- Esa suite fija que una identidad tier-0 inofensiva no rompa la presencia o
  ausencia de señal visible cuando el solver cae en `needs-complex`, en especial
  en `Assume + Wildcard` con residual guiado por “complex/preset”.
- La ampliación actual fija además ausencia correcta de esa señal en familias
  `conditional`/`reciprocal` que no necesitan ayuda educativa de dominio,
  incluida la ruta `2^x = sqrt(y)`.
- Esa suite convive ahora con una cobertura más completa de `wildcard residual`:
  la señal visible `complex/preset` sigue siendo el canal correcto, mientras que
  `required/assumed/warnings/assumption section` permanecen ausentes.
- En contraste, `nested product-zero` en `Generic + Real` queda ya fijado como
  residual sin transparencia visible `complex/preset`, lo que ayuda a separar
  dos clases distintas de residual:
  - residual por necesidad de dominio complejo (`wildcard residual`)
  - residual por rama exponencial simbólica no soportada (`nested product-zero`)
- Existe además una suite explícita de preservación de `output_scopes` del solver:
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/equation_scope_contract_cases.csv`
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/metamorphic_equation_tests.rs`
- Esa suite fija que identidades tier-0 inofensivas no borren ni inventen scopes
  estructurados del solver, en particular `rule(QuadraticFormula)` para rutas
  cuadráticas y ausencia de ese scope en ecuaciones lineales.
- Existe además una suite explícita de didáctica mínima en `steps` del solver:
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/equation_step_contract_cases.csv`
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/metamorphic_equation_tests.rs`
- Esa suite fija que identidades tier-0 inofensivas no rompan señales didácticas
  mínimas de rutas ya estabilizadas, hoy:
  - activación de `quadratic formula`
  - presencia de `collect` en `linear collect`
  - presencia de `collect terms in x` en la factorización de términos semejantes
    (`a*x + b*x = c`)
  - preservación de `take log base e of both sides`
    en la ruta `log-linear` (`2^x = y`)
  - preservación de `combine fractions` y `reciprocal` en la ruta recíproca
    (`1/R = 1/R1 + 1/R2`)
  - preservación de `multiply both sides by t` y `divide both sides by`
    en la descomposición de aislamiento de denominador (`P*V/T = n*R`)
- La suite ya soporta además ausencia correcta de pasos malos vía keywords
  prohibidas (`!keyword`) en el mismo CSV. Hoy fija también:
  - ausencia de `isolate denominator` en las rutas `reciprocal` e `ideal gas`
  - ausencia de `divide both sides by k` en `y = k*x/(x+c)`
- Las rutas más sensibles al wording fino siguen fuera del metamórfico; la suite
  de `steps` fija solo keywords pedagógicas gruesas y ausencias claras, no
  secuencias exactas completas.
- Existe además una primera suite explícita de contratos curados para Strategy 3 (pares equivalentes):
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/equation_pair_contract_cases.csv`
  - `/Users/javiergimenezmoya/developer/math/crates/cas_solver/tests/metamorphic_equation_tests.rs`
- Esa suite fija:
  - familias básicas del corpus general también en modo curado:
    - `2*x = 6` ↔ `6 = 2*x`
    - `x - 2 = 5` ↔ `3*(x - 2) = 15`
    - `x = 5` ↔ `x + 3 = 8`
  - el par paramétrico `y = 2*x + 1` ↔ `y - 1 = 2*x` como regresión verificada
  - el par `a*x = b` ↔ `x = b/a` como caso que hoy sigue necesitando `numeric fallback`
  - y además familias solver más ricas:
    - `A = P + P*r*t` ↔ `A = P*(1 + r*t)`
    - `a*x + b*x = c` ↔ `x*(a+b) = c`
    - `(x-1)/(x+1) = y` ↔ `x - 1 = y*(x + 1)`
    - `1/R = 1/R1 + 1/R2` ↔ `R = (R1*R2)/(R1+R2)`
    - `x^2 - 6*x = -9` ↔ `x^2 - 6*x + 9 = 0`
    - `|x| = 3` ↔ `|x| - 3 = 0`
- Ya existe además un runner agregado de contratos de ecuaciones:
  - `metatest_equation_contract_suites`
- El harness principal de ecuaciones ya está endurecido:
  - `Strategy 1` falla también si reaparecen `solver/parse/timeout`
  - `Strategy 3` falla también si reaparecen `solver/parse/timeout`
  - el benchmark agregado ya cuenta `errors` y `timeouts` de `Strategy 2/3` como fallo duro

Comando canónico actual de Fase 4:
- `cargo test -p cas_solver --test metamorphic_simplification_tests metatest_simplify_phase4_contract_suites -- --ignored --nocapture`
- `cargo test -p cas_solver --test metamorphic_equation_tests metatest_equation_transform_contracts -- --ignored --nocapture`
- `cargo test -p cas_solver --test metamorphic_equation_tests metatest_equation_solution_kind_contracts -- --ignored --nocapture`
- `cargo test -p cas_solver --test metamorphic_equation_tests metatest_equation_required_contracts -- --ignored --nocapture`
- `cargo test -p cas_solver --test metamorphic_equation_tests metatest_equation_assumption_contracts -- --ignored --nocapture`
- `cargo test -p cas_solver --test metamorphic_equation_tests metatest_equation_assumption_record_contracts -- --ignored --nocapture`
- `cargo test -p cas_solver --test metamorphic_equation_tests metatest_equation_warning_contracts -- --ignored --nocapture`
- `cargo test -p cas_solver --test metamorphic_equation_tests metatest_equation_assumption_section_contracts -- --ignored --nocapture`
- `cargo test -p cas_solver --test metamorphic_equation_tests metatest_equation_transparency_contracts -- --ignored --nocapture`
- `cargo test -p cas_solver --test metamorphic_equation_tests metatest_equation_scope_contracts -- --ignored --nocapture`
- `cargo test -p cas_solver --test metamorphic_equation_tests metatest_equation_step_contracts -- --ignored --nocapture`
- `cargo test -p cas_solver --test metamorphic_equation_tests metatest_equation_pair_contracts -- --ignored --nocapture`
- `cargo test -p cas_solver --test metamorphic_equation_tests metatest_equation_contract_suites -- --ignored --nocapture`

### Objetivo

Detectar regresiones que la equivalencia algebraica pura no ve.

### Suites nuevas o extensiones

#### 4.1. Required conditions preservation

Si `A ≡ B`, verificar:

- que no se pierdan `required_conditions` relevantes
- que no aparezcan condiciones espurias fuertes sin motivo

#### 4.2. Warning preservation

Verificar:

- que identidades equivalentes no introduzcan warnings falsos
- que no desaparezcan warnings importantes de dominio

#### 4.3. Solve invariance

Para equivalencias usadas dentro de ecuaciones:

- misma solución discreta
- misma familia de soluciones
- mismo comportamiento en `Strict / Generic / Assume`
- y, cuando la identidad contrae el dominio, reclasificación honesta a `domain-changed`

#### 4.4. Idempotence

- `simplify(simplify(x)) == simplify(x)`
- y su versión contextual cuando aplique

### Criterio de salida

Al menos una suite explícita de contratos, separada de la equivalencia pura.

## Fase 5. Numeric-only Diagnostics como Fuente de Trabajo

### Objetivo

Convertir el log en backlog accionable.

### Modo de trabajo

Cada tanda de `numeric-only` debe acabar en una de estas salidas:

1. se promueve a `contextual_pair` curado
2. se re-clasifica mejor en el harness
3. se corrige el checker numérico
4. se confirma que es una limitación real del motor y se documenta como tal

### Resultado esperado

El log deja de ser solo observación y pasa a ser pipeline de mejora.

## Acciones Concretas Inmediatas

## P0. Clasificación

- añadir motivo explícito a `numeric-only`
- hacer visible `proved_composed_fallback` como última categoría, nunca antes del fallback numérico

## P0. Contextualización

Crear las primeras suites nuevas:

- `contextual_rational_pairs.csv`
- `contextual_trig_pairs.csv`

Semillas iniciales:

- `1/x + 1/(x+1)`
- `1/(x-1) + 1/(x+1)`
- `sec^2 - tan^2`
- `sin(u) + sin(3*u)`
- logistic identity

## P1. Sampling mejorado

- sampler con rechazo de polos más fino
- sampler multivariante explícito

## P1. Contratos

- required conditions
- warnings

## P2. Exact checker pequeño

- racionales/polinomios pequeños

## Casos Candidatos a Añadir Ya

### Racionales

- `(1/x + 1/(x+1)) + (u-1)^5`
- `(1/x + 1/(x+1)) + (u^3 + u^2 + u + 1)`
- `(1/(x-1) + 1/(x+1)) + (u+1)^2`
- `(1/(x-1) + 1/(x+1)) + (u-1)^5`
- `(1/(x-1) + 1/(x+1)) + (u^6 - 1)`

### Trig

- `sec(x)^2 - tan(x)^2 + (u+1/2)^2`
- `sec(x)^2 - tan(x)^2 + (u-1)^5`
- `sin(u) + sin(3*u)` con constantes y con radicales conocidos
- `tanh(2*u)` dentro de polinomios equivalentes

### Exp / logistic

- `exp(u)/(exp(u)+1) + 1/(exp(u)+1)`
- embebida en sumas racionales y radicales

### Multivariante

- `(x^2+y^2)(a^2+b^2) + P(u)`
- donde `P(u)` sea:
  - factorizado
  - expandido
  - cúbico
  - grado 6

## Criterios de Éxito de la Fase

Daremos esta mejora por razonablemente completada cuando:

1. `numeric-only` deje de ser un bucket opaco.
2. Las familias repetidas del log tengan suites contextuales propias.
3. El checker numérico distinga mejor problemas de dominio/polos de equivalencias no probadas.
4. Existan metamórficos de contrato además de equivalencia.
5. Cada tanda de `numeric-only` nueva se pueda clasificar rápidamente como:
   - limitación real del motor
   - limitación del oracle
   - caso aún no curado

## Comandos Útiles

### Suite contextual

```bash
cargo test -p cas_solver --test metamorphic_simplification_tests metatest_csv_contextual_pairs -- --ignored --nocapture
```

### Combinaciones completas

```bash
METATEST_VERBOSE=1 cargo test -p cas_solver \
  --test metamorphic_simplification_tests metatest_csv_combinations_full \
  -- --nocapture --ignored
```

### Slice para inspección

```bash
METATEST_VERBOSE=1 METATEST_START_OFFSET=300 cargo test -p cas_engine \
  --test metamorphic_simplification_tests metatest_csv_combinations_full \
  -- --nocapture --ignored
```

### Diagnóstico de numeric-only

```bash
cargo test -p cas_solver --test numeric_only_diagnostic -- --nocapture
```

## Decisión de Mantenimiento

No volver a meter identidades complejas en el corpus global solo porque "matemáticamente caben".

Regla:

- corpus global = identidades base robustas
- suites contextuales = composiciones difíciles
- logs = fuente de promoción a nuevas suites

## Estado

Documento base creado. Sirve como guía viva para desarrollar la siguiente iteración de la batería metamórfica.
