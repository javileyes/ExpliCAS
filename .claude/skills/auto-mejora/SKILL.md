---
name: auto-mejora
description: Ejecuta N ciclos de auto-mejora del engine CAS siguiendo el proceso documentado del repo - selección de candidato por ROI, iteración acotada, validación con huellas de scorecard, ledger y commit por ciclo. Usar cuando el usuario pida ciclos de mejora del engine, "haz un ciclo de mejora", o un goal de varios ciclos.
---

# Auto-mejora del engine

Eres el operador del bucle de auto-mejora del engine CAS de este repo. El
argumento de la invocación es el número de ciclos a completar, `N`
(default: 1). Cada ciclo es **una iteración acotada y retenible**, con su
propio commit y su propio informe. No mezcles dos ciclos en un commit.

## Fuentes de verdad

Lee al arrancar (y vuelve a consultarlas tras cualquier compactación de
contexto — son tu memoria externa):

1. `docs/ENGINE_AUTO_IMPROVEMENT_PROMPT.txt` — el proceso maestro
   (captura obligatoria, criterios de retención/rechazo, cadencia).
2. `docs/ENGINE_COMBINATION_LEDGER.md` — memoria de trabajo: qué se
   retuvo/rechazó y los aprendizajes; tu entrada nueva va al final.
3. `docs/CALCULUS_ENGINE_STRATEGY.md` — el north star (engine de cálculo
   diferencial/integral **universal Y educativo** en dominio real) y los
   guardrails de Deferred Horizons.
4. `docs/GENERAL_INTEGRATION_BACKEND_ROADMAP.md` — fases, items con
   estado, y la recomendación de siguiente iteración.
5. `docs/ENGINE_COHESION_REFACTORING_STRATEGY.md` — cuándo una iteración
   debe ser arquitectura (extracción) en vez de capacidad.

## Protocolo de un ciclo

### 0. Precondiciones
- `git status` limpio (si no, para y repórtalo: nunca trabajes sobre un
  árbol sucio).
- Copia baselines de huella:
  `cp docs/generated/engine_improvement_scorecard.json /tmp/scorecard_guardrail_before.json`
  `cp docs/generated/engine_improvement_scorecard_pressure.json /tmp/scorecard_pressure_before.json`

### 1. Selección del candidato (mayor ROI)
- Parte de la "siguiente iteración recomendada" del informe del ciclo
  anterior (ledger/roadmap) y de los items `[in progress]`/`[pending]`.
- Criterios: avanza un gate incompleto del north star; acotado a un
  ciclo; retenible (capacidad nueva verificable, fix de soundness, o
  extracción behavior-preserving); reutiliza maquinaria existente antes
  de inventar.
- Alterna frentes cuando uno acumula varios ciclos seguidos: la mitad
  educativa (steps, Phase 6) cuenta lo mismo que la universal.

### 2. Sondeo antes de implementar
- Sonda la frontera real con probes del CLI (`target/release/cas_cli
  eval "..." --format json`): lo que crees residual puede ya funcionar,
  y viceversa. La frontera real define el alcance.
- Si hay incógnitas de diseño (APIs internas, riesgo de huella, punto de
  inserción), lanza un workflow de scoping con agentes paralelos antes de
  tocar código. Pregunta SIEMPRE por el riesgo de huella: ¿qué fixtures
  de las lanes capturaría el cambio?
- Decide guards que protejan la huella como **intención declarada** (p.ej.
  ventanas de grado, formas con dueño existente), no como parches.

### 3. Implementación acotada
- Toca las zonas de crecimiento designadas (p.ej.
  `general_integration_backend/methods.rs`,
  `verification_normalization.rs`), no los god files.
- Reutiliza: solver lineal compartido, builders arctan/log, verificador
  algebraico, `Polynomial`. Extraer antes de abstraer.
- Los residuales fuera de alcance se quedan residuales **honestos** y se
  anotan en el roadmap como siguiente peldaño.

### 4. Tests y harness
- Tests unitarios para la matemática nueva (descomposiciones con valores
  exactos esperados) y para los **rechazos** (cada bail con su razón).
- Si promueves capacidad pública: filas nuevas en
  `scripts/engine_integrate_command_matrix_smoke.py` (con
  `expected_direct_diff_integrate_result` para el round-trip) y
  contadores en `scripts/test_engine_integrate_command_matrix_smoke.py`
  (len(cases), supported, block12, boundary_verified,
  verified_by_direct_diff, direct_diff_exact, verified_supported,
  domain_regime). Corre `python3 -m unittest
  scripts.test_engine_integrate_command_matrix_smoke` y el smoke entero.

### 5. Cadena de validación (toda verde antes de commitear)
```bash
cargo test --workspace          # NUNCA por crate: el total debe ser failed:0
cargo clippy --workspace --all-targets -- -D warnings   # --all-targets: el clippy por crate NO cubre tests
rustfmt --edition 2021 <archivos tocados>
make engine-fast
make engine-scorecard           # guardrail, 16 suites
make engine-scorecard-pressure  # 3 suites
```
Si falla un test existente, primero juzga la intención: si fijaba como
residual algo que tu ciclo convierte en soportado, actualiza el contrato;
si fija soundness (condiciones, dominios), tu cambio es el problema.

### 6. Comparación de huella
```python
# Para guardrail y pressure: estado/passed/failed por suite vs baseline.
# Único delta admisible: las lanes cuyos contadores actualizaste a
# propósito (p.ej. la matriz al añadir filas). La lane
# calculus_integrate_backend_observability debe quedar ESTRUCTURALMENTE
# idéntica salvo decisión explícita documentada en el ledger.
# Filtra claves de runtime: runtime/elapsed/ms/seconds/duration.
```
Cualquier delta no intencionado ⇒ investigar; si no se resuelve ⇒
**rechazar la iteración** (revert), anotar el aprendizaje en el ledger y
pasar al siguiente candidato (el ciclo cuenta como rechazado, no lo
repitas en bucle).

### 7. Documentación
- Entrada nueva al final de `docs/ENGINE_COMBINATION_LEDGER.md` con el
  formato de las últimas entradas (area / status / capture / observed /
  decision / retained learning). Los aprendizajes deben ser
  generalizables, no un changelog.
- `python3 scripts/engine_combination_ledger_tool.py --reindex`
- Actualiza el estado en el roadmap (items done/in-progress, "siguiente
  peldaño" para lo que quede fuera).

### 8. Commit e informe
- Un commit por ciclo: título imperativo describiendo la capacidad o el
  cambio (no "cycle N"), cuerpo con el porqué, el diseño, los fixes que
  surgieron y los números de validación. Termina con:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- Informe del ciclo: qué cambió y por qué era el mayor ROI, resultados
  ejemplo, validación (tests/clippy/matriz/huellas), aprendizaje
  retenido, y **siguiente iteración recomendada** (alimenta el ciclo
  siguiente).

## Protocolo multi-ciclo (N > 1)

- Mantén un TodoWrite con `ciclo i/N` y el estado del ciclo en curso —
  sobrevive a compactaciones.
- Tras el commit y el informe de un ciclo, **continúa inmediatamente**
  con el siguiente sin pedir permiso; la selección del candidato i+1
  sale del informe del ciclo i.
- Paradas duras (termina el goal y repórtalo): árbol sucio que no creaste,
  validación irrecuperable tras un revert (2 rechazos seguidos del mismo
  candidato), o cualquier indicio de pérdida de datos.
- Verificación adversarial: para ciclos que introducen procedimientos de
  decisión o verificadores (no para extensiones mecánicas de familias),
  lanza un workflow adversarial de 2 lentes (soundness con probes de
  refutación + regresión de contadores) antes de retener — ha cazado
  defectos reales que los tests unitarios verdes no veían.
- Presupuesto de fricción: si un ciclo lleva 3+ intentos fallidos de la
  misma edición, relee el archivo real (rustfmt reordena imports y
  reformatea anclas) en vez de reintentar a ciegas.

## Lecciones operativas acumuladas (no re-aprender)

- El motor **expande** productos sintácticos antes de `integrate`: las
  frontera racionales se miden sobre polinomios expandidos.
- `numeric_value` solo casa literales: usa `numeric_eval::as_rational_const`
  para formas plegables (`x^(2-1)`, `6-3*2`) en matchers y traducciones.
- Las políticas de aceptación pública escritas para la primera familia de
  un método bloquean a sucesores mejor portados: exprésalas como
  intención (qué conjuntos de condiciones son confiables).
- La completitud de condiciones es una obligación de prueba separada de
  la identidad de la derivada: el verificador algebraico no ve una
  condición de dominio ausente (polos con residuo cero, radicandos).
- gcd de `Polynomial` devuelve escala racional arbitraria: renormaliza
  mónico tras cada gcd/div.
- En la web/server, los multiline-replace por indentación (12⊂16⊂20
  espacios) cascadean: ancla en bloques de llamada completos.
- `make ci` tiene lints EXTRA que la cadena del ciclo no cubre: fmt
  global (declaraciones `mod` en orden alfabético — los inserts por
  ancla de wiring lo rompen) y `lint_string_compares` (prohibido
  `sym_name(...) == "..."` sobre nombres de función en cas_engine: usa
  `is_call_named` o interna el símbolo una vez y compara por SymbolId).
  Tras tocar cas_engine, corre `cargo fmt -p cas_engine` y, si añadiste
  comparaciones de nombres de función, verifica con `make ci`.
