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
6. `docs/CALCULUS_FRONTIER_AUDIT.md` — la cola priorizada de huecos
   medidos contra curso universitario y CAS profesional (P0 soundness,
   P1 capítulos a 0%, P2 familias, P3 educativo), con clase de ciclo
   (F familia / A arquitectónico / I fuera del norte) y la lista de
   residuales honestos que NUNCA deben "resolverse".
7. `docs/CALCULUS_ENGINE_DEVELOPMENT_PHASES.md` — la SECUENCIA de fases del
   north star (Fase 1 real-univariable-elemental+educativo → Fase 2 complejo
   elemental + vectorial multivariable → Fase 3 capas analíticas), con los
   items por fase y los **guardrails inter-fase** que mantienen baratas las
   fases futuras. Define a qué apuntar; las fuentes 4/6 dicen qué item.
8. `docs/AUDITORIA_P0_SOUNDNESS_<fecha>.md` — el informe de la última
   auditoría P0 multi-agente (workflow `cas-frontier-audit`): wrong-answers
   confirmados, agrupados por familia de causa-raíz, con estado de fix y
   backlog restante. Si existe uno reciente, su cola P0 manda sobre la
   fuente 6 para soundness.

> ⚠️ **Las fuentes de verdad se pudren.** Sus afirmaciones de CAPACIDAD (porcentajes,
> "X a ~0%", "X inalcanzable", "no implementado", items `[ ]`) reflejan lo que era
> verdad cuando se escribieron y ENVEJECEN sin avisar. Antes de dejar que una afirmación
> de estado dirija la selección de candidato, **verifícala contra el CLI vivo**
> (`target/release/cas_cli eval "..." --steps on --format json`), imprimiendo el ÁRBOL de
> substeps (`steps[].substeps[].title`), no solo el `rule` de nivel-1 — un audit de
> nivel-1 pierde la riqueza real (2026-07-15: una fila stale "límites ~45-50%, `e`
> inalcanzable" causó una recomendación equivocada; el sondeo la falsificó). Los LOGS
> históricos (ledger, `AUDITORIA_*_<fecha>`, notas `*(graduado …)*`) son INMUTABLES — no
> los "corrijas"; registran lo que era verdad en un commit. Ver "Meta-mantenimiento" al final.

## Fases del north star (a qué apuntar)

El north star tiene un ORDEN de fases deliberado (detalle en la fuente 7).
**ACTUALIZACIÓN 2026-07-23: las Fases 1-3 están CERRADAS (F3 en su núcleo) y la
Fase 4 (EDOs, abierta por el usuario el 2026-07-19) quedó COMPLETA salvo series
en 5 tandas con 0 rechazos. También cerró el frente S (sistemas en solve,
álgebra-soporte señalado por el usuario). El detalle por fase y las fuentes de
candidatos vigentes están en los bullets de abajo.** La regla de fondo se
mantiene: cada fase se abre SOLO al cruzar el umbral de la anterior — y las
aperturas nuevas (series, Gröbner 3+, análisis complejo) exigen decisión
explícita del usuario + mini-scoping, nunca inercia del bucle.

**Qué ordena la restricción de fase (y qué NO).** La restricción de fase ordena
SOLO el trabajo de **nueva capacidad de cálculo**. **NO** están sujetos a ella y
van SIEMPRE primero:
- **Fixes de soundness/honestidad** — cualquier wrong-answer o condición de dominio
  perdida, en **CUALQUIER** comando (solve, inecuaciones, factor, gcd, series, abs,
  matrices incluidos), aunque no sean cálculo real-univariable elemental. P0 antes
  que capacidad.
- **Ciclos arquitectónicos / de extracción (clase A)** — cohesión, ownership,
  architecture-pressure-first.

Esto es deliberado: los ~30 ciclos recientes retenidos (gcd que devolvía 1, signos
de denominador en inecuaciones, abs de imaginario, power-tower, factor, inversa de
matriz, series geométricas) son soundness/arquitectura, no cálculo elemental — y
fueron correctos. "Alterna frentes" y "mayor ROI retenible" del prompt maestro
operan DENTRO de este marco: el north-star de fase ORDENA la capacidad nueva, no
deroga el ROI retenible. Cuando la cola P0 de la fuente 6 y la restricción de fase
de la fuente 7 chocan, **gana P0/soundness**.

- **Fase 1 (COMPLETADA 2026-07-15 — umbral cruzado) — serio y universal: real,
  univariable, elemental + educativo básico.** Sus dos gatekeepers quedaron
  CERRADOS: **G1** (integración racional universal, hasta la clausura RootSum
  Cap. E-iv) y **G2** (narrativa de límites, MADURA). Los residuales que quedan
  son ESTRECHOS y NO bloquean (narración ∞−∞ con denom log/exp, algoritmo de
  VALOR general tipo Gruntz, Track R de presentación): siguen siendo candidatos
  válidos de pulido, no gatekeepers.
- **Fase 2 (COMPLETADA 2026-07-17/18) — complejo elemental + vectorial
  multivariable.** Ambas mitades cerradas (complejo 15 ciclos + residuales;
  vectorial tanda 8/8 + cierre formal con las 4 preguntas resueltas). El
  guardrail complejo sigue vigente para TODO ciclo futuro: reglas nuevas
  value-dependent auto-gateadas `RealOnly→None`; decline honesto antes que
  valor sin red de verificación.
- **Fase 3 (NÚCLEO CERRADO 2026-07-19) — capas analíticas.** F0-F11+F8b
  graduados (Taylor multivariable, verbos vectoriales gateados por
  verificación, límites multivariable con DNE-por-caminos, squeeze polar).
  Quedan los OPCIONALES F11b/F12 (`docs/FASE3_ANALYTIC_LAYERS_SCOPING.md`)
  como candidatos válidos de capacidad.
- **Fase 4 (COMPLETA salvo series, 2026-07-19→23) — EDOs elementales.**
  dsolve cubre el curso entero: O0-O9 + factor integrante μ(x)/μ(y)
  (1er orden clásico completo, 2º orden coef constantes + Cauchy-Euler,
  sistemas 2×2, superficie). Las SERIES de potencias solo entran con
  mini-scoping propio Y decisión explícita del usuario.
- **Frente S (sistemas en solve, COMPLETO 2026-07-23)** — fuera de la
  numeración de fases (álgebra-soporte): lineal n×n exacto, paramétrico
  2×2 con det≠0 como condición, no-lineales 2×2 por sustitución Y
  resultante de Sylvester. Sus peldaños nombrados (3×3 paramétrico, rank
  det=0, Res≡0) viven en el ledger/memoria como candidatos S/M.
- **Frente E (calidad educativa de steps, CAMPAÑA NÚCLEO COMPLETA 2026-07-23→24)**
  — fuera de la numeración de fases (mitad educativa): 3 ciclos E1-E3 + 3 tandas
  sobre el corpus de 210 examples (`docs/AUDITORIA_EDUCATIVA_2026-07-23.md`,
  informe HISTÓRICO — inmutable). Estado real: A 506→31 (heterogéneo), B/K/C/D
  todos 0, E highlights −75%, fugas de inglés 0, **27/42 solves del corpus narran
  por solve_steps** (trig completo, poly-en-átomo ×5 familias, abs completo salvo
  paramétrico, recíproco). El harness del corpus es la MEDIDA del frente
  (audit_extract/analyze2, regenerable; anclar comparaciones por EXPRESIÓN, no por
  índice — el csv se reordena). Residuales nombrados en
  [[frente-e-calidad-educativa]] (memoria): dueños trace-negativos de
  `|afín|⋚c`/`|f|<|g|`, `x^(2/3)>c`, E2b precisión path global, F 44.
- **De dónde sale la capacidad nueva HOY** (en orden): (1) peldaños
  nombrados de frentes cerrados (ledger + memoria — acotados y de bajo
  riesgo); (2) opcionales F11b/F12 de Fase 3; (3) residuales estrechos de
  Fase 1 (pulido); (4) series de Fase 4 SOLO con decisión de usuario; (5)
  si nada retenible: frontier-audit nuevo (workflow) para re-descubrir la
  frontera real — no elegir de memoria.
- El complejo multivaluado / análisis complejo completo y Gröbner para
  3+ incógnitas polinómicas generales están FUERA del norte (Gröbner
  entraría solo con mini-scoping propio).

**Guardrails inter-fase — OBLIGATORIOS en cada ciclo (aplican igual en Fase 2).**
No cuestan más hoy y son la razón de que el orden real-primero fue correcto
(volvieron la Fase 2 ≈ M; ahora preparan la Fase 3 igual de barata):
1. En toda regla nueva **cuyo resultado dependa del dominio de valores**
   (log/sqrt/exp/potencias/inversas) enhebra `ValueDomain` y gatea real-only
   (`value_domain() == RealOnly => return None`); nunca hard-codees RealOnly en un
   contrato público. Las reglas puramente sintácticas, de presentación o de
   narración (p.ej. `diff(x,n)`, linealidad de sumatorios, trazas) NO necesitan el
   gate — no lo añadas como ceremonia (código muerto + contrato RealOnly engañoso).
2. Mantén diff/integrate parametrizados por variable (per-variable, sin sesgo
   single-var).
3. Predicados de condición estructurados/extensibles (cortes de rama, dominio),
   no supuestos real-only horneados.
4. Backstop de soundness domain-aware y EXACTO (`BigRational`, patrón
   `*_in_domain`); nunca f64 para keep/drop.
5. Resultados como contrato (cargan decisiones de rama/dominio).

## Cerrar el dominio real = preparar el dominio complejo (nexo clave)

Cerrar soundness en dominio real y preparar Fase 2 (complejo) son el MISMO trabajo
cuando se hace bien: el chokepoint compartido es la **capa de decisión EXACTA y
parametrizada por dominio** (signo, condiciones de dominio, cortes de rama). Cada
fix real que hace una decisión exacta y la carga como CONTRATO es directamente
reutilizable en complejo (donde la misma decisión devuelve una condición de
rama en vez de un intervalo real).

- **La capa de signo/constante es el chokepoint transversal.** Enseñar a los
  probadores de signo (`cas_math::prove_sign::prove_positive`/`prove_nonnegative`,
  `cas_solver_core::isolation_utils::is_known_negative`, el discriminante en
  `quadratic_formula.rs`, el umbral radical) que un **surd/transcendental constante
  `A+B√n`, `e−3` es un valor real DECIDIBLE** — vía `cas_math::root_forms::provable_sign_vs_zero`
  (Option<Ordering>) o `cas_math::const_sign::provable_const_sign` (superset: también
  e/π) — cerró **5+ familias P0 de una vez** (2026-07). Esa misma capa exacta es lo
  que Fase 2 necesita: no es de usar y tirar.
- **Nunca hard-codees RealOnly; devuelve la CONDICIÓN.** Un fix que devuelve `Empty`
  para un radicando real-negativo debe poder devolver la condición estructurada, para
  que el dominio complejo la voltee a una raíz compleja sin re-derivar (guardrails
  inter-fase #1/#3/#5). El "surd es constante decidible, no coeficiente simbólico" es
  precisamente esta preparación.

## Estrategias de reducción, soundness y completitud (validadas 2026-07)

**Reduce-a-canónico — el patrón de mayor ROI para universalidad.** Identifica el
ATOM invertible y delega en su solver robusto; NO parchees el caso:
- radicales `√f = ±g`, `√f ± √g = c`; exponenciales `m^x → p^(k·x)`,
  Laurent-en-`b^x` (recíprocas/hiperbólicas `e^x+e^(-x)`), dos-bases-distintas
  `A·m^x = B·n^x → log`; trig `a·sin+b·cos = 0 → tan(g) = −b/a`, inhomogénea
  `→ R·sin(g+φ) = c/R` (ángulo auxiliar), argumento afín/desfase-π; abs
  `|E| = 0 ⟺ E = 0`, `|f| = g → f = ±g`; poly-en-atom (`u = ln/√/trig/exp/x^(1/q)`).
- La VERIFICACIÓN de raíces contra la ecuación ORIGINAL subsume las condiciones de
  dominio (radicandos ≥ 0, surdos que cancelan, `g ≥ 0` en abs) — no re-derives el
  dominio: verifica. Y trabaja en el árbol CRUDO si `simplify` colapsa la estructura
  que detectas (`e^x+e^(-x) → cosh`, `sin² →` doble-ángulo).

**Chokepoint > parche por caso.** Cuando varios wrong-answers comparten una
meta-forma ("un guard dispara solo para el caso racional/nombrado y pierde el
hermano surd/negado/compuesto/recíproco"), arregla la CAPA compartida UNA vez, no
caso a caso. Colector que hornea un signo/forma fija → devuelve el signo/forma como
DATO. Al mover el RHS a un lado, `Sub(lhs, 0)` deja una constante `0` que un colector
estructural debe DESCARTAR explícitamente (una no-nula ⇒ otra forma ⇒ declina).

**Disciplina de barrido adversarial (procedimientos de decisión).** Verde en tests
unitarios NO basta — ha cazado wrong-answers que los tests verdes no veían:
- incluye el BORDE de la constante sobre la que se ramifica (`c = 0`), no solo
  `c ∈ {1,2,3}`; e incluye coeficientes COMPUESTOS (`2√2`, surd×surd), no solo
  atómicos (un descarte de coeficiente interno anidado es invisible a los atómicos).
- oráculo independiente (sympy) + verificación por SUSTITUCIÓN para familias
  periódicas (raíces tangentes: se comprueban por sustitución, no por cambio de signo).
- al AMPLIAR el alcance de un colector, enumera los casos que ahora TAMBIÉN captura y
  re-deriva su verificación — una heredada demasiado estricta convierte un acierto de
  otro handler en un wrong-answer (p.ej. `√A=√B` con surdos que cancelan).

**Descubrimiento y scoping con workflows multi-agente (ultracode / opt-in).** La
frontera real la descubres exhaustivamente, no de memoria:
- FRONTIER-AUDIT: N scouts (uno por frente: solve / inecuaciones / radicales / exp /
  log / trig / abs / derivadas / integrales / límites / series-matrices) probando ~40
  inputs c/u → verificación adversarial 2-lentes (confirma wrong-answer vs falso
  positivo de convención) → síntesis rankeada por ROI. Guarda el informe en
  `docs/AUDITORIA_P0_SOUNDNESS_<fecha>.md`.
- SCOPING: un investigador READ-ONLY por bug → `file:line` exacto + fix mínimo
  verificado + blast-radius + dificultad; convierte cada P0 en un ciclo acotado.
- COMMITEA antes de lanzar cualquier workflow (los agentes pueden tocar el árbol).
- Falsos positivos a NO reportar: `log(a,b) = log_a(b)`, familias tangentes de una
  sola rama, y la omisión CORRECTA de raíces complejas en dominio real.

**Arquitectura: extraer antes de abstraer (corte de menor riesgo).** Cuando un god
file (`solve_backend_local.rs`) acumula handlers, extrae PRIMERO las utilidades PURAS
(matchers estructurales sobre `&Context`, sin deps de la infra de solve, no usadas por
los tests inline) en bloques CONTIGUOS a un módulo hermano `pub(crate)`: `cargo check`
valida la visibilidad al instante y la huella 0-delta prueba que es behavior-preserving.
Los handlers y los helpers entrelazados con la infra de inecuaciones son un corte
posterior de mayor riesgo (necesitan `pub(crate)` en la maquinaria compartida).

## Protocolo de un ciclo

### 0. Precondiciones
- `git status` limpio (si no, para y repórtalo: nunca trabajes sobre un
  árbol sucio).
- Copia baselines de huella:
  `cp docs/generated/engine_improvement_scorecard.json /tmp/scorecard_guardrail_before.json`
  `cp docs/generated/engine_improvement_scorecard_pressure.json /tmp/scorecard_pressure_before.json`

### 1. Selección del candidato (mayor ROI)
- **Si el candidato es capacidad NUEVA, DEBE estar en Fase 1** (ver "Fases del
  north star"); no abras Fase 2/3 hasta cruzar su umbral. **Los fixes de
  soundness/honestidad (cualquier comando) y los ciclos clase A NO están sujetos a
  la fase: van primero.** A igualdad de coste dentro de Fase 1, prioriza los dos
  gatekeepers y luego los wins P1 baratos.
- **Los gatekeepers son clase L** (G1 ~8-12 ciclos, G2 ~6-10): NUNCA se entran como
  un solo ciclo. Se entran SIEMPRE como **scoping workflow que produce una SECUENCIA
  de sub-ciclos acotados y retenibles** (cada uno con su commit y su green-before-
  commit). Si en este ciclo no hay un sub-paso retenible del gatekeeper, **cae a los
  wins P1 baratos o a un candidato de soundness/arquitectura** — no arranques un LRT
  o una cadena didáctica a medio construir que falle el verde y fuerce revert.
- Parte de la "siguiente iteración recomendada" del informe del ciclo
  anterior (ledger/roadmap), de los items `[in progress]`/`[pending]`,
  y de la cola priorizada de `docs/CALCULUS_FRONTIER_AUDIT.md`
  (P0 antes que P1 antes que P2/P3 a igualdad de coste; los items
  clase A exigen scoping workflow primero; los clase I no son ciclos).
  **Salvedad del único P0 abierto:** el de FTC con borde singular es un
  *under-answer conservador no urgente* (un-answer, no wrong-answer); no compromete
  soundness, así que los dos gatekeepers van por delante. La precedencia
  "wrong-answer-P0-primero" sigue intacta para cualquier P0 real de respuesta
  incorrecta.
- Criterios: avanza un gate incompleto del north star; acotado a un
  ciclo; retenible (capacidad nueva verificable, fix de soundness, o
  extracción behavior-preserving); reutiliza maquinaria existente antes
  de inventar.
- Alterna frentes cuando uno acumula varios ciclos seguidos: la mitad
  educativa (steps, Phase 6) cuenta lo mismo que la universal. *(El educativo de
  límites, gatekeeper G2, ya está sustancialmente cerrado 2026-07-15 — ya NO a
  ~0%; ver la nota en "Fases del north star".)*

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
- **Respeta los guardrails inter-fase** (ver "Fases del north star"): regla nueva
  **value-dependent** (log/sqrt/exp/potencias/inversas) enhebra `ValueDomain` y
  gatea real-only — las sintácticas/de presentación no; diff/integrate per-variable;
  condiciones estructuradas; backstop exacto domain-aware. No cuesta más hoy y es
  lo que mantiene baratas las Fases 2/3 — incumplirlos es deuda que se paga L.
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
make engine-scorecard           # guardrail, 18 suites
make engine-scorecard-pressure  # 3 suites
make wasm-check                 # cargo check wasm32 de cas_wasm (~1 min): tipos/cfg para Pages
```
Si falla un test existente, primero juzga la intención: si fijaba como
residual algo que tu ciclo convierte en soportado, actualiza el contrato;
si fija soundness (condiciones, dominios), tu cambio es el problema.

**Gate condicional de Pages (build completo + E2E).** `wasm-check` NO cubre
codegen/linker (lección W3: check ≠ build). Si el ciclo toca `cas_wasm`,
`web/`, `Cargo.toml`/`Cargo.lock` o código `cfg(target_arch)`, añade ANTES de
commitear: `scripts/build_pages_site.sh` (wasm-pack nightly, ~5-10 min) y la
verificación E2E en navegador con el server `pages-static` de
`.claude/launch.json` sirviendo `dist/pages` (probar la superficie tocada:
declaraciones/paneles/import/…, como en los fixes W7). Para ciclos que no
tocan esas zonas, el build completo por-ciclo es un impuesto sin cazas
históricas — el workflow de Actions es el gate real post-push.

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
- Si el ciclo gradúa (total o parcialmente) un item de
  `docs/CALCULUS_FRONTIER_AUDIT.md`, márcalo ahí con `[x]` +
  `*(graduado FECHA commit: qué quedó cubierto y qué queda como
  peldaño)*` — con el hash REAL del commit del ciclo, nunca
  marcar por adelantado. Las verificaciones de honestidad del audit
  (residuales no-elementales) cuentan como contratos: si un ciclo las
  rompe, es soundness, no capacidad.
- **Disciplina de hash-stamps: nunca estampes el hash del PROPIO commit
  vía amend** — el amend crea un commit nuevo y el hash estampado queda
  COLGANTE (resuelve con `git cat-file` pero no es ancestro de main;
  así se acumularon 6 stamps rotos hasta el audit 2026-07-15). El hash
  de un ciclo se estampa en el commit del ciclo SIGUIENTE, o el doc
  cita "hash en el ledger". Auditar stamps existentes con
  `git merge-base --is-ancestor <hash> HEAD`, no con "el hash resuelve".

### 8. Commit e informe
- Un commit por ciclo: título imperativo describiendo la capacidad o el
  cambio (no "cycle N"), cuerpo con el porqué, el diseño, los fixes que
  surgieron y los números de validación. Termina con:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- **Cada push a `main` DESPLIEGA GitHub Pages** (workflow `pages`,
  build_type=workflow): commitear en local no publica; push sí. Si el build
  de Actions falla, el sitio conserva el último deploy bueno (no se cae) y
  el aviso llega en ~5 min — para cambios que tocan la superficie web, espera
  el verde de Actions (`gh run watch`) y verifica en vivo antes de dar el
  ciclo por cerrado.
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
  barre adversarialmente antes de retener — ha cazado defectos reales que los
  tests unitarios verdes no veían. Cubre los BORDES (`c = 0`) y los coeficientes
  COMPUESTOS (`2√2`, surd×surd), oráculo sympy + sustitución para periódicas
  (ver "Disciplina de barrido adversarial" arriba). Para lotes grandes de
  soundness, un workflow de 2 lentes (refutación + regresión de contadores).
- Presupuesto de fricción: si un ciclo lleva 3+ intentos fallidos de la
  misma edición, relee el archivo real (rustfmt reordena imports y
  reformatea anclas) en vez de reintentar a ciegas.

## Lecciones operativas acumuladas (no re-aprender)

- **Sitios de la capa de decisión de signo/constante** (donde arreglar el chokepoint
  surd/transcendental): `cas_math::root_forms::provable_sign_vs_zero` (surd lineal,
  Option<Ordering>) y `cas_math::const_sign::provable_const_sign` (superset con e/π);
  consumidos por `cas_math::prove_sign::{prove_positive,prove_nonnegative}_depth_inner`,
  `cas_solver_core::isolation_utils::is_known_negative`, el discriminante SymbolicEq en
  `quadratic_formula.rs`, y el umbral even-root en `solve_backend_local.rs`. Un `_ =>
  false`/`as_rational_const(...)`-only en un guard de signo es casi siempre un
  wrong-answer surd latente.
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

- **Ritual de steps para maquinaria interna**: un step-listener de sesión ve
  TODOS los simplify — cualquier solve/simplify interno que corra bajo un
  comando con steps activos debe aislarse con save/off/restore de
  `step_listener` Y `steps_mode` (2 leaks cazados: micro-pasos haciéndose
  pasar por narración). Y la lane que protege una superficie con steps debe
  CORRER con `--steps on` o no ve el leak.
- **Delegar en el handler verificado convierte un método nuevo en
  solo-su-matcher** (5 instancias: Bernoulli→lineal, homogénea→separable,
  Cauchy-Euler→D9, surd→sustitución, μ→exacta). Antes de construir
  maquinaria, busca el handler graduado al que reducir; y el techo del
  delegado ES el techo de la composición — verifícalo como pre-existente
  con la forma transformada como input directo antes de culpar al ciclo.
- **«La excepción vive donde vive el contexto»** (4 instancias: ∂→d de
  dsolve, eco fiel, desugar compartido, constantes-nombradas-declaradas):
  las cortesías de canal se interceptan en el chokepoint del canal (spec
  parse), jamás en el significado global.
- **Separar «emitir» de «afirmar vacío»**: la emisión puede ser agresiva si
  la gatea verificación exacta incondicional; la afirmación de conjunto
  vacío exige un argumento de COMPLETITUD aparte (candidatos exhaustivos) —
  sin él, decline honesto, no "no solution".
- **La forma degenerada rompe al delegado** (6ª aparición: cero aparcado ×5,
  `solve(9=0)` Err): antes de delegar una forma construida, atajar sus casos
  triviales/degenerados (constante, cero) en el caller.

- **Instrumenta ANTES de implementar, y el trace va en el CALL-SITE** (2
  instancias): un scoping heredado es hipótesis — 2 eprintln en los boundaries
  sospechosos lo falsifican en una corrida ([052]: el descarte no era el
  ensamblado de estrategia sino las auto-recursiones del propio handler); con
  recursión, «quién se ejecuta» es ruido y «quién responde» es la señal. El
  trace-NEGATIVO también es entregable (candidatos descartados no se
  re-investigan).
- **Detector de narración muda de coste cero**: `grep 'let (sol, _)' /
  'let (xs, _)'` sobre delegadores a `solver_entrypoints_solve::solve` — la
  narración venía en la tupla y se tiraba (3 tandas de peldaños salieron de ahí).
  Al narrar un sustrato compartido, contar su fan-out de wrappers antes que casos
  del corpus (poly-en-átomo: 1 función → 5 familias).
- **Narrar ≠ avalar (candidatos vs respuesta)**: los pasos de RAMA solo se
  anexan si la rama produce respuesta directa; si produce candidatos que una
  verificación/intersección posterior puede descartar, narra solo la ESTRUCTURA
  (líneas de split/caso). Y las líneas de identidad se predican de la ECUACIÓN
  DEL USUARIO (jamás de formas internas con símbolos sintéticos `__u`).
- **La forma que narra y la que resuelve pueden divergir legítimamente**: el
  clasificador necesita la normal simplificada; el alumno el cociente crudo
  (√3/2, no 3/2·3^(-1/2)). Construir ambas en el punto de reducción; en líneas
  de narración plegar SOLO el ruido nombrado (base 0, −0), nunca simplify
  general (factoriza a formas ilegibles).
- **`git add -A` con edición humana concurrente barre cambios ajenos**: el
  status limpio al INICIO no cubre ediciones DURANTE el ciclo — antes de
  commitear, revisa `git diff --stat` y cuestiona archivos fuera del área
  tocada (un reorden de examples.csv entró así y movió los índices del harness).

## Meta-mantenimiento: revisiones periódicas (docs y esta skill)

El bucle mejora el ENGINE; estas dos revisiones mantienen honesto el bucle mismo. Ninguna
es un ciclo de capacidad — no llevan huella; son higiene de las fuentes de verdad. Hazlas
**de vez en cuando** (buen disparador: cada ~8-12 ciclos, tras una tanda de graduaciones, o
en cuanto una afirmación de doc choque con lo que ves en el CLI). No son opcionales-para-
siempre: una fuente de verdad podrida dirige mal la selección de candidato y desperdicia
ciclos (lección viva del 2026-07-15).

### A. Auditoría de veracidad de la documentación
Las afirmaciones de capacidad de los docs (fuentes 3-8) se pudren. Barre las que puedan
DIRIGIR la selección de candidato (porcentajes, "~0%"/"inalcanzable"/"no implementado",
items `[ ]` que podrían estar secretamente graduados) y verifícalas contra el CLI vivo
(imprime el árbol de substeps, no el `rule` de nivel-1). Corrige el estado-ACTUAL; deja
intactos los LOGS históricos (ledger, `AUDITORIA_*_<fecha>`, notas `*(graduado …)*`). Un
workflow READ-ONLY de 2-3 auditores (uno por doc/sección, cada uno probando el CLI) lo hace
exhaustivo. Commit docs-only, sin huella. *(Patrón validado 2026-07-15: `91b42728e`.)*

### B. Auto-revisión de esta skill
La estrategia de auto-mejora también aprende. Con el conocimiento nuevo de las últimas
tandas, relee ESTA skill y pregunta: ¿la priorización sigue apuntando al norte real
(**universal Y educativo** en dominio real)? ¿algún gatekeeper/fase ya cerrado se sigue
tratando como abierto — o al revés? ¿alguna "lección operativa" quedó obsoleta, o hay una
nueva recurrente que merece entrar? ¿los criterios de retención/rechazo y la cadencia
siguen sirviendo? Actualiza la skill para que la ESTRATEGIA refleje el estado real y las
lecciones acumuladas — el objetivo es acercar el engine a lo **más universal posible SIN
perder lo más educativo posible**, y esta skill es la palanca que lo dirige. Registra el
porqué del cambio en el commit. *(2026-07-15: G2 —límites educativos— se marcó cerrado aquí
tras confirmarse maduro; `7812f4109`. 2026-07-24: pasada B tras la campaña del
frente E — estado E añadido a fases, 5 lecciones nuevas; la pasada A verificó
contra el CLI que las filas P3 de CALCULUS_FRONTIER_AUDIT que parecían
graduables NO lo están —`ln(e)` sigue vivo en diff orden ≥4, el corpus da D=0
porque no ejercita orden 4— y se dejaron intactas: el chequeo vivo evitó un
falso stamp.)*
