# Plan de auditoría de calidad del código — multi-agente, paralelizada

**Fecha**: 2026-07-29 · **Estado**: PLAN (sin ejecutar) · **Autor**: sesión de perf campaña 2ª

## 0. Tesis y correcciones de rumbo

Tesis del proyecto: optimizar dirigido por métricas "desenreda" el código —
quita ineficiencias, sanea y simplifica — y eso da más garantías de alcanzar
**universalidad con steps educativos**.

Evidencia a favor (campaña 2ª de perf, commits `2918f13a2`→`08ebf09d1`):
- El memo P16 estaba neutralizado por wipes mal concebidos → revivirlo fue
  *arreglar un diseño*, no "tunear".
- El reductor del quotient ring era término-a-término con rebuild completo →
  el batching es *mejor algoritmo*, no un truco.
- `atanh_ln_bounds` sin retícula era una *asimetría interna* (sus vecinos
  Newton ya la tenían).

Dos correcciones que este plan incorpora:
1. **La perf es el detector, no el objetivo.** La flecha causal es
   medir → entender → sanear. Por eso el plan usa las métricas de rendimiento
   como UNA lente (L4) y añade lentes que la perf no ve.
2. **La universalidad vive en la larga cola, no en los hot paths.** Sus
   riesgos son: gates inexactos, contratos sin test, fugas entre capas,
   capacidad a medio cablear. Lentes L2, L3, L5, L8.

Principio rector heredado del Frente E: **publicado ⇒ verificado**. Ningún
hallazgo se acepta sin prueba mecánica; ningún cambio se fusiona sin gate
conductual completo.

---

## 1. Invariantes intocables (contrato para TODOS los agentes)

Todo agente recibe esta lista literal. Violar una = rechazo automático.

- **I1** `as_rational_const` no se modifica (regla R1 del proyecto).
- **I2** Los gates de soundness son exactos: jamás f64 para decidir drop/keep.
- **I3** `__hold`/`Expr::Hold`: contrato documentado en `cas_ast/src/hold.rs`.
  No se altera su semántica; strip solo en fronteras de salida.
- **I4** Sticky domain/root y memos (P16, sondas, exact-zero): su ciclo de
  vida está razonado en comentarios; cambiarlo exige el análisis completo de
  invalidación por escrito en el PR.
- **I5** El ORDEN de reglas y de root-shortcuts es semántico. No se reordena
  "por estética".
- **I6** Budgets y caps: se pueden subir/instrumentar, nunca eliminar ni
  convertir en silenciosos (lección "no silent caps").
- **I7** Cambio de COMPORTAMIENTO (resultado, steps, warnings) = fuera de
  alcance de esta auditoría. Si un agente cree que el comportamiento actual
  es un bug, lo cataloga en `HALLAZGOS_CAPACIDAD.md` — no lo "arregla".
- **I8** Dead code en crates de dominio (`cas_math`, `cas_engine`,
  `cas_solver*`) NO se borra: se cataloga como posible capacidad a medio
  cablear (lección dead-code-detector). Solo el plumbing 0-refs es borrable.
- **I9** Nombres de reglas/steps visibles al usuario no cambian (hay pins).
- **I10** Push = deploy. Nadie pushea. Commit por lote, mensaje en español.

---

## 2. Gates mecánicos (definición única, usada por todas las fases)

**GATE-A (conductual)** — obligatorio por lote de cambio:
1. `cargo test --workspace` = exit 0 (preservar log crudo: `> log 2>&1;
   echo EXIT=$?` — un grep en el pipe enmascara el exit).
2. Diff conductual del corpus VACÍO: los 221 de `web/examples.csv` por
   `eval --steps on --format json`, comparando result + steps_count +
   solve_steps_count + substeps_count + warnings (harness:
   `scripts/` — portar `corpus_results.py`/`corpus_timing.py` del scratchpad
   de la sesión b5a0d3cf; F0 los materializa).
3. `cargo clippy --workspace` sin warnings nuevos.

**GATE-B (rendimiento, no-regresión)** — por lote que toque crates calientes:
- Los 8 trazadores de la campaña con presupuesto = medición base × 1.25:
  `solve(e^x+e^(-x)=4,x)`, `solve(x^3-6x^2+11x-6=0,x)`, `solve(x+1/x>2,x)`,
  `solve(abs(x^2-1)=x+1,x)`, `integrate(1/(x^8+1),x)`,
  `integrate(1/(x^8+16),x)`, `dsolve(diff(y,x,2)+y=cos(x),y,x)`,
  `solve([x^2+y^2=25,x+y=7],[x,y])` (medir por `timings_us.simplify_us`,
  mediana de 3; el ruido run-to-run es ±15% — lección scorecard-huella:
  juzgar tendencia, no un run).

**GATE-C (huella)** — al cierre de cada fase de mutación:
- `make engine-scorecard`: contadores estructurales (state/passed/failed/
  total) y slot identity idénticos; los campos de latencia se ignoran.

---

## 3. Fase 0 — Preparación (secuencial, ~1 agente, barato)

Sin esto no arranca nada. Entregables:

- **F0.1** Working tree limpio y commit de partida anotado (lección:
  workflow-agents-mutate-working-tree — commitear ANTES de lanzar agentes).
- **F0.2** Materializar los harness del scratchpad como scripts del repo:
  `scripts/corpus_behavior_diff.py` (GATE-A.2) y
  `scripts/corpus_timing.py` (GATE-B), con baseline commiteado en
  `docs/generated/quality_audit_baseline/`.
- **F0.3** Inventario base: `tokei` por crate (o `wc -l` si no está),
  lista de archivos >5k líneas, funciones >300 líneas
  (`rg -n "^    (pub )?fn " + awk`), `cargo tree -d` (deps duplicadas),
  salida de los lints existentes del Makefile (`lint-allowlist`,
  `lint-budget`, `lint-limits`, `audit-utils`, `lint-string-compares`,
  `lint-no-panic-prod`). Todo a `docs/generated/quality_audit_baseline/`.
- **F0.4** Congelar este plan + invariantes en el prompt-plantilla (ver §8).

## 4. Fase 1 — Barrido de lectura (8 lentes × particiones, TODO paralelo)

Agentes de SOLO LECTURA (tipo Explore/general-purpose sin Write). Cada lente
produce `docs/generated/quality_audit/L<k>_<slug>.md` con hallazgos en
formato fijo (§7). Ninguna lente muta código ⟹ paralelismo total (8-16
agentes; las lentes grandes se parten por crates).

- **L1 — Duplicación de moldes** (partición: por pares de crates).
  Buscar el mismo patrón reimplementado con formas distintas. Semillas
  conocidas: 3+ memos thread-local con contratos de invalidación distintos
  (`CANCELLATION_MATCH_MEMO`, `VARIABLE_SQUARE_GATE_MEMO`,
  `DEFAULT_SIMPLIFY_PROBE_MEMO`, `ISOLATED_SIMPLIFY_PROBE_MEMO`, memos de
  `perf-recurrence`); 2 mecanismos de hold (`Expr::Hold` vs `__hold`);
  variantes de `as_rational_const`/eval numérico; walkers contains-X
  repetidos (`contains_radical`, `is_surd_like`, `expr_contains_sqrt_or_half_power`…).
  Salida: mapa de familias duplicadas + propuesta de primitivo canónico +
  coste de migración. **No migrar aún.**
- **L2 — Capas y acoplamiento** (partición: por crate).
  Verificar la dirección declarada `cas_ast → cas_math → cas_engine →
  cas_solver_core → cas_solver → cas_session* → cli/wasm/didactic`.
  Detectar: re-exports que perforan capas, lógica de dominio en crates de
  plumbing y viceversa, dependencias cíclicas lógicas (aunque compilen),
  tipos públicos que deberían ser pub(crate). Semilla: el catálogo de
  re-exports de `cas_solver::runtime`.
- **L3 — Código muerto CON clasificación domain/plumbing** (partición: por
  crate). Para cada símbolo 0-refs: (a) plumbing → candidato a borrar;
  (b) dominio → ficha de "capacidad a medio cablear" con qué faltaría para
  cablearla (esto ALIMENTA el backlog de universalidad, no la papelera).
  Herramienta: `cargo +nightly udeps` si está, si no `rg` de nombres +
  `#[allow(dead_code)]` existentes.
- **L4 — Calidad adyacente a hotspots** (partición: orchestrator.rs /
  arithmetic.rs / solve_backend_local.rs / focused_rule_substeps.rs — los
  4 monstruos de >10k líneas). Medido en campaña: `orchestrator.rs` ~30k
  líneas con `simplify_pipeline_inner` de ~3.5k y ~66 root-shortcuts (solo
  13 instrumentados). Salida: plan de partición POR COHESIÓN en módulos
  (sin mover semántica), lista de shortcuts sin instrumentar (candidatos a
  `orchestrator_shortcut_profiler`), funciones >300 líneas con corte
  natural. **Solo diseño de partición; la ejecución va en F3.**
- **L5 — Contratos sin test** (partición: por crate). Todo comentario
  normativo ("MUST", "never", "always", "invariant", "sound") sin test que
  lo pinne → ficha de test de contrato propuesto (con esqueleto). Semillas:
  contrato de hold.rs, ciclo de vida de sticky, exactitud de gates, "el
  resultado de expand queda expandido".
- **L6 — Errores y paths de pánico** (partición: por crate). unwrap/expect/
  panic en rutas de producción (fuera de tests), índices sin guard,
  aritmética que puede desbordar. Cruzar con `lint-no-panic-prod` existente
  y catalogar las excepciones toleradas. Semilla: el crash cross-arena de
  esta campaña (`Context::get` index-out-of-bounds) — ¿hay más caches
  ExprId-keyed sin `instance_tag`?
- **L7 — Consistencia de presupuestos y caps** (transversal, 1 agente).
  Inventario de TODOS los budgets/caps/timeouts (`max_terms`, `MAX_TERMS`,
  `PROBE_BUDGET`, `time_budget_ms`, N=60-style constants). Para cada uno:
  ¿está documentado su porqué con MEDICIÓN (como el 8916 del reductor)?
  ¿reporta cuando recorta o traga silenciosamente? ¿es alcanzable por
  inputs razonables? Los comentarios de presupuesto documentan el coste:
  leerlos como perfiles (lección tanda 5).
- **L8 — Métricas de perf como síntoma** (transversal, 1 agente).
  Re-correr `scripts/corpus_timing.py`, tomar el top-20 y para cada caso
  >50ms: ¿el coste es esencial (trabajo matemático real) o accidental
  (repetición, rebuild, churn)? Etiquetar con el patrón de la campaña:
  rewriter-término-a-término, memo-neutralizado, sonda-sin-gate,
  serie-sin-retícula, registry-por-sonda. Salida: candidatos de perf con
  su clase de defecto — la cola de la próxima campaña.

## 5. Fase 2 — Verificación adversarial (paralela por hallazgo)

Cada hallazgo de F1 pasa por un agente verificador INDEPENDIENTE (no el
autor) con instrucción explícita de **REFUTAR** (lección audit5: hunter y
verifier pueden compartir el mismo misread; lección C1.9: el fixture
never-confirm caza unsoundness):

1. Confirmar contra el código ACTUAL (cita file:line viva, no de memoria).
2. Clasificar: ¿toca invariante I1-I10? ¿hot path (GATE-B)? ¿cubierto por
   tests hoy?
3. Redactar el criterio de aceptación mecánico del fix (qué test/gate
   demuestra que el cambio es correcto Y que el defecto existía).
4. Veredicto: CONFIRMADO / REFUTADO / REDIRIGIDO (el defecto real es otro —
   patrón frecuente: 3 de 5 atribuciones de esta campaña se redirigieron al
   medir).

Presupuesto: kill-rate esperado ≥40%. Solo lo CONFIRMADO pasa a F3.

## 6. Fase 3 — Remediación por lotes desacoplados (paralela por worktree)

- Agrupar hallazgos confirmados en lotes **sin intersección de archivos**
  (el grafo de conflictos se calcula con las rutas de las fichas). Lotes
  típicos: "partición de orchestrator.rs en módulos" (L4), "unificación de
  memos sobre instance_tag" (L1), "tests de contrato de hold" (L5),
  "borrado plumbing 0-refs" (L3a), "catálogo capacidad a-medio-cablear"
  (L3b, solo documentación).
- Cada lote: un agente en **worktree aislado**, entrega cambio + tests
  nuevos + GATE-A completo (+ GATE-B si toca crates calientes). Un commit
  por lote (bisect-friendly), mensaje en español estilo repo.
- Prioridad de lotes (ROI para universalidad):
  1. L5 (contratos sin test) — convierte invariantes implícitos en red.
  2. L1 (primitivos unificados) — cada duplicado es un futuro divergente.
  3. L4 (partición de monstruos) — habilita los audits siguientes.
  4. L6 (paths de pánico) — robustez.
  5. L3a (plumbing muerto) — última, la más barata y la más arriesgada de
     hacer a ciegas.
- Integración: los lotes aterrizan SECUENCIALMENTE en main (rebase +
  GATE-A re-corrido tras cada aterrizaje). Paralelo en desarrollo,
  serializado en merge.

## 7. Formato de ficha de hallazgo (obligatorio, machine-friendly)

```markdown
### [L4-017] simplify_pipeline_inner: bloque de shortcuts trig sin cohesión
- **Archivo**: crates/cas_engine/src/orchestrator.rs:26496-26544
- **Lente**: L4 (partición por cohesión)
- **Defecto**: 3 shortcuts consecutivos comparten el mismo pre-check
  duplicado 3 veces (líneas X, Y, Z).
- **Propuesta**: extraer `try_trig_zero_family_shortcuts(...)` en
  `orchestrator/shortcuts_trig.rs`; sin cambio semántico.
- **Riesgo**: I5 (orden) — la extracción DEBE preservar el orden de probes.
- **Aceptación**: GATE-A + GATE-B(solve trazadores) + diff de orden de
  secciones del shortcut-profiler vacío.
- **Verificación (F2)**: CONFIRMADO por <agente> — <fecha> — <cita>.
```

## 8. Plantilla de prompt por agente (F1; F2/F3 análogas)

> Eres el agente de la lente **L<k>** sobre la partición **<crates/archivos>**
> del repo ExpliCAS. SOLO LECTURA: no editas nada. Tu contrato: (1) la lista
> de invariantes I1-I10 de docs/PLAN_AUDITORIA_CALIDAD_2026-07-29.md §1 es
> intocable y todo hallazgo que la roce debe declararlo; (2) cada hallazgo en
> el formato §7, con file:line del código ACTUAL; (3) nada de gustos: cada
> ficha necesita un defecto ARGUMENTADO (duplicación medible, capa violada,
> contrato sin test, coste medido) y un criterio de aceptación mecánico;
> (4) si detectas lo que parece un bug de comportamiento, va a
> HALLAZGOS_CAPACIDAD.md, no lo arregles; (5) máximo 25 fichas: prioriza por
> impacto en universalidad (soundness > contratos > duplicación > estética,
> y estética NO entra).

## 9. DAG de paralelización

```
F0 (1 agente, secuencial)
 └─> F1: L1×3 ∥ L2×5 ∥ L3×5 ∥ L4×4 ∥ L5×5 ∥ L6×5 ∥ L7×1 ∥ L8×1  (≈29 tareas, sin conflictos: solo lectura)
      └─> F2: 1 verificador por ficha, agrupados por lente (∥ total; presupuesto: ~2 fichas/agente)
           └─> F3: lotes desacoplados por archivos (∥ en worktrees; merge secuencial con GATE-A)
                └─> F4 (1 agente): re-baseline completo (F0.3 + GATE-C + corpus timing),
                     comparación antes/después, actualización de ledger/memoria,
                     residuales → backlog
```

Presupuesto orientativo: F1 ≈ 29 agentes de lectura; F2 ≈ 30-60 verificadores
cortos; F3 ≈ 6-10 lotes; F4 = 1. Criterio de parada por fase: F1/F2 se cierran
por lista completa, no por tiempo; F3 se corta cuando el lote siguiente no
justifica su GATE-A (~15 min de suite por lote).

## 10. Qué NO es este plan

- No es una campaña de renombrados, reformateos ni "DRY" especulativo.
- No es una campaña de capacidades: los bugs de comportamiento que aparezcan
  se catalogan y se atacan con el proceso de auto-mejora normal.
- No borra dead code de dominio: lo convierte en backlog de universalidad.
- No introduce dependencias nuevas (herramientas: las del repo + rg/tokei).

## 11. Métricas de éxito del audit completo

- Contratos: N invariantes pineados con test nuevo (objetivo: ≥15).
- Duplicación: familias de primitivos unificadas (objetivo: memos → 1 molde
  sobre `Context::instance_tag`; holds documentados como 2-por-diseño o
  unificados).
- Tamaño: ningún archivo nuevo >5k líneas; orchestrator.rs particionado o
  con plan aprobado.
- Conducta: GATE-A vacío en TODOS los lotes (cero cambios de resultado).
- Perf: GATE-B sin regresión; la cola L8 alimenta la campaña 3ª.
- Universalidad: catálogo L3b de capacidad a medio cablear entregado como
  backlog priorizado.
