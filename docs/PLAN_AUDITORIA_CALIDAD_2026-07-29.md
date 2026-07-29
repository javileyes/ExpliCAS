# Plan de auditoría integral — soundness × universalidad × didáctica, multi-agente

**Fecha**: 2026-07-29 (v2) · **Estado**: PLAN (sin ejecutar) · **Supersede** la
v1 "auditoría de calidad": el soundness pasa de invariante protegido a
OBJETIVO de primera clase, y el plan se reorganiza alrededor de la meta.

## 0. La meta y su descomposición

Meta del proyecto: **universalidad con pasos didácticos**. Descompuesta en las
tres dimensiones que la definen — y en su multiplicador:

- **S — Soundness**: nada falso. Ni un resultado incorrecto ni un gate que
  decida con aproximaciones. Un CAS que responde mal no es universal por
  mucho que responda.
- **U — Universalidad (completitud)**: menos residuales honestos, más
  familias cerradas, capacidad a medio cablear terminada. Un CAS que declina
  es honesto pero incompleto.
- **D — Didáctica**: cada paso publicado es verdadero, no vacío, y pedagógico
  (el contrato «publicado ⇒ verificado» del Frente E).
- **Q — Calidad de código (multiplicador)**: cohesión, desacoplo, sin
  duplicados divergentes, sin enredo. No es meta en sí: es lo que hace baratas
  y seguras las otras tres. La campaña de perf 2ª lo demostró en ambos
  sentidos (defectos de diseño → lentitud; medir → sanear).

Precedencia ante conflicto: **S > D > U > Q**. Un P0 de soundness bloquea
cualquier otro lote que toque los mismos archivos.

## 1. Reglas del juego (contrato literal para TODOS los agentes)

- **R1** `as_rational_const` no se modifica.
- **R2** Gates de soundness exactos: jamás f64 para drop/keep. (En pista S
  esto es además CAZA ACTIVA: encontrar los que hoy no lo son.)
- **R3** `__hold`/`Expr::Hold`: contrato de `cas_ast/src/hold.rs`; strip solo
  en fronteras de salida.
- **R4** Ciclos de vida de sticky/memos (P16, sondas): cambiarlos exige el
  análisis de invalidación por escrito (la campaña 2ª documenta el molde).
- **R5** El orden de reglas y root-shortcuts es semántico: no se reordena.
- **R6** Budgets/caps: instrumentables y ajustables, nunca eliminados ni
  silenciosos.
- **R7** Cambios de COMPORTAMIENTO: prohibidos en pistas Q/U/D. En pista S
  están PERMITIDOS únicamente para fichas S-P0 CONFIRMADAS con oráculo
  exacto (la corrección de un wrong answer cambia el comportamiento porque
  el comportamiento era falso); siguen el proceso de la skill auto-mejora
  (candidato → fix acotado → huella → ledger → commit propio).
- **R8** Dead code en crates de dominio no se borra: se cataloga (pista U).
  Solo plumbing 0-refs es borrable (pista Q).
- **R9** Nombres visibles de reglas/steps no cambian (hay pins).
- **R10** Push = deploy: nadie pushea. Commit por lote, mensaje en español.
- **R11** «La “correct answer” de un audit es hipótesis, no oráculo»: todo
  P0 de soundness exige verificación NUMÉRICA EXACTA independiente
  (Fraction/BigRational literal, jamás float — lección
  sympy-subs-masks-poles) antes de tocar código.
- **R12** Atribución antes de arreglar: contadores/perfil primero (3 de 5
  atribuciones de la campaña 2ª se redirigieron al medir).

## 2. Gates mecánicos únicos (compartidos por todas las pistas)

- **GATE-A (conducta)**: (1) `cargo test --workspace` exit 0 con log crudo
  (`> log 2>&1; echo EXIT=$?` — nunca juzgar por un pipe); (2) diff
  conductual VACÍO del corpus 221 (`scripts/corpus_behavior_diff.py`,
  result+steps_count+solve_steps+substeps+warnings); (3) clippy limpio.
  *Excepción R7*: una ficha S-P0 tiene diff NO vacío esperado — su gate es
  «diff EXACTAMENTE las filas previstas por la ficha, ni una más».
- **GATE-B (perf no-regresión)**: los 8 trazadores de la campaña 2ª
  (`simplify_us` mediana de 3, presupuesto = base × 1.25).
- **GATE-C (huella)**: `make engine-scorecard` — contadores estructurales y
  slot identity idénticos (campos de latencia se ignoran).
- **GATE-D (steps)**: (1) barrido diferencial de steps del gate
  anti-divergencia (dos perfiles) sin divergencias nuevas; (2) verificador
  de claims (`substep_claim`/`verify_claim`, maquinaria C1.9) con 0
  refutaciones sobre la muestra tocada.
- Lección de la campaña 2ª sobre cobertura: **el diff de corpus NO cazó el
  crash cross-arena; el workspace sí** — por eso GATE-A exige ambos, y por
  eso la pista S usa generadores además del corpus fijo.

## 3. Fase 0 — Preparación común (secuencial, 1 agente)

- **F0.1** Working tree limpio, commit de partida anotado (los agentes de
  workflow mutan el árbol: commitear antes de lanzar nada).
- **F0.2** Materializar harnesses como `scripts/` del repo:
  `corpus_behavior_diff.py`, `corpus_timing.py` (portar del scratchpad de la
  sesión b5a0d3cf), y `sound_probe.py` (nuevo: evalúa una identidad en
  puntos racionales con `Fraction` exacto — el oráculo R11).
- **F0.3** Baseline commiteado en `docs/generated/quality_audit_baseline/`:
  tokei/wc por crate, archivos >5k líneas, funciones >300 líneas,
  `cargo tree -d`, salidas de los lints del Makefile, corpus timing, huella.
- **F0.4** Consolidar el backlog previo: fichas abiertas de los
  frontier-audits (11 P2 del 2026-07-13, F13/F3/F14 del 07-14, pendientes
  del Frente E §Pendientes, residuales de límites) en un índice único
  `docs/generated/quality_audit/BACKLOG_PREVIO.md` — la pista U parte de ahí,
  no de cero.

## 4. Fase 1 — Cuatro pistas de lectura/medición (paralelismo total)

Todos los agentes de F1 son de SOLO LECTURA (+ ejecución de binarios de
medición). Nadie muta código ⟹ sin conflictos. Cada lente produce
`docs/generated/quality_audit/<pista><n>_<slug>.md` con fichas (§7).

### Pista S — Caza de unsoundness (la nueva primera clase)

- **S1 — Differential/metamórfico generativo** (×3 agentes: álgebra/trig,
  calculus, solve). Generar familias paramétricas (plantilla: las 40
  familias del frontier-audit 2026-07-09) y comparar cada
  `simplify/derive/integrate` contra el oráculo exacto de `sound_probe.py`
  en ≥5 puntos racionales seguros (fuera de polos — detectarlos con
  denominador exacto). Discrepancia ⟹ ficha S-P0. Barrer formas desnudas Y
  simplificadas (lección: el audit de formas desnudas pierde las
  simplificadas) y variantes de signo/paridad (lección: la reducción cubre
  el caso impar y pierde el par/negado).
- **S2 — Auditoría de gates de decisión** (×2: cas_engine, cas_math+solver).
  Inventario de todo punto drop/keep (`provable_*`, `poly_eq`, `are_equal`,
  `is_zero`, comparaciones de score): ¿exacto, conservador o aproximado?
  Grep dirigido de f64/`to_f64`/floats en rutas de decisión + clasificación.
  Cada gate aproximado ⟹ ficha S (severidad por alcance).
- **S3 — Solve: emisión sin verificación** (×1). Mapear handlers que emiten
  SolutionSet sin verificación por sustitución exacta o con verificación
  más débil que la emisión (precedente: el radical-product filter). Para
  inecuaciones: muestreo exacto dentro/fuera de cada intervalo emitido.
- **S4 — Condiciones perdidas** (×1). Requires/domain conditions que se
  derivan y luego se pierden por el camino al wire (precedentes: E3 y el
  filtro de dominios de radicales). Muestreo de casos con `sqrt/ln/1/x` y
  comprobación de que las condiciones llegan al usuario.
- **S5 — Fixtures never-confirm** (×1). Extender el detector que cazó 2 P0
  en Fase 2 compleja: catálogo de equivalencias FALSAS plausibles (con
  |x|, ramas, dominios) que el equivalence-checker jamás debe confirmar,
  como suite permanente.

### Pista Q — Calidad de código (multiplicador; lentes de la v1 podadas)

- **Q1 — Duplicación de moldes** (×2). Semillas: los 4+ memos thread-local
  con contratos de invalidación distintos → un molde canónico sobre
  `Context::instance_tag`; 2 mecanismos de hold (¿2-por-diseño? documentar
  o unificar); walkers `contains_*` repetidos; variantes de eval numérico.
- **Q2 — Capas y acoplamiento** (×2). Dirección declarada
  `ast → math → engine → solver_core → solver → session → cli/wasm/didactic`:
  re-exports que perforan, dominio en plumbing y viceversa, pub que debería
  ser pub(crate).
- **Q3 — Monstruos por cohesión** (×4: orchestrator.rs ~30k,
  arithmetic.rs ~30k, solve_backend_local.rs, focused_rule_substeps.rs).
  SOLO diseño de partición en módulos cohesivos sin mover semántica (R5),
  + lista de los ~53 root-shortcuts sin instrumentar en el
  shortcut-profiler.
- **Q4 — Paths de pánico** (×2). unwrap/expect/index fuera de tests; cruzar
  con `lint-no-panic-prod`; pregunta heredada de la campaña: ¿más caches
  ExprId-keyed sin `instance_tag`?
- **Q5 — Presupuestos y caps** (×1). Inventario completo; para cada uno:
  ¿su porqué está MEDIDO y documentado (el «8916 pasos» del reductor era un
  perfil escrito en un comentario)? ¿recorta en silencio (R6)?

### Pista U — Universalidad (completitud)

- **U1 — Mapa de residuales** (×2). Correr corpus + familias de F0.4 y
  catalogar TODOS los declines/residuales honestos (integrate/solve/limit/
  dsolve) con causa y frecuencia; salida = backlog priorizado por ROI
  (familia × frecuencia × coste estimado).
- **U2 — Capacidad a medio cablear** (×2). El «dead code» de dominio
  clasificado: qué es, qué le falta para cablearse, qué familia cerraría.
  Alimenta U1, no la papelera.
- **U3 — Paridad de formas en handlers** (×1). Barrido sistemático de
  matchers con casos nombrados: ¿cubren negativo/par/recíproco/wrapper
  afín? (lecciones scout-workflow y reaudit-post-fixes).

### Pista D — Didáctica

- **D1 — Muestreo ciego de steps** (×2). N=30 filas aleatorias del corpus
  con steps on, verificación humana-grado de cada paso (claims con la
  maquinaria C1.9 donde aplique; ojo → ficha donde no). La lección del
  re-audit del Frente E: una campaña cerrada por métricas propias necesita
  muestreo ciego externo.
- **D2 — Pasos vacíos / no-mejorantes** (×1). Detector sobre el corpus de
  pasos cuyo before/after empatan en score NF sin estar en la lista
  didáctica (el «Factor out 3» de esta sesión como semilla); + pasos cuyo
  texto no re-parsea (lección Canonicalize Roots).
- **D3 — Cobertura de narración** (×1). Familias con resultado correcto y
  0 narración (las 15/42 de solve que no narran, inecuaciones familia I,
  |x|=a paramétrico) — consolidar el mapa del Frente E con estado actual.

## 5. Fase 2 — Verificación adversarial (paralela por ficha)

Cada ficha pasa por un verificador INDEPENDIENTE con instrucción de REFUTAR
(hunter y verifier pueden compartir el mismo misread):

1. Confirmar contra código/salida ACTUAL (cita viva file:line o
   reproducción CLI).
2. **Fichas S además**: reproducir el wrong answer con `sound_probe.py`
   (oráculo exacto, R11). Sin reproducción numérica exacta no hay P0.
3. Clasificar riesgo (¿toca R1-R12? ¿hot path? ¿cubierto por tests?).
4. Redactar el criterio de aceptación mecánico del fix.
5. Veredicto: CONFIRMADO / REFUTADO / REDIRIGIDO.

Kill-rate esperado ≥40%. Solo lo CONFIRMADO pasa a F3.

## 6. Fase 3 — Remediación por lotes desacoplados (paralela por worktree)

- Orden de aterrizaje: **S-P0 → D-falsos → Q-contratos/tests → U-cierres →
  Q-restos → borrado plumbing** (lo más barato y ciego, al final).
- Lotes sin intersección de archivos; un agente por lote en worktree
  aislado; entrega = cambio + tests nuevos + GATE-A(+B si caliente,
  +D si toca steps) verdes; un commit por lote.
- Fichas S-P0 usan el proceso de la skill auto-mejora (ciclo completo con
  huella y ledger) — no el carril rápido de Q.
- Merge secuencial en main con GATE-A re-corrido tras cada aterrizaje.
- Cada lote Q de «partición de monstruo» debe ser un move-only verificable:
  mismo set de símbolos antes/después (`rg` de firmas), GATE-C idéntico.

## 7. Ficha de hallazgo (formato único, machine-friendly)

```markdown
### [S1-004] simplify colapsa |x|·sign(x) a x sin condición
- **Pista/Lente**: S1 (differential trig/álgebra)
- **Reproducción**: `expli eval "abs(x)*sign(x)" ...` → `x`;
  sound_probe.py en x=-3/7 (exacto): motor=-3/7, identidad=−3/7 ✓/✗ …
- **Severidad**: P0 (wrong answer) / P1 (gate aproximado) / P2 (cosmético)
- **Riesgo**: toca R-…; hot path sí/no; tests que lo cubren hoy: …
- **Aceptación**: test exacto nuevo + diff conductual = exactamente las
  filas previstas + GATE-D sin divergencias.
- **Verificación (F2)**: CONFIRMADO por <agente> — <cita/reproducción>.
```

## 8. Plantillas de prompt (esqueleto por pista)

Común: «Contrato R1-R12 de docs/PLAN_AUDITORIA_CALIDAD_2026-07-29.md §1.
SOLO LECTURA/medición. Fichas en formato §7, máx 25, priorizadas por impacto
en la meta (S > D > U > Q). Nada de estética. Si dudas entre dos lentes, la
ficha va a la más severa.»

- S: «Tu oráculo es sound_probe.py con racionales exactos; una discrepancia
  sin reproducción exacta NO es ficha. Barre formas desnudas Y simplificadas,
  y las variantes par/negado/recíproco de cada patrón.»
- Q: «Cada ficha necesita defecto ARGUMENTADO (duplicación medible, capa
  violada, contrato sin test, coste medido) — un gusto no es un defecto.»
- U: «Tu salida es un backlog priorizado por familia×frecuencia×coste; un
  residual honesto no es un bug, es un candidato.»
- D: «Un paso es defecto si es falso, vacío (sin mejora y sin valor
  didáctico declarado), no re-parseable o mudo donde debería narrar.»

## 9. DAG de ejecución

```
F0 (1 agente)
 └─> F1  S1×3 ∥ S2×2 ∥ S3 ∥ S4 ∥ S5        (8)
        ∥ Q1×2 ∥ Q2×2 ∥ Q3×4 ∥ Q4×2 ∥ Q5  (11)
        ∥ U1×2 ∥ U2×2 ∥ U3                 (5)
        ∥ D1×2 ∥ D2 ∥ D3                   (4)   ≈ 28 lectores, cero conflictos
      └─> F2  1 verificador/ficha, agrupado por pista (kill-rate ≥40%)
           └─> F3  lotes desacoplados en worktrees; merge secuencial
                   orden: S-P0 → D-falsos → Q-tests → U → Q-restos → plumbing
                └─> F4 (1 agente) re-baseline F0.3 + GATE-C/D globales +
                     actualización ledger/memoria + backlog residual v2
```

Criterios de parada: F1/F2 por lista completa; F3 se corta cuando el
siguiente lote no justifica su GATE-A (~15 min de suite); las fichas S-P0
NUNCA se cortan por presupuesto — se aparcan explícitamente en el ledger si
no caben.

## 10. Métricas de éxito (alineadas con la meta)

- **S**: 0 wrong answers reproducibles abiertos al cierre; inventario de
  gates con su clase (exacto/conservador) publicado; suite never-confirm
  ampliada y en CI.
- **D**: 0 pasos falsos en el muestreo ciego final (N=30 nuevo); detector de
  pasos vacíos integrado al harness del Frente E.
- **U**: backlog priorizado único (residuales + capacidad a medio cablear +
  paridad de formas) con ROI estimado — el alimento de los próximos ciclos
  de auto-mejora.
- **Q**: ≥15 contratos pineados con test; memos unificados sobre
  `instance_tag`; plan de partición de los 4 monstruos aprobado o ejecutado;
  GATE-A vacío en todos los lotes Q/U/D.
- **Global**: GATE-B sin regresión; huella idéntica salvo fichas S-P0
  documentadas.

## 11. Qué NO es este plan

- No es renombrado/reformateo/DRY especulativo (un gusto no es un defecto).
- No es una campaña de capacidades nuevas: U produce el BACKLOG; los cierres
  de familia siguen el proceso de auto-mejora habitual.
- No borra dead code de dominio; lo convierte en mapa de capacidad.
- No introduce dependencias nuevas.
- No corrige comportamiento fuera del carril S-P0 (R7).
