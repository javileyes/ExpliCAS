# Plan de remediación de steps y highlights — 2026-07-25

**Estado:** plan de ejecución. Sustituye a la sección «Prioridad sugerida» del informe
`docs/AUDITORIA_STEPS_HIGHLIGHTS_2026-07-25.md`, que queda como diagnóstico y como fuente de
hallazgos, no como orden de trabajo.

**Base:** commit `bdeb56429` (rama `auditoria/steps-highlights-2026-07-25`, motor intacto).
546 hallazgos / 185 filas / 53 P0 sobre las 210 filas de `web/examples.csv`.

**Procedencia de las cifras.** Las de hallazgos salen de
`docs/auditoria_steps_highlights_2026-07-25_findings.json` (546 registros, recontados hoy). Las de
resaltado salen de una sonda instrumentada sobre `Simplifier::with_default_rules()` +
`to_display_steps` (480 pasos de display) y de dos scripts de análisis del wire
(`scratchpad/audit/probe_guard.py`, `probe_order.py`). Las de sub-paso salen del prototipo del guard
numérico (`scratchpad/audit/guard_probe.py` + `guard_probe.json`, 214 sub-pasos). Las de solve e
integrales salen de `scratchpad/audit/run_corpus.py` + `detect.py` sobre `scratchpad/audit/json/`.
Ninguna sonda dejó ficheros en el repo.

---

## 1. Tesis y principio rector

### Principio rector: **publicado ⇒ verificado**

El motor no publicará ninguna afirmación matemática que no pueda comprobar en el momento de
emitirla. Lo que no verifica **declina**, y declinar es una salida legítima que además queda
**contada**. El path grabado, el nombre de la regla y el método interno del integrador dejan de ser
autoridades y pasan a ser **candidatos que hay que probar contra el delta semántico del paso**.

Esto es un criterio de **admisión**, no una prioridad más dentro de una lista. Bajo él, los 546
hallazgos no se ordenan por ROI sino por tipo de daño:

1. **El motor AFIRMA algo falso** (`3·(−3) → (−3)²`, `∫−2sin x dx = 2x·sin x + (2−x²)cos x`,
   un `F(b)−F(a)` que evalúa π/4 donde vale π/12, «x²+1 no tiene solución» sin decir «en ℝ»).
2. **El motor CALLA** (magia, filas mudas, sub-pasos ausentes).
3. **El motor es FEO o INCONSISTENTE** (orden de términos, fugas de idioma, contadores).

El tipo (1) va primero **siempre**, y se ataca preferentemente con guards que convierten falso en
ausente — es decir, moviendo daño de (1) a (2) — antes que con arreglos de foco que pueden volver a
fallar mañana. El tipo (3) va al final, **salvo cuando corrompe la medida con la que juzgamos (1) y
(2)**: ese es el caso de `steps_count`, y por eso el ciclo 1 es de instrumento.

### Tesis

El sustrato que falta **no es** un espacio de árboles compartido ni un canal de traza
método→didáctica (los dos arreglos «de arquitectura» que el informe deduce de su propio
diagnóstico). Es un **invariante de publicación** instalado en los tres chokepoints por los que sale
la narración:

| Chokepoint | Fichero | Qué publica | Verificador |
|---|---|---|---|
| Plan de resaltado | `cas_didactic/src/timeline/simplify_highlights/global.rs:14` | `before_latex`/`after_latex`/`rule_latex` con color | **Reconstrucción**: sustituir el rojo por el verde debe reproducir `global_after` (ExprId) |
| Sub-paso | `cas_didactic/src/didactic/focused_rule_substeps.rs` (219 de los 235 emisores) | `SubStep` | **Relación tipada** (Equality / Derivative / Antiderivative / DefiniteEval / abstención) |
| Cadena de narradores | `focused_rule_substeps.rs:49` (`generate_focused_rule_substeps`) | sub-pasos hijos | **Re-derivación**: partir, re-derivar y comprobar que la suma reproduce el `after` |

Coste: **23 ciclos acotados**, uno por entrada de ledger. Resultado: **0 falsedades publicadas, 0
publicaciones sin verificar**, y un inventario medido y decreciente de declives y fronteras abiertas.
**No** deja el corpus en 0 hallazgos, y eso se dice por adelantado.

---

## 2. Las cuatro refutaciones que reordenan el informe

Cuatro medidas mandan sobre el orden que sugiere el documento de auditoría:

**(a) El arreglo nº1 del informe (RC-1, «transportar el path a través de
`normalize_expr_for_display`») es inviable.** `Context::add` (`cas_ast/src/expression.rs:356`) no
permuta hijos: **aplana** a n-ario con `collect_add_terms`, **ordena** con `compare_add_terms` y
**reconstruye** con `build_balanced_add` (las tres privadas en `cas_ast`), de modo que el path cambia
de **longitud**, no de índice. `normalize_expr_for_display`
(`cas_solver_core/src/eval_step_pipeline.rs:278`) además colapsa niveles y llama a
`compact_root_denominator_fraction_product_for_display`, que desarma y rearma el producto entero
desde listas de factores: ahí no hay transporte definible. Y aun con transporte perfecto, **en 87 de
480 pasos (18 %) el nodo foco no tiene imagen en el árbol normalizado**; el «fallback por
estructura» de `find_absolute_path` recupera **0 de esos 87** (`diff_find_paths_by_structure` usa
`compare_expr`, que sobre árboles canónicos es igualdad de ExprId: es código que parece red de
seguridad y no lo es). → Se hace la opción (b), **foco por contenido**, con el guard de desempate.

**(b) El guard (RC-3) está mal ubicado en el informe.** Situado en
`step_payloads/build/latex.rs:145` opera sobre **strings** y solo protege `rule_latex`, dejando
`before_latex`/`after_latex` publicando el rojo/verde falso — que es justo lo que la web pinta
(`web/index.html:2373-2398 renderSteps`). A nivel string declina 37/250 pasos del wire, de los cuales
**12 son declinaciones benignas** (6 de reordenación canónica, 4 de plegado colateral). Movido al
chokepoint y **a nivel de expresión** recupera 10 de esas 12 y sube la precisión de ~68 % a ~93 %.
→ El guard va a `simplify_highlights/global.rs:14`, y va **antes** que el foco por contenido.

**(c) Los P0 más graves no tienen causa raíz asignada.** RC-1..RC-16 no cubren los 7 defectos de
sub-paso; el propio informe lo admite («los sub-pasos nunca entraron en el harness»). Son 12 de los
53 P0 y **los más baratos de cerrar**: dos ciclos S sin un solo pin externo en riesgo cierran 52
hallazgos.

**(d) La clase de fuga de idioma está infra-medida 6×.** «9 nombres de regla, paridad estructural
210/210 correcta» describe `web/examples.csv`, que es una **vitrina** que enruta por las rutas de
cálculo bien traducidas. Sobre `identity_pairs.csv` (corpus guardrail del propio repo) fuga el
**26 % de las filas** (79/300 muestreadas, 134 hits, 55 cadenas distintas), y
`cas_cli eval "sqrt(x^2)" --steps on --lang en` sale **íntegro en español**. Es el mismo error que la
auditoría denuncia, cometido dentro de la auditoría. → Todo contador nuevo corre sobre los corpus
**guardrail**, no solo sobre la vitrina.

**Corrección menor pero operativa:** el renderizador de la web es **MathJax 3 tex-svg**
(`web/index.html:18`, `MathJax.typesetPromise` en :2314/:2483/:2971), **no KaTeX** como dice el
informe. El diagnóstico no cambia; la elección del escaper del ciclo C1.1 sí.

---

## 3. Desacuerdos resueltos entre las tres estrategias

| Punto | Alternativas | Decisión | Razón |
|---|---|---|---|
| ¿Ciclo 1? | guard / fallback-latex / contadores+carril | **Contadores + carril** | Todos los criterios de salida son contadores y el contador miente en 24/210 filas y no ve 214 sub-pasos. Es exactamente el instrumento con el que el frente E se declaró completo dejando su residual A=31 intacto (hoy D4=31: nada se movió y nadie lo notó). Coste 1 ciclo, riesgo 0, cero cambio visible para el alumno. |
| ¿Guard antes o después de los P0 de sub-paso? | guard 1º / sub-pasos 1º | **Dos ciclos S de sub-paso, luego el guard** | Los dos S cierran 52 hallazgos (7 P0) sin un pin externo y **mejoran** lo que ve el alumno; el guard es M, tumba 12 pins que exigen diagnóstico a mano y **empeora la densidad visual**. Se aterriza con dos mejoras visibles delante y con el carril ya midiendo. |
| RC-2 (`Mul`/`Neg` en `preferred_local_scope`) | ciclo propio / absorbido | **Absorbido en C2.1** (3 líneas) | Está dominado: vacía la ruta DEFAULT-POSITIONAL pero la manda a la búsqueda por contenido, que es RC-1. Con foco por contenido esa lista deja de decidir nada; no merece entrada de ledger. |
| Multi-span (ruta aditiva) | ciclo propio / diferido / ignorado | **Ciclo propio S al final de F2** | «Publicado ⇒ verificado» no puede tener una excepción permanente. Son 9/292 pasos del wire; el guard los declara excepción declarada en C1.3 y C2.2 la cierra. |
| Contadores | estricto / «minimal» (`combined` en todas las rutas) | **Estricto** | Medido: de los 163 pares (test, expresión) que asertan `steps_count`, **cero** cae bajo el estricto; el «minimal» **rompe** `solve_system_educational_s3_contract`, que asserta `"steps_count": 0` como prueba de cero fugas. El estricto refuerza ese pin, el minimal lo afloja. |
| RC-13 (texto plano en substeps) | ciclo M / rider | **Rider de C0.1** | `SubStep` YA lleva `before_expr`/`after_expr` en texto plano (`didactic/types/substep.rs:8-11`); es `SubStepWire` (`cas_api_models/src/wire_types.rs:172-181`) el que no los publica. Dos campos y un productor. Corrige además el diagnóstico del informe: `SolveSubStepWire` **sí** lleva texto plano. |
| i18n | fase 3 / fase 5 | **Fase 5**, salvo el aviso de dominio complejo, que **sube a F1** | El aviso es verdad (un absoluto sin dominio), no léxico. El resto es tipo (3). |
| Verbos vectoriales (P0-S3-6) | ciclo L / tanda / fuera | **Fuera del plan**, frontera abierta con contador | No es un bug local sino una decisión de diseño declarada en los doc-comments («narración a nivel de fórmula», `:11996` y `:12040`): 5 verbos × ~4 sub-pasos + locale. Es una tanda propia. Su mitad de **mentira** (el fallback declarado LaTeX) sí la cierra C1.1. |
| `navigate_to_subexpr` → `Option` | chip / rider | **Rider de C2.1** | Es el mecanismo por el que el drift es SILENCIOSO (ante un índice fuera de rango hace `break` y devuelve el vecino), y C2.1 ya toca esa capa. |
| Re-sondeo del backend (F8) | dentro / opcional | **Opcional, último de F3** | Es el único ítem con riesgo real de **huella de tiempo** (6-27 ms medidos contra `CALCULUS_RUNTIME_PRESSURE_WATCH_MAX_MS=150 / P95=75`). |

---

## 4. Fases y criterio de salida

### F0 — La regla de medir (1 ciclo)

Que exista un instrumento honesto **antes** de mover nada.

**Criterio de salida (medible):**
- `sum(steps_count) == sum(len(steps)) == 292`; filas con `steps_count != len(steps)`: **24 → 0**.
- Campos nuevos publicados y cuadrando: `solve_steps_count` total **128**, `substeps_count` total
  **263** (214 canal steps + 49 canal solve).
- `SubStepWire` publica `before`/`after` en texto plano en los 214 sub-pasos.
- Carril `steps_quality_gate_tests.rs` registrado en el scorecard con **D5 = D6 = D9 = 0** y
  `steps_count_mismatch_rows = 0` como aserción dura, y `steps_total/substeps_total/solve_steps_total`
  publicados como medida.
- `cargo test --workspace` failed:0, clippy 0, **huellas guardrail/fast/pressure con 0 deltas**.

### F1 — Nada falso en pantalla (8 ciclos)

Que ninguna de las dos superficies publique una afirmación que no pueda verificar al emitirla.

**Criterio de salida:**
- `substep_wrong_math`: **8 P0 → 0**; `substep_noop_or_false_claim` 16 → ≤2.
- Campos `*_latex` con texto de display **crudo**: **37 → 0** (14 filas); filas con el patrón `^(`:
  **11 → 0**.
- `E8_substep_noop` (sub-pasos con `before_latex == after_latex`): **5 → 0**.
- **0 pasos publican un span de color parcial sin pasar el guard** (excepción declarada: multi-span,
  9/292, hasta C2.2). `D1_red_equals_green` 6 → 0, `D1b` 5 → 0,
  `D2_hl_substitution_mismatch` 39 → ≤12.
- Los 5 P0 de identidad falsa citados a mano en el resumen ejecutivo ([001] `ln(x^(-3/2))`,
  [017] `∂/∂x` vs `∂/∂y`, [153] `2x²+2y²−4x²`, [180] `3·(−3)`, [192] `sin(u)→0²`) verificados **uno a
  uno con el CLI** como AUSENTES (los 5 están dentro del conjunto que el guard declina — comprobado).
- Los 7 testigos de la sonda 3 convertidos en pins explícitos, cada uno con su medida numérica.
- 4 filas del grupo Complejo dejan de publicar un absoluto sin dominio.
- `SubStep::checked` cubre el 100 % de los sub-pasos KEYED; el contador de emisores `Unchecked` (126)
  solo puede bajar.
- **Huellas guardrail IDÉNTICAS**: ningún ciclo de esta fase toca un resultado matemático.

### F2 — Recuperar la precisión del color (4 ciclos)

Devolver el resaltado fino a los pasos que **sí** admiten un span veraz, con el guard de juez. Por
construcción, ningún cambio de esta fase puede subir el contador de publicaciones no verificadas:
solo puede convertir declives en verificaciones.

**Criterio de salida:**
- guard-pass entre los que publican: **338/480 → ≥417/480** (86,9 %, el techo medido con span único).
- Declives: **142 → ≤63**, clasificados (DEFAULT 20, LOCAL-meta 12, LOCAL-shape 31).
- Ruta DEFAULT-POSITIONAL (78 pasos, hoy 50/78 fallando el guard): **→ 0**.
- `D2_hl_substitution_mismatch` ≤2; `D2b_hl_arity_mismatch` 19 → ≤5.
- `hl_published_unverified` = 0 **sin excepciones** (multi-span incluido).
- Paridad de renderizadores: **542/960 → ≥842/960**; las 118 divergencias estructurales pasan a eje
  de inventario con número.
- `wrong_rule_name` deja de ser anécdota: lista cerrable con inventario `--ignored`.

### F3 — Narración por re-derivación: integrales (5 ciclos)

La capa didáctica deja de ser un adivino pasivo (oráculos espejo sobre un `Context` clonado) y pasa a
ser un **re-derivador con obligación de prueba**: parte el problema, re-deriva cada parte y solo
publica si la reconstrucción reproduce el `after` del motor.

**Criterio de salida:**
- `mute_integration_steps` (pasos «Calcular la integral» con `substeps` vacío): **14 → 2**, y esos 2
  (fila 021 gaussiana definida, fila 025 by-parts cíclica) quedan **declarados** como residuales
  honestos con su familia nombrada.
- Reparto de las 22 filas de integrales (021-042): hoy 8 método real / 2 polo / 5 cáscara FTC /
  7 mudas → **20 con método real y 2 mudas honestas**.
- `E1_container_sin_substeps` 36 → ≤26.
- Invariante permanente: ningún narrador recursivo publica sin que su verificación pase.
- `E8_substep_noop` sigue en 0 (ningún narrador nuevo introduce un no-op).

### F4 — Solve deja de enmudecer (2 ciclos + inventario)

**Criterio de salida:**
- Filas mudas reales (ok=true, familia solve/dsolve, resultado **no residual**, **sin warnings**,
  `steps + solve_steps == 0`): **14 → 11**, y las 11 restantes listadas por familia y **dueño exacto**
  en el eje `--ignored`.
- Asimetría: inecuaciones mudas **12/16 → 9/16** (ecuaciones están en 2/21).
- Pin de dos niveles instalado: nivel 1 verde en el carril default; nivel 2 rojo por diseño.

### F5 — Léxico, avisos y UX medidos sobre los corpus guardrail (3 ciclos)

**Criterio de salida:**
- `spanish_residue_in_en_wire`: **29 → 0** sobre `examples.csv`; **134 → 0** sobre `identity_pairs.csv`
  (con inventario `--ignored` de las 55 cadenas mientras baja).
- `english_residue_in_es_wire`: **18 → 0**, incluida la clase «plantilla localizada con ARGUMENTO sin
  traducir» (`Solución condicional: Cannot prove RHS > 0 for logarithm`).
- Valla **por construcción**: todo literal español que `visible_rule_name` puede devolver es clave de
  `rule_name_es_to_en` (test que scrapea el fuente con `include_str!`).
- 0 avisos que remitan al alumno de la web a un comando de REPL inexistente.

---

## 5. Tabla de ciclos

Esfuerzo: S ≈ media jornada · M ≈ 1-2 jornadas · L ≈ 3+ jornadas o tanda.
«Cierra» = hallazgos de los 546 (P0 entre paréntesis).

| # | Título | RC / ítem | Esf. | Riesgo | Cierra | Depende de |
|---|---|---|---|---|---|---|
| **C0.1** | Un solo contrato de contadores + el carril nace verde + riders | RC-14, RC-13, RC-15 | M | bajo | 1 (0) | — |
| **C1.1** | Un fallback NUNCA se declara LaTeX | P0-S3-7 | S | medio | 44 (3) | C0.1 |
| **C1.2** | `F(b) − F(a)` es un nodo `Sub`, no una cadena (y el `\lim` se delimita) | P0-S3-1 | S | bajo | 8 (7) | — |
| **C1.3** | Guard de veracidad del resaltado a nivel de EXPRESIÓN en el chokepoint | RC-3 | M | medio | 11 (5) + neutraliza 17 P0 | C0.1 |
| **C1.4** | Fracciones parciales honestas: `before` = integrando, guardia de identidad, `∫dx/x` | P0-S3-4, RC-11 | M | medio | 15 (2) | — |
| **C1.5** | El `before` es el operando real: gate de fracción anidada + fila k del hessiano | P0-S3-3, P0-S3-5 | M | bajo | 8 (4) | C1.1 |
| **C1.6** | Por partes repetida: el cierre integra SU integrando + recomposición | P0-S3-2 | M | medio | 2 (1) | C1.8 |
| **C1.7** | Honestidad de dominio: avisar cuando ℝ DESCARTÓ algo (no cuando el texto dice `i`) | RC-14 (B3) | S | bajo | 8 (2) | — |
| **C1.8** | `SubStep::checked`: constructor verificador TIPADO POR RELACIÓN + inventario | P0-S3-GUARD | L | medio | 0 (red permanente) | C0.1 |
| **C2.1** | Foco por CONTENIDO con el guard de desempate (absorbe RC-2 y RC-5) + fold + riders | RC-1, RC-5, RC-2, RC-15, RC-16 | L | medio | 39 (22) | C1.3 |
| **C2.2** | Predicado multi-span: el contrato se queda sin excepciones | RC-3, RC-1 | S | medio | 0 | C2.1 |
| **C2.3** | Orden aditivo con comparador CONSCIENTE DEL SIGNO + el span que empieza tras el `−` | RC-6 | M | medio | ~30 (1) | C1.3 |
| **C2.4** | Nombre visible por (rule_name, description) + inventario de descripciones | RC-4 | S | bajo | 2 (1) | — |
| **C3.1** | Recursión acotada + narrador aditivo verificado POR RE-INTEGRACIÓN | RC-7, RC-10 | M | bajo | 3 (0) | C0.1 |
| **C3.2** | Brazo vectorial del mismo molde + fold de `gcd` n-ario | RC-7 | M | bajo | 3 (0) | C3.1 |
| **C3.3** | Narrador de `root_sum` leído del PROPIO resultado | RC-9, RC-8 | M | bajo | 10 (0) | — |
| **C3.4** | Reinyección de la cadena de narradores dentro del envoltorio FTC | RC-10 | S | bajo | 1 (0) | C3.1, **C1.4** |
| **C3.5** | *(opcional)* Tag de método por re-sondeo diagnóstico del backend | RC-8, RC-9 | M | medio | 4 (0) | C3.3 |
| **C4.1** | Pin de DOS NIVELES + inventario de solve mudo | RC-12 | S | bajo | 0 | C0.1 |
| **C4.2** | Rebanada barata de E5: trig periódica + log por sustitución | RC-12 | M | medio | 8 (0) | C4.1 |
| **C5.1** | Contador `spanish_residue_in_en_wire` sobre corpus GUARDRAIL + valla de cobertura | RC-4 | M | bajo | 0 | C0.1 |
| **C5.2** | Rellenar tablas ES→EN y adoptar claves en títulos de sub-paso | RC-4 | M | bajo | 9 (0) | C5.1 |
| **C5.3** | Warnings por el catálogo i18n + UX del grupo Complejo en la web | RC-4, RC-14 | M | medio | 12 (0) | C5.1 |

**Total: 23 ciclos.** Cierre estimado: **≈260 de 546 hallazgos (48 %) y ≈45 de 53 P0 (85 %)**, más
17 P0 (`highlight_wrong_subexpression`) que dejan de publicarse en C1.3 y quedan **cerrados con color
correcto** en C2.1.

**Los dos ejemplos denunciados por el usuario:**
`taylor(sin(x),x,0,5)` paso 3 deja de mentir en **C1.3** (declina al estado entero + zoom local
correcto), queda **bien resaltado** en C2.1 y su nombre de regla se corrige en C2.4.
`integrate(2*x/sqrt(4+x^4)+1,x)` se cierra en **C3.1**, que solo depende de C0.1 — es la palanca de
reordenación si se quiere cerrar ambos testigos pronto (ver §8, riesgo 1).

---

## 6. Ciclo 1 paso a paso

**C0.1 — «Un solo contrato de contadores y el carril de calidad nace con sus tres invariantes
gratis».** Un commit, una entrada de ledger. Media jornada a jornada. Esfuerzo M, riesgo bajo.

### 6.0 Pre-vuelo (obligatorio, 30-40 min)

1. `git status --porcelain` vacío. Rama nueva desde `bdeb56429`.
2. `cargo test --workspace 2>&1 | tail -40` → anotar `failed:0` **antes de tocar nada**. La memoria
   del repo es explícita: *git limpio ≠ tests verdes*. Guardar el log.
3. Huella base: `make engine-scorecard`; copiar `docs/generated/engine_improvement_scorecard.json` a
   `/tmp/scorecard_guardrail_before.json`.
4. Línea base del corpus: `python3 scratchpad/audit/run_corpus.py`. Debe reproducir
   `sum(steps_count)=352`, `sum(len(steps))=292`, `filas discrepantes=24`, `sum(substeps)=214`,
   `sum(solve_steps)=128`.

### 6.1 Parte A — motor (la ÚNICA parte con huella; validarla antes de escribir el test)

1. `crates/cas_solver/src/eval_output_finalize_input/types/shared.rs:37-45` — dejar
   `primary_steps_count()`; añadir `solve_steps_count(&self) -> usize { self.solve_steps.len() }` y
   `substeps_count(&self) -> usize` sumando los sub-pasos de **ambos** canales. Tras el punto 2,
   `combined_steps_count()` queda sin llamantes: **borrarlo** (doctrina de barrido 0-ref en plumbing;
   dejarlo es sembrar la próxima confusión).
2. `crates/cas_solver/src/eval_output_finalize_nonexpr.rs:29` —
   `combined_steps_count()` → `primary_steps_count()`. Es el **único** de los 4 call-sites que usaba
   `combined` (`:42` bool, `:55` text y `eval_output_finalize_expr.rs:50` ya usan `primary`).
3. `crates/cas_api_models/src/wire_types.rs:360` (`EvalOutputWire`) y `:397` (`EvalOutputBuild`) —
   dos campos nuevos `solve_steps_count` y `substeps_count`, propagados en el builder
   `eval_output_finalize/build/output/build.rs:52,96`. **Serializar siempre**, sin
   `skip_serializing_if`: el objetivo del ciclo es que la narración se pueda CONTAR, y un campo
   ausente vuelve a obligar al consumidor a adivinar.
4. **Rider RC-13** — `crates/cas_api_models/src/wire_types.rs:172-181` (`SubStepWire`): añadir
   `before: String` y `after: String`, rellenados desde `SubStep::before_expr` / `after_expr`
   (`cas_didactic/src/didactic/types/substep.rs:8-11`, que **ya** los lleva en texto plano) en
   `cas_didactic/src/step_payloads/build/substeps.rs`. Corrige de paso el diagnóstico del informe:
   `SolveSubStepWire` ya publica `equation` en texto plano; el ciego era el canal primario
   (medido sobre `integrate(sin(x)*x^2,x)`: keys exactamente
   `['title','before_latex','after_latex']` × 8).
5. `crates/cas_api_models/src/wire.rs:275-281` — **obligatorio y fácil de olvidar**: el mensaje
   «N step(s)» se emite solo si `steps_count > 0`; bajo el contrato estricto las 24 filas SolutionSet
   lo **perderían** en la salida de texto y **ningún pin lo vigila** (grep de `step(s)` sobre
   `crates/`: los 3 pins existentes son de derive/expresión). Alimentarlo con `steps + solve_steps`.
6. `crates/cas_wasm/src/lib.rs` — actualizar el doc-comment que enumera los campos del wire que la web
   necesita (disciplina W5: la paridad wasm se pinea POR CAMPOS). La adición es segura (el test wasm
   solo asserta presencia de claves), pero el doc no debe nacer podrido.
7. **Rider RC-15**, 1 línea — `crates/cas_didactic/src/step_payloads.rs:55`:
   `w.rule.starts_with("Conservar")` → las tres constantes `RULE_CONSERVAR_*` de
   `cas_solver_core::rule_names` (la de integral ya está importada en `:10` y usada en `:228`).
   Desactiva por adelantado el acoplamiento latente² entre F5 (traducir ese nombre) y el borrado del
   contrato de honestidad.
8. Validar la parte A **sola**: `cargo test --workspace`; `make engine-scorecard` y diff contra la base.

### 6.2 Parte B — carril (test-only, sin huella)

Nuevo `crates/cas_cli/tests/steps_quality_gate_tests.rs`, calcado de
`crates/cas_cli/tests/steps_divergence_gate_tests.rs`:

- `TERMINATION_NET` (30 s por fila, un disparo es **fallo duro**, nunca skip silencioso),
  `min_expected = 200` filas contra loaders rotos, y `load_web_examples()` (`:319`).
- **Un** eval por fila vía `evaluate_eval_command_in_memory_with_state`
  (`crates/cas_session/src/eval_command/session.rs:123`, devuelve `EvalWireOutput` con
  steps/substeps/solve_steps/warnings públicos), con `EvalStepsMode::On`, `Language::Es` y
  `cli_default_config` — idéntico a lo que sirve la web.
- **Solo cuatro aserciones duras**, las que hoy ya valen cero: `D5_noop_step`
  (`before == after` y la regla no es `RULE_CONSERVAR_*`), `D6_duplicate_consecutive`,
  `D9_unbalanced_braces` (llaves balanceadas sobre todos los campos `*_latex`; una llave
  desbalanceada rompe el render de la fila entera) y `steps_count_mismatch_rows = 0`.
- **Cuatro medidas publicadas** con el formato que el parser espera (`NOMBRE hits=N rows=M`):
  `steps_total` (292), `substeps_total` (263), `solve_steps_total` (128), `E8_substep_noop` (5, con
  techo declarado que baja a 0 en C1.4).
- Dos tests: `steps_quality_canary` **no-ignored** con 10 filas a mano — incluidas
  **obligatoriamente** `taylor(sin(x),x,0,5)` e `integrate(2*x/sqrt(4+x^4)+1,x)` — (~3 s en debug,
  muere si el arnés se rompe) y `steps_quality_corpus_gate` `#[ignore]` con las 210 filas (~11 s en
  release; medido: 6,7 s de evals + overhead).
- **NO** portar `tex2txt.py` (D3/D3b): es un traductor heurístico LaTeX→texto y es la fuente de los
  194 falsos positivos que el informe descartó en bloque.
- **NO** crear todavía `fixtures/steps_quality_inventory.csv`: llega con el primer detector no-cero
  (C1.3). Aterrizar hoy un volcado de ~400 líneas que nadie audita destruye la fuerza del contrato
  «la lista solo encoge».

**Determinismo verificado empíricamente:** dos corridas completas del corpus dan hash idéntico en
210/210 filas tras quitar `timings_us`. El carril no tiene fuente de flakiness.

### 6.3 Parte C — scorecard

- `SuiteSpec` nueva en el dict `SUITES` de `scripts/engine_improvement_scorecard.py:152`, con
  `profile_tags=("guardrail","full")` y
  `command=["cargo","test","--release","-q","-p","cas_cli","--test","steps_quality_gate_tests","--","--ignored","--nocapture"]`.
- `parse_steps_quality` (~15 líneas) junto a `parse_corpus` (`:978`), que captura las líneas
  `NOMBRE hits=N rows=M` y las mete en `docs/generated/engine_improvement_scorecard.json`.
- A partir de aquí, la verificación de **todos** los ciclos siguientes es el paso 6 del protocolo de
  la skill sin inventar nada: `make engine-scorecard` + diff de contadores.

### 6.4 Tests que caen y qué hacer con ellos

| Test | Predicción | Acción |
|---|---|---|
| Los **163 pares (test, expresión)** que asertan `steps_count` (`semantics_cli_contract_tests.rs` 159, `wire_smoke_tests.rs` 8, `cli_contract_tests.rs` 2, `limit_contract_tests.rs` 1, `cas_session/src/eval_command_tests.rs` 2) | **0 caídas** (medido: todos son derive/diff/limit/integrate, ruta de expresión que ya usaba `primary_steps_count()`) | Ninguna. Si alguno cae, es hallazgo del ciclo y se documenta. |
| `cli_contract_tests.rs::solve_system_educational_s3_contract` (`"steps_count": 0`) | **Sigue verde y se REFUERZA** | Añadir comentario: el estricto es lo que garantiza «cero fugas de micro-pasos internos al canal steps». |
| `wire_smoke_tests.rs::test_wire_steps_summary_when_enabled` | No rompe (es auto-consistente), pero es el **único guardián** del mensaje de wire, cuyo alimentador cambia | **AMPLIAR** con una fila solve que hoy perdería la línea. |
| `cli_contract_tests.rs:85-112` (`integrate(exp(x^2),x)`, `steps_count 1`, regla «Conservar integral residual») | Verde | Es exactamente el pin que **valida** el rider RC-15. |

### 6.5 Criterio de aceptación (todo o se revierte)

- `cargo test --workspace` failed:0; `cargo clippy --workspace --all-targets` 0 warnings (**gatear con
  `&&`, nunca con `; echo OK`**: la memoria del repo registra ese falso verde).
- `cargo check -p cas_wasm --target wasm32-unknown-unknown` (nightly) verde.
- `make engine-scorecard` → **0 deltas de contadores** contra `/tmp/scorecard_guardrail_before.json`.
  Predicción **fuerte y falsable**: `scripts/engine_diff_command_matrix_smoke.py:1647` ya calcula el
  contador como `len(parsed["steps"])`, o sea que el arnés de guardrail **ya** implementa el contrato
  estricto y era el motor el que discrepaba. Si algo se mueve, hay una fila guardrail SolutionSet no
  inventariada y **ese es el hallazgo del ciclo** (se documenta, no se enmascara).
- `run_corpus.py` re-corrido: `sum(steps_count)` 352 → **292** == `sum(len(steps))`; filas
  discrepantes 24 → **0**; `solve_steps_count` 128; `substeps_count` 263; los 214 sub-pasos con
  `before`/`after` en texto plano.

### 6.6 Ledger y commit

Entrada nueva al final de `docs/ENGINE_COMBINATION_LEDGER.md` con el molde vigente
(`area` / `status` / `capture.investment_class` / `observed` / `decision` / `retained learning`).

*Retained learning* candidato:

> **Un contador que mezcla dos canales no es un defecto cosmético: es el instrumento con el que se
> declaró completa la campaña anterior.** El frente E cerró contra métricas que no veían los 214
> sub-pasos ni las 16 filas con `solve_steps`; su residual declarado A=31 coincide EXACTAMENTE con el
> D4 de hoy (31 hits) — nada se movió y nadie lo notó durante meses. El primer ciclo de una campaña
> de calidad arregla la REGLA de medir, y el arnés nace con los invariantes que ya valen cero para
> que su primer día sea verde y su crecimiento sea **por tandas de detector, nunca por volcado**.

**Lo que este ciclo NO toca (declararlo en el ledger):** ni una línea de highlights, sub-pasos o
narración. Si al portar los detectores tienta añadir D1/D2, se aparca: llegan en F1 y F2 **con el
arreglo que los baja en el mismo ciclo**, no antes.

---

## 7. El arnés permanente

Dos carriles: `crates/cas_cli/tests/steps_quality_gate_tests.rs` (nuevo, calidad de steps) y el
módulo de veracidad de resaltado dentro de `cas_didactic` (C1.3). Ambos con el molde de
`steps_divergence_gate_tests.rs`: twin in-memory sin spawn, `min_expected` contra loaders rotos, red
de terminación como fallo duro, y **cuarentena autoinvalidante**.

**Regla de aterrizaje:** cada contador entra en el ciclo que lo lleva a su objetivo, jamás antes.
El eje que **falla por diseño** no se registra en el scorecard (un suite siempre-rojo ensucia la
huella); solo se registra el que **mide**.

**Regla del fixture:** el inventario se ancla **por expresión** en
`crates/cas_cli/tests/fixtures/steps_quality_inventory.csv` (columnas `expression,detector,hits`,
cargado con `include_str!`) — nunca por índice de fila (el csv se reordena) ni por índice de paso
(el dedup renumera). El carril falla **tanto si aparece una entrada nueva como si una entrada deja de
disparar** (STALE): eso impide el trueque silencioso «arreglo una fila, rompo otra». **La lista solo
encoge.**

### 7.1 Contrato CERO (aserción dura, sin techo)

| Contador | Hoy | Entra en | Nota |
|---|---|---|---|
| `D5_noop_step` | 0 | C0.1 | valla gratis |
| `D6_duplicate_consecutive` | 0 | C0.1 | valla gratis |
| `D9_unbalanced_braces` | 0 | C0.1 | well-formedness de MathJax |
| `steps_count_mismatch_rows` | 24→0 | C0.1 | lo arregla el propio ciclo |
| `substep_raw_display_in_latex` | 37→0 | C1.1 | 14 filas |
| `D1_red_equals_green` | 6→0 | C1.3 | subcaso trivial del guard |
| `D1b_hl_identical_but_state_changed` | 5→0 | C1.3 | |
| `hl_published_unverified` | —→0 | C1.3 | excepción multi-span declarada hasta C2.2 |
| `E8_substep_noop` | 5→0 | C1.4 | invariante, no cobertura |
| `substep_checked_failures` | —→0 | C1.8 | 0 fallos no cuarentenados |

### 7.2 Techo declarado que SOLO puede bajar (con inventario anclado)

| Contador | Hoy | Objetivo | Dueño |
|---|---|---|---|
| `D2_hl_substitution_mismatch` | 39 | ≤12 (C1.3) → ≤2 (C2.1) | RC-3 / RC-1 |
| `D2b_hl_arity_mismatch` | 19 | ≤5 | C2.1 |
| `hl_declined` | 142 (tras C1.3) | ≤63 | C2.1, C2.3 |
| `mute_integration_steps` | 14 | 2 | F3 |
| `E1_container_sin_substeps` | 36 | ≤26 | F3 |
| `solve_mute_rows` | 14 | 11 (F4) | F4 / frente E5 |
| `substep_unchecked_emitters` | 126 | monótono decreciente | migración incremental |
| `D4_chain_discontinuity` | 31 | — (baja al abrir granularidad) | **nace calibrado**: coincide exactamente con el residual A=31 del frente E |
| `D8_raw_exponent_artifact` | 20 | — | |
| `D12_zero_steps_but_changed` | 27 | — | 4 falsos positivos `:=` **documentados en el fixture, no borrados** |
| `E2_substep_chain_break` | 50 | — | |
| `D7_rule_burst` | 2 | — | |

### 7.3 Inventario SIN techo (`--ignored`, FALLA POR DISEÑO)

Molde literal de `input_associativity_pairs_inventory` (`steps_divergence_gate_tests.rs:537`): el rojo
manual **es** el inventario vivo — se regenera solo y no envejece como un doc.

| Eje | Hoy | Por qué no tiene techo |
|---|---|---|
| `D3b_text_vs_latex_order_only` | 194 | Clase entera de RC-6; su residual estructural no se cierra |
| `hl_render_parity_structural` | 118 | Cerrarlas exige desmontar el aplanado de `Sub`, que es lo que da granularidad al color |
| `E3/E3b_substep_vs_parent` | 103 / 88 | Unificación local-vs-global (el E2 que el ledger aplazó con conocimiento de causa) |
| `spanish_residue_in_en_wire` (identity_pairs) | 134 (55 cadenas) | Clase abierta hasta F5 |
| `solve_narration_level2` | 14 filas | Exige ≥1 paso que no sea re-enunciado ni respuesta |
| `unmapped_rule_name_description_pairs` | por medir | C2.4 |

### 7.4 Medida descriptiva, JAMÁS aserción

`D11_single_step` (85 filas) — el informe demostró que **49 de esas 85 narran bien dentro de
`substeps`** y que tratarlo como defecto es lo que produjo la falsa alarma de regresión anterior.
Igual `steps_total` / `substeps_total` / `solve_steps_total` (292 / 214 / 128): son la fotografía
honesta de la narración, no un umbral.

---

## 8. Lo que NO se va a arreglar (y por qué)

1. **RC-1 opción (a), transportar el path** — el arreglo nº1 del informe. Refutado por medida
   (§2a). Sería un mes en `cas_ast` para un techo peor que el de la resolución por contenido.
2. **Paridad total de renderizadores** (`to_latex(e) == PathHighlightedLatexRenderer(config vacía)`).
   De las 418 divergencias, **118 son estructurales**: el renderizador con paths distribuye
   `A − (B + C − D)` en `A − B − C + D` mientras el plano conserva el paréntesis
   (`format_sub:445-450`), y ese aplanado es exactamente lo que da granularidad al color por término.
   Se cierran las 300 de permutación; las 118 se publican con número.
3. **`IntegrationTrace` completo (RC-8, `Option<ExprId>` → `Option<IntegrationTrace>`)**. XL y con
   riesgo alto: ~216 rutas de retorno, `IntegrationTraceKind` consumido en 3 sitios de producción con
   semánticas distintas (supresión de condiciones en `integration_result_pipeline.rs:49`, envoltura en
   `Hold` en `:73`, gate de presentación en `integral_derivative_shortcut_presentation.rs:265`) y 10
   pins de contrato. Y **el render ni siquiera está cableado**: `collect_step_payload_substeps`
   (`step_payload_render/substeps.rs:9-14`) ignora `step.substeps()`, así que se rellenaría una traza
   que no se publica. C3.3 y C3.5 capturan el valor narrativo sin tocar **una sola firma** del motor.
4. **El «piso de narración» de solve como respuesta didáctica.** Su tercer elemento **no existe**:
   `EvalOutputView.strategy` está hardcodeado a `None`
   (`solve_command_eval_core/eval/output.rs:69`) y su único productor en todo el repo es
   `derive_command.rs:437`; el backend devuelve `(SolutionSet, Vec<SolveStep>)` sin etiqueta y
   `solve_local_core_inner` es una cascada de ~56 `try_*` anónimos. Lo construible es «re-enunciar el
   problema + dar la respuesta», y las condiciones de dominio **ya** se publican vía
   `required_display` (medido: `x ≠ 2`, `x ≠ 0`, `x ≥ 0`, `x > 0` en las filas 79/80/83/88). Convierte
   25 `missing_narration` en 25 `magic_step`: cierre honesto neto ≈ 0. Solo entraría como red del
   nivel 1 del pin si F4 se atasca.
5. **E5 completo (narración de inecuaciones, 9 familias restantes).** Medido: 39 call-sites devuelven
   `(set, Vec::new())` frente a 17 que propagan, y **21 de los 39 mudos son manejadores de
   inequality** — la narración no se pierde, **no existe**. La narrativa (raíces y polos → tabla de
   signos → ensamblado del intervalo) no está escrita en ningún punto del repo; el lado `=` solo
   regala tipos (`Equation` lleva `op`, el wire publica `relop`) y la tabla i18n de 145 plantillas:
   fontanería a coste cero, contenido a coste completo. Es un frente propio, un ciclo por familia.
   Y **no** se meten las filas 85/86/90 (abs-ineq) en la rebanada barata: la memoria del repo
   (commit `04e7cda0f`) ya registra que el detector barato está **AGOTADO** ahí (cero disparos en los
   3 `let (sol, _)` instrumentados) y que el dueño real exige mapear la estrategia piecewise desde el
   entrypoint.
6. **Los mega-sub-pasos de verbos vectoriales/analíticos (P0-S3-6, 14 hallazgos).** No es un bug con
   arreglo local sino una decisión de diseño declarada en los doc-comments de
   `focused_rule_substeps.rs:11996` y `:12040`; sustituirla por narración por maniobra son 5 verbos ×
   ~4 sub-pasos + locale es/en. Tanda propia. Lo único que sí reciben ya es que **dejan de mentir en
   el formato** (C1.1 les quita el fallback declarado LaTeX).
7. **RC-15 y RC-16 como ciclo propio.** RC-16 afecta a 0 filas. RC-15 exige **dos** condiciones
   independientes y hoy no se da ninguna (medido: `integrate(e^(x^2),x)` sale «Conservar integral
   residual» en ES **y** en EN — el nombre nunca se traduce): latente al cuadrado. Vale el rider de
   1 línea de C0.1; el arreglo sustantivo comparte chokepoint con la política de fold
   (`step_payloads/build/expr.rs:49-93`) que C2.1 tiene que abrir de todas formas.
8. **Portar D3 / `tex2txt.py` al carril.** Traductor heurístico LaTeX→texto, fuente de los 194 falsos
   positivos. Su clase se mide con inventario, no con un detector ruidoso.
9. **`D11_single_step` como defecto.** Ver §7.4.
10. **Los 63/480 pasos sin span único veraz.** DECLINAN para siempre (DEFAULT 20, LOCAL-meta 12,
    LOCAL-shape 31). El techo del enfoque de span único es 417/480; el 13 % restante exigiría un
    modelo de resaltado multi-span general que este plan no propone. **No es deuda: es el resultado
    correcto.**
11. **Las dos familias sin narrador:** fila 021 (gaussiana definida sobre ℝ, «√π aparece de la
    nada») y fila 025 (by-parts cíclica `∫e^x·sin x`). Ninguna cae en RC-7 — la 025 es un
    **producto**, no una suma. Escribir esos narradores es capacidad nueva, no remediación.
12. **Revertir el aparcado del listener de eventos de solve/dsolve** (`83481111e` / `797a93c21` /
    `ce968c793`). El confeti era real y su vuelta reabriría `chain_discontinuity` 506.
13. **Tocar un solo resultado matemático.** Esta campaña cambia campos `latex`, títulos, contadores y
    decisiones de publicar/declinar. `as_rational_const` y la capa numérica quedan intactas (R1 de la
    memoria).

**Superficie declarada NO cubierta, para que nadie dé la clase por cerrada:** los 7 sub-pasos que
cuelgan de `solve_steps` usan otro tipo (`SolveSubStepWire`) y otra ruta
(`cas_solver/src/eval_output_presentation_solve_steps.rs:40-46`). Ni el chokepoint de la tubería ni
`SubStep::checked` los tocan. Es una segunda instalación pendiente, no un olvido.

**Tres hallazgos colaterales que NO están en los 546 y que van al backlog:**
(1) `integrate(1/(x^5-1), x)` devuelve `steps[0].after` en texto plano **corrupto**
(`fracln(|x - 1|)5 + ln(x · (sqrtfrac54 + frac12) + ...`, nombres de macro LaTeX filtrados al
renderizador de texto) mientras su `after_latex` y su `result` están bien;
(2) `approx(subs(asinh(x^2/2) + x - (...), x, 1.23456789))` **no termina en 25 s** — cuelgue
reproducible en `approx` sobre `asinh`;
(3) `equiv(i^2,-1)` y `equiv(sin(i), i*sinh(1))` devuelven `false` mudo igual que la identidad de
Euler (los cubre C1.7).

---

## 9. Riesgos y señales tempranas de que el plan va mal

### R1 — El guard QUITA color el día que aterriza (el más visible)
Al aterrizar C1.3, **142 de 480 pasos de display** (~27 de 292 en el wire) dejan de publicar span
parcial y pasan a estado-entero rojo/verde + zoom local. Es un intercambio deliberado de riqueza por
honestidad, pero el usuario **lo verá**.

*Mitigaciones:* (a) **avisar antes de aterrizar el ciclo**, con el número exacto por ruta;
(b) el fallback está verificado bueno — inspeccionados los 42 pasos que hoy ya caen en
`render_normalized_rule_latex` (`build/latex.rs:221`) y las 10 muestras revisadas son identidades
locales correctas (`{x}^{0}→1`, `ln(sqrt(e))→1/2`, `3/2·ln(x)−3/2·ln(x)→0`, `e^x+1/e^x→2·cosh(x)`);
(c) **nunca** dejar el paso sin información: el estado entero SÍ cambió y eso es cierto;
(d) C2.1 recupera ~79 de esos 142 en el ciclo siguiente — planificar C1.3 y C2.1 como una sola
entrega hacia el usuario aunque sean dos commits; (e) **palanca de reordenación disponible**: C3.1
solo depende de C0.1 y cierra el segundo testigo del usuario; si se quiere una entrega visiblemente
positiva antes del guard, se adelanta C3.1 a la posición 2.

### R2 — Los 12 pins de C1.3 (el mayor riesgo de calendario)
8 en `crates/cas_cli/tests/semantics_cli_contract_tests.rs`, 3 en
`crates/cas_didactic/tests/step_wire_tests.rs`, 1 en
`crates/cas_didactic/tests/timeline_render_test.rs`. Cada rojo exige decidir **a mano** si el
resaltado que fija es cierto o falso. Es diagnóstico, no tecleo.

*Doctrina obligatoria:* si el pin exigía un span que el guard declina y ese span **es falso**, es un
pin de defecto codificado y se **REESCRIBE al contrato nuevo** en el mismo commit. Si el span **es
cierto** y el guard lo declina, **el bug está en el guard**, no en el pin — esa es la señal de
diagnóstico. Bajo ninguna circunstancia se afloja un assert para poner verde.

Dos pins ya identificados como **defecto codificado**:
`integrate_contract_quadratic_exp_by_parts_exposes_didactic_substep`
(`integrate_contract_tests.rs:3054`, assertea literalmente
`closer["after_latex"] == "e^{x}\\cdot ({x}^{2} + 2 - 2\\cdot x)"` con el comentario *«closer should
land on the final antiderivative»* — eso **es** el P0 de C1.6) y
`render_substep_side_preserves_latexish_fallbacks` (`substeps.rs:78`, pide que un fallback con
`\frac` se pase crudo — eso **es** el P0 de C1.1). Cada ciclo que tumba un pin deja en el ledger la
frase «**contrato nuevo: …**» y el pin reescrito en el mismo commit.

### R3 — Regresionar la superficie SANA
El drift **solo** afecta al wire: la timeline (CLI/HTML) usa snapshots RAW
(`timeline/simplify_highlights/snapshots.rs:5`, sin `normalize`) donde el path grabado **es válido**.
Un cambio de resolución de foco que no corra el guard en **ambas** superficies puede empeorar la que
hoy funciona. El chokepoint único lo hace factible, pero exige hilar `&mut Context` por la ruta
timeline (`render_timeline_step_math` / `math.rs:6`) y **compartir un solo clon** con el cálculo de
snapshots — no añadir una clase de coste nueva (el wire ya clona el Context 4-5 veces por paso).

### R4 — Medir el corpus y llamar a eso la clase
Es el error que la propia auditoría cometió (§2d). *Regla de la campaña:* todo contador nuevo corre
sobre los corpus **guardrail** (`web_examples`, `identity_pairs`, `derive_pairs`,
`substitution_identities`, `equation_corpus`, `limits`), no solo sobre la vitrina. Ningún «X está
cerrado» vale si solo se comprobó en `examples.csv`.

### R5 — El arnés aterriza rojo
Con los 11 detectores y ~400 hits de golpe, el primer commit es un fixture inauditable y «la lista
solo encoge» pierde toda su fuerza. *Mitigación ya adoptada:* tandas de detector, cada una montada
sobre el ciclo que lleva su contador a cero. Ciclo de vida del ledger: **descubrir → medir en
`--ignored` → cerrar → mover al tier default y vigilar**.

### R6 — Huella
Ninguna de F0, F1, F2 y F5 toca matemática: los 18 contadores guardrail deben quedar **idénticos**, y
cualquier delta es **motivo de rechazo de la iteración**, no de investigación diferida. El único ítem
con riesgo real de huella de tiempo es C3.5 (6-27 ms de re-sondeo contra vigilancias 150 ms /
P95 75 ms) — por eso es opcional y va al final.

### R7 — Churn doble sobre el fold
La política de pliegue de `step_payloads/build/expr.rs:49-93`
(`cleanup_symbolic_diff_after_for_display` + `normalize_expr_for_display`) es la **misma** que tocan
RC-15, RC-16 y el drift de RC-1. Quien la abra, cierra los tres de una pasada. Declarado: el rider de
C0.1 **no** abre el fold; el fold solo se toca en **C2.1**.

### R8 — Que un piso evapore la presión
Por eso el pin de solve es de **dos niveles** y el nivel 2 (`--ignored`, falla por diseño) exige ≥1
paso que no sea ni el re-enunciado ni la respuesta.

### Señales tempranas de que el plan va mal (revisar en cada ciclo)

| Señal | Significado | Acción |
|---|---|---|
| C0.1 mueve **cualquier** contador guardrail | Hay una fila guardrail SolutionSet no inventariada | Parar, inventariarla, documentarla; no commitear hasta entenderla |
| Tras C1.3, `hl_declined` > 180/480 | El guard está mal escrito (o mal ubicado), no «es agresivo» | Desglosar por **ruta** y por `rule_name`: si una regla concreta declina en masa, es su foco lo que está mal |
| Un pin de C1.3 cae y su resaltado **sí** verifica | Bug del guard | Parar el ciclo y arreglar el guard antes de seguir |
| Tras C2.1, guard-pass < 400/480 | El foco por contenido no rinde lo medido | Re-scopear C2.1 antes de C2.2/C2.3; el techo era 417 |
| El fixture de inventario **crece** en cualquier ciclo | Trueque silencioso «arreglo una fila, rompo otra» | Rechazar la iteración |
| Una entrada del fixture deja de disparar sin que el ciclo lo declare | Entrada STALE / detector roto | Rechazar la iteración |
| C1.3 consume > 2× su presupuesto en triage de pins | El chokepoint estaba más acoplado de lo medido | Parar, reportar y decidir si se parte en dos commits |
| Al final de F1, re-corrida del corpus da > 10 P0 publicados | El marco «verdad primero» no está rindiendo | Revisar el reparto de F1 antes de entrar en F2 |
| Un ciclo de presentación mueve un contador **matemático** | Se tocó algo que no se creía tocar | Rechazar la iteración e investigar |

---

## 10. Apéndice — línea base de contadores (para el diff)

**Corpus** (210 filas de `web/examples.csv`, `--steps on --lang es --format json`):
292 steps · 214 substeps · 128 solve_steps · `steps_count` publicado **352** · 24 filas discrepantes.

**Hallazgos (546).** Por severidad: P0 53 · P1 218 · P2 215 · P3 60.
P0 por categoría: `highlight_wrong_subexpression` 17 · `substep_wrong_math` 8 · `wrong_rule_name` 7 ·
`latex_render_bug` 7 · `highlight_red_equals_green` 5 · `anti_pedagogical` 4 · `other` 3 ·
`text_latex_divergence` 1 · `substep_noop_or_false_claim` 1.
Top categorías: `magic_step` 112 · `latex_render_bug` 61 · `anti_pedagogical` 57 ·
`missing_narration` 38 · `chain_discontinuity` 36 · `other` 36 · `noop_or_trivial_step` 25 ·
`highlight_wrong_subexpression` 25 · `wrong_rule_name` 24 · `text_latex_divergence` 23 ·
`rule_name_misleading` 20 · `substep_noop_or_false_claim` 16 · `substep_duplicates_parent` 14 ·
`duplicate_or_burst` 13 · `substep_wrong_math` 12 · `substep_chain_break` 8 ·
`highlight_stale_form` 7 · `highlight_red_equals_green` 7 · `language_leak` 6 · `lang_parity` 3.

**Resaltado** (480 pasos de display): guard-pass **338/480** (70,4 %) · declives 142 · techo con span
único **417/480** (86,9 %) · sin span veraz 63 (DEFAULT 20 / LOCAL-meta 12 / LOCAL-shape 31) · drift
119/480 (25 %) · ruta DEFAULT-POSITIONAL 78 pasos con guard-pass 28/78 · focos ambiguos 25 ·
multi-span 9/292 del wire (6 filas) · paridad de renderizadores 542/960 iguales, 418 distintos
(300 permutación + 118 estructurales).

**Sub-pasos** (214): prototipo del guard tipado → 45 OK · 80 MISMATCH (de los cuales ~51 son
relaciones legítimas no-igualdad) · 52 ABSTAIN · 36 NOEVAL · 1 TIMEOUT. Campos `*_latex` con texto de
display: 46 en 17 filas, de los cuales **37 crudos** en 14 filas (54, 147, 149-153, 176, 180, 183,
185, 193, 194, 195); patrón `^(` en 11 filas. `E8_substep_noop` = 5. Emisores: 219 en
`focused_rule_substeps.rs` (115 con `latex_expr`, 30 con `format!`, 20 sin latex);
159 `SubStep::new` vs 107 `SubStep::keyed`.

**Integrales:** `mute_integration_steps` = 14 (filas 021, 025, 028, 034, 035, 036, 039, 136, 137,
156, 186, 187, 188, 191). Filas 021-042: 8 método real / 2 polo / 5 cáscara FTC / 7 mudas.

**Solve:** mudas reales = 14 (68, 70, 79, 80, 81, 83, 84, 85, 86, 88, 89, 90, 189, 190).
Por clase: ecuaciones 2/21 (10 %) · **inecuaciones 12/16 (75 %)** · sistemas 0/9 · dsolve 1/9
(y esa una declina honesto con warning). 39 call-sites `(set, Vec::new())` vs 17 que propagan;
21 de los 39 mudos son manejadores de inequality.

**Idioma:** `spanish_residue_in_en_wire` = 29 sobre `examples.csv` (18 cadenas, 3 canales:
5 nombres de regla + 4 títulos de sub-paso + 9 warnings) y **134 sobre 300 filas de
`identity_pairs.csv`** (79 filas, 55 cadenas). `english_residue_in_es_wire` = 18 (4 cadenas).
77 de 84 títulos literales de `SubStep::new(...)` no están en `description_en`.

**Detectores:** D1 6 · D1b 5 · D2 39 · D2b 19 · D3b 194 · D4 31 · D5 0 · D6 0 · D7 2 · D8 20 ·
D9 0 · D11 85 · D12 27 · E1 36 · E2 50 · E3/E3b 103/88 · E8 5.