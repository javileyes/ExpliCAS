# Ledger de saneamiento

Sospechosos observados **mientras** se reestructura (protocolo de
`AUDITORIA_PARETO_ARQUITECTURA_2026-07-31.md`, §6): se anotan sin actuar, para
no abrir madrigueras a mitad de un movimiento. Las tandas de saneamiento
posteriores se alimentan de aquí, no de la memoria.

Estados: `abierto` · `en curso` · `cerrado` · `descartado`.

---

## Abiertos

### L1 — El repo no está limpio de `rustfmt` (preexistente)
- **Origen:** P5, troceo de `semantics_cli_contract_tests` (2026-07-31).
- **Qué:** dos hunks que `rustfmt --check` rechaza desde antes del troceo, en
  el test `bignum_available_mirrors_materialization_gates` y un vecino. Tras el
  troceo viven en `semantics_cli_contract_tests/misc_numeric.rs` y
  `misc_core.rs`, con el mismo contenido (verificado comparando el diff de
  rustfmt antes y después).
- **Por qué importa:** mientras `cargo fmt --check` esté rojo de base, no puede
  usarse como gate: cualquier suciedad nueva se camufla entre la vieja.
- **Acción propuesta:** un commit de formato aislado (`cargo fmt`) sobre esos
  hunks; después, `cargo fmt --check` sirve ya como gate binario.
- **Riesgo:** nulo (solo espacio en blanco), pero es cirugía → commit propio.

### L2 — Helpers de test duplicados por todo el workspace
- **Origen:** P5, barrido de `crates/*/tests` (2026-07-31).
- **Qué:** el mismo helper reescrito en muchos ficheros de test:
  `solve_display` ×22, `simplify_str` ×19, `solve` ×16, `label` ×9,
  `create_full_simplifier` ×9, `parse_wire` ×5, `run_cli` ×3.
- **Por qué importa:** es el mismo patrón de deriva silenciosa que
  `collect_add_terms` ×13 y `unary_builtin_arg` ×14 en `src`. Un arreglo en una
  copia no llega a las otras, y en helpers de test eso se traduce en suites que
  creen estar comprobando lo mismo y no lo comprueban.
- **Acción propuesta:** **primero diffear** las implementaciones (§4 del
  protocolo). Las idénticas → crate `cas_test_support` como `dev-dependency`.
  Las divergentes → cada divergencia es un hallazgo propio antes de fusionar.
- **Riesgo:** medio. No fusionar a ciegas.

### L3 — `metamorphic_simplification_tests.rs` es infraestructura, no casos
- **Origen:** P5 (2026-07-31), decidido NO trocear.
- **Qué:** 18.923 líneas con **139 tests y 316 helpers**. La proporción está
  invertida respecto a los otros monolitos: el contenido real es la maquinaria
  (carga de CSV, chequeo numérico, chequeo estructural, clasificadores).
- **Por qué importa:** el troceo mecánico por bloques `#[test]` aquí no aporta
  —dejaría un `main.rs` de 18k líneas—. Necesita mover maquinaria a módulos,
  que es cirugía.
- **Acción propuesta:** pasada propia que extraiga la infraestructura por
  responsabilidad, con tests verdes entre cada extracción.

### L4 — Rutas `--exact` obsoletas en la documentación
- **Origen:** P5 (2026-07-31).
- **Qué:** al pasar los tests a submódulos, sus rutas quedan cualificadas
  (`solving::test_eval_...`). Los repro con `--exact <nombre_desnudo>` de
  `SLOW_CI_TEST_LEDGER.md` ya no casan. Los `--test <binario>` sí siguen
  válidos: la convención `tests/<nombre>/main.rs` conserva el nombre del target.
- **Acción propuesta:** barrer los `--exact` de `docs/` y prefijarlos, o
  quitarles el `--exact` (el filtro por subcadena sigue funcionando).
- **Riesgo:** nulo, es documentación.

### L5 — La atribución de tiempo de CI puede estar inflada
- **Origen:** P5 (2026-07-31), medición incidental.
- **Qué:** `.claude/skills/auto-mejora/SKILL.md` da `cli_contract_tests` = 267 s
  y sostiene que cuatro suites se llevan 12 de los ~19 min. Medido aislado hoy:
  **57 s**. La cifra de 267 s salió de una corrida `--workspace` completa, es
  decir con contención de CPU entre binarios.
- **Por qué importa:** es exactamente el patrón ya registrado en memoria de
  «atribución heredada FALSA: medir, no heredar». Si el reparto real del tiempo
  es otro, el plan de acelerar CI está optimizando el objetivo equivocado.
- **Acción propuesta:** re-medir por binario en aislamiento antes de invertir en
  acelerar «las cuatro suites lentas».

### L6 — El grafo de llamadas del orquestador es una bola sin costuras
- **Origen:** P1, medición previa al troceo (2026-07-31).
- **Qué:** las 692 fns de producción de `orchestrator.rs` tienen 1.795 aristas
  de llamada internas. Una agrupación guiada por el grafo (propagación de
  etiquetas) alcanza 98,8% de cohesión pero metiendo **627 de las 692 en un
  único grupo**: no hay clústeres naturales que seguir. La partición por
  familia de nombre deja el 34,1% de las llamadas dentro del módulo, frente al
  11,1% de una partición seudoaleatoria.
- **Por qué importa:** fija la expectativa para P2–P4. Trocear estos ficheros
  compra navegabilidad, tamaño de fichero y menos conflictos — **no compra
  desacoplamiento**. Quien prometa lo segundo con un movimiento de módulos se
  equivoca; desacoplar aquí exige rediseño (introducir interfaces, invertir
  dependencias), que es trabajo de diseño y no de mudanza.
- **Dato útil que sí salió:** existen 41 primitivas compartidas (llamadas
  desde 4 o más familias) que ahora viven en `orchestrator/support.rs`. Esa
  caja de herramientas sí es una capa real, no una etiqueta.
- **Hallazgo negativo, para no repetir la búsqueda:** el compilador no emite
  **ni un** `dead_code` tras estrechar la visibilidad (548 `pub(super)`, 144
  privadas). El orquestador no tiene funciones huérfanas de nivel superior: es
  sedimento *vivo*, no muerto.

### L7 — Nombres de módulo autogenerados que ensombrecen crates de std
- **Origen:** P1 (2026-07-31), pisado y corregido en el momento.
- **Qué:** el bucket residual del troceo se llamaba `core`, y `mod core` +
  `use core::*` ensombrece el crate `core` de Rust. Compilaba y los tests
  pasaban, pero dejaba la trampa de que un futuro `core::mem::…` escrito en ese
  módulo resolviera al módulo local. Renombrado a `general`.
- **Acción para P2–P4:** al generar nombres de módulo, comprobarlos contra
  `core`, `std`, `alloc`, `test`, `proc_macro` y los nombres de crates del
  workspace antes de emitir.

### L8 — `rules/arithmetic.rs` no es un archivador: es un motor de cancelación
- **Origen:** P2, medición previa al troceo (2026-07-31).
- **Qué:** la auditoría proponía repartir sus 25 reglas por la taxonomía
  existente de `rules/`. La medición del cierre transitivo de helpers dice que
  no se puede: de los 151 helpers que necesita la trigonometría solo **15 son
  exclusivos suyos** (los otros 136 los usan también las reglas aritméticas);
  logaritmos, 3 exclusivos de 34; álgebra, 0 de 3. El token dominante de los
  713 helpers es `cancellation` (134), por delante de `zero` (126) y `trig`
  (117).
- **Por qué importa:** `ExpandTrigSumToProductToEnableCancellationRule` no es
  una regla de trigonometría mal archivada — es una regla de **cancelación** con
  disparador trigonométrico. Dispersar las reglas habría duplicado maquinaria o
  tejido una telaraña entre directorios peor que el fichero de partida. La
  recomendación original de P2 en la auditoría queda **superada por la
  medición**.
- **Qué queda vivo de aquella idea:** si algún día se quiere de verdad mover
  familias a la taxonomía, primero hay que separar el motor de cancelación de
  sus disparadores. Eso es rediseño, no mudanza.

### L9 — Los tests con rutas `super::` explícitas no son movibles sin reescritura
- **Origen:** P2 (2026-07-31), pisado y resuelto.
- **Qué:** los 398 tests de `arithmetic` llaman a los helpers con rutas
  explícitas `super::foo` (325 ocurrencias). Al bajarlos un nivel de
  anidamiento, `super` deja de apuntar al módulo padre y todo revienta con
  E0425. Los del orquestador no tenían el problema porque usan `use super::*` y
  llamadas sin cualificar.
- **Acción aplicada:** reescribir `super::` → `super::super::` en los bloques
  movidos (misma referencia relativa, un nivel más abajo). Verificado
  deshaciendo la reescritura al comparar contra HEAD.
- **Para P3/P4:** comprobar `grep -c 'super::'` en el bloque a mover ANTES de
  moverlo; si sale distinto de cero, la reescritura es obligatoria y hay que
  declararla (deja de ser movimiento puro).

### L10 — `cargo build --workspace` verde NO implica que compilen los tests
- **Origen:** P2 (2026-07-31), fallo real en el gate final.
- **Qué:** tras repartir los helpers de `arithmetic`, `cargo build --workspace`
  salió limpio y `cargo test -p cas_engine --lib` pasó. El gate final
  (`cargo test --workspace`) reventó con E0603: `register` era `pub`, la llama
  otro crate, y el reexport `pub(crate) use general::*` la había estrechado.
  Solo la llamaba código de **test** (`cas_cli/tests/advanced_simplification.rs`),
  y ningún `build` compila eso.
- **Por qué importa:** es la misma familia de falso verde que `| tail` y
  `; echo OK`, pero más sutil porque el comando en sí es correcto: lo engañoso
  es el ALCANCE. Un `build` limpio no dice nada sobre los objetivos de test.
- **Regla para P3/P4:** tras cualquier cambio de visibilidad, verificar con
  `cargo test --workspace --no-run` (compila todo lo que el gate ejecutará)
  antes de dar el paso por bueno; no basta con `build` ni con los tests del
  crate tocado.

---

## Cerrados

_(ninguno todavía)_

---

## Nota de método

Un aviso que costó un rato en P5: verificar un troceo comparando ficheros con
`f.endswith("main.rs")` **se salta `misc_domain.rs` en silencio** —
`"misc_domain.rs".endswith("main.rs")` es `True`—. El verificador daba 500 de
505 tests y parecía un fallo del troceo. Comparar el nombre completo
(`Path(f).name == "main.rs"`), no el sufijo. Es la misma familia de error que
los falsos verdes de `| tail` y `; echo OK` ya registrados: **cuando una
verificación acusa al código, sospecha primero de la verificación.**
