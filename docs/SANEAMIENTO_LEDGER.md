# Ledger de saneamiento

Sospechosos observados **mientras** se reestructura (protocolo de
`AUDITORIA_PARETO_ARQUITECTURA_2026-07-31.md`, §6): se anotan sin actuar, para
no abrir madrigueras a mitad de un movimiento. Las tandas de saneamiento
posteriores se alimentan de aquí, no de la memoria.

Estados: `abierto` · `en curso` · `cerrado` · `descartado`.

---

## Abiertos

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

### L12 — La superficie pública de cas_math es ancha pero REAL
- **Origen:** P3 (2026-07-31), medición que corrige la propia auditoría.
- **Qué:** la auditoría señalaba las 91 `pub fn` de
  `symbolic_integration_support` como superficie inflada que impedía refactorizar
  con seguridad local. Medido: **88 de las 91 se usan fuera de cas_math**, y las
  8 de `limits_support`, las 8. No hay ninguna candidata a `pub(crate)`.
- **Por qué importa:** la parte «API curada» de P3 no tiene recorrido — no hay
  superficie que recortar. Lo que sí queda es lo que se hizo: trocear para
  navegar. Y cuidado con la conclusión inversa: que la API se use no significa
  que esté bien diseñada, significa que estrecharla es un cambio de contrato con
  39 ficheros, no una limpieza.

### L13 — Los helpers duplicados HAN DERIVADO: el nombre no es un contrato
- **Origen:** P6, paso previo obligatorio de diffear antes de fusionar (2026-07-31).
- **Qué:** agrupadas por cuerpo normalizado, las copias del mismo nombre no son
  copias. `unary_builtin_arg` llegó a tener **cuatro semánticas distintas**
  bajo un solo nombre: una que no desenvuelve nada, una que desenvuelve
  `__hold`, una equivalente a ésa escrita de otro modo, y una que desenvuelve
  **`abs`**.
- **Avance 2026-07-31** (`7d7799400`, `1e0ba51da`, y el commit de este cierre
  parcial):

  | helper | al empezar | ahora |
  |---|---|---|
  | `collect_add_terms` | 18 definiciones / 15 variantes | 14 / 14 |
  | `unary_builtin_arg` | 14 definiciones / 10 variantes | **6 / 6** |

  Consolidadas las 4 copias idénticas de `collect_add_terms` y las 7 de
  `unary_builtin_arg` (4 equivalentes + 3 idénticas), en dos canónicos de
  `cas_math::expr_destructure` cuyos nombres dicen lo que hacen:
  `unary_builtin_arg_through_hold` y `unary_builtin_arg_no_hold`. La de
  cas_solver_core se **renombró** a `unary_builtin_arg_through_abs` en vez de
  fusionarse, que es el arreglo correcto cuando la copia hace otra cosa.
- **Hipótesis descartada por el camino:** los dos `reciprocal_trig_log_domain.rs`
  de cas_math y cas_engine NO son un módulo copiado; comparten el nombre de
  fichero y un helper de 7 líneas, nada más.
- **Pregunta abierta → DECIDIDA 2026-08-01:** `no_hold` es el default CORRECTO
  para las policies de integración. Evidencia: (a) la entrada
  `integrate_symbolic_expr` no desenvuelve holds, así que un hold en un punto
  de match produce integral SIN evaluar — residual visible y seguro, nunca
  wrong answer; (b) en pipelines reales el hold se disuelve por distribución
  antes de llegar al backend (sonda `integrate(cos·expand((sin+1)²))` integra
  bien); (c) donde los holds SÍ llegan, el código ya los desenvuelve costura a
  costura (`general.rs`, `by_parts.rs`, `logs_exp.rs`); (d) un see-through
  indiscriminado sería MÁS arriesgado: casar a través del hold y reconstruir
  con los nodos internos tira la barrera que expand/factor instaló contra
  manglings conocidos. Decisión grabada en el doc-comment del canónico.
- **Lo que queda:** las 14 variantes de `collect_add_terms` son todas
  singletons; ahí no hay dedup posible, solo el trabajo caso por caso de decidir
  si cada nombre miente.
- **Barrido de helpers de TEST ejecutado 2026-08-01:** `solve_display` 22
  definiciones → 7 variantes (un cluster de 8 idénticas y otro de 7),
  `simplify_str` 19 → 5, `create_full_simplifier` 9 → 8 (deriva casi total,
  no fusionable), `parse_wire`/`run_cli` singletons. Consolidado el cluster de
  8 en `tests/inequality_utils/mod.rs` — en módulo PROPIO y no en
  `test_utils`, porque los wrappers del engine incluyen `test_utils` bajo
  `extern crate cas_engine as cas_solver` y todo lo que entre ahí debe
  resolver bajo ambas identidades (`display_solution_set` no existe en la
  superficie del engine; el primer intento rompió los 3 wrappers y se movió).
  Los clusters restantes (7×solve_display, 6/4/4×simplify_str) quedan medidos
  y anotados para una pasada igual.

### L16 — El input del 7/3 ahora CUELGA, y el presupuesto no lo poda
- **Origen:** destapado por el fix de L15 (2026-08-01).
- **Qué:** con el colapso erróneo eliminado, `integrate(cos(x)*(sin(x)+1)^2, x)`
  pasa de mentir en segundos a moler >240 s: el colapso actuaba de válvula de
  escape de una búsqueda patológica preexistente. El molino medido NO es la
  ruta opaca (devuelve None al instante tras el fix) sino la **estrategia 2**
  de `div_expand_cancel` (expand-then-compare con simplifies completos sobre
  formas trig expandidas, ~150 s por invocación) repetida por el router de
  integración, que además eligió Weierstrass/medio-ángulo para un integrando
  con sustitución u=1+sin evidente.
- **Hallazgo agravante:** `--budget standard` NO poda el bucle (exit 124 con
  presupuesto activo) — hay trabajo no medido por el sistema de presupuesto.
- **Doctrina aplicable:** familia C5 («HANG de oscilación expand↔factor — fix
  de orquestación, no apresurar»). No se parchea en caliente: mejor un hang
  honesto que una respuesta incorrecta instantánea.
- **Pista (a) EJECUTADA 2026-08-01 (`79ed4ce58`):** la vía u-du simbólica
  (`symbolic_power_substitution_from_base`: deriva la base con el
  diferenciador completo y exige cofactor = s·u' exacto) caza esta familia
  ANTES del carril Weierstrass — `∫cos·(sin+1)²` → `(sin+1)³/3` compuesta, el
  hang muere de rebote y sin tocar la zona de orquestación. Quedan latentes
  para otras familias patológicas: **(b)** metrar la estrategia 2 en el
  presupuesto (verificado que `--budget standard` no la poda) y **(c)**
  abstención de estrategia 2 con abstracción opaca parcial.

### L14 — Tres suposiciones de utillaje que el troceo destapó (todas corregidas)
- **Origen:** P7, troceo de `focused_rule_substeps.rs` (2026-07-31).
- **Qué falló, y por qué vale registrarlo:** las tres son la misma clase de
  error —dar por hecho un patrón sin comprobarlo— y las tres fallaron
  *ruidosamente*, que es lo que salvó el paso:
  1. **Los módulos de test no siempre se llaman `tests`.** Aquí son cinco con
     nombre propio (`limit_notable_tests`, `named_identity_matcher_tests`…).
     El extractor solo reconocía `mod tests`, y el analizador de visibilidad
     solo leía un fichero `tests.rs`, así que dejó privadas funciones que los
     tests importan por nombre → E0432. Ahora se lee el directorio entero.
  2. **Las rutas `super::` también están en producción, no solo en tests.**
     L9 se quedó corta: `super::visible_rule_names::…` y
     `super::nested_fraction_analysis::…` dejan de resolver al bajar un nivel.
     La reescritura a `super::super::` hay que aplicarla a TODO bloque movido.
  3. **Una función puede usarse sin llamarse.** `reduce(gcd_usize)` la pasa
     como valor; un detector de llamadas que busca `nombre(` no la ve y la deja
     privada. Ahora cuenta también que otro módulo la NOMBRE.
- **Regla general que dejan las tres, y la de L10 y la del `endswith`:** cuando
  una herramienta de análisis decide *visibilidad* o *pertenencia*, equivocarse
  por exceso es inocuo y por defecto rompe. Elegir siempre la
  sobreaproximación.

---

## Cerrados

### L3 — Harness metamórfico → CERRADO 2026-08-01 (`6789317fa`)
La «cirugía» que P5 declinó resultó mecánica con el utillaje de P1-P7:
18.923 → main.rs de 1.358 + 9 submódulos por familia de infraestructura, 455
fns idénticas contra HEAD, reparto exacto 74 passed / 65 ignored conservado en
los DOS binarios. Dos trampas nuevas cazadas por los gates: atributos
`cfg_attr` MULTILÍNEA que dejaban 5 `#[test]` huérfanos (extractor corregido
con balance de corchetes), y el wrapper cross-crate de cas_engine con
`extern crate cas_engine as cas_solver` — un consumidor por `#[path]`+ALIAS
invisible a cualquier escaneo de imports, que además soporta los repros de
PERFORMANCE_TRACK_PLAN (se conserva apuntado al nuevo main.rs). Nota: ese
wrapper compila la suite DOS veces por corrida de workspace — coste conocido,
no tocado.

### L17 — CERRADO: barrido de la clase L15 — el segundo inquilino estaba en el gate de cero exacto
- **Origen:** ciclos 2026-08-01 (2ª tanda); fix `a31bff030`.
- **Qué:** auditados los 6 generadores de variables sintéticas del workspace
  (la clase del bug 7/3). Dos ya sembraban anti-colisión
  (`polynomial_identity_support`, `verification_algebraic` — tercer
  implementador del mismo patrón, candidato a unificar); los `uc{N}` de dsolve
  son **inalcanzables** (cualquier símbolo de usuario en la ODE declina la
  ruta antes de crear los frescos — sondeado con control); y
  `poly_compare::poly_is_zero_opaque` tenía el agujero real: sus átomos
  `__polyzero_*` de nombre fijo fusionaban con variables homónimas del árbol y
  el gate declaraba cero `sin(x) − __polyzero_atom_0` (test escrito en rojo
  primero).
- **Alcance honesto:** no se encontró ruta pública que lo dispare hoy (solve,
  cancel y powfold idénticos al control), pero el gate alimenta decisiones
  drop/keep y la doctrina es exactitud en el gate, no confianza en los
  consumidores. Cerrado con el patrón de bases desplazadas de L15.
- **Pendiente menor anotado:** unificar los TRES implementadores del patrón
  «nombre fresco esquivando el árbol» (engine, verification_algebraic,
  poly_compare + el de div_expand) en un helper de cas_ast — misma familia
  L13, sin urgencia.

### L11 — Detectores sec/csc de potencia impar → CERRADO 2026-08-01 (borrados)
La cuarentena se resolvió haciendo el trabajo que el informe de saneamiento del
2026-07-02 (§11, Clase A) dejaba encargado a «la campaña de universalidad»:
**verificar antes de borrar**. Verificado:
- Los tres predicados (`integrate_symbolic_is_{sec,csc}_third_affine_target`,
  `integrate_symbolic_is_polynomial_times_constant_base_power_target`) eran
  **espejos redundantes**: sus handlers están cableados directamente en el
  router (`support.rs:799` sirve `∫sec³` desde el commit `863ee59fd` de junio),
  y las capacidades responden bien HOY (`∫sec³`, `∫csc³`, `∫sec⁵`, `∫csc⁵`,
  `∫x·2^x`, comprobadas antes y después del borrado).
- Las potencias PARES consumen sus predicados desde las rutas de presentación;
  las impares nunca los necesitaron.
- El frontier que los querría (F2) está cerrado desde 2026-07-31.
Borrados los 3 predicados + su único test (que solo ejercitaba al predicado).
El compilador no emite ni un dead_code tras el borrado: los handlers quedan
vivos vía router, como predecía el análisis.

### L15 — CERRADO: wrong answer 7/3 por colisión de temps opacos (P0, preexistente)
- **Origen:** sondas de la pregunta abierta de L13 (2026-07-31/08-01); fix en
  `a8a7dbdc2`.
- **Qué:** `integrate(cos(x)*(sin(x)+1)^2, x)` devolvía `7/3` — una constante
  como primitiva. Atribución por worktree: preexistente a toda la campaña.
  Causa: `prepare_opaque_shared_substitution` (cas_math) generaba temps
  `__opq0…` sin comprobar colisiones; en rondas opacas ANIDADAS el árbol ya
  contiene `__opq0` del nivel exterior y `sin(x/2) := __opq0` fusionaba dos
  átomos (verificación algebraica: con s=o, N/D = 56/24 = 7/3 exacto). Segundo
  defecto en la misma copia: el matcher greedy quemaba el `shared_limit` en
  pares duplicados del mismo átomo.
- **La parte que confirma L13 con daño real:** el asignador hermano de
  `cas_engine::polynomial_identity_support` YA tenía ambos fixes (siembra con
  `collect_variables` + `dedup_expr_ids`). El fix se aplicó una vez y nunca
  viajó a la copia de cas_math. «Un fix en una copia no llega a las otras» dejó
  de ser un riesgo teórico: era un P0 en producción.
- **Tests:** 2 regresiones unitarias rápidas en cas_math + 2 end-to-end
  `#[ignore]` (~150 s cada uno) en `cas_engine/tests/opaque_quotient_soundness.rs`.

### L1 — rustfmt sucio preexistente → CERRADO 2026-07-31 (`711914960`)
Eran **seis** hunks en cinco ficheros, no dos: la entrada original solo vio el
que el troceo de P5 puso delante. Todos formato puro (encadenados que caben en
una línea, un `pub use` desordenado, una tabla de comentarios realineada).
`cargo fmt --all --check` da exit 0 y ya sirve como gate binario.
**Lección:** una entrada de ledger escrita desde un solo punto de observación
subestima el alcance; al cerrarla, volver a medir en todo el workspace.

### L4 — Rutas `--exact` obsoletas → CERRADO 2026-07-31 (`3b282cc84`)
Ocho comandos prefijados con su submódulo real (buscado, no adivinado) y
verificado que vuelven a ejecutar. Las otras cuatro referencias eran filtros por
subcadena y no necesitaban cambio.
`ENGINE_COMBINATION_LEDGER_ARCHIVE_2026_05.md` se deja intacto pese a sus 15
referencias: es un archivo histórico y reescribirlo falsearía lo ejecutado
entonces.

### L5 — Atribución de tiempo de CI → CERRADO 2026-07-31
La sospecha se confirma, y por más margen del previsto. Suma de tiempos de test
de los 361 binarios: **265 s (4,4 min)**, no los ~19 min medidos el 07-28.

| suite | 07-28 | 07-31 |
|---|---:|---:|
| `cli_contract_tests` | 267 s | 59 s |
| `steps_divergence_gate_tests` | 138 s | 63 s |
| `stress_solve_tests` | 139 s | **2 s** |
| `nonaffine_trig_principal_drop_contract_tests` | 185 s | **1 s** |

Los dos últimos corren el mismo número de tests que entonces (80 y 3), así que
el speedup es real —campaña de perf del orquestador y fixes de F13—, no un
filtro que se los salte. Corregido `.claude/skills/auto-mejora/SKILL.md`, que
era el documento que habría dirigido el trabajo.
**Lección:** un plan de acelerar CI sobre las cifras viejas habría optimizado
dos suites que ya tardan 1 y 2 segundos. Medir, no heredar.

### L2 — Helpers de test duplicados por todo el workspace → reformulado
Sustituido por L13, que mide lo mismo con más precisión y sobre el código de
producción: el problema no es el número de copias sino que han divergido.
El barrido de helpers de test (`solve_display` ×22, `simplify_str` ×19…) sigue
pendiente, pero hay que abordarlo con el mismo método: diffear primero.

---

## Nota de método

Un aviso que costó un rato en P5: verificar un troceo comparando ficheros con
`f.endswith("main.rs")` **se salta `misc_domain.rs` en silencio** —
`"misc_domain.rs".endswith("main.rs")` es `True`—. El verificador daba 500 de
505 tests y parecía un fallo del troceo. Comparar el nombre completo
(`Path(f).name == "main.rs"`), no el sufijo. Es la misma familia de error que
los falsos verdes de `| tail` y `; echo OK` ya registrados: **cuando una
verificación acusa al código, sospecha primero de la verificación.**
