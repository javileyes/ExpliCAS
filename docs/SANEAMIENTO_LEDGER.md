# Ledger de saneamiento

Sospechosos observados **mientras** se reestructura (protocolo de
`AUDITORIA_PARETO_ARQUITECTURA_2026-07-31.md`, §6): se anotan sin actuar, para
no abrir madrigueras a mitad de un movimiento. Las tandas de saneamiento
posteriores se alimentan de aquí, no de la memoria.

Estados: `abierto` · `en curso` · `cerrado` · `descartado`.

---

## Abiertos

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

### L11 — Detectores de sec/csc de potencia IMPAR sin cablear (superados, no pendientes)
- **Origen:** P3, medición de superficie pública (2026-07-31).
- **Qué:** en `symbolic_integration_support` la familia de predicados
  `integrate_symbolic_is_{sec,csc}_{third,fourth,sixth,eighth}_affine_target`
  tiene una asimetría limpia: las potencias **pares** se usan en 3 ficheros
  externos cada una; las dos de potencia **tercera**, en **ninguno**.
- **Comprobado contra el motor antes de juzgar:** `integrate(sec(x)^3, x)`
  devuelve el resultado correcto `(ln|tan(x)+sec(x)| + tan(x)·sec(x))/2`. O sea
  que la capacidad existe y se sirve por OTRA ruta: no son semillas de trabajo
  pendiente, son **detectores superados**.
- **Estado:** cuarentena, no borrado. Son `pub`, así que el compilador no puede
  certificarlos, y el criterio del protocolo (compilador + registries + frentes
  vivos) no se cumple entero. Antes de quitarlos hay que contrastar con
  `docs/G1_RATIONAL_INTEGRATION_SCOPING.md` y el frontier de cálculo.
- **Tercer candidato del mismo barrido:**
  `integrate_symbolic_is_polynomial_times_constant_base_power_target`, sin uso
  ni siquiera en tests.

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
  copias:

  | helper | definiciones | variantes distintas |
  |---|---:|---:|
  | `collect_add_terms` | 18 | **15** |
  | `unary_builtin_arg` | 14 | **10** |

  Una de las de `collect_add_terms` ni siquiera comparte firma
  (`&mut Context` en `div_add_common_factor_from_den_support.rs`), lo que
  delata semántica distinta, no un simple retoque.
- **Por qué importa:** el riesgo que se temía —«un fix en una copia no llega a
  las otras»— es peor de lo previsto: quien lea `collect_add_terms` en un
  fichero y asuma el comportamiento del que conoce se equivocará en 13 de cada
  18 casos. Y confirma que **fusionar a ciegas habría sido un cambio de
  comportamiento en trece sitios**, no una limpieza.
- **Hecho:** consolidado el único cluster genuinamente idéntico (4 copias en
  los `div_*_support` de cas_math → `collect_additive_terms_flat_add` de
  `expr_terms.rs`).
- **Pendiente, y NO es mecánico:** las 14 variantes restantes piden un análisis
  caso por caso —¿es deriva accidental o especialización legítima?—. Si es lo
  segundo, el arreglo no es fusionar sino **renombrar**, para que el nombre deje
  de prometer algo que no cumple. Quedan además los clusters idénticos de
  `unary_builtin_arg` (3 en cas_math, 2 en cas_solver y 2 cruzando
  cas_engine/cas_math, estos últimos en ficheros con el mismo nombre
  `reciprocal_trig_log_domain.rs`, que huele a módulo copiado entero).

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
