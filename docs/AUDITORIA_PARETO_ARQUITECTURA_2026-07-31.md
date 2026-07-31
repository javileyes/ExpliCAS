# Auditoría Pareto de arquitectura — cuellos de botella del engine (2026-07-31)

**Método:** análisis económico sin agentes ni profiling nuevo. Tres señales baratas cruzadas:
tamaño de fichero × churn de git (4 meses: 2026-04-01 → hoy) × acoplamiento (fan-in/fan-out
medido con grep + grafo real de crates desde los Cargo.toml). El runtime ya tuvo campaña de
perf reciente (corpus 5,37s → 1,64s, memos por nodo) y no se re-perfila aquí.

## Resumen ejecutivo

**El cuello de botella del engine hoy no es el runtime: es el bucle de desarrollo.**
Seis ficheros fuente concentran ~173k de las 814k líneas del workspace (21%) y reciben el
12,5% de todos los toques de git (635 de 5.086 file-touches en 4 meses, repartidos entre
1.308 ficheros distintos) — una densidad de cambio **27× la media**. Cada fix aterriza en un
fichero de 15k–42k líneas, dentro de la cadena de compilación más larga del workspace.

**La buena noticia estructural:** todos son *icebergs* — API pública mínima (1–4 items) con
decenas de miles de líneas privadas debajo. Se pueden trocear en submódulos **sin tocar a
ningún consumidor**: riesgo de API cero, refactor puramente mecánico.

## El 20% de Pareto: los 6 ficheros

| Fichero | Líneas | Código real¹ | Toques 4m | Anatomía | API pública |
|---|---:|---:|---:|---|---|
| `cas_engine/src/orchestrator.rs` | 42.307 | ~30.000 | 78 | 1.445 fns | 1 struct, 3 métodos |
| `cas_engine/src/rules/arithmetic.rs` | 39.346 | ~30.750 | 57 | 1.091 fns, 30 reglas | solo los structs de regla |
| `cas_math/src/symbolic_integration_support.rs` | 29.999 | ~23.700 | 105 | 960 fns, 0 impl | 91 pub fns planas |
| `cas_didactic/src/didactic/focused_rule_substeps.rs` | 26.006 | — | 123 | 842 fns, 54 tipos | interna |
| `cas_math/src/limits_support.rs` | 19.957 | — | 88 | bolsa de fns | pub fns planas |
| `cas_solver/src/solve_backend_local.rs` | 15.616 | ~15.000 | **184** | 220 fns | 1 item |

¹ Restando el `mod tests` inline (orchestrator lleva ~12k líneas de tests dentro del propio fichero).

A esto se suman los **test-monolitos**: `cli_contract_tests.rs` (10,4k líneas, **278 toques
— el fichero más tocado del repo**), `semantics_cli_contract_tests.rs` (12,2k / 118),
`integrate_contract_tests.rs` (16,8k / 96).

## Cuellos de botella en detalle

### 1. `orchestrator.rs` — el monolito privado (P1)

42k líneas donde la API real (`Orchestrator::new/for_expand/simplify_pipeline`) aparece en la
**línea 26.193**: hay 26k líneas de helpers privados antes de llegar al struct que da nombre al
fichero. Clustering de nombres: 272 `simplify_*`, 128 `extract_*`, 123 `try_*`, 65 `is_*`,
46 `build_*` — familias claras que ya dibujan los submódulos naturales. Además hay un
**sedimento de shortcuts de regresión** con nombres kilométricos
(`zero_product_with_exact_zero_child_shortcut_handles_...`): cada fix de auditoría añadió su
caso especial aquí en vez de una regla. Fan-in: solo `lib.rs` y `engine/orchestration.rs`.

### 2. `rules/arithmetic.rs` — el estrato pre-taxonomía (P2)

39k líneas = **37% de todo `rules/`** (106k); el siguiente fichero de reglas mide 4,2k. Contiene
30 de las 101 `impl Rule` del engine más ~1.060 helpers. El directorio ya tiene la taxonomía
buena (`algebra/`, `calculus/`, `exponents/`, `hyperbolic/`…) — `arithmetic.rs` es el estrato
geológico anterior a esa taxonomía. Los 28 ficheros que lo importan solo consumen los structs
de regla (`MulOneRule`, `CombineConstantsRule`…), nunca los helpers: re-exportando los structs
desde el mismo path, el troceo es invisible hacia fuera.

### 3. `cas_math` y el patrón `*_support.rs` — la raíz de la cadena de compilación (P3)

180k líneas, **1.044 pub fns** (vs 238 en cas_engine, que es más grande). El patrón "un fichero
plano `<tema>_support.rs` con bolsa de funciones públicas y 0 impl blocks" escala mal:
`symbolic_integration_support.rs` expone 91 pub fns — todo consumidor puede alcanzar cualquier
interior, así que ninguna función es refactorizable con seguridad local. Y `cas_math` está en la
raíz del DAG: tocarla recompila engine + solver + didactic + session + cli/wasm (~530k líneas
aguas abajo). El propio `Cargo.toml` del workspace tuvo que apagar debuginfo porque el linker
moría en macOS — señal física de que el tamaño de compilación está en el límite.

### 4. `solve_backend_local.rs` — el campeón de churn (P4)

**184 toques en 4 meses (más de 1/día)** en un fichero de 15,6k líneas: es donde aterriza cada
fix de solve. Máximo riesgo de conflicto con trabajo paralelo (los workflows adversariales
mutan el working tree — lección ya registrada en memoria). Partirlo por familia de ecuación
reparte el churn y elimina el punto único de conflicto.

### 5. `focused_rule_substeps.rs` — crecimiento lineal garantizado (P5)

26k líneas, 123 toques, 842 fns / 54 tipos: el dispatch de narración didáctica. Cada regla que
gana narración (Frente E) añade su builder **aquí**. Sin partición por familia de reglas seguirá
creciendo linealmente con la campaña educativa.

### 6. Duplicación de helpers con el mismo nombre

`collect_add_terms` ×13 definiciones, `unary_builtin_arg` ×14, repartidas por crates y ficheros.
Riesgo real: deriva silenciosa de comportamiento — un fix en una copia no llega a las otras 12
(patrón ya visto en auditorías: "la reducción cubre el caso nombrado y pierde el par/negado").
Candidatos a consolidarse en `cas_ast::views` / helpers comunes de `cas_math`.

## Lo que está sano (no tocar)

- **El grafo de crates es un DAG limpio, sin ciclos.** Los aparentes ciclos engine↔solver↔session
  son solo `dev-dependencies` para tests de integración. Capas correctas:
  `ast → math → solver_core → engine → solver → didactic/session → cli/wasm`.
- **Las APIs iceberg** hacen el refactor barato: el 95% del contenido de los god-files es privado.
- **Runtime**: campaña reciente cerrada (memos por nodo con eje de opciones en la clave, corpus
  1,64s); hangs de pow gigante (F13) cerrados con el carril sci. Residual conocido: oscilación
  expand↔factor (C5), que es fix de orquestación pendiente y no debe apresurarse.
- `cas_ast` pequeño (6k) y estable — buen lecho de roca.

## Plan de ensanchado recomendado (orden ROI)

| # | Acción | Esfuerzo | Beneficio |
|---|---|---|---|
| P1 | ~~`orchestrator.rs` → directorio `orchestrator/`~~ | Medio, mecánico | **HECHO 2026-07-31** — ver más abajo |
| P2 | `rules/arithmetic.rs` → repartir las 30 reglas en la taxonomía existente de `rules/`, helpers a `rules/support/`; re-export desde el path original | Medio, mecánico | El catch-all desaparece; cada regla vive con su familia |
| P3 | `symbolic_integration_support.rs` y `limits_support.rs` → directorios con `mod.rs` de API curada (re-export explícito) e internals privados | Medio | Reduce la superficie de 91 pub; recorta el acoplamiento aferente a cas_math |
| P4 | `solve_backend_local.rs` → partir por familia de ecuación | Medio | Reparte el churn más alto del repo; menos conflictos en paralelo |
| P5 | ~~Test-monolitos (`cli_contract_tests.rs` y compañía) → un fichero por dominio~~ | Bajo | **HECHO 2026-07-31** — ver más abajo |
| P6 | Dedup de `collect_add_terms` / `unary_builtin_arg` y similares | Bajo | Elimina deriva silenciosa entre 13–14 copias |

P1, P2 y P5 son independientes entre sí y se pueden hacer en sesiones separadas sin pisarse.

## P1 ejecutado (2026-07-31)

`orchestrator.rs`, **42.307 → 4.065 líneas** en el padre, repartido en 20
ficheros, en tres commits (`1ef2121d1`, `51ce1d0c2`, `079eeffc9`):

| Paso | Qué | Resultado |
|---|---|---|
| 1/4 | sale el `mod tests` inline (711 tests, 12.225 líneas) | 42.307 → 30.081 |
| 2/4 | las 692 fns de producción → 10 submódulos por familia + `support` | padre: 4.065 |
| 3/4 | los 711 tests → 9 submódulos con **las mismas familias** | `tests.rs`: 31 líneas |

Ningún fichero pasa de 4.787 líneas. La API pública (`Orchestrator`, `new`,
`for_expand`, `simplify_pipeline`) no se toca, y los dos únicos consumidores
(`lib.rs` y `engine/orchestration.rs`) no se enteran.

### Lo que se midió ANTES de partir, y cambia la expectativa

El grafo de llamadas de las 692 fns **es una bola**. Agrupar guiándose por el
grafo (propagación de etiquetas) da 98,8% de cohesión metiendo 627 de las 692
en un solo grupo: no hay costuras naturales. La partición por familia de nombre
deja el 34,1% de las llamadas dentro del módulo, contra el 11,1% de una
partición al azar — los nombres llevan estructura real, pero dos tercios de las
llamadas siguen cruzando módulo.

**Conclusión que vale para P2, P3 y P4: trocear compra navegabilidad, tamaño de
fichero y menos conflictos; NO compra desacoplamiento.** Desacoplar de verdad
exige rediseño —interfaces, inversión de dependencias—, que es trabajo de
diseño, no de mudanza. Prometer lo segundo moviendo ficheros sería falso.

Lo que sí apareció como capa real: **41 primitivas compartidas**, llamadas
desde 4 o más familias (`build_mul_expr_from_factors_root`,
`isolated_simplify_rewrites_to_zero`, `finish_standard_root_shortcut`…), ahora
en `orchestrator/support.rs`.

### Visibilidad mínima: el paso 2 del protocolo de código muerto

En vez de marcar las 692 como `pub(super)` —cómodo, y habría cegado el lint—,
se marcaron solo las **548** que se usan fuera de su módulo; **144 quedan
privadas**. Con eso el compilador ya puede certificar, y su veredicto es un
hallazgo negativo que ahorra futuras búsquedas: **ni un solo `dead_code`**. El
orquestador no tiene funciones huérfanas de nivel superior; su sedimento está
vivo.

### Retoques inevitables (no son movimiento puro, y se declaran)

Desangrar el bloque de tests y añadir `pub(super)` alarga líneas, así que
rustfmt vuelve a partirlas: 86 firmas quedaron reformateadas. El fichero estaba
limpio de rustfmt en HEAD y sigue limpio. La verificación se hizo comparando
las 692 fns contra HEAD normalizando espacios, prefijo de visibilidad y el
reajuste de rustfmt: **0 cuerpos alterados**, ninguna fn perdida, duplicada ni
añadida.

Y una trampa que conviene no repetir en P2–P4: el bucket residual se llamaba
`core`, y `mod core` **ensombrece el crate `core` de Rust**. Compilaba y pasaba
los tests, pero envenenaba cualquier `core::mem::…` futuro escrito en ese
módulo. Renombrado a `general`; anotado como L7.

## P5 ejecutado (2026-07-31)

Cuatro monolitos, **60.611 líneas planas → 56 ficheros**, en cuatro commits de
movimiento puro (`efd79191a`, `0b553e318`, `b67ffa1fa`, `befbd71ca`):

| Fichero | Antes | Tests | Después | Eje de troceo |
|---|---:|---:|---|---|
| `cas_cli/tests/cli_contract_tests` | 10.393 | 228 | main + 15 | dominio matemático |
| `cas_cli/tests/integrate_contract_tests` | 16.778 | 381 | main + 14 | técnica de integración |
| `cas_cli/tests/semantics_cli_contract_tests` | 12.284 | 505 | main + 20 | comando × tema |
| `cas_solver/tests/diff_step_contract_tests` | 21.156 | 264 | main + 7 | familia de función derivada |

**Decisión de diseño: un solo binario por suite.** La convención de cargo
`tests/<nombre>/main.rs` (no `tests/<nombre>.rs` — en un crate raíz los `mod`
se resuelven en el *mismo* directorio, de ahí un `E0583` inicial) mantiene el
nombre del target, así que los `cargo test --test <bin>` del
`SLOW_CI_TEST_LEDGER` siguen valiendo y no se añaden 56 binarios al enlazado de
un workspace que ya tuvo que apagar debuginfo. Sin `nextest` ni CI de tests,
cargo ejecuta los binarios en serie: más binarios no habría acelerado nada.
Los submódulos heredan imports y helpers vía `use super::*`.

**Lección sobre los ejes.** El eje por dominio matemático solo sirve para
suites genuinamente multidominio. En un fichero monotemático no separa nada
(371 de 381 de integrate caían en «integración»): ahí la costura útil es *cómo*
se hace, no *qué* se hace. Y los criterios transversales —`verification`,
`steps`— deben ir los ÚLTIMOS de la lista de reglas: como criterio primario se
tragan todo (`steps` capturaba 175 de 225 tests de derive) porque casi todo
test verifica y casi todo test mira los pasos.

**Verificación.** Cada troceo se comprobó con un verificador independiente que
compara los bloques `#[test]` contra el commit padre: 1.378 bloques idénticos
byte a byte, ningún nombre perdido ni añadido. Suite completa verde antes
(12.771 tests) y después. Único retoque no-movimiento: colapsar las rachas de
líneas en blanco que dejan los tests al salir, exigido por rustfmt.

**Lo que NO se hizo, y por qué.** `metamorphic_simplification_tests.rs` (18.923
líneas) queda intacto: son 139 tests contra 316 helpers, o sea un fichero de
infraestructura. Trocearlo exige mover maquinaria, que es cirugía y no
movimiento. Anotado como L3 en `SANEAMIENTO_LEDGER.md`.

Los sospechosos observados durante los movimientos (rustfmt sucio preexistente,
helpers de test duplicados ×22/×19/×16, rutas `--exact` obsoletas en docs,
atribución de tiempo de CI probablemente inflada) están en
`SANEAMIENTO_LEDGER.md` — anotados, no tocados.

## Protocolo: sanear mientras se reestructura

La reestructuración es la mejor oportunidad de censo del código (al mover 1.445 funciones se
*ve* cada una), así que conviene auditar en el mismo pase. Pero la disciplina es estricta:
**observar mientras se mueve, sí; operar mientras se mueve, no.** Cada tipo de cambio va en su
propio commit.

### 1. Separación estricta move / cirugía

Un commit de movimiento puro es verificable casi mecánicamente: `git diff --color-moved`
muestra los bloques trasladados, los tests quedan verdes trivialmente, y ante una regresión
posterior `git bisect` distingue al instante "move" de "cirugía". Mezclar en un commit "moví
esto" con "de paso borré aquello" hace el diff inauditable y esconde regresiones (cicatrices
propias del repo: *bisect antes de creer la atribución*, *git limpio ≠ tests verdes*).

- Commits de move: **cero cambios de lógica**, ni un rename de variable.
- Commits de saneamiento: pequeños y temáticos (un duplicado, un cluster muerto), para que
  bisect y revert sigan siendo baratos.

### 2. Código muerto: la reestructuración ES el detector

`cas_math` expone 1.044 funciones `pub`, y el lint `dead_code` **no puede señalar nada
público** (para el compilador, "alguien externo podría usarla"). Por eso detectar muerto hoy es
adivinar con grep. La secuencia correcta invierte el problema:

1. **Mover** a submódulos (P1–P3) — verde.
2. **Estrechar visibilidad**: API curada en `mod.rs`, internals a `pub(crate)`/privado — verde.
3. **Dejar que el compilador señale**: lo que quede sin llamar tras el estrechamiento es
   candidato certificado, no especulado. Grep especula; el compilador certifica.
4. Solo entonces, decidir cortes (con las salvaguardas del punto 3).

### 3. Salvaguardas contra el falso-muerto

Dos trampas concretas de este repo hacen mentiroso el "0 referencias" de grep:

- **Despacho dinámico**: las 101 reglas se registran vía registry (`register`,
  `target_types`, allowlist de funciones del engine que además fallan **en silencio** si no se
  cablean). Un helper puede tener cero referencias directas y estar vivo vía registro. Lección
  ya registrada: el sweep de 0-referencias es seguro en plumbing, **cargado en crates de
  dominio**.
- **Semillas de universalidad compleja**: los residuales vivos de Fase 2 (sinh↔exp con
  argumento constante, `|i|` anidado, equivalencia tri-estado, roots-of-unity) tienen
  infraestructura a medio usar que hoy parece muerta y es el andamio de mañana.

Protocolo: **cuarentena antes que borrado**. Todo candidato se contrasta contra los backlogs
de frentes vivos (docs de fases y auditorías); en caso de duda se marca con
`#[allow(dead_code)]` + comentario de *por qué vive* (o módulo de semillas explícito) en vez
de borrarse. Borrar es barato de deshacer en git pero caro de *re-descubrir*; la cuarentena
mantiene la semilla visible. Un candidato solo se borra si: el compilador lo señala tras el
estrechamiento **y** no aparece en registries/allowlists **y** no casa con ningún frente vivo.

### 4. Duplicados: diffear antes de fusionar

Las 13 copias de `collect_add_terms` (y las 14 de `unary_builtin_arg`) probablemente **ya no
son idénticas** — alguna habrá recibido un fix que las otras no. Fusionar a ciegas cambia el
comportamiento en 12 sitios de golpe. Paso previo obligatorio: diff de implementaciones.

- Idénticas (byte a byte o semánticamente verificadas) → consolidar sin miedo.
- Divergentes → la divergencia es un **hallazgo en sí misma** (¿bug de deriva o divergencia
  intencional?) que merece su propio mini-análisis antes de tocar nada. No se fusiona en el
  mismo commit que se investiga.

### 5. Acoplamientos: salen gratis en el move

Al mover una función a su submódulo, los `use` que hay que arrastrar *son* la medida de su
acoplamiento — el move los hace visibles uno a uno. No hay que buscarlos: hay que anotarlos.

### 6. El ledger de sospechosos

Durante cada move se lleva un ledger (p. ej. `docs/SANEAMIENTO_LEDGER.md`) donde se anota
sin actuar: muerto-aparente, duplicado, acoplamiento raro, shortcut de regresión que podría
generalizarse. Coste cero, no frena el move ni abre madrigueras a mitad de refactor. Las
tandas de saneamiento posteriores se alimentan del ledger, no de la memoria.

### 7. Ritmo y verificación

```
move (verde) → estrechar visibilidad (verde) → sanear con el compilador como oráculo (verde + gate)
```

- `cargo test --workspace` completo **antes** de empezar (clean git ≠ green tests) y tras
  cada paso — con `&&` en las cadenas de shell, nunca `| tail` ni `; echo OK` (falsos verdes
  ya documentados).
- Gate steps-on/off si el corte roza simplificación.
- Commit por paso; sin push (push = deploy); jamás amend sobre hash-stamps.
- Si un saneamiento rompe algo no trivial de atribuir: worktree-bisect antes de revertir.
