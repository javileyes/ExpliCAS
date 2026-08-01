# D0 — Arnés de medición y baseline del desacoplo (2026-08-01)

Ejecución de la fase D0 de `PLAN_DESACOPLO_2026-08.md` sobre el commit
`aa6244597`. Tres entregables: (1) las herramientas de medición de la campaña,
versionadas y generalizadas; (2) las métricas de desacoplo por costura,
definidas operativamente y MEDIDAS hoy (el baseline que D1-D3 deben mover);
(3) los baselines de compilación y de huella funcional contra pre-campaña
(`2d4ad8c35`).

## 1. Herramientas versionadas (`scripts/engine_decoupling_*.py`)

Las cuatro herramientas que decidieron la campaña (4 de 7 recomendaciones del
plan corregidas por la medición) vivían en scratchpad de sesión; ahora están
en `scripts/`, generalizadas para operar sobre los DIRECTORIOS post-troceo
además de sobre ficheros únicos:

| Script | Qué mide | Uso mínimo |
|---|---|---|
| `engine_decoupling_callgraph.py` | grafo de llamadas, % aristas intra-grupo, flujos entre grupos, primitivas candidatas a `support` | `… <fichero.rs\|dir>` |
| `engine_decoupling_closure.py` | cierre transitivo de helpers por punto de entrada (`define_rule!`), solape entre temas, API de facto | `… <dir> --per-entry` |
| `engine_decoupling_dup_diff.py` | variantes reales de un helper duplicado (cuerpo normalizado), con `--diff N M` | `… <nombre_fn>` |
| `engine_decoupling_verify_move.py` | equivalencia de un troceo contra un ref (fns normalizadas o bloques `#[test]` byte a byte); exit≠0 si difiere | `… <original.rs> --ref <REF>` |
| `engine_decoupling_metrics.py` | snapshot integrado de las métricas por costura (JSON versionable) | `… --json <out>` |

**Validación por reproducción**: sobre el árbol actual reproducen los números
publicados de la campaña — orquestador 692 fns y 41 primitivas de `support`
usadas; arithmetic 25 entries / 713 helpers con el cierre por tema EXACTO de
la auditoría (trig 151/15, hyperbolic 42/13, logarithms 34/3, exponents 14/8,
algebra 3/0); 548 `pub(super)`.

## 2. Métricas por costura — baseline en `aa6244597`

Snapshot versionado: `docs/generated/decoupling_metrics_baseline.json`
(regenerable con `python3 scripts/engine_decoupling_metrics.py --seam all`).

### D1 — `rules/arithmetic` (motor de cancelación)

| métrica | baseline | éxito D1 |
|---|---:|---|
| entries (`define_rule!`) | 25 | disparadores que SOLO detectan y delegan |
| helpers nivel superior | 713 | núcleo `cancellation` con API estrecha |
| helpers compartidos entre temas | 162 | → «solo la API» |
| % aristas intra-fichero del dir | 36,4% | sube al migrar disparadores |

La **API de facto** ya es visible (helpers alcanzados desde más entries):
`exprs_equal_up_to_add_term_order` (16/25), `collect_add_terms` (16),
`run_default_simplify` (15), `ambient_pipeline_value_domain` (15),
`signed_term_expr` (15). Nótese que `collect_add_terms` — el helper con 14
variantes divergentes en el workspace (L13) — es a la vez API de facto del
motor: la costura D1 y el saneamiento L13 se tocan.

### D2 — `orchestrator/`

| métrica | baseline | éxito D2 |
|---|---:|---|
| % aristas intra-fichero | **28,1%** | sube por peldaños |
| fan-in de `support` | 606 aristas desde 9 módulos (305 fns llamadoras, 41 primitivas) | superficie documentada y estrecha |
| `pub(super)` | 548 | baja al formalizar la API interna |
| `pub(crate)` | 0 | — |

⚠️ El plan citaba «34%»: era la cohesión de la partición por REGEX de nombre
pre-troceo (sin `support` separado). Con la partición FÍSICA por fichero —la
métrica canónica en adelante— el baseline es 28,1%: mover 52 primitivas a
`support` convierte sus aristas entrantes en cross por definición. Comparar
siempre contra 28,1%, no contra 34%.

### D3 — `cas_math` (crates)

- Imports cross-familia (ficheros con `crate::<módulo>` de otra familia):
  trig→resto 59, resto→poly 50, integration→resto 45, resto→numeric 36,
  numeric→resto 29, integration→numeric 13, limits→resto 12.
- Módulos puente (importan de más familias): `limits_support` (5),
  `symbolic_integration_support` (5), `general_integration_backend` (4),
  `symbolic_differentiation_support` (4), `root_forms` (3).
- Tamaños: cas_math 181k líneas / 738 `pub fn`; aguas abajo cas_engine 202k,
  cas_solver 98k, cas_solver_core 76k, cas_didactic 46k.

### Baseline de compilación (la métrica que D3 debe mover)

Protocolo: estabilizar → cambio real de contenido en cas_math (una const
añadida a `lib.rs`) → medir → revertir → re-medir. Perfil dev, incremental,
máquina de la sesión. Pared:

| escenario | `cargo build --workspace` | `cargo test --workspace --no-run` |
|---|---:|---:|
| sin cambios (suelo sano) | ~0,2 s | **0,9 s** |
| touch de cas_math (mtime, contenido idéntico) | 3,9 s | — |
| cambio real en cas_math | 4,3 s (6,2 s CPU) | **26,0 s (185 s CPU)** |

- **Invalidación**: tocar `cas_math` invalida **9 crates** (cas_math,
  solver_core, engine, solver, didactic, session, android_ffi, wasm, cli) —
  la cadena entera, como afirmaba el plan.
- **El coste real del bucle editar→validar es ~26 s**, dominado por los TEST
  TARGETS (compilar + enlazar los binarios de la cadena), no por las libs
  (4,3 s; por crate: engine 1,31 s, math 1,14 s, solver 0,80 s, cli 0,74 s,
  solver_core 0,63 s, didactic 0,57 s — HTML en `target/cargo-timings/`).
- Iterar con `cargo test -p <crate> <filtro>` (la cadencia de la skill) paga
  solo el prefijo de la cadena hasta ese crate; los 26 s son el techo
  (validación workspace completa).
- Lo que D3 compraría: reducir el conjunto invalidado al tocar UNA familia
  (integration dejaría de arrastrar poly/limits y sus binarios). Techo de
  mejora ~20 s por iteración workspace. **El orden del plan (D3 al final)
  queda revalidado por la medición**: hay techo real, pero no la urgencia que
  el tamaño de cas_math sugería.

**Trampa medida y purgada durante D0**: un fingerprint STALE heredado en la
raíz del DAG (`cas_ast/src/lib.rs` con mtime de la cosecha 01-ago > dep-info
del 25-jul) convertía CADA validación en cascada completa: tres corridas
consecutivas de `--no-run` a ~27 s / 185 s CPU **sin cambio alguno en el
árbol** — y de paso fabricó una atribución falsa provisional («cli/wasm
volátiles») que la re-medición desmontó. Un `cargo build -p cas_ast` purga el
estado. Si las validaciones van sistemáticamente lentas SIN cambios:
`CARGO_LOG=cargo::core::compiler::fingerprint=info cargo build -p <crate>`
sobre la raíz antes de culpar a la máquina o a los crates hoja (misma familia
que «medir, no heredar»).

## 3. Baseline funcional contra pre-campaña (`2d4ad8c35`)

### El orquestador sigue siendo byte-equivalente

`engine_decoupling_verify_move.py crates/cas_engine/src/orchestrator.rs --ref
2d4ad8c35` certifica: las **692/692 fns** pre-campaña presentes, **0 cuerpos
alterados** (módulo visibilidad/rustfmt), 2 fns nuevas legítimas de la cosecha
(`render`, `simplify_render`). Toda la campaña + cosecha dejó el orquestador
equivalente función a función.

### Delta funcional de huella (scorecard guardrail + pressure)

Diff contra `2d4ad8c35` filtrando claves runtime. Deltas REALES, todos
intencionados y con dueño en el ledger:

- `calculus_integrate_contract.passed` 380→391 (+11 tests u-du de la cosecha).
- `steps_quality_gate.substep_claim_abstained_{hits,rows}` 981→1030 (fallback
  de substeps `d91aa2b80`).
- Listas de comando de 3 suites (nombres de tests nuevos).

El resto del diff son rankings dependientes del timing de la corrida — ruido
conocido (lección `scorecard-huella-latency-noise`), reconocible por familia
de clave: `*_slowest_*`, `*hotspots*`, `*_heavy_rows`, sondas con `*_us`.
Juzgar huellas por contadores e identidad de slots, jamás por esas familias.

## 4. Qué queda fuera de D0 (deliberado)

- El baseline de perf de SUITES ya está re-medido y vigente en la skill
  (2026-07-31: 265 s totales; los repartos por suite, allí) — no se repite.
- `--timings` en frío total (clean build): se difiere a la entrada de D3;
  el incremental-tras-cambio-real es la métrica operativa del bucle de
  desarrollo y ya está anclada.
- Las 110 fichas sin verificar del informe integral y la cola P0 del
  frontier-audit 2026-07-14 no cambian de estado por este ciclo.
