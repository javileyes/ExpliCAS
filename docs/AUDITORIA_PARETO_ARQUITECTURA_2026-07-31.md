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
| P1 | `orchestrator.rs` → directorio `orchestrator/` (mod.rs re-exporta los 4 items pub; submódulos por familia de prefijo; shortcuts de regresión a `regression_shortcuts/`; tests inline a fichero propio) | Medio, mecánico | Navegabilidad, menos conflictos, sedimento visible y auditable |
| P2 | `rules/arithmetic.rs` → repartir las 30 reglas en la taxonomía existente de `rules/`, helpers a `rules/support/`; re-export desde el path original | Medio, mecánico | El catch-all desaparece; cada regla vive con su familia |
| P3 | `symbolic_integration_support.rs` y `limits_support.rs` → directorios con `mod.rs` de API curada (re-export explícito) e internals privados | Medio | Reduce la superficie de 91 pub; recorta el acoplamiento aferente a cas_math |
| P4 | `solve_backend_local.rs` → partir por familia de ecuación | Medio | Reparte el churn más alto del repo; menos conflictos en paralelo |
| P5 | Test-monolitos (`cli_contract_tests.rs` y compañía) → un fichero por dominio | Bajo | Riesgo casi nulo; el fichero más tocado del repo deja de ser imán de conflictos |
| P6 | Dedup de `collect_add_terms` / `unary_builtin_arg` y similares | Bajo | Elimina deriva silenciosa entre 13–14 copias |

**Regla de oro para ejecutar:** solo movimientos de módulo, cero cambios de lógica.
`cargo test --workspace` completo **antes** de empezar (clean git ≠ green tests) y tras cada
paso; gate steps-on/off si se roza simplificación; commit por paso; sin push (push = deploy);
jamás amend sobre hash-stamps. P1, P2 y P5 son independientes entre sí y se pueden hacer en
sesiones separadas sin pisarse.
