# Informe de saneamiento del código — motor CAS

**Fecha:** 2026-07-02 · **Commit base:** `ab5587aa1` · **Objetivo:** dejar el código en el estado más limpio posible antes de abordar la campaña de universalidad/completitud (solve / integrate / limits).

> **Estado del informe — se construye por fases (presupuesto de cómputo acotado).**
> - **Parte 1: COMPLETA.** Resumen ejecutivo, chokepoints A-D, y los **40 hallazgos verificados adversarialmente** (33 CONFIRMED + 7 ADJUSTED, **0 refutados**) con su plan de saneamiento.
> - **Parte 2a: COMPLETA (§8).** Lote de 28 hallazgos (**13 CONF + 15 ADJ, 0 refutados**). Añade el **chokepoint E** (prefijos de error duplicados) y P2 de seguridad/rendimiento.
> - **Parte 2b: COMPLETA (§9).** Lote de 35 P2 dup/coupling (**20 CONF + 15 ADJ, 0 refutados**). Chokepoints **F** (colectores de factores ×4-6, adyacentes a P0-G) y **G** (god file `solve_outcome.rs`), cluster copy-paste en `derive/*`, duplicación del harness Python, meta-hallazgo del lint.
> - **Parte 2c: COMPLETA (§10) — TIER P2 CERRADO.** Lote de 58 P2 coupling/perf/dead (**34 CONF + 24 ADJ, 0 refutados**). Mapea los **7 god files restantes** (`symbolic_integration_support.rs` 30k, `orchestrator.rs` 42k, `arithmetic.rs` 39k, `focused_rule_substeps.rs` 23k, `limits_support.rs` 17.5k, `engine_improvement_scorecard.py` 12.6k, `derive_command.rs` 8k), un **cluster de rendimiento nuevo** (7 items en el hot loop), la extensión del chokepoint D a **5 crates + web + python**, y código muerto adicional (session_api 217/401, archivos huérfanos, contratos wire muertos).
> - **TOTAL verificado adversarialmente: 161 verificaciones, 0 refutados** (~120 hallazgos distintos tras dedup). **Todo el tier P1+P2 está verificado.**
> - **Pendiente (opcional):** los ~113 P3 quedan como inventario crudo en `docs/.saneamiento_pendiente_raw.json` (verificables bajo demanda; son menores por definición).
>
> Metodología: workflow multi-agente (30 escáneres read-only = 5 dimensiones × 6 grupos de código; verificadores adversariales 1-por-hallazgo con instrucción de *refutar por defecto*, que re-miden cada cifra, trazan cada ruta y reproducen con el binario release cuando aplica). 274 hallazgos en bruto. **Nada del trabajo caro se perdió** en el corte de presupuesto (todo cosechado de los transcripts en disco).

---

## 1. Resumen ejecutivo

1. **El código es sólido pero tiene deuda arquitectónica concentrada, no dispersa.** Clippy pasa con `-D warnings` y no hay `unsafe`; los problemas están en lo que los lints no ven. De los 40 hallazgos verificados, **la mayor parte se agrupa en 4 chokepoints** que se reportaron una y otra vez desde escáneres independientes (§2). Arreglar esos 4 rinde más que los otros 36 juntos para la campaña.

2. **Hay riesgos de robustez reales alcanzables desde input de usuario** (P1 seguridad, todos CONFIRMED por reproducción en vivo): el parser aborta el proceso por *stack overflow* con paréntesis anidados (~2000-3000 de profundidad, `exit 134`, no capturable); `2^100000000` y `1234...891^(1/2)` cuelgan el motor sin límite (saltan el guard `MAX_ABS_POW=1000` documentado); `(n + 1099511627776)!/n!` cuelga con y sin `--steps`. El servidor web escucha en `0.0.0.0` con CORS `*` sin auth y `/api/import` permite un OOM remoto con ~60 bytes.

3. **La capa de decisión exacta de signo de surds está duplicada 3-4 veces en 3 crates, y ya ha divergido.** Es precisamente el chokepoint que cerró 5+ familias P0 en la campaña de julio (memoria `surd-sign-guard-upgrade-closes-families`). Con 3-4 copias, cada mejora de decidibilidad futura (nth-roots, dos surds, φ, cotas transcendentales) hay que portarla a mano N veces o una rama reabre en silencio un wrong-answer. **Consolidar esto es prerequisito de la campaña, no cosmética.**

4. **La familia `solve_runtime_flow_*` es una torre de callbacks de ~109 funciones**, con una función de **40-41 parámetros (34-35 genéricos)** cuyo cuerpo es solo re-envolver `RefCell` para reenviar a la siguiente capa. 48 funciones con ≥10 parámetros, 289 `#[allow(clippy::too_many_arguments)]`. Es la espina dorsal por la que pasa **toda** extensión de `solve()`: añadir un hook nuevo es cirugía en escopeta por 8-9 archivos en lockstep. Es el mayor impuesto por ciclo de la campaña de solve.

5. **El `dead_code` lint está desactivado de facto en los 2 crates grandes de solver/math** por `pub` general: 49% de los ítems `pub` de `cas_solver_core` (903/1833) y 23% de `cas_math` (333/1434) no se referencian fuera de su archivo. Esto es la **causa raíz** de que capas muertas (módulos superseded, variantes hermanas de refactors de callbacks) se acumulen invisiblemente. Un barrido `pub`→`pub(crate)` devuelve el lint al servicio.

6. **Patrón recurrente de "hermano superseded presente":** cada refactor "añade un callback con default" deja compilada la variante N-1 sin llamar (`predicate_proofs`, `strategy_kernels`, módulo `preflight` entero muerto); cada grado de trig se enumera a mano (58 funciones tan/cot/sec/csc). Es exactamente la forma de bug "el fix cubre el caso nombrado, pierde el hermano" que el historial de auditorías documenta.

7. **Contrato stringly-typed cruzado entre crates:** `cas_solver_core` (crate inferior) hard-codea nombres de reglas producidos por `cas_engine` (crate superior) para decidir keep/drop de pasos didácticos. Renombrar o localizar una regla cambia el filtrado sin error de compilación — la forma exacta del RED heredado donde un paso "Canonicalize Reciprocal Sqrt" desapareció.

8. **Ninguno de los 40 fue refutado.** El verificador adversarial ajustó 7 severidades/detalles (siempre corroborando el fondo, a veces *reforzándolo* — p.ej. la torre de callbacks es peor de lo reportado: 109 funciones, no 92) pero no tumbó ninguno. Alta confianza en accionarlos.

---

## 2. Los chokepoints (máximo apalancamiento)

Estos concentran los reportes repetidos de escáneres independientes. **Son los que la campaña de universalidad tocará en cada ciclo; sanearlos primero cambia el coste marginal de todo lo demás.**

### CHOKEPOINT A — Comparadores exactos de surds/constantes duplicados 3-4× en 3 crates *(P1, dup, CONFIRMED ×5 reportes independientes)*

**Qué:** el mismo algoritmo `sign(A+B√n) = sign(B)·sign(B²n−A²)` (squaring con tracking de signo) vive en:
- `cas_math/src/root_forms.rs:1876` `provable_sign_vs_zero` (canónico, API pública documentada)
- `cas_solver_core/src/solution_set.rs:117` `sign_of_linear_surd` (reimplementado sobre `(p,q,n)` descompuesto; byte-equivalente)
- `cas_solver/src/solve_backend_local.rs:28` `linear_surd_sign` (su propio doc-comment admite "the same exact surd-sign logic as `cas_math::root_forms::provable_sign_vs_zero`")
- `cas_solver/src/solve_backend_local.rs:1330` `cmp_rational_to_quadratic_surd` (variante rational-vs-surd, = `reverse(sign_of_linear_surd(a−r, b, n))`)

Y los satélites también: comparación de nth-roots duplicada entre `solution_set.rs:230/286/306` y `solve_backend_local.rs:1365/1419/1436`; φ=½+½√5 caso-especial dos veces (`as_surd_value` y en solve_backend). El verificador **elevó el conteo a 4 copias** del kernel de signo, no 3.

**Drift confirmado:** las copias ya divergen en capacidades (una conoce φ/Neg, otra no; una hace fallback a `const_sign::provable_const_sign`, otras no). Esta es la forma exacta del bug "el fix cubre el caso nombrado, el hermano produce wrong-answer".

**Por qué importa:** es LA capa que cerró las familias P0 de julio. Toda mejora de decidibilidad futura debe aterrizar N veces o una familia reabre en silencio.

**Fix (M):** hacer `cas_math::root_forms` la única casa. Exportar `sign_of_linear_surd(p,q,n)`, `sign_of_sum_two_surds`, `cmp_rational_to_nth_root`, `compare_positive_nth_roots` (puros `BigRational`, sin `Context`), más un decompositor `as_surd_value`/`as_nth_root_value` que incluya Phi/Neg(Phi) y el caso coeficiente-Mul; enseñar `as_linear_surd` sobre `Constant::Phi`. `solution_set.rs` y `solve_backend_local.rs` pasan a ser delegaciones finas. La delegación pura es huella-safe; unificar las copias divergidas requiere validar la huella + barrido adversarial de surds (ya existe el patrón). `cas_solver_core` y `cas_solver` ya dependen de `cas_math`.

### CHOKEPOINT B — Torre de callbacks `solve_runtime_flow_*` *(P1, coupling, CONFIRMED ×6 reportes independientes)*

**Qué (medido, cifras del verificador que re-contó):** ~**109 funciones `pub`** en 61 archivos `solve_runtime_*.rs` (~6146 líneas); **48 con ≥10 parámetros, 27-31 con ≥15, 15 con ≥20**; la peor es `dispatch_isolation_with_default_kernels_and_default_arithmetic_pow_function_routes_with_state` (`solve_runtime_flow_isolation_default_routes.rs:15`) con **40-41 parámetros / 34-35 genéricos** y where-clause de ~120 líneas — su cuerpo es solo 18 `RefCell::new` para reenviar a la capa siguiente. **289 `#[allow(clippy::too_many_arguments)]`** en el crate. Cadena de aislamiento trazada de **10 capas** de profundidad. Un archivo entero (`solve_runtime_flow_isolation_dispatch_base.rs`, 96 líneas, 22 params verbatim) tiene **cero call sites** (capa muerta).

**Por qué importa:** es la superficie de extensión de `solve()`. Añadir un hook (el shape recurrente de los fixes de la campaña) = editar lista genérica + params + where-clause + re-wrapping de closures en lockstep por 8-9 archivos. El hook `register_blocked_hint`, reciente, aparece en 8 archivos de plumbing.

**Fix (L, por capas, un ciclo cada una, 0-delta huella):** el repo **ya tiene los dos precedentes**: el trait `RuntimeSolveAdapterState` (`cas_engine/src/solve_runtime_state_impl.rs` ya liga 15+ de estos callbacks como métodos) y `PowIsolationRuntimeConfig`, y ya hay un `dispatch_isolation_with_runtime_ctx_and_options_and_state` que empaqueta escalares en `RuntimeSolveCtx`+`SolverOptions` y baja de 41→~23 params. Continuar ese seam: (1) borrar la capa pass-through muerta `..._dispatch_base.rs`; (2) agrupar los callbacks `Fn(&mut T,...)` repetidos (`FContextRef`/`FContextMut`/`FRenderExpr`/`FMapStep`/`FSimplify...`, cada uno ~111 apariciones) en un struct `IsolationHooks<T,S,E>`; (3) convertir la capa más interna primero, dejando las 43 firmas como adaptadores finos que construyen el struct. Colapsar archivos wrapper a medida que migran los call sites.

### CHOKEPOINT C — `pub` general apaga el `dead_code` lint en los crates grandes *(P1+P2, dead, CONFIRMED ×2)*

**Qué:** `cas_solver_core/src/lib.rs` declara los 181 módulos `pub mod` con 0 `pub use`; **903/1833 (49%) ítems `pub` sin referencia externa al archivo**. `cas_math`: **333/1434 `pub fn` (23%)** igual. Como todo es `pub` en un lib crate, `rustc` nunca puede emitir `dead_code` (y el repo compila con `-D warnings`), así que las capas muertas se acumulan invisibles. Peores archivos: `solve_outcome.rs` 326/416 file-local, `number_theory_support.rs` 33, `limits_support.rs` 22, `summation_support.rs` 21.

**Por qué importa:** es la **causa raíz** de que todos los demás hallazgos de código muerto fueran invisibles a clippy. Sin disciplina de visibilidad la campaña seguirá acumulando helpers inalcanzables en los god files.

**Fix (L, mecánico, un god-file por ciclo, huella-idéntica):** demote de ítems file-local a `pub(crate)`/privado, luego dejar que `clippy -D warnings` marque los realmente muertos automáticamente. Empezar por `solve_outcome.rs`. Alternativa/complemento: añadir el script de escaneo usado aquí a `scripts/` como chequeo periódico.

### CHOKEPOINT D — Contrato stringly-typed de nombres de regla cruzando crates *(P1, coupling, CONFIRMED ×2)*

**Qué:** `cas_solver_core` (NO depende de `cas_engine`: sus deps son `cas_ast`/`cas_formatter`/`cas_math`) sin embargo `step_rules.rs:9-25` hace string-match de nombres de regla definidos en `cas_engine` ("Sum Exponents" `arithmetic.rs:39117`, "Cancel Exact Additive Pairs" `arithmetic.rs:14250`, "Conservar integral residual" `eval/simplify_action.rs:2632`, prefijo "Canonicalize"). `is_always_keep_step_rule_name` alimenta `optimize_steps`, y `cas_engine/rule_application.rs:137,391` **llama de vuelta** a `cas_solver_core::step_rules` — el vocabulario de strings fluye en ambos sentidos. "Conservar integral residual" abarca **5 crates**.

**Por qué importa:** renombrar/localizar una regla (la campaña es/en toca exactamente estos títulos en español) cambia keep/drop sin error de compilación — la forma exacta del RED heredado donde "Canonicalize Reciprocal Sqrt" desapareció (steps_count=0).

**Fix (M, huella-safe):** mover los literales a `pub const RULE_* : &str` en `cas_solver_core` (crate más bajo de la cadena) y que `cas_engine`/`cas_didactic`/`cas_solver` referencien esas constantes al construir y matchear. Strings byte-idénticos → 0-delta. Empezar por los 3 "Conservar * residual" y la lista `is_always_keep` (10 nombres). Follow-up: `StepCategory`/`RuleId` tipado en `Step`.

---

## 3. Hallazgos por dimensión (verificados)

### 3.1 Seguridad y robustez (P1 — atacar primero)

| # | Fichero:línea | Qué | Fix | Esf. |
|---|---|---|---|---|
| S1 | `cas_parser/src/parser.rs` (parse_expr:720 y cadena) | **Parser sin guard de profundidad**: `('*3000` aborta el proceso (`exit 134`, stack overflow no capturable). Alcanzable desde CLI, subprocess web, y FFI Android (cuyo doc-comment afirma falsamente capturar todos los panics). `grep depth/recursion/MAX_` en `cas_parser/src` = 0 resultados. | Contador de profundidad enhebrado por `parse_expr/parse_term/parse_unary/parse_atom` (+ cadena `latex_parser.rs`) que devuelve `ParseError::Syntax` pasado un límite (~256). Corregir el doc-comment del FFI. | M |
| S2 | `cas_math/src/power_eval_support.rs:175` | **Materialización de potencia entera ilimitada**: `2^100000000` computa `Pow::pow` sin cota (solo i32), saltándose el `MAX_ABS_POW=1000` documentado en `const_eval.rs` que `EvaluatePowerRule` sí aplica antes de caer a esta ruta. Cuelga el proceso e ignora el time-budget (no preemptible dentro de `rule.apply`). | Antes de `Pow::pow`, rechazar cuando `exp.unsigned_abs() > MAX_ABS_POW` (reusar la constante) o cuando `base.bits()*exp` excede una cota; devolver `None` para dejar la potencia simbólica. | S |
| S3 | `cas_math/src/root_forms.rs:2489` | **Trial division BigInt ilimitada** en `extract_root_factor` (`while &d*&d <= n_abs`): `1234...891^(1/2)` gira 10s+ CPU hasta kill. Caller vivo y caliente: `EvaluatePowerRule` en cada `Number^(p/q)`. Sin cota, sin gate de bits, sin poll de deadline. 4 loops hermanos iguales. | Early-out cuando `n.bits() > ~96-128` devolviendo `(1, n)` (surd sin factorizar = sound), o cap del divisor ~2^20. Aplicar a los 4 loops hermanos. Riesgo de huella ~0 (hoy nunca terminan). | S |
| S4 | `cas_math/src/number_theory_support.rs:484` (**re-atribuido** desde `cas_didactic/.../focused_rule_substeps.rs:8228`) | **Expansión de ratio factorial ilimitada**: `(n + 1099511627776)!/n!` cuelga con Y sin `--steps` (el verificador probó que el choke está en la regla del CORE, no en el didáctico). `gap = num_offset − den_offset` de offsets controlados por el usuario, `Vec::with_capacity(gap)` sin cota ni `checked_sub`. | Cap de `gap` (~64) con `checked_sub` antes de alocar/iterar, y/o poll del deadline dentro del loop. Corregir en el sitio del core (`number_theory_support.rs`) además del didáctico. | S |
| S5 | `web/server.py:994` | **Superficie web insegura**: bind `('', PORT)` = `0.0.0.0` (LAN), `Access-Control-Allow-Origin: *` sin auth (session_id elegido por cliente), `Content-Length` sin cap (buffer de GBs), `HTTPServer` no-threading manejando subprocesos de 8-120s (DoS de una request). | Bind por defecto `127.0.0.1` con opt-in env para `0.0.0.0`; quitar ACAO:* (UI same-origin) o allowlist; cap `Content-Length` (~2MB→413); `ThreadingHTTPServer`; mensaje de error genérico en vez de `str(e)`/stderr. | S |
| S6 | `web/server.py:443` | **OOM remoto `/api/import`**: `max_ref` viene del body sin cota, luego `for i in range(1, max_ref+1): append(...)`. ~60 bytes → ~10⁹ dicts (100+ GB). (Nota del verificador: el CORS es *red herring* aquí — el vector real es el bind LAN + no-auth.) | Clamp `max_ref = min(max_ref, len(results)+1000)` o rechazar `max_ref > len(results)*2`; saltar refs por encima del clamp en vez de rellenar. | S |

### 3.2 Rendimiento (P1 — en el loop más caliente del motor)

| # | Fichero:línea | Qué | Fix | Esf. |
|---|---|---|---|---|
| P1 | `cas_engine/src/engine/transform/mod.rs:336` | **Backstop de soundness por-nodo sin guard de cambio**: tras `apply_rules`, `rewrite_unsoundly_drops_nonfinite_in_domain` se llama SIN comprobar `result != input` — camina todo el subárbol (con zero-proofs multipoly) aunque no haya habido reescritura. Caso común (fixpoint) paga el precio completo. O(n·depth) por pase en el loop más caliente. Verificado por profiling. | Early-out de una línea: saltar el backstop cuando `result == expr_with_simplified_children` (solo puede disparar si hubo reescritura). Luego memoizar `expr_carries_undefined/nonfinite` por `ExprId`. Huella-safe. | S |
| P2 | `cas_math/src/arithmetic_cancel_support.rs:1770` | **`SubSelfToZeroRule`/`AddInverseRule` caminan el subárbol antes del check de forma**: primera línea `if expr_carries_nonfinite_or_undefined(...)` (walk completo) y solo después rechaza no-Sub/no-Add. Reglas GLOBALES (sin target) → corren en cada nodo de cada tipo. Otro O(n²)-por-pase. | Hoistear el check de forma: `match Expr::Sub`/add-inverse PRIMERO, luego el walk solo en nodos que matchean. Reordenar 2 líneas, huella-safe. | S |
| P3 | `cas_math/src/semantic_equality.rs:244` | **Clona el arena `Context` entero en cada comparación Div-vs-no-Div**: `div_add_common_factor_rewrite_matches` primera línea `let mut scratch = self.context.clone()` ANTES de los gates read-only baratos del callee. Dentro del matcher multiset O(n²) → O(pares × arena) de alocación. Convierte paths lentos de derive/solve en declines por budget. | Hoistear el gate de forma read-only (`as_div` + numerador Add/Sub, solo `&Context`) antes de clonar. Huella-safe (reordenación pura). | S |

*(S3 y S6 arriba también son perf/safety mixtos; contados una vez.)*

### 3.3 Acoplamiento y cohesión (P1 — más allá de los chokepoints B y D)

| # | Fichero:línea | Qué | Fix | Esf. |
|---|---|---|---|---|
| C1 | `cas_solver/src/derive_command.rs:605` | **`try_supported_derive_strategies_inner` = función de 1.333 líneas** (la siguiente del archivo: 251) que memoiza a mano 35 `Option<DeriveStageOutput>` locales sobre un match de 40 variantes con arms copy-paste; 9 variantes `SimplifyThen*` son clones combinatorios. `derive` es un frente de universalidad; cada familia nueva paga el impuesto de 6 sitios en lockstep. | Table-ification behavior-preserving: `DeriveStageCache` con `get(strategy, ||runner)`; arms genéricos sobre una tabla `fn runner(DeriveStrategy)->StageRunner` (las 4 con quirks quedan explícitas); modelar `SimplifyThenX` como `(pre, base)`. | M |
| C2 | `cas_solver/src/solve_backend_local.rs:6080` | **`solve_local_core` = dispatcher de 454 líneas con 43 llamadas `try_*` order-sensitive** cuyo orden load-bearing vive solo en comentarios (fuente documentada de wrong-answers pasados); archivo de 8242 líneas con 3 convenciones de firma distintas para los 40 handlers `try_*`. | Seams de movimiento puro primero (0-delta): helpers surd/trig-threshold (líneas 28-250) → `surd_guards.rs`; handlers de inecuación vs ecuación → 2 módulos hermanos; luego normalizar firmas a una forma y convertir la cadena en `&[Handler]` ordenado con doc por restricción de orden. | M |
| C3 | `cas_math/src/symbolic_integration_support.rs:2527` (ADJUSTED) | **Integrandos trig/hiperbólicos de potencia enumerados a mano por grado**: 58 funciones `trig_{tan,cot,sec,csc}_{3,4,5,6,8}`, wrappers byte-idénticos salvo el literal de grado, predicados `is_*_target` casi idénticos (5807-6027), arms de dispatcher por grado. (Ajuste: los impares 7/9 bare SÍ funcionan por otra ruta; el hueco es par y en formas afines.) La arquitectura que genera la familia recurrente "cubre el grado nombrado, pierde el hermano". | Extract-before-abstract: UN builder de reducción recursiva por familia (`∫tanⁿ = tanⁿ⁻¹/((n−1)a) − ∫tanⁿ⁻²`, ídem cot; sec/csc por partes) tras un predicado parametrizado por grado y un colector; los enumerados quedan como delegados finos (0-delta) y luego se retiran. Empezar por tan/cot (puros). | L |

### 3.4 Código muerto (P2/P3 — todos huella-safe al borrar)

| # | Fichero:línea | Qué | Fix | Esf. | Sev |
|---|---|---|---|---|---|
| D1 | `cas_solver_core/src/solve_runtime_pipeline_preflight_context_runtime.rs:8` | Módulo entero muerto: `build_solve_preflight_state_with_existing_condition_derivation_with_state` (14 genéricos), 0 callers en todo el repo. Lee como entry-point vivo y engaña. | Borrar el archivo + su `pub mod` en `lib.rs:133`. | S | P2 |
| D2 | `cas_engine/src/rules/trig_canonicalization.rs:221` | 6 reglas Pythagoras definidas pero **nunca registradas** (`register_pythagorean_identities` sin callers); 2 duplican reglas VIVAS del mismo nombre en `values_rules.rs`. Trampa "superseded ambas presentes" — el dev puede editar la copia muerta sin efecto. | Borrar los 6 `define_rule!` + `register_pythagorean_identities` + imports huérfanos. Huella-safe (nunca registradas). | S | P2 |
| D3 | `cas_solver/src/health_suite_runner.rs:9` | `run_suite` y `format_report` = wrappers superseded muertos (ambas capas con `#[allow(dead_code)]`); vivos son los `*_filtered`. Dos implementaciones report/runner paralelas. | Borrar `run_suite` (2 capas), `format_report` (2 capas), `render/basic.rs`, `push_basic_result_line`, y los `allow(dead_code)`. Mantener `render/header.rs::push_report_header` (compartido). | S | P2 |
| D4 | `cas_engine/src/profiler.rs:116` | `record_domain_assumption` sin callers → la sección "Domain assumptions used" del health report **nunca se renderiza** (observabilidad muerta que miente en auditorías de soundness). | Cablear la llamada donde se comiten assumptions de dominio, o borrar la fn + campo + agregación + reset + sección de report. Borrar es huella-safe. | M | P2 |
| D5 | `cas_math/src/multipoly/arithmetic.rs:200` | `mul_with_stats`/`div_exact_with_stats` (0 refs) **contradicen `BUDGET_POLICY.md`**, que los documenta como los hooks de charging de poly-ops. La doc afirma cobertura de budget que no existe (defensa DoS falsamente "completa"). Ajuste: `div_exact` vivo tampoco tiene param de budget. | Cablear los `*_with_stats` en los callers que deben cobrar, o borrar ambas fns y corregir las 2 filas de doc. | S | P2 |
| D6 ✅ **HECHO** (`ab57211ec`) | `cas_solver_core/src/predicate_proofs.rs` + `strategy_kernels.rs` | **Patrón hermano-superseded** (generador dominante de código muerto del crate): cada refactor "añade callback con default" deja la variante N-1 compilada sin llamar. Escaneo full-repo → **6 `pub fn` con 1 sola referencia (su def)**, todas hojas (0 cascada): las 3 de proofs + las 3 `solve_*_result_pipeline_with_item*` de kernels (superseded por `execute_*`). Borradas. Regla "nueva variante REEMPLAZA a la predecesora" adoptada. | S | P2 |
| D7 | `cas_engine/benches/mm_gcd.rs:1` | Benches "deshabilitados" comentando `[[bench]]`, pero autodiscovery los sigue compilando como targets inejecutables (`criterion_main!` bajo harness libtest = no-op silencioso). Pagan compilación en cada ciclo. | Borrar los 2 archivos bench (+ `common.rs` si huérfano) o `autobenches = false` + `[[bench]]` explícitos para los 3 reales. | S | P2 |
| D8 | `cas_math/src/distribution_guard_support.rs:412` (ADJUSTED) | Planificador de distribución-división duplicado; la copia de `guard_support` es **test-only**, superseded por la viva en `distribution_division_support.rs`. Trampa de divergencia (como el over-cancel de `DivAddSymmetricFactorRule`). | Borrar el cluster (más grande de lo reportado según verificador) y portar sus 2 tests a la fn viva. | S | P2 |
| D9 | `cas_solver/src/assignment_command/message.rs:35` (ADJUSTED) | `evaluate_let_assignment_command_message_with` muerta (0 callers incl. tests), superseded por `_with_context`; hermana `evaluate_assignment_command_message_with` es test-only. Ambas tras `#[allow(dead_code)]`. (Módulo privado → no rompe API externa.) | Borrar la muerta + re-exports; mover la test-only bajo `#[cfg(test)]`. | S | P3 |
| D10 | `cas_math/src/trig_pattern_detection.rs:1` (ADJUSTED) | `#![allow(dead_code)]` de módulo entero oculta cluster Pythagoras superseded (~260 líneas) + `println` comentados. `should_preserve_trig_function` (0 refs). (Ajuste: KEEP `collect_add_chain` y los 4 `is_*_squared` — sí usados por `pattern_scanner.rs`.) | Borrar el cluster muerto (lista corregida por verificador), quitar el `#![allow(dead_code)]`, mantener las 5 vivas. | M | P3 |
| D11 | `cas_formatter/src/latex_core.rs:1349` (ADJUSTED) | 4 structs renderer LaTeX superseded (~110 líneas): `SimpleLatexRenderer`/`HighlightedLatexRenderer`/`HintedLatexRenderer`/`FullLatexRenderer`, 0 refs externas; vivo es `PathHighlightedLatexRenderer` (23 sitios). Cada uno tiene equivalente vivo en `latex.rs`/`latex_highlight.rs`. Semántica de hint stale (produciría pasos mal-renderizados si alguien los toma). | Borrar los 4 structs + impls (líneas ~1344-1447). Sin callers → huella-safe. | S | P3 |
| D12 | `cas_engine/Cargo.toml:19` (ADJUSTED) | 4 deps sin usar (`regex`, `thiserror`, `rustc-hash`, `rayon` opcional colgante) confirmadas por `-Wunused_crate_dependencies`. `regex` es árbol transitivo pesado recompilado en cada build del crate de 192k. (Bonus del verificador: `regex` también sin usar en `cas_ast`.) | Borrar las 3 de `[dependencies]`; `parallel = ["cas_math/parallel"]` y quitar el `rayon` opcional. Verificar con build+test. También `cas_ast`. | S | P3 |

---

## 4. Plan de saneamiento propuesto (secuencia de ciclos acotados)

Orden: **(a) seguridad/robustez → (b) chokepoints que la campaña tocará → (c) extracciones de god-file por seams de bajo riesgo → (d) limpieza barata.** Un commit por ciclo, huella 0-delta donde aplica, extract-before-abstract.

**Bloque A — robustez (P1 seguridad, cada uno S salvo parser):**
1. **Guard de profundidad en el parser** (S1) — contador en `parse_expr/…` + cadena latex; corregir doc FFI. *Gate:* input `('*5000` devuelve `ParseError` en vez de abortar; workspace verde. *Desbloquea:* la campaña puede fuzzear solve/integrate sin abortar el proceso. **[M]**
2. **Cap de potencias/raíces** (S2+S3, mismo ciclo) — `MAX_ABS_POW` en `power_eval_support`, cap de bits en `extract_root_factor` (×4 loops). *Gate:* `2^100000000` y `…891^(1/2)` declinan/simbólico en <100ms; huella 0-delta (hoy no terminan). **[S]**
3. **Cap de ratio factorial** (S4) — en `number_theory_support.rs` (core) + didáctico, `checked_sub` + cap gap. *Gate:* `(n+2^40)!/n!` declina; equivfuzz sin regresión. **[S]**
4. **Endurecer `web/server.py`** (S5+S6) — bind localhost por defecto, cap Content-Length, clamp `max_ref`, ThreadingHTTPServer, error genérico. *Gate:* smoke web verde; probar los dos repros de DoS. **[S]**

**Bloque B — chokepoints (el corazón del saneamiento para la campaña):**
5. **Perf del loop de simplify** (P1+P2+P3 perf, un ciclo) — early-out `result==input` en el backstop; hoist de shape-check en Sub/Add; hoist del gate antes del clone en `semantic_equality`. *Gate:* huella 0-delta (resultados idénticos), equivfuzz sin nuevos hangs, profiling muestra caída. *Desbloquea:* ciclos de campaña más rápidos (el loop es el más caliente). **[S]**
6. **Consolidar comparadores de surds** (Chokepoint A) — `cas_math::root_forms` canónico; `solution_set.rs` y `solve_backend_local.rs` delegan. *Gate:* barrido adversarial de surds (patrón existente) 0 wrong; huella 0-delta en delegación pura; unificar drift valida por separado. *Desbloquea:* toda mejora futura de decidibilidad aterriza en 1 sitio. **[M]**
7. ✅ **HECHO** (`f19023842`) **Constantes de nombres de regla** (Chokepoint D, alcance recomendado) — `cas_solver_core::rule_names::RULE_*` con el vocabulario always-keep (10 nombres, incluye los 3 "Conservar * residual"); 49 refs de producción en 19 ficheros de 4 crates; tests/smokes conservan literales como anclas de contrato; `CHECK 11` blinda. Huella 0-delta. *Follow-up pendiente:* `StepCategory`/`RuleId` tipado en `Step` + el resto del vocabulario (nombres solo-visibles y es/en de `visible_rule_names`). **[M]**
8. ✅ **HECHO** (`de9e999dd`) **Barrido `pub(crate)` en `solve_outcome.rs`** (Chokepoint C, primer god-file) — 315 fns demotadas; el compilador reveló **121 muertas en cadenas** (el detector grep de D6 solo veía 3) + 160 tests que pineaban ese plumbing; todo borrado. El god file baja 21.4k→12.7k (−41%) y desde ahora el código muerto ahí es warning de compilación. *Repetir por god-file:* `orchestrator.rs` (42k), `arithmetic.rs` (39k), `symbolic_integration_support.rs` (30k), `focused_rule_substeps.rs` (23k)… **[M por fichero]**

**Bloque C — colapsar la torre de callbacks (Chokepoint B), una capa por ciclo:**
9. Borrar la capa pass-through muerta `solve_runtime_flow_isolation_dispatch_base.rs` + módulo `preflight` muerto (D1). *Gate:* huella 0-delta. **[S]**
10. Introducir `IsolationHooks` struct-of-callbacks y convertir la capa más interna, dejando firmas como adaptadores finos. *Gate:* huella 0-delta por capa. **[L, multi-ciclo]** *Desbloquea:* extender `solve()` deja de ser cirugía en escopeta.

**Bloque D — limpieza barata (borrados huella-safe, agrupables):**
11. Borrar código muerto confirmado (D2–D8) en 1-2 ciclos agrupados por crate. *Gate:* huella 0-delta, workspace verde. **[S]**
12. Borrar deps sin usar (D12) + renderers/assignment/trig muertos (D9–D11). *Gate:* build+test; `-Wunused_crate_dependencies` limpio. **[S]**

**Bloque E — cohesión de god-files (tras los chokepoints):**
13. Seams de `solve_backend_local.rs` (C2) — mover helpers surd/trig a `surd_guards.rs`, separar inecuación/ecuación. **[M]**
14. Table-ificar `try_supported_derive_strategies_inner` (C1) y reducción recursiva de potencias trig (C3). **[M/L]**

---

## 5. Lista NO-TOCAR (capa deliberada / residuales honestos)

- **La separación core/runtime `cas_solver_core` vs `cas_engine`/`cas_solver` es deliberada** (memoria: "core has no Simplifier → split fix by layer"). La torre de callbacks es el *cómo* accidental; la *separación* en sí es correcta — el fix es un context-struct, no fusionar crates.
- **`prove_positive`/`prove_nonzero` con fallback a `provable_const_sign` NO es duplicación** — es la capa de decisión de constantes deliberadamente estratificada (linear-surd rápido → superset transcendental). No colapsar; sí unificar los *comparadores* de surds bajo ella (Chokepoint A).
- **El hang de C5** (`diff((x+tan x)^2)`, oscilación inter-nodo expand↔factor, ~2.3% de formas trig) es no-terminación honesta conocida, con análisis completo en memoria y presupuesto en `engine-equivfuzz`. NO es objetivo de este saneamiento (fix de orquestación, superficie de huella enorme).
- **Los residuales honestos del audit** (formas no-elementales que declinan a propósito) son contratos: si un ciclo de saneamiento los rompe, es soundness, no limpieza.
- **`web/server_colab.py`**: está roto (2 breakages confirmados) y su propio docstring dice usar `server.py`. Borrarlo es la opción limpia — pero **confirmar con el usuario** antes (es un artefacto de deploy, decisión suya).

---

## 6. Apéndice — reclamaciones refutadas

**Ninguna de las 40 verificadas fue refutada** (0 REFUTED). El verificador adversarial (instruido a refutar por defecto) confirmó 33 y ajustó 7 — y varios ajustes *reforzaron* el hallazgo (torre de callbacks: 109 fns no 92; kernel de surd: 4 copias no 3; ratio factorial: el choke está en el core, no en el didáctico). La verificación de las ~231 restantes (P2/P3 en bruto) está pendiente para la Parte 2; hasta entonces se tratan como *no confirmadas*.

---

## 7. Apéndice — cobertura

- **30/30 escáneres completaron** (5 dimensiones × 6 grupos). Grupos: `cas_engine`, `cas_math`, `cas_solver_core`, `cas_solver` (por separado); crates de base (`cas_ast`/`cas_parser`/`cas_formatter`/`cas_api_models`/`cas_session*`/`cas_cli`/`cas_didactic`/`cas_android_ffi`); harness (`scripts/*.py`, `web/*`).
- **40/~131 hallazgos P1+P2 verificados** antes del corte de presupuesto. Todos los P1 salvo 3 quedaron verificados; el grueso pendiente es P2 dup/coupling/perf.
- **No cubierto en profundidad todavía (Parte 2):** verificación de los ~114 P2 y ~117 P3 en bruto; entre ellos, semillas prometedoras sin verificar: duplicación en `symbolic_integration_support.rs`/`limits_support.rs`, tablas trig repetidas (`trig_table.rs`), helpers CLI-runner/JSON repetidos en `scripts/*.py`, `ConditionSet` sort por `format!("{:?}")` (perf en tipo core), y varios de cohesión en `orchestrator.rs` (42k) y `arithmetic.rs` (39k).
- **Tests/benches** excluidos del scope de hallazgos salvo cuando código de producción resulta ser test-only.

---

## 8. Parte 2a — lote verificado (28 hallazgos: 13 CONF + 15 ADJ, 0 refutados)

Verificación adversarial de los 3 P1 pendientes + 25 P2 (sesgo safety/perf). **19 hallazgos nuevos distintos**, 8 corroboraciones de la Parte 1, y **1 upgrade** (`semantic_equality.rs`). Cero refutados.

### CHOKEPOINT E (nuevo) — Prefijos de error duplicados por re-wrapping en capas *(P2, safety, CONFIRMED ×6 sitios)*

**Qué:** `CasError::SolverError` ya hace `Display` como `"Solver error: {0}"` (`cas_solver_core/src/error_model.rs:19`), pero varios call sites vuelven a prepender el literal. Reproducido en vivo (debug y release): `eval "solve(a*x^2+b*x+c>0,x)"` → `"Solver error: Solver error: Inequalities with symbolic coefficients not yet supported"`. Sitios: `execute.rs:44` (`format!("Solver error: {error}")`), `eval/actions.rs:435`, y análogamente el parser (`error.rs:15` → `"Parse error: Parse error at ..."`). **El comando `envelope` produce prefijo TRIPLE.** Peor que cosmético: `actions.rs:435` además **corrompe el routing machine-readable** — el JSON devuelve `"kind":"InternalError","code":"E_INTERNAL"` en vez del código real del error.

**Por qué importa:** la campaña de universalidad añade familias que declinan con mensajes de error; el doble/triple prefijo y el `E_INTERNAL` espurio contaminan tanto la UX como los oráculos de los tests/harness que parsean el error.

**Fix (M):** elegir una capa dueña del prefijo. O quitar el texto de los `Display` (`error.rs:15`, `error_model.rs:19`), o que los callers (`execute.rs:44`, `actions.rs:435`, `eval_input/build.rs:18`, `envelope.rs:35`) propaguen el `CasError` sin re-prependar. **Cuidado (nota del verificador):** el borrado ingenuo del literal NO es incondicional — las variantes que NO se auto-prefijan sí necesitan contexto; matchear por variante. Corregir también el corruptor de routing en `actions.rs:435`.

### 8.1 Nuevos — seguridad/robustez

| # | Fichero:línea | Qué | Fix | Esf. | Sev |
|---|---|---|---|---|---|
| S7 | `cas_math/src/summation_support.rs:260` | **2ª ruta de potencia ilimitada** (hermana de S2/S3): `rational_pow_int` computa `num_traits::pow(base, n)` sin cota, con `n` = coeficiente de exponente del sumando controlado por el usuario, vía `extract_geometric_term`. (Ajuste: en release `n=10⁶` tarda ~5s, no cuelga infinito como en debug — pero sigue siendo un hang práctico y salta el time-budget.) | Guard de `|si|,|ti|` (cap pequeño) antes de materializar; devolver `None` (suma simbólica). Mismo patrón que S2. | S | P2 |
| S8 | `cas_engine/src/engine/transform/mod.rs:237` | **Escritura de fichero como efecto colateral de librería**: en cada depth-overflow (input normal de usuario) `cas_engine` hace `append` a la ruta fija world-shared `/tmp/cas_depth_overflow_expressions.log`. Reproducido: `diff((x+tan(x))^3,x)` añade 4-7 líneas; se acumulan cientos rápido (el flag `depth_overflow_warned` es per-transformer, no per-proceso). Hazard de symlink/colisión/disk-fill; una librería no debe escribir ficheros. | Quitar el write y dejar solo el `tracing::warn!` (ya emitido 3 líneas abajo), o gatear tras env var opt-in (patrón `CAS_TRACE_RULES`) con `tempfile` per-proceso, nunca ruta fija. | S | P2 |
| E (chokepoint) | `execute.rs:44`, `actions.rs:435`, `parser/error.rs:15` | Prefijos de error duplicados/triples + corrupción de `code` a `E_INTERNAL`. Ver chokepoint E arriba. | Una capa dueña del prefijo; match por variante. | M | P2 |

### 8.2 Nuevos — rendimiento

| # | Fichero:línea | Qué | Fix | Esf. | Sev |
|---|---|---|---|---|---|
| P4 | `cas_math/src/semantic_equality.rs:244` | **UPGRADE P3→P1** (re-verificado): clona el `Context` arena entero en cada comparación semántica Div-vs-no-Div del loop de reescritura, ANTES de los gates read-only. Ya estaba en §3.2 como P3; el verificador lo eleva a P1 por estar en el loop más caliente. Fix idéntico (hoist del gate). | Gate read-only (`source` es Div y numerador Add/Sub, solo `&Context`) antes de clonar; mejor, partir en fase read-only + fase build. | S | P1 |
| P5 | `cas_solver_core/src/ground_eval_runtime.rs:33` | **Fallback de decisión de signo clona el arena + reconstruye un Simplifier de ~340 reglas por invocación**, sin memoización de proofs. Microbench: 12.9ms@0 nodos → 88.8ms@50k. (Ajuste P1→P2: hay un factor mitigante — no es cada nodo, sino cada fallback de constante no-decidible por rutas baratas.) | (1) helper de ~40 líneas que re-interna solo el subárbol constante en un `Context` fresco en vez de clonar todo; (2) memoizar `Proof` por `ExprId` (thread_local). | M | P2 |
| P6 | `cas_solver_core/src/cycle_detection.rs:166` | **Fingerprint del hot-loop hashea cada `Number` con dos `to_string()` decimales de BigInt** y cada `Constant` con `format!("{:?}")`. Microbench: ~2× más lento vs hash por limbs. En el detector de ciclos que corre en cada nodo. | Hash del `BigRational` sin conversión decimal (FNV sobre `to_bytes_le()` de numer/denom); `Constant` por `discriminant` + tag, no Debug. Huella-safe (fingerprint interno). | S | P2 |
| P7 | `cas_solver_core/src/fingerprint.rs:5` | **Segundo `expr_fingerprint` sin memoizar** que duplica el de `cycle_detection.rs` y recorre el árbol DAG-desplegado en cada entrada de solve recursivo. Microbench (torre `Mul(A,A)`): unmemo 264ms vs memo 2µs a 26 nodos (explosión exponencial por DAG). | Borrar la traversal de `fingerprint.rs` y re-expresar `equation_fingerprint` sobre `cycle_detection::expr_fingerprint` con un `FingerprintMemo` local. | S | P3 |
| P8 | `cas_solver_core/src/domain_normalization.rs:1701/3678/4182` | **`normalize_and_dedupe_conditions` es O(n²)** con comparadores que (a) clonan el arena por PAR de condiciones, (b) corren `Polynomial::factor()` + `multipoly` sin cache per-ExprId dentro de los loops de pares (2 pasadas), (c) renderizan Strings de display por par. Bench: 42-79ms con 16-24 condiciones. (Ajuste a P3: es ruta de display/output una-vez-por-resultado, no el inner loop de rule-application; patológico solo a n grande.) | Memoizar `factor`/`multipoly` por `ExprId` durante la llamada; clave canónica por condición computada una vez; early-out sin NonZero. | S/M | P3 |

### 8.3 Nuevo — acoplamiento

| # | Fichero:línea | Qué | Fix | Esf. | Sev |
|---|---|---|---|---|---|
| C4 | `cas_engine/src/solve_core_runtime.rs:14` | **Dos pipelines públicos de solve divergidos**: `cas_engine::api::solve` usa un `solve_inner` byte-igual al de `cas_solver` PERO sin el guard periódico-trig (no puede llamarlo: `cas_solver` depende de `cas_engine`, no al revés). El commit `2753e6ce8` que arregló la unión periódica tocó la copia de `cas_solver` y NO ésta. **Divergencia real a nivel librería** (probado: `sin(x)*cos(x)=0` da el `{0,π/2}` incorrecto por esta ruta). (Ajuste P1→P3: la ruta NO es alcanzable por usuario — `EvalAction::Solve` nunca se construye, `eval_solve` es código muerto; el `solve(...)` de usuario va por la copia guardada de `cas_solver`. El shim `golden_corpus_tests.rs` valida la copia sin guardia pero el corpus no tiene cobertura trig-producto.) | (a) borrar/deprecar `cas_engine::api::solve` + el shim del golden-corpus (sin consumidor de producción), o (b) hoistear el pre-check periódico a `cas_solver_core::solve_inner` como `Option<hook>` compartido por ambos bridges. (a) es huella-safe. | S | P3 |

### 8.4 Corroboraciones (refuerzan la Parte 1)

Los siguientes re-verificaron hallazgos ya en la Parte 1, todos CONFIRMED, varios con reproducción en vivo adicional: **parser sin depth-guard** (S1; el verificador confirma abort a profundidad ~2000-3000 y el shim FFI), **potencia/raíz ilimitadas** (S2/S3), **ratio factorial** (S4), **`web/server.py` bind+CORS+DoS** (S5/S6), y **`web/server_colab.py` roto** (§5; `_coerce_domain_mode` devuelve None para todo dominio válido + crash de arranque `env_int`, confirmado por ejecución). Ninguna cambió de severidad a la baja de forma que invalide la Parte 1.

### 8.5 Ajuste al plan de saneamiento

Añadir al **Bloque A** (robustez): S7 (cap `rational_pow_int`, mismo ciclo que S2/S3), S8 (quitar el write de `/tmp` — S, trivial), y el **chokepoint E** (prefijos de error — M, alto valor porque corrige también el `code` corrupto que los oráculos leen). Añadir al **Bloque B** (perf del loop): P4 (upgrade, ya contemplado), P6 (fingerprint por limbs). P5/P7/P8 y C4 caben en el **Bloque D** (limpieza) como ciclos S/M independientes.

---

## 9. Parte 2b — lote verificado (35 P2 dup/coupling: 20 CONF + 15 ADJ, 0 refutados)

Este lote confirma que **la duplicación es el problema dominante de mantenibilidad del repo** y añade dos clusters de grado chokepoint. Ninguno refutado.

### CHOKEPOINT F (nuevo) — Colectores de factores duplicados 4-6× en `cas_math` *(P2, dup, CONFIRMED ×2)*

**Qué:** `factors_to_vec` es **byte-idéntica en 4 módulos** `div_*` de cancelación (`div_add_symmetric_factor_support.rs:79`, `div_add_common_factor_from_den_support.rs`, y 2 más); `collect_mul_factors` re-implementada 6×; `add_terms_signed` varias veces. **Esto es adyacente a la raíz de P0-G** (el bug del ciclo reciente estaba exactamente en un colector de factores que devolvía bases repetidas). Con 4-6 copias, el fix de P0-G podría no haberse propagado a todas.

**Por qué importa:** los colectores de factores son la maquinaria de cancelación de fracciones — territorio directo de la campaña de integración/simplificación. Una copia divergida = un wrong-answer latente de la forma ya vista.

**Fix (S, huella-safe):** mover `factors_to_vec` a `cas_math/src/fraction_factors.rs` como `pub(crate) fn merge_factor_multiset` (byte-idéntico → 0-delta); plegar las copias de `collect_mul_factors` en la canónica de `fraction_factors`; **extender `scripts/lint_no_duplicate_utils.sh`** para cubrir esta familia (ya designa homes canónicos para otras).

### CHOKEPOINT G (nuevo) — God file `solve_outcome.rs` (21.471 líneas) + wrappers combinatorios *(P2, coupling+dup, CONFIRMED ×4)*

**Qué:** 21.471 líneas, 436 ítems top-level, **320 `pub fn`** mezclando ≥8 responsabilidades del solver. **114 variantes `_with_existing_steps` + 71 `pipeline_with_item`**; ~**29 twins `_with`/`_with_state`** que duplican cuerpos enteros en vez de delegar (uno ya divergió estructuralmente). 264/320 `pub fn` sin uso externo (el 90% file-local del chokepoint C se concentra aquí). Es el equivalente solver-side de la torre de callbacks (B).

**Fix (L, extract-before-abstract, un cluster por ciclo, 0-delta):** extraer clusters contiguos a módulos hermanos (bloque pow-exponent ~1326-2193 → `solve_outcome_pow_exponent.rs`, etc.); reimplementar cada twin `_with` como wrapper fino sobre `_with_state`; borrar los 3+5 `pub fn` muertos + tests huérfanos; **este es el primer objetivo del barrido `pub(crate)` del chokepoint C**.

### 9.1 Clusters de duplicación confirmados (consolidar hacia un home canónico)

| Cluster | Copias | Home canónico propuesto | Esf. |
|---|---|---|---|
| `expr_contains_*` (builtin-traversal) | 3× en `cas_engine` **divergidas** (orchestrator salta Hold/Matrix; lista hiperbólica 3 vs 6) | `cas_ast::traversal` (ya es el home lint-designado); estandarizar en la variante con descenso Neg\|Hold\|Matrix (la de orchestrator es la peligrosa) | M |
| `format_trig_canonical_identity_desc` | 3 copias hand-synced, 2 con `unreachable!()` | método exhaustivo `impl TrigCanonicalIdentityKind::desc` en `cas_math` (sin `_` arm → compilador enforcea) | S |
| `rational_constant_value` | byte-idéntica en integration + differentiation god files, **ambas sin el guard de depth** | borrar ambas, llamar `crate::numeric_eval::as_rational_const` (convierte un posible stack overflow en `None` sound) | S |
| `root_forms::rational_bounds` | parálelo débil de `const_sign::const_value_bounds` (3 dígitos π/e vs 50 + sqrt) | reemplazar cuerpo por llamada a `const_value_bounds` (**amplía** el conjunto de comparaciones probables; es parte del chokepoint A) | S |
| Tablas de ángulos especiales | `trig_table.rs` vs `trig_values.rs`, **ya fuera de sync** | `trig_values.rs` como fact store canónico; `trig_table` como adaptador | M |
| `derive/*` helpers | 29 nombres definidos 2-8×, varios byte-idénticos + **31 twins "negados"** en `trig.rs`(18)+`hyperbolic.rs`(13) | mover a `derive/match_support.rs`; un combinador de adaptación de negación colapsa los 31 twins | M/L |
| Renderers LaTeX | `PathHighlightedLatexRenderer` re-implementa ~1000 líneas del pipeline; plain-text duplica el cluster reciprocal-sqrt, **con drift** | parametrizar el renderer con `Option<(&ExprPath, &PathHighlightConfig)>`; extraer detectores estructurales compartidos | L |
| Dedupe de condiciones | implementada 2× (wires estructurados vs re-parseo de strings de display) | el wire como única fuente; derivar display de los wires supervivientes | M |
| Inverse-fn by-parts skeleton | arctan/atanh/asinh/acosh copy-paste 4× (~500 líneas) | un `inverse_fn_affine_term` genérico + driver by-parts parametrizado | M |
| Twins `_with`/`_with_state` | ~29 en `solve_outcome.rs` (ver chokepoint G) | wrapper fino sobre `_with_state` | M |
| Strings de descripción didáctica | tabla paralela hand-synced entre crates (builders vs tabla i18n de 74 entradas) — **relacionado con chokepoint D** | test que feed-ea cada builder por `localize_solve_description` y falla en drift; largo plazo, payload estructurado | S |

### 9.2 Duplicación en el harness Python *(P2, dup, CONFIRMED ×3)*

- ✅ **HECHO** (`c61f4db38`) **`ensure_release_cas_cli` tiene una 2ª implementación más débil** en `engine_calculus_residual_probe_smoke.py:1115` → el smoke de residuales podía validar un **binario `cas_cli` STALE** (falsos verdes en el propio harness). Fix aplicado: `from cas_cli_release import ensure_release_cas_cli` como los otros 3 smokes.
- ✅ **HECHO** (`c61f4db38`) **Decoder del wire de warnings duplicado 4×**, la copia del residual-probe había **drift-eado** y descartaba warnings con forma rule/assumption. Consolidado en `engine_command_matrix_observability.py::extract_warning_messages` (los 4 smokes lo importan); `CHECK 10` del lint lo blinda. Verificado 730/730 residual en vivo (0 flips).
- ✅ **HECHO** (`37ec450e2`) **Plumbing CLI-runner/JSON** consolidado en `engine_smoke_common.py` (`parse_json` ×4→1, `terminate_process_group` ×5→1 — el scorecard conserva la suya a propósito, contrato distinto documentado —, `extract_cli_timings_us` ×2→1) + `CHECK 12`. Validado: 186 unit tests, residual-probe vivo 730/730. **§9.2 completo (3/3).**

### 9.3 Meta-hallazgo — el lint anti-duplicación existe pero está incompleto

`scripts/lint_no_duplicate_utils.sh` **ya designa homes canónicos** (CHECK 7 para `count_nodes*` → `cas_ast::traversal`) pero **no cubre** `expr_contains_*`, `factors_to_vec`, `collect_mul_factors`, ni los helpers de `derive/*`. Cada consolidación de arriba debería **añadir su check al lint** para que la duplicación no reaparezca — es la mayor palanca preventiva del saneamiento (convierte un fix puntual en un guardrail permanente, como `engine-equivfuzz` para soundness).

### 9.4 Ajuste al plan de saneamiento

- **Bloque B/chokepoints:** F (colectores de factores, S — hacer PRIMERO, es adyacente a P0-G y trivial) y el arranque de G (`solve_outcome.rs`, primer cluster) se suman al barrido `pub(crate)` del chokepoint C (§4 ciclo 8), que ahora tiene a `solve_outcome.rs` como objetivo #1 concreto.
- **Nuevo ciclo transversal (alto valor):** extender `lint_no_duplicate_utils.sh` con checks para cada familia consolidada (§9.3) — un ciclo S que blinda todo el resto.
- **Bloque D (limpieza):** los clusters de §9.1 son en su mayoría S/M huella-safe, agrupables por crate. El harness Python (§9.2) es un ciclo S independiente y **urgente** (el `ensure_release_cas_cli` stale compromete la fiabilidad del propio harness de validación que usa la campaña).

---

## 10. Parte 2c — lote final P2 (58 coupling/perf/dead: 34 CONF + 24 ADJ, 0 refutados) — TIER P2 CERRADO

Este lote completa la verificación de todo el tier P1+P2. Revela **el mapa completo de god files** y un **cluster de rendimiento en el hot loop** no visto en las partes anteriores.

### 10.1 Mapa completo de god files *(P2, cohesion, CONFIRMED)*

Todos comparten el mismo fix: **extract-before-abstract, mover clusters contiguos a módulos hermanos, pure helpers primero, un cluster por ciclo con huella 0-delta.** Varios ya tienen el precedente en el propio repo (módulos hermanos ya extraídos).

| Fichero | Líneas | Qué mezcla | Precedente / seam | Esf. |
|---|---|---|---|---|
| `cas_engine/src/orchestrator.rs` | **42.018** | ~25.9k de matchers-shortcut privados + pipeline 3.9k + **12.2k de tests inline** | mover tests fuera (0 riesgo), luego profiling helpers, luego shortcuts por familia | M |
| `cas_engine/src/rules/arithmetic.rs` | **39.182** | motor de probes de cancelación disfrazado de fichero de reglas; posee lifecycle de pipeline + ~8.5k tests | mover tests; extraer el probe-budget scope | M |
| `cas_math/src/symbolic_integration_support.rs` | **29.927** | ~90 familias de forma + dispatcher de 194 arms/1.077 líneas + 6.283 tests; `get_linear_coeffs` (93 call sites) enterrada en L23.529 | **7 módulos hermanos ya extraídos** (patrón probado); seguir con partial-fractions y log/by-parts | M |
| `cas_didactic/src/didactic/focused_rule_substeps.rs` | **23.181** (61% del crate) | 783 funciones en ~15 familias didácticas no relacionadas | `nested_fractions.rs`/`generic_rule_substeps.rs` ya son el precedente hermano | M |
| `cas_solver_core/src/solve_outcome.rs` | 21.471 | ver **chokepoint G** (§9) | — | L |
| `cas_math/src/limits_support.rs` | **17.517** | motor de límites tras un entry-point mal nombrado (`eval_limit_at_infinity` despacha finite/one-sided/∞); if-chain de 37 arms; 40% tests | split por secciones medidas; renombrar entry a `eval_limit` | M |
| `scripts/engine_improvement_scorecard.py` | **12.663** | 9 responsabilidades; `render_markdown` = 1 función de 3.705 líneas; `parse_algorithmic_backend_observability` = 2.073 líneas copy-paste; **shotgun surgery: 1 métrica nueva = ~37 ediciones coordinadas** | paquete `scripts/scorecard/`, helpers puros primero | L |
| `cas_solver/src/derive_command.rs` | 7.981 | 8 responsabilidades (ver también C1 §3.3) | módulos hermanos `derive/` ya muestran la estructura buscada | M |
| `cas_solver_core/src/domain_normalization.rs` | 10.099 | ver §8/§9 | split por las 5 pub-fn boundaries | M |

> **Nota de secuenciación:** `symbolic_integration_support.rs` y `limits_support.rs` son los **más urgentes** para la campaña — son literalmente los ficheros más calientes de integrate/limits, los frentes de universalidad. `root_forms.rs` (§9, chokepoint A) también: mezcla la capa soundness-crítica de signo de surds (455 líneas) con 3.635 de rewriting de presentación — extraer `surd_sign.rs` es S y prepara el chokepoint A.

### 10.2 Cluster de rendimiento nuevo (hot loop) *(P2, perf, CONFIRMED)*

Todos en el loop de rule-application / solve-dispatch. Van al **Bloque B** del plan.

| # | Fichero:línea | Qué | Fix | Esf. |
|---|---|---|---|---|
| P9 | `cas_math/src/prove_sign.rs:297` | **CONFIRMADA Y CORREGIDA (medida).** `prove_positive`/`prove_nonnegative` corrían AMBOS oráculos de constante en cada nodo de recursión, incl. subárboles con variable. **Regresión real que introduje en `1250a156e`**: bench per-nodo → el `const_sign` añadido cuesta 280 ns/nodo (×1.66 vs el surd pre-existente) en el caso común var-bearing, y **436 µs/nodo** en constantes transcendentales (serie atanh 50-díg). End-to-end en 10 exprs var realistas: **34.2 → 30.7 µs/call (~10%)**, peor con constantes transcendentales como subtérmino. | **HECHO** (pendiente de commit): gate `if !contains_variable(expr)` sobre ambos oráculos — behavior-idéntico (ambos devuelven `None` en var-bearing) y además mejora el baseline original. Workspace verde, huella GUARD/PRESS 0-delta, equivfuzz limpio, CLI probes idénticos. | S |
| P10 | `cas_math/src/const_sign.rs:48` | Sin caché en el oráculo de bounds: constantes π/e de 50 dígitos re-parseadas de string, serie atanh de ln(2) y Newton 1e-40 recomputados en cada llamada. | `OnceLock<(BigRational,BigRational)>` para π/e/φ/ln2/ln10; memo por `ExprId` en `compare_endpoints`. | S |
| P11 | `cas_math/src/expr_semantic_hash.rs:26` | `semantic_hash` sin memoización por `ExprId` (**exponencial en DAGs compartidos**) + `String` por cada `Number` vía `to_string()`. | Memoizar `u64` por `ExprId`; hashear numer/denom por palabras en vez de `to_string`. | S |
| P12 | `cas_engine/.../rule_application.rs:27` | `build_parent_context` deep-clona `ImplicitDomain` (un HashSet) **una vez por nodo visitado**. | `Rc<ImplicitDomain>` como ya hace `pattern_marks` (read-only durante el pase). | S |
| P13 | `cas_ast/src/expression.rs:405` | Cada canonicalización de `Mul` en `Context::add` camina ambos subárboles (unmemoized) para probar conmutatividad. | Bit `contains_noncommutative` por nodo en creación (O(1) bottom-up). | M |
| P14 | `cas_math/src/expr_complexity.rs:23` | El budget anti-worsen cuenta ambos árboles por expansión completa (sin dedup, sin early-exit) en cada rewrite aceptado — exponencial en DAGs. | Contar `before` una vez, `after` con early-exit al superar el umbral; visited-set. | S |
| P15 | `.../rationalize.rs:233`, `derive/solve_prep.rs:340` | `Simplifier::with_default_rules()` **re-registra las 337 reglas por probe/llamada** (57 call sites en solve_prep, 9 seguidos en sitios). | Registro por defecto una vez (`LazyLock`/`thread_local`, reglas son `Arc<dyn Rule>` stateless); `mem::swap` del Context. | M |
| P16 | `solve_backend_local.rs:6131` | `solve_local_core` corre el pipeline de simplify completo sobre el MISMO `diff`/`lhs`/`rhs` interned **4-8 veces por solve** (cada handler re-simplifica). | Simplificar una vez arriba y pasar los simplificados a los handlers. | M |
| P17 | `step_payloads/build/expr.rs:176` | `render_human_expr` clona el arena `Context` entero por string renderizado; **3-6 clones de arena por step-wire**. | Un `Context` scratch por timeline, enhebrado `&mut`. | M |

### 10.3 Chokepoint D a escala real *(P2, coupling, CONFIRMED)*

El contrato stringly-typed de nombres de regla es **mayor de lo que la Parte 1 estimó**: confirmado cruzando **5 crates + `web/server.py` + el harness Python**. Sitios adicionales: `actions.rs:540`, `visible_rule_names.rs:606` (doble pivote de strings engine→es→en), `localization.rs:121` (tabla de 74 entradas que duplica los strings del productor por reverse-parsing), `server.py:725` (el web filtra pasos didácticos por display-name), `eval.rs:68` (`is_known_eval_engine_function` hard-codea ~90 name/arity de otro crate). Refuerza el fix del chokepoint D (constantes tipadas) y lo eleva en prioridad.

### 10.4 Código muerto adicional *(P2, dead, CONFIRMED — todos huella-safe)*

- `cas_solver/src/session_api/`: **217 de 401 nombres exportados sin consumidor** en ningún crate/web/script (borrar los `pub use`). **[M]**
- `cas_solver_core/src/isolation_power.rs`: **22 pub fns totalmente muertas** (una de ~121 líneas / ~20 genéricos). **[M]**
- `cas_solver_core/src/isolation_arithmetic.rs`: 155 pub items referenciados solo desde su propio `#[cfg(test)]`. **[M]**
- `cas_didactic/src/step_payload_render/path.rs` (+ hermano): **2 ficheros huérfanos nunca compilados** (sin `mod`); `git rm` es bit-idéntico. **[S]**
- `cas_api_models/src/wire_types.rs`: `OutputEnvelope V1` anuncia un `steps` que ningún path puebla; 2 contratos wire muertos (script, mm-gcd-modp) + API substitute-mode muerta. **[S/M]**
- `Makefile:236`: `test_engine_combination_ledger_tool.py` existe pero **ningún harness lo ejecuta** (pasa hoy; añadirlo es huella-safe). **[S]**
- `cas_solver/src/lib.rs:177`: `#[allow(dead_code)]` de módulo oculta ~135 líneas de diagnósticos test-only. **[S]**

### 10.5 Ajuste final al plan de saneamiento

- **⚠️ Verificar P9 primero (posible regresión propia):** confirmar si el doble-oráculo de `prove_sign` degradó el hot path desde el ciclo de condicionales vacuos; si sí, es un fix S de alto valor (gate var-free) que además es soundness-neutral.
- **Bloque B (perf) crece a un ciclo sustancial:** P9-P17 son casi todos S, en el loop más caliente; agrupables en 2-3 ciclos por crate. Junto con P1-P8 (§3.2, §8) forman el "ciclo de perf del hot loop" de mayor ROI para acelerar la propia campaña.
- **God files (Bloque E):** priorizar `symbolic_integration_support.rs`, `limits_support.rs` y `root_forms.rs` (surd_sign) — son los ficheros que la campaña de universalidad tocará en cada ciclo. `orchestrator.rs`/`arithmetic.rs`: empezar por mover los ~20k líneas de tests inline fuera (0 riesgo, gran alivio de navegación).
- **Harness Python (urgente, S):** el `ensure_release_cas_cli` stale (§9.2) + el scorecard god file + shotgun surgery de métricas comprometen la fiabilidad del propio harness que valida la campaña.
- **Chokepoint D subió de prioridad** (5 crates + web + python); hacerlo antes de la campaña es/en evita RED por renombrado.

---

## Estado de ejecución (2026-07-02)

**9 ciclos completados, todos con workspace verde (12511-12512 / 0), clippy limpio, huella GUARD/PRESS 0-delta, y equivfuzz de soundness limpio donde aplica.** Los chokepoints de duplicación se blindaron con checks nuevos del lint anti-duplicación (7 → 9).

| Commit | Ciclo | Qué | Medido |
|---|---|---|---|
| `93cb5e2bd` | P9 | Gate `!contains_variable` en `prove_sign` (regresión propia de `1250a156e`) | 34.2→30.7 µs/call (~10%) |
| `5fffc1a15` | **F** | `factors_to_vec`/`find_factor_exp` ×4→1 en `fraction_factors` + `CHECK 8` | — |
| `e0067e13e` | P10 | Caché `OnceLock` pi/e/ln2/ln10 en `const_sign` | 1662→884 ns/nodo (1.88×) |
| `947b93f81` | **A1** | `sign_of_linear_surd` ×3→1 en `root_forms` + `CHECK 9` | 120 surds, 0 wrong |
| `73b4e78e0` | P12+P1 | `Rc<ImplicitDomain>` + early-out del backstop en fixpoint | — |
| `fbc9f3d21` | P2+P3 | Hoisting de shape-check antes de walk/clone | — |
| `d2b1c7ce9` | P14 | Early-exit del guard anti-worsen (`count_all_nodes_capped`) | — |
| `45f9d0869` | P11 | `semantic_hash` sin `to_string`/`format!` (hash directo) | — |
| `c0232cd3e` | P15 | Registro de reglas por `thread_local` (no reconstruir ~340/llamada) | 35.6→2.07 µs (17×) |
| `38a86d451` | **A2** | 4 comparadores exactos surds/nth-roots → `root_forms`; `CHECK 9` a 6 nombres | 72 formas adversariales, 0 wrong |
| `c61f4db38` | **infra** | Harness Python §9.2: `ensure_release_cas_cli` stale + decoder warnings ×4→1 + `CHECK 10` | 730/730 residual, 92 tests |
| `ab57211ec` | **D6** | 6 `pub fn` muertas sibling-superseded borradas (`predicate_proofs`+`strategy_kernels`); 0 cascada | 12512/0, huella 0-delta |
| `3931e3dca` | **D6+** | Detector aplicado a TODO `cas_solver_core`: 13 `pub fn` muertas más (~357 líneas) + 1 fichero-módulo vacío borrado; 0 cascada | 12512/0, huella 0-delta |
| `f19023842` | **D (parcial)** | Vocabulario always-keep (10 nombres) → `cas_solver_core::rule_names::RULE_*`; 49 refs de producción en 19 ficheros de 4 crates; tests/smokes conservan literales como anclas; `CHECK 11` | 12512/0, huella 0-delta, lint 11/11 |
| `de9e999dd` | **C ciclo 1** | `solve_outcome.rs`: 315 `pub`→`pub(crate)` (2 quedan pub) → el compilador marca **121 fns muertas** (cadenas que el grep no veía) → borradas + sus 160 tests-de-plumbing-muerto. **21.4k→12.7k líneas (−41%)** | 12352/0 (delta = exactamente los 160 tests), huella 0-delta |
| `96bf5ee82` | **C ciclo 2** | Crate-wide: 704 fns demotadas (138 quedan pub) + **28 re-exports de las fachadas flow demotados** → rustc ve a través de la torre: **146 fns muertas en 30 ficheros** (incl. la capa pass-through de B/D1) + 198 tests que las pineaban. **−10.0k líneas en 129 ficheros**; el crate queda visibility-honest de punta a punta | 12154/0 (delta = exactamente los 198 tests), huella 0-delta |
| `84276b747` | **C ciclo 3** | `cas_math` (primer crate de DOMINIO, con paso de clasificación): 483 demotadas (683 pub) → 29 muertas → **3 KEPT** (predicados `integrate_symbolic_is_*_target`, capacidad sin cablear, pub+NOTE) + **26 borradas** (capa plan/build de distribution_guard — guards vivos —, wrappers/collectores superseded, Clase B §11) + 29 tests | 12125/0 (delta exacto), huella 0-delta, equivfuzz 120/0 |
| `511bee94a` | **C ciclo 4** | `cas_engine`: 80 fns top-level demotadas (70 pub) → 1 sola muerta (`register_pythagorean_identities`, propiedad de la tarea paralela — no tocada). Grafo top-level del engine completamente vivo. Los impl-methods (orchestrator/arithmetic) = pasada aparte | 12125/0, huella 0-delta |
| `9f293a9f7` | **C ciclo 5** (redo exitoso tras un intento abortado) | Los 7 crates restantes: 318 fns demotadas + **56 fn-sites re-promocionados por la invariante de alias** (la familia `*_on_repl_core` que cas_cli consume). Con el **workspace completo** en el loop, el conjunto muerto real es **18 fns** (no las ~199 de la vista lib-only — ¡180 estaban VIVAS vía session_api!): generaciones wire superseded (autoexpand/config/context/health/substitute/unary/weierstrass), utils de path de cas_ast, y la capa strategy local del substitute. + sus 51 tests (reconciliados **nombre a nombre** contra un worktree HEAD con `--list`) y 2 de cas_ast | 12072/0 (delta −53 exacto por nombre), clippy -D warnings workspace, huella 0-delta, baselines byte-idénticos |

| *(ABORTADO y revertido)* | **C ciclo 6** (impl-methods) | Demote de 226 métodos inherentes (9 crates) → 42 lib-dead. **Abortado al descubrir la frontera de la política**: los accessors de tipos de dominio vivos (`is_present`, `surd_count`, `allows_level`, `failure_class`…) son lib-dead pero **test-alive — son API de test**, vehículos de aserción de tests cuyo sujeto es comportamiento vivo (backend de integración, SurdSumView, RationalizePolicy). Borrarlos + sus tests perdía cobertura real (detectado en `general_integration_backend/tests.rs`: 4 tests de comportamiento + 71 helpers huérfanos). El botín genuinamente muerto era mínimo (~15 variantes superseded, ~150 líneas). *Regla para un futuro redo:* muerto-en-lib + usado-en-tests ⇒ KEEP (o `cfg_attr(not(test), allow(dead_code))`); borrar solo si sin uso en lib Y tests. Hallazgo aparte (chip): `record_domain_assumption` = observabilidad sin cablear (el health report muestra domain-assumptions siempre 0) | revertido a `9f293a9f7`; build 0 err |

| `d15161e5f` | **pitagóricas** (sesión paralela) | `register_pythagorean_identities` + 6 reglas huérfanas borradas (caso-b redundante: cubiertas por reglas vivas — veredicto verificado empíricamente con las 6 formas también en esta sesión) + test de contrato (6 tests) | 12078/0 |
| `7a9088e61` | **fix ci** | Ancla del lint presimplify visibility-agnostic (rota por el demote correcto del ciclo 3) | 18/18 lints |
| `87ca94ba6` | **telemetría** | `record_domain_assumption` cableado end-to-end — eran **3 eslabones rotos**: producer (por assumption_event en ambos sitios de aplicación), `health on` nunca llegaba al profiler del engine (`set_health_tracking`), y los rebuilds de semantics perdían el flag. Verificado vivo: `Domain assumptions used: 1 (Abs Under Positivity / Cancel Identical N/D)` — la sección se renderiza por primera vez | 12078/0, huella 0-delta, clippy -D |
| `1d8174069` | **C ciclo 7** | Redo seguro de impl-methods con política **def-only** (nombre con exactamente 1 aparición en el repo = su def): 28 métodos borrados en 18 ficheros, **0 tests tocados** — los accessors test-API quedan excluidos por construcción | 12078/0 (delta 0 exacto), huella 0-delta |
| `6181b766a` | **P16** | Memo de `simplify()` scoped al solve (hasta **85% de llamadas redundantes** medidas): key (expr, sticky_root), replay de los 3 side-channels `last_*`, invalidación en sticky, off con step_listener. **−23% a −38% por solve típico** (x²−5x+6: 60→37ms; corpus 8 eq: −13%) | 12078/0, huella 0-delta, **equivfuzz 120/0** |
| `641db00c0` | **P17** | `render_step_wire_exprs` reutiliza un scratch `Context` (antes 3-5 clones de arena por step). Sin delta medible en CLI one-shot (arenas pequeñas); el win escala con arenas de sesión largas — anotado honestamente | 12078/0, huella 0-delta |
| `a0f222e0d` | **C ciclo 9** | **36 tipos def-only borrados** (compañeros huérfanos de las fns de ciclos 1-3 + 2 contratos wire muertos `MmGcdModpWireOutput`/`ScriptWireOutput`). KEPT: `LimitValueKind` (anotado "Reserved for V2 composition" — scaffolding declarado de la capability de límites) | 12078/0, huella 0-delta |
| `720eaae94` | **S8** | cas_engine ya no escribe `/tmp/cas_depth_overflow_expressions.log` en cada depth-overflow (hazard symlink/colisión/disk-fill; una librería no escribe ficheros). El `tracing::warn!` conserva el payload. Verificado con el repro `diff((x+tan(x))^3,x)` | 12078/0, huella 0-delta |
| `4d14f5d25` | **Chokepoint E** | Prefijos de error: **una capa dueña** — de-dup variant-aware en solver (execute.rs/actions.rs), `parse_error_message()` canónico para los 6 wrappers parse, y el clasificador wire enruta `Solver error:`→**E_SOLVER** (antes corrompía a E_INTERNAL) y reconoce la forma con span de parse. 4 repros verificados en release; 1 test recontractado (pineaba el doble prefijo) + 1 test nuevo del owner | 12079/0 (+1 exacto), huella 0-delta |

**P13 — CERRADO por falsación empírica (2026-07-03):** se implementó el bit `contains_noncommutative` por nodo completo (bottom-up en los 3 push-sites incl. restore de snapshots, con oracle `debug_assert` bit==walk, 305 tests verdes) y se midió A/B en release: **0% de diferencia** tanto en el corpus típico (271.6 vs 275.1 ms) como en cargas patológicas de construcción (`expand((a+b+c+d)^8)`, `(x²+x+1)^12`: 1224.8 vs 1225.8 ms). La hipótesis del audit queda falsada: el walk corta rápido en operandos ya aplanados y el coste de construcción lo dominan BigRational + sort + interning-hash. El parche se **revirtió** (un Vec paralelo + invariante mantenible por 0% medido no se justifica). El cluster de perf P1-P17 queda **completamente cerrado**.

**Corrección al hallazgo del audit "session_api 217/401 muertas":** con linkage completo (bins+FFI), la mayor parte de esa capa está **viva** a través de los aliases `*_on_repl_core` que consume el REPL de cas_cli. El muerto real a nivel de fn en los crates wire era 18. La cifra del audit medía otra cosa (probablemente accesibilidad de la API pública, no reachability).
| `bd19e28d0` | **higiene** | `PAR_DEPTH`/`PAR_TERM_THRESHOLD` gated bajo `#[cfg(feature="parallel")]` (build por defecto sin warnings) | 0 warnings ambas features |

**Chokepoint A — CERRADO** (A1+A2): `sign_of_linear_surd`, `sign_of_sum_two_surds`, `cmp_rational_to_quadratic_surd`, `cmp_rational_to_nth_root`, `compare_positive_nth_roots` viven todos en `cas_math::root_forms`, blindados por `CHECK 9`. **Residual honesto:** los decompositores Context-level (`as_surd_value`/`bound_surd` con φ, `as_nth_root_value`/`bound_nth_root`) siguen en sus ficheros — NO son duplicados literales (convenciones distintas `(sign,q,n)` vs `(q,n,neg)`; ambos manejan φ), unificarlos aporta poco frente a su riesgo.

**Pendientes — requieren tratamiento individual (mayor riesgo/superficie, NO quick-wins):**
- **P13** (`Context::add` camina subárboles para probar conmutatividad de `Mul`): toca el core de creación de nodos; necesita un bit por nodo.
- **P16** (`solve_local_core` simplifica el mismo `diff` 4-8×): threading de firmas por múltiples handlers.
- **P17** (`render_human_expr` clona el arena por string): ruta didáctica, scratch `Context` enhebrado.
- **A2-decompositores** (residual honesto, arriba): baja prioridad.

## 11. Hallazgo nuevo (2026-07-02): `pub fn` muertas en crates core — NO borrar como limpieza

Aplicando el detector de D6 (`pub fn` cuya única referencia en todo el repo es su
propia definición) a los crates core, `cas_solver_core` dio 19 muertas **obsoletas**
(borradas, `ab57211ec`+`3931e3dca`). Pero `cas_engine` (3) y `cas_math` (6) dan
candidatas **semánticamente cargadas** — NO son variantes-predecesoras mecánicas,
sino detectores/registradores que pueden ser **capacidad a medio cablear**. Se
dejan intactas; clasificarlas es trabajo de la campaña de universalidad, no de
saneamiento mecánico.

**Clase A — CAPACIDAD LATENTE (evaluar wire-up en universalidad, NO borrar):**
- `cas_engine/rules/trig_canonicalization.rs:221` **`register_pythagorean_identities`** — registra 6 reglas pitagóricas pero **nunca se llama**. `SecTanPythagoreanRule`/`CscCotPythagoreanRule` se registran además vía `sum_to_product_rules.rs:165-166` (viva), pero **`TanToSecPythagoreanRule`, `CotToCscPythagoreanRule`, `SecTanMinusOneIdentityRule`, `CscCotMinusOneIdentityRule` SOLO aquí** → esas 4 identidades **nunca se registran**. Gap de capacidad real (o desactivación deliberada sin documentar). El comentario dice "HIGHEST PRIORITY / must fire BEFORE conversion".
- `cas_math/symbolic_integration_support.rs:5913` **`integrate_symbolic_is_csc_third_affine_target`** — predicado csc³-afín muerto mientras sus hermanos csc⁴/⁶/⁸ (`:5947/:5975/:6003`) y el handler `trig_csc_third_affine_antiderivative:3553` existen. Asimetría sospechosa.
- `cas_math/symbolic_integration_support.rs:15474` **`integrate_symbolic_is_polynomial_times_constant_base_power_target`** — predicado muerto, pero el handler `polynomial_times_constant_base_power_antiderivative:15392` y su test (`:24345`) existen → probablemente detector superseded (el otro predicado sí alcanza el handler); verificar antes de borrar.

**Clase B — utilidades/diagnósticos huérfanos (borrables en un ciclo cuidadoso, bajo valor):**
- `cas_engine/recursion_guard.rs:91` `get_all_max_depths` (diagnóstico), `:102` `with_stack` (utilidad de stack profundo, con doc-example, 0 llamadores).
- `cas_math/limits_support.rs:112` `mk_inf` (constructor), `poly_store.rs:651` `try_render_poly_result_latex` (render no cableado), `fraction_add_rule_support.rs:141` `is_inside_trig_ancestor_with` (helper `_with` sin hermano), `trig_pattern_detection.rs:263` `should_preserve_trig_function` (predicado).

**Lección de proceso:** el detector "0 referencias" es sólido para variantes-predecesoras mecánicas (crate de plumbing como `cas_solver_core`), pero en crates de dominio (`cas_engine`/`cas_math`) una `pub fn` sin llamadores puede ser una feature a medio construir. Distinguir requiere leer el dominio, no solo contar referencias.

## Cierre

**Verificación adversarial completa del tier P1+P2: 161 verificaciones, 0 refutados, ~120 hallazgos distintos.** El código es fundamentalmente sólido (0 `unsafe`, clippy limpio, 0 wrong-answers P0 abiertos) pero carga deuda concentrada en **7 chokepoints (A-G)** y **9 god files**, más un cluster de ~17 items de rendimiento en el hot loop y varios riesgos de robustez alcanzables por input. El orden recomendado no cambia: **robustez → chokepoints → perf del hot loop → god files → limpieza**, cada uno como ciclo acotado con huella 0-delta, y **extendiendo `lint_no_duplicate_utils.sh`** en cada consolidación para que la deuda no reaparezca. Los ~113 P3 quedan catalogados en `docs/.saneamiento_pendiente_raw.json` para verificar bajo demanda.

*Generado con [Claude Code](https://claude.com/claude-code) — auditoría multi-agente, verificación adversarial. Partes 1 + 2a + 2b + 2c completas (tier P1+P2 cerrado).*
