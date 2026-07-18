# Fase 2 · Frente Complejo elemental (principal-branch): scoping en sub-ciclos acotados

- **Fecha:** 2026-07-16
- **HEAD:** `e069031f2`
- **Clase:** M total (dos evaluaciones de audit lo estimaron ≈M sin reescritura fundamental). Se entra como **secuencia de sub-ciclos acotados**, nunca como un solo ciclo.
- **Método:** scoping workflow READ-ONLY (6 mappers de subsistema convergentes + síntesis + **doble verificación adversarial**: verificador de anclas `file:line` en vivo + crítico de completitud). 9 agentes, 0 errores. Toda ancla verificada contra el árbol y toda fila de frontera re-sondeada con el binario fresco. Journal crudo: `subagents/workflows/wf_b05c3772-7f9/journal.jsonl`.
- **Relacionado:** `docs/CALCULUS_ENGINE_DEVELOPMENT_PHASES.md` (Fase 2, líneas ~176‑201 — los tres items del frente), `docs/ENGINE_VS_SYMPY_ASSESSMENT_2026-07-14.md` (fila F12: `solve(x^2+1=0)` complejo = el gap emblemático), `docs/G1_RATIONAL_INTEGRATION_SCOPING.md` (molde de formato: C-i/E-i = "primitivo duro reusable, huella byte-idéntica, antes de tocar lo caro").

Abrir este frente gradúa la mitad **compleja** de la Fase 2 (la otra mitad, vectorial multivariable, es un frente separado). Modelo: **single-valued PRINCIPAL BRANCH** (no Riemann, no multivaluado — eso está fuera del norte para siempre).

---

## La frontera exacta (probes verificados en vivo, HEAD `e069031f2`)
*(re-audit 2026-07-18: TABLA CERRADA — los 18 probes verdes (criterio de 'frente cerrado' de §Cómo ejecutar cumplido); además graduaron los residuales post-cierre: (z)^(-1), unimodularidad, trig-de-i, C2)*

> Cada fila re-sondeada con `./target/release/cas_cli eval "…" --value-domain complex`. Los ✅ son la máquina reusable; cada ❌ lleva su **punto de decline** exacto.

| Input `--value-domain complex` | Estado | Punto de decline |
|---|---|---|
| `(3+4i)/(1-2i)` | ✅ `2i-1` | `GaussianDivRule` `complex.rs:75` → `try_rewrite_gaussian_div_expr` `complex_support.rs:284` |
| `i^3` | ✅ `-i` | `ImaginaryPowerRule` `complex.rs:27` → `complex_support.rs:154` |
| `sqrt(-4)` | ✅ `2i` | `SqrtNegativeRule` `complex.rs:88` → `negative_abs_to_i_sqrt` `complex_support.rs:355` |
| `(-8)^(1/3)` | ✅ `1+i·√3` (principal) | `ComplexNegativeBaseRootRule` `power_rules.rs:402` (gate `:420`, emite `cos+i·sin` `:435‑448`) |
| **`(1+i)^2`** | ❌ `(1+i)^2` (no expande) | `extract_gaussian` **sin brazo `Pow`** (catch-all `_ => None` `complex_support.rs:149`); no existe `GaussianPowRule` en `register()` `complex.rs:116‑124` |
| **`abs(3+4i)`** | ❌ `\|3+4·i\|` (debe `5`) | `abs_support.rs:299` (`return None`, sin rama módulo Gaussiano) + eval numérico rechaza `i` `evaluator_f64.rs:534` |
| **`conjugate/Re/Im(3+4i)`** | ❌ "función no definida" | no está en `BuiltinFn` `cas_ast/src/builtin.rs`; corta en `is_known_eval_engine_function` `cas_session_core/src/eval.rs:68` |
| **`Arg(i)` / `Arg(-1)`** | ❌ "función no definida" | net-new **y** `atan2` transcendental → depende de la red numérica (B1) |
| **`solve(x^2+1, x)`** | ❌ `No solution` (debe `{i,-i}`) | `solution_set.rs:527‑529` `quadratic_numeric_solution`: `Δ<0 ∧ RelOp::Eq → SolutionSet::Empty`; **nunca usa `r1/r2` ya construidos** en `quadratic_formula.rs` |
| **`solve(x^2+x+1, x)` / `x^2-2x+5`** | ❌ `No solution` | misma rama `Δ<0 ∧ Eq → Empty` |
| **`solve(x^4-1, x)`** | ❌ `{-1,1}` (**dropea `±i` en silencio**) | `rational_roots.rs:961‑962` `solve_residual_degree_leq_two`: raíces racionales peladas, residual `x^2+1` con `Δ<0 → else { vec![] }` |
| **`solve(x^3-1, x)`** | ❌ `{1}` (dropea par conjugado) | `rational_roots.rs:961‑962` (residual `x^2+x+1`, `Δ=-3<0 → vec![]`) |
| **`e^(i*pi)`** | ❌ `e^(pi·i)` (debe `-1`) | no existe `EulerRule`; nada consume `Pow(E, i·θ)` |
| **`exp(i*x)`** | ❌ `e^(i·x)` | igual — **y** `exp(iθ)` queda `Function(exp,·)` (no `Pow`), ver chokepoint del bloque B |
| **`ln(-1)` / `ln(-2)`** | ❌ `undefined` (debe `i·π` / `ln 2 + i·π`) | `logarithm_inverse_support.rs:375` (`n<0 → Undefined`, **incondicional, sin value_domain**), vía `EvaluateLogRule` `logarithms/mod.rs:52` (`define_rule!` corto, sin `parent_ctx`) |
| **`ln(i)`** | ❌ `ln(i)` simbólico (debe `i·π/2`) | `try_rewrite_evaluate_log_expr` requiere `arg==Number` (`logarithm_inverse_support.rs:319`) |
| **`i^i` / `2^i`** | ❌ sin evaluar | ningún handler `z^w` complejo general (`z^w = e^(w·Log z)` → depende de Euler+Log) |
| **`approx`/`abs(i)` numérico** | ❌ sin valor | **CHOKEPOINT numérico**: `evaluator_f64.rs:534` `Constant::I => None`, `:187` `Err(Domain)`, `:165` gate `Pow`; `numeric_eval.rs:440/:540` |

---

## Arquitectura: la máquina reusable y los tres chokepoints

### Lo que ya existe (verificado)

1. **Eje `ValueDomain` enhebrado END-TO-END y probado.** `cli_args.rs:338` (flag, default `Real`) → `commands/eval.rs:132` → **junción maestra** `eval_option_axes/apply.rs:158` (`EvalValueDomain → ComplexEnabled` en `opts.shared.semantics.value_domain`) → `parent_context.rs:248` `parent_ctx.value_domain()`. Enum canónico `cas_solver_core/src/value_domain.rs:5` (`RealOnly` `#[default]` / `ComplexEnabled`). ⚠️ **Enum PARALELO** `abs_support.rs:85` `ValueDomainMode` — no confundir.
2. **Tipo exacto ℚ[i].** `complex_support.rs:9` `GaussianRational { real, imag: BigRational }` — **struct PASIVA**: solo `new/is_real/is_pure_imag/to_expr`; **CERO ops aritméticas como método** (toda la C-álgebra vive INLINE en las free-fns `try_rewrite_*`). `extract_gaussian` `:89` capta `Number` / `Constant::I` / `Neg` / `Mul(Number·I)` / `Add` / `Sub`; **pierde `Pow`, `Div`, y `Mul` de dos gaussianos**. `to_expr` `:28` materializa `a+bi`. Átomo dato único = `Expr::Constant(Constant::I)` (`expression.rs:56`).
3. **7 reglas Gaussianas** `rules/complex.rs`, cada una **auto-gateada** `if parent_ctx.value_domain()==RealOnly { return None }` (`:31,:41,:53,:66,:79,:92,:105`), `register()` `:116‑124`. Patrón = `define_rule!` brazo domain-aware corto (`macros.rs:235`). **En modo real (default) TODAS inertes → huella byte-idéntica.** El enum `ComplexRewriteKind` `:78‑85` tiene 6 variantes (sin `GaussianPow`/`Abs`).
4. **Template transcendental YA en producción:** `ComplexNegativeBaseRootRule` `power_rules.rs:402` (`impl Rule` estructurada, gate `:420`, construye `cos θ + i·sin θ` `:435‑448`). El trig de π-racional y la forma `ln 2 + i·π` **se pliegan solos** aguas abajo: Euler/Log solo tienen que EMITIR la forma y el pipeline la reduce.
5. **Eje `BranchPolicy`** `cas_solver_core/src/branch_policy.rs` = solo `Principal` (no-op hoy). Principal-branch NO necesita consultarlo; `apply.rs:163` descarta `axes.complex_branch` (consistente con el modelo principal-branch-only).

### Los tres chokepoints (el equivalente al "solo factoriza sobre ℚ" de G1)

- **CHOKEPOINT-A · numérico (el análogo más fuerte del gate de G1).** `eval_f64`/`numeric_eval` **rechazan `Constant::I`**: `evaluator_f64.rs:534` (`I=>None`), `:187` (`I=>Err(Domain)`), `:165`; `numeric_eval.rs:440/:540`. Consecuencia crítica: las identidades **a nivel VALOR** (`e^(iπ)=-1`, `ln(-1)=iπ`) **no tienen variable** → el diff-back simbólico es **inaplicable** → la ÚNICA red de cross-check independiente es un probe numérico complejo. Sin él, un engine soundness-first **declina** cualquier regla transcendental. Es el primitivo caro → **bloque B, sub-ciclo B1**.
- **CHOKEPOINT-B · algebraico exacto.** La capa de equivalencia EXACTA trata `i` como **indeterminada opaca**: `multipoly_from_expr` (`expr_domain.rs:10`) y `as_rational_const` (`semantic_equality.rs:16`) **no saben `i²=-1`**. `(x+i)(x-i) ≡ x²+1` no es confirmable exactamente. Enseñar `i²=-1` (reducción módulo `i²+1`) habilita la confirmación exacta de identidades algebraicas complejas y el verificador de integración complejo → **sub-ciclo A3** (independiente, sin dependencia nueva).
*(re-audit 2026-07-18: el probe emblemático YA confirma en vivo (i²=-1 pliega en la capa de rewrite antes de comparar); A3 queda reducido a enseñar i²=-1 a multipoly/semantic_equality SOLO si un ciclo futuro necesita esa capa específica)*
- **CHOKEPOINT-C · solve.** Los kernels de generación de conjunto-solución son **real-only** y descartan las raíces `√Δ<0` **aguas arriba** (`solution_set.rs:527` `Eq→Empty`; `rational_roots.rs:961` `else vec![]`), **antes** del gate `drop-non-real` que **ya es domain-aware** (`solve_backend_local.rs:11124`). Las expresiones-raíz `(-b±√Δ)/2a` se **construyen** (`quadratic_formula.rs:38‑46`, `sqrt_expr` incondicional) y se **tiran** sin mirar el dominio. El render complejo ya emite `±i`; el fix = enhebrar `value_domain` hasta esos kernels y no descartar bajo `ComplexEnabled` → **sub-ciclos A4/A5**.

---

## Secuencia de sub-ciclos (cada uno = un `/auto-mejora`, un commit)

Orden por **dependencia + blast**: el más pequeño/zero-blast/reusable primero (análogo C-i/E-i de G1). Tres bloques: **A algebraico-exacto** (sin dep nueva, verificable por ℚ[i]), **B transcendental** (necesita la red numérica), **C presentación/pedagogía** (transversal).

### Bloque A — algebraico exacto (sin dependencia nueva)

#### ☑ A1 — Potencia entera Gaussiana `(a+bi)^n` **[S] — HECHO** *(2026-07-16 `da2a2f151`)*
- **Graduado:** `(1+i)^2→2i`, `(1+i)^3→2(i-1)`, `(2+i)^4→-7+24i`, `(3+4i)^2`, `(1-2i)^3→-11+2i`, `(1/2+i)^2→-3/4+i` (la canonicalización racional pliega `Div(1,2)` antes de la regla — extract_gaussian NO se ensanchó, lección widening-collector). **BONUS: `(1+i)^(-2)→-i/2` gradúa GRATIS** (canonicaliza a `1/(1+i)^2`, el fold interior + GaussianDivRule rematan — la composición anticipada en el scoping). Composiciones verifican: `(1+i)^2+(1-i)^2→0`, `(1+i)^2·(1-i)^2→4`. Ownership intacto (`^0`,`^1`, `(2i)^3`, `i^7`, `(x+i)^2` simbólico); modo real byte-idéntico; cap 4096 declina honesto (`(1+i)^5000`). `GaussianRational` estrena `mul()`/`pow()` como métodos (repeated squaring exacto). Narración al nivel de las 7 hermanas (rule + traza LaTeX). Residual documentado: `(1+i)^(-1)` sigue emitiendo el parcial `(1/2·2-i)/(2)` (ruta recíproca previa, dueño C1-display).
- **Gradúa:** `(1+i)^2 → 2·i`, `(1+i)^3 → 2·i-2`, `(2+i)^4`, `(3+4i)^2`.
- **Inserción:** `complex_support.rs:85` (variante `ComplexRewriteKind::GaussianPower`, antes del `}` en `:86`); `complex_support.rs:70` (impl `GaussianRational`: métodos `mul()/pow()` exactos sobre `BigRational` — hoy CERO ops son método); `complex_support.rs:~405` (`try_rewrite_gaussian_power_expr`, con **guard `g.is_real() || g.is_pure_imag() → None`** para no colisionar con `ImaginaryPowerRule`, exponente entero ≥0); `complex.rs:24` (brazo desc); `complex.rs:114` (`GaussianPowRule`, gate `RealOnly→None`); `complex.rs:123` (`add_rule` en `register()`).
- **Reuso:** `extract_gaussian`, `to_expr`, fórmula mul inline `:239‑244`, `define_rule!` domain-aware.
- **Net-new:** variante enum + free-fn + los primeros métodos `mul/pow` de `GaussianRational` (el primitivo reusable que A2/A4/A5 aprovechan).
- **Blast:** **BAJO** — auto-gateado (real-mode byte-idéntico). Trampas: (1) loop si re-emite `Pow` → materializar el gaussiano **ya multiplicado** vía `to_expr`; (2) colisión con `ImaginaryPowerRule` → el guard excluye base pura-`i`; (3) nombre de regla único (`assert_unique_rule_names` `engine/simplifier.rs:539`).
- **Depende:** nada.
- **Retención:** `complex_tests.rs:130` pinea `to_expr == '3 + 2 * i'` (contrato de orden `a+bi`) — respetar. Workspace verde; huella real byte-idéntica.
- **Residual conocido (peldaño, no bloquea):** exponente **negativo** `(1+i)^(-n)` — hoy `(1+i)^(-1)` devuelve el parcial sin terminar `(1/2·2 - i)/(2)`; A1 scopea `n≥0`. Extensión opcional: potencia positiva + recíproco Gaussiano (o dejar como residual honesto hasta C1-display).
*(re-audit 2026-07-18: CERRADO — tanda tanda-2 ciclo 1: AddFractions pliega Number×Number en la emisión; `(1+i)^(-1)→1/2 - 1/2·i`)*

#### ☑ A2 — Módulo `|a+bi|` + builtins `Re`/`Im`/`conjugate` (exactos ℚ[i]) **[M] — HECHO** *(2026-07-17 `ed3e8e61b` — incluye el batch de 10 gates de soundness abs/signo destapado por composición A1×A2; ver ledger)*
- **Gradúa:** `abs(3+4i) → 5`, `conjugate(3+4i) → 3-4·i`, `Re(3+4i) → 3`, `Im(3+4i) → 4`.
- **Inserción:** `cas_ast/src/builtin.rs` (**5 sitios sincronizados**: enum + `name()` + `from_name()` + `ALL_BUILTINS` + `COUNT 46→49`); `cas_session_core/src/eval.rs:68` (`is_known_eval_engine_function` pasa auto si `from_name` es `Some`); `abs_support.rs:299` (rama módulo Gaussiano reusando `negative_abs_to_i_sqrt` para el fold de cuadrado perfecto); reglas de despacho `Re/Im/conjugate` target `FUNCTION` en `complex.rs:116`; helpers puros en `complex_support.rs`.
- **Reuso:** `extract_gaussian`, `to_expr`, `negative_abs_to_i_sqrt`.
- **Blast:** **MEDIO** — `builtin.rs` de 5 sitios (`COUNT` desincronizado = fallo SILENCIOSO). `Arg` **NO** va aquí (necesita `atan2` transcendental → B3).
- **Decisión de gating (TOMADA en la revisión 2026-07-16):** las 7 reglas existentes gatean OFF en `RealOnly`. Un `|a+bi|` **sin gate** haría `abs(3+4i)→5` en modo REAL mientras la álgebra Gaussiana sigue congelada; además, en `RealOnly` `i` es un **símbolo ordinario** (`domain_contract_tests.rs:653`), así que evaluar su módulo sería incoherente con la semántica del modo. → **Módulo y builtins gateados a `ComplexEnabled`**, como sus 7 hermanas.
- **Depende:** nada.
- **Retención:** `domain_contract_tests.rs:394` (`prove_positive(i)==Unknown`) DEBE sobrevivir — un módulo/comparación nuevo NO debe hacer `prove_positive(i)=Proven`.

#### ☐ A3 — Reducción exacta `i²=-1` en la capa de equivalencia (CHOKEPOINT-B) **[S/M — DIFERIDO, on-demand]**
> **Decisión de revisión (2026-07-16):** NO es prerequisito duro de A1/A2/A4/A5 (la aritmética Gaussiana y las raíces cuadráticas son exactas por construcción). Su valor real aparece cuando un ciclo necesite CONFIRMAR una identidad algebraica compleja (integración compleja, verificación de antiderivadas con `i`). **Se difiere hasta que un ciclo lo requiera** — en ese momento se ejecuta como prerequisito nombrado de ese ciclo (patrón nivel-2 de G1), no como ciclo especulativo.
- **Gradúa:** `(x+i)(x-i) ≡ x²+1` confirmable exactamente; el verificador de integración complejo puede confirmar antiderivadas con `i` una vez conocido `i²=-1`.
- **Inserción:** `expr_domain.rs:10` (reducción `i²=-1` en `multipoly_from_expr` — módulo `i²+1`) y `semantic_equality.rs:16`; `general_integration_backend/verification.rs:176` (confirm de antiderivada compleja).
- **Reuso:** la maquinaria multipoly existente; `GaussianRational` para el confirm cerrado.
- **Blast:** **MEDIO** — toca la capa de equivalencia exacta (superficie de soundness). Regresión explícita: byte-identidad de todos los probes reales; la reducción `i²=-1` es EXACTA (sin f64).
- **Depende:** nada (independiente; era la mitad "M4a" que la síntesis había fundido en un solo ciclo con la red numérica — **la verificación la separó**, tienen dep y consumidores distintos). Puede aterrizar en cualquier punto del bloque A.
- **Nota:** habilita CONFIRMACIÓN exacta; NO es prerequisito duro de A1/A2/A4/A5 (la aritmética Gaussiana y las raíces cuadráticas son exactas por construcción).

#### ☑ A4 — Solve complejo cuadrático desnudo (F12, `Δ<0`) **[S] — HECHO** *(2026-07-17 `1857f1f9c`)*
- **Graduado:** `solve(x^2+1,x)→{i,-i}`, `x^2+4→{±2i}`, `x^2+x+1→{(-1±i√3)/2}`, `x^2-2x+5→{1±2i}`, `2x^2+3→{±3i/√6}` — **F12 CERRADO** (la fila emblemática vs sympy). El fix real fue DOBLE: (1) el gate de dominio en la rama `Δ<0∧Eq` (como scopeado), y (2) **un chokepoint NO scopeado**: el `simplify()` interno del solver corre con `SimplifyOptions::default()` = RealOnly SIEMPRE → las raíces emitían sin plegar (`(-3)^(1/2)` en vez de `i√3`). Cierre: campo sticky `sticky_value_domain` en el Simplifier (patrón `set_sticky_implicit_domain`, dropea solve-memo al cambiar) consumido por `simplify()`, seteado con save/restore en `solve_with_ctx_and_options`. Bordes verificados: `x^2=0→{0}`, `x^2=1-√2` complex→`±(1-√2)^(1/2)` (valor correcto, surd no-literal no pliega a forma-i — residual de presentación) y real→`No solution` (guard R1 intacto), coefs Gaussianos `x^2+2i=0` emiten (√ de complejo = territorio bloque B), `≠`/inecuaciones conservan semántica real (SolutionSet no representa ℂ∖{±i} — decisión documentada). Narración: hereda la cadena cuadrática ("fórmula cuadrática") vía `solve_steps`; prosa "Δ<0→par conjugado" queda para C2.
- **Gradúa:** `solve(x^2+1, x) → {i,-i}`, `solve(x^2+x+1, x) → {-1/2 ± √3/2·i}`, `solve(x^2-2x+5, x) → {1±2i}`.
- **Inserción:** `solution_set.rs:449` (añadir `value_domain` a la firma) y `:527‑529` (rama `else Δ<0`: si `ComplexEnabled ∧ op==Eq → Discrete(vec![r1,r2])` en vez de `Empty`); `quadratic_formula.rs:244` (propagar dominio) y `:225‑234` (invertir el guard de Δ-simbólica-negativa SOLO en `ComplexEnabled`); `quadratic_strategy.rs:127` (pasar `is_real_only` al plan).
- **Reuso:** `roots_from_a_b_delta` `quadratic_formula.rs:38‑46` (ya construye `√Δ` negativo con `sqrt_expr` incondicional), render complejo ya emite `±i`, gate `drop-non-real` `solve_backend_local.rs:11124` ya domain-aware (**NO tocar**).
- **Blast:** **BAJO-MEDIO** — el cambio de firma toca ~5 llamadores de `quadratic_numeric_solution` (4 tests `solution_set.rs:1565+` + `quadratic_formula.rs:247`). **SCOPE-OUT:** inecuaciones `Δ<0` (orden indefinido en ℂ) → cambiar SOLO la rama `RelOp::Eq`.
- **Depende:** nada duro (el `bool is_real_only` ya llega a la frontera de la estrategia).
- **Retención:** los tests reales que esperan `Empty` en modo real siguen verdes — fix **gateado SOLO en `ComplexEnabled`**. (Nota de verificación: `solution_set.rs:1565` es en realidad un test `Δ>0`; no hay test dedicado que pinee `Empty` para `x²+1 ∧ Δ<0∧Eq` — cablear uno nuevo al graduar.)

#### ☑ A5 — Solve complejo grado ≥3 (deflación + par conjugado) **[L] — HECHO** *(2026-07-17, hash en el ledger)*
- **Graduado:** `x^4-1→{±1,±i}`, `x^3-1→{1,(-1±i√3)/2}`, `x^3+8→{-2,1±i√3}`, `x^4-16→{±2,±2i}`, `x^4+5x^2+4→{±i,±2i}`, `x^3+x=0→{0,±i}`. Kernel rational-roots domain-aware (threading `is_real_only` 6 firmas); biquadrático `Option<Vec>` (Δ_z<0→None honesto); **completitud grado≥3 en ℂ = siempre false** (Sturm cuenta reales — trampa F4 edición compleja evitada). Real intacto. RESIDUAL NOMBRADO: familia pow-isolation (`x^4+1→No solution`, `x^5-1→{1}`, `x^3=8→{2}` — aislamiento n-ésimo real-only; pre-existente, dueño exacto en el ledger).
*(re-audit 2026-07-18: estado reducido — `x^3=8` complex resuelto completo; `x^4+1`/`x^5-1` complex ya no dan subconjunto/No-solution sino ECO residual honesto (queda el aislamiento n-ésimo))*
- **Gradúa:** `solve(x^4-1, x) → {-1,1,-i,i}`, `solve(x^3-1, x) → {1, -1/2 ± √3/2·i}`.
- **Inserción:** `rational_roots.rs:931/961` (`solve_residual_degree_leq_two`: `value_domain` + rama `Δ<0 → roots_from_a_b_delta`); `rational_roots.rs:851/857` (`solve_residual_biquadratic`: `z<0 → ±i√|z|`); `rational_roots.rs:810/835` (`extract_candidate_roots`: `value_domain` + **completitud contada por GRADO, no `count_real_roots` Sturm**); `cas_solver_core/src/solve_runtime_flow_strategy_kernels_equation.rs:28` (origen del enhebrado; `is_real_only` ya disponible); `solve_backend_local.rs:11724/:12354` (helpers cuártico/cúbico: emitir par conjugado o ceder a la ruta general).
- **Reuso:** los cambios de kernel de A4 + `roots_from_a_b_delta`; render `±i`.
- **Blast:** **ALTO** — la firma toca múltiples llamadores internos + tests `rational_roots.rs:1000+`; el verificador f64 de raíces (`solve_backend_local.rs:11757`, `eval_f64` rechaza `i`) **no vale** para complejas → saltar (las raíces de cuadráticas racionales son exactas por construcción).
- **Depende:** **A4.**
- **Retención (REGRESIÓN P0 de completitud):** `extract_candidate_roots:841` usa `count_real_roots()==0` — al cambiar a "grado" verificar que **NO se declare "completo con 0 raíces"** un residual irreducible grado≥3 (memoria [[frontier-audit-2026-07-13b-8-familias]] F4: declarar completo un subconjunto = **P0 wrong-answer**). Modo real `x^4-1→{-1,1}`, `x^3-1→{1}` intactos.

### Bloque B — transcendental **[RE-SCOPEADO 2026-07-17 con el bloque A aterrizado]**

> **Re-scope** (workflow 3 mappers + síntesis + verificador: 53 anclas + ~25 probes confirmados a HEAD `688425f9c`; journal `wf_3f023d4b-47f`). **8 decisiones cerradas con evidencia** — las claves:
> 1. **B1 = HAND-ROLLED POR EXTRACCIÓN, cero dep nueva.** El struct privado `C` de `rootsum_numeric.rs:22‑83` (G1 E-iv-d1) YA es el núcleo: add/mul/div/abs(hypot) **y `ln` PRINCIPAL-BRANCH vía `f64::atan2`** (`:63‑70`), testeado vs mpmath 30 dígitos (`:352‑374`), con la decisión anti-dep escrita en el propio archivo (`:21` "kept local: no new dependency"). Faltan ~35 líneas de ops (exp/sin/cos/sqrt/powc/arg) + ~150 de walker. `num-complex` queda descartado (cero precedente de dep de conveniencia en 13 crates; dejaría dos tipos complejos).
> 2. **La premisa "B2 debe casar dos formas AST" queda DEGRADADA a hardening**: el parser desugariza `exp(x)→Pow(E,x)` **en parse** (`parser.rs:396‑404`, incondicional) — `Pow(E,·)` es el match primario; el brazo `Function(exp,·)` no tiene productor vivo (~5 líneas defensivas). **El bloqueador REAL de B2 es otro**: `extract_gaussian` NO extrae `i·π` (el brazo Mul exige `Number` junto a `I`; `π` es `Constant::Pi`) → hace falta el splitter simbólico **`split_i_factor`** (cadenas Mul anidadas, `Div(Mul(Pi,I),2)`, Add mixto).
> 3. **`Arg` es EXACTO, sin f64**: tabla atan2 de 9 casos por signos sobre el Gaussiano cerrado, emitiendo `π` racional / `atan(q)` simbólico (`atan(±1)` ya pliegan). B1 deja de ser prerequisito computacional de B3 — queda como **verificador refute-only pre-commit**.
> 4. **Dependencias reales**: la ÚNICA dura es **B3→B4** (B4 consume `ln(z)`/`Arg`). B2 tiene red de retención EXACTA propia (metamórfica ODE `diff(f)≡i·f` — probes byte-idénticos hoy — + pin `f(0)=1` + unimodularidad). `i^i` cierra con B3+B4 solos; B2 es critical-path solo para `2^i` (su test de aceptación VIVE EN B4) y `e^(1+i)`.
> 5. **Dueño de `sqrt(i)`/`(1+i)^(1/2)`**: B4 (polar), con fast-path exacto `GaussianSqrtRule` (`√(a+bi)` cerrado cuando `|z|²` es cuadrado racional perfecto: `sqrt(3+4i)→2+i`). NO generalizar `ComplexNegativeBaseRootRule`.
> 6. **Condición estructurada de rama**: variante nueva `ImplicitCondition::PrincipalBranch{func,arg}` con **`is_trivial()→false` SIEMPRE** — ⚠️ trampa verificada: `domain_condition.rs:172` devuelve trivial para toda expresión sin variable y `diagnostics_model.rs:117` la dropea; el caso común de B3/B4 ES constante cerrada (`ln(-1)`, `i^i`) → el short-circuit va ANTES de `:172` (reestructurar el cuerpo compartido, no añadir un brazo al final). Auditar también el `TryFrom<&ConditionPredicate>` con wildcard (`:1084`) — el compilador NO señala ese round-trip.

**Orden recomendado: B1 → B2 → B3 → B4** (contingencia validada: si B1 se atasca, B2 puede adelantarse sin pérdida de soundness).

#### ☑ B1 — `eval_complex` refute-only por EXTRACCIÓN de rootsum_numeric **[M] — HECHO** *(2026-07-17, hash en el ledger)*
- **Gradúa:** `equiv(e^(i·π), 1)` complex → `false` (REFUTA vía probe complejo) mientras `equiv(e^(i·π), -1)` → `unknown` (JAMÁS true desde probe); brazo complejo de `numeric_poly_zero_check` refuta identidades falsas con `i`. *(El criterio del borrador `approx(abs(i))→1` YA pasa hoy vía fold exacto de A2 — probe débil, sustituido.)*
- **Inserción:** `rootsum_numeric.rs:22‑83` (extraer `C` → `pub Complex64` en módulo NUEVO `cas_math/src/evaluator_complex.rs`; renombrar la fn privada `eval_complex` de `:147`); walker modelado sobre `evaluator_f64.rs:416‑541` con `Constant::I→Complex64{0,1}`; fallback refute-only en `engine/equivalence.rs:340‑352` (⚠️ NO en `are_equivalent:72‑83`, que SÍ confirma desde probe — heredar esa forma sería unsound) y `numeric_eval.rs:374‑393` (PROHIBIDO contribuir al camino true: el argumento Schwartz-Zippel de ahí es solo polinomial); análogos probe-map/eps en `cas_solver_core/equivalence.rs:35/:46` con semillas off-diagonal; re-exports ADITIVOS (`cas_math/lib.rs:201`, `cas_engine/api.rs:79`, `cas_solver/api.rs:22`) — `eval_f64` INTACTO.
- **Retención:** modo real byte-idéntico gratis (todo aditivo); la suite mpmath de rootsum valida la delegación; contrato documentado citando `actions.rs:461‑465` ("un probe JAMÁS confirma") + el disclaimer display-grade de Durand-Kerner (`:86‑88`, "NOT a decision procedure").
- **⚠️ Trampa pow-convención:** `pow_real` (`evaluator_f64.rs:393‑413`) implementa raíz impar REAL (`(-8)^(1/3)=-2`) vs. powc principal (`1+i√3`) — mezclar convenciones en un probe = REFUTACIONES FALSAS; `eval_complex` corre solo bajo semántica compleja.
- **Residual nombrado:** `approx()` de valores complejos cerrados (`approx(ln(i))`, `approx(e^i)`) queda SIN cablear en B1-B4 (requiere ensanchar el retorno de approx a `a+bi`; `meta_functions_support.rs:56‑66`) — sub-ciclo aparte post-B. *(GRADUADO 2026-07-17, T4·ciclo-4, hash en el ledger: fallback complejo en el brazo approx vía `eval_complex`, gated ComplexEnabled — `approx(ln(i))→i·1.5707…`, `approx(2^i)→0.7692…+i·0.6389…`; real byte-idéntico.)*

#### ☑ B2 — `EulerRule`: `e^(i·θ)` y `e^(a+bi)` → forma trigonométrica **[M] — HECHO** *(2026-07-17, commit `963ce8656`)*
- **Gradúa:** `e^(i·π)→-1`, `e^(i·π/2)→i`, `e^(i·π/4)`, `e^(2πi)→1`, `exp(i·x)→cos(x)+i·sin(x)`, `e^(1+i)→e·(cos 1+i·sin 1)`.
- **Inserción:** `complex.rs:186/:200` (13ª hermana, gate `==RealOnly→None`, target FUNCTION|POW); **`split_i_factor` NUEVO** en `complex_support.rs:~548` (el bloqueador real — ver decisión 2; `extract_gaussian` solo como fallback del exponente Gaussiano racional); template de construcción `cos+i·sin` en `power_rules.rs:435‑448`; `is_e_constant_expr` (`expr_predicates.rs:235`).
- **Depende:** NINGUNA dura — red de retención exacta propia: **metamórfica ODE** (`diff(cos x+i·sin x, x) ≡ expand(i·(cos x+i·sin x))` — byte-idéntico HOY), pin `f(0)=1`, unimodularidad, pins en π-racionales (por equivalencia, NO string exacto: `sin(π/3)` imprime no-canónico).
- **Retención:** convención ONE-DIRECTION (PROHIBIDA la inversa `cosθ+i·sinθ→e^(iθ)` — ping-pong; anotar en el commit); NO tocar `canonicalization.rs:314‑317` ni el parser. Residuales nombrados: `e^(-i·x)` queda `1/(cos x+i·sin x)` (recíproco pre-Euler, presentación); `abs(e^(i·x))→1` (unimodularidad como simplificación) sin dueño — residual honesto, no regresión.
*(re-audit 2026-07-18: unimodularidad GRADUADA tanda-2 ciclo 2 con re-scope V0: pliega para θ real DECIDIBLE; θ simbólico declina CORRECTO (x puede ser compleja); `e^(-i·x)` recíproco sigue vivo)*

#### ☑ B3 — Log principal + builtin `Arg` EXACTO **[L → M efectivo] — HECHO (núcleo exacto)** *(2026-07-17, commit `cfee09381`)*
- **Graduado:** `ln(-1)→π·i`, `ln(i)→π/2·i`, `ln(-2)→ln 2+π·i`, `ln(1+i)→½·ln 2+¼·π·i`, `arg(i)→π/2`, `arg(-1)→π`, `arg(1+i)→π/4`, `arg(-1-i)→-¾·π`, `arg(0)→undefined` explícito, `arg(3)→0`; tabla atan2 de 9 casos EXACTA (π racionales / `atan(q)` simbólico); `ln|z|` vía fold de cuadrado perfecto de `a²+b²` o `ln(norm)/2`; verificación independiente contra la red B1 (`eval_complex` antes/después, incl. brazo `atan` real añadido al walker). Modo real byte-idéntico (`ln(-1)→undefined`, `ln(2)` y `arg` simbólicos intactos).
- **Decisión de alcance (2026-07-17):** la condición estructurada `PrincipalBranch` se MUEVE a B4 — cruza 4 crates con 2 trampas verificadas (short-circuit ANTES de `domain_condition.rs:172`; wildcard `TryFrom` `:1084`) y su segundo emisor es `z^w`: un solo cableado para ambos consumidores en el ciclo que ya toca esa capa. Residual nombrado, no incumplimiento del guardrail #3 — los folds B3 son sound sin la condición (rama principal única documentada en la desc de regla). *(El graduate `ln(-e)` quedó RETIRADO en la revisión: requiere split factor-negativo-real que extract_gaussian no capta — peldaño de B4 o posterior.)*
- **Inserción:** `builtin.rs` 5 sitios (`Arg` tras Conjugate `:126`, COUNT 49→50 `:131`, name `~:206`, from_name con DOS arms `"Arg"`/`"arg"` `:283`, ALL_BUILTINS `:347`); `try_rewrite_arg_expr` (tabla 9 casos, `complex_support.rs:~502`); `ArgRule` en `complex.rs`; `ComplexLogRule` en `logarithms/mod.rs:52/:203` (struct-impl con parent_ctx, **priority 16** para interceptar antes de EvaluateLogRule, DECLINA racional positivo puro — la narración de `ln(2)` en complejo no cambia); variante `PrincipalBranch` en `domain_condition.rs:17‑26/:172` (short-circuit ANTES del early-return de trivialidad) + `domain.rs:59` + `domain_normalization.rs:84‑104` + `eval_output_public_conditions.rs:26‑38`.
- **Depende:** computacionalmente NADA (núcleo exacto); B1 como verificador pre-commit. Independiente de B2. `logarithm_inverse_support.rs:375` NO se toca (decline real intacto).

#### ☑ B4 — `z^w = e^(w·Log z)` + fast-path `GaussianSqrtRule` + condición `PrincipalBranch` **[partido en B4a+B4b] — HECHO** *(2026-07-17; B4a commit `14dabe3b8`, B4b hash en el ledger)*

**☑ B4a — wire `PrincipalBranch` (2026-07-17, hash en el ledger):** `ConditionPredicate::PrincipalBranch{func,arg}` (cas_ast, junto a InvTrigPrincipalRange) + `ImplicitCondition::PrincipalBranch` (solver_core) con `is_trivial()→false` vía short-circuit EN EL MATCH INICIAL (antes del early-return `!contains_variable` — la trampa :172 desactivada estructuralmente) + round-trip `TryFrom` pinneado (la trampa del wildcard :1084) + 18 brazos en 8 crates (inference/normalization/airbag/diagnostics/didactic/solve-backend/wire-JSON/formatter, cada uno decisión explícita nombrada, no wildcard). Emisores: ComplexLogRule (`func:"ln"`) y ArgRule (`func:"arg"`, salvo verdict undefined). Retenible: `eval "ln(-1)" --value-domain complex` → `required_conditions: [{kind:"PrincipalBranch", expr:"-1"}]` + display `"ln(-1) via principal branch (Arg ∈ (-π, π])"`; modo real byte-idéntico.
**☑ B4b — `z^w` + `GaussianSqrtRule` (2026-07-17, hash en el ledger):** GRADUADO: `i^i→e^(-π/2)`, `2^i→cos(ln 2)+i·sin(ln 2)`, `2^(1+i)`, `(-2)^i→(cos(ln 2)+i·sin(ln 2))/e^π`, `sqrt(3±4i)→2±i`, `sqrt(i)→sqrt(1/2)·(1+i)`, `sqrt(-5+12i)→2+3i`, `(1+i)^(1/2)` polar EXACTO (half-angle pliega), `i^(1/3)→√3/2+i/2`; todos con condición `PrincipalBranch`. Rama directa Euler para base real positiva (evita ping-pong con el canonicalizador exp-log `e^(a·ln b)→b^a` cuando `ln z` queda simbólico). **BONUS P0 cazado por el fixture nuevo:** `Neg` IEEE produce `im=-0.0` y `atan2(-0.0,-x)=-π` → la red B1 REFUTABA `ln(-e)=1+iπ` (verdadera); fix `signed_zero_fix` en `arg`/`ln` del walker + pin. Fixtures never-confirm re-anclados: `ln(-e)≡1+iπ` (split factor-negativo-real pendiente) y `sin(i)≡i·(e−1/e)/2` (trig-de-i, fuera del bloque B). Scoping original (histórico): gradúa `i^i→e^(-π/2)` (cadena verificada por tramos), `2^i→cos(ln 2)+i·sin(ln 2)` (**el test de aceptación de `2^i` vive AQUÍ**, requiere B2), `sqrt(i)→√2/2·(1+i)` EXACTO vía fast-path, `sqrt(3+4i)→2+i`, `(1+i)^(1/2)` polar honesto; el emisor `z^w` ata la condición B4a ya cableada. **Del scoping original:** variante `ImplicitCondition::PrincipalBranch{func,arg}` con `is_trivial()→false` SIEMPRE — ⚠️ short-circuit ANTES del early-return `!contains_variable` de `domain_condition.rs:172` (reestructurar, no añadir brazo al final) + auditar el wildcard `TryFrom<&ConditionPredicate>` `:1084` (dropea variantes nuevas en SILENCIO); emitida por `ComplexLogRule`/`ArgRule` (B3, ya aterrizados) y por `z^w` — un solo cableado, dos emisores. Al cerrar, actualizar el fixture never-confirm de `equivalence.rs` (`i^i ≡ e^(-π/2)` gradúa → nuevo fixture).
*(re-audit 2026-07-18: trig-de-i GRADUADO tanda-2 ciclo 3 (puente entero, 6 brazos + walker sinh/cosh/tanh); el fixture never-confirm sigue anclado en `ln(-e)` — verificado eco)*
- **Inserción:** `ComplexGeneralPowerRule` en `exponents/mod.rs:33` (registro antes de EvaluatePowerRule, priority 16, template `power_rules.rs:402‑460`); emite `Pow(E, Mul(w, ln(z)))` SOLO para constantes cerradas; **DECLINA**: exponente entero (dueño GaussianPowRule), base racional real (EvaluatePowerRule/ComplexNegativeBaseRoot/SqrtNegative), **base==E** (⚠️ gap del verificador: sin este decline, `Pow(E, i·π)` re-emitiría `e^(w·ln e)` = churn de fixpoint + robo a EulerRule), simbólicos (residual honesto). `GaussianSqrtRule` (`√(a+bi)=√((|z|+a)/2)+i·sign(b)·√((|z|-a)/2)`, gate `a²+b²` cuadrado perfecto ∧ `b≠0`) junto a NegativeBaseHalfPower.
- **Depende:** **B3 DURA** (consume `ln(z)`/`Arg` — una sola implementación de rama, JAMÁS una segunda atan2); B2 solo para `2^i`; B1 pre-commit.
- **Retención:** pins de regresión en ambos dominios sobre `(-8)^(1/3)`, `sqrt(-4)`, `(2+i)^2` (las formas exactas NO se degradan a polar).

### Bloque C — presentación / pedagogía (transversal)

#### ☑ C1 — Normalización de forma cartesiana `a+bi` **[S] — HECHO (orden)** *(2026-07-17, hash en el ledger)*
- **GRADUADO:** orden cartesiano en display — `(3+4i)/(1-2i)→-1+2·i` (antes `2·i-1`), `solve(x²+2x+5)→{-1-2·i, -1+2·i}`, `(1+i)^3→2·(-1+i)`, `(-8)^(1/3)→1+3·i·3^(-1/2)`. Implementación: override en `cmp_term_for_display` (`ordering.rs`) ANTES de la regla positivo-antes-que-negativo, shape-gated por presencia de `i` (término i-free primero SIEMPRE) — UN comparador alimenta texto, hints y LaTeX. Solo 2 pins migrados (ambos se auto-documentaban como "espera C1"). Expresiones sin `i` intactas byte-a-byte (pins `1-x`, `x²-3x+2`).
- **RESIDUAL nombrado (el parcial `(z)^(-1)`):** `(1+i)^(-1) → (1/2·2 - i)/(2)` NO es de display — el ÁRBOL final llega mangled SOLO por la ruta `Pow(z,-1)` (la ruta directa `1/(1+i)` da `1/2 - 1/2·i` limpio). Diagnóstico avanzado: un pase de recombinación de fracciones con holds internos deja `Mul(1/2, 2)` sin plegar (los holds bloquean el fold Number×Number); el emisor vive en la maquinaria AddFractions (`algebra/fractions/addition_rules.rs`), no localizado el sitio exacto en el timebox. Peldaño futuro: cazar el emisor con panic-trampa condicional (patrón T2-pipeline-layer).
*(re-audit 2026-07-18: GRADUADO tanda-2 ciclo 1 (hash en el ledger): la panic-trampa cazó el emisor (`build_add_fraction_rewrite→mul2_raw`) y el fold en emisión lo cerró)*

#### ☐ C2 — Pulido/localización de la narración compleja **[S — reducido en la revisión 2026-07-16]**
*(re-audit 2026-07-18: NÚCLEO GRADUADO tanda-2 ciclo 4: los 21 nombres de regla complejos localizados es/en en visible_rule_names; queda pulido puntual (elevar narraciones-cáscara multi-paso))*
- **Contexto de la reducción:** la narración didáctica dejó de ser un batch final — es **entregable per-ciclo** dentro del `graduates` de cada A/B (ver "Orden recomendado"). Dejar las reglas nacer mudas para narrarlas después repetiría el error histórico de "límites a 0% educativo".
- **Gradúa (lo que queda aquí):** localización es/en de las descripciones (`format_complex_rewrite_desc` emite solo inglés hoy), elevación de narraciones-cáscara a cadenas multi-paso donde aporte (p.ej. la división por conjugado narrada en 2 pasos), y coherencia de estilo con la narrativa G2.
- **Inserción:** templates es/en en `locale.rs` + builders en `cas_didactic` (mismo patrón que la narrativa de límites G2).
- **Blast:** **BAJO** — capa didáctica, huella NONE sobre resultados.
- **Depende:** las capacidades ya narradas per-ciclo; va al final del frente.

---

## Orden recomendado y primer ciclo

**Ejecutar A1 primero.** Es el análogo exacto del **C-i/E-i de G1**: el más pequeño (S), **zero-blast** (auto-gateado `RealOnly→None`, huella byte-idéntica en modo real por defecto), **sin dependencia nueva**, y **extiende el primitivo reusable central** (`GaussianRational`: introduce por fin métodos `mul()/pow()` como `impl`, hoy cero ops son método) que A2/A4/A5 reaprovechan. Cierra un fallo de frontera visible (`(1+i)^2`) por la ruta puramente exacta (verificable por ℚ[i], sin f64, sin verificador nuevo). De-risquea la mecánica de "añadir regla Gaussiana" antes de tocar nada caro.

**Resolución explícita de la dependencia M3↔M4** (por qué la red numérica NO va primero): el evaluador numérico complejo (**B1**) **no** es prerequisito de los ciclos algebraicos (A1‑A5 son EXACTOS, verificables por ℚ[i] o keep/drop-safe). B1 **sí** es prerequisito **duro y único** de los transcendentales (B2/B3/B4): las identidades a-nivel-valor (`e^(iπ)=-1`, `ln(-1)=iπ`) no tienen variable → el diff-back simbólico es inaplicable → el probe numérico complejo es el ÚNICO cross-check independiente; sin él el engine soundness-first declina. Por eso B1 es el "primitivo duro reusable, huella byte-idéntica" **pero su lugar es al ENCABEZAR el bloque B**, no en el arranque absoluto.

**Orden global (revisado 2026-07-16):** `A1 → A4 → A2 → A5` (bloque A) `→ B1 → B2 → B3 → B4` (bloque B) + `C1` intercalable. Ajustes de la revisión sobre el orden original `A1→A2→A3→A4→A5`:

- **A4 adelantado al 2º ciclo:** es el gap emblemático vs. sympy (fila F12 del assessment), es [S], y es independiente de A2/A3. Los dos fallos más visibles del frente (`(1+i)^2` y `solve(x²+1)`) quedan verdes en los dos primeros ciclos. Coste asumido: un context-switch reglas↔solver antes de lo ideal.
- **A3 diferido on-demand** (ver su entrada): no es prereq duro de nada del bloque A; se ejecuta como prerequisito nombrado del primer ciclo que necesite confirmación exacta de identidades complejas.
- **Compromiso por bloques:** el greenlight de esta revisión cubre el **bloque A completo**. El bloque B se **RE-SCOPEA al aterrizar A** (patrón G1: Cap. E se scopeó con la máquina A‑D ya verde) — B introduce la primera dependencia externa candidata (`num-complex`, decisión abierta en B1) y la superficie principal-branch, y merece decidirse con la experiencia del bloque A en la mano. Las entradas B1‑B4 de este doc son el borrador de partida de ese re-scoping, no un compromiso de diseño.
- **Narración didáctica PER-CICLO, no batch:** cada sub-ciclo A/B incluye su traza `--steps` narrada (es/en) como parte de su `graduates` — el engine es mitad educativo, y dejar las reglas nacer mudas para narrarlas en un batch final repetiría el error histórico de "límites a 0% educativo". **C2 queda reducido a pulido/localización**, no a deuda acumulada. (Las 7 reglas Gaussianas existentes ya emiten descripción vía `format_complex_rewrite_desc` — el listón es mantener ese contrato en cada regla nueva y elevarlo donde la narración sea cáscara.)

Dependencias netas: A5 depende de A4; B1 antes de B2/B3; B4 tras B2+B3; A1/A2/A4 independientes entre sí.

---

## Riesgos (trampas a evitar)

- **Soundness principal-branch (B3 sobre todo):** una rama mal elegida = **WRONG-ANSWER, no residual**. `Log/Arg` deben respetar `Arg(z) ∈ (-π, π]`; `z^w` (B4) hereda el corte. Sin B1 (probe complejo) estas reglas no tienen verificador independiente → **declinar antes que emitir sin red**.
- **Huella (byte-identidad en modo REAL):** toda regla net-new (A1, A2, B2, B3, B4) DEBE auto-gatearse `value_domain()==RealOnly→None` como las 7 existentes; los cambios de solve (A4, A5) gateados por `is_real_only`/`ComplexEnabled`. Pins que lo verifican: `const_fold_contract_tests.rs`, `semantics_contract_tests.rs`, `complex_tests.rs`.
- **Exactitud de la red numérica (B1):** el probe `Complex<f64>` SOLO puede REFUTAR, nunca CONFIRMAR ([[soundness-gates-must-be-exact]] + `actions.rs:450`). Cablearlo para confirmar = unsound. El confirm queda en la capa exacta ℚ[i]/multipoly-mod-`(i²+1)` (A3).
- **Regresión de completitud en solve (A5, P0):** cambiar `count_real_roots()==0` (Sturm) por "grado" en `extract_candidate_roots:841` puede declarar "completo con 0 raíces" un residual irreducible grado≥3 → subconjunto silencioso (memoria F4 = P0). Verificar **decline honesto vs. emisión completa**.
- **Blast de firma en A5/B1:** añadir `value_domain` a los kernels de `rational_roots` toca múltiples llamadores internos + tests; `eval_f64` se re-exporta en `cas_engine/api.rs:79` y `cas_solver/api.rs:22` → B1 debe añadir `eval_complex` **PARALELO**, NO ensanchar `eval_f64` (rompe ~8 sitios de `solve_backend_local` + round-trip/metamorphic).
- **`builtin.rs` (A2, B3):** un builtin nuevo requiere 5 ediciones sincronizadas (enum/name/from_name/ALL_BUILTINS/COUNT) o el fallo es **SILENCIOSO**; + gate `is_known_eval_engine_function` `eval.rs:68` (+ posible allowlist `budget_exempt`, memoria [[new-engine-function-wiring-gotchas]]).
- **Loop/colisión de reglas (A1):** `GaussianPowRule` debe materializar el gaussiano ya multiplicado (`to_expr`) para no re-matchear su propio `Pow`, y excluir base pura-`i` (guard) para no oscilar/duplicar con `ImaginaryPowerRule`; nombre único (`engine/simplifier.rs:539`).
- **Fast-path que hardcodea RealOnly:** `cas_solver/src/eval_command_runtime/prepare.rs:48‑61` fuerza `ValueDomain::RealOnly` en `infer_implicit_domain` del path cacheado; verificar que ningún frente complejo dependa de esa rama o cablear `config.value_domain`.
- **`z^w`/Euler/Log NO caben en `GaussianRational`** (solo `BigRational`): B2/B3/B4 emiten forma simbólica (`cos+i·sin` / `ln|z|+i·Arg`) y/o el eval complejo de B1 — no forzar el tipo racional. **Multivaluado/Riemann FUERA DE SCOPE siempre.**

---

## Guardrails inter-fase aplicados al frente complejo

1. **`ValueDomain`-threading** (CUMPLIDO parcial): el hilo llega intacto hasta `apply.rs:158` y `parent_ctx.value_domain()` (`parent_context.rs:248`). RESPETAR: todo consumidor net-new lee de ahí; **NO reintroducir hardcodes** (vigilar `prepare.rs:48‑61`). Cada regla nueva auto-gateada como las 7 existentes.
2. **diff/integrate per-variable** (CUMPLIDO): respetan variable + dominio. RESPETAR la firma per-variable si se añade integración compleja.
3. **Predicados de condición ESTRUCTURADOS** (NO cumplido para complejo): `branch='principal'` es solo etiqueta; `required_conditions` (`finalize.rs:26`) no se puebla. **CABLEAR en B4** (decisión 2026-07-17: un solo cableado para los dos emisores — Log/Arg de B3 ya aterrizados le atachan la condición retroactivamente en el mismo ciclo): un resultado que dependa de un corte de rama emite su condición estructurada, no prosa.
*(re-audit 2026-07-18: CUMPLIDO desde B4a: `ConditionPredicate::PrincipalBranch` end-to-end, `required_conditions` poblado en el wire por ambos emisores)*
4. **Backstop de soundness EXACTO** (PARCIAL): `prove_nonzero(i)`/`prove_positive(i)` exactos. PERO `solve(x^4-1)`--complex dropea `±i` en silencio (conjunto incompleto = violación). **Regla para A4/A5/B2/B3:** en complejo un DECLINE honesto es preferible a un conjunto/valor incompleto; nunca subconjunto sin avisar; el probe B1 solo refuta.
5. **Resultado-como-contrato** (CUMPLIDO): el sobre `EvalWireOutput` expone `semantics{value_domain, branch, …}` estable y re-enterable (`wire_types.rs:463`). RESPETAR: todo resultado complejo debe re-parsear como input (forma `a+bi`, `±i`).

---

## Cómo ejecutar

Cada sub-ciclo es un `/auto-mejora 1` (o encadenar `/auto-mejora N`). Marca aquí `☑` con el hash del commit al graduar. El criterio de "frente complejo cerrado" son los 18 probes de la tabla de frontera verdes + verificados (algebraicos por ℚ[i], transcendentales por el probe refute-only de B1). Los ciclos A gradúan la C-álgebra y el solve complejo (F12); los B, lo transcendental; los C, presentación y pedagogía.

> **Nota de disciplina (hash-stamps):** regla del ledger de G1 — nunca estampar el hash del PROPIO commit vía amend (el amend lo invalida); estampar el hash de un ciclo en el commit del ciclo SIGUIENTE, o citar "hash en el ledger".
