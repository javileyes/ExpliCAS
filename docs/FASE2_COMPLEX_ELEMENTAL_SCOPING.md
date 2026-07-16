# Fase 2 · Frente Complejo elemental (principal-branch): scoping en sub-ciclos acotados

- **Fecha:** 2026-07-16
- **HEAD:** `e069031f2`
- **Clase:** M total (dos evaluaciones de audit lo estimaron ≈M sin reescritura fundamental). Se entra como **secuencia de sub-ciclos acotados**, nunca como un solo ciclo.
- **Método:** scoping workflow READ-ONLY (6 mappers de subsistema convergentes + síntesis + **doble verificación adversarial**: verificador de anclas `file:line` en vivo + crítico de completitud). 9 agentes, 0 errores. Toda ancla verificada contra el árbol y toda fila de frontera re-sondeada con el binario fresco. Journal crudo: `subagents/workflows/wf_b05c3772-7f9/journal.jsonl`.
- **Relacionado:** `docs/CALCULUS_ENGINE_DEVELOPMENT_PHASES.md` (Fase 2, líneas ~176‑201 — los tres items del frente), `docs/ENGINE_VS_SYMPY_ASSESSMENT_2026-07-14.md` (fila F12: `solve(x^2+1=0)` complejo = el gap emblemático), `docs/G1_RATIONAL_INTEGRATION_SCOPING.md` (molde de formato: C-i/E-i = "primitivo duro reusable, huella byte-idéntica, antes de tocar lo caro").

Abrir este frente gradúa la mitad **compleja** de la Fase 2 (la otra mitad, vectorial multivariable, es un frente separado). Modelo: **single-valued PRINCIPAL BRANCH** (no Riemann, no multivaluado — eso está fuera del norte para siempre).

---

## La frontera exacta (probes verificados en vivo, HEAD `e069031f2`)

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
- **CHOKEPOINT-C · solve.** Los kernels de generación de conjunto-solución son **real-only** y descartan las raíces `√Δ<0` **aguas arriba** (`solution_set.rs:527` `Eq→Empty`; `rational_roots.rs:961` `else vec![]`), **antes** del gate `drop-non-real` que **ya es domain-aware** (`solve_backend_local.rs:11124`). Las expresiones-raíz `(-b±√Δ)/2a` se **construyen** (`quadratic_formula.rs:38‑46`, `sqrt_expr` incondicional) y se **tiran** sin mirar el dominio. El render complejo ya emite `±i`; el fix = enhebrar `value_domain` hasta esos kernels y no descartar bajo `ComplexEnabled` → **sub-ciclos A4/A5**.

---

## Secuencia de sub-ciclos (cada uno = un `/auto-mejora`, un commit)

Orden por **dependencia + blast**: el más pequeño/zero-blast/reusable primero (análogo C-i/E-i de G1). Tres bloques: **A algebraico-exacto** (sin dep nueva, verificable por ℚ[i]), **B transcendental** (necesita la red numérica), **C presentación/pedagogía** (transversal).

### Bloque A — algebraico exacto (sin dependencia nueva)

#### ☐ A1 — Potencia entera Gaussiana `(a+bi)^n` **[S] — PRIMER CICLO**
- **Gradúa:** `(1+i)^2 → 2·i`, `(1+i)^3 → 2·i-2`, `(2+i)^4`, `(3+4i)^2`.
- **Inserción:** `complex_support.rs:85` (variante `ComplexRewriteKind::GaussianPower`, antes del `}` en `:86`); `complex_support.rs:70` (impl `GaussianRational`: métodos `mul()/pow()` exactos sobre `BigRational` — hoy CERO ops son método); `complex_support.rs:~405` (`try_rewrite_gaussian_power_expr`, con **guard `g.is_real() || g.is_pure_imag() → None`** para no colisionar con `ImaginaryPowerRule`, exponente entero ≥0); `complex.rs:24` (brazo desc); `complex.rs:114` (`GaussianPowRule`, gate `RealOnly→None`); `complex.rs:123` (`add_rule` en `register()`).
- **Reuso:** `extract_gaussian`, `to_expr`, fórmula mul inline `:239‑244`, `define_rule!` domain-aware.
- **Net-new:** variante enum + free-fn + los primeros métodos `mul/pow` de `GaussianRational` (el primitivo reusable que A2/A4/A5 aprovechan).
- **Blast:** **BAJO** — auto-gateado (real-mode byte-idéntico). Trampas: (1) loop si re-emite `Pow` → materializar el gaussiano **ya multiplicado** vía `to_expr`; (2) colisión con `ImaginaryPowerRule` → el guard excluye base pura-`i`; (3) nombre de regla único (`assert_unique_rule_names` `engine/simplifier.rs:539`).
- **Depende:** nada.
- **Retención:** `complex_tests.rs:130` pinea `to_expr == '3 + 2 * i'` (contrato de orden `a+bi`) — respetar. Workspace verde; huella real byte-idéntica.
- **Residual conocido (peldaño, no bloquea):** exponente **negativo** `(1+i)^(-n)` — hoy `(1+i)^(-1)` devuelve el parcial sin terminar `(1/2·2 - i)/(2)`; A1 scopea `n≥0`. Extensión opcional: potencia positiva + recíproco Gaussiano (o dejar como residual honesto hasta C1-display).

#### ☐ A2 — Módulo `|a+bi|` + builtins `Re`/`Im`/`conjugate` (exactos ℚ[i]) **[M]**
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

#### ☐ A4 — Solve complejo cuadrático desnudo (F12, `Δ<0`) **[S]**
- **Gradúa:** `solve(x^2+1, x) → {i,-i}`, `solve(x^2+x+1, x) → {-1/2 ± √3/2·i}`, `solve(x^2-2x+5, x) → {1±2i}`.
- **Inserción:** `solution_set.rs:449` (añadir `value_domain` a la firma) y `:527‑529` (rama `else Δ<0`: si `ComplexEnabled ∧ op==Eq → Discrete(vec![r1,r2])` en vez de `Empty`); `quadratic_formula.rs:244` (propagar dominio) y `:225‑234` (invertir el guard de Δ-simbólica-negativa SOLO en `ComplexEnabled`); `quadratic_strategy.rs:127` (pasar `is_real_only` al plan).
- **Reuso:** `roots_from_a_b_delta` `quadratic_formula.rs:38‑46` (ya construye `√Δ` negativo con `sqrt_expr` incondicional), render complejo ya emite `±i`, gate `drop-non-real` `solve_backend_local.rs:11124` ya domain-aware (**NO tocar**).
- **Blast:** **BAJO-MEDIO** — el cambio de firma toca ~5 llamadores de `quadratic_numeric_solution` (4 tests `solution_set.rs:1565+` + `quadratic_formula.rs:247`). **SCOPE-OUT:** inecuaciones `Δ<0` (orden indefinido en ℂ) → cambiar SOLO la rama `RelOp::Eq`.
- **Depende:** nada duro (el `bool is_real_only` ya llega a la frontera de la estrategia).
- **Retención:** los tests reales que esperan `Empty` en modo real siguen verdes — fix **gateado SOLO en `ComplexEnabled`**. (Nota de verificación: `solution_set.rs:1565` es en realidad un test `Δ>0`; no hay test dedicado que pinee `Empty` para `x²+1 ∧ Δ<0∧Eq` — cablear uno nuevo al graduar.)

#### ☐ A5 — Solve complejo grado ≥3 (deflación + par conjugado) **[L]**
- **Gradúa:** `solve(x^4-1, x) → {-1,1,-i,i}`, `solve(x^3-1, x) → {1, -1/2 ± √3/2·i}`.
- **Inserción:** `rational_roots.rs:931/961` (`solve_residual_degree_leq_two`: `value_domain` + rama `Δ<0 → roots_from_a_b_delta`); `rational_roots.rs:851/857` (`solve_residual_biquadratic`: `z<0 → ±i√|z|`); `rational_roots.rs:810/835` (`extract_candidate_roots`: `value_domain` + **completitud contada por GRADO, no `count_real_roots` Sturm**); `cas_solver_core/src/solve_runtime_flow_strategy_kernels_equation.rs:28` (origen del enhebrado; `is_real_only` ya disponible); `solve_backend_local.rs:11724/:12354` (helpers cuártico/cúbico: emitir par conjugado o ceder a la ruta general).
- **Reuso:** los cambios de kernel de A4 + `roots_from_a_b_delta`; render `±i`.
- **Blast:** **ALTO** — la firma toca múltiples llamadores internos + tests `rational_roots.rs:1000+`; el verificador f64 de raíces (`solve_backend_local.rs:11757`, `eval_f64` rechaza `i`) **no vale** para complejas → saltar (las raíces de cuadráticas racionales son exactas por construcción).
- **Depende:** **A4.**
- **Retención (REGRESIÓN P0 de completitud):** `extract_candidate_roots:841` usa `count_real_roots()==0` — al cambiar a "grado" verificar que **NO se declare "completo con 0 raíces"** un residual irreducible grado≥3 (memoria [[frontier-audit-2026-07-13b-8-familias]] F4: declarar completo un subconjunto = **P0 wrong-answer**). Modo real `x^4-1→{-1,1}`, `x^3-1→{1}` intactos.

### Bloque B — transcendental (necesita la red numérica)

#### ☐ B1 — Evaluador numérico complejo (`Complex<f64>`, **refute-only**) **[M] — el primitivo E-i del bloque**
- **Gradúa:** unit-tests de `eval_complex` (exp/ln/powc principal-branch) verdes + probe refute-only cableado en la equivalencia; `approx(abs(i)) → 1.0`.
- **Inserción:** `cas_math/Cargo.toml` (dep NUEVA `num-complex = "0.4"` — hoy AUSENTE; `num-rational` `:17` y `num-bigint` `:19` son 0.4, `num-traits` `:18` es 0.2, `num-integer` `:20` es 0.1); `evaluator_f64.rs:534` (añadir `eval_complex` **PARALELO, NO ensanchar la firma de `eval_f64`**); `numeric_eval.rs:290` (`numeric_poly_zero_check`: fallback probe complejo **refute-only**); `engine/equivalence.rs:78/:344` (fallback numérico complejo refute-only; `:65` confirm exacto sigue en ℚ[i]).
- **Decisión abierta (evaluar al re-scopear el bloque B):** `num-complex` como dep vs. `Complex<f64>` hand-rolled (~50 líneas: add/mul/exp/ln/powc/sqrt principal-branch). El repo es soundness-first y casi todo hand-rolled; el probe es refute-only (tolerante a f64), así que ambas vías son sound — decidir por mantenimiento, no por soundness.
- **Reuso:** la asimetría de exactitud YA codificada `eval/actions.rs:450` ("un probe puede REFUTAR, nunca CONFIRMAR"); `numeric_poly_zero_check` ya la implementa para reales. `num-complex` trae `exp/ln/powc/sqrt` principal-branch out-of-the-box.
- **Blast:** **MEDIO** — add PARALELO (no widening). `eval_f64` se re-exporta en `cas_engine/api.rs:79`, `cas_solver/api.rs:22` y ~8 sitios de `solve_backend_local` → **NO cambiar su firma** (rompería suites round-trip/metamorphic).
- **Depende:** nada duro, pero **DEBE ir antes de B2/B3/B4**.
- **REGLA DE ORO (soundness):** el probe complejo **SOLO refuta** (`false`), **JAMÁS confirma** (memoria [[soundness-gates-must-be-exact]] + `actions.rs:450`); el confirm queda en la capa exacta ℚ[i]/multipoly-mod-`(i²+1)` (A3).
- **Nota:** era la mitad "M4b" del ciclo `C5` original; **la verificación separó M4a (A3, exacto, sin dep) de M4b (B1, num-complex, transcendental)** por tener dep y consumidores distintos.

#### ☐ B2 — Fórmula de Euler `e^(iθ)=cos θ + i·sin θ` (y `e^(a+bi)`) **[M]**
- **Gradúa:** `e^(i*pi) → -1`, `e^(i*pi/2) → i`, `exp(i*x) → cos(x)+i·sin(x)`.
- **Inserción:** `complex.rs:116` (`EulerRule`, gate `ComplexEnabled`, modelada EXACTAMENTE sobre `ComplexNegativeBaseRootRule` `power_rules.rs:402/435‑448`); `complex_support.rs:~405` (`try_rewrite_euler_expr`: `extract_gaussian(exp)=a+bi → e^a(cos b + i·sin b)`).
- **⚠️ CORRECCIÓN DE DISEÑO (verificación de anclas):** `ExpToEPowRule` (`canonicalization.rs:291`) está **gateada `!= RealOnly → None`** (`:307`, con comentario explícito: en complejo `exp` se mantiene univaluada). Por tanto en modo complejo `exp(iθ)` queda como **`Function(exp,·)`** mientras `e^(iθ)` es **`Pow(E,·)`** — **dos formas AST distintas**. La premisa original "una regla cubre ambas" es **falsa**: `EulerRule` **debe casar AMBAS** (`Function(exp,·)` Y `Pow(E, ·)`), o despachar sobre el argumento sin importar el envoltorio. (Ambas imprimen `e^…` pero es elección del formatter, no identidad estructural.)
- **Reuso:** `cos/sin` de π-racional se pliegan solos; el verificador refute de B1.
- **Blast:** **MEDIO** — gate `ComplexEnabled` obligatorio (Euler NO es domain-neutral). La verificación depende de B1.
- **Depende:** **B1** (único verificador independiente de identidades a-nivel-valor: `e^(iπ)=-1` no tiene variable).
- **Sub-scope surfaceado:** `e^(a+bi)` general (`a≠0`, p.ej. `e^(1+i)`) necesita `e^a` numérico/simbólico y su verificación cabalga sobre B1 — cablearlo explícitamente (el caso puro `a=0` es limpio, `e^a=1`).
- **Retención:** modo real `e^x` intacto (regla inerte en `RealOnly`).

#### ☐ B3 — Logaritmo principal `ln(z)=ln|z|+i·Arg(z)` + builtin `Arg` **[M]**
- **Gradúa:** `ln(-1) → i·π`, `ln(-2) → ln 2 + i·π`, `ln(i) → i·π/2`, `Arg(i) → π/2`, `Arg(-1) → π`.
- **Inserción:** `logarithms/mod.rs:52` (`ComplexLogRule` **estructurada con `parent_ctx`** — `EvaluateLogRule` usa `define_rule!` corto SIN `parent_ctx`, no ve el dominio; prioridad > `EvaluateLogRule` en `register()` `:179`); `logarithm_inverse_support.rs:375` (dejar el decline `n<0→Undefined` intacto; regla nueva ANTES); `builtin.rs` (`Arg` + `COUNT` bump) + `eval.rs:68`; despacho `Arg` vía `atan2` en `complex.rs:116`; `eval_command_runtime/present/finalize.rs:26` (**poblar `required_conditions` de corte de rama** — guardrail #3).
- **Reuso:** la forma `ln 2 + i·π` ya es foldable; template `ComplexNegativeBaseRootRule`; verificador B1.
- **Blast:** **MEDIO-ALTO** — `EvaluateLogRule` corto no gatea por dominio → regla estructurada nueva. **SOUNDNESS principal-branch: una rama mal elegida = WRONG-ANSWER, no residual** (`Arg(z) ∈ (-π, π]`).
- **Depende:** **B1** (`Arg=atan2` transcendental).
- **Retención:** modo real `ln(-1)→undefined` SIGUE (gateado); el decline real con sugerencia de modo complejo (`cas_cli/tests/solver_domain_contract_tests.rs`) debe seguir vivo.

#### ☐ B4 — Potencia compleja general `z^w = e^(w·Log z)` **[M]**
- **Gradúa:** `i^i → e^(-π/2)`, `2^i → cos(ln 2)+i·sin(ln 2)`.
- **Inserción:** `rules/exponents/mod.rs:33` (`ComplexGeneralPowerRule`, junto a `ComplexNegativeBaseRootRule`, gate `ComplexEnabled`); `const_fold/helpers.rs:205` (slot `_branch` "Wired for future use" — al ampliar `z^w` branch-aware).
- **Reuso:** `EulerRule` (B2) + `ComplexLogRule` (B3); verificador B1.
- **Blast:** **MEDIO** — gate `ComplexEnabled`. **Hereda el corte de rama de B3** → la condición estructurada del guardrail #3 debe propagarse aquí (no quedar solo en B3).
- **Depende:** **B2 + B3.**
- **Retención:** `i^n` desnudo sigue en `ImaginaryPowerRule`; `(a+bi)^n` entero en A1 (no re-rutar por `z^w`).

### Bloque C — presentación / pedagogía (transversal)

#### ☐ C1 — Normalización de forma cartesiana `a+bi` **[S]**
- **Gradúa:** salida en orden canónico `-1 + 2·i` (hoy `2·i - 1`, imag-first / como resta); limpia el parcial sin terminar `(1+i)^(-1) → (1/2·2 - i)/(2)`.
- **Inserción:** normalizador de orden cartesiano en la capa de display/`to_expr`; migrar conscientemente el pin `complex_tests.rs:130` (`'3 + 2 * i'`).
- **Blast:** **BAJO** (cosmético) pero interactúa con pins de formato → migración consciente.
- **Depende:** nada; ROI mayor tras A1/A2.

#### ☐ C2 — Pulido/localización de la narración compleja **[S — reducido en la revisión 2026-07-16]**
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
3. **Predicados de condición ESTRUCTURADOS** (NO cumplido para complejo): `branch='principal'` es solo etiqueta; `required_conditions` (`finalize.rs:26`) no se puebla. **CABLEAR en B3 (Log/Arg) y B4 (z^w):** un resultado que dependa de un corte de rama emite su condición estructurada, no prosa.
4. **Backstop de soundness EXACTO** (PARCIAL): `prove_nonzero(i)`/`prove_positive(i)` exactos. PERO `solve(x^4-1)`--complex dropea `±i` en silencio (conjunto incompleto = violación). **Regla para A4/A5/B2/B3:** en complejo un DECLINE honesto es preferible a un conjunto/valor incompleto; nunca subconjunto sin avisar; el probe B1 solo refuta.
5. **Resultado-como-contrato** (CUMPLIDO): el sobre `EvalWireOutput` expone `semantics{value_domain, branch, …}` estable y re-enterable (`wire_types.rs:463`). RESPETAR: todo resultado complejo debe re-parsear como input (forma `a+bi`, `±i`).

---

## Cómo ejecutar

Cada sub-ciclo es un `/auto-mejora 1` (o encadenar `/auto-mejora N`). Marca aquí `☑` con el hash del commit al graduar. El criterio de "frente complejo cerrado" son los 18 probes de la tabla de frontera verdes + verificados (algebraicos por ℚ[i], transcendentales por el probe refute-only de B1). Los ciclos A gradúan la C-álgebra y el solve complejo (F12); los B, lo transcendental; los C, presentación y pedagogía.

> **Nota de disciplina (hash-stamps):** regla del ledger de G1 — nunca estampar el hash del PROPIO commit vía amend (el amend lo invalida); estampar el hash de un ciclo en el commit del ciclo SIGUIENTE, o citar "hash en el ledger".
