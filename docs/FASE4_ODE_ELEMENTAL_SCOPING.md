# Fase 4 · EDOs elementales (`dsolve`): scoping en sub-ciclos acotados

- **Fecha:** 2026-07-19
- **Decisión del usuario:** 2026-07-19 — las EDOs ELEMENTALES entran al north star como **Fase 4** ("engine realmente universal Y educativo"). Commit del cambio de norte: `cb2e0c80e`. Supersede el scope-out de EDOs del doc Fase 3 (su nota de actualización 2026-07-19 ya apunta aquí).
- **Clase:** **MIXTA, mayormente S/M de orquestación sobre maquinaria viva — con UN costo fijo L (O0) y DOS hallazgos estructurales que mandan sobre todo el diseño.** El oráculo curricular cierra 36/39 verificaciones por sustitución HOY sin escribir una línea (mapper 5); los kernels de `integrate` que cada método necesita cierran TODOS; exactas tiene su corazón YA construido (`try_potential_expr`, F6); la característica de 2º orden tiene el molde de discriminante exacto vivo. El trabajo neto-nuevo real: el sustrato wire+acción (O0), el collector-por-base de coeficientes indeterminados (O5) y el álgebra 2×2 compleja/defectiva de sistemas (O6).
- **Hallazgo estructural #1 (verificado a mano, re-confirmado por 4 mappers):** la maquinaria actual **COLAPSA la notación de EDOs** — `eval "diff(y,x)"` → `0` (y tratada como constante: brazo `Expr::Variable` de `differentiate_symbolic_expr`, `crates/cas_math/src/symbolic_differentiation_support.rs:6513-6518`) y `eval "solve(diff(y,x)=y, y)"` → `{ 0 }` (DiffRule mata la EDO antes del solver). **La intercepción a nivel WIRE es obligatoria**: `dsolve` debe capturar el texto crudo ANTES de que simplify toque el árbol (molde exacto: `parse_solve_command`/`parse_solve_system_list_command`, `crates/cas_api_models/src/wire_types.rs:625/:708`).
- **Hallazgo estructural #2:** `dsolve` debe ser **ACCIÓN (`EvalSpecialCommand` → `EvalAction`), no rule** — la verificación por sustitución necesita el evaluador COMPLETO (`eval_simplify` vía el molde `equiv_difference_evaluates_to_zero`, `crates/cas_engine/src/eval/actions.rs:496-509`) y el canal de warnings de la capa de acción (`Vec<DomainWarning>`, `crates/cas_engine/src/eval/mod.rs:16-25`). Consecuencia (verificada por mapper 6 contra el gate): como special-command, `dsolve` **NO entra** en `is_known_eval_engine_function` ni necesita el flag por-regla `budget_exempt` (`rule.rs:187` — es de Rewrites, y una acción no pasa por reglas) — `solve` tampoco está (el gate `crates/cas_session_core/src/eval.rs:68-139` solo lista `linsolve`).
- **Journal:** `subagents/workflows/wf_d06339c8-cec/journal.jsonl` (9 agentes, 0 errores). Doble verificación adversarial: verificador de anclas (98 verificadas, 8 correcciones aplicadas — 0 FATAL) + crítico de completitud (13 gaps integrados).
- **Método:** scoping workflow READ-ONLY, 6 mappers convergentes (representación-wire, métodos-vs-maquinaria, verificación, contrato-salida, currículo-oráculo, huella-gobernanza) + síntesis. Todas las anclas file:line verificadas en sesión; probes contra `./target/release/cas_cli` del día. ⚠ `wire_types.rs` y `cas_session_core/src/eval.rs` están git-modified (F7/F9 de Fase 3 en vuelo) — **re-anclar líneas al abrir O0**.
- **Relacionado:** `docs/FASE3_ANALYTIC_LAYERS_SCOPING.md` (molde de formato; decisión D10 del never-confirm; doctrina F6 verificación-gatea-emisión), `docs/FASE2_VECTORIAL_MULTIVARIABLE_SCOPING.md` (patrón verbo-sobre-máquina-viva), `docs/CALCULUS_ENGINE_DEVELOPMENT_PHASES.md` (§Fase 4 tras cb2e0c80e).

**SCOPE-OUT explícito (FUERA de esta fase):** funciones especiales como salida (`erf`/`Γ`/Bessel/Airy/elípticas — sus presas never-confirm SIGUEN vigiladas); EDOs no-lineales sin método clásico (catálogo Z abajo — su decline residual ES contrato); transformada de Laplace; factor integrante μ(x,y) general (μ(x) simple = extensión opcional nombrada); series de Frobenius; existencia/unicidad (Picard) como capacidad; **reducción de orden** (y'' sin y / autónomas) y **variación de parámetros** (RHS fuera de la tabla UC declina con ese nombre); azúcar de notación `y'` en la ECUACIÓN (solo en condiciones iniciales, donde el scanner textual del wire lo controla — ver D1).

---

## La frontera exacta: catálogo curricular (~35 graduate + 7 never-fabricate)

> Verificación del oráculo (mapper 5): candidata sustituida en la EDO, residuo `LHS−RHS` por el evaluador actual → **36/39 cierran a `0` exacto HOY**. Los 3 no-cierres (H19, O23, N27) tienen causa aislada y diseño de mitigación (ver D5 y riesgos). Sintaxis normalizada a la forma canónica D1; 2º orden SIEMPRE `diff(y,x,2)` (la forma `diff(y,x,x)` con `y` desnuda tropieza el guard de ambigüedad `symbolic_calculus_call_support.rs:169-179` — declina).

### A. Separables (ciclo O0)

| # | Probe | Método | Solución esperada | Oráculo |
|---|---|---|---|---|
| S1 | `dsolve(diff(y,x)=x*y, y, x)` | separable | `y = C·e^(x²/2)` | 0 ✓ |
| S2 | `dsolve(diff(y,x)=y^2, y, x)` | separable | `y = −1/(x+C)` | 0 ✓ |
| S3 | `dsolve(diff(y,x)=1+y^2, y, x)` | separable | `y = tan(x+C)` | 0 ✓ |
| S4 | `dsolve(diff(y,x)=-x/y, y, x)` | separable (implícita) | `x²+y² = C` | 0 ✓ |
| S5 | `dsolve(diff(y,x)=x/y, y, x)` | separable | `y = √(x²+C)` | 0 ✓ |
| S6 | `dsolve(diff(y,x)=y/x, y, x)` | separable | `y = C·x` | 0 ✓ |
| S7 | `dsolve(diff(y,x)=2*x*y^2, y, x)` | separable | `y = −1/(x²+C)` | 0 ✓ |
| L12 | `dsolve(diff(y,x)=-y, y, x)` | separable/lineal | `y = C·e^(−x)` | 0 ✓ |
| S8 | `dsolve(diff(y,x)=k*y, y, x)` | separable (param) | `y = C·e^(kx)` — el staple crecimiento/decaimiento; la verificación con `k` simbólico debe cerrar a 0 | 0 ✓ |
| S9 | `dsolve(diff(y,x)=cos(x), y, x)` | integración directa (g(y)=1) | `y = sin(x)+C` | 0 ✓ |

Kernels verificados: `integrate(1/y,y)`→`ln(|y|)`, `integrate(1/(1+y^2),y)`→`arctan(y)`, `integrate(y^-2,y)`→`−1/y`; el solve inverso maneja el ± del abs: `solve(ln(abs(y))=x+C,y)`→`{e^(C+x), −e^(C+x)}`.

### B. Lineal 1er orden — factor integrante (ciclo O1)

| # | Probe | Método | Solución esperada | Oráculo |
|---|---|---|---|---|
| L8 | `dsolve(diff(y,x)+y=x, y, x)` | μ=e^x | `y = x−1+C·e^(−x)` | 0 ✓ |
| L9 | `dsolve(diff(y,x)+y/x=x^2, y, x)` | μ=x (wrinkle μ=\|x\|, ver D12) | `y = x³/4 + C/x` | 0 ✓ |
| L10 | `dsolve(diff(y,x)-y=exp(x), y, x)` | resonancia 1er orden | `y = (x+C)·e^x` | 0 ✓ |
| L11 | `dsolve(diff(y,x)+2*y=sin(x), y, x)` | μ=e^(2x) | `y = (2sin x − cos x)/5 + C·e^(−2x)` | 0 ✓ |

Kernels: `integrate(x*e^x,x)`→`(x−1)e^x`; `integrate(sin(2*x)*e^(3*x),x)` exacto; `integrate(exp(2*x)*sin(x),x)`→`(1/5)e^(2x)(2sin x−cos x)`.

### C. Exactas — `try_potential_expr` reutilizado (ciclo O2)

| # | Probe | Método | Solución (implícita) | Oráculo |
|---|---|---|---|---|
| E13 | `dsolve(2*x*y + x^2*diff(y,x) = 0, y, x)` | exacta | `x²y = C` (fixture F6 vivo: `vector_calculus.rs:634-639`) | parciales 0 ✓ |
| E14 | `dsolve((2*x*y+1) + (x^2+2*y)*diff(y,x) = 0, y, x)` | exacta | `x²y + x + y² = C` | 0 ✓ |
| E15 | `dsolve((3*x^2+2*y) + (2*x+3*y^2)*diff(y,x) = 0, y, x)` | exacta | `x³ + 2xy + y³ = C` | 0 ✓ |
| E-neg | `dsolve(y + 2*x*diff(y,x) = 0, y, x)` forma M+N·y′ no exacta | — | **decline del camino exacto** (gate `poly_eq` no verifica — GRATIS) → cae a separable si aplica, o residual honesto | contrato |

### D. IVP — condiciones iniciales (ciclo O3; V31 gradúa en O4)

| # | Probe | Método | Solución esperada | Oráculo |
|---|---|---|---|---|
| V30 | `dsolve(diff(y,x)=-y, y, x, y(0)=3)` | separable+IVP | `y = 3·e^(−x)` | 0 ✓ |
| V32 | `dsolve(diff(y,x)=x*y, y, x, y(0)=2)` | separable+IVP | `y = 2·e^(x²/2)` | 0 ✓ |
| V33 | `dsolve(diff(y,x)+y=x, y, x, y(0)=0)` | lineal+IVP | `y = x−1+e^(−x)` | 0 ✓ |
| V31 | `dsolve(diff(y,x,2)+4*y=0, y, x, y(0)=0, y'(0)=2)` | 2º orden+IVP | `y = sin(2x)` | 0 ✓ |

Primitivas verificadas: `subs(x-1+C*exp(-x), x, 0)`→`C−1`; `solve(C*e^(-2)=5, C)`→`{5·e^2}`; paramétrico `solve(C*e^(-a)=b, C)`→`{b·e^a}`. **Gap conocido**: `solve_system` rechaza `c1`/`C1` (`Invalid variable name` — validador `crates/cas_solver/src/linear_system_command_parse/vars.rs:18-20` exige `is_alphabetic() || '_'` — sin dígitos) → fix D16.

### E. 2º orden lineal homogénea coef. constantes (ciclo O4)

| # | Probe | Raíces caract. | Solución esperada | Oráculo |
|---|---|---|---|---|
| O20 | `dsolve(diff(y,x,2)-y=0, y, x)` | ±1 | `C1·e^x + C2·e^(−x)` | 0 ✓ |
| O21 | `dsolve(diff(y,x,2)+4*y=0, y, x)` | ±2i | `C1·sin(2x) + C2·cos(2x)` | 0 ✓ |
| O22 | `dsolve(diff(y,x,2)+2*diff(y,x)+y=0, y, x)` | doble −1 | `(C1+C2·x)·e^(−x)` | 0 ✓ |
| O23 | `dsolve(diff(y,x,2)+2*diff(y,x)+5*y=0, y, x)` | complejas −1±2i | `e^(−x)(C1·sin(2x)+C2·cos(2x))` | **HANG con C1/C2; base sin constantes → 0 ✓** (D5-linealidad) |
| O24 | `dsolve(diff(y,x,2)-3*diff(y,x)+2*y=0, y, x)` | 1, 2 | `C1·e^x + C2·e^(2x)` | 0 ✓ |
| O25 | `dsolve(diff(y,x,2)=0, y, x)` | doble 0 | `C1 + C2·x` | 0 ✓ |

Hechos del solve característico (mapper 5): `solve(r^2+2*r+1=0,r)`→`{−1}` (el SET **colapsa multiplicidad**); raíces complejas SOLO bajo `--value-domain complex` (default real → "No solution") → por eso D9: discriminante exacto interno, jamás solve desnudo.

### F. 2º orden no-homogénea — coeficientes indeterminados (ciclo O5)

| # | Probe | Método | Solución esperada | Oráculo |
|---|---|---|---|---|
| N26 | `dsolve(diff(y,x,2)+y=x, y, x)` | UC polinomio | `x + C1·sin x + C2·cos x` | 0 ✓ |
| N27 | `dsolve(diff(y,x,2)-y=exp(2*x), y, x)` | UC exponencial | `C1·e^x + C2·e^(−x) + e^(2x)/3` | 0 ✓ (WARN depth transitorio en stderr — falso-rojo) |
| N28 | `dsolve(diff(y,x,2)+y=cos(x), y, x)` | UC **resonancia** | `(x/2)·sin x + C1·sin x + C2·cos x` | 0 ✓ |
| N29 | `dsolve(diff(y,x,2)-3*diff(y,x)+2*y=exp(x), y, x)` | UC resonancia raíz simple | `C1·e^x + C2·e^(2x) − x·e^x` | 0 ✓ |

Primitivas: `diff(A*x*e^(2*x),x)`→`e^(2x)(2Ax+A)`; `solve_system(A+B=1; 2*A-B=0; A; B)`→`{A=1/3, B=2/3}` (3×3 también, `examples.csv:84`).

### G. Bernoulli y homogéneas-sustitución (ciclo O8 — graduate, NO opcional)

| # | Probe | Método | Solución esperada | Oráculo |
|---|---|---|---|---|
| B16 | `dsolve(diff(y,x)+y=y^2, y, x)` | Bernoulli n=2 | `y = 1/(1+C·e^x)` | 0 ✓ |
| B17 | `dsolve(diff(y,x)+y/x=x*y^2, y, x)` | Bernoulli n=2 | `y = 1/(Cx−x²)` (re-verificar a mano en el ciclo) | 0 ✓ |
| H18 | `dsolve(diff(y,x)=(x+y)/x, y, x)` | homogénea v=y/x | `y = x·ln(x) + C·x` | 0 ✓ |
| H19 | `dsolve(diff(y,x)=(x^2+y^2)/(x*y), y, x)` | homogénea | `y² = x²(2·ln x + C)` | **residual surd no colapsa** (0 a mano; con base abstracta `u` el expand SÍ cierra — mitigación D5) |

Detector de homogéneas verificado: `subs(RHS, y, v*x)` + check x-free — `subs((x^2+y^2)/(x*y), y, v*x)`→`(v²+1)/v`.

### H. Sistemas 2×2 (ciclo O6)

| # | Probe | Método | Solución esperada | Oráculo |
|---|---|---|---|---|
| Y34 | `dsolve([diff(x,t)=y, diff(y,t)=x], [x,y], t)` | eigen λ=±1 | `x=C1·e^t+C2·e^(−t)`, `y=C1·e^t−C2·e^(−t)` | ambos 0 ✓ |
| Y35 | `dsolve([diff(x,t)=-y, diff(y,t)=x], [x,y], t)` | eigen λ=±i | `x=C1·cos t+C2·sin t`, `y=C1·sin t−C2·cos t` | ambos 0 ✓ |
| Y-def | `dsolve([diff(x,t)=2*x+y, diff(y,t)=2*y], [x,y], t)` | defectiva λ=2 doble | `e^(2t)(C1·(1,0)+C2·(t·(1,0)+(0,1)))` | — |

Eigen exacto vivo: `eigenvalues([[1,2],[2,1]])`→`[3,-1]`, `eigenvectors` ✓; defectiva detectable (base incompleta); `charpoly`+`solve` complejo ✓; nullspace racional ✓. **Gaps**: `try_matrix_eigenvalues` declina disc<0 (`matrix_rule_support.rs:550-552`) y `try_matrix_eigenvectors` (fn en `:644`) declina λ no racional (decline real en `:673-688` — `rational_sqrt(&discriminant)?` :682, `_ => return None` :687) — incluso con `--value-domain complex` → ruta interna D17.

### Z. NEVER-FABRICATE (residual honesto obligatorio — contrato permanente de la fase)

| # | Probe | Por qué no-elemental |
|---|---|---|
| Z1 | `dsolve(diff(y,x)=x^2+y^2, y, x)` | Riccati general → Bessel |
| Z2 | `dsolve(diff(y,x,2)+x*y=0, y, x)` | Airy (coef. variables) |
| Z3 | `dsolve(diff(y,x)=sin(x*y), y, x)` | sin método clásico |
| Z4 | `dsolve(diff(y,x,2)=y^2, y, x)` | no-lineal → Weierstrass ℘ |
| Z5 | `dsolve(diff(y,x)=y^2-x, y, x)` | Riccati sin particular obvia |
| Z6 | `dsolve(x^2*diff(y,x,2)+x*diff(y,x)+(x^2-1)*y=0, y, x)` | Bessel orden 1 |
| Z7 | `dsolve(diff(y,x,2)+sin(y)=0, y, x)` | péndulo → elíptica |

Coherente con la doctrina del repo: `integrate(e^(-x^2),x)` es residual honesto verificado — `dsolve` HEREDA los declines del integrador como residuales honestos, jamás los fuerza.

---

## Arquitectura: chokepoints y máquina reusable (anclas verificadas)

### La cadena de intercepción (una inserción cubre CLI + wire JSON + envelope)

1. **CH-WIRE · chokepoint textual**: `parse_eval_special_command(input: &str)` (`crates/cas_api_models/src/wire_types.rs:522`) prueba en orden Solve (:523, def :625) → SolveSystem list (:529, def :708) → SolveSystem (:534, def :788) → derive/equiv/limit. Sin colisión de prefijo: `parse_solve_command` usa `starts_with("solve(")` sobre trimmed lowercase (:627) y `dsolve(` NO lo satisface — **pinear igualmente**. Helpers de split textual listos: `split_top_level_commas` (:661) y `split_by_comma_at_depth_0` (:939). Pre-pass de usage-error molde `parse_eval_limit_command_error` (:557) — sin él, `dsolve(a=b,...)` malformado cae a statement-parse y muere con "Parse error" críptico por el `=` interno (probado).
2. **CH-BUILD · build tipado**: `build_prepared_eval_request_for_input` (`crates/cas_solver/src/eval_input/build.rs:9-22`) → `build_special_command_request` (`crates/cas_solver/src/eval_input/build/special.rs:6`); la ecuación TEXTO se parsea con `parse_solve_input_for_eval_request` (`crates/cas_solver/src/eval_input_special.rs:30-46`: `Statement::Equation` → nodo `Equal(lhs,rhs)`). Trampa a esquivar: sin intercepción, `y(0)=1` cae al auto-Solve de `statement.rs:5-31` ("Variable 'x' not found", probado).
3. **CH-RUNTIME**: `evaluate_prepared_request_with_session` (`crates/cas_solver/src/eval_request_runtime.rs:16`); `map_non_solve_action` (:7-13) → `engine.eval` (:72). Enums a espejar: `EvalSpecialCommand` (`wire_types.rs:495-514`), `EvalNonSolveAction` (`cas_solver/src/eval_input.rs:8-17`), `PreparedEvalRequest` (:20-45), `EvalAction` (`crates/cas_solver_core/src/eval_models.rs:6-22`).
4. **CH-DISPATCH**: `Engine::eval` (`crates/cas_engine/src/eval/dispatch.rs:78`) → `resolve_and_prepare_dispatch` (`crates/cas_session_core/src/eval.rs:734-776` — **solo resuelve session-refs `#N`, NO simplifica**: `eval_dsolve` recibe el árbol CRUDO intacto; `Equal` es `BuiltinFn` (`cas_ast/src/builtin.rs:113/:273`) y pasa el gate de función desconocida `dispatch.rs:138-155`) → `dispatch_eval_action` (`dispatch.rs:212`; match en `actions.rs:366-386`).
5. **CH-REPL · segunda entrada**: `repl_command_preprocess.rs:22-24` reescribe `solve(...)`→`solve ...`; route en `crates/cas_cli/src/repl/dispatch/route.rs:74-78`. **Cablear dsolve en AMBAS entradas o wire y REPL divergen en silencio** (riesgo mapper 6 #3).

### El verificador (corazón de la fase)

- `equiv_difference_evaluates_to_zero` (`actions.rs:496-509`, privada de `impl Engine`): `Sub(a,b)` → `eval_simplify` (evaluador COMPLETO, `simplify_action.rs:4144`) → acepta SOLO `Expr::Number(0)` literal. **Ritual obligatorio del caller** (`eval_equiv`, `actions.rs:447-485`): save/disable/restore de `self.simplifier.allow_numerical_verification` (:464-468) — un probe f64 jamás confirma.
- Presupuesto: `Budget::preset_cli` (`crates/cas_solver_core/src/budget_model.rs:260-269`) sobrado — 5 probes estilo-dsolve cierran en <0.01s (mapper 6f).

### Máquina reusable por método

| Método | Maquinaria (ancla) | Gap real |
|---|---|---|
| Separables | `integrate_with_trace` (`integration.rs:32-43`; `:45` abre la variante `_with_backend_config`, ambas `pub(crate)` — misma crate que la acción), `factor`, solve inverso con ± | splitter f(x)·g(y) por partición de free-vars (nuevo, pequeño) |
| Lineal 1er | ídem + kernels μ·q exactos | orquestación μ→∫μq→despeje (glue) |
| Exactas | **`try_potential_expr`** (`vector_calculus.rs:547-592`, `pub(crate)`, verificación `poly_eq` gateando :581-590, decline con `required_conditions` :556-559) | extraer M,N y emitir φ=C — casi nada |
| Bernoulli | subs (F3), potencias, reduce a lineal | matcher y'+py=qy^n pelando coef (lección 2026-07-08b) |
| Homogéneas | detector subs+free-vars verificado | ∫1/(F(v)−v) puede declinar honesto; back-subs |
| 2º orden homog | `discriminant`/`sqrt_expr` (`quadratic_formula.rs:73/:6`); molde numérico exacto `matrix_rule_support.rs:539-563` | ramificación 3 casos interna (D9) |
| No-homog UC | diff de trial, `solve_system` exacto (`linear_system_command_entry.rs:7`) | **collector de coeficientes por base {x^j·e^(kx)·sin/cos(bx)}** — walker nuevo (el trabajo real de O5) |
| Sistemas 2×2 | eigen (`try_matrix_eigenvalues` :510, `try_matrix_eigenvectors` :644, charpoly call-site :665) / RREF `rational_rref_in_place` :584-614 / `rational_null_space` :618-636 | autovalores complejos/surds y autovector 2×2 a mano; defectiva vía `solve_system` (D17) |
| Orden superior | `try_extract_diff_call` (`symbolic_calculus_call_support.rs:119-136`), `try_desugar_higher_order_diff` (:150-179, necesita `&mut Context` — disponible en la acción) | el extractor casa ambos shapes crudos: `diff(y,x,2)` sin desugar y `diff(diff(y,x),x)` |

### El contrato de salida

- **No existe nodo ecuación en `Expr`** (`cas_ast/src/expression.rs:95-123`) pero SÍ el wrapper en-arena `__eq__`: `wrap_eq`/`unwrap_eq` (`cas_ast/src/eq.rs:44-59`, `BuiltinFn::Eq`), render `lhs = rhs` plain (`cas_formatter/src/display/expr.rs:586-587`) y LaTeX (`latex_core.rs:996-1005/:2611-2618`).
- `EvalResult` (`eval_models.rs:33-48`) y `SolutionSet` (`cas_ast/src/domain.rs:362-407`) cubren todo sin variante nueva (D6). `ActionResult` = tupla de 8 slots (`eval/mod.rs:16-25`) con `steps` Y `solve_steps` simultáneos.
- Render: `finalize_solution_set_output` (`eval_output_finalize_nonexpr.rs:22-35`); solve_steps → wire + localización en `collect_output_solve_steps` (`eval_output_presentation_solve_steps.rs:11-78`; los steps primarios con `global_after = wrap_eq(...)` se FUSIONAN como substeps del primer solve step, :57-65/:91-113 — así la traza de `integrate_with_trace` entra como substeps de "Integrar ambos lados").
- Rechazado: `Text{plain,latex}` (molde solve_system) — `stored_id: None` (`linear_system_command_eval/runtime.rs:67`), pierde storabilidad `#N`, stats y el walker numeric-display (`present/finalize.rs:24-53`).
- LaTeX gotcha: `diff_is_partial` (`latex_core.rs:49-61`) cuenta `{y,x}` = 2 vars en `diff(y,x)` ⇒ renderizaría `∂` para una EDO ordinaria → D14.

---

## Decisiones cerradas

1. **D1 · Sintaxis canónica**: `dsolve(<ecuación>, <incógnita>, <var>[, <condiciones>...])` — p.ej. `dsolve(diff(y,x)=x*y, y, x)`, `dsolve(diff(y,x,2)+4*y=0, y, x, y(0)=0, y'(0)=2)`. La ecuación viaja como TEXTO crudo (molde `parse_solve_command`); 2º orden **SIEMPRE `diff(y,x,2)`** (convención sympy; `diff(y,x,x)` con `y` desnuda declina por el guard de ambigüedad y se documenta). Condiciones iniciales = args extra TEXTUALES con scanner `<func>['](<punto>)=<valor>` en el wire — el head `y(0)`/`y'(0)` **jamás llega a cas_parser** (esquiva el apóstrofe no soportado y el gate de función desconocida); punto y valor se parsean POR SEPARADO con `cas_parser::parse`. Expresión sin `=` se acepta como `expr = 0` (convención sympy); ecuación sin ningún nodo `diff(<incógnita>,·)` (o incógnita que no aparece derivada) → usage-error "la ecuación no contiene diff(y,·): no es una EDO". Azúcar `y'=...` en la ecuación: DIFERIDA (fuera de fase). Sistemas (O6): forma lista `dsolve([eq1, eq2], [x,y], t)` — molde `parse_solve_system_list_command`.
2. **D2 · Intercepción WIRE + doble entrada + pre-pass**: variante `EvalSpecialCommand::Dsolve { equation, func, var, conditions }` con gancho en `parse_eval_special_command` Y en el preprocess/route del REPL **en el mismo commit O0**; pre-pass de usage-error que captura TODO prefijo `dsolve(` malformado (molde limit :557) — incluido el estilo sympy `y(x)` (detección textual de `<incógnita>(` en la ecuación/2º arg → "escribe y, no y(x): dsolve(diff(y,x)=…, y, x)", fixture pineando el mensaje; hoy `y(x)` muere críptico en el gate de función desconocida); test de no-colisión de prefijo `solve(`/`dsolve(` pineado.
3. **D3 · dsolve es ACCIÓN, no rule, y NO entra al gate de nombres**: `EvalAction::Dsolve` despachada en `dispatch_eval_action`; NO se registra en `is_known_eval_engine_function` ni budget_exempt (precedente verificado: `solve` tampoco está — mapper 6e; la sugerencia contraria de mapper 5 queda SUPERSEDIDA por esa verificación). Formas anidadas (`1+dsolve(...)`) siguen declinando como hoy — es contrato, no bug.
4. **D4 · Invariante anti-colapso (P0 de diseño)**: `eval_dsolve` extrae la estructura de la EDO del árbol CRUDO **antes de cualquier simplify**; solo simplifica subárboles libres-de-y (coeficientes) y el residuo de verificación post-sustitución (donde `diff(φ(x),x)` ya es honesto). Cualquier pasada de simplify sobre subárbol con `diff(y,·)` + `y` desnuda colapsa a 0 en silencio (`symbolic_differentiation_support.rs:6513-6518`). Fixture obligatorio O0: metamórfico never-confirm de AMBAS formas del colapso — `{0}`/`y=0` Y el condicional `All real numbers if -y = 0` (la ecuación-statement top-level entra por el auto-Solve de `statement.rs:5-31` y colapsa a Conditional — probado) — dsolve JAMÁS emite ninguna.
5. **D5 · La verificación GATEA la emisión (doctrina F6) — por LINEALIDAD y con presupuesto**: sustituir la candidata, residuo LHS−RHS → `eval_simplify` → `Number(0)` literal, con `allow_numerical_verification` desactivado (ritual `actions.rs:464-468`); sin cero exacto NO se emite (residual honesto + warning "no verificada"). Para soluciones `Σ Ci·ui(x)`: **pelar las constantes multiplicativas y verificar las funciones base por separado**; para las AFINES de O5 (`y_p + Σ Ci·ui`): verificar `L[ui]≡0` por base contra la homogénea asociada Y `L[y_p]≡RHS` por separado contra la completa — la suma con constantes adheridas jamás se sustituye (pin N28/N29 por esta vía) — único camino que cierra O23 hoy (con `C1·e^(−x)·sin(2x)` el verificador HANG re-confirmado >30s por la oscilación expand↔factor conocida de C5; la base sin constante cierra a 0 exacto en ~1.7s wall — juzgar por contadores, no timing) y rima con "pelar el coef antes de casar". **Candidata IMPLÍCITA `φ(x,y)=C`** (S4 gradúa en O0, E13-E15 en O2): verificar por diferenciación implícita — residuo `∂φ/∂x + ∂φ/∂y·f(x,y)` con `f` = RHS de `y'` despejada de la EDO → `eval_simplify` → `Number(0)` (para exactas equivale al gate `poly_eq` de `try_potential_expr` :581-590). H19 (base surd compuesta): atomizar la base (`2·ln x+C → u`) antes de expand, o verificar elevando al cuadrado. **Budget NO tiene timeout de reloj — solo contadores** (`budget_model.rs:255-270`; el HANG O23 ocurrió BAJO esos contadores): el gate corre bajo un preset de Budget DEDICADO y más estricto (límites RewriteSteps/PolyOps bajos, molde `preset_cli` :260) + **fixture O0 de TERMINACIÓN del gate** con el caso hostil conocido (candidata O23 CON C1/C2 adherida) asertando decline acotado; si ni el preset estricto termina, la linealidad se eleva de mitigación a REGLA DURA: el gate jamás verifica términos con constante simbólica adherida.
6. **D6 · Contrato de salida SIN variante nueva de `EvalResult`** — con CRITERIO de forma: tras integrar, intentar el despeje con el solve inverso; si cierra en ≤2 ramas limpias → explícita (1 rama = `wrap_eq(y,f)`; 2 = `SolutionSet` Discrete); si solve declina o el despeje exige surds anidados → implícita `wrap_eq(φ(x,y),C)` + DomainWarning "solución implícita". Formas: solución explícita → `Expr(wrap_eq(y, f(x,C)))` (render `y = C·e^x`, storable `#N`, stats, walker numeric-display); multi-rama → `SolutionSet(Discrete([wrap_eq(y,f1), wrap_eq(y,f2)]))`; exacta implícita → `Expr(wrap_eq(φ(x,y), C))`; paramétrico → `SolutionSet::Conditional`; no-resuelto → `SolutionSet::Residual(eco dsolve(...))`. **Jamás `Text`** (pierde storabilidad — molde solve_system descartado con causa).
7. **D7 · Constantes `C`/`C1`/`C2` = símbolos reservados del RESULTADO**: opacas verificadas (diff/simplify/approx/equiv/solve — cero colapsos, mapper 3b); 1er orden emite `C`, 2º orden `C1`,`C2`. Si la ENTRADA del usuario ya contiene esos nombres → constantes frescas (`K1`,...) + warning. Todo resultado general lleva warning estándar por el canal `DomainWarning`: "solución general; C constante arbitraria".
8. **D8 · Residual honesto = eco `dsolve(...)` + warning del método más cercano**: `SolutionSet::Residual` con la call simbólica re-emitible + `DomainWarning` nombrando por qué declinó ("no separable ni lineal; el candidato exacto no verificó", "la integral ∫... no cierra en forma elemental", "solución singular y=0 descartada al dividir por g(y)"). Los Z1-Z7 son fixtures never-fabricate permanentes.
9. **D9 · Característica de 2º orden por discriminante exacto INTERNO**: jamás `solve` desnudo (el set colapsa multiplicidad; complejas exigen value-domain de sesión). `dsolve` computa disc `BigRational` (molde `quadratic_formula.rs:73` + `matrix_rule_support.rs:539-563`) y ramifica: >0 dos exponenciales; =0 `(C1+C2·x)e^(rx)`; <0 α=−b/2a, β=√(−disc)/2a → `e^(αx)(C1·cos βx + C2·sin βx)`. Independiente del `--value-domain` de la sesión (cálculo interno, no gate de superficie — guardrail #1 en su modo no-ceremonia).
10. **D10 · Migración never-confirm en O0 (semántica special-command)**: sacar `"dsolve(y, x)"` del array (`cli_contract_tests.rs:3573`), reescribir el comentario stale :3570-3572 (dsolve ya NO está "fuera del norte"); el pin migrado es BIDIRECCIONAL: (a) la forma-comando bien formada computa; (b) `dsolve(y, x)` malformado → usage-error asertado EXPLÍCITO (el pre-pass lo captura — ya no "no definida"); (c) forma embebida en expresión (`dsolve(y)+1`) sigue declinando por el eval genérico. `erf`/`gamma`/`residue` permanecen presas intactas.
11. **D11 · Exactas delegan en `try_potential_expr` con gate dos-niveles**: comps=[M,N], vars=[x,y] → φ; fast-path `poly_eq` (vector_calculus.rs:587) + fallback estilo `equiv_difference_evaluates_to_zero` por componente (el evaluador completo SÍ verifica trig: `diff(sin(x)+cos(y),x)−cos(x)`→`0` probado) con flag numérico off. Un campo no conservativo NUNCA verifica → la detección de exactitud es GRATIS. Graduar el mismo upgrade a `PotentialRule` (F6) queda como chip opcional fuera de fase.
12. **D12 · Wrinkles didácticos de presentación (capa display, no valor)**: μ = e^(∫dx/x) = `|x|` → se toma μ = `x` (convención textbook; cualquier μ funcional es legítimo — análogo doctrinal a la capa numeric-display: preferencia de PRESENTACIÓN); el ± de `e^(ln|y|)` se ABSORBE en C (solve ya emite ambas ramas) con warning "C ≠ 0; y = 0 solución singular" cuando se dividió por g(y). Vetables por el usuario en la revisión de este doc.
13. **D13 · Narración por `solve_steps` (estados-ecuación), keyed es/en, PER-CICLO**: el canal natural es `SolveStep{description, equation_after}` (`solve_types.rs:43-48`); cada descripción nueva entra en `SOLVE_DESCRIPTIONS` (`eval_output_presentation_solve_steps/localization.rs:119+` — sin entrada pasa INTACTA en inglés crudo: prohibido); la traza del integrador se fusiona como substeps (:57-65). Guiones por método (mapper 4c): separables (identificar → separar → integrar ambos lados → despejar → verificar), lineal (identificar p,q → μ → (μy)′=μq → integrar → despejar), exactas (exactitud ∂M/∂y=∂N/∂x → reconstruir φ → φ=C), 2º orden (característica → raíces → base 3-casos → general). Las reglas no nacen mudas (lección Fase 3). **Warnings**: siguen la convención actual del repo (String única por `DomainWarning`, sin keying es/en — la disciplina `SOLVE_DESCRIPTIONS` aplica SOLO a solve_steps); el idioma se fija en O0 y NO se abre maquinaria i18n nueva en esta fase.
14. **D14 · LaTeX: `y` es variable DEPENDIENTE**: render `d/dx`, no `∂` (excepción en `diff_is_partial` scoped al canal de dsolve, o render propio del lhs vía `render_equation_strings` del canal Equation). Notación prima de SALIDA: no existe y no se introduce en esta fase.
15. **D15 · Lane matrix-smoke propia NACE en O0**: cero suites del scorecard ejercitan el pipeline solve/special-command (verificado mapper 6b — la única red serán los cli_contract_tests + la lane nueva). `engine_dsolve_command_matrix_smoke.py` molde limit (`scripts/engine_limit_command_matrix_smoke.py`, dataclass de ejes :41-56, contadores :3299-3348): ejes `family` (separable/lineal_1o/exacta/coef_const_2o/UC/bernoulli/homogenea/sistema), `order_regime`, `verification_regime` (verified_by_substitution/declined), `constant_regime` (general/IVP), `outcome` (supported/residual), `residual_cause`, `trace_regime`, `presentation_regime`; `SuiteSpec` nueva en `engine_improvement_scorecard.py` (molde :533). Cada ciclo que registre una familia la añade a la matrix EN EL MISMO ciclo.
16. **D16 · Constantes en `solve_system`**: relajar `is_valid_linear_system_var` (`vars.rs:18-20`) a alfanumérico-tras-alfabético (fix de una línea + tests propios) — aterriza en O3 como dependencia de las ICs acopladas de 2º orden; hasta entonces el solve univariado (que SÍ acepta `c1`/`C1`) cubre 1er orden.
17. **D17 · Sistemas 2×2 por ruta interna exacta**: charpoly + disc exacto (molde D9) + autovector por eliminación de Gauss 2×2 a mano (racional/complejo-racional — nuevo, pequeño); defectiva → vector generalizado `(A−λI)w=v` vía `solve_system` (verde); **los verbos `eigenvalues`/`eigenvectors` NO se tocan** (sus declines disc<0/λ-no-racional quedan pineados como contrato del verbo).

---

## Secuencia de sub-ciclos O0..O9 (cada uno = un `/auto-mejora`, un commit)

Orden por dependencia + ROI. El costo fijo grande va primero (O0); después cada método es un ensamblador S/M sobre máquina viva.

#### O0 — Sustrato dsolve: wire + acción + contrato + verificador + SEPARABLES **[L — el ciclo dos-cables grande]** ☑ *(graduado 2026-07-19, hash en el ledger: S1-S9+L12 con verificación-gate exacta y narración es/en; S3/S5 implícitas honestas — el despeje `arctan(y)=RHS` simbólico es gap del solve nombrado como peldaño; Z1-Z7 + anti-colapso + never-confirm D10 pineados; azúcar aridad-2; lane `calculus_dsolve_command_matrix_smoke` en 4 perfiles; paridad REPL vía gancho en `build_eval_request` — el fallback-eval del REPL NO pasaba por el wire, riesgo mapper 6 #3 confirmado y cerrado)*
- **Gradúa:** S1-S7 + L12 computan con verificación-gate y narración keyed; `dsolve(...)` parsea en CLI, wire JSON, envelope y REPL (ambas entradas, D2); todo método no implementado → `SolutionSet::Residual` eco + warning (D8) — los Z1-Z7 nacen pineados never-fabricate; formas con condiciones iniciales PARSEAN y declinan honesto ("condiciones: ciclo O3"); usage-error pre-pass para `dsolve(` malformado; migración never-confirm D10 (tridireccional); lane `engine_dsolve_command_matrix_smoke` registrada en el scorecard con las familias separable/residual (D15); warning solución singular/± (D12); fixture metamórfico anti-colapso (D4: never-confirm de `{0}`).
- **Inserción (orden compiler-driven, mapper 1b):** `wire_types.rs` variante enum + `parse_dsolve_command` + gancho :522 + pre-pass; `eval_models.rs:6` `EvalAction::Dsolve`; `eval_input.rs:8` espejo; `special.rs:12` brazo build; `eval_request_runtime.rs:7-13` map; `actions.rs:373-385` brazo dispatch → `eval_dsolve` (molde `eval_solve` :389 para el shape del ActionResult, molde `eval_limit` :512 para warnings); matches exhaustivos forzados: `eval_output_presentation_input.rs:126-165` (LaTeX de entrada, molde Solve :162) + revisar a mano los patrones no-exhaustivos (`eval_output_presentation_conditions.rs:727`, `present/collect.rs:55`); REPL preprocess+route; splitter separable f(x)·g(y) nuevo (factor + partición free-vars; SPEC: todo factor libre-de-y va al lado f(x) — incluidos parámetros libres de ambas, p.ej. `k`; `g(y)=1` es partición válida = integración directa).
- **Reuso:** `integrate_with_trace`, solve inverso (maneja el ± del abs), `wrap_eq`, `equiv_difference_evaluates_to_zero` (misma `impl Engine` — sin cambio de visibilidad; hoist `pub(crate)` solo si el handler vive en otro módulo), canal `DomainWarning`, `SOLVE_DESCRIPTIONS`.
- **Blast:** **MEDIO** — enums cross-crate compiler-driven (precedente B4a: 18 brazos, 8 crates, sin wildcards); el fixture never-confirm migra; cero drift en real fuera de dsolve (la intercepción es prefijo-exacta).
- **Depende:** nada. Prohibición O0: `#N` en condiciones/ecuación NO se resuelve dentro de la acción (R-B) — declinar con motivo.
- **Retención (pins):** `solve(x^2=4,x)`→`{−2,2}` y todo el pipeline solve byte-idéntico; `diff(y,x)`→`0` en eval plano SIGUE (es el contrato de diff, no de dsolve); no-colisión `solve(`/`dsolve(`; `1+dsolve(...)` declina; `erf`/`gamma`/`residue` presas; metamórfico anti-colapso.

#### O1 — Lineal 1er orden (factor integrante) **[S]** ☑ *(graduado 2026-07-19, hash en el ledger: L8-L11 + siblings del barrido — forma reordenada `y'=x−y`, coef no-unitario, `a(x)=x`, paramétrico `k`; pin L9 μ=x cumplido vía strip-abs; candidata en forma SPLIT `∫μq/μ + C/μ` con cancelación estructural de μ — la forma split es a la vez textbook y la única que el verificador reduce; familia `lineal_1o` en la matrix)*
- **Gradúa:** L8-L11; μ=e^∫p con strip-abs D12 (pin explícito L9: μ=x, no |x|); resonancia de 1er orden L10 sale sola (∫e^0·e^x); narración D13 lineal; familia `lineal_1o` en la matrix.
- **Inserción:** orquestador μ→∫μq→despeje dentro de `eval_dsolve` (glue); clave locale "Identificar forma lineal: y' + p·y = q con p = {0}, q = {1}" (molde template cuadrático existente).
- **Reuso:** kernels ∫μq verificados (`x·e^x`, `sin(2x)e^(3x)`, `e^(2x)sin x` — todos exactos); despeje trivial (y = (∫μq + C)/μ).
- **Blast:** **BAJO** — aditivo dentro del handler.
- **Depende:** O0.
- **Retención (pins):** S1-S7 intactos (el dispatcher de métodos prueba lineal DESPUÉS de separable o con matcher excluyente — pin de no-regresión de forma); L9 μ-display.

#### O2 — Exactas (delegación en `try_potential_expr`) **[S]** ☑ *(graduado 2026-07-20, hash en el ledger: E14/E15 nivel-1 poly_eq + el TRASCENDENTE `x·e^y+y²=C` graduado por el nivel-2 D11 — la misma reconstrucción con el evaluador completo en el caller, no solo el pin de decline; gate POR COMPONENTE ∂φ/∂x−M→0 ∧ ∂φ/∂y−N→0; E13/E-neg caen antes al lineal con explícitas equivalentes; pins potential() intactos — el upgrade del verbo sigue siendo chip fuera de fase; familia `exacta` en la matrix)*
- **Gradúa:** E13-E15 → `φ(x,y) = C` implícita (D6.3); no-exacta → decline del camino con motivo (el gate poly_eq no verifica — E-neg pineado); gate dos-niveles D11 (fallback full-eval para exactas trig — graduar al menos un caso trascendente o pinear su decline honesto); narración D13 exactas; familia `exacta` en la matrix.
- **Inserción:** extractor M,N de la forma `M + N·diff(y,x) = 0` (árbol crudo, D4) + llamada directa `try_potential_expr(ctx, &[M,N], &["x","y"])` (misma crate); brazo fallback de verificación por componente.
- **Reuso:** TODO el corazón F6 (`vector_calculus.rs:547-592`) as-is; `required_conditions` decline (:556-559) heredado como residual honesto (M=1/x etc.).
- **Blast:** **BAJO** — `try_potential_expr` no se modifica en O2 (el fallback vive en el caller dsolve).
- **Depende:** O0. Independiente de O1 (intercalable).
- **Retención (pins):** `potential([2*x*y,x^2],[x,y])`→`y·x²` intacto (metamórfico F6); `potential([cos(x),-sin(y)],[x,y])` sigue residual (el upgrade de PotentialRule NO entra aquí — chip aparte); E-neg decline.

#### O3 — Condiciones iniciales (IVP) **[S/M]**
- **Gradúa:** V30/V32/V33 (1er orden); scanner textual de condiciones `y(x0)=y0` / `y'(x0)=v0` en el wire (D1 — heads jamás llegan a cas_parser); resolución de C: `subs` del punto + `solve` univariado (verde incl. paramétrico: `solve(C*e^(-a)=b,C)`→`{b·e^a}`); fix D16 de `vars.rs:18-20` + tests (deja listo el 2×2 de constantes para O4); condición inconsistente o solve declina → residual honesto con warning; `constant_regime=IVP` en la matrix; verificación FINAL doble: la solución particular verifica la EDO Y las condiciones (subs exacto).
- **Inserción:** scanner en `parse_dsolve_command`; resolutor de constantes en `eval_dsolve` (subs a nivel de árbol + solve); `vars.rs:18-20` una línea.
- **Reuso:** `substitute_expr_by_id`, solve univariado, `solve_system` (post-fix).
- **Blast:** **BAJO-MEDIO** — el fix de vars.rs toca superficie de solve_system: pins de sus mensajes de error actuales (nombres inválidos que DEBEN seguir siéndolo: `1a`, `x-y`).
- **Depende:** O0 (y O1 recomendado para V33).
- **Retención (pins):** `solve_system(a+b=3; 2*a-2*b=4; a; b)` intacto; formas sin condiciones byte-idénticas; V31 queda EXPLÍCITAMENTE pendiente (gradúa en O4 con esta maquinaria).

#### O4 — 2º orden homogénea coef. constantes **[S/M]**
- **Gradúa:** O20-O25 por las 3 ramas del disc exacto (D9); V31 (IVP 2º orden con el 2×2 de constantes post-D16); verificación por linealidad D5 (pelar C1/C2, verificar bases — cierra O23 sin tocar el HANG expand↔factor, que NO se apresura: memoria C5); extractor de orden 2 casa `diff(y,x,2)` crudo y `diff(diff(y,x),x)` (reusa `try_desugar_higher_order_diff` para normalizar); orden ≥3 → residual honesto nombrado; narración D13 característica (con los solve_steps del cuadrático como substeps); familia `coef_const_2o`.
- **Inserción:** rama 2º orden en el extractor de EDO; módulo característica (disc BigRational interno); ensamblador de base 3-casos.
- **Reuso:** `discriminant`/`sqrt_expr` (`quadratic_formula.rs`), molde `matrix_rule_support.rs:539-563`, wronskian disponible para narración de independencia (opcional).
- **Blast:** **BAJO-MEDIO** — neto-nuevo dentro del handler; la verificación por linealidad es el punto delicado (pin O23 obligatorio).
- **Depende:** O0 + O3 (V31).
- **Retención (pins):** O23 emite Y verifica (el pin del ciclo — sin él no gradúa); Z2/Z6 (coef. variables) siguen residual; `solve(r^2+2*r+1=0,r)`→`{−1}` intacto (el verbo solve no cambia).

#### O5 — 2º orden no-homogénea (coeficientes indeterminados) **[M — el trabajo neto-nuevo mayor]**
- **Gradúa:** N26-N29 con RHS ∈ {polinomio, e^(kx), sin/cos(kx)} y productos simples; shift de resonancia x^s (s = multiplicidad de k/±ib como raíz característica — sale del disc de O4); trial → diff → **collector de coeficientes por función-base** (walker nuevo: MultiPoly no cubre bases trascendentes) → `solve_system` exacto (coeficientes internos con la disciplina de FRESCURA D7: nombres ausentes de las free-vars de la EDO; si el RHS trae parámetros simbólicos libres, decidir en el ciclo soporte-paramétrico vs decline nombrado — pin); RHS fuera de la tabla UC → residual honesto ("variación de parámetros: fuera de fase o ciclo futuro"); narración D13; familia `UC`.
- **Inserción:** tabla de formas trial + walker collect-por-base {x^j·e^(kx)·sin/cos(bx)} (el módulo nuevo del ciclo); ensamblador y_general = y_h + y_p.
- **Reuso:** O4 entero (y_h), diff simbólico, `solve_system` (`linear_system_command_entry.rs:7`).
- **Blast:** **MEDIO** — walker nuevo con matcher de formas (riesgo conocido: coeficiente ≠1 rompe matchers de forma-desnuda — pelar el coef primero); N27 documenta el WARN depth transitorio (falso-rojo de stderr — eje stderr-fragility en la matrix).
- **Depende:** O4.
- **Retención (pins):** N28/N29 resonancia (los pins de corrección del shift); O20-O25 byte-idénticos; residual honesto para RHS tan(x) (fuera de tabla UC).

#### O6 — Sistemas 2×2 `X' = AX` **[M]**
- **Gradúa:** Y34 (eigen real — la ruta feliz existente), Y35 (λ=±i por ruta interna D17: charpoly + disc + autovector 2×2 a mano → solución REAL vía Re/Im), Y-def (defectiva: vector generalizado por `solve_system`); forma lista `dsolve([...],[x,y],t)` parsea (molde `parse_solve_system_list_command` :708); A no constante o n>2 → residual honesto; condiciones iniciales en la forma lista (`x(0)=1, y(0)=0`) parsean con el mismo scanner y en O6 DECLINAN honesto con motivo ("IVP de sistemas: ciclo futuro") — o se soportan si el 2×2 en C1,C2 post-D16 sale barato: decidir al abrir O6 y pinear; verificación componente a componente (ambas ecuaciones a 0); narración por autovalores; familia `sistema`.
- **Inserción:** brazo lista en `parse_dsolve_command`; extractor de A desde las dos ecuaciones (coeficientes libres-de-{x,y}); módulo eigen-interno 2×2 (D17).
- **Reuso:** `charpoly`, molde disc D9, `solve_system`, `nullspace` racional; eigen verbos SOLO para el caso racional feliz.
- **Blast:** **MEDIO** — parse de forma lista + álgebra compleja-racional nueva (acotada a 2×2).
- **Depende:** O0 (+O4 por el molde disc; independiente de O5 — **intercambiables O5↔O6**).
- **Retención (pins):** `eigenvalues([[0,-1],[1,0]])` SIGUE residual (el verbo no cambia — pin de contrato D17); Y35 emite sin `i` en el resultado final (solución real).

#### O7 — Superficie de usuario + pulido de narración **[S/M]**
- **Gradúa:** brazo `"dsolve"` en `help_topics.rs` (moldes solve :149 / limit :395, con sección de residuales honestos); `completer.rs` (:16-79); grupo nuevo en `web/examples.csv` ("Ecuaciones diferenciales") + filtro `EXPECTED_DSOLVE_RESULTS` en `scripts/test_web_examples_smoke.py` (sin él las filas NO se auto-verifican — riesgo mapper 6 #5); render LaTeX `d/dx` para la incógnita (D14); auditoría de narración completa es/en (ninguna descripción pasa cruda por `localize_solve_description` — grep-gate); OPCIONAL: hint didáctico "¿quisiste decir dsolve?" cubriendo las DOS entradas del colapso — `solve(diff(y,·)=…)` Y la ecuación-statement top-level (auto-Solve de `statement.rs:5-31`) — por la vía didáctica, no tocando solve.
- **Inserción / Reuso / Blast:** superficies conocidas, BAJO.
- **Depende:** O0-O6 (audita lo existente).
- **Retención (pins):** smoke de examples con resultados esperados; help/completer no rompen tests de CLI.

#### O8 — Bernoulli + homogéneas-sustitución **[S-M / M] — cierre curricular, NO opcional (probes graduate)**
- **Gradúa:** B16-B17 (matcher y'+py=qy^n pelando coef; v=y^(1−n) → lineal O1 → back-subs); H18-H19 (detector subs+free-vars verificado; v=y/x → separable O0 en v → back-subs); H19 con la mitigación D5 (base atomizada o verificación al cuadrado) — si aún no verifica, residual honesto CON DUEÑO (jamás emitir sin gate); ∫1/(F(v)−v) que declina → residual honesto; familias `bernoulli`/`homogenea`.
- **Inserción:** dos matchers + dos transformaciones dentro del dispatcher de métodos.
- **Reuso:** O0 (separables) + O1 (lineal) COMPLETOS — cero primitivas nuevas.
- **Blast:** **BAJO** — composición de métodos ya graduados.
- **Depende:** O0+O1. **Puede adelantarse** (independiente de O4-O6) si se quiere cerrar 1er orden antes de abrir 2º.
- **Retención (pins):** el dispatcher no roba formas (S/L/E byte-idénticos); B17 re-verificado a mano en el ciclo (nota del catálogo); H19 gate.

#### O9 — (OPCIONAL) Series de potencias vía Taylor **[varianza alta — se scopea al llegar]**
- Solución en serie `y = Σ aₙxⁿ` para coef. variables curriculares (Airy Z2 como serie truncada NOMBRADA como aproximación, jamás como solución cerrada). Fuera del catálogo verificado; reuso F1/F2 de Fase 3. Solo entra por decisión explícita del usuario al cerrar O0-O8.
- Extensión opcional hermana (mismo estatus): **factor integrante μ(x) simple** para no-exactas (`μ = e^∫((M_y−N_x)/N)` — mapper 2 la aisló como ciclo M aparte).
- Extensión opcional hermana 2: **Cauchy-Euler** `x²y''+axy'+by=0` — S puro sobre el molde D9 (característica `r(r−1)+ar+b=0`, mismas 3 ramas); HOY `solve(x^2*diff(y,x,2)+x*diff(y,x)-y=0,y)` → `{0}` (colapso probado) y hasta su ciclo queda cubierta por el decline Z-coef-variables — el residual la nombra para no ambiguar el contrato.

**Orden global propuesto:** `O0 → O1 → O2 → O3 → O4 → O5 → O6 → O7 → O8 (→ O9)`, con O2 intercalable tras O0, O8 adelantable tras O1, y O5↔O6 intercambiables. Dependencias netas: O1←O0; O2←O0; O3←O0(+O1); O4←O0+O3; O5←O4; O6←O0(+O4 molde); O7←todos; O8←O0+O1.

---

## Riesgos (trampas a evitar)

- **El colapso es SILENCIOSO y ok:true** (`solve(diff(y,x)=y,y)`→`{0}` sin error): cualquier subárbol con `diff(y,·)` que toque el simplificador fabrica. El invariante D4 + el metamórfico anti-colapso son el contrato de soundness central de la fase — se cablean en O0 y ningún ciclo posterior los relaja.
- **El HANG del verificador es real y conocido** (O23: constante simbólica × exp × trig → oscilación expand↔factor, familia C5, depth_overflow 51): la verificación por linealidad D5 lo ESQUIVA; el fix de orquestación del detector NO se apresura (memoria C5: fix de orquestación, no de regla). Budget de contadores ESTRICTO en el gate (no existe timeout de reloj — gap verificado) + fixture de terminación O0: un gate que no decide degrada a residual, jamás bloquea el eval.
- **Falso-rojo de stderr** (N27: WARN depth_overflow transitorio con resultado correcto): juzgar por resultado + contadores, eje stderr-fragility en la matrix (molde limit).
- **Doble entrada wire/REPL**: cablear solo una divierte las superficies en silencio (mapper 6 #3) — ambas en O0, mismo commit, con probes de paridad.
- **dsolve nace SIN red heredada de scorecard** (cero lanes cubren el pipeline special-command): la matrix propia D15 no es opcional ni diferible — llega con cada familia en su ciclo.
- **Session-refs en la acción**: `resolve_prepare_config` solo resuelve `parsed`/`equiv_other` (`dispatch.rs:49-60`) — ExprIds extra de condiciones NO reciben resolución `#N`. O0 los prohíbe con motivo; si un ciclo futuro los quiere, resuelve DENTRO de `eval_dsolve`.
- **Los verbos eigen/solve NO se modifican por conveniencia de dsolve** (D17/D9): sus declines y colapsos de multiplicidad son contrato propio; dsolve computa internamente. Tocarlos sería blast transversal no presupuestado.
- **Working tree en vuelo**: `wire_types.rs` y `cas_session_core/src/eval.rs` tienen cambios F7/F9 sin commitear — re-anclar TODAS las líneas de este doc al abrir O0 (y committear antes de lanzar workflows adversariales — memoria).
- **Matchers de forma**: coeficiente ≠1 rompe matchers desnudos (lección 2026-07-08b) — pelar coef antes de casar en Bernoulli/UC/lineal.
- **La "solución esperada" del catálogo es hipótesis, no oráculo** (lección FTC 2026-07-08c): el gate por sustitución es el juez; si el engine emite una forma equivalente distinta, verificar equivalencia, no bytes.

---

## Guardrails inter-fase aplicados

1. **`ValueDomain`-threading**: dsolve es SINTÁCTICO en superficie (cero gate-ceremonia); el dominio complejo se usa solo como CÁLCULO INTERNO de la característica (D9/D17) sin depender del value-domain de sesión — ni copiar el gate por inercia ni depender de la sesión para corrección.
2. **diff/integrate per-variable**: todo diff/integrate del handler es por-variable explícita; jamás estado global de variable; el extractor respeta la frontera del guard de ambigüedad.
3. **Soundness EXACTO**: la emisión exige `Number(0)` literal del evaluador completo con verificación numérica DESACTIVADA; disc/coeficientes sobre `BigRational`; cero f64 en decisiones drop/keep (regla del repo).
4. **Backstop never-confirm/never-fabricate per-ciclo**: metamórfico anti-colapso (O0), Z1-Z7 permanentes, never-confirm D10 bidireccional, pins de no-robo de formas del dispatcher de métodos, E-neg/H19 gates.
5. **Resultado-como-contrato**: la salida re-parsea y compone — `wrap_eq` storable `#N`; fixture de round-trip con RECETA verificada en O0: (a) probar que subs/diff/simplify TRAVERSAN `BuiltinFn::Eq` sin declinar ni colapsar (dsolve sería el primer comando que ALMACENA `__eq__` como `#N` — pin); si no traversan, (b) el fixture usa el RHS textual del render y se documenta `#N`-de-ecuación como storable-pero-opaco — elección de contrato del ciclo, no descubrimiento en fixture rojo; el residual conserva la call `dsolve(...)` re-emitible.

---

## Preguntas abiertas — RESUELTAS 2026-07-19 (el usuario delegó la respuesta con su criterio: «lo más universal y más educativo posible al mismo tiempo»)

1. **Azúcar de aridad corta** → **SÍ, en O0**: `dsolve(eq, y)` infiere la variable del primer `diff(y, <var>)` del texto; usage-error si la inferencia es ambigua (dos vars distintas derivando y). Universal (acepta la entrada natural) y educativo (menos fricción); la canónica 3-args sigue siendo la documentada.
2. **Prioridad O5 vs O6** → **O5 primero** (como está el orden): coeficientes indeterminados es el corazón curricular del 2º curso — el criterio educativo manda; los sistemas lucen la maquinaria pero enseñan menos por ciclo. No se recorta nada: ambos entran.
3. **Convención de constantes** → **`C`, `C1`, `C2`** (mayúsculas): la convención dominante de los textos — el resultado se lee como en el libro (criterio educativo); el fix D16 igual se necesita.
4. **Los wrinkles D12** → **SIN veto, cerrados como están**: μ=x es la convención textbook (capa de PRESENTACIÓN — cualquier μ funcional es legítimo) y el warning de solución singular al dividir por g(y) ENSEÑA exactamente el punto que un curso serio subraya — presentación de libro + honestidad explícita es la síntesis universal+educativo.
5. **Los opcionales O9** → **PRE-APROBADOS con matiz**: μ(x) simple y Cauchy-Euler ENTRAN a la fase como ciclos post-O8 (son S/M sobre maquinaria viva y ensanchan cobertura real — criterio universal); las SERIES conservan su «se scopea al llegar» (varianza alta — un mini-scoping propio al abrirlas, no un cheque en blanco). El orden del núcleo O0-O8 no se altera.

---

## Cómo ejecutar

Cada sub-ciclo es un `/auto-mejora 1` (o encadenar). Marcar aquí `☑` con hash al graduar (disciplina hash-stamps: jamás estampar el propio commit vía amend — estampar en el ciclo siguiente o citar "hash en el ledger"; auditar con `--is-ancestor`). Verificación por ciclo: probes de `graduates` + pins de retención nombrados + `cargo test --workspace` completo AL INICIO y al cierre (clean git status ≠ green tests; gatear pipes con `&&`, jamás `| tail` sin gate). Los workflows adversariales solo sobre árbol commiteado.

**Criterio de "Fase 4 cerrada" (checklist mecánico):**
1. Los ~35 probes graduate del catálogo (S/L/E/V/O/N/B/H/Y) verdes, cada uno emitido SOLO tras verificación-gate a `Number(0)` exacto.
2. Z1-Z7 residual honesto pineado never-fabricate (su decline es parte del contrato — resolverlos sería bug, no avance).
3. Metamórfico anti-colapso verde: dsolve jamás emite `{0}`/`y=0`-del-colapso NI el condicional All-reals-if del statement top-level; `diff(y,x)`→`0` en eval plano intacto (contrato de diff).
4. Never-confirm D10 migrado bidireccional; `erf`/`gamma`/`residue` siguen presas.
5. Lane `engine_dsolve_command_matrix_smoke` registrada en el scorecard con TODAS las familias graduadas y contadores por eje; drift real en las lanes existentes (16 = perfil guardrail; 25 SuiteSpec totales: fast=13, fast_embedded=14, full=20) = SOLO el declarado (esperado: cero — dsolve es aditivo; NINGUNA suite existente ejercita el pipeline special-command).
6. Narración `solve_steps` keyed es/en presente en cada método (grep-gate: ninguna descripción sin entrada en `SOLVE_DESCRIPTIONS`); LaTeX `d/dx` (no `∂`) para la incógnita.
7. Grupo examples.csv + filtro smoke propio verdes; help/completer presentes.
8. Fix D16 (`vars.rs`) con pins de nombres que siguen inválidos.
