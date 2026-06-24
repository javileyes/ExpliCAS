# Auditoría de frontera del engine de cálculo

Fuente de candidatos del bucle `/auto-mejora`. Cada ciclo retenido que
gradúe un item lo marca aquí (`[x]` + `*(graduado YYYY-MM-DD commit:
qué quedó cubierto y qué queda como peldaño)*`) en vez de borrarlo;
las re-auditorías añaden una sección nueva con fecha y dejan las
anteriores como historia de progreso.

## Auditoría 2026-06-12

Metodología: 6 agentes paralelos — 5 sondando el CLI release
(`target/release/cas_cli eval`) con ~150 expresiones de dificultad
creciente clasificadas en funciona / falla / residual honesto, y 1
leyendo el modelo de madurez, roadmap y velocidad del ledger.

### Cobertura medida

| Dimensión | vs. curso universitario | vs. CAS profesional | Nota |
|---|---|---|---|
| Diferenciación | ~80% | ~45-50% | mecánica sólida (cadena profunda, x^x, parciales, condiciones); bloqueada por estabilidad del simplificador |
| Integral indefinida | ~60-65% (58/92 sondas) | ~35-40% | racionales/por-partes/potencias trig casi completos; 6/6 no-elementales correctamente residuales |
| Límites | ~45-50% | ~20-25% | allowlist de patrones, no algoritmo; `e` inalcanzable por límite |
| Definidas/impropias | ~70-75% | ~35-40% | FTC/touches/impropias elementales sólidos; pre-simplificador sabotea casos |
| Calidad educativa | ~58% bien narrado | ~35-40% | condiciones de dominio sistemáticas (punto fuerte); por partes narrada en las TRES asignaciones u/dv: polinomio·ln, polinomio(lineal)·{eˣ,sin,cos,sinh}, y u=función/dv=dx (inverse-trig/hiperbólicas + ln(x) solo) (quedan repetida grado≥2 y cíclico eˣ·sin) |

### Horizontes del propio repo (CALCULUS_ENGINE_STRATEGY.md, citas)

1. "serious educational ... dozens more retained ROI cycles" — a la
   velocidad observada (~19 ciclos retenidos/día de sesión en
   2026-06-10/11/12), días de sesión.
2. "mature elementary ... on the order of one hundred or more retained
   cycles" — 1-3 semanas de sesiones intensas, CON la advertencia de
   que la velocidad actual está inflada por ciclos baratos (espejos de
   signo, brazos Div); lo restante incluye ciclos arquitectónicos.
3. "a universal integration engine is not a bounded target for
   ordinary ROI cycles" — track del backend híbrido (Fases 0-3 done,
   4 in progress) + componentes de grado investigación (Risch, Gruntz,
   assumptions, funciones especiales).

## Cola priorizada (consumir desde /auto-mejora paso 1)

Clase F = ciclo-familia (patrón conocido, 1 ciclo). Clase A = ciclo
arquitectónico (scoping workflow primero, riesgo de huella alto).
Clase I = grado investigación / Deferred Horizons (no es un ciclo).

### P0 — soundness y confianza (antes que capacidad)

- [x] **(S) `diff` soltaba la condición de dominio de un factor recíproco-trig que se cancela**:
  `diff(tan(x)·cos(x), x)` devolvía `cos(x)` SIN condición (válida solo donde `cos(x)≠0`, ya que
  `tan(x)·cos(x)` es indefinida en `cos(x)=0`); igual `diff(sec x·cos x)→0`, `diff(cot x·sin x)→−sin(x)`,
  `diff(sin x·cot x)`, `diff(csc x·sin x)`. El motor era CONSISTENTE en casos paralelos
  (`diff(tan x·cot x)→0` con `[cos≠0,sin≠0]`, `diff((x²−1)/(x−1))` con `[x≠1]`, `diff(x/x)` con `[x≠0]`):
  solo la familia trig-que-cancela lo soltaba. (P0 soundness en `diff`; exento del orden de fase.)
  *(graduado 2026-06-23 eeced5d5c: hallado por el HUNT ADVERSARIAL MULTIAGENTE (ultracode) y verificado
  por 2 lentes vs sympy/mpmath. Causa raíz: las `required_conditions` se derivan de la estructura del
  RESULTADO, así que un factor restrictor que se cancela (`tan·cos → sin`) pierde su condición;
  `infer_implicit_domain` tampoco modela los dominios de tan/cot/sec/csc (es de solo-lectura). Fix en la
  rama `diff` de `eval_simplify`: recorre el diferando y re-adjunta `tan`/`sec`→`cos≠0`,
  `cot`/`csc`→`sin≠0` (con `&mut ctx` para construir el `cos`/`sin`), dedup; gateado a `diff` (plain eval
  intacto); funciones solas/sumas no duplican; argumentos constantes no aportan. Form-only: el valor de la
  derivada no cambia. 5 fixtures actualizados (mejoras legítimas), huella 0 deltas, smokes verdes, test
  nuevo. Peldaño: análogos sqrt/log que se cancelan — re-adjuntar el dominio COMPLETO del diferando.)*
- [x] **(F) Inecuaciones polinómicas de grado≥3 expandidas**: `solve(x^4-5x^2+4<0)` devolvía ∅
  (real `(-2,-1)∪(1,2)`) y `solve(x^3-x<0)` un aislamiento garbled `solve(x=x^(1/3))`; las formas
  factorizadas equivalentes sí resolvían. Las estrategias degree-aware (Quadratic, RationalRoots)
  gatean `op==Eq`, así que las inecuaciones caían a aislamiento de variable.
  *(graduado 2026-06-22 b3089a272: nuevo `try_factor_polynomial_inequality` — antes de aislar, si
  es inecuación de polinomio univariado grado≥3 REDUCIBLE, reescribe a `factor(p) OP 0` y re-aísla
  → ruta product-sign existente (que ya resolvía factorizadas). Guard anti-loop sobre la forma
  CRUDA + re-entrada por `isolate_equation`. Irreducibles (`x^4-10`, `x^3-2`), cuadráticas,
  ya-factorizadas y ecuaciones declinan/intactas. Factorizador exacto BigRational. Scoping por
  subagente; brute-force 10/10 vs sympy; huella NONE. Las cúbicas IRREDUCIBLES ya no dan garble:
  ahora residual honesto (2026-06-22, guard de no-progreso en `residual_solution_set`:
  `solve(x^3+x+1<0) → solve(x^3+x+1 = 0, x)`). Peldaño restante: no hay raíz cerrada (Cardano) ni
  el op de la inecuación en el residual.)*
- [x] **(F) Límite ∞−∞ del mismo signo colapsaba a 0**: `limit(1/sin²x − 1/x², x, 0)` devolvía `0`
  (valor real 1/3); igual `csc²x − 1/x²` y `1/x² − 1/(x²+x³)` (real divergente). La resta de dos
  sub-límites infinitos iguales se colapsaba por un atajo estructural.
  *(graduado 2026-06-22 bcdbd3317: `finite_sub_result` tenía un atajo `lhs == rhs ⇒ 0` que mordía
  porque `mk_infinity` interna `Constant(Infinity)` — ambos sub-límites devuelven el MISMO ExprId.
  Ahora declina (`Option`) si AMBOS operandos son infinitos del MISMO signo (∞−∞ indeterminado),
  consistente con el rechazo existente de `ln(x)−ln(x)`; ∞−finito y signo-opuesto siguen
  resolviendo a ±∞. Wrong-answer → residual honesto. Localizado por subagente de scoping; huella
  NONE. El VALOR 1/3 se entrega después (2026-06-22, `combine_difference_over_common_denominator`:
  al declinar ∞−∞ se combina `lhs−rhs` sobre denominador común y se reintenta → `1/sin²x−1/x² = 1/3`).
  Las grafías de recíproco-trig se cierran después (2026-06-22, `as_fraction` extendido a
  csc/sec/cot y `(a/b)^k`: `csc²x−1/x²=1/3`, `cot²x−1/x²=-2/3`). Peldaño restante: `1/tan²x−1/x²` y
  otras con tan crudo cuyo sub-límite no resuelve (las reglas de polo no cubren tan).)*
- [x] **(F) gcd de polinomios devolvía un NO-divisor**: `gcd(x²+x, x²-x)` devolvía `x²+x` (que no
  divide a `x²-x`) en vez de `x`, y `gcd(x²+x+1, x²-x+1)` devolvía `x²+x+1` en vez de `1` (coprimos).
  La clave AC del gcd estructural ignoraba el signo de los términos aditivos, colisionando `x²+x`
  con `x²-x`.
  *(graduado 2026-06-22 41a827a5b: la rama `Add` de `expr_key_hash` usaba `add_terms_no_sign`, que
  aplanaba `Add`/`Sub`/`Neg` DESCARTANDO el signo → ambos hasheaban a `{x²,x}` → `expr_equal_ac`
  true → el "factor común" era el primer argumento entero (no-divisor). Reemplazado por
  `add_terms_signed` (rastrea el signo, hashea negados vía `expr_key_neg`, igual que la rama `Sub`
  ya hacía). El bug solo mordía cuando los términos sin-signo coincidían (`+x`/`-x`);
  `gcd(x²+2x, x²-3x)` ya era correcto. Cazado por hunt adversarial; brute-force 16/16 vs sympy;
  huella NONE. Peldaño: grupos B (ineq grado 4 → ∅) y D (lim ∞−∞ → 0) del mismo hunt.)*
- [x] **(F) Endpoints surd de inecuaciones sin ordenar por valor**: `solve(x²-3<0)` devolvía
  `(√3, -√3)` (intervalo invertido = ∅) en vez de `(-√3, √3)`, y `solve(x²-3>0)` una unión que
  cubría todo ℝ; misma falla en `x²-2`, `2x²-6`, `x²-x-1` (raíz φ). Los endpoints RACIONALES
  (`x²-4<0 → (-2,2)`) sí se ordenaban.
  *(graduado 2026-06-22 86a77d28d: `compare_values` resolvía racionales (rama Number) pero caía a
  comparación ESTRUCTURAL para surds — que no refleja el valor. Añadido `compare_quadratic_surds`:
  ordena por el signo EXACTO de `a−b` como surd `P+Q√n` (n≥0, misma fórmula que
  `provable_sign_vs_zero`, nunca f64), reusando `as_linear_surd`; φ=(½+½√5) reconocido localmente.
  Radicandos distintos/transcendentes declinan → fallback estructural. Cazado por hunt adversarial
  5-lente con refutación sympy/mpmath; brute-force 14/14 ordenados; huella NONE. Peldaño: surds de
  radicando distinto y raíces de orden ≥3 siguen estructurales; display sin racionalizar es otro
  wart.)*
- [x] **(F) `log(b, 0)` ignoraba la base (signo erróneo)**: `log(1/2, 0)` devolvía `-∞` cuando el
  valor real es `+∞` (`log_b(0)=ln(0)/ln(b)`, y `ln(b)<0` para `0<b<1` invierte el signo), y
  `log(1, 0)` devolvía `-∞` en vez de `undefined` (base 1 degenerada). La rama `n=0` emitía `-∞`
  incondicionalmente, sound solo para `b>1`.
  *(graduado 2026-06-22 b3cb0efd5: la rama `n.is_zero()` de `try_rewrite_evaluate_log_expr`
  ahora gatea por la base: `b>1 → -∞`, `0<b<1 → +∞`, `b=1` o `b≤0 → undefined`; `e`/`π`/base
  simbólica conservan `-∞` (régimen `b>1`). Cazado por verificación adversarial 2-lente (sonda +
  refutación con mpmath/sympy) — los tests verdes solo ejercían bases >1, enmascarando el defecto.
  Huella NONE. La base FRACCIONARIA se cierra después (2026-06-22, `eval_log_rational_full`:
  `log(1/2,16)→-4`, `log(2/3,9/4)→-2` vía vector de exponentes primos con signo). La base
  DEGENERADA también se cierra (2026-06-22, guard de validez de base: `log(1,1)→undefined` —antes
  `0`—, y `log(0,·)`/`log(neg,·)→undefined`).)*
- [x] **(S) Multiplicación de matrices no-cuadradas → broadcast malformado**: `[[1,2],[3,4]]·[[5,6,7],[8,9,10]]`
  (2x2·2x3) devolvía una matriz-de-matrices malformada en vez de `[[21,24,27],[47,54,61]]`;
  `[[1],[2],[3]]·[[4,5,6]]` (3x1·1x3) igual; y un producto de dimensiones incompatibles (`2x3·2x2`)
  fabricaba una matriz finita en vez de quedar sin evaluar. Los productos CUADRADOS pequeños
  (2x2·2x2) sí funcionaban, ocultando el defecto. (P0 soundness en comando no-cálculo; exento del
  orden de fase.)
  *(graduado 2026-06-22 329f5534f: dos capas que se enmascaraban — `MatrixMultiplyRule` (registrada
  ANTES que `ScalarMatrixRule`) SÍ producía el producto correcto, pero cada entrada de salida es una
  suma sin plegar de `inner_dim` productos → el resultado cruzaba el presupuesto anti-empeoramiento
  (>30 nodos abs, >1.5×) y el simplificador lo rechazaba (`continue`), cayendo a `ScalarMatrixRule`
  que difundía una matriz como escalar. Fix: (1) `MatrixMultiplyRule` declara `budget_exempt`
  ACOTADO (MAX_N=16, celdas≤256, inner≤16) para que la reducción definitoria se confirme antes de
  plegar; (2) `ScalarMatrixRule` declina cuando AMBOS operandos son matrices → mismatch queda como
  residual honesto. Diagnosticado instrumentando `try_eval_matrix_mul_expr` + leyendo el bucle de
  aceptación `apply_rules`. Verificado vs sympy (3x3·I, 1x3·3x3, 3x2·2x4, 1x1, fracciones); huella
  `filtered_out` +7 (tests añadidos), passed/failed idénticos. Peldaño: `M·M` no-cuadrada se colecta
  a `M^2` sin evaluar y los productos que exceden los caps quedan sin evaluar — ambos residuales
  honestos.)*

- [x] **(S) Conmutador de matrices `A·B − B·A` colapsaba a `0`**: `[[1,2],[3,4]]·[[5,6],[7,8]] −
  [[5,6],[7,8]]·[[1,2],[3,4]]` devolvía `0` cuando el valor real es `[[-4,-12],[12,4]]` (la
  multiplicación de matrices NO es conmutativa). El defecto solo aparecía con `--steps off` (la ruta
  rápida sin escucha de pasos); con `--steps on/compact` la regla de multiplicación de matrices se
  aplicaba primero y daba el valor correcto, ocultando el bug. (P0 soundness en comando no-cálculo —
  wrong-answer; exento del orden de fase. Hallado por el hunt adversarial multiagente ultracode,
  confirmado 2/2 lentes y verificado vs sympy.)
  *(graduado 2026-06-23 a5c13a907: la capa de root-shortcuts del orquestador (`try_standard_*` exact-zero /
  equivalent-pair) y los matchers de cancelación aditiva (`exprs_equal_up_to_mul_factor_order_and_sign`,
  `pairwise_matches`) comparan productos como MULTICONJUNTOS de factores conmutativos → `A·B` y `B·A`
  se tratan como iguales y se cancelan. Fix en capas con un único predicado `term_has_matrix_product_factor`
  (una matriz literal como factor de un `Mul`/`Div`/`Pow`): (1) los dos bloques de root-shortcuts
  (`if matches!(context_mode, Standard|Auto)`, uno gateado por `!has_step_listener()` — de ahí la
  divergencia steps on/off) se saltan enteros si hay producto matricial → el pipeline normal evalúa al
  valor real; (2) `try_standard_exact_zero_equivalence_shortcut` declina; (3)
  `exprs_equal_up_to_mul_factor_order_and_sign` NO ordena factores cuando hay una matriz (compara
  posicionalmente); (4) `exprs_match_for_cancellation`(`_leaf`) restringen a igualdad estructural
  order-preserving (`compare_expr`). El predicado es bounds-safe (salta ExprId centinela como
  `ln_base_sentinel`, que hacía panic en `Context::get` desde la travesía completa del árbol).
  Cancelaciones genuinas preservadas: `A·B − A·B → 0`, `M − M → 0`, escalares `x·y − y·x → 0`,
  `2·M − M·2 → 0` (el escalar SÍ conmuta), `det(M)·x − x·det(M) → 0` (la matriz solo está dentro del
  arg de la función). Verificado vs sympy (2x2, nilpotentes, 3x3); huella guardrail+pressure sin deltas
  estructurales (solo timing + `filtered_out`); tests de contrato nuevos en ambos modos de pasos.
  Peldaño: la exponenciación de matrices (`M^2`) y `M^2 − N^2` quedan SIN evaluar — under-answer
  honesto preexistente, no regresión.)*

- [x] **(S) gcd multivariable devolvía `1` (coprimalidad falsa)**: `gcd(x²−y², x²+2xy+y²)` devolvía
  `1` cuando el gcd real es `x+y`; igual `gcd((x+y)², (x+y)(x−y))` → `1` y `gcd(x³−y³, x²−y²)` → `1`.
  El gcd con factor común monomial-extraíble (`gcd(x²+xy, xy+y²)` → `x+y`) sí funcionaba, ocultando el
  defecto. Afirmar coprimalidad sin probarla es un wrong-answer. (P0 soundness en comando no-cálculo;
  hallado por el hunt adversarial multiagente ultracode #2, verificado vs sympy.)
  *(graduado 2026-06-23 84e1c6ef5: bare `gcd` → `select_poly_gcd_mode` = Structural → `gcd_exact` (capas exactas
  Layer 1/2/2.5) cae a un fallback trivial "return 1" cuando sus heurísticas multivariables FALLAN en
  encontrar el factor — devolviendo `1` tanto para coprimos como para fallos. Fix: en el modo Structural
  de `compute_poly_gcd_unified_with`, cuando las capas exactas devuelven `1`, consultar el Zippel modp
  (completo para estos casos) y aceptar su candidato SOLO tras verificación por división exacta
  (`MultiPoly::div_exact`) de que divide ambas entradas — el modp de un primo es probabilístico, la
  verificación lo vuelve sólido (nunca un factor espurio). `None` → se mantiene `1` (coprimos genuinos).
  Gateado a `goal != CancelFraction`: cancelar un factor de una fracción puede soltar una condición de
  polo removible, así que la cancelación de fracciones queda conservadora (preserva el bloqueo de modp
  existente). Verificado vs sympy (factor lineal, multiplicidad `(x+y)²`, productos, coprimos, signo);
  gcd entero/univariable intactos. Huella workspace 12310/0; guardrail+pressure sin deltas estructurales
  (el delta de `calculus_integrate_command_matrix_smoke` es no-determinismo preexistente, reproducible con
  el mismo código; integración no llama esa ruta de gcd). Peldaño: implementar un PRS subresultante exacto
  determinista evitaría depender del modp+verificación; n>2 variables y casos fuera de presupuesto del
  modp quedan como residuales.)*

- [x] **(S) `cosh(3x) − cosh(x)` colapsaba a `0`**: devolvía `0` cuando el valor real es
  `4·cosh³(x) − 4·cosh(x) = 4·cosh(x)·sinh²(x)` (≈8.5246 en x=1). Igual `cosh³(x) − cosh(x)` → `0`
  (= `cosh·sinh²`). Los análogos circulares (`cos(3x)−cos(x)`, `sin³−sin`) ya eran correctos, ocultando
  el defecto hiperbólico. (P0 soundness en simplificación no-cálculo; hallado por el hunt ultracode #3.)
  *(graduado 2026-06-23 504776a17: la regla "puente de cancelación pitagórica hiperbólica"
  (`ExpandHyperbolicPythagoreanFactorToEnableCancellationRule`) reconoce `k·cosh³ − k·cosh` (modo
  `FactorThenRewrite`) para reescribirlo a `k·cosh·sinh²` y habilitar cancelación con términos vecinos —
  pero su ruta DIRECTA a profundidad 0 construía incondicionalmente un rewrite a `0`, asumiendo que toda
  la expresión se anula. Falso para una diferencia AISLADA (`cosh³−cosh = cosh·(cosh²−1) = cosh·sinh²`,
  nunca idénticamente 0). Fix: la ruta directa DECLINA el modo `FactorThenRewrite` (deja la forma expandida
  correcta, igual que `y³−y` no se factoriza con avidez); solo `AlreadyFactored` (término factorizado que
  cancela a su compañero, p.ej. `sinh·(cosh²−1)−sinh³`) sigue → 0. El scope-rewrite multi-término (que
  verifica negación de los términos restantes) queda intacto: las identidades genuinamente cero siguen
  colapsando. Verificado numéricamente + identidades cero preservadas. Huella workspace 12311/0;
  guardrail+pressure sin deltas de estado. **Coste educativo aceptado** (decisión del operador, soundness >
  detalle): la prueba `derive` desnuda de `2sinh(2x)sinh(x)=4cosh³−4cosh` pasa de 2 pasos visibles a 1 (el
  paso de ángulo-triple se funde en la normalización); la variante con passthrough conserva los 2. Peldaño:
  restaurar el paso explícito de ángulo-triple en la narrativa `derive` desnuda — lógica del motor `derive`,
  follow-up separado.)*

- [x] **(S) Inecuaciones de SUMA de valores absolutos devolvían "No solution"**: `|x|+|x-1| < 5`
  devolvía "No solution" cuando la solución es `(-2, 3)`; igual `<= 3` → "No solution" (real `[-1,2]`),
  y la forma `>` dejaba un residual malformado. El abs SIMPLE (`|x| < 5 → (-5,5)`) sí funcionaba,
  ocultando el defecto en las SUMAS. Afirmar conjunto vacío cuando hay un intervalo es wrong-answer.
  (P0 soundness en `solve` de inecuaciones; hallado por el hunt ultracode #4.)
  *(graduado 2026-06-24 038fc9c79: el solver usaba *aislar-un-abs-y-dividir-casos*; para una suma pierde los demás
  términos y la intersección de ramas incompletas colapsa a vacío. Nuevo solver EXACTO por breakpoints
  (`try_solve_sum_of_abs_inequality`): recast `lhs-rhs {op} 0`, descomponer en `Σ kᵢ·|mᵢ·x+bᵢ| + afín`
  (inner lineal), breakpoints `aᵢ=-bᵢ/mᵢ` (BigRational), en cada tramo entre breakpoints el LHS es lineal
  (signo de cada `|·|` por punto de prueba interior) → resolver `M·x+B {op} 0` exacto → intersecar tramo →
  unir. Sin supuesto de convexidad (correcto con coeficientes negativos). Cableado robusto en
  `solve_with_ctx_and_options` (nivel superior, antes del enrutado de aislamiento), gateado a inecuaciones
  con ≥2 términos abs (single-abs / ecuaciones / no-abs intactos). Verificado: oráculo independiente
  Fraction-based, 600 casos aleatorios, 0 fallos; + sympy (coeficientes, bps racionales, no convexos, 3
  términos, los 4 operadores, bordes abiertos/cerrados, vacíos genuinos). Workspace 12312/0; guardrail+
  pressure sin deltas de estado (solo no-determinismo preexistente de smokes diff/integrate, ajeno al
  solver). Peldaño: ECUACIONES de suma de abs (`|x|+|x-1|=3`) siguen dando residual — misma técnica
  piecewise, follow-up; y pasos didácticos del solver piecewise.)*
  *(peldaño ECUACIONES graduado 2026-06-24 baf9fbb52: el solver `try_solve_sum_of_abs_inequality` se gateaba
  solo a inecuaciones; las ecuaciones caían al viejo aislar-un-abs y daban residual basura
  (`|x|+|x-1|=3`) o un WRONG-ANSWER (`|x|+|x-1|=1 → (-∞,1]`, real `[0,1]`). Generalizado a `Eq` (renombrado a
  `try_solve_sum_of_abs_relation`): por segmento, pendiente 0 ⇒ se cumple sii la constante es 0 (segmento entero,
  mínimo plano); pendiente ≠0 ⇒ cruce único `x=-c/m` como intervalo cerrado degenerado `[p,p]` (intersect no
  maneja Continuous∩Discrete), colapsado a `Discrete` al final solo si todo son puntos. Sin supuesto de
  convexidad: coef con signo dan rayos (`|x|-|x-1|=-1 → (-∞,0]`) y remanente afín funciona (`|x|+|x-1|+x=3 →
  {-2,4/3}`). Resultados: `{-1,2}`, `[0,1]`, `No solution`, `{-1/3,7/3}`. Verificación adversarial: oráculo
  independiente `fractions` 400 sumas aleatorias, 0 mismatches. Workspace 12316/0; huella sin deltas. Sigue
  pendiente: pasos didácticos del solver piecewise y valor absoluto ANIDADO.)*

- [x] **(S) Ecuación de UN abs reorientada a `var = c − |arg|` fugaba residual malformado**: `x + |x-1| = 3`
  y `|x-1| = 3 - x` devolvían `Solve: solve(x - (3 - |x-1|) = 0, x) = 0` (sintaxis interna filtrada con
  ok=true) en vez de `{2}`; el valor absoluto ANIDADO `|x + |x-1|| = 3` igual (correcto `{2}`); coef≠1 y sumas
  divididas (`2x - |x| = 1`, `(|x|+|x-1|)/2 = 1`) también fugaban. (P0 honestidad en `solve`; hallado al sondear
  el peldaño de abs anidado de #4.)
  *(graduado 2026-06-24 e144dbb2d: cuando una ecuación de un abs se reorienta a `var = α·|arg| + β`,
  `try_abs_self_equation` solo reconocía el caso estructural `|f|=±f` y fugaba el resto. Es piecewise-lineal con
  un breakpoint → se resuelve con el MISMO core exacto del solver de sumas: extraído `solve_decomposed_abs_relation`
  (loop por segmentos ya verificado con 400 casos) y compartido; nuevo `try_single_abs_affine_equation` descompone
  `var - rhs`, exige 1 término abs y delega al core, cableado como fallback tras `try_abs_self_equation` (preserva
  su huella). Además `decompose_sum_of_abs.collect` ahora lleva un `scale` racional y DISTRIBUYE factores
  constantes `Mul(const,·)`/`Div(·,const)`, exponiendo los abs ocultos tras `(…)/2` — arregla también el
  top-level `(|x|+|x-1|)/2=1`. Sin convexidad: `x=|x| → [0,∞)`. Verificación adversarial: 2 oráculos `fractions`
  (sumas 400/0, un-abso=afín 296/0). Workspace 12317/0; huella sin deltas. Peldaño de presentación: la ruta
  `|f|=f` rinde medio-rectas como "All real numbers if x≥0" en vez de `[0,∞)`.)*

- [x] **(S) Ecuaciones polinómicas en x^(1/q) fugaban residual malformado y perdían todas las raíces**:
  `solve(x-3·√x+2=0)` devolvía `ok=true` con `result="Solve: solve(x - (3·x^(1/2) - 2) = 0, x) = 0"`
  (sintaxis interna del solver fugada, ambas raíces {1,4} soltadas); igual `x^(2/3)-x^(1/3)-2` (→{-1,8}),
  `x-5√x+6` (→{4,9}), `x+√x-6` (→{4}). Son cuadráticas-en-disfraz (polinomio de grado ≥2 en u=x^(1/q)).
  (P0 wrong-answer/leak en `solve`; hallado por el hunt adversarial ultracode de 12 frentes, Cluster A1.)
  *(graduado 2026-06-24 d5695b7ed: la estrategia de Substitution solo cubría EXPONENCIALES (e^x, a^x); para
  potencias racionales la ruta de Isolation reorientaba a `x=f(x)` y `try_recover_isolated_eq` no la cerraba →
  fuga. Nuevo hook top-level `try_solve_rational_power_polynomial` (antes del routing): simplifica lhs-rhs
  (√x→x^(1/2)), exige que x aparezca solo como potencias racionales positivas, q=lcm de denominadores,
  reconstruye el polinomio en u (x^e→u^(q·e)), exige grado ≥2 en u (grado 1 recursaría infinito en la
  retro-sub), resuelve en u recursivamente, y retro-sustituye `x^(1/q)=u_root` recursivamente — el solver
  recursivo aplica el dominio de raíz real (q par descarta u_root<0; q impar la conserva). Verificación
  adversarial: oráculo `fractions` independiente, 300 casos (q∈{2,3,4,5}, grado 2-3), 0 mismatches. Workspace
  12326/0; huella sin deltas. Peldaños del Cluster A: A2 (polinomio en ln(x)) y A3 (suma de dos radicales
  distintos) siguen fugando.)*

- [x] **(S) Inecuación radical con argumento compuesto soltaba el dominio**: `sqrt(x-1) < 3` devolvía
  `(-∞, 10)` cuando la solución es `[1, 10)` — incluyendo puntos donde el radicando `x-1 < 0` y `√` no
  existe en ℝ. Igual `sqrt(2x-1) ≤ 3` → `(-∞, 5]` (real `[1/2, 5]`); `sqrt(x²-4) < 3` daba un solo intervalo
  sin el split del dominio `|x|≥2`. El radical con variable PELADA (`√x < 2 → [0,4)`) sí aplicaba el
  dominio, ocultando el fallo. (P0 soundness en `solve` de inecuaciones; hallado por el hunt ultracode #5.)
  *(graduado 2026-06-24 6668d00a5: `intersect_inequality_with_function_domain` gateaba en `arg_is_var` y
  devolvía el set SIN tocar para argumentos compuestos → la inversión solo restringía `g(x)` contra el
  umbral y dejaba la región `g(x) < 0`. Fix: calcular el dominio como la solución de `arg ≥ 0` (even root) /
  `arg > 0` (log) resolviéndola para la variable (recursión acotada: el arg ya es no-radical), e intersecar.
  Variable pelada (fast-path), `>`/`≥` (el bound implica el dominio), corrección de rango (`√<c≤0` → ∅), y
  no-radical intactos; fallback honesto si el dominio no reduce. Verificado: oráculo independiente
  Fraction-based 500 casos (afines, 4 ops, coeficientes, `≤0` discretos), 0 fallos; + sympy (afín,
  cuadrático con split, ln, rango). Workspace 12313/0; guardrail+pressure sin deltas de estado (solo
  no-determinismo preexistente de smokes diff/integrate). Peldaño cosmético: la cota inferior surd `-√13`
  renderiza como `-13·13^(-1/2)` (estilo preexistente del path de cotas surd) — no soundness.)*

- [x] **(S) El TEXTO de una potencia anidada se renderizaba sin paréntesis (round-trip incorrecto)**:
  `(4·x²)^(1/2)` mostraba el texto `2·x^2^(1/2)`, que re-parsea como `2·x^(2^(1/2)) = 2·x^√2` — una
  expresión DISTINTA (el LaTeX sí era correcto). Como `^` es asociativo por la derecha, una potencia cuya
  base es a su vez una potencia debe parentizarse. (Relacionado con el frente #6/#7 del hunt; el otro
  síntoma —`(x²)^(1/3) → x^(2/3)`— NO es bug: es correcto bajo la semántica de potencia REAL del engine,
  ver abajo.)
  *(graduado 2026-06-24 1b17bcf02: el formateador de TEXTO solo parentizaba la base de una potencia si
  `base_prec < op_prec`; `Pow` y el operador `^` comparten precedencia (3), así que una base-potencia (`x^2`)
  no se parentizaba → `x^2^(1/2)`. Fix de una línea: `base_prec <= op_prec` (parentiza la base de igual
  precedencia; `Pow` es el único nodo de precedencia 3). Verificado con fuzzer round-trip (texto re-evaluado
  vía el engine) y consistencia del engine en bases negativas; 10 tests que enshrinaban el rendering ambiguo
  (`(u²+1)^(1/2)^2`, `x^2^2`) actualizados al correcto. Workspace 12314/0; formatter 125/0; huella sin
  deltas de estado. **NO-bug documentado**: `(x²)^(1/3) → x^(2/3)` es CORRECTO — el engine usa potencia
  REAL (`(-8)^(2/3)=4`, `(-3)^(2/3)-|−3|^(2/3)=0`, denominador par sí lleva abs `(x²)^(1/4)→|x|^(1/2)`); un
  oráculo sympy/mpmath de rama compleja lo marca como wrong-answer pero es FALSO POSITIVO. Peldaño:
  sub-simplificación residual (`(4x²)^(1/2)` no llega a `2|x|` en una pasada) — idempotencia, follow-up.)*
- [x] **(S) La identidad complementaria arcsin+arccos colapsaba con argumento fuera de dominio**:
  `arcsec(1/2)+arccsc(1/2)` devolvía `π/2`, pero para `|x|<1` AMBOS términos son indefinidos en ℝ
  (`arcsec`/`arccsc` exigen `|x|≥1`). Internamente se reduce a `arccos(2)+arcsin(2)`, y la identidad
  `arcsin(x)+arccos(x)=π/2` se disparaba sin verificar `x∈[-1,1]`, pese a que `arcsin(2)`/`arccos(2)` no
  existen en ℝ. (P0 soundness en `eval` de inversas trig; hallado por el hunt ultracode #8.)
  *(graduado 2026-06-24 7856578b9: `try_plan_inverse_trig_sum_pair_expr` reconocía la pareja
  complementaria `arcsin(arg)+arccos(arg)` (y la forma `arcsec/arccsc` que se reescribe a ella) por
  IGUALDAD de argumentos y colapsaba a `π/2` sin gatear el dominio. Fix: dentro de la rama `args_equal`,
  `if is_number_outside_unit_interval(ctx, arg_i) { return None; }` — un argumento concreto provablemente
  FUERA de `[-1,1]` deja la suma como residual honesto (`arcsin(2)+arccos(2)`), no la colapsa. Casos válidos
  intactos: `arccos(1/2)+arcsin(1/2)→π/2`, `arccos(±1)+arcsin(±1)→π/2`, `arcsec(2)+arccsc(2)→π/2`,
  simbólico `arccos(x)+arcsin(x)→π/2` con `-1≤x≤1`. Test de regresión
  `test_eval_complementary_inverse_trig_respects_domain`. Workspace 12315/0; clippy/fmt limpios;
  guardrail+pressure sin deltas de estado. **Peldaño (b)**: el simbólico `arcsec(x)+arccsc(x)→π/2` NO
  arrastra la condición `|x|≥1` (required_display vacío), mientras que la forma `arccos(1/x)+arcsin(1/x)`
  sí mantiene `x≤-1 or x≥1`: hueco de propagación de condición a través de la reescritura multi-paso
  arcsec→arccos — follow-up de honestidad, NO wrong-answer (el valor `π/2` es correcto donde `arcsec`
  existe). **RESUELTO 2026-06-24 9e82cbb13**: `push_intrinsic_function_requires` (diagnostics.rs) —el
  escaneo que emite las condiciones de dominio inverse-trig recorriendo la expresión— no tenía arm para
  Arcsec/Asec/Arccsc/Acsc; el individual `arcsec(x)→arccos(1/x)` sobrevive porque el arccos queda en el OUTPUT,
  pero la SUMA colapsa a `π/2` sin inverse-trig que anclar → condición vacía. Fix: arm nuevo que construye el
  recíproco `1/arg` y reusa `inverse_unit_interval_intrinsic_requirement` → `NonNegative(1-(1/arg)²)`, idéntico
  a lo que emite la reescritura; ahora `arcsec(x)+arccsc(x)→π/2` arrastra `x ≤ -1 or x ≥ 1`, y afín/desplazado
  salen gratis (`arcsec(2x)`→`x ≤ -1/2 or x ≥ 1/2`, `arcsec(x+1)`→`x ≤ -2 or x ≥ 0`). Gate `!inside_calculus_call`
  preserva `diff(arcsec(...))` vacío; dedup colapsa el duplicado del individual. Test de regresión ampliado;
  workspace 12315/0; huella sin deltas. Peldaño: barrer otros recolectores por value-dependent sin arm
  (recíprocas hiperbólicas).)*
- [x] **(F) Raíz extraña fuera de dominio con radicando transcendente**: `solve(ln(x)+ln(x-3)=1)`
  devolvía también la raíz extraña `(3−√(9+4e))/2 ≈ −0.73`, que viola el dominio `x>3` que el
  propio solver deriva (el filtro de raíces extrañas declinaba en radicando NO racional `9+4e`).
  *(graduado 2026-06-22 7100e3afb: `provable_sign_vs_zero_const_radicand` decide el signo de
  `A+B·√R` con R constante transcendente por `sign(B)·sign(R−(A/B)²)`, probando la comparación
  `R vs (A/B)²` con `rational_bounds` —aritmética de intervalos con cotas racionales PROVABLES
  `2.718<e<2.719`, `3.141<π<3.142`, exacta, decide solo en separación estricta; frontera ⇒ None ⇒
  conserva—; encadenado tras el probador racional en `root_violates_required_condition` por
  `.or_else`. Cubre `9+4e`, `e²+1`, π y grados ≤16. Barrido adversarial conteo-de-raíces
  `solve(ln(x)+ln(x-a)=c)` (25 casos) = nº real de raíces válidas, 0 mismatches; huella NONE.
  NUNCA f64 para keep/drop. Peldaño: bases-potencia de base negativa y divisiones por
  no-constante declinan (conservan, sound).)*
- [x] **(F) Fórmula cuadrática pierde un factor del radical con discriminante factorizado**:
  `solve(x^2-4*x-e=0)` devolvía `(4±√(4+e))/2` (≈3.30/0.70, valor FALSO que NO satisface la
  ecuación) en vez de `2±√(4+e)`; mismo error en toda cuadrática cuyo discriminante simplifica a
  la forma factorizada `k²·(suma)` con término simbólico (`4·(4+e)`, `4·(1+e)`). Se propagaba a
  `solve(ln(x)+ln(x-a)=c)` (raíz válida perdida → `No solution`).
  *(graduado 2026-06-22 1dbe290ee: `pull_square_from_sqrt` dividía el cofactor `R` —ya extraído
  por la rama Mul de `split_numeric_factor` de `√(k²·R)`— por `k` una SEGUNDA vez, dando
  `k·√(R/k²)=√R` (mitad del valor real). La división por `k` solo es válida en la rama Add (donde
  `split` devuelve la suma entera sin dividir); ahora se gatea en `base_is_additive`, leyendo la
  forma del ARGUMENTO original, no la del resto extraído. `sqrt(16+4e)` standalone ya daba
  `2√(4+e)` correcto — el bug solo vivía dentro del constructor de raíces con el discriminante
  pre-factorizado. Unit-test-locked (valor numérico, no forma); huella guardrail+pressure NONE.
  Descubierto por verificación adversarial del filtro de raíces extrañas (ciclo hermano). Peldaño:
  el filtro de dominio para radicandos transcendentes es el ciclo siguiente, ortogonal.)*
- [x] **(F) `0·∞` plegado a 0 en punto finito**: `limit(x·sinh(1/x²),x,0)`
  devolvía `0` cuando el límite real es `+∞` (sinh(1/x²)→+∞, y 0·∞ es
  indeterminado, no 0); mismo error en `x·cosh(1/x²)`, `x·exp(1/x²)`,
  `x·cosh(1/x)` (valor finito FALSO, no residual honesto).
  *(graduado 2026-06-14 e0710101b: dos causas — (1)
  `finite_total_real_unary_result` devolvía el cofactor como `sinh(∞)` SIN
  plegar (el fold de saturación cubría exp/abs/sin/cos/atan pero no
  sinh/cosh/tanh), leyéndose como acotado aguas abajo; ahora el fallback
  corre `fold_infinity_saturation` (sinh(∞)→∞, cosh(∞)→∞, tanh(∞)→1,
  exp(−∞)→0); (2) `finite_mul_result` devolvía 0 si CUALQUIER factor era 0
  sin comprobar que el otro fuese finito; ahora declina (residual honesto)
  cuando un factor es 0 y el otro ∞. `x·exp(−1/x²)→0` y `x·tanh(1/x²)→0`
  siguen correctos. Descubierto al prototipar la sustitución recíproca
  u=1/x para límites en ∞ — revertida por disparar justo este hueco —;
  huella de scorecard sin cambios, ningún fixture público lo capturaba.)*
- [x] **(F) `(x^m)^n` con m par perdía el valor absoluto**: `(x^2)^(3/2)`
  se aplanaba a `x^3` (signo FALSO para x<0) y `diff((x^2)^(3/2),x)` daba
  `3·x^2` en lugar de `3x·√(x²)`. Reportado por el operador. El motor ya
  hacía bien `(x^2)^(1/2)→|x|` pero el resto de la familia se filtraba.
  *(graduado 2026-06-15 6be8b3b79: `try_rewrite_power_power_even_root_abs_expr`
  estaba gateado por `is_half(outer)` —solo el exponente exacto 1/2—; generalizado
  al INVARIANTE real: con exponente interno m PAR, `x^m=|x|^m≥0` ⇒ `(x^m)^n=|x|^(m·n)`
  exacto; se emite `x^(m·n)` solo cuando el numerador de m·n es PAR (signo ya absorbido),
  si no se conserva `|x|`. Ahora `(x²)^(3/2)=|x|·x²`, `(x²)^(1/4)=|x|^(1/2)`,
  `((x+1)²)^(3/2)=|x+1|·(x+1)²`; intactos `(x⁴)^(1/2)=x²`, `(x²)^(1/3)=x^(2/3)`, m impar,
  y los radicales de base-Add. Verificación adversarial 3-lente ~730 probes, 0 defectos
  de soundness; huellas guardrail+pressure byte-idénticas. Peldaño: la ruta de coeficiente
  cuadrado-perfecto `(4·x²)^(1/2)→2·x^2^(1/2)` (valor interno correcto 2|x|, pero el display
  de la potencia anidada re-parsea a `2·x^(√2)` — hazard de presentación PRE-EXISTENTE y
  ortogonal, distinto root cause: distribución del radical sobre el producto + formatter
  sin paréntesis en bases-potencia).)*
- [~] **(F) FTC en borde inferior singular**: `d/dx ∫_0^x f(t) dt = f(x)`
  presentado por las rutas de diff-de-integral SIN comprobar convergencia en el
  borde constante. Para integrandos divergentes en 0 (`t^(-3/2)`, `ln(t)/t`,
  `1/t`) el término de borde `f(0)·0` colapsa a 0 y se devuelve una derivada
  FINITA para una integral DIVERGENTE — valor falso, no residual honesto. El
  hueco vive en VARIAS rutas (la regla Leibniz y las rutas de presentación
  verificadas/log-potencia), así que un gate solo-Leibniz no basta.
  *(parcial 2026-06-15 782736593: el PANIC subyacente `0^(-1/2)` →
  `division by zero` (recip de un factor externo 0 en `try_rewrite_evaluate_power_expr`)
  ya está arreglado: `0^(-p)` → `undefined` (valor correcto, como `0^(-1)`). Esto
  cierra la familia de POTENCIA fraccionaria divergente — `t^(-3/2)`, `t^(-4/3)`,
  `t^(-5/2)`, `t^(-7/3)` `[0,x]` → undefined (antes panic o `x^(-p)` falso); el
  convergente `t^(-2/3)` (p<1) mantiene `x^(-2/3)` correcto. Queda la familia
  `ln(t)^k/t` `[0,x]` → `ln(x)^k/x` (falso, ruta sibling) y la capacidad removible
  `sin(t)/t→sin(x)/x`. Peldaño: certificado de convergencia compartido (reusar el
  motor de definidas, que ya devuelve finito sii `∫_0^c f` converge) gateando TODAS
  las rutas de presentación FTC. Cazado por verificación adversarial 2-lente.)*
  *(fuga `infinity^k` cerrada 2026-06-15 4ae92d1ea: el ÚLTIMO eslabón divergente —
  la rama de bornes simbólicos del FTC (`integrate(f,t,a,x)` con borne constante en una
  singularidad de la antiderivada) devolvía `F(x) − F(a)` arrastrando un término
  `infinity^k`/`+infinity` que un `diff` posterior tira, filtrando una derivada finita
  FALSA (`d/dx ∫_0^x ln(t)/t = ln(x)/x`). `boundary_is_genuinely_nonfinite` lo bloquea
  ANTES de formar `F(x)−F(a)`: chequeo ESTRUCTURAL y consciente de dominio (`ln` de
  constante ≤0, `c/0`, `0^neg`, infinity; un FACTOR cero literal mata singularidades
  removibles `0·ln(0)`, y `ln(0²+1)=ln(1)` es finito) con `as_rational_const` + `eval_f64`
  para argumentos función-de-cero (`sinh(0)`, `e^0−1`). Cierra `ln(t)^k/t`, la familia
  hiperbólica/exp `coth`/`1/tanh`/`1/sinh`/`1/(e^t−1)`→undefined, y los mezclados con
  infinito enmascarado; mantiene los convergentes (`∫ln(t)`, `∫arctan(t)`, `∫sinh(t)`) y
  ordinarios. Se DESCARTÓ un gate basado en LÍMITES (incompleto: no resuelve `t·arctan(t)`
  en 0 → falso positivo; ni distingue residual-divergente de residual-no-resoluble).
  Verificado adversarialmente 3 rondas, 0 unsound, 0 regresiones. Quedan: capacidad
  removible `sin(t)/t→sin(x)/x` y unos pocos convergentes-pero-no-probables (`tan(t)/t`)
  conservadoramente undefined — ambos esperan un certificado de convergencia real.)*
- [x] **(A) Cuelgue del simplificador**: `diff(sin(x)^3*cos(x)^2, x)`
  timeout >30s con `depth_overflow depth=51 phase=Core`; mismo patrón
  da 12s en `diff((x^2*sin(x))/(x+1), x)` y
  `diff((x*ln(x))/(exp(x)*sin(x)), x)`. Respuestas correctas que no
  llegan valen cero. Relacionado: el bucle tan↔sin/cos que obligó a
  construir tan⁵ en forma expandida (ledger 2026-06-12).
  *(graduado 2026-06-12 ab8591792: la clase-cuelgue era explosión
  de probes especulativos de equivalencia-cero — cap de profundidad 2
  + presupuesto 48/pipeline con guard save/restore + franja de 24
  pipelines completos; sin²cos²−sin⁴ 0.4s, diff(sin³cos²) 50ms, el
  hermano preexistente sin⁴+cos⁴−1+2sin²cos² ahora prueba 0 en 1.5s,
  2 timeouts del corpus baseline ahora prueban 0. Quedan como
  peldaños: los cocientes-con-producto de 10-12s — preexistente,
  mejorado de 23.9s→12.5s de rebote —, la suma ancha de 3 identidades
  9s→45s, y el ruido WARN depth=51 en PostCleanup)*
- [x] **(F) `inf` como símbolo libre**: `integrate(e^(-x), x, 0, inf)`
  produce `(e^inf-1)/e^inf` sin aviso; solo `infinity` activa la
  maquinaria. Parsear `inf`/`oo` como infinito o rechazarlos con
  mensaje. *(graduado 2026-06-12 ca78c8164: inf/oo → Constant::
  Infinity en el mapa de constantes del parser + oo reservado;
  el glifo ∞ sigue rechazando con error claro — peldaño cosmético;
  -inf suelto en CLI es parseo de flags del shell, no del parser)*
- [x] **(F) Pasos corruptos**: `integrate(sin(x),x,0,pi)` etiqueta
  "Expandir secante" al evaluar cos(π); `sec(x)^2` tiene traza rota
  (el resultado tan(x) no aparece en ningún paso); pasos no-op
  before==after y ciclos expandir/refactorizar. Filtro de saneado de
  traza (eliminar no-ops, verificar que el último after == resultado).
  *(parcial 2026-06-12 3a43f063e: etiqueta falsa corregida — el
  preámbulo de valores exactos de sec/csc/cot llamaba la tabla trig
  SIN restringir el builtin y reclamaba cos(π)/cos(0) bajo su nombre;
  ahora gatea por su propia función + 2 traducciones nuevas. Quedan:
  la traza rota de sec²/csc² (el paso de integración desaparece
  cuando hay pre-pasos — entre el ensamblado y el optimizador
  semántico, necesita ciclo propio) y el filtro de no-ops. BONUS:
  integrate(sec(x)²+1) residual por el pre-simplificador — ejemplo
  vivo del item P2)*
  *(parcial 2026-06-13 025d98333: traza rota de sec²/csc² resuelta —
  el filtro de productividad de pasos truncaba los ciclos por ÍNDICE
  de estado, pero los pasos always-keep crecen la lista sin registrar
  estado, así que el ciclo tan→sin/cos→tan de sec² recortaba una
  posición de más y borraba el paso "Calcular la integral". Fix:
  registrar filtered.len() por estado y truncar ahí + descartar
  always-keep que sean no-ops de display. Verificado adversarialmente
  (2 lentes, 0 regresiones; huellas byte-idénticas). BONUS confirmado:
  integrate(sec(x)²+1) ahora resuelve vía la ruta Weierstrass del
  ciclo c6107abd5. Quedan como peldaño: el filtro de no-ops para
  reglas NON-always-keep — "Convert exp to Power" y "Agrupar términos
  semejantes" emiten pasos before==after en integrate(x/e^x),
  integrate(sin(x)²,x,0,π) — y las trazas FTC definidas terminan una
  reducción aritmética antes del resultado (muestran 1+1, resultado 2))*

### P1 — capítulos universitarios enteros a 0% (mayor densidad de valor)

- [x] **(F×3) Sustitución trigonométrica**: `sqrt(1-x^2)`,
  `x^2*sqrt(1-x^2)`, `sqrt(4-x^2)/x`, `1/(x*sqrt(x^2-1))` (→ arcsec),
  `1/(x^2*sqrt(x^2+4))`, `sqrt(x^2±a^2)` — capítulo completo de Calc
  II ausente; el semicírculo `∫√(1-x²) [-1,1] = π/2` falla. Nota: las
  formas 1/√(cuadrática) y p(x)/√(cuadrática) SÍ están (split
  Hermite); lo que falta es √(cuadrática) en el NUMERADOR y los
  cocientes con x en el denominador.
  *(parcial 2026-06-12 77ea29595: el lado NUMERADOR completo vía
  p·√q = (p·q)/√q delegado al split Hermite — √(1−x²), x²√(1−x²),
  √(4−x²), √(x²±1) asinh/acosh, √(2x−x²), y el semicírculo
  ∫₋₁¹√(1−x²) = π/2 por doble touch. Quedan: los cocientes con x en
  el DENOMINADOR — √(4−x²)/x, 1/(x·√(x²−1)) → arcsec,
  1/(x²·√(x²+4)) — que necesitan sustitución real u otra identidad)*
  *(completado 2026-06-13 069c38a1d: el lado DENOMINADOR entero vía
  u=√q sobre denominadores monomiales — 1/(x√(x²−1))→arctan(√(x²−1))
  (arcsec, condición honesta x<−1 or x>1), 1/(x²√(x²+4))=−√(x²+4)/(4x),
  √(4−x²)/x, √(1−x²)/x², √(x²±1)/x, m=3. El capítulo de sustitución
  trigonométrica de Calc II queda cubierto en ambos lados. Peldaños:
  radicandos con término lineal (completar cuadrado), m par ≥4,
  denominadores no monomiales (x+1)·√q)*
- [x] **(F) Sustitución algebraica general** (u=eˣ, u=√x, u=ax+b bajo
  radical): bloquea DOS familias de golpe — racionales de eˣ
  (`1/(1+e^x)`, `e^x/(1+e^(2x))` → arctan(eˣ), `1/(e^x+e^(-x))`) y
  radicales lineales (`x*sqrt(x+1)`, `exp(sqrt(x))/sqrt(x)`,
  `1/(sqrt(x)+1)`). Mejor ROI según la sonda: una pasada de
  rewrite+integrate-recursivo resuelve ambas.
  *(graduado 2026-06-12 8298dce96: la mitad u=√(ax+b) completa —
  racionales de (x, √(ax+b)) con coeficientes racionales vía
  x=(u²−b)/a, dx=(2u/a)du delegando a los dueños racionales:
  x·√(x+1), x²·√(x+1), x·√(2x−1), x·(x+1)^(3/2), 1/(√x+1),
  √x/(1+x), √(x+1)/x, pendientes negativas (2x+3)√(5−x), con
  condiciones de dominio honestas por canal (x≥−1 integral vs x>−1
  derivada). Quedan como peldaños: cofactores no racionales en u
  (e^√x/√x, sin(√x)), radicandos mixtos √x·√(x+1), y el cierre
  simbólico de diff(F)−integrando para superficies racionalizadas —
  dos filas van como verification_gap con round-trip numérico)*
  *(parcial 2026-06-12 ec314325b: la mitad u=eˣ completa — todo
  racional sobre átomos e^(kx) con k racional integra vía u=e^(cx),
  c=gcd de pendientes, delegando al backend racional y
  back-sustituyendo: 1/(1+eˣ), eˣ/(1+e²ˣ)→arctan(eˣ), e²ˣ/(1+eˣ),
  (eˣ−1)/(eˣ+1), 1/(e²ˣ−1) con su condición de polo, e^(x/2)/(1+eˣ).
  Quedan: u=√x radicales lineales, y la superficie 1/(eˣ+e⁻ˣ) que el
  pre-simplificador reescribe a 1/(2cosh(x)) antes de llegar al
  integrador — peldaño hiperbólico aparte)*
  *(peldaño hiperbólico 2026-06-14 02bd6d0d2: las recíprocas hiperbólicas
  de primer orden ya integran — `∫1/cosh(u)=arctan(sinh(u))`,
  `∫1/sinh(u)=ln(|tanh(u/2)|)` [sinh≠0] vía la ruta de tabla extendida a
  n=1 (`HyperbolicReciprocalTablePower::First`); arg desde `Function`
  desnuda (1/cosh es `Div(1,Function)`, no `Pow`), escala 1/u' afín, forma
  cerrada (no el verificador estricto, que trata cosh/sinh como átomos
  opacos). `2/(eˣ+e⁻ˣ)=1/cosh` cierra. Round-trip diff−integrando=0
  simbólico y numérico (~1e-11) en ambos signos. Quedan: la versión
  CONSTANTE-en-denominador `1/(2cosh(x))=1/(eˣ+e⁻ˣ)` —gap hiperbólico
  preexistente que afecta a TODAS las potencias (`1/(2cosh²)` también
  residual; el trig sí lo hace), y los múltiplos enteros puros `1/sinh(3x)`
  (el motor expande sinh(3x) antes de integrar))*
  *(peldaño f(√x) 2026-06-15 aa8c9e5f7: el cofactor no-racional `sin(√x)`
  —y de paso `cos(√x)`, `sinh(√x)`, `cosh(√x)`— ya integra vía u=√x →
  2∫u·f(u)du: `∫sin(√x)=2sin(√x)−2√x·cos(√x)`,
  `∫cos(√x)=2cos(√x)+2√x·sin(√x)`, los hiperbólicos análogos. Fue un
  ensanche de despacho del dueño inverse-trig-of-sqrt del ciclo previo
  (renombrado `function_of_sqrt_antiderivative`): el cuerpo ya era genérico
  sobre el builtin y el delegado ∫u·f(u) auto-cierra elementalidad —`tan(√x)`
  se queda residual honesto porque ∫u·tan(u) no es elemental. Unit-test-locked
  (no diff-verifica simbólico), sin delta de scorecard. Quedan: `e^√x` y
  `e^√x/√x` (forma Pow(e,√x), no Function → otro punto de despacho))*
  *(peldaño e^√x 2026-06-15 552d4ee30: cerrado. `∫e^√x=2(√x−1)e^√x` vía punto
  de despacho propio en el brazo Pow (`Pow(E,√x)`, no Function), delegando ∫u·e^u;
  y la familia cofactor `∫H(√x)/√x=2∫H(u)du` —`∫e^√x/√x=2e^√x`,
  `∫sin(√x)/√x=−2cos(√x)`, cos/sinh/cosh análogos— donde 1/√x cancela el u de
  dx=2u du. Se factorizó el tail del dueño en `complete_sqrt_substitution`
  (delegar→back-sub u→√x→plegar→×2). El 1/√x vive como Mul `H(√x)·x^(-1/2)` (el
  motor reescribe /√x a ·x^(-1/2)), con gemelo Div para entrada recursiva. Self-gate
  honesto: e^x/√x (erf), e^√x/x (Ei), sin(x)/√x (Fresnel) siguen residuales.
  Unit-test-locked, sin delta de scorecard. Queda el peldaño de exponente radical
  escalado e^(c√x) y radicandos lineales bajo raíz)*
- [x] **(F) Weierstrass t=tan(x/2)**: `1/(2+cos(x))`, `1/(1+sin(x))` —
  estándar de examen universitario.
  *(graduado 2026-06-13 c6107abd5: racionales de sin(kx)/cos(kx) con
  argumento lineal compartido vía t=tan(kx/2) + pares polinómicos +
  gcd mónico + fallback al backend solo-incondicional: 1/(2+cos x)→
  (2/√3)arctan(tan(x/2)/√3), 1/(1+sin x), 1/(1+cos x)→tan(x/2),
  1/(3+2cos x), 1/(5+4sin x), sin x/(1+sin x), 1/(2+cos 2x),
  1/(sin x+cos x)→atanh. Quedan como peldaños: múltiplos mixtos
  (sin x con cos 2x, necesita pre-expansión de ángulo doble), offsets
  de fase, átomos tan/sec, canal de condiciones a través de la ruta
  de soporte, y el techo de profundidad del simplificador que deja
  4/6 filas como verification_gap)*
- [~] **(A) Motor 0/0 componible en punto finito**: la allowlist no
  invierte (`x/sin(x)` falla siendo `sin(x)/x` soportado), no compone
  (`sin(3x)/sin(5x)` → 3/5, `(1-cos x)/x²` → 1/2, `(sin x - x)/x³` →
  −1/6, `asin(x)/x`, `sinh(x)/x`), no encadena L'Hôpital/Taylor. El
  item de mayor frecuencia en cualquier curso.
  *(parcial 2026-06-13 339496d6e: motor de INFINITÉSIMOS EQUIVALENTES de
  primer orden — `first_order_equivalent_poly` extrae el equivalente
  polinómico de AMBOS lados (`f(u)~u` para sin/tan/asin/arcsin/atan/
  arctan/sinh/tanh con guard `u→0`, `e^u−1~u`, polinomios exactos,
  productos, Neg) y delega al `finite_rational_polynomial_value`
  existente (L'Hôpital polinómico). Cubre INVERSIÓN (`x/sin x=1`),
  COMPOSICIÓN (`sin 3x/sin 5x=3/5`, `sin x/sin 2x=1/2`, `tan 2x/sin 3x=
  2/3`) y los átomos que faltaban (`tan/asin/arctan/sinh/tanh /x=1`).
  Footprint-mínimo: corre tras las reglas sin/exp/log, solo dispara en
  0/0 genuino previamente residual. Verificado adversarialmente (3-lente
  scoping + 2-lente refutación, 141 sondas, 0 violaciones).)*
  *(parcial 2026-06-13 4ccd1b930: peldaño (1) el orden SUPERIOR / Taylor con
  cancelación de sumas GRADUADO — `apply_finite_taylor_quotient_rule` añade
  un motor de Maclaurin autocontenido (orden de truncado 12) que corre TRAS
  la regla de equivalentes y solo sobre 0/0 en x=0 que esos pasos dejaron
  residual. `taylor_at_zero` expande estructuralmente (polinomios exactos,
  Add/Sub/Neg/Mul, Pow(E,arg), Pow(base,n entero≥0), y Function(f,[arg])
  componiendo la serie estándar de f vía Horner; tan=sin/cos por división
  de series). Compara los órdenes mínimos no-nulos: num>den→0, num==den→
  cociente exacto de coeficientes líderes, si no declina. SOUND: el
  truncado es EXACTO para un límite de orden líder (solo descarta órdenes
  estrictamente mayores); la lista honesta sobrevive gratis (`sin(1/x)`
  declina porque su argumento `1/x` es `Pow(x,-1)`, que el constructor de
  series rechaza). Cubre `(1-cos x)/x²=1/2`, `(sin x−x)/x³=−1/6`,
  `(tan x−x)/x³=1/3`, `(e^x−1−x)/x²=1/2`, `(cosh x−1)/x²=1/2`,
  `(arctan x−x)/x³=−1/3`, `(arcsin x−x)/x³=1/6`, composiciones anidadas
  (`(sin(sin x)−x)/x³=−1/3`) y `(sin(tan x)−tan(sin x))/x⁷=−1/30`. Verificado
  adversarialmente (2-lente, 55 sondas vs SymPy, 0 unsound).)*
  *(parcial 2026-06-14 9ae1f606c: la franja EXPONENCIAL de (4) L'Hôpital
  general — combinaciones lineales de exponenciales de base general sobre un
  polinomio de primer orden — GRADUADA vía
  `apply_finite_exp_linear_combination_quotient_rule`: lee el numerador como
  Σ c_i a_i^(g_i) (+ ctes), acumula valor en 0 (=0 para 0/0 genuino) y
  derivada N'(0)=Σ c_i g_i'(0) ln(a_i) simbólica, y devuelve N'(0)/h'(0).
  `(2^x−3^x)/x=ln2−ln3`, `(2^(3x)−3^x)/x=3ln2−ln3`, `(e^x−2^x)/x=1−ln2`.
  Sound por construcción (L'Hôpital de primer orden, derivadas exactas);
  declina fuera de la clase. Peldaños restantes del item: (2) átomo con
  argumento NO-cero en el punto
  (`tan x/sin x` en π=−1, `sin x/(x−π)` en π=−1) — necesita el
  equivalente local en el cero del argumento; (3) log en el numerador/
  composición (`ln(1+x)/sin x`) — excluido por la ruta de base no-natural
  `valor/ln(base)`; (4) encadenamiento L'Hôpital general)*
  *(parcial 2026-06-14 08141ef4d: peldaño (2) átomo con argumento NO-cero
  GRADUADO + L'Hôpital general (4) para punto no-cero — el 0/0 transcendente
  cuya anulación ocurre en un punto desplazado ya resuelve vía
  `apply_finite_lhopital_nonzero_point_quotient_rule`: deriva numerador y
  denominador y reevalúa el límite del cociente, iterando mientras siga 0/0,
  reutilizando la cascada finita (que ya pliega sin(π)=0, cos(π)=−1 en el caso
  continuo). `sin(x)/(x−π)=−1`, `tan(x)/sin(x)=−1` en π, `cos(x)/(x−π/2)=−1`,
  `(1−cos(x−1))/(x−1)²=1/2` (2 aplicaciones), `(sin(x−1)−(x−1))/(x−1)³=−1/6`
  (3 aplicaciones), `ln(x)/(x−1)=1`. GATEADO a punto NO-cero (el 0 conserva sus
  dueños equivalent/Taylor y su narración de ángulo pequeño), último en la
  cascada (solo dispara sobre residual). SOUND por la hipótesis de L'Hôpital:
  emite SOLO valor finito racional definido; declina en polos (den'→0,
  num'≠0), formas unilaterales/cambio-de-signo, f'/g' oscilante (la trampa de
  falso-positivo) y valores irracionales (−sin(2), e²). Verificado
  adversarialmente: fuzz de 272 combinaciones 0 desacuerdos vs mpmath, trampas
  de polo-disfrazado/sign-flip/hipótesis-falla todas declinadas; punto 0 e
  infinito sin cambio. Aprendizaje: el diferenciador emite exponentes sin
  plegar `(x−1)^(2−1)`; hay que plegar todo subexpr constante con
  `as_rational_const` antes de retomar el límite. Quedan: encadenamiento
  L'Hôpital EN punto 0, peldaño (3) log/composición, y valores irracionales
  como salida simbólica.)*
- [x] **(A) Formas exponenciales 1^∞/0^0/∞^0** vía `exp(lim g·ln f)`:
  `(1+1/x)^x → e`, `(1+2/x)^x → e²`, `(1+x)^(1/x) → e`, `x^x → 1 en
  0+`, `(2^x+3^x)^(1/x) → 3`. Hoy la constante `e` es inalcanzable
  por límite — invalida un capítulo del temario.
  *(GRADUADO — capítulo completo: la constante `e` ya es alcanzable por
  límite y las tres formas indeterminadas exponenciales resuelven. Las tres
  reducen a `exp(lim exp·ln base)` con la maquinaria del sub-límite por
  forma.)*
  *(parcial 2026-06-14 0a2672c98: la forma ∞^0 graduada, CERRANDO el
  capítulo — dos fundamentos acoplados: `general_base_exponential_limit_at_
  infinity` (`b^x→∞/0/1` por análisis de signo; el motor crecía `e^x` pero
  dejaba `2^x` residual) e `inf_to_zero_power_limit_at_infinity` (base→+∞,
  exp→0 → `exp(lim exp·ln base)`, racionalizado+presimplificado para que la
  dominancia log-exp-suma vea el ln desnudo; `(2^x+3^x)^(1/x)=3`). Verificado
  adversarialmente (2-lente, 48 sondas, 0 unsound). Peldaños menores: ∞^0 con
  coeficiente en el exponente (`(2^x+3^x)^(2/x)=9` queda residual por el
  `c·ln`), bases e-mixtas, y las segundas-órdenes transcendentes del 1^∞.)*
  *(parcial 2026-06-14 a723ff67d: la forma 1^∞ EN INFINITO graduada — la
  constante `e` ya es alcanzable por límite. `one_to_infinity_power_limit_
  at_infinity` reduce 1^∞ a `exp(lim exp·(base−1))` usando `ln(1+h)~h`
  (válido porque la base→1 fuerza h→0), racionaliza el producto sobre
  denominador común (`rationalize_to_fraction`) y reutiliza el límite
  racional; pliega `e^0=1, e^1=e, e^(±∞)=∞/0`. Cubre `(1+a/x)^x=e^a`,
  `(1+1/x)^(kx)=e^k`, `((2x+1)/(2x-1))^x=e`, `(1+1/x²)^x=1`,
  `(1+1/x)^(x²)=∞`. Verificado adversarialmente (2-lente, 85 sondas, 0
  unsound): la trampa de SEGUNDO ORDEN `cos(1/x)^(x²)=e^(−1/2)` declina
  (base transcendente opaca al racionalizador) — nunca emite valor erróneo.)*
  *(parcial 2026-06-14 976efd869: la forma 1^∞ EN PUNTO FINITO graduada —
  la OTRA definición de e, `(1+x)^(1/x)=e`, resuelve, y las bases de SEGUNDO
  ORDEN también. `apply_finite_one_to_infinity_power_rule` gatea por el
  PRODUCTO (base→1 y `L=lim exp·(base−1)` no-nulo, que fuerza exponente
  divergente) en vez de por el exponente (1/x en 0 no tiene límite bilateral
  con signo). Como L lo evalúa la maquinaria finita COMPLETA (Taylor +
  infinitésimos equivalentes), es ESTRICTAMENTE más fuerte que la hermana en
  ∞: `cos(x)^(1/x²)=e^(−1/2)`, `(sin x/x)^(1/x²)=e^(−1/6)`,
  `(1+sin x)^(1/x)=e`. Sound: `lim g·ln(1+h)=lim(g·h)·lim(ln(1+h)/h)=L·1`,
  el término `−h²/2` se absorbe en el factor `ln(1+h)/h→1`. Verificado
  adversarialmente (2-lente, 47 sondas, 0 unsound, cross-check mpmath ~14
  cifras).)*
  *(parcial 2026-06-14 bbe89428f: la forma 0^0 graduada — `x^x → 1` en 0+.
  `apply_finite_zero_base_power_rule`: como x>0 a la DERECHA de 0, `x^g=
  exp(g ln x)` es real, y el límite es `exp(lim g ln x)`; `x^x=exp(lim x ln
  x)=exp(0)=1`. Gateado al lado derecho con base = la variable desnuda (signo
  positivo conocido en la aproximación); el bilateral `x^x` queda residual
  (complejo para x<0) y una base no-variable (`sin(x)^x`) declina (signo no
  probado). Peldaño restante: ∞^0 con base exponencial dominante
  (`(2^x+3^x)^(1/x)=3` necesita `ln(2^x+3^x)/x → ln 3`).)*
- [x] **(F) ∞−∞ con radicales** (racionalización por conjugado):
  `sqrt(x^2+x)-x → 1/2`, `sqrt(x+1)-sqrt(x) → 0`, y en punto finito
  `(sqrt(x)-2)/(x-4) → 1/4`.
  *(parcial 2026-06-13 d78ce2c0e: `sqrt(ax²+bx+c) − (dx+e)` a ±∞ con
  √a racional y cancelación de términos líderes ya resuelve vía forma
  cerrada `b/(2√a)−e` — `sqrt(x²+x)−x=1/2`, `sqrt(x²+1)−x=0`,
  `sqrt(4x²+x)−2x=1/4`, `x−sqrt(x²−x)=1/2`. Gate de cancelación exacta
  (los divergentes declinan).)*
  *(parcial 2026-06-13 d7dd00024: sqrt−sqrt completado para radicandos
  del mismo grado (1 o 2) y mismo líder — `sqrt(x+1)−sqrt(x)=0`,
  `sqrt(x²+x)−sqrt(x²−x)=1`, `sqrt(4x²+x)−sqrt(4x²−x)=1/2` vía
  `(b_P−b_Q)/(2√a)`. El lado +∞ del item queda cubierto. Peldaños:
  √a irracional (`sqrt(2x²+x)−sqrt(2x²−x)=1/√2`), grado ≥3)*
  *(graduado 2026-06-13 15bc39585: el lado PUNTO FINITO completado —
  `(scale·√(ax+b)+k)/den` en 0/0 vía conjugado: `(√x−2)/(x−4)=1/4`,
  `(√x−3)/(x−9)=1/6`, `(√(2x+1)−3)/(x−4)=1/3`, denominador cuadrático
  `(√x−2)/(x²−16)=1/32`. Gate de seguridad: numerador 0 en el punto +
  raíz racional + conjugado ≠0; los polos no-0/0 y las raíces
  irracionales declinan, con condiciones de dominio honestas. Item
  cerrado salvo los peldaños √a irracional y grado ≥3 anotados)*
  *(sqrt−sqrt punto finito 2026-06-15 39e11685f: el complemento de DOS
  radicales `(s1√(L1)+s2√(L2))/den` en 0/0 vía conjugado — `apply_finite_radical_
  difference_conjugate_rule`, hermano del sqrt−constante de arriba. El conjugado
  `s1√(L1)−s2√(L2)` cancela AMBOS radicales en el polinomio `s1²L1−s2²L2`, así que
  el límite es `[ese polinomio sobre den, removible] / (s1√(L1(pt))−s2√(L2(pt)))`.
  `(√(1+x)−√(1−x))/x=1`, `(√(4+x)−√(4−x))/x=1/2`, `(√(x+3)−√(2x+2))/(x−1)=−1/4`
  (punto no-cero), `(√(1−x)−√(1+x))/x=−1` (signo invertido). Gate idéntico:
  0/0 genuino + radicandos lineales + raíces racionales en el punto + conjugado
  ≠0; polos, SUMA de raíces, radicandos no lineales e irracionales declinan.
  Verificado numéricamente (mpmath dps 40). Quedan: √a irracional y radicandos
  grado ≥2.)*
  *(hermano RACIONAL 2026-06-14 1881980a6: el ∞−∞ de funciones racionales
  (sin radicales) también resuelve — `rational_difference_limit_at_infinity`
  pone los operandos sobre denominador común y reutiliza
  `rational_poly_limit`: `(x²+1)/(x+1)−x=−1`, `x²/(x−1)−x=1`,
  `x²/(x+1)−x²/(x+2)=1`, `x³/(x+1)−x=+∞`. Corre al final de la cadena (las
  diferencias con límites finitos conservan su traza aditiva; operandos no
  racionales declinan al conjugado/dominancia).)*
  *(compañera 0·∞ 2026-06-14 b91912327: el PRODUCTO `factor·(diferencia
  conjugada→0)` —la forma 0·∞ que el bare-difference dejaba al multiplicativo,
  que declinaba— ya resuelve a +∞ vía `radical_conjugate_product_limit_at_
  infinity`: racionaliza la diferencia (numerador conjugado `s²Q−L²` sobre la
  suma conjugada `~2s√a·x`) para leer su decaimiento como término líder
  `K·xᵖ`, lee el factor (polinomio o `escala·√(poli)` con líder racional) como
  `c·xᵠ`, y devuelve el límite por la suma de exponentes: `c·K` si `p+q=0`, `0`
  si `<0`, declina (deja `+∞` a dominancia) si `>0`. Términos aditivos
  aplanados y partidos en √ vs resto polinómico, así que cola lineal partida
  (`x·(√(x²+2x)−x−1)=−1/2`), ambas orientaciones, y `√−√` cuadrático
  (`x·(√(x²+x+1)−√(x²+x))=1/2`) caen igual. `x·(√(x²+1)−x)=1/2`,
  `x·(√(x²+4)−x)=2`, `√x·(√(x+1)−√x)=1/2`. Gate de cancelación líder
  (`s√a+r1=0`); SOLO a +∞ (el lado −∞ es trampa: misma forma diverge ahí, se
  deja residual honesto); declina líder irracional (`√(2x)·…`) y factor que
  supera el decaimiento (`x²·(√(x²+1)−x)` diverge). Peldaños: análogos cbrt
  (`x²·(∛(x³+1)−x)=1/3`), lado −∞ con valor finito, coeficientes irracionales.
  Verificado adversarialmente con 37 sondas mpmath dps=60.)*
  *(análogo CBRT 2026-06-14 6e3257810: el peldaño cbrt cierra — la diferencia
  conjugada cúbica `∛(P)−x` y su producto 0·∞ resuelven vía
  `cbrt_conjugate_limit_at_infinity`. Donde √ racionaliza por `a+b` (`~2dx`),
  la cúbica usa `a³−b³=(a−b)(a²+ab+b²)`, suma conjugada de TRES términos
  `~3d²x²`; lee `N=s³P−L³` (grado ≤2 tras cancelar x³) sobre `3d²x²`: término
  x² → constante, x → K/x, constante → K/x². `∛(x³+x²)−x=1/3`,
  `∛(x³+2x²)−x=2/3`, `∛(x³−3x²)−x=−1`, `∛(8x³+x²)−2x=1/12`,
  `x²·(∛(x³+1)−x)=1/3`, `x·(∛(x³+3x)−x)=1`; forma Pow `(x³+x²)^(1/3)−x=1/3`
  también. SOLO +∞ (−∞ es trampa: el producto puede invertir signo); líder
  cbrt racional (`∛(2x³…)` declina, irracional). Adversarial RETAIN (37 sondas
  mpmath dps=60, incluidas las trampas −∞ e irracionales, sin desacuerdos).
  Peldaño general n-ésima raíz: el conjugado es la suma de n términos
  `~n·d^(n-1)·x^(n-1)`, parametrización de los casos √ (n=2) y cbrt (n=3).)*
  *(GENERAL n-ésima raíz 2026-06-14 31201e550: el peldaño cierra —
  `nth_root_conjugate_limit_at_infinity` resuelve `(P)^(1/n)−L` y su producto
  0·∞ para cualquier `n≥2` en forma Pow. Lee `N=s^n·P−L^n` por el binomio
  (`N_k=s^n·P_k−C(n,k)·d^k·e^(n-k)`) sobre `n·d^(n-1)·x^(n-1)`, con
  `rational_nth_root` nuevo (generaliza rational_sqrt/cbrt vía BigInt::nth_root
  con re-chequeo root^n==value y rechazo de raíz par de negativo). Corre tras
  √/cbrt, así que sólo añade n≥4. `(x^4+x^3)^(1/4)−x=1/4`,
  `(x^5+x^4)^(1/5)−x=1/5`, `(16x^4+x^3)^(1/4)−2x=1/32`,
  `((1/16)x^4+x^3)^(1/4)−x/2=2`, `x^3·((x^4+1)^(1/4)−x)=1/4`. Adversarial RETAIN
  (33 sondas mpmath dps 60-200; trampas líder irracional, líder negativo
  (sin raíz par real), −∞, grado≠n, sobre-potencia, todas residuales). SOLO +∞;
  líder racional. Peldaño restante: lado −∞ con n impar (converge a otro valor).)*

### P2 — familias y mejoras de alto valor (1 ciclo cada una)

- [x] **(F) Touch con límite x^a·ln(x)^b → 0**: `ln(x)^2 [0,1]` (=2),
  `x*ln(x) [0,1]` (=−1/4), `ln(x)/sqrt(x) [0,1]` (=−4) residuales con
  antiderivadas elementales; la dominancia potencia-log existe en el
  lado lateral pero no cubre estas combinaciones. Arregla 3+ familias.
  *(graduado 2026-06-13 52f0fb4f9: el hueco estaba en el MOTOR DE
  LÍMITES, no en el integrador — las antiderivadas ya se conocen y el
  borde definido ya las evalúa por límite lateral de F. La dominancia
  `power_log_dominance_zero_limit` resolvía solo el monomio `u^p·ln(u)^q`;
  `apply_finite_one_sided_power_log_polynomial_zero` la generaliza a
  `Σ c·(var-pt)^a·P(ln(var-pt))` con todos a>0 → 0 (potencia × polinomio
  en ln, sumados). Resuelve `∫₀¹ ln²=2`, `x·ln=−1/4`, `ln/√x=−4`,
  `x²·ln=−1/9`, `√x·ln=−4/9`. Gate de soundness: potencia neta
  estrictamente positiva + al menos una potencia presente (un término
  constante o de log-puro bloquea: `x ln x + 5 → 5`), y ln gateado a
  potencias ENTERAS no-negativas (ln<0 cerca de 0). Verificado
  adversarialmente (2 lentes, 115 sondas, 0 violaciones; el trap
  `x·e^(1/x)` queda residual = +∞). Peldaños: touches con exp/trig en el
  borde, y la forma exponencial `1^∞/0^0` que necesita el interno `∞·0`
  robusto)*
- [x] **(F) Gaussiana/Gamma por tabla**: `e^(-x^2) [0,∞) = √π/2`,
  `(-∞,∞) = √π`, `x^2*e^(-x^2) [0,∞) = √π/4`, `e^(-x)/sqrt(x) = √π` —
  la impropia más famosa de la universidad; tabla pequeña de formas
  patrón (la indefinida debe SEGUIR residual).
  *(graduado 2026-06-13 cda9fbca5: la familia Gaussiana de momentos
  `∫ x^(2n) e^(-a x²)` sobre semirrecta o recta completa vía
  `(1/2)(2n)!/(4^n n!)√π/a^(n+1/2)` — `e^(-x²)[0,∞)=√π/2`,
  `[-∞,∞]=√π`, `x²e^(-x²)=√π/4`, `x⁴e^(-x²)=3√π/8`,
  `e^(-2x²)=½√(π/2)`. Gating fuerte: bounds infinitos, exponente puro
  cuadrático, cofactor par, a>0 — indefinida/bounds finitos/no-cuadrático
  declinan (honestidad intacta, verificada adversarialmente — cazó y
  arregló un bug de coeficiente perdido). Peldaños: las formas Gamma
  (`e^(-x)/√x`, `x^n e^(-x)=n!`), el cofactor con cuadrado completado
  `e^(-x²+x)`, y `c·e^(-x²)` en forma Mul anidada)*
  *(Gamma graduada 2026-06-13 5ee37ea63: la familia Gamma de
  medio-entero `∫₀^∞ x^(m-1/2) e^(-ax) = (2m)!/(4^m m!)/a^m √(π/a)` —
  `e^(-x)/√x=√π`, `√x·e^(-x)=√π/2`, `x^(3/2)e^(-x)=3√π/4`,
  `e^(-2x)/√x=√(π/2)`. El entero `x^n e^(-x)=n!` ya resolvía vía
  antiderivada elemental. `match_gamma_integrand` ACUMULA sobre Mul/Div/Neg
  (potencia neta + decay lineal + constante) — un walker para todas las
  formas. Gating: exponente decay lineal puro, potencia medio-entera
  (los enteros caen a la antiderivada), s≥−1/2 (divergentes residuales),
  a>0, solo [0,∞). Verificado adversarialmente (58 sondas, valores
  exactos sin coefficient/sign drop; cazó un gap de coeficiente −1 unitario
  → arreglado con el brazo Neg). Peldaños restantes del item Gaussiano:
  cuadrado completado `e^(-x²+x)` y `c·e^(-x²)` Mul anidada)*
- [ ] **(A) Pre-simplificador vs integrador**: reescribe
  `1/(sqrt(x)*(1+x))` a `(x^(3/2)-x^(1/2))/(x^3-x)` y `cos(5x)` a
  Chebyshev en cos(x), destruyendo la sustitución obvia
  (`[0,∞) = π`) y la ortogonalidad de Fourier `sin(3x)cos(5x)
  [-π,π] = 0`. Integrar sobre la forma original primero, o enseñar al
  integrador las formas reescritas (precedente: reconocedor Chebyshev
  del ledger 2026-06-12). Ejemplo vivo adicional:
  `integrate(sec(x)^2 + 1, x)` se vuelve residual porque el
  pre-simplificador lo machaca a `(2cos²−1+3)/(2cos²)`.
- [~] **(F) Detección estructural sin antiderivada**: imparidad en
  `[-a,a]` para integrandos no elementales (`sin(x)/(1+x^2) [-1,1] =
  0`), abs por tramos (`|x| [-1,1] = 1`, `e^(-|x|) (-∞,∞) = 2`),
  test-p completo (`1/sqrt(x) [1,∞) = ∞`, hoy residual mientras `1/x`
  sí diverge), divergencia oscilatoria declarada (`sin(x) [0,∞)`).
  *(parcial 2026-06-14 b5f80b09f: IMPARIDAD en `[-a,a]` GRADUADA —
  `odd_symmetric_definite_integral_rewrite` corre como fallback estructural
  donde la antiderivada es None y resuelve a 0 bajo tres obligaciones
  independientes: bornes finitos simétricos (lower=-upper sobre el endpoint
  racional+pi+e), imparidad probada por `parity_in_var` (clasificador sound
  y conservador {Odd, Even, Unknown}: símbolo ajeno=par, suma conserva
  paridad solo si ambos términos coinciden, producto/cociente suma paridades,
  potencia entera por paridad del exponente, base constante positiva b^g par
  sii g par, composición por la clase del builtin externo) e INTEGRABILIDAD
  vía el MISMO certificado que hace `int(1/x,-1,1)` undefined (Certified
  estricto). Resuelve `sin(x)/(1+x^2)`, `sin(x)e^(x^2)`, `sin(x^3)`,
  `tan(x)e^(x^2)` en `[-1,1]`; declina con corrección `1/x` (undefined),
  `tan e^(x^2) [-2,2]` (polo en π/2), integrandos pares e intervalos
  asimétricos. Verificado adversarialmente (2-lente, 80 sondas, 0 unsound).
  Peldaños restantes: abs por tramos (`|x| [-1,1]=1`, `e^(-|x|)`), test-p
  completo, divergencia oscilatoria declarada, y la narración educativa
  específica de simetría — hoy el paso es el envoltorio genérico "Calcular
  la integral".)*
  *(abs lineal por tramos 2026-06-15 1afca174c: `∫_a^b |c x + d|` sobre bornes
  RACIONALES ya resuelve — `abs_linear_definite_integral_rewrite` parte en la raíz
  `r=−d/c`: la antiderivada de `c x+d` es `G(x)=c x²/2+d x` y, como el interior
  tiene signo constante a cada lado de r, la integral es `|G(r)−G(lo)|+|G(hi)−G(r)|`
  si r∈(lo,hi), si no `|G(hi)−G(lo)|`. Aritmética BigRational exacta, antes del
  intento FTC (|lineal| no tiene antiderivada que el FTC halle); abs es continuo,
  sin certificado de polo. `∫|x|[-1,1]=1`, `∫|x-1|[0,2]=1`, `∫|2x-1|[0,1]=1/2`,
  `∫|x|[-2,3]=13/2`, raíz fuera `∫|x-1|[2,5]=15/2`. Gateado a |lineal| desnudo con
  bornes racionales: el producto `x·|x|` (no es Function(Abs)) lo conserva la
  simetría impar (→0); inner cuadrático `|x²-1|`, borne π/e e indefinida declinan.
  Quedan: inner cuadrático/polinómico (raíces múltiples), `e^(-|x|)` impropia, y la
  narración de simetría.)*
  *(narración abs-lineal 2026-06-15 4509c4651: la NARRACIÓN educativa de `∫|c x+d|`
  ya aterriza — `generate_abs_linear_definite_integral_substeps` en el didactic narra
  "Localizar la raíz" → partir en la raíz (raíz dentro) / signo constante (raíz fuera) →
  "Integrar por tramos con G(x)=c x²/2+d x". Antes el paso era el envoltorio genérico
  "Calcular la integral" con substeps vacíos (el narrador FTC reintegra y `∫|lineal|` no
  tiene antiderivada única). Honesto: narra desde los MISMOS hechos estructurales (raíz,
  signo por tramo) que usó el rewrite del ciclo 5, no de una re-derivación. Locked por test
  de contrato cas_cli; las filas abs de la matriz pasan sin cambio (matcher por substring).
  Queda la narración de SIMETRÍA IMPAR: `parity_in_var` es privada de cas_engine y el
  didactic no la alcanza — necesita subir un clasificador de paridad a cas_math primero.)*
  *(límite prerequisito del test-p 2026-06-15 e8b1e5d27: `lim_{x→+∞} x^q` con q
  racional NO entero ya resuelve — `apply_rational_power_rule` extrae el exponente con
  `as_rational_const`, declina enteros (los conserva `apply_power_rule`) y, como la base
  x→+∞ es positiva, devuelve +∞ si q>0 y 0 si q<0; x→−∞ DECLINA (x^q no es real para x<0
  con q no entero) igual que los exponentes simbólicos/irracionales. `x^(1/2)→∞`,
  `x^(-1/2)→0`, `x^(2/3)→∞`. Esto desbloquea el LÍMITE que el test-p necesita, pero el
  despachador de impropias aún gatea los integrandos de potencia fraccionaria ANTES del
  paso límite-de-la-antiderivada, así que `∫1/x^(3/2)[1,∞)` sigue residual — ese gate es
  el siguiente peldaño del test-p. Verificado adversarialmente (2-lente, ~40 sondas +
  cross-check numérico a x=1e18, 0 unsound).)*
- [~] **(F) Por partes narrada**: la plantilla completa ('Elegir u y
  dv' → 'Calcular du y v' → 'Aplicar la fórmula') existe y la usa
  `x·ln(x)`, pero `x·eˣ`, `x·cos x`, `arctan(x)`, `x²eˣ`, `eˣ·sin x`
  solo dicen "Usar integración por partes" (o nada: `ln(x)` da cero
  substeps). Es LA técnica central del curso y es cableado.
  *(parcial 2026-06-14 b08e182b0: la familia `polinomio(lineal)·{eˣ,
  sin,cos,sinh}` ya narra u/dv/du/v — `generate_polynomial_elementary_
  by_parts_substeps` clona el narrador de ln con la asignación inversa
  (u=polinomio, dv=factor elemental), computa `v` con
  `integrate_symbolic_expr` y `du` con `differentiate_symbolic_expr`:
  `x·cos x` → u=x, dv=cos(x)dx, du=1dx, v=sin(x), `x·sin(x)−∫sin(x)dx`;
  argumento afín (`x·cos(2x+1)`, v=½sin(2x+1)) y coeficiente no-unidad
  (`(2x+3)·eˣ`, du=2) incluidos; resultado byte-idéntico (sólo
  presentación). Desambiguación de factores por eliminación (u=el
  polinomio de grado 1; dv=el otro si no es polinomio ni ln) — la familia
  ln conserva su narración (u=ln) sin duplicar. Quedan: el caso repetido
  grado≥2 (`x²eˣ`, queda sólo-título "repetida"), inverse-trig
  (`arctan(x)`, `arcsin(x)` con u=función-inversa) y el cíclico `eˣ·sin x`
  (sin factor polinómico).)*
  *(parcial 2026-06-14 acb6ba04a: la familia inverse-trig / inverse-hiperbólica
  desnuda ya narra — `generate_single_inverse_by_parts_substeps` es el tercer
  hermano del dispatcher (u=f, dv=dx, v=x, du=f'): `∫arctan(x)` → u=arctan(x),
  dv=dx, du=1/(x²+1)dx, v=x; `arcsin/arccos/asinh/acosh/atanh` y argumento afín
  `arctan(2x+1)` incluidos. El título ya disparaba (vía
  is_bounded_inverse_trig_variable_target), narración pura sin tocar el gate.
  Quedan: `x²eˣ` repetida (grado≥2), el cíclico `eˣ·sin x`, y `ln(x)` SOLO —
  que hoy da CERO substeps porque ningún target by-partes lo reclama: no hay
  título, requiere cambio de gate del dispatcher, ciclo aparte.)*
  *(parcial 2026-06-14 4c0c0a24d: `ln(x)` SOLO ya narra — el gap era el GATE,
  no el narrador: `contains_linear_integration_by_parts_target` ahora reclama
  `Function(Ln,[afín])` (dispara el título) y `Ln` se une al narrador
  single-inverse (u=ln, dv=dx, v=x, du=1/x). `∫ln(x)` → u=ln(x), dv=dx, du=1/x
  dx, v=x; `∫ln(2x+1)` (afín, du=2/(2x+1)) incluido. Cambio de gate de bajo
  footprint (la única fixture de ln(x) es una fila FTC-definida con matcher
  subset). Sin doble-narración: `x·ln(x)` es Mul, conserva su narración
  cycle-2 (u=ln, dv=x dx). El motor narra ya las TRES asignaciones u/dv:
  u=ln sobre dv polinómico, u=polinomio sobre dv elemental, u=función sobre
  dv=dx (inversas + ln). Quedan: `x²eˣ` repetida y el cíclico `eˣ·sin x`.)*
  *(parcial 2026-06-14 87deecf8f: la repetida grado≥2 `p(x)·{eˣ,sin,cos,
  sinh,cosh}(afín)` ya DESENROLLA cada aplicación — `generate_repeated_
  polynomial_elementary_by_parts_substeps` es el cuarto hermano del dispatcher.
  El motor integra estos por el método tabular cerrado (derivadas iteradas con
  signo alterno = exactamente N aplicaciones de partes), así que el título
  "repetida" ya era honesto; el narrador recomputa u/dv/du/v por nivel
  (`integrate_symbolic_expr` para v, `differentiate_symbolic_expr` para du)
  bajando el grado del polinomio en 1 cada nivel hasta constante, donde el
  término restante elemental cierra en la antiderivada final. `x²eˣ` → nivel 1
  u=x², dv=eˣdx, du=2x, v=eˣ ⇒ x²eˣ−∫eˣ·2x; nivel 2 u=2x, du=2 ⇒ 2x·eˣ−∫eˣ·2;
  "Integrar el término restante" ⇒ eˣ(x²−2x+2). Cubre x²cos, x³sin(2x+1),
  x²sinh, x³eˣ (hasta grado 8). El grado=1 conserva su narrador lineal (una sola
  aplicación), ln su narrador propio, y no-targets (cos·eˣ) no disparan.
  Presentación pura: resultado byte-idéntico, huella guardrail+pressure sin
  deltas. Queda SÓLO el cíclico `eˣ·sin x` — su ruta interna NO es partes
  (distribuye/expande), narrarla como partes mentiría: ciclo aparte que
  primero necesitaría una ruta interna cíclica honesta.)*
- [ ] **(F) Residuales con motivo**: 'Conservar integral residual' no
  distingue "no elemental (necesitaría erf/Si/Ei)" de "el motor aún
  no lo soporta" (`sqrt(1-x²)`, `|x|`, `1/(x⁴+1)` son elementales y
  quedan igual que `e^(-x²)`). Campo de motivo como el warning de
  límites.
- [~] **(F) Cuárticas+ irreducibles**: `1/(x^4+1)`, `1/(x^6-1)` —
  factorización real en cuadráticas con coeficientes irracionales
  (√2) para fracciones parciales.
  *(precursor 2026-06-13 3a267bdf2: el kernel `1/(cuadrática con
  raíces irracionales)` ya integra — `1/(x²−2)`, `(x+b)²−a` vía forma
  log con √c simbólico, desbloqueando `1/(u²−2)`.)*
  *(parcial 2026-06-13 962a01ddb: `(ax²+b)/(x⁴+1)` graduado vía
  sustitución simétrica u=x∓1/x — `1/(x⁴+1)`, `x²/(x⁴+1)`,
  `(x²±1)/(x⁴+1)`, con condición honesta x≠0 (la sustitución salta en
  0). Peldaños restantes: generalizar a `(ax²+b)/(x⁴+e)` con e no
  cuadrado perfecto (radicales anidados); la forma continua para
  integrales definidas que crucen 0; `1/(x⁶−1)` levantar el cap de
  factores del backend racional; `1/(x⁵−1)` las cuárticas ciclotómicas
  Φ5 con √5)*
  *(precursor `factor` 2026-06-21 f2313025cb7f61e3c5e58603303943a764475835: `factor` ahora parte
  las cuárticas pares REDUCIBLES `a·x⁴+b·x²+c` en dos cuadráticas sobre ℚ
  — biquadrática `(x²+r)(x²+s)` y Sophie-Germain `(x²+ex+f)(x²-ex+f)`:
  `factor(x⁴+x²+1)→(x²+x+1)(x²-x+1)`, `factor(4x⁴+1)→(2x²+2x+1)(2x²-2x+1)`,
  `factor(x⁶-1)→(x-1)(x+1)(x²+x+1)(x²-x+1)`; las irreducibles `x⁴+1`,
  `x⁴-x²+1`, Φ5 quedan enteras.)*
  *(`1/(x⁶−1)` GRADUADO dd2ad48bf 2026-06-21: sube el budget del multipoly del
  `algebraic_rational_zero_test` (max_terms 64→256, grado 16→32) — el verifier YA hacía √c↦t,
  t²=c, solo no cabía el residual de grado 6. `1/(x⁶−1)`, `1/(x⁶−64)` integran y round-trip-an.
  Diagnóstico corregido: NO faltaba un rationalizador de radicales, faltaba budget. Quedan
  residuales honestos los que requieren factor-over-ℝ: `1/(x⁴+1)` ya cubierto por la sustitución
  simétrica, pero `1/(x⁸−1)`, `1/(x⁶+1)`, `1/(x⁵−1)` (Φ5/√5), `1/(x⁴−4)` (√2) necesitan LRT.)*
  *(extensión `factor` even-poly 2026-06-21 bfd669727b629bdbba8340e380e493cc773cd3ae: `factor` parte
  también polinomios PARES REDUCIBLES de grado ≥6 vía t=x²:
  `factor(x⁶+1)→(x²+1)(x⁴-x²+1)`, `factor(x⁶+x⁴+x²+1)→(x²+1)(x⁴+1)`,
  `factor(x⁸-1)→(x-1)(x+1)(x²+1)(x⁴+1)`, `factor(x⁸+x⁴+1)→(x²+x+1)(x²-x+1)(x⁴-x²+1)`;
  irreducibles `x⁶+x³+1` (Φ9) enteras. Verificado exacto (expand∘factor=id).)*
  *(re-diagnóstico EXACTO del hueco de INTEGRACIÓN — corrige la nota previa:
  NO es factorización NI una antiderivada incorrecta. El backend racional
  `general_rational_partial_fraction_antiderivative` produce la antiderivada
  CORRECTA de `1/(x⁶-1)` (coincide byte-a-byte con sympy: 4 logs + 2 arctan).
  El `verify=Failed` es un FALSO NEGATIVO de la verificación: `differentiate_symbolic_expr`
  deja `sqrt(3)·sqrt(3)` y `(a/sqrt(3))²` SIN reducir en las derivadas de arctan,
  y la normalización del verificador no combina los 6 términos racionales para el caso
  más rico (2 lineales + 2 cuadráticas, grado 6). TODO lo de grado ≤5 verifica
  (`1/(x³-1)`, `1/(x⁴-1)`, `1/((x-1)(x²+x+1)(x²-x+1))`) y `1/(x⁴+x²+1)` sola (2 cuadráticas)
  también. Peldaño: reducir `sqrt(k)·sqrt(k)→k` en el verificador (o verificar la
  DESCOMPOSICIÓN en vez de derivar el arctan) — toca el verificador del bloque-12
  (huella-sensible, exige verificación adversarial), no es un ciclo rápido.)*
- [x] **(F) Composición de límites con interno conocido**:
  `e^(1/x) en 0±` (→ ∞ / 0), `atan(1/x) en 0+` (→ π/2) fallan aunque
  `1/x → ±∞` resuelve; regla de composición continua/monótona barata
  (la tabla saturante en ∞ ya existe — reutilizarla desde laterales).
  *(parcial 2026-06-13 457b8d5d8: el lado BILATERAL con interno → ∞ con
  signo definido ya resuelve — `e^(-1/x²)→0`, `e^(1/x²)→∞`,
  `atan(1/x²)→π/2`, `tanh(1/x²)→1` — vía fold de saturación f(±∞) sobre
  la salida del límite (arctan/tanh/exp/ln/sqrt/sinh/cosh; excluye
  sin/cos/tan oscilantes). Verificado adversarialmente: el caso
  bilateral con laterales DISTINTOS (`e^(1/x)` en 0) queda correctamente
  residual. Nota de soundness: el fold NO se
  registró como regla global (ensanchaba el bug preexistente ∞−∞=0 en
  aritmética cruda); queda confinado a salidas de límite vetadas)*
  *(graduado 2026-06-13 393388fbb: cubierto el peldaño UNILATERAL —
  `apply_finite_one_sided_composition_rule` ganó ramas `Pow(E,g)` y
  `Function(f,[g])` que resuelven el límite interno lateral, leen su
  signo de ∞ y pliegan f(±∞) con el MISMO `fold_infinity_saturation`. La
  puerta de oscilación es el propio fold (`folded != candidate`), sin
  lista explícita. Honestidad estructural: el bilateral solo resuelve vía
  `matching_finite_bilateral_one_sided_result` (ambos lados deben
  coincidir), así que `e^(1/x)`/`atan(1/x)`/`tanh(1/x)` en 0 siguen
  residuales. Verificación adversarial 60+ sondas: cero violaciones de
  soundness, oscilantes declinan, bilaterales de poste par del ciclo 7
  sin regresión. Bonus: el fold `0·finito→0` en `combine_limit_product`
  normaliza productos cuyos dos factores ahora resuelven. Peldaños
  abiertos fuera de alcance: `cosh(1/x)` bilateral (ambos lados → +∞
  pero declina), `asinh/acosh/coth/sech` (folds no implementados),
  coeficientes irracionales `e^(π/x)`, y `∞·(π/2)` lateral que necesita
  cofactor racional)*
  *(peldaño cosh cerrado 2026-06-13 b349e056e:
  `apply_finite_bilateral_even_saturating_pole_rule` resuelve
  `cosh(1/x)→∞` bilateral — cosh es PAR, así que cosh(±∞)=+∞ vale en
  ambos lados pese a que el polo impar 1/x diverge con signos opuestos.
  Gateado a Cosh + inner divergente en ambos lados (reutiliza
  `one_sided_inner_infinity_sign` y `saturate_outer_at_infinity`). Nota
  de alcance: se prototipó la regla GENERAL "bilateral = valor lateral
  común" (sólida por el teorema, resolvía `1/|x|`, `log_b(|x|)`,
  `sqrt|x|`, `exp(ln|x|)`) pero flipeaba ~6 contratos conservadores de
  "composición finita no soportada con seguridad" de golpe; se redujo a
  cosh (par, fold independiente del signo), que NO toca ningún contrato.
  Verificado adversarialmente (52 sondas, 0 violaciones; semi-definidos
  `cosh(1/√x)`/`cosh(ln x)` declinan por lado indefinido). Peldaño
  preexistente anotado: `cos(1/x²)`/`sin(1/x²)` filtran `cos(infinity)`
  sin plegar — el combine bilateral de inner oscilante de potencia par)*
  *(fuga cos/sin cerrada 2026-06-13 055929883:
  `apply_finite_total_real_unary_composition_rule` declina cuando el outer
  es Sin/Cos y el argumento → ±∞ — sin/cos oscilan en ∞, sin límite. Era
  honestidad: el outer saturante (atan/exp/tanh/cosh) filtra
  `outer(infinity)` que la capa eval pliega, pero sin/cos no pliegan y
  filtraban `cos(infinity)` como pseudo-valor en la familia `sin(1/x)` que
  nunca debe resolver. Sustractivo (solo añade decline); el polo impar
  `cos(1/x)` ya estaba bien por el gate "fold changed it" unilateral.
  Verificado adversarialmente (41 sondas, 0 violaciones; saturantes y
  squeeze intactos))*
- [x] **(F) Squeeze y dominancia fraccionaria**: `x*sin(1/x) → 0 en
  0`, `(x+sin x)/x → 1 en ∞`, `ln(x)/sqrt(x) → 0 en ∞` (la dominancia
  entera `ln(x)/x` sí funciona).
  *(parcial 2026-06-13 74544e793: cubierto el SQUEEZE en punto finito —
  `apply_finite_squeeze_bounded_product_rule` resuelve a 0 todo producto
  con un factor infinitésimo y un factor oscilante globalmente acotado
  sin límite (`sin/cos/atan/arctan/tanh` de una función racional de la
  variable): `x·sin(1/x)`, `x²·cos(1/x)`, `sin(x)·sin(1/x)`,
  `(x-2)·sin(1/(x-2))` en 2 → 0. Footprint-mínimo: solo dispara cuando
  hay un factor acotado SIN límite, así que `x·sin(x)` sigue por la ruta
  genérica. Honestidad triple-gateada: `sin(1/x)` solo y `2·sin(1/x)`
  (sin infinitésimo) quedan residuales, `(1/x)·sin(1/x)` normaliza a Div.
  Verificado adversarialmente (2 pasadas, ~230 sondas): cazado y
  corregido un bug de soundness — denominador idénticamente cero
  `1/(x-x)` daba `sin(1/0)` indefinido como "acotado"; el gate ahora
  exige denominador no-cero. Peldaños restantes: el cociente con ruido
  aditivo acotado `(x+sin x)/x → 1 en ∞` (la maquinaria
  `polynomial_growth_info_with_bounded_additive_noise` existe pero no
  está cableada al cociente racional general) y la dominancia
  log-potencia FRACCIONARIA `ln(x)/√x → 0 en ∞` (`ln(x)/x` y `ln(x)/x²`
  enteras ya funcionan; falta extender a `x^(p/q)`). Gaps cosméticos de
  completitud del squeeze: argumentos sin normalizar `x^(-1)`, `1/x+x²`
  declinan conservadoramente aunque el límite real es 0)*
  *(dominancia fraccionaria graduada 2026-06-13 da56c3a08:
  `polylog_power_dominance_limit_at_infinity` resuelve `c·ln(x)^a/x^b→0`
  y `c·x^b/(c'·ln(x)^a)→sign(c/c')·∞` con a≥1 entero y b>0 racional —
  `ln(x)/√x=0`, `ln(x)²/x=0`, `ln(x)³/x=0`, `ln(x)/x^(1/3)=0`,
  `√x/ln(x)=∞`, `x/ln(x)²=∞`. `positive_power_tail` reconoce el exponente
  RACIONAL (x^(1/2), x^(2/3) de primera clase). Gating: solo +∞ (ln
  indefinido en −∞), coeficientes no-cero, potencia genuinamente positiva
  (`ln(x)/x^(-2)=ln·x²→∞`, no 0). Verificado adversarialmente (70 sondas,
  0 violaciones; cerrado un brazo Neg faltante). Peldaño restante: el
  cociente con ruido aditivo acotado `(x+sin x)/x → 1`)*
  *(ruido aditivo graduado 2026-06-13 ac4dd379f — ITEM CERRADO:
  `bounded_noise_rational_limit_at_infinity` resuelve cocientes
  `poly+ruido_acotado / poly+ruido_acotado` por la parte polinómica —
  `(x+sin x)/x=1`, `(2x+cos x)/x=2`, `(x²+sin x)/(x²-1)=1`,
  `(x+sin x)/(2x+1)=1/2`, `x/(x+sin x)=1`, `(x+cos x)/x²=0`,
  `(x²+sin x)/x=∞`. Cablea `polynomial_growth_info_with_bounded_additive_
  noise` al cociente racional con la misma comparación de grados. El ruido
  NO acotado (`x·sin x`) declina. Verificado adversarialmente (54 sondas,
  0 violaciones))*
- [x] **(F) Producto-a-suma residual mutilado**: `sin(3x)cos(5x)`
  indefinida queda residual Y mutilada (expandida en potencias de
  cos); el reconocedor producto-a-suma cubre frecuencias distintas
  con a≠±b — revisar por qué esta combinación escapa.
  *(graduado 2026-06-13 2cd81323b: la mutilación era una regla — la
  "Quintuple Angle Identity" expandía cos(5x)/sin(5x) antes del
  producto-a-suma y NO estaba en la lista de desactivadas de
  IntegratePrep. Añadida a la lista (scoping de 3 agentes: footprint
  bajo, además ARREGLA `integrate(cos(5x))→sin(5x)/5` y
  `integrate(cos(5x)²)` que era residual). Y un sibling Werner
  sin-coeficiente con `/2` explícito cubre sin·cos/cos·cos/sin·sin
  con gate A≠B: `sin(3x)cos(5x)=1/16(4cos2x−cos8x)`,
  `cos(3x)cos(5x)`, `sin(3x)sin(5x)`. Resuelve parcialmente el item
  clase A "Pre-simplificador vs integrador" para este caso)*
- [x] **(F) Potencias trig mixtas incoherentes**: `sin^2(x)cos^3(x)`
  residual mientras `sin^3*cos` y `sin^5` funcionan.
  *(graduado 2026-06-13 59c742081: productos sin(kx)^m·cos(kx)^n con
  argumento lineal compartido y una potencia impar (ambas ≥2) vía
  u=sin (cos impar) o u=cos (sin impar) → integrando polinómico
  u^kept·(1−u²)^spare delegado al integrador de polinomios. Cubre
  sin²cos³, sin³cos², sin⁴cos³, sin⁵cos², sin³cos⁴, sin(2x)³cos(2x)².
  Gate de intención min(m,n)≥2: los casos f^n·f' (sin³cos) y
  ambas-pares (sin²cos²) conservan su dueño. sin⁵cos² va como
  verification_gap — ambos canales simbólicos no recolapsan la derivada
  de grado 7, verificado numéricamente)*
- [x] **(F) Polinomio·potencia trig par**: `x·sin(x)^2`, `x^2·cos(x)^2`
  residuales mientras `sin(x)^2` solo ya reducía.
  *(graduado 2026-06-14 161e410b9: p(x)·sin(ax+b)^2 / p(x)·cos(ax+b)^2 con deg p≥1 e
  inner afín vía la identidad de ángulo mitad sin²u=½−½cos2u; el reescrito
  DISTRIBUYE en ½p(x)∓½p(x)cos2u y delega en el integrador de polinomios y
  en el de polinomio·cos(afín) por partes. Gate: un único factor
  Pow(Sin|Cos,2) afín, cofactor polinómico deg≥1. Peldaño honesto que queda:
  inner no afín substitución-amenable `x·sin(x²)²` — es ELEMENTAL
  (=x²/4−sin(2x²)/8 por u=x², el cofactor x aporta el du) pero hoy residual;
  y potencias pares ≥4 `x·sin⁴x`)*
  *(potencias pares ≥4 graduadas 2026-06-14 1f613fc39: p(x)·sin(ax+b)^n /
  p(x)·cos(ax+b)^n con n par en 4..=8 vía la reducción binómica
  sin^(2m)(u)=C(2m,m)/4^m + (2/4^m)Σ_{j=1}^{m}(−1)^j C(2m,m−j)cos(2j u) (coseno
  sin el (−1)^j). Multiplica por p, DISTRIBUYE en p·c₀ + Σ(cⱼ·p)cos(2j u) y delega
  igual que n=2; reutiliza el `combinatorics::binomial_coeff` compartido. Corre
  tras el dueño n=2 (solo añade n≥4), cap n≤8 a juego con los handlers desnudos
  sin⁴/⁶/⁸. `x·sin⁴x`, `x²·sin⁴x`, `x·cos⁴x`, `x·sin⁶x`, `x·sin⁸x`,
  `x·cos(2x+1)⁴`. Impar, inner no afín, n≥10 y cofactor constante declinan.
  Correcto verificado NUMÉRICAMENTE (sympy/mpmath, máx |d/dx F − f| ~1e-16); SIN
  filas de matriz porque el round-trip de verificación (derivar el resultado y
  recerrarlo contra el integrando) desborda la profundidad del simplificador
  para las salidas multi-ángulo de grado alto — el techo de round-trip del
  simplificador queda como peldaño. Queda aún el inner substitución-amenable
  `x·sin(x²)²`.)*
  *(inner substitución-amenable graduado 2026-06-14 98dc0e42b: p(x)·sin(g(x))² /
  cos(g(x))² con inner NO afín cuya cofactor aporta la derivada de la sustitución
  ya integra — `polynomial_times_trig_square_substitution_antiderivative`, hermano
  del dueño afín (corre después). El reescrito de ángulo mitad sin²(g)=(1−cos2g)/2
  es idéntico para cualquier g; lo único que cambia es si los términos p·cos(2g)
  distribuidos son integrables. `x·sin(x²)²→x/2−(x/2)cos(2x²)` es ELEMENTAL por
  u=x² (la cofactor x es du/2). AUTO-GATING por la linealidad de
  integrate_symbolic_expr (propaga None por `?`): cofactor que NO aporta el du
  (`x²·sin(x²)²` no-elemental/Fresnel, `x·sin(x³)²` necesita du=3x²) deja un término
  sin integrar y todo el reescrito da None → residual honesto, sin gate de
  paridad/grado. `x·sin(x²)²`, `x·cos(x²)²`, `x³·sin(x²)²`, `x·sin(2x²)²`. Verificado
  numéricamente (máx |d/dx F−f| ~1e-16); 2 filas de matriz para los casos
  cuadráticos simples (round-trip limpio). ITEM cerrado salvo potencias pares ≥4
  con inner no afín.)*
- [x] **(F) sech/csch no parsean**: `sech(x)^2` da "función no
  definida".
  *(graduado 2026-06-13 758e54e73: el parser desugariza sech→1/cosh,
  csch→1/sinh, coth→cosh/sinh en el lowering — `sech(x)²`,
  `integrate(sech(x)²)=tanh(x)`, `integrate(csch(x)²)=−coth(x)`,
  `diff(sech(x))=−sinh/cosh²`, `diff(coth(x))=−1/sinh²`, `sech(0)=1`.
  Gated a los tres nombres exactos con un argumento. `integrate(sech(x))`
  sigue residual — gudermannian es peldaño aparte)*
- [~] **(F) Bounds con e**: `integrate(1/x, x, 1, e)` residual
  (`ln(e)` no se evalúa); `tan(x) [0,π/4]` devuelve
  `ln(|cos(0)/cos(π/4)|)` sin plegar `cos(0)=1`.
  *(parcial 2026-06-13 a41cc8e55: la mitad de los bounds con e ya
  certifica — `e` y múltiplos racionales (2e, e/2, −e) son endpoints
  finitos con enclosure racional en el certificado de polo:
  `∫1/x [1,e]=1`, `∫1/x² [1,e]=(e−1)/e`, `∫2/x [1,e]=2`, y los polos
  se ubican (1/(x−2) en [1,e] diverge, 1/(x−3) certifica). Peldaños:
  `e²` (potencia, no múltiplo) y `√2` (algebraico) siguen Symbolic;
  el plegado `cos(0)=1` en la sustitución FTC de `tan [0,π/4]` es
  cosmético aparte)*
- [~] **(F) FTC/Leibniz en diff**: `diff(integrate(f(t),t,0,x), x)`
  → `f(x)` (+ regla de Leibniz con límites variables).
  *(parcial 2026-06-13 aea379c97: la regla de Leibniz
  `d/dx ∫_a(x)^b(x) f = f(b)b' − f(a)a'` ya aplica a integrandos
  ELEMENTALES no-integrables — `diff(∫e^(t²) [0,x]) = e^(x²)` (gaussiana),
  Fresnel `sin(t²)`, Si `sin(t)/t`, con regla de la cadena (`[0,x²] →
  2x·e^(x⁴)`) y bound inferior con cambio de signo. Las indefinidas
  siguen residuales (honestidad intacta): demuestra que esos residuales
  son frontera de PRESENTACIÓN, no de conocimiento. Peldaño restante:
  `f` OPACA simbólica (`diff(∫f(t)[0,x])=f(x)`) — el engine rechaza
  funciones desconocidas; necesita soporte de funciones simbólicas)*
- [ ] **(F) Diagnóstico de no-existencia en límites**: `sin(1/x)` en 0
  y laterales discrepantes deberían reportar "no existe" con motivo,
  no un residual genérico.

### P3 — educativo transversal

- [~] **(F) Límites con pedagogía**: los soportados no justifican nada
  (`sin(x)/x = 1` sin nombrar el límite notable/L'Hôpital/sandwich);
  impropias muestran `lim` sin evaluarlo con justificación
  (`lim e^(-x)(-x-1) = 0` por dominancia).
  *(parcial f73db6948..6dfaa5479 2026-06-21: 4 sub-ciclos del gatekeeper G2 vía el
  pipeline de enriquecimiento de cas_didactic, todo sound por chequeo de resultado/grado y huella
  NONE — (1) límites NOTABLES `sin/tan/arcsin/arctan/sinh/tanh(u)/u→1`, `(eᵘ−1)/u→1`,
  `(aᵘ−1)/u→ln(a)`, `ln(1+u)/u→1`, `(1−cos u)/u²→1/2`, `(1+u)^(1/u)→e`; (2) teorema del SÁNDWICH
  `u^k·sin/cos(1/u)→0`; (3) CONTINUIDAD/sustitución directa (polinomios) y FACTOR-Y-CANCELA (0/0
  removible); (4) DOMINANCIA en infinito (cociente de coeficientes líderes / grado mayor → 0/±∞).)*
  *(parcial 2026-06-22 335fb440e: sub-ciclo (5) la INDETERMINACIÓN 0/0 de orden superior en x=0
  narrada como L'Hôpital/Taylor — `(x−sin x)/x³`, `(eˣ−1−x)/x²`, `(tan x−x)/x³`, `(sin x−x)/x³`,
  `(cos x−1)/x²`, `(arctan x−x)/x³`, `(1−cos 2x)/x²`. El motor ya las calcula; faltaba nombrarlas.
  SOUND: un denominador `u^k` se anula solo en 0, así que la narración "0/0 en 0" es correcta exactamente
  cuando el punto del límite es 0 (chequeado literal); dado punto 0, den→0 y un resultado finito fuerza
  num→0, luego es 0/0 demostrable. Requirió cablear el punto al paso (`StepMeta.limit_point`). Falsos
  positivos refutados: `(x+1)/x` en 2 y `sin(πx)/x` en 1 (punto≠0, sustitución directa) declinan. Huella
  0 deltas; valores verificados vs sympy.)*
  *(parcial 2026-06-22 51a481fc0: sub-ciclo (5b) el 0/0 narrado en punto DESPLAZADO — generaliza
  (5) de punto-0 a cualquier punto racional verificando que el denominador POLINÓMICO se anule EN EL
  PUNTO del límite (`Polynomial::eval(punto)`); `ln(x)/(x−1)→1`, `(1−cos(x−1))/(x−1)²→1/2`,
  `(sin(x−1)−(x−1))/(x−1)³→−1/6` en 1, `(eˣ⁻²−1)/(x−2)→1` en 2 narran "0/0 en x=1/2". SOUND: den→0 en
  el punto + resultado finito ⟹ num→0; `ln(x)/(x−1)` declina en 0 (den→−1) y `cos(x)/(x−1)` en 1
  declina (num≠0→∞). vs sympy, huella 0 deltas.)*
  *(parcial 2026-06-22 03045f628: sub-ciclo (5c) el 0/0 con denominador TRIG/HIPERBÓLICO que se anula
  en el punto — el oráculo `limit_denominator_vanishes_at` se hace recursivo: polinomio ∪ `f(g)` con
  `f∈{sin,tan,sinh,tanh,arcsin,arctan,...}` (todas `f(0)=0`) y `g→0` ∪ `d^k` con `d→0`. `(cos x−1)/sin x→0`,
  `(1−cos x)/tan x→0`, `x²/sin x→0`, `(x−sin x)/sin x³→1/6`, `(eˣ−1)/sin x→1` narran 0/0 en x=0. SOUND:
  `f(g)→f(0)=0` cuando `g→0`; `cos`/`cosh` (valor 1 en 0) NO certifican. `1/cos(x)`, infinitos y puntos
  irracionales declinan. vs sympy, huella 0 deltas.)*
  Quedan: mostrar la SUSTITUCIÓN concreta y la factorización explícita (el punto YA está cableado;
  falta el factor), L'Hôpital/Taylor DERIVADOS paso a paso (no solo nombrados), y el 0/0 en punto
  IRRACIONAL (`tan x/sin x` en π — el cero del argumento es transcendente, fuera de `as_rational_const`).
- [~] **(F) Presentación**: ~10 nombres de regla en inglés dentro de
  narración española ('Normalize Negative Exponent', 'Identity
  Power'...); `--steps` ignorado en modo texto del CLI (los pasos solo
  viven en JSON); sin `+C` en antiderivadas; artefactos `ln(e)`,
  `x^(2-1)` en substeps de derivadas anidadas.
  *(parcial 2026-06-22 e8d20481c: los ~10 nombres de regla en inglés TRADUCIDOS al
  español — añadidas 10 entradas al chokepoint de display `visible_rule_name`
  ("Evaluate Meta Functions", "Cancel Same-Base Powers", "Factor Polynomial",
  "Identity Power", "Distribute Division Into Sum", "Normalize Negative Exponent",
  "Evaluate Logarithms", "N-ary Mul Combine Powers", "Combine Constants", "Abs Of
  Even Power"). El `rule_name` interno (clave de detección de ciclos/dispatch) queda
  intacto; solo cambia la etiqueta de display. Huella 0 deltas; smokes verdes. Quedan
  los OTROS sub-items: `--steps` en modo texto, `+C` en antiderivadas, y los artefactos
  `ln(e)`/`x^(2-1)` en substeps de derivadas anidadas.)*
- [ ] **(F) Etiquetas legibles en pre-cálculo**: `factor(x^2-9)` narra
  "Factor Polynomial" sin diferencia de cuadrados;
  `expand((x+2)^3)` narra "Evaluate Meta Functions".
- [~] **(F) Cosmético diff**: `e^(3x^2)/e^(4x^2)` no se combina a
  `e^(-x^2)` en derivadas 2ª/3ª de `exp(-x^2)`; derivadas anidadas de
  orden 4-5 devuelven blobs con `ln(e)`, `x^0`, `x^(2-1)`.
  *(parcial 2026-06-22 b44842e6: el COCIENTE `e^a/e^b` ahora combina aunque haya un
  co-factor en el producto — `try_rewrite_exp_quotient_expr` escanea el `Mul` con
  `mul_leaves`, extrae el `e`-power y recompone el resto. `diff(e^(−x²),x,1) = -2·x/e^(x²)`
  (antes `-2·e^(x²)·x/e^(2x²)`), y las derivadas 2ª/3ª pierden el `e^(7x²)/e^(8x²)` quedando
  un único `/e^(x²)`. Identidad exacta `e^a/e^b=e^(a-b)` (e≠0), sin gate; smokes diff/integrate
  verdes, huella solo +3 filtered_out. Quedan: las derivadas de orden ≥4 con `e`-powers
  MULTIPLICADOS sobre una SUMA (`e^(−6x²)·(...e^(5x²)...)`) — necesitan DISTRIBUIR antes de
  combinar; y los artefactos `ln(e)`/`x^0`/`x^(2-1)` en substeps (que en el RESULTADO ya se
  pliegan).)*
  *(parcial 2026-06-22 b5b65f11: la misma combinación GENERALIZADA a base positiva — `2^a/2^b →
  2^(a-b)`, `x·2^a/2^b → x·2^(a-b)`, `3^x/3^y → 3^(x-y)` (antes ninguno combinaba, solo `e`). Dos
  gates de soundness: base `e` o numérico>0 (incondicional), y para base numérica al menos un
  exponente SIMBÓLICO (deja los radicales `2^(1/2)=√2` a las reglas de raíz). Base simbólica /
  negativa / distinta declinan.)*

### Fuera del norte actual (clase I — no son ciclos)

Derivación implícita y `diff(f,x,n)`; funciones abstractas `f(x)`;
piecewise en el parser; funciones especiales (erf, Γ/digamma, Si/Ei,
LambertW) como *valores de salida*; assumptions; valor principal;
Risch completo; Gruntz completo; dominio complejo y multivariable
(Deferred Horizons). Si alguno se promueve, exige decisión explícita
de estrategia, no un ciclo de auto-mejora.

### Confirmaciones de honestidad (no tocar)

Los residuales no-elementales correctos deben seguir residuales:
`e^(-x^2)` (indefinida), `sin(x)/x`, `1/ln(x)`, `x^x`, `sin(1/x)` en 0
(no existe), `diff(floor(x))`. Cualquier ciclo que los "resuelva" es
un bug de soundness, no una mejora.
