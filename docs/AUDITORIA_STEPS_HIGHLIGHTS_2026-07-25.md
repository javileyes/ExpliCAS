# Auditoría sistemática de steps y highlights — 2026-07-25

**Encargo del usuario:** revisar sistemáticamente (a) que los steps no sean «mágicos», triviales
ni absurdos y (b) que los resaltados rojo/verde de LaTeX sean correctos. Disparada por dos
ejemplos elegidos al azar de `web/examples.csv` que fallaban pese a estar dados por revisados.

**Alcance:** las 210 filas de `web/examples.csv`, evaluadas con
`cas_cli eval "<expr>" --steps on --lang es --format json` (exactamente lo que sirve la web —
verificado contra `web/index.html:2373 renderSteps`, que pinta `rule_latex`, `before_latex`,
`after_latex` y `substeps[]` literalmente con MathJax 3 tex-svg, `web/index.html:18`). Corpus completo: **292 steps de
simplificación + 214 sub-pasos + 128 solve_steps**. Se auditaron las tres superficies.

**Este documento es un INFORME: no se ha tocado ni una línea de código del motor.**

**Método:** 11 detectores mecánicos sobre el corpus (candidatos, con falsos positivos) +
24 agentes de auditoría en dos oleadas (7 auditores por tramo + 7 verificadores adversariales
+ 7 sobre sub-pasos/magia/léxico + 3 de causa raíz sobre el código). Los verificadores
refutaron 45 de 368 hallazgos del primer barrido. Los hallazgos P0 citados en el resumen
ejecutivo fueron además re-verificados a mano contra el CLI.

---

## Resumen ejecutivo

| | |
|---|---|
| Hallazgos totales | **546** |
| Filas afectadas | **185 / 210** (25 limpias) |
| P0 (matemáticamente falso o engañoso) | **53** en 32 filas |
| P1 (rompe la didáctica) | 218 |
| P2 (ruido) | 215 |
| P3 (cosmético) | 60 |

Las dos quejas concretas del usuario son ciertas, y ninguna de las dos es un caso aislado:
**cada una es la punta de una familia con causa raíz medida e instrumentada.**

### Los dos ejemplos denunciados

**1. `taylor(sin(x), x, 0, 5)`, paso 3 — la regla «x → x».**
```
rule       = "Repartir el denominador entre los sumandos"
before     = (6·x^5 - 120·x^3)/720 + x
after      = (x^5 - 20·x^3)/120 + x
rule_latex = {\color{red}{x}} \rightarrow {\color{green}{x}}
```
El resaltado marca el sumando `x` — el único término que NO cambia — y deja sin marcar la
fracción, que es lo que realmente se reduce. Además el nombre de la regla miente: no reparte
ningún denominador, cancela el factor 6.

La causa está medida con una sonda instrumentada (RC-1 abajo): **PATH DRIFT**. El motor graba
el `path` del paso contra un árbol RAW (`ctx.add_raw`, que preserva el orden de los operandos),
pero la capa de presentación renderiza el árbol después de `normalize_expr_for_display`, que
lo reconstruye con `ctx.add(...)` — y `Context::add` **canonicaliza**: ordena los sumandos por
rank y colapsa niveles. En el testigo:
```
raw(global_before).ast  = Add( 1/720·(6x⁵−120x³) , x )     ← el path [0] apunta a la fracción
norm(global_before).ast = Add( x , 1/720·(6x⁵−120x³) )     ← el path [0] apunta a x
```
(`Variable` tiene rank 2, `Mul` rank 6, así que `ctx.add` los intercambia.) El índice grabado
sobrevive, el árbol al que apuntaba no. **125 de 411 pasos crudos del corpus (30 %) tienen
path drift**; en la ruta puramente posicional, 32 de 56 (57 %).

Y no es sólo un resaltado feo: cuando el rojo/verde caen sobre nodos equivocados, `rule_latex`
publica **identidades literalmente falsas** bajo el nombre de una regla. Verificadas a mano:
```
[001] ln(x^(-3/2)) → (3/2)·ln(x)        el signo desaparece (el paso hace −3/2·ln(x))
[180] 3·(−3) → (−3)²                     −9 = 9
[017] ∂/∂x(y·x²) → ∂/∂y(2·x·y)          dos operadores distintos igualados
[153] 2x²+2y²−4x² → 2x²                  falta el 2y²−2x²
[192] sin(u) → 0²
```
La web pinta `rule_latex` como la línea de regla justo bajo el nombre del paso
(`web/index.html:2387`), así que el alumno lo lee como una identidad enseñada.

**2. `integrate(2*x/sqrt(4+x^4)+1, x)` — un solo paso mágico.**
Un único «Calcular la integral» de `∫(2x/√(4+x⁴) + 1)dx` a `asinh(x²/2) + x`, con
`substeps` vacío: ni linealidad, ni sustitución, ni fórmula.

La causa raíz está localizada (RC-7): en toda la cadena de ~23 narradores de integración
hay **una sola descomposición aditiva del integrando**, y está detrás de un gate duro que
exige que el integrando **completo** sea un polinomio
(`focused_rule_substeps.rs:12887`). Todos los demás matchers exigen que el integrando entero
case UNA forma; `nested_inverse_polynomial_result` —el dueño de asinh— cae en `_ => None`
en cuanto ve un `Expr::Add` (`:18199`). Consecuencia comprobada: **quitar el `+1` restaura
la narración**.
```
integrate(2*x/sqrt(4+x^4), x)      → 2 sub-pasos («regla u'/√(1+u²) → asinh(u)», «identificar u y du»)
integrate(2*x/sqrt(4+x^4)+1, x)    → 0
integrate(cos(x), x)               → 1 sub-paso
integrate(cos(x)+1, x)             → 0
```
Es el ejemplo más elemental de linealidad de cualquier curso. El mismo patrón «la suma / el
vector / la aridad>2 se come la narración» aparece en cuatro familias más:
`diff([x²,sin(x)],x)`, `integrate([cos(x),e^x],x)`, `gcd(48,36,60)`, `limit((1+2/x)^x,∞)`.

### Lo más grave que apareció y que nadie había pedido buscar

Los sub-pasos —una superficie que la auditoría anterior no midió— contienen **matemáticas
falsas**, no sólo pedagogía pobre:

- **`F(b) − F(a)` sin paréntesis** (`focused_rule_substeps.rs:14225`,
  `format!("{} - {}", upper, lower)`): el signo menos sólo alcanza al primer término de `F(a)`.
  Afecta a **toda integral definida cuya antiderivada tenga ≥2 términos**. Reproducido y
  verificado a mano:
  ```
  integrate(cos(t)^2, t, pi/6, pi/3)   result = π/12
  sub-paso «Evaluar la antiderivada en los límites»:
      sin(2π/3)/4 + (π/3)/2 − sin(2π/6)/4 + (π/6)/2   = π/4   ← FALSO, y en pantalla
  ```
  Filas del corpus afectadas: 23, 40, 42, 185.
- **`integrate(x^2*sin(x), x)`**, sub-paso 8 «Integrar el término restante»:
  afirma `∫−2·sin(x)dx = 2x·sin(x) + (2−x²)·cos(x)`. La primitiva de −2sin(x) es 2cos(x).
- **`diff(arctan((1+x)/(1-x)), x)`**, sub-paso «sacar factor común (1−x)²»: afirma `A = (1−x)²·A`.
- **`integrate(1/(x^3-2), x)`** y **`integrate(1/(x^4-5), x)`**: el sub-paso «Factorizar el
  denominador» tiene `before == after` — anuncia una factorización y no factoriza, dejando al
  alumno con la idea de que x³−2 es irreducible sobre ℝ, cuando el propio resultado de la fila
  usa ∛2.
- **`equiv(e^(i*pi), -1)` → `false`** sin ningún warning (única fila del grupo Complejo que no
  emite el aviso): el alumno lee que la identidad de Euler es falsa. Con `--value-domain complex`
  devuelve `true` con 7 pasos correctos.

### Por qué esto seguía ahí después de la campaña «frente E»

No es que la campaña fallara: es que **su alcance no incluía estas dos dimensiones**, y el
propio ledger lo dice. `docs/ENGINE_COMBINATION_LEDGER.md:21441` cierra la tanda declarando
el corpus en «**A 31, D 0, E 24/25, F 44**», y `:21349` deja E **diagnosticado y aplazado**:
_«el rojo/verde del rule_latex renderiza nodos LOCALES mientras before_latex/after_latex
renderizan el GLOBAL […]; la unificación local-vs-global es E2»_.

- La **continuidad de cadena** residual (A=31) coincide exactamente con lo que mide hoy el
  detector D4: 31 hits. Nada se ha movido ni a mejor ni a peor.
- Los **highlights** (E) se redujeron un 75 % pero nunca a cero, y el residual declarado
  (24/25) es del mismo orden que los 32 highlight_* que salen hoy.
- **«Magia»** y **«el nombre de la regla dice la verdad»** no eran métricas del frente E.
  Por eso el ejemplo de `integrate` estaba intacto: nadie lo estaba midiendo.
- Los **sub-pasos** nunca entraron en el harness. Ahí es donde han aparecido los P0
  matemáticos.

La lectura honesta: «campaña completa» significaba _completa respecto de sus propias métricas_
(A/B/C/D/K y fugas de idioma), no _corpus limpio_. Elegir dos filas al azar y que ambas fallen
es el resultado esperado con 185/210 filas tocadas por algún hallazgo.

### Un aviso sobre la medida de partida

`steps_count` **mezcla dos contratos** (RC-14) y es engañoso como métrica:
- 16 filas reportan `steps_count: 0` llevando 3–7 `solve_steps` (sistemas, dsolve).
- 24 filas reportan 1–6 con el array `steps` **vacío** (el contador incluye solve_steps).
- Los `substeps` —donde vive casi la mitad de la narración— **no se cuentan nunca**.

Por eso «85 filas con un solo paso» NO es una regresión (RC-13): 49 de ellas narran bien
dentro de `substeps`, y sólo 36 son magia auténtica. Y de las 50 filas «con 0 pasos», sólo 15
son huecos reales; 12 son el grupo Complejo sin su modo, 4 son definiciones `:=` y 3 son
fallos del runner sin sesión.

Lo que **sí** es una pérdida real: el aparcado del listener de eventos en solve/dsolve
(commits `83481111e`, `797a93c21`, `ce968c793`) borró 759 pasos de confeti —correctamente,
eran ruido— pero sólo 11 de las 25 filas afectadas recibieron narración de reemplazo por
`solve_steps`. **14 filas quedaron completamente mudas**: 68, 70, 79, 80, 81, 83, 85, 88, 89,
90, 158, 189, 190, 209. De ahí sale la asimetría medida ecuación-vs-inecuación: de las 16 filas
de inecuación, 12 no narran nada (75 %); de las 19 de ecuación, sólo 2. Y en cada pareja el
lado `=` narra y el lado `<`/`>` no, con el mismo miembro izquierdo:
```
solve(abs(x-2)=1,x)  → 4 pasos    |  solve(abs(x-2)>1,x)  → 0
solve(sin(x)=1/2,x)  → 4          |  solve(sin(x)>1/2,x)  → 0
solve(x^(2/3)=2,x)   → 5          |  solve(x^(2/3)>2,x)   → 0
```
Los extremos de los intervalos que devuelve la inecuación son exactamente las raíces que el
lado ecuación sí sabe narrar: el trabajo ya está hecho, sólo falta emitirlo.

---

## Prioridad sugerida (por ROI, no ejecutada)

1. **RC-1 path drift** — transportar el path a través de `normalize_expr_for_display`, o
   resolver el foco por contenido restringido al diff `global_before`/`global_after`. Cierra de
   golpe los 32 highlight_* y buena parte de los 61 latex_render_bug. Es el arreglo con más
   alcance del informe.
2. **Guard de veracidad en `rule_latex`** (RC-3) — sustituir rojo→verde dentro de `before_latex`
   debe reproducir `after_latex`; si no, declinar el resaltado en vez de publicar una identidad
   falsa. Es exactamente el detector D2 de este informe convertido en invariante del motor, y
   es barato.
3. **Paréntesis en `F(b) − F(a)`** (`focused_rule_substeps.rs:14225`) — un `format!`. Mata 4
   P0 matemáticos y protege toda la familia de integrales definidas.
4. **Narrador aditivo genérico para `integrate`** (RC-7) — recuperar por linealidad las filas
   28, 48, 50, 117, 148, 156 con un solo molde recursivo.
5. **Piso de narración en solve** (RC-12) — cuando `steps` y `solve_steps` quedan ambos vacíos,
   emitir el esqueleto mínimo desde datos que ya existen (ecuación normalizada → estrategia →
   conjunto solución). Cierra las 14 filas mudas.
6. **Unificar el orden de términos entre los tres renderizadores** (RC-6) — hoy el mismo nodo
   se imprime en tres órdenes distintos (texto, LaTeX plano, LaTeX resaltado). 194 pasos.
7. **Contrato de contadores** (RC-14) — `steps_count == steps.len()` y publicar
   `solve_steps_count` / `substeps_count`. Sin esto, cualquier medición futura vuelve a mentir.
8. Léxico y paridad: 9 nombres de regla se quedan en español con `--lang en`; los `warnings`
   **no pasan por el catálogo i18n en ninguna dirección** (16 avisos en inglés con `--lang es`,
   13 en español con `--lang en`). Estructuralmente la paridad es buena: 210/210 filas dan el
   mismo número de pasos y la misma matemática en ambos idiomas.

---

## Causas raíz (16, con código y evidencia)

### El path posicional del paso (`step.path()`) se aplica a un árbol RE-CANONICALIZADO por `normalize_expr_for_display`, no al árbol contra el que se grabó

**Síntoma.** El resaltado rojo/verde cae en un subárbol que NO cambió (en el testigo, el sumando `x`) mientras la subexpresión realmente reescrita (la fracción) queda sin resaltar. Categorías: highlight_wrong_subexpression (20), highlight_red_equals_green (6), highlight_stale_form (7).

**Mecanismo.** El motor graba `global_before/global_after` reconstruyendo con `rewrite_at_expr_path_raw`, que usa `ctx.add_raw` (preserva el orden de los operandos), de modo que el `path` grabado ES válido sobre ese árbol RAW. La capa de presentación, en cambio, no renderiza ese árbol: `step_wire_presentation_snapshots` lo pasa por `normalize_expr_for_display`, que reconstruye TODO nodo a nodo con `ctx.add(...)`. Y `Context::add` no es identidad: canonicaliza Add/Mul (ordena los términos con `compare_add_terms` → `compare_expr` → `get_rank`) y además ELIMINA niveles (`Add(0,x)→x`, `Mul(0,·)→0`, `Mul(1,x)→x`, `Div(x,1)→x`, `Sub(x,0)→x`, `Neg(0)→0`). El resultado es que el mismo índice designa otro hijo (permutación) o incluso otra profundidad (colapso de niveles). En el testigo, medido con una sonda: `raw(gb).ast = Add(ExprId(1610612789)/*1/720·(6x⁵−120x³)*/, ExprId(536870912)/*x*/)` pero `norm(gb).ast = Add(ExprId(536870912)/*x*/, ExprId(1610612789))` — `Variable` tiene rank 2 y `Mul` rank 6, así que `ctx.add` los intercambia. El path grabado `[Left] → [0]` navegaba a `1/720·(6x⁵−120x³)` en el árbol raw y navega a `x` en el normalizado. Confirmado imprimiendo ambos: `path@raw = 1/720 * (6 * x^5 - 120 * x^3)` vs `path@norm = x  <-- PATH DRIFT`. El caso `surface_integral(...)` muestra la otra variante (colapso de nivel): el término `0*0` desaparece del árbol normalizado y el path `[0,0,0,0]` aterriza sobre `sin(u)`.

**Evidencia.** Sonda instrumentada (test temporal ya borrado) sobre el corpus de 160 expresiones extraídas de detector_report.json: 411 pasos crudos, 125 (30 %) con PATH DRIFT (el nodo al que apunta el path difiere entre el árbol raw y el normalizado). Desglose del testigo `taylor(sin(x), x, 0, 5)`, paso crudo 3: rule_name=`Distribute Division Into Sum`, path=`[Left]`, expr_path=`[0]`, global_before=`1/720 * (6*x^5 - 120*x^3) + x`, raw(gb).ast=`Add(Mul, x)`, norm(gb).ast=`Add(x, Mul)`, `path@raw`=la fracción, `path@norm`=`x`. Coincide exactamente con `before_latex = {\color{red}{x}} + \frac{6\cdot{x}^{5}-120\cdot{x}^{3}}{720}` del CLI. Reproducido idéntico en `taylor(sin(x)/x, x, 0, 4)` (D1 idx 182) y en `laplacian(ln(x^2+y^2),[x,y])` (D1 idx 153, 7 pasos consecutivos con drift).

**Alcance.** Toda familia cuyo paso se resuelva por path posicional. Medido: de los 56 pasos que toman la ruta DEFAULT-POSITIONAL, 32 (57 %) tienen drift → resaltado erróneo garantizado. Reglas afectadas más frecuentes en esa ruta: Combine Constants (10), Normalize Negation in Product (5), Canonicalize Negation (5), N-ary Mul Combine Powers (5), Product of Powers (4), Distribute Division Into Sum (2). Además `render_direct_focus_transition` PREFIJA `pathsteps_to_expr_path(step.path())` al path hallado por contenido, así que las rutas LOCAL-SCOPE con focus_path no vacío heredan el mismo defecto (50+43 pasos con drift en esas rutas).

**Código.** `crates/cas_didactic/src/timeline/simplify_highlights/global/default_transition.rs:14`, `crates/cas_didactic/src/timeline/simplify_highlights/global/direct.rs:15`, `crates/cas_didactic/src/timeline/simplify_highlights.rs:84`, `crates/cas_didactic/src/timeline/simplify_highlights.rs:90`, `crates/cas_solver_core/src/eval_step_pipeline.rs:278`, `crates/cas_ast/src/expression.rs:356`, `crates/cas_ast/src/expression.rs:375`, `crates/cas_ast/src/expression.rs:346`, `crates/cas_ast/src/ordering.rs:124`, `crates/cas_engine/src/engine/transform/step_recording.rs:74`, `crates/cas_math/src/expr_path_rewrite.rs:147`

**Arreglo propuesto.** El path grabado y el árbol renderizado tienen que provenir del mismo espacio. Tres opciones, de menor a mayor coste: (a) hacer que `normalize_expr_for_display` devuelva también un mapa de transporte de paths (o una variante `normalize_expr_for_display_with_path(ctx, expr, path) -> (ExprId, Option<ExprPath>)`) y que `default_transition`/`direct` usen el path transportado; si el nodo desaparece por colapso (Add(0,x)→x) el transporte devuelve None y hay que declinar el resaltado en vez de pintar un vecino. (b) Dejar de usar el path como fuente primaria: resolver SIEMPRE por contenido, buscando `step.before`/`step.after` (o `before_local`/`after_local`) dentro del árbol normalizado con `diff_find_path_to_expr`/`diff_find_paths_by_structure`, y usar el path solo como desempate cuando haya varias ocurrencias (esto es lo que ya hace `render_absolute_scope_transition` y por eso los pasos Add/Sub salen bien). (c) Renderizar el estado resaltado a partir del árbol RAW (`step.global_before`) en lugar del normalizado, aceptando que el orden mostrado sea el del AST crudo — barato pero rompería la coherencia de cadena que la auditoría 2026-07-23 introdujo. Añadir un gate de regresión: para cada paso, `navigate_to_subexpr(norm_global_before, path_usado)` debe ser estructuralmente igual a `step.before_local().unwrap_or(step.before)`; hoy falla en 30 % del corpus.

_(confianza: high)_

---

### `preferred_local_scope` no reconoce `Expr::Mul`, así que la regla que fue el testigo cae al camino puramente posicional en vez del basado en contenido

**Síntoma.** El paso del testigo (`Distribute Division Into Sum`) es exactamente el tipo de paso que debería resaltar el sitio de la regla por contenido, pero toma la rama `default` (solo path) y por eso el bug de path-drift le pega de lleno.

**Mecanismo.** `preferred_local_scope` (global.rs:43-56) decide la estrategia: si `step.before_local()` es `Some`, usa el scope local; si no, mira la FORMA de `step.before` y solo acepta `Function | Add | Sub | Div | Pow`. `Expr::Mul` NO está en la lista. Pero la forma canónica de una fracción con coeficiente racional es precisamente `Mul(Number(1/720), Add(...))` — de hecho `DivScalarIntoAddRule` se registra con `TargetKindSet::MUL`, o sea que TODAS sus aplicaciones tienen `step.before` de tipo Mul. Y la regla construye su `Rewrite` con `Rewrite::new(...).desc(...)`, sin `.local(before, after)`, de modo que `before_local`/`after_local` quedan en `None` (verificado en la sonda: `before_local = None`). Resultado: `preferred_local_scope` devuelve `None` → `render_default_global_transition` → path posicional puro. Si `Mul` estuviera en la lista, se iría por `render_absolute_scope_transition`, que localiza `before_local` y `after_local` por CONTENIDO dentro de los snapshots normalizados (`find_absolute_path`) y habría encontrado el índice 1 correcto en ambos lados.

**Evidencia.** La sonda imprime la decisión de ruta replicando `preferred_local_scope`: para el paso crudo 3 del testigo → `route = DEFAULT-POSITIONAL (before is Discriminant(5))`, siendo 5 el discriminante de `Expr::Mul`. Los pasos vecinos (`Add Fractions`, cuyo `before` es `Sub`) salen por `LOCAL-SCOPE(shape)` y su resaltado en el CLI es CORRECTO: `before_latex` del paso 2 marca las dos fracciones reales y `after_latex` la fracción resultante.

**Alcance.** Todo paso cuyo `before` sea Mul/Neg/Number/Variable/Constant/Matrix/Hold y no traiga `before_local`. En el corpus medido son los 56 pasos de la ruta DEFAULT-POSITIONAL (13,6 % de los pasos). Es un multiplicador del RC1: es la puerta por la que se entra al camino frágil.

**Código.** `crates/cas_didactic/src/timeline/simplify_highlights/global.rs:43`, `crates/cas_didactic/src/timeline/simplify_highlights/global.rs:48`, `crates/cas_didactic/src/timeline/simplify_highlights/global/scope.rs:23`, `crates/cas_didactic/src/timeline/simplify_highlights/global/scope.rs:59`, `crates/cas_engine/src/rules/algebra/fractions/small_rules.rs:47`, `crates/cas_engine/src/rules/algebra/fractions/small_rules.rs:55`

**Arreglo propuesto.** Añadir `Expr::Mul(_, _)` (y probablemente `Expr::Neg(_)`) a la lista de formas de `preferred_local_scope`, de modo que la localización pase por `render_absolute_scope_transition` (búsqueda por contenido) en lugar del path. Alternativa complementaria y más robusta: que `DivScalarIntoAddRule` (y las demás reglas que hoy no lo hacen) emita `.local(before_local, after_local)` con el subárbol exacto reescrito — `Rewrite::local` ya existe en `crates/cas_engine/src/rule.rs:122`/`:275`; con `before_local` presente, `preferred_local_scope` acierta sin depender de la forma. Ojo: arreglar solo esto no elimina el RC1, porque `render_direct_focus_transition` sigue prefijando el path grabado.

_(confianza: high)_

---

### `rule_latex` se re-deriva extrayendo los spans de color de `before_latex`/`after_latex`, sin comprobar que rojo ≠ verde ni que el span corresponda al cambio local real

**Síntoma.** `rule_latex = {\color{red}{x}} \rightarrow {\color{green}{x}}`: la "regla" mostrada al alumno es una identidad vacía. Categoría D1_red_equals_green (6 casos).

**Mecanismo.** `render_step_wire_latex` calcula primero `before_latex`/`after_latex` (con el resaltado, ya erróneo por el RC1) y después obtiene `rule_latex` con `derive_rule_latex_from_global(step, &before_latex, &after_latex, is_first)`, que simplemente recorta con `single_color_span` el único span `\color{red}{…}` del estado global y el único `\color{green}{…}` y los pega con `\rightarrow`. Su único guard es `red_full && !full_red_ok` (rechaza el rojo a pantalla completa fuera del primer paso y de los verbos de cálculo); NO comprueba que el span rojo sea distinto del verde, ni que se corresponda con `step.before_local()/step.before`. Por eso convierte un fallo de resaltado en una narración no-op. Lo notable es que el fallback que se salta, `render_normalized_rule_latex` (línea 221), SÍ usa `step.before_local().unwrap_or(step.before)` y `after_local().unwrap_or(step.after)` — habría producido el correcto `\frac{6x^5-120x^3}{720} \rightarrow \frac{x^5-20x^3}{120}`. O sea: la política "coherencia local-vs-global" degrada activamente este caso.

**Evidencia.** Salida del CLI para el testigo, paso 3: `rule_latex` es exactamente el span rojo de `before_latex` (`x`) y el span verde de `after_latex` (`x`). Los otros 5 hallazgos D1 tienen la misma estructura (`{\color{red}{1}} → {\color{green}{1}}` en laplacian, `{\color{red}{\sin(u)}} → {\color{green}{\sin(u)}}` en surface_integral, `{\color{red}{{y}^{2}}} → {\color{green}{{y}^{2}}}`), siempre con el span extraído textualmente del before/after resaltado.

**Alcance.** Los 6 casos D1_red_equals_green del informe, más cualquier paso futuro en que el resaltado sea incorrecto: el mecanismo garantiza que un fallo de resaltado se propague al zoom de la regla. También cubre parte de noop_or_trivial_step (23), porque un `rule_latex` X→X hace que el paso parezca vacío.

**Código.** `crates/cas_didactic/src/step_payloads/build/latex.rs:145`, `crates/cas_didactic/src/step_payloads/build/latex.rs:191`, `crates/cas_didactic/src/step_payloads/build/latex.rs:197`, `crates/cas_didactic/src/step_payloads/build/latex.rs:206`, `crates/cas_didactic/src/step_payloads/build/latex.rs:221`, `crates/cas_didactic/src/step_payloads/build/latex.rs:157`

**Arreglo propuesto.** Añadir dos guards a `derive_rule_latex_from_global`: (1) devolver `None` si `red == green` (nada que enseñar; el fallback `render_normalized_rule_latex` sí sabe qué cambió); (2) validar que el span extraído sea estructuralmente el `step.before_local().unwrap_or(step.before)` renderizado — si no coincide, devolver `None`. Ese segundo guard es, además, un DETECTOR barato del RC1: si el rojo del estado global no es el sitio de la regla, el resaltado está mal y el paso debería avisarlo (o caer al render local) en vez de publicarlo.

_(confianza: high)_

---

### El nombre visible de la regla sale de una tabla estática indexada por `rule_name` que ignora el `kind`/`description` de la aplicación concreta

**Síntoma.** El paso se titula «Repartir el denominador entre los sumandos» mientras la transformación mostrada es sacar el factor común 6 del numerador y absorberlo en el denominador ((6x⁵−120x³)/720 → (x⁵−20x³)/120). Categoría wrong_rule_name (24).

**Mecanismo.** `DivScalarIntoAddRule` tiene DOS comportamientos, discriminados en tiempo de aplicación por `DivScalarIntoAddRewriteKind`: `AllTermsCancel` y `FactorCommonCoefficientFromSum`. La regla vuelca ese discriminante SOLO en la `description` (`format_div_scalar_into_add_desc` → «Factor common coefficient from sum»), mientras el `rule_name` es siempre la cadena de registro «Distribute Division Into Sum». En la capa didáctica, `build_step_wire` llama a `visible_rule_name_for_step(&step.rule_name, &step.description)`; esa función tiene un puñado de casos sensibles a la descripción (Collect Terms, Finite Product, Reciprocal Trig Identity…) pero «Distribute Division Into Sum» NO está entre ellos, así que cae en `visible_rule_name(rule_name)` y sale la traducción fija de la línea 124. El nombre y el before/after vienen del MISMO `Step` (no de fases distintas del pipeline), pero de campos distintos con distinta fidelidad: el nombre viene de la identidad estática de la regla, el before/after del reescrito realmente aplicado. Nota: el `kind` no llega ni siquiera al `Step` como dato estructurado, solo aplanado en texto inglés dentro de `description`.

**Evidencia.** La sonda muestra para el paso crudo 3: `rule_name = Distribute Division Into Sum`, `description = Factor common coefficient from sum`. En `visible_rule_names.rs:499` existe una entrada por descripción para «Distribute a sum over the common denominator» (la que sí reparte el denominador, emitida por `crates/cas_solver/src/derive/fractions.rs:118`), lo que confirma que la coletilla «entre los sumandos» pertenece a esa OTRA transformación, no a la que se ejecutó.

**Alcance.** Toda regla multi-comportamiento cuyo discriminante viva solo en la descripción y no esté cableado en la tabla de nombres. Confirmado en `Distribute Division Into Sum`; el patrón se repite en las 24 filas de wrong_rule_name del informe y es estructuralmente idéntico al ya parcheado a mano para Collect Terms / Finite Product / Reciprocal Trig Identity.

**Código.** `crates/cas_didactic/src/step_payloads/build.rs:50`, `crates/cas_didactic/src/didactic/visible_rule_names.rs:276`, `crates/cas_didactic/src/didactic/visible_rule_names.rs:124`, `crates/cas_didactic/src/didactic/visible_rule_names.rs:499`, `crates/cas_engine/src/rules/algebra/fractions/small_rules.rs:60`, `crates/cas_math/src/div_scalar_into_add_support.rs:17`

**Arreglo propuesto.** Añadir en `visible_rule_name_for_step` un caso `"Distribute Division Into Sum" if description == "Factor common coefficient from sum" => "Simplificar la fracción sacando el factor común del numerador"` (y otro para `"All terms cancel"`), más su entrada en la tabla es→en de `rule_name_es_to_en`. Arreglo estructural preferible: que la tabla de nombres visibles se indexe por (rule_name, description) de forma sistemática y que un test de contrato falle cuando una regla emita una `description` que la tabla no cubra — hoy el fallback silencioso al nombre estático es lo que deja pasar la mentira.

_(confianza: high)_

---

### La localización por contenido devuelve la PRIMERA ocurrencia; con hash-consing todos los subárboles iguales comparten ExprId, así que resalta una copia que no cambió

**Síntoma.** En los pasos que sí toman la ruta local, el rojo aparece sobre una ocurrencia distinta de la reescrita (p. ej. el `1` de `x^{2-1}` en un término que no se tocó, o el `sin(u)` de otro factor). Categorías highlight_wrong_subexpression y parte de D2_hl_substitution_mismatch (39).

**Mecanismo.** `render_absolute_scope_transition` localiza el foco con `find_absolute_path`, que es `diff_find_path_to_expr(...).or_else(|| diff_find_paths_by_structure(...).into_iter().next())`. `diff_find_path_to_expr` hace DFS y devuelve el PRIMER `current == target`; `diff_find_paths_by_structure` recoge todas pero se toma `.next()`. Como el `Context` internaliza expresiones (`Context::add` deduplica por hash), `1`, `sin(u)`, `y^2` o `2-1` tienen un único ExprId compartido por todas sus apariciones — el buscador no puede distinguir cuál reescribió la regla y elige la primera en preorden. El único desempate existente es un parche muy específico (`narrow_before_path_to_changed_additive_child`, que solo estrecha hacia el hijo aditivo de un `Div`), no una política general. La señal que sí desambiguaría — el `path` grabado — está disponible pero no se usa aquí (y de hecho, por el RC1, tampoco sería fiable tal cual).

**Evidencia.** D1 idx 153 (`laplacian(ln(x^2+y^2),[x,y])`, paso 2, regla `Identity Power` con `before = x^0`): el before_latex resalta el `1` de `{x}^{2 - {\color{red}{1}}}` dentro del PRIMER término, cuando el reescrito era el exponente del segundo factor; y before/after muestran el cambio real en `x^{2-1-1} → 1`, que no es lo resaltado. D1 idx 192 (`surface_integral`) resalta `\sin(u)` dentro de `(0 + \sin(u))` mientras la regla combinaba potencias en otro factor. En ambos, el ExprId buscado aparece varias veces en el árbol.

**Alcance.** Todos los pasos que llegan a `render_absolute_scope_transition`/`resolve_focus_path` cuyo foco sea un subárbol repetido (números pequeños, variables, funciones trig con el mismo argumento). Es la parte de highlight_wrong_subexpression (20) y de D2_hl_substitution_mismatch (39) que NO explica el RC1.

**Código.** `crates/cas_formatter/src/path.rs:37`, `crates/cas_formatter/src/path.rs:44`, `crates/cas_formatter/src/path.rs:176`, `crates/cas_didactic/src/timeline/simplify_highlights/global/scope.rs:122`, `crates/cas_didactic/src/timeline/simplify_highlights/global/scope.rs:130`, `crates/cas_didactic/src/timeline/simplify_highlights/global/scope/focus_path.rs:6`

**Arreglo propuesto.** Desempatar por proximidad al path grabado (una vez transportado correctamente al árbol normalizado, ver RC1): de todas las ocurrencias que devuelva `diff_find_all_paths_to_expr`/`diff_find_paths_by_structure`, escoger la de mayor prefijo común con el path del paso. Si hay empate o el transporte falló, resaltar TODAS las ocurrencias (`render_with_paths` ya lo soporta) o declinar el resaltado, pero nunca elegir la primera arbitrariamente. Complementariamente, usar el diff before/after global para restringir la búsqueda a los subárboles que realmente difieren entre `global_before_expr` y `global_after_expr`.

_(confianza: medium)_

---

### Dos renderizadores con criterios de orden distintos: el renderizador plano ordena los sumandos por grado, el que resalta por path los emite en orden del AST

**Síntoma.** `before` (texto) y `before_latex` muestran los mismos términos en ORDEN DISTINTO — `(6·x^5 - 120·x^3)/720 + x` frente a `x + \frac{6x^5-120x^3}{720}` — lo que hace que el resaltado parezca aún más descolocado y rompe la lectura de la cadena. Categorías text_latex_divergence (18) y D3b_text_vs_latex_order_only (194).

**Mecanismo.** `LaTeXRenderer::format_add` (el camino sin resaltado) aplana la cadena Add y la ORDENA con `crate::display::cmp_term_for_display` (grado descendente, positivos primero). `PathHighlightedLatexRenderer::format_additive_path` (el camino con resaltado, el que produce `before_latex`/`after_latex`) recoge los términos con `collect_signed_add_terms_path` en orden izquierda-derecha del AST y NO los ordena — no puede, porque los índices del path tienen que seguir la estructura real. El renderizador de texto (`render_expr`) usa su propio orden de display, tercer criterio. Así, la MISMA expresión normalizada `Add(x, Mul(1/720, ...))` se imprime como `1/720 * (6*x^5 - 120*x^3) + x` en texto, como `\frac{6x^5-120x^3}{720} + x` en LaTeX plano y como `x + \frac{...}{720}` en LaTeX resaltado. Verificado en la sonda imprimiendo los tres del mismo ExprId.

**Evidencia.** Sonda sobre el testigo, paso 3: `norm(gb).ast = Add(ExprId(536870912)/*x*/, ExprId(1610612789))` pero `norm(gb).text = 1/720 * (6 * x^5 - 120 * x^3) + x` y `norm(gb).tex = \frac{6\cdot{x}^{5} - 120\cdot{x}^{3}}{720} + x`, mientras el `before_latex` publicado por el CLI (vía el renderizador con path) es `{\color{red}{x}} + \frac{6\cdot{x}^{5} - 120\cdot{x}^{3}}{720}`. Tres órdenes para el mismo nodo.

**Alcance.** Es el detector más numeroso del informe: 194 pasos con divergencia de orden texto-vs-LaTeX y 18 clasificados como text_latex_divergence. Afecta a cualquier expresión con ≥2 sumandos cuyo orden canónico del AST difiera del orden de display, es decir, prácticamente toda suma donde un monomio de bajo rank convive con un producto/cociente.

**Código.** `crates/cas_formatter/src/latex_core.rs:268`, `crates/cas_formatter/src/latex_core.rs:276`, `crates/cas_formatter/src/latex_core.rs:1712`, `crates/cas_formatter/src/latex_core.rs:1640`, `crates/cas_didactic/src/step_payloads/build/expr.rs:15`, `crates/cas_didactic/src/step_payloads/build/latex.rs:145`

**Arreglo propuesto.** Unificar el orden en UNA sola política. Lo correcto es que `PathHighlightedLatexRenderer::format_additive_path` ordene igual que `format_add` (`cmp_term_for_display`) llevándose consigo el `ExprPath` de cada término — `collect_signed_add_terms_path` ya devuelve tuplas `(ExprId, ExprPath, bool)`, así que basta ordenar ese vector por el mismo comparador antes de emitir; el resaltado sigue siendo correcto porque el color se decide por path, no por posición. Existe ya un comentario de contrato en `crates/cas_formatter/src/function_latex.rs:4` que dice que ambos renderizadores deben producir lo MISMO: hoy ese contrato no se cumple para Add/Sub y no hay test que lo fije. Añadir una prueba de propiedad: para toda expresión, `LaTeXExpr::to_latex(e)` == `PathHighlightedLatexRenderer` con config vacía.

_(confianza: high)_

---

### La narración de integrales NO descompone sumas: solo el integrando polinómico (y por accidente el racional) llega a la linealidad

**Síntoma.** `integrate(2*x/sqrt(4+x^4)+1, x)` emite 1 paso «Calcular la integral» con CERO substeps, mientras que `integrate(2*x/sqrt(4+x^4), x)` (el mismo integrando sin el `+1`) sí narra «Usar la regla de u'/sqrt(1+u^2) -> asinh(u)» + «Identificar u y du». Lo mismo pasa con `cos(x)+sin(x)`, `sin(x)+x`, `e^x+x^2`, `sqrt(x)+x`, `x*cos(x^2)+1`, `x^2*e^x+1`.

**Mecanismo.** En toda la cadena de ~23 narradores de `Symbolic Integration` de focused_rule_substeps.rs hay UNA SOLA descomposición aditiva del integrando: `AddView::from_expr(ctx, args[0])` en la línea 12891, dentro de `generate_basic_polynomial_integration_substeps`. Y está precedida por un gate duro en la línea 12887: `if Polynomial::from_expr(ctx, args[0], var_name).is_err() { return Vec::new(); }` — o sea, la linealidad («Usar linealidad de la integral» + «Integrar cada término») solo se emite si el integrando COMPLETO es un polinomio en la variable. Todos los demás matchers exigen que el integrando entero case UNA forma única, y además que el RESULTADO case una forma única: `nested_inverse_polynomial_result` (el dueño de asinh) recorre Neg/Hold/Mul-por-constante/Div-por-constante y cae en `_ => None` (línea 18199) en cuanto ve un `Expr::Add`. El resultado del testigo es `asinh(x^2/2) + x` — un Add — así que el matcher del asinh muere por los dos extremos a la vez (integrando `... + 1` y resultado `... + x`). Las sumas racionales narran por accidente, absorbidas por el matcher de fracciones parciales (`x + 1/x`, `1/(x^2+1)+x^3`), no por linealidad.

**Evidencia.** Barrido CLI (--steps on --lang es --format json) sobre 14 integrandos aditivos: narran 6, TODOS polinómicos (x^2+1, 3x^4-2x+5, x^3-2x) o racionales (1/(1+x^2)+1, 1/(x^2+1)+x^3, x+1/x); mudos 8 (sin(x)+x, e^x+x^2, sqrt(x)+x, cos(x)+sin(x), e^x*sin(x)+1, x*cos(x^2)+1, x^2*e^x+1, y el testigo 2x/sqrt(4+x^4)+1). El par de control es concluyente: quitar el `+1` al testigo enciende la narración. `grep` de `AddView|Expr::Add` en el tramo 12800–19200 del fichero devuelve un único punto de descomposición del integrando (12891). El `match` final del dispatcher termina en `_ => Vec::new()` (línea 448) y `Symbolic Integration` no aparece en él.

**Alcance.** Todo integrando que sea una SUMA y no sea polinomio ni función racional en la variable. En el tramo de corpus 021–042: 028 (testigo) y 025 (el paso integral de e^x·sin(x)) caen aquí. En el probe ampliado: 8/14 sumas mudas (57%). Incluye el ejemplo más elemental de linealidad de cualquier curso, `∫(cos x + sin x)dx`.

**Código.** `/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/focused_rule_substeps.rs:12887`, `/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/focused_rule_substeps.rs:12891`, `/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/focused_rule_substeps.rs:18130`, `/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/focused_rule_substeps.rs:18199`, `/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/focused_rule_substeps.rs:49`, `/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/focused_rule_substeps.rs:448`

**Arreglo propuesto.** Anteponer al resto de la cadena un narrador aditivo genérico: si `before` es `integrate(Add(...), x)` y `after` también parte en el mismo número de sumandos, emitir (a) un substep «Usar linealidad de la integral» con el desarrollo concreto `∫t1 dx + ∫t2 dx + ...` (ya existen los helpers `integral_sum_display`/`integral_sum_latex` usados en 12893–12901) y (b) recursión: para cada término, re-invocar `generate_focused_rule_substeps` sobre un `Step` sintético `integrate(t_i, x) -> a_i`, concatenando los substeps hijos. Eso reutiliza los 23 matchers existentes sin tocarlos y hace que el testigo herede la narración de asinh que ya funciona en solitario. El emparejamiento término↔antiderivada debe verificarse (derivar `a_i` y comparar con `t_i`) antes de emitir, para no inventar correspondencias.

_(confianza: high)_

---

### No existe canal de traza método→didáctica: el integrador devuelve solo un ExprId y la capa didáctica ADIVINA la ruta re-ejecutando oráculos sobre un Context clonado

**Síntoma.** La narración de una integral no depende de qué método usó realmente el motor, sino de si alguno de ~23 reconocedores de forma escritos aparte logra re-identificar la familia mirando (integrando, resultado). Cuando el integrador usa una ruta sin oráculo público espejo, el paso sale mudo aunque el motor sepa perfectamente qué hizo.

**Mecanismo.** `cas_math::symbolic_integration_support::integrate_symbolic_expr` tiene la firma `(ctx, expr, var) -> Option<ExprId>`: devuelve la antiderivada desnuda, sin ninguna estructura de traza. El único metadato que sube es `IntegrationOutcome.trace_kind`, un enum de DOS valores (`EducationalRule` | `AlgorithmicBackendSummary`) que además se consume solo para suprimir condiciones y para envolver en `Hold` — nunca se adjunta al `Step`. Como consecuencia, `cas_didactic` no puede leer la ruta y la reconstruye a ciegas: `generate_integration_substitution_substeps` hace `let mut scratch = ctx.clone();` y encadena ~15 predicados `integrate_symbolic_is_*_target(&mut scratch, ...)` con `&&`/`!` para conjeturar cuál se aplicó; `generate_definite_integral_substeps` llega a RE-INTEGRAR desde cero (`integrate_symbolic_expr(&mut scratch, integrand, &var_name)` en 14101, más un probe del backend). El desajuste es cuantificable: el fichero exporta 79 predicados `pub fn integrate_symbolic_is_*` frente a 216 funciones de ruta `*antiderivative` — ~137 rutas no tienen espejo y son mudas por construcción. Sobre la pregunta explícita del encargo: el wire NO pierde nada — `StepWire.substeps` se serializa (con `skip_serializing_if = "Vec::is_empty"`) y `integrate(x*e^x,x)` sale con 4 substeps tanto en JSON como en texto; lo que falta es aguas arriba. El canal de substeps del MOTOR (`Rewrite::substep()`, rule.rs:352, propagado a `Step.meta.substeps`) sí existe pero (a) no lo usa ningún fichero de `rules/calculus/` — solo arithmetic.rs (83 usos), hyperbolic.rs (6) y values_rules.rs (1) — y (b) el render solo itera `enriched.sub_steps`, nunca `step.substeps()`, que se lee únicamente como señal de visibilidad en prepare.rs:74.

**Evidencia.** `grep -c 'pub fn integrate_symbolic_is_' symbolic_integration_support.rs` = 79; `grep -c '^fn .*antiderivative|^pub fn .*antiderivative'` = 216. `grep -c 'ctx.clone()' focused_rule_substeps.rs` = 101 (la capa didáctica clona el Context una vez por matcher para poder correr oráculos `&mut`). `grep -rn '\.substep(' crates/cas_engine/src/rules/calculus/` no devuelve nada. Prueba positiva de que el wire sí expone substeps: `cas_cli eval "integrate(x*e^x, x)" --steps on --format json` incluye un array `substeps` con 4 entradas; el mismo comando sobre el testigo no emite la clave (vec vacío, elidido por serde).

**Alcance.** Todas las familias de integración: la calidad narrativa es proporcional a cuántos oráculos-espejo se hayan escrito a mano, no a lo que el motor sabe. Además coste de rendimiento: hasta 101 clonados de Context por paso de integración enriquecido.

**Código.** `/Users/javiergimenezmoya/developer/math/crates/cas_math/src/symbolic_integration_support.rs:22484`, `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/calculus/integration.rs:15`, `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/calculus/integration.rs:51`, `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/calculus/integrate_rule.rs:8`, `/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/focused_rule_substeps.rs:20099`, `/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/focused_rule_substeps.rs:14101`, `/Users/javiergimenezmoya/developer/math/crates/cas_api_models/src/wire_types.rs:159`, `/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/step_payload_render/substeps.rs:9`, `/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/step_payloads/prepare.rs:74`, `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rule.rs:352`

**Arreglo propuesto.** Sustituir `Option<ExprId>` por `Option<IntegrationTrace>` en el núcleo, donde `IntegrationTrace { result: ExprId, method: IntegrationMethodTag, children: Vec<IntegrationTrace> }` y `IntegrationMethodTag` nombre la ruta (Linearity, PowerRule, ByParts{u,dv}, USub{u,du}, TableInverse{fn}, PartialFractions{factored,decomp}, Backend{method}). Como la migración de 216 rutas es inviable de golpe: (a) añadir el campo con `#[default] Unknown` y rellenarlo incrementalmente empezando por los ~15 puntos de retorno de más tráfico; (b) propagarlo por `IntegrationOutcome` → `Rewrite::substep()` → `Step.meta.substeps` (el canal ya existe, rule.rs:182); (c) cablear el render para volcar `step.substeps()` cuando `enriched.sub_steps` esté vacío, en `collect_step_payload_substeps`. Los 23 matchers de focused_rule_substeps.rs quedan como fallback y se van retirando a medida que las rutas emiten traza propia.

_(confianza: high)_

---

### La ruta del backend algorítmico (RootSum/Hermite/racional general) tira una traza rica en el punto de retorno público

**Síntoma.** `integrate(1/(x^5-x-1), x)`, `integrate(1/(x^7-1), x)`, `integrate(1/(x^5-2), x)`, `integrate(x/(x^4+x+1), x)` devuelven un `root_sum(...)` correcto con CERO substeps — ni siquiera el nombre del método ni el hecho de que se pasó al backend algorítmico.

**Mecanismo.** `try_algorithmic_integration_backend` produce un `AlgorithmicIntegrationCandidate` con `method`, `method_probe_attempts`, `method_probe_no_match_reasons`, `verification_status`, `verification_evidence`, `trace_level` (model.rs:512–540) — es decir, la traza que la narración necesitaría ya está calculada. Pero `public_algorithmic_backend_fallback` la colapsa a `Option<(ExprId, Vec<ConditionPredicate>)>`: usa `candidate.method` solo para un `match` de admisión de condiciones y luego devuelve `Some((result, candidate.required_conditions))`, descartando todo lo demás. Aguas abajo, `integrate_with_result_preservation` envuelve el resultado en `Hold` cuando `trace_kind == AlgorithmicBackendSummary` (integration_result_pipeline.rs:73–75), lo que además rompe los matchers didácticos que hacen `match ctx.get(after)` sin desenvolver el Hold.

**Evidencia.** Corpus: 034 (1/(x^5-x-1)), 035 (1/(x^7-1)), 036 (1/(x^5-2)), 039 (x/(x^4+x+1)) → `substeps_on_integral = 0` en los cuatro. `AlgorithmicIntegrationMethod` ya tiene `metric_label()` con nombres publicables («rational», «hermite», «heurisch_probe», «table_reused») en model.rs:17–25, sin ningún consumidor didáctico. Contraste: 032 (1/(x^3-2)) sí narra 3 substeps porque cayó en la ruta educativa de fracciones parciales, no en el backend.

**Alcance.** Toda integral racional que escape a las rutas educativas y termine en el backend algorítmico — el frente G1 completo (RootSum, Hermite, Ostrogradsky). 4/22 filas del tramo 021–042 del corpus.

**Código.** `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/calculus/integration.rs:70`, `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/calculus/integration.rs:126`, `/Users/javiergimenezmoya/developer/math/crates/cas_math/src/general_integration_backend/model.rs:512`, `/Users/javiergimenezmoya/developer/math/crates/cas_math/src/general_integration_backend/model.rs:8`, `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/calculus/integration_result_pipeline.rs:48`, `/Users/javiergimenezmoya/developer/math/crates/cas_engine/src/rules/calculus/integration_result_pipeline.rs:73`

**Arreglo propuesto.** Cambiar la firma de `public_algorithmic_backend_fallback` a devolver también `candidate.method` (y opcionalmente `verification_status`), y hacer que `IntegrationOutcome.trace_kind` pase de enum binario a `Backend(AlgorithmicIntegrationMethod)`. Con eso `generate_general_rational_integration_substeps` (que ya sabe narrar Ostrogradsky + factorizar + descomponer, líneas 14296–14343) puede dispararse por MÉTODO en vez de por re-detección de forma, y para RootSum se puede emitir la narración honesta que hoy no existe: «Las raíces del denominador no son expresables por radicales → se usa la suma sobre raíces del resolvente R(t)», con `R` concreto. Desenvolver el `Hold` con `unwrap_internal_hold` al entrar en los matchers (algunos ya lo hacen, 14271–14277; otros no).

_(confianza: high)_

---

### El envoltorio FTC de la integral definida narra la cáscara pero nunca el método de la antiderivada

**Síntoma.** `integrate(1/(x^5-x-1), x, 2, 3)` «narra» 2 substeps («Hallar la antiderivada», «Evaluar la antiderivada en los límites») mientras su gemela indefinida `integrate(1/(x^5-x-1), x)` sale muda. El substep «Hallar la antiderivada» tiene como after un `root_sum(...)` de 200 caracteres aparecido de la nada: enuncia QUÉ se obtuvo, nunca CÓMO.

**Mecanismo.** `generate_definite_integral_substeps` emite una plantilla fija de 2 substeps: recalcula la antiderivada clonando el contexto (`integrate_symbolic_expr(&mut scratch, integrand, &var_name)`, línea 14101, con fallback a un probe `diagnostic_only` del backend) y luego narra Barrow. En ningún punto delega en el resto de la cadena de narradores para explicar cómo se obtuvo esa antiderivada: no construye un `Step` sintético `integrate(integrand, x) -> F` ni re-invoca `generate_focused_rule_substeps`. Por eso la fila definida siempre marca «narra» en cualquier métrica de cobertura aunque su contenido matemático sea opaco, enmascarando el hueco real.

**Evidencia.** Corpus 037.json: substep «Hallar la antiderivada» con before `\frac{1}{x^5-x-1}` y after `\sum_{t:80t^2+1-2869t^5-...}` — salto sin explicación. Corpus 034.json (misma integral, indefinida): 0 substeps. Corpus 022/040/042 muestran el mismo par fijo de títulos. Las únicas dos variantes son el caso polo («Detectar un polo dentro del intervalo», 038/041) y el split de |afín| (14093–14098).

**Alcance.** Todas las integrales definidas: 8 filas del tramo 021–042 (021,022,023,037,038,040,041,042). Además distorsiona cualquier medición de cobertura narrativa al contar como «narra» lo que es una cáscara.

**Código.** `/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/focused_rule_substeps.rs:14047`, `/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/focused_rule_substeps.rs:14101`, `/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/focused_rule_substeps.rs:14122`

**Arreglo propuesto.** Tras calcular `antiderivative` (14101), construir un `Step` sintético con `before = integrate(integrand, var)` / `after = antiderivative` y `rule_name = "Symbolic Integration"`, pasarlo por `generate_focused_rule_substeps`, y INSERTAR los substeps resultantes entre «Hallar la antiderivada» y «Evaluar en los límites». Con la recursión aditiva de la causa 1 ya implementada, esto propaga la mejora a todo el frente definido sin más código. Cuidado con la recursión infinita: el `Step` sintético tiene 2 args, así que `generate_definite_integral_substeps` (que exige `args.len() == 4`) declina solo.

_(confianza: high)_

---

### Substeps de fracciones parciales no-op: se anuncia una maniobra que no ocurre, violando el contrato de DIDACTIC_SUBSTEP_NORMALIZATION.md

**Síntoma.** `integrate(1/x, x)` emite el substep «Descomponer en fracciones parciales» con before `\frac{1}{x}` y after `\frac{1}{x}` — before y after IDÉNTICOS. `integrate(x + 1/x, x)` emite el mismo título para lo que en realidad es linealidad (`(x^2+1)/x → 1/x + x`), no una descomposición en fracciones parciales.

**Mecanismo.** `generate_rational_linear_partial_fraction_integration_substeps` llama a tres oráculos de descomposición encadenados con `.or_else()` y, en cuanto uno devuelve `Some(decomposition)`, emite incondicionalmente el par de substeps con clave `partial_fractions.decompose` + `integral.integrate_simple_terms` (líneas 15161–15177). No hay ninguna comprobación de que `decomposition != args[0]`: cuando el integrando ya está descompuesto (denominador lineal único, `1/x`), el oráculo devuelve la identidad y el narrador la presenta como paso didáctico. Esto contradice explícitamente el contrato del repo: «if the only possible substep would duplicate the parent step, emit no substep» y «generic template math is worse than no substep» (DIDACTIC_STEPS.md, sección SubSteps; DIDACTIC_SUBSTEP_NORMALIZATION.md, regla 3).

**Evidencia.** `cas_cli eval "integrate(1/x, x)" --steps on --format json` → substep con `before_latex == after_latex == "\\frac{1}{x}"`. `cas_cli eval "integrate(x + 1/x, x)"` → título «Descomponer en fracciones parciales» sobre `(x^2+1)/x → 1/x + x`, que es reparto de una suma, no descomposición de Heaviside. El `let Some(decomposition) = decomposition else { return Vec::new(); }` de la línea 15155 solo filtra el `None`, nunca la identidad.

**Alcance.** Integrandos racionales con denominador ya irreducible/lineal simple y sumas que el simplificador ha unificado sobre denominador común antes de llegar al matcher. En el corpus, 024 y 027 pasan por este generador; el efecto visible más claro está fuera del tramo (`1/x`). Es un problema de VERACIDAD narrativa, no de cobertura: aquí el motor habla, pero dice algo falso.

**Código.** `/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/focused_rule_substeps.rs:15110`, `/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/focused_rule_substeps.rs:15155`, `/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/focused_rule_substeps.rs:15164`, `/Users/javiergimenezmoya/developer/math/docs/DIDACTIC_SUBSTEP_NORMALIZATION.md:1`, `/Users/javiergimenezmoya/developer/math/docs/DIDACTIC_STEPS.md:170`

**Arreglo propuesto.** Añadir tras la línea 15155 un guardia estructural: si `decomposition` y `args[0]` renderizan igual (o son el mismo `ExprId` tras `simplify` en el scratch), devolver `Vec::new()` y dejar que la cadena siga hasta un narrador más honesto (regla de tabla `∫dx/x = ln|x|`). Y separar títulos: cuando la «descomposición» solo reparte una suma sobre denominador común, usar la clave `integral.use_linearity` en vez de `partial_fractions.decompose`. Nota: el guardia de identidad es el mismo patrón que ya usa `is_noop_wire_step` en step_payloads.rs:122 para los pasos principales — conviene reutilizar el comparador de display en lugar de escribir uno nuevo.

_(confianza: high)_

---

### El aparcado del listener de eventos en solve/dsolve borró 759 de los 764 pasos perdidos — y solo 11 de las 25 filas afectadas recibieron narración de reemplazo

**Síntoma.** 25 filas del corpus pasaron de N pasos a exactamente 0 entre 94f378a92 y HEAD. 14 de ellas quedan COMPLETAMENTE mudas (0 steps Y 0 solve_steps): 68, 70, 79, 80, 81, 83, 85, 88, 89, 90, 158, 189, 190, 209. Las filas mudas totales del corpus suben 21 -> 34.

**Mecanismo.** E3 (83481111e) envuelve todo `evaluate_solve_parsed_with_session` en `replace_step_listener(None)` y su espejo 797a93c21 hace lo mismo en los dos brazos Dsolve/DsolveSystem del dispatch (más ce968c793 en sistemas no-lineales). El diagnóstico era correcto: lo que viajaba por ese canal era scratch interno (verifiqué los dumps PRE: [079] son OCHO pasos «Combinar las constantes: 0 + 2 => 2»; [070] «expand(0) => 0»; [190] expande tan->sin/cos y lo vuelve a plegar en round-trip). El problema no es el filtro, es que el canal de reemplazo (`solve_steps`) se construyó por FAMILIAS en los peldaños 4097be7c4/5668c3acb/2d68b2c2c/c690ac28b/2071febfd/d1b722fa3/d4112a7a7/abbd0a7e8, y 14 familias del corpus se quedaron sin kernel: inecuaciones racionales, con raíz, con abs, logarítmicas, trigonométricas periódicas, ecuaciones exponenciales de base distinta, radicales, y el decline de Riccati. Para esas, aparcar el listener equivale a apagar la narración sin encender nada.

**Evidencia.** Medición directa: construí el binario release de 94f378a92 (worktree temporal, ya eliminado; JSONs en /private/tmp/claude-501/-Users-javiergimenezmoya-developer-math/b0999919-11c4-479b-a988-69f2e2260f62/scratchpad/audit/json_pre/) y pasé el MISMO web/examples.csv. PRE: len(steps)=1056 (reproduce exacto el 1056 del informe docs/AUDITORIA_EDUCATIVA_2026-07-23.md:3), solve_steps=97, substeps=263. POST (HEAD 1dae38514): len(steps)=292, solve_steps=128, substeps=263. Delta = -764 steps. Solo 31 filas cambian; 25 de ellas van a 0 y suman -759 (81: 141->0, 71: 117->0, 67: 94->0, 80: 84->0, 75: 80->0, 85: 51->0, 65: 34->0, 74: 25->0, 88: 21->0, 209: 18->0, 89/90: 13->0, 62: 10->0, 79/76/208: 8->0, 64: 7->0, 190: 6->0, 61/63: 5->0, 189: 4->0, 70: 3->0, 68: 2->0, 83/158: 1->0). De esas 25, 11 recuperaron solve_steps (+31 en total); las otras 14 perdieron 366 pasos sin reemplazo alguno.

**Alcance.** 55 filas de familia solve/solve_system/dsolve en el corpus; 15 de ellas hoy no narran nada (14 nuevas + 84/86 que ya estaban mudas antes). En la web es el caso peor: el alumno recibe un intervalo o una familia periódica correcta y CERO explicación. Es la causa dominante de la categoría missing_narration (17) y contribuye a magic_step. Fuera del corpus afecta a toda inecuación no cubierta por los kernels E5.

**Código.** `crates/cas_solver/src/solve_command_eval_core/eval.rs:37`, `crates/cas_solver/src/solve_command_eval_core/eval.rs:52`, `crates/cas_engine/src/eval/actions.rs:395`, `crates/cas_engine/src/eval/actions.rs:406`, `crates/cas_solver/src/linear_system_command_eval/nonlinear.rs:174`

**Arreglo propuesto.** No revertir el aparcado (el confeti era real y su vuelta reabriría chain_discontinuity 506). Dos opciones acumulables: (a) piso de narración por defecto — cuando `solve` termina con `steps.is_empty() && solve_steps.is_empty()`, emitir un esqueleto mínimo derivado de datos que YA existen en el execution output (ecuación normalizada -> nombre de la estrategia usada (`strategy`) -> conjunto solución), en vez de devolver el canal vacío; (b) seguir la campaña E5 con las 14 familias nombradas arriba, empezando por las inecuaciones racionales/abs/log que son las más clásicas del curso. Añadir además un pin de contrato «ninguna fila de examples.csv con resultado ok puede tener steps+solve_steps == 0» para que la lane detecte el hueco en vez de dejarlo invisible.

_(confianza: high)_

---

### Las 85 filas de «exactamente 1 paso» NO son una regresión: es el diseño de verbo-contenedor, y `steps_count` no cuenta los substeps donde vive la narración

**Síntoma.** 93 filas con steps_count==1 (85 con len(steps)==1). Grupos enteros colapsados: Límites 15/15, Integrales 18/22, Teoría de números 9/9. La fila muestra un único salto input => resultado.

**Mecanismo.** Los verbos de cálculo emiten UN solo `Step` contenedor por construcción y meten la didáctica dentro de `substeps`. En límites es literal: `let mut steps = Vec::new(); ... steps.push(step);` — un único Step con importancia Medium y categoría Limits. Igual con «Symbolic Integration» -> «Calcular la integral» y con teoría de números. Los `substeps` NO se suman a `steps_count` (que es `self.steps.len()`), y además los `SolveStepWire`/substeps solo llevan campos `*_latex` (title/before_latex/after_latex), sin texto plano — así que cualquier detector o lector que mire `before`/`after` ve una fila vacía de contenido.

**Evidencia.** PRE tenía 86 filas con exactamente 1 paso, POST tiene 85: los conjuntos difieren en 3 elementos (salen 83 y 158, entra 152). Los ciclos de julio NO tocaron esto. Los substeps son IDÉNTICOS antes y después: 263 en ambos. Verificado en vivo con el binario de HEAD: `integrate(sin(x)*x^2,x)` -> steps_count 1 con 8 substeps que narran integración por partes repetida completa (elegir u y dv, calcular du y v, aplicar fórmula, x2, integrar el término restante); `limit(sin(x)/x,x,0)` -> 1 step con 2 substeps («La sustitución directa da la indeterminación 0/0», «Aplicar el límite notable»). El residual REAL son las 36 filas de las 85 cuyo contenedor tiene CERO substeps: Matrices 5/5, Complejo 3/3, sympy-wrong 3/3, Integrales 6, Teoría de números 4, Series 3, Analítico 3, Álgebra 3, Vectorial 2, Límites 2, Verificación 1, Funciones 1.

**Alcance.** 85 filas del corpus (40%). De ellas 49 narran bien pero la métrica no lo ve; 36 son magic_step auténtico. Es la explicación principal de la categoría magic_step: 68 y de la sensación de «narración insuficiente» que no viene de ningún filtro.

**Código.** `crates/cas_engine/src/eval/actions.rs:592`, `crates/cas_engine/src/eval/actions.rs:617`, `crates/cas_engine/src/eval/actions.rs:630`, `crates/cas_didactic/src/didactic/visible_rule_names.rs:194`, `crates/cas_solver/src/eval_output_finalize_input/types/shared.rs:38`

**Arreglo propuesto.** (1) Métrica: publicar `substeps_count` en el wire (o hacer que `steps_count` sume substeps) para dejar de medir 292 cuando hay 683 unidades de narración. (2) Contenido: dar a `SolveStepWire`/substep los campos `before`/`after` en texto plano igual que `StepWire` — hoy la mitad del corpus narra solo en LaTeX. (3) Cerrar las 36 filas con contenedor vacío empezando por los grupos con 100% de vacío (Matrices, Complejo): son verbos que ya conocen su algoritmo y solo necesitan el molde `focused_rule_substeps`.

_(confianza: high)_

---

### `steps_count` mezcla dos contratos: 16 filas con narración real reportan 0, y 24 filas reportan un número que no coincide con `len(steps)`

**Síntoma.** Filas 91-98 (sistemas), 181, 201-207 (dsolve) reportan `steps_count: 0` llevando 3-7 `solve_steps` cada una. En sentido contrario, 24 filas (58-78, 82, 87, 164, 165, 208) reportan steps_count 1-6 con el array `steps` VACÍO. La suma de steps_count del corpus (352) no es el número de pasos (292).

**Mecanismo.** El finalize elige el contador según la FORMA del resultado, no según los canales realmente poblados: `finalize_solution_set_output` usa `combined_steps_count()` (steps + solve_steps) mientras `finalize_bool_output`, `finalize_text_output` y el finalize de expresión usan `primary_steps_count()` (solo steps). Como dsolve devuelve una ecuación (Expr) y solve_system devuelve texto, su narración en `solve_steps` no entra en el contador; como las inecuaciones/ecuaciones devuelven SolutionSet, sus solve_steps sí entran y encima se serializan bajo otra clave, de modo que steps_count no describe ninguno de los dos arrays.

**Evidencia.** Medido sobre los 210 JSON de HEAD: sum(steps_count)=352, len(steps)=292, solve_steps=128 — los 60 de diferencia son los solve_steps de las filas SolutionSet contados dos veces. 24 filas con steps_count != len(steps) (todas de familia solve). 16 filas con steps_count 0 y solve_steps > 0 (91,92,93,94,95,96,97,98,181,201,202,203,204,205,206,207): de las 50 filas medidas como «0 pasos», 16 SÍ narran. Ejemplo comprobado: [201] dsolve separable, steps_count 0, con 5 solve_steps que identifican la EDO, separan variables, integran, despejan y verifican.

**Alcance.** Falsea la medida de partida: de las 50 filas «con 0 pasos», 16 son artefacto de contador, 12 son el grupo Complejo devolviendo el input sin evaluar (value domain real por defecto, ya reportado en la auditoría P3 #10), 4 son definiciones `:=` que nada tienen que narrar y 3 son fallos del runner stateless. Solo 15 son huecos de narración auténticos (los de la causa 1). También rompe el mensaje de wire «N step(s)» en crates/cas_api_models/src/wire.rs:276.

**Código.** `crates/cas_solver/src/eval_output_finalize_input/types/shared.rs:38`, `crates/cas_solver/src/eval_output_finalize_input/types/shared.rs:42`, `crates/cas_solver/src/eval_output_finalize_nonexpr.rs:29`, `crates/cas_solver/src/eval_output_finalize_nonexpr.rs:42`, `crates/cas_solver/src/eval_output_finalize_nonexpr.rs:55`, `crates/cas_solver/src/eval_output_finalize_expr.rs:50`

**Arreglo propuesto.** Un solo contrato: `steps_count == steps.len()` SIEMPRE, y añadir `solve_steps_count` (y `substeps_count`) como campos propios del wire. Eso obliga a un cambio en los pins que hoy afirman `steps_count == 4` para filas con `steps: []` (crates/cas_cli/tests/semantics_cli_contract_tests.rs y wire_smoke_tests.rs), pero elimina la ambigüedad de raíz. Mínimo alternativo si se quiere evitar el churn: usar `combined_steps_count()` en las tres rutas nonexpr/expr para que dsolve y sistemas dejen de reportar 0.

_(confianza: high)_

---

### El `retain(before != after)` de E1 opera sobre cadenas YA plegadas por display — mecanismo de sobre-frenada real, pero en este corpus solo eliminó 5 pasos y todos eran ruido

**Síntoma.** Sospecha de que el dedup/prune de E1 (d443ee002) hubiera colapsado la narración. Filas 15, 137, 152, 175, 186 pierden exactamente 1 paso cada una.

**Mecanismo.** E1 introdujo dos cambios que se componen: (a) todo estado intermedio se renderiza plegado+normalizado (`cleanup_symbolic_diff_after_for_display` + `normalize_expr_for_display`) tanto en `before` como en `after`; (b) inmediatamente después, el dedup descarta cualquier paso cuyo `before` renderizado sea igual al `after` renderizado. Un paso cuyo trabajo consiste precisamente en lo que el fold ya hace (aritmética de exponentes x^(2-1), ln(e), recíprocos anidados) sale con before==after y desaparece. El pliegue es GLOBAL, así que una reescritura local absorbida por el fold global también se pierde. Es un peligro estructural: la cantidad de pasos borrados crece con lo agresivo que sea el fold, sin ningún tope ni traza.

**Evidencia.** Diff PRE/POST paso a paso de las 5 filas afectadas: [015] cae «Presentar resultado de cálculo en forma compacta» que iba de (1/(2·√x))/(x+1) a 1/(2·√x·(x+1)) — es EXACTAMENTE el paso redundante que la auditoría denunció en su ejemplo [004]; [152] cae «Agrupar términos semejantes: 2·x^(2-1) + 2·y^(2-1) => 2·y^(2-1) + 2·x», artefacto de maquinaria; [175] cae un «Quitar paréntesis tras el signo menos» duplicado idéntico al anterior; [186] cae «Cancelar términos opuestos» con before==after literal; [137] el fold absorbe el «Combinar las constantes» que convertía x^(2-1) en x. Total: -5 pasos sobre 1056, es decir el 0.65% de la reducción. La hipótesis de que los filtros de E1 vaciaron la narración queda FALSADA.

**Alcance.** Hoy, 5 pasos del corpus. El riesgo latente es general (cualquier paso de aritmética de exponentes o de canonicalización que el fold adelante), y crece si se endurece el fold. La excepción de honestidad `w.rule.starts_with("Conservar")` es una comparación por prefijo de cadena en español que solo funciona porque el dedup corre ANTES de `localize_step_payloads`: si alguien reordena esas dos líneas, los pasos «Keep … residual» del modo inglés se borrarían y se perdería el contrato de honestidad.

**Código.** `crates/cas_didactic/src/step_payloads.rs:55`, `crates/cas_didactic/src/step_payloads.rs:56`, `crates/cas_didactic/src/step_payloads/build/expr.rs:52`, `crates/cas_didactic/src/step_payloads/build/expr.rs:93`, `crates/cas_didactic/src/step_payloads/prepare.rs:42`, `crates/cas_didactic/src/step_payloads.rs:122`

**Arreglo propuesto.** Que el `retain` decida sobre la IDENTIDAD SEMÁNTICA del paso (los ExprId before/after antes del fold, que ya se comparan en prepare.rs:88-100 vía `should_drop_semantically_noop_step_payload`) y no sobre la cadena renderizada post-fold: si el paso cambió el árbol pero el fold lo oculta, la respuesta correcta es no plegar ese paso, no borrarlo. Y sustituir el `starts_with("Conservar")` por el `RULE_CONSERVAR_*` de cas_solver_core::rule_names (ya importado en el mismo fichero, línea 10), que es inmune a la localización y al reordenamiento de fases.

_(confianza: high)_

---

### `is_first` se decide al construir y el dedup posterior renumera: el paso promovido a índice 1 pierde el eco verbatim de la entrada (latente)

**Síntoma.** Riesgo de que la cadena visible arranque en una forma plegada que el alumno nunca tecleó — el patrón que la auditoría clasificó como chain_discontinuity/magic_step en el primer paso.

**Mecanismo.** `build_step_wire` decide `is_first = index == 1` con `index = payloads.len() + 1` en el momento de construir, y solo el primer paso conserva el `before` sin plegar («Step 0 keeps the user's input echo verbatim»). Pero `dedup_consecutive_step_payloads` corre DESPUÉS de todo el bucle de construcción y puede eliminar ese primer paso (por before==after) para luego reasignar `wire.index = i + 1`. El paso que queda en el índice 1 se construyó con `is_first = false`, o sea con el `before` plegado y normalizado: la cadena empieza en un estado que no es la entrada.

**Evidencia.** El orden de las llamadas es inequívoco en el código (construcción completa en `collect_step_payloads_inner`, dedup y renumerado después, en la entrada compartida). En el corpus la trampa apenas se dispara: solo 1 fila tiene un primer paso con before==after (PRE [158] `(3+4i)/(1-2i)`, «Quitar paréntesis tras el signo menos», before==after) y era su único paso, así que la fila fue a 0 en vez de promover a otro. En POST no queda ninguna fila con primer paso before==after.

**Alcance.** 0 filas hoy; se activa en cuanto una expresión tenga un primer paso noop-tras-fold seguido de más pasos. Como `is_noop_wire_step` (step_payloads.rs:122) solo cubre 5 nombres de regla mientras el `retain` cubre TODAS, la puerta está abierta para cualquier regla nueva.

**Código.** `crates/cas_didactic/src/step_payloads.rs:41`, `crates/cas_didactic/src/step_payloads.rs:57`, `crates/cas_didactic/src/step_payloads/build.rs:47`, `crates/cas_didactic/src/step_payloads/build/expr.rs:49`

**Arreglo propuesto.** Decidir el pliegue por posición FINAL, no por posición de construcción: correr dedup/prune primero sobre los EnrichedStep (o sobre wires sin renderizar) y renderizar después, o re-renderizar el `before` del paso que quede en índice 1 con la política verbatim tras el renumerado en step_payloads.rs:57-59. Añadir un pin: «el `before` del paso 1 coincide con `input_latex`» sobre unas cuantas filas del corpus.

_(confianza: medium)_

---

## Apéndice A — los 53 hallazgos P0

| fila | expresión | categoría | ubicación | lo que se muestra |
|---|---|---|---|---|
| 1 | `(ln(x*sqrt(x)) + ln(sqrt(x)/x^2)) + (sqrt(y)/(sqrt(y` | latex_render_bug | steps[index=3].rule_latex (rule 'Sacar un exponente fuera de | {\color{red}{\ln({x}^{-\frac{3}{2}})}} \rightarrow {\color{green}{\frac{3}{2}\cdot \ln(x)}}  →  se renderiza como  ln(x^(-3/2)) →  |
| 9 | `(sqrt(5 + 2*sqrt(6))) + (1/(u*(u+2))) - ((sqrt(2) + ` | latex_render_bug | steps[3] ('Sumar fracciones'), campo before_latex: el estado | steps[3] ('Sumar fracciones'), campo before_latex: el estado se renderiza con VALOR DISTINTO del real. Reproducido en vivo: before |
| 11 | `diff(arctan((1+x)/(1-x)), x)` | anti_pedagogical | steps[3].substeps[0] ('Reescribir el denominador sacando fac | before_latex "{(\frac{x + 1}{1 - x})}^{2} + 1" → after_latex "\left({(1 - x)}^{2}\right)\cdot \left(\frac{{(x + 1)}^{2}}{{(1 - x)} |
| 23 | `integrate(1/(x^4-1), x, 2, oo)` | anti_pedagogical | steps[1].substeps[1] ('Evaluar la antiderivada en los límite | after_latex = "\lim_{x \to \infty} \frac{\ln(/x - 1/)}{4} - \frac{1}{2}\cdot \arctan(x) - \frac{1}{4}\cdot \ln(/x + 1/) - \frac{\l |
| 23 | `integrate(1/(x^4-1), x, 2, oo)` | substep_wrong_math | paso 1 «Calcular la integral» → sub-paso 2 «Evaluar la antid | \lim_{x \to \infty} \frac{\ln(/x - 1/)}{4} - \frac{1}{2}\cdot \arctan(x) - \frac{1}{4}\cdot \ln(/x + 1/) - \frac{\ln(/2 - 1/)}{4}  |
| 25 | `integrate(e^x*sin(x), x)` | latex_render_bug | steps[index=2].rule_latex ('Expandir la expresión') y steps[ | paso 2: {\color{red}{\sin(x) - \cos(x)}} \rightarrow {\color{green}{\sin(x)\cdot {e}^{x} - \cos(x)\cdot {e}^{x}}}   /   paso 3: {\ |
| 26 | `integrate(x^2 * sin(x), x)` | substep_wrong_math | paso 1 «Calcular la integral» → sub-paso 8 «Integrar el térm | \int -\sin(x)\cdot 2\,dx  ->  2\cdot x\cdot \sin(x) + (2 - {x}^{2})\cdot \cos(x) |
| 32 | `integrate(1/(x^3-2), x)` | wrong_rule_name | steps[0].substeps[1] ("Descomponer en fracciones parciales") | before_latex "{x}^{3} - 2" → after_latex "\frac{1}{{x}^{3} - 2}" |
| 33 | `integrate(1/(x^4-5), x)` | wrong_rule_name | steps[0].substeps[1] ("Descomponer en fracciones parciales") | before_latex "{x}^{4} - 5" → after_latex "\frac{1}{{x}^{4} - 5}" |
| 40 | `integrate(1/(x^3-2), x, 2, 3)` | other | steps[0].substeps[1] ("Evaluar la antiderivada en los límite | ...\frac{\ln(/3 - \text{cbrt}(2)/)}{6}\cdot \text{cbrt}(2) - -\frac{1}{6}\cdot \arctan(\frac{\text{cbrt}(2) + 2\cdot 2}{\sqrt{3}\c |
| 40 | `integrate(1/(x^3-2), x, 2, 3)` | substep_wrong_math | paso 1 «Calcular la integral» → sub-paso 2 «Evaluar la antid | ... + \frac{\ln(/3 - \text{cbrt}(2)/)}{6}\cdot \text{cbrt}(2) - -\frac{1}{6}\cdot \arctan(...)\cdot \sqrt{3}\cdot \text{cbrt}(2) - |
| 42 | `approx(integrate(1/(x^3-2), x, 2, 3))` | other | steps[0].substeps[1] ("Evaluar la antiderivada en los límite | ... - -\frac{1}{6}\cdot \arctan(\frac{\text{cbrt}(2) + 2\cdot 2}{\sqrt{3}\cdot \text{cbrt}(2)})\cdot \sqrt{3}\cdot \text{cbrt}(2)  |
| 42 | `approx(integrate(1/(x^3-2), x, 2, 3))` | substep_wrong_math | paso 1 «Calcular la integral» → sub-paso 2 «Evaluar la antid | ... \frac{\ln(/3 - \text{cbrt}(2)/)}{6}\cdot \text{cbrt}(2) - -\frac{1}{6}\cdot \arctan(\frac{\text{cbrt}(2) + 2\cdot 2}{\sqrt{3}\ |
| 72 | `solve(x^2=1-sqrt(2),x)` | anti_pedagogical | solve_steps[0].substeps[5] y [6] (índices 1.6 y 1.7) | 1.6 "Tomar raíz cuadrada en ambos lados" → equation "/x/ = (1 - sqrt(2))^(1/2)", rhs_latex "\\sqrt{1 - \\sqrt{2}}"; 1.7 "/u/ = a s |
| 72 | `solve(x^2=1-sqrt(2),x)` | substep_wrong_math | solve_steps[0].substeps 1.6 y 1.7 | 1.6 «Tomar raíz cuadrada en ambos lados» ⇒ /x/ = \sqrt{1 - \sqrt{2}} ; 1.7 «/u/ = a se descompone en u = a y u = -a. Despejando x  |
| 78 | `solve((x-1)*(x-2)*(x-3)>0,x)` | wrong_rule_name | solve_steps[0] y solve_steps[2] | {"index":1,"description":"Mueve los términos a un lado","equation":"x > 3"} … {"index":3,"description":"Mueve los términos a un la |
| 103 | `product(1 - 1/k^2, k, 2, n)` | latex_render_bug | steps[2].substeps[1].after_latex y substeps[2].before_latex | "\\frac{2 - 1\\cdot n + 1}{2\\cdot n}" — sin paréntesis se lee (2 - n + 1)/(2n) = (3 - n)/(2n), que es FALSO; y el substep siguien |
| 103 | `product(1 - 1/k^2, k, 2, n)` | substep_wrong_math | steps[1] (paso 2 «Evaluar producto telescópico finito») → su | \frac{2 - 1\cdot n + 1}{2\cdot n} |
| 104 | `taylor(sin(x), x, 0, 5)` | highlight_red_equals_green | steps[3] | rule_latex = "{\\color{red}{x}} \\rightarrow {\\color{green}{x}}": se resalta el sumando x, que es exactamente el único término qu |
| 104 | `taylor(sin(x), x, 0, 5)` | wrong_rule_name | steps[3] | rule = "Repartir el denominador entre los sumandos" (interno "Distribute Division Into Sum", crates/cas_didactic/src/didactic/visi |
| 126 | `cross([1,0,0],[0,1,0])` | anti_pedagogical | steps[1] | El paso va HACIA ATRÁS: before="[[0], [0], [1]]" → after="[[0], [0], [1 - 0^2]]" (introduce un 0^2 inexistente) y además el último |
| 126 | `cross([1,0,0],[0,1,0])` | wrong_rule_name | steps[1] | rule="Sumar exponentes de la misma base" para una transformación que es 0 → 0^2 (rule_latex="{\color{red}{0}} \rightarrow {\color{ |
| 136 | `diff(integrate(2*x/sqrt(4+x^4)+1, x), x) - (2*x/sqrt` | highlight_wrong_subexpression | steps[1].rule_latex | "{\color{red}{\frac{d}{dx}(x + \operatorname{asinh}(\frac{{x}^{2}}{2}))}} \rightarrow {\color{green}{\frac{\frac{2}{2}\cdot x}{\sq |
| 136 | `diff(integrate(2*x/sqrt(4+x^4)+1, x), x) - (2*x/sqrt` | highlight_wrong_subexpression | steps[3].rule_latex | «Sumar fracciones»: "{\color{red}{\frac{x}{\sqrt{1 + \frac{{{x}^{2}}^{2}}{4}}}}} \rightarrow {\color{green}{\frac{4 + {x}^{4}}{4}} |
| 137 | `equiv(diff(integrate(2*x/sqrt(4+x^4)+1, x), x), 2*x/` | highlight_wrong_subexpression | steps[1].rule_latex | "{\color{red}{\frac{d}{dx}(x + \operatorname{asinh}(\frac{{x}^{2}}{2}))}} \rightarrow {\color{green}{\frac{\frac{2}{2}\cdot x}{\sq |
| 137 | `equiv(diff(integrate(2*x/sqrt(4+x^4)+1, x), x), 2*x/` | highlight_wrong_subexpression | steps[5].rule_latex | «Sumar fracciones»: "{\color{red}{\frac{x}{\sqrt{1 + \frac{{{x}^{2}}^{2}}{4}}}}} \rightarrow {\color{green}{\frac{4 + {x}^{4}}{4}} |
| 140 | `equiv((x^3+y^3)/(x+y), x^2-x*y+y^2)` | highlight_wrong_subexpression | steps[1].rule_latex | "{\color{red}{{x}^{2} - x\cdot y}} \rightarrow {\color{green}{\text{expand}({x}^{2} + {y}^{2} - x\cdot y - {x}^{2} - {y}^{2} + x\c |
| 142 | `diff(integrate(x*e^x, x), x) - x*e^x` | text_latex_divergence | steps[0] | after (texto) = "(x - 1)·e^x" — pierde el operador d/dx y el término −x·e^x — mientras after_latex="\frac{d}{dx}({\color{green}{{e |
| 142 | `diff(integrate(x*e^x, x), x) - x*e^x` | highlight_wrong_subexpression | steps[1].rule_latex | "{\color{red}{\frac{d}{dx}({e}^{x}\cdot (x - 1))}} \rightarrow {\color{green}{{e}^{x} + {e}^{x}\cdot (x - 1) - x\cdot {e}^{x}}}":  |
| 142 | `diff(integrate(x*e^x, x), x) - x*e^x` | wrong_rule_name | steps[1].substeps[0] | title="Usar regla de la cadena" con before_latex="(x - 1)\cdot {e}^{x}" y after_latex="{e}^{x} + (x - 1)\cdot {e}^{x}\cdot \ln(e)" |
| 144 | `integrate(diff(x^3, x), x) - x^3` | highlight_wrong_subexpression | steps[0].rule_latex | "{\color{red}{\frac{d}{dx}({x}^{3})}} \rightarrow {\color{green}{\int 3\cdot {x}^{2} \, dx - {x}^{3}}}": la regla afirma que d/dx( |
| 151 | `det(hessian(x^2*y, [x,y]))` | highlight_wrong_subexpression | steps[2] | rule_latex: {\color{red}{{x}^{0}}} \rightarrow {\color{green}{1}} — pero el verde se ancla en after_latex sobre el exponente de OT |
| 153 | `laplacian(ln(x^2+y^2), [x,y])` | highlight_red_equals_green | steps[2] | rule_latex: {\color{red}{1}} \rightarrow {\color{green}{1}} — y el ancla está dentro de 2\cdot 2\cdot {x}^{2 - {\color{red}{1}}}\c |
| 153 | `laplacian(ln(x^2+y^2), [x,y])` | highlight_red_equals_green | steps[4] | rule_latex: {\color{red}{{y}^{2}}} \rightarrow {\color{green}{{y}^{2}}} — anclado en el DENOMINADOR {({x}^{2} + {\color{red}{{y}^{ |
| 153 | `laplacian(ln(x^2+y^2), [x,y])` | highlight_wrong_subexpression | steps[5] y steps[6] | steps[5].rule_latex = {\color{red}{2\cdot {x}^{2} + 2\cdot {y}^{2} - 4\cdot {x}^{2}}} \rightarrow {\color{green}{2\cdot {x}^{2}}}  |
| 153 | `laplacian(ln(x^2+y^2), [x,y])` | latex_render_bug | steps[index=5].rule_latex y steps[index=6].rule_latex (regla | paso 5: {\color{red}{2\cdot {x}^{2} + 2\cdot {y}^{2} - 4\cdot {x}^{2}}} \rightarrow {\color{green}{2\cdot {x}^{2}}}   /   paso 6:  |
| 170 | `i^i` | highlight_wrong_subexpression | i^i con --value-domain complex (la fila solo evalúa con el s | i^i con --value-domain complex (la fila solo evalúa con el selector Complejo). steps[2].rule_latex = '{\\color{red}{\\ln(i)}} \\ri |
| 175 | `equiv(e^(i*pi), -1)` | substep_noop_or_false_claim | output.result + steps[1..3] (grupo Complejo, flags por defec | result: "false"; warnings: []; equivalence_diagnostics.residual: "1 + e^(pi·i)"; los 3 pasos son: 1) "Quitar paréntesis tras el si |
| 176 | `hessian(x^2*y, [x,y])` | highlight_wrong_subexpression | steps[2] | rule_latex: {\color{red}{{x}^{0}}} \rightarrow {\color{green}{1}} ; after_latex: \begin{bmatrix} 2\cdot y & 2\cdot {x}^{2 - {\colo |
| 178 | `diff([x^2, x^3], x, 2)` | highlight_wrong_subexpression | steps[2] | rule_latex: {\color{red}{\frac{d}{dx}(\begin{bmatrix}{x}^{2} \\ {x}^{3}\end{bmatrix})}} \rightarrow {\color{green}{\frac{d}{dx}(\b |
| 179 | `subs(subs(diff(x^2*y,x), x, 1), y, 2)` | highlight_wrong_subexpression | steps[1] | rule_latex: {\color{red}{\frac{\partial}{\partial x}(y\cdot {x}^{2})}} \rightarrow {\color{green}{\text{subs}(\text{subs}(2\cdot x |
| 180 | `subs(subs(det(hessian(x^3+y^3-3*x*y,[x,y])), x, 1), ` | highlight_wrong_subexpression | steps[3] | rule_latex: {\color{red}{3\cdot -3}} \rightarrow {\color{green}{{(-3)}^{2}}} |
| 180 | `subs(subs(det(hessian(x^3+y^3-3*x*y,[x,y])), x, 1), ` | latex_render_bug | steps[index=3].rule_latex (regla 'Sumar exponentes de la mis | {\color{red}{3\cdot -3}} \rightarrow {\color{green}{{(-3)}^{2}}}   →   se renderiza como   3·−3 → (−3)² |
| 182 | `taylor(sin(x)/x, x, 0, 4)` | highlight_red_equals_green | steps[3].rule_latex | {\color{red}{1}} \rightarrow {\color{green}{1}} — se resalta el sumando «1», que es exactamente lo único que NO cambia |
| 182 | `taylor(sin(x)/x, x, 0, 4)` | wrong_rule_name | steps[3].rule | «Repartir el denominador entre los sumandos» |
| 185 | `lineintegral(x^2, [x,y], [cos(t),sin(t)], t, 0, pi)` | highlight_wrong_subexpression | steps[6].after_latex | after_latex '\frac{\pi}{2} - \frac{\sin({\color{green}{0}})}{4} - \frac{0}{2}' — el verde marca el argumento 0 del sin(0) que SOBR |
| 185 | `lineintegral(x^2, [x,y], [cos(t),sin(t)], t, 0, pi)` | highlight_wrong_subexpression | steps[7].after_latex | after_latex '\frac{\pi}{2} - \frac{{\color{green}{0}}}{2}' — el verde cae sobre el 0 del término 0/2, que ya estaba ahí antes del  |
| 185 | `lineintegral(x^2, [x,y], [cos(t),sin(t)], t, 0, pi)` | substep_wrong_math | paso 5 'Calcular la integral' → substep 2 'Evaluar la antide | before_latex '\frac{\sin(2\cdot t)}{4} + \frac{t}{2}' → after_latex '\frac{\sin(2\cdot \pi)}{4} + \frac{\pi}{2} - \frac{\sin(0\cdo |
| 192 | `surface_integral([x,y,0], [x,y,z], [cos(u),sin(u),v]` | highlight_wrong_subexpression | steps[2].rule_latex | '{\color{red}{\sin(u)}} \rightarrow {\color{green}{{0}^{2}}}' — afirma que sin(u) se transforma en 0^2 |
| 192 | `surface_integral([x,y,0], [x,y,z], [cos(u),sin(u),v]` | highlight_red_equals_green | steps[3].rule_latex | '{\color{red}{\sin(u)}} \rightarrow {\color{green}{\sin(u)}}' — regla x → x sobre el factor que no cambia |
| 192 | `surface_integral([x,y,0], [x,y,z], [cos(u),sin(u),v]` | other | paso 2 'Sumar exponentes de la misma base' (rule_latex) — y  | paso 2: rule_latex '{\color{red}{\sin(u)}} \rightarrow {\color{green}{{0}^{2}}}'; paso 3: rule_latex '{\color{red}{\sin(u)}} \righ |
| 192 | `surface_integral([x,y,0], [x,y,z], [cos(u),sin(u),v]` | latex_render_bug | steps[index=2].rule_latex (regla 'Sumar exponentes de la mis | {\color{red}{\sin(u)}} \rightarrow {\color{green}{{0}^{2}}} |
| 199 | `limit(exp(z), z, i*pi)` | substep_wrong_math | paso 1 'Conservar límite residual' → substep 1 (único) | title 'La política segura no decide este límite. Para investigarlo, calcula los límites laterales en z = pi·i (por la izquierda y  |

---

## Apéndice B — incidencia por fila (185 filas con al menos un hallazgo)

`fila` = índice 0-based del corpus; `csv` = línea de `web/examples.csv`.

| fila | csv | grupo | expresión | severidades | categorías |
|---|---|---|---|---|---|
| 1 | 3 | Álgebra y simplificación | `(ln(x*sqrt(x)) + ln(sqrt(x)/x^2)) + (sqrt(y)/(sqrt(y)-1) - ...` | P0×1 P1×2 P2×4 P3×1 | chain_discontinuity, latex_render_bug, magic_step, substep_chain_break, substep_generic_or_empty |
| 2 | 4 | Álgebra y simplificación | `1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)` | P2×1 | chain_discontinuity |
| 4 | 6 | Álgebra y simplificación | `factor(x^6-1)` | P1×1 | magic_step |
| 5 | 7 | Álgebra y simplificación | `apart(x^3/(x^2-1))` | P1×1 | magic_step |
| 6 | 8 | Álgebra y simplificación | `sqrt(5+2*sqrt(6))` | P1×1 P2×1 P3×1 | chain_discontinuity, latex_render_bug, magic_step |
| 7 | 9 | Álgebra y simplificación | `simplify((e^x - e^(-x))/(e^x + e^(-x)))` | P2×1 P3×1 | highlight_stale_form, rule_name_misleading |
| 9 | 11 | Álgebra y simplificación | `(sqrt(5 + 2*sqrt(6))) + (1/(u*(u+2))) - ((sqrt(2) + sqrt(3)...` | P0×1 P2×2 | chain_discontinuity, latex_render_bug, substep_duplicates_parent |
| 10 | 12 | Álgebra y simplificación | `approx(e^pi)` | P1×2 | latex_render_bug |
| 11 | 13 | Derivadas | `diff(arctan((1+x)/(1-x)), x)` | P0×1 | anti_pedagogical |
| 12 | 14 | Derivadas | `diff(x^x, x)` | P1×1 P2×1 | magic_step, substep_duplicates_parent |
| 13 | 15 | Derivadas | `diff(sin(e^(x^2)), x)` | P1×1 P2×1 | anti_pedagogical, substep_noop_or_false_claim |
| 14 | 16 | Derivadas | `diff(ln((x-1)/(x+1)), x)` | P1×1 P2×1 | chain_discontinuity, rule_name_misleading |
| 15 | 17 | Derivadas | `diff(arctan(sqrt(x)), x)` | P2×2 | highlight_wrong_subexpression, substep_generic_or_empty |
| 16 | 18 | Derivadas | `diff(sin(x)/x, x)` | P2×1 | substep_duplicates_parent |
| 17 | 19 | Derivadas | `diff(x^2*y, x, y)` | P2×1 | substep_chain_break |
| 18 | 20 | Derivadas | `diff(sin(x), x, 4)` | P2×1 | substep_duplicates_parent |
| 19 | 21 | Derivadas | `diff(tanh(x), x)` | P2×1 | substep_duplicates_parent |
| 20 | 22 | Derivadas | `diff(x^3 + 2*x^2 - 5*x + 1, x)` | P1×1 P2×2 | anti_pedagogical, chain_discontinuity, rule_name_misleading |
| 21 | 23 | Integrales | `integrate(e^(-x^2), x, -oo, oo)` | P1×1 P3×2 | anti_pedagogical, magic_step, text_latex_divergence |
| 22 | 24 | Integrales | `integrate(1/(x^2+1), x, 0, oo)` | P2×1 | magic_step |
| 23 | 25 | Integrales | `integrate(1/(x^4-1), x, 2, oo)` | P0×2 P2×1 | anti_pedagogical, substep_wrong_math |
| 25 | 27 | Integrales | `integrate(e^x*sin(x), x)` | P0×1 P2×1 | latex_render_bug, noop_or_trivial_step |
| 26 | 28 | Integrales | `integrate(x^2 * sin(x), x)` | P0×1 P1×1 | anti_pedagogical, substep_wrong_math |
| 27 | 29 | Integrales | `integrate((2*x+3)/(x^2+x+1), x)` | P2×3 | magic_step, rule_name_misleading, wrong_rule_name |
| 28 | 30 | Integrales | `integrate(2*x/sqrt(4+x^4)+1, x)` | P1×1 P3×1 | magic_step, other |
| 29 | 31 | Integrales | `integrate(x*cos(x^2), x)` | P2×1 P3×1 | latex_render_bug, substep_duplicates_parent |
| 30 | 32 | Integrales | `integrate(x^2, x)` | P2×1 P3×2 | latex_render_bug, other, text_latex_divergence |
| 31 | 33 | Integrales | `integrate(1/(x^5-1), x)` | P1×3 P2×2 P3×2 | chain_discontinuity, magic_step, noop_or_trivial_step, other, substep_wrong_math, text_latex_divergence |
| 32 | 34 | Integrales | `integrate(1/(x^3-2), x)` | P0×1 P1×2 P2×2 | latex_render_bug, magic_step, noop_or_trivial_step, substep_noop_or_false_claim, wrong_rule_name |
| 33 | 35 | Integrales | `integrate(1/(x^4-5), x)` | P0×1 P1×2 P2×1 P3×1 | anti_pedagogical, magic_step, noop_or_trivial_step, substep_noop_or_false_claim, wrong_rule_name |
| 34 | 36 | Integrales | `integrate(1/(x^5-x-1), x)` | P1×2 P2×2 | highlight_stale_form, magic_step, text_latex_divergence |
| 35 | 37 | Integrales | `integrate(1/(x^7-1), x)` | P1×2 | magic_step |
| 36 | 38 | Integrales | `integrate(1/(x^5-2), x)` | P1×2 | magic_step |
| 37 | 39 | Integrales | `integrate(1/(x^5-x-1), x, 2, 3)` | P2×2 | highlight_stale_form, magic_step |
| 38 | 40 | Integrales | `integrate(1/(x^3-x-1), x, 1, 2)` | P1×1 P2×1 | anti_pedagogical, missing_narration |
| 39 | 41 | Integrales | `integrate(x/(x^4+x+1), x)` | P1×2 | magic_step |
| 40 | 42 | Integrales | `integrate(1/(x^3-2), x, 2, 3)` | P0×2 P2×1 | anti_pedagogical, other, substep_wrong_math |
| 41 | 43 | Integrales | `integrate(1/(x^4-5), x, 1, 2)` | P1×1 P2×2 | anti_pedagogical, missing_narration |
| 42 | 44 | Integrales | `approx(integrate(1/(x^3-2), x, 2, 3))` | P0×2 P1×1 | latex_render_bug, other, substep_wrong_math |
| 43 | 45 | Límites | `limit((1+1/n)^n, n, infinity)` | P3×2 | substep_chain_break, text_latex_divergence |
| 44 | 46 | Límites | `limit(sin(x)/x, x, 0)` | P2×1 P3×1 | anti_pedagogical, substep_duplicates_parent |
| 47 | 49 | Límites | `limit(sqrt(x^2+x)-x, x, infinity)` | P2×1 | substep_math_missing |
| 48 | 50 | Límites | `limit((1+2/x)^x, x, infinity)` | P1×3 | magic_step |
| 50 | 52 | Límites | `limit(x/ln(1-x), x, -infinity)` | P1×3 P3×1 | magic_step, text_latex_divergence |
| 54 | 56 | Límites | `limit(sqrt(x)*exp(-x), x, infinity)` | P1×2 | latex_render_bug |
| 56 | 58 | Límites | `limit(1/x-1/sin(x), x, 0)` | P1×1 P2×1 | magic_step |
| 57 | 59 | Límites | `limit(1/tan(x)^2-1/x^2, x, 0)` | P1×2 | magic_step |
| 58 | 60 | Ecuaciones | `solve(a*x^2+b*x+c=0,x)` | P2×2 P3×1 | anti_pedagogical, chain_discontinuity, substep_noop_or_false_claim |
| 59 | 61 | Ecuaciones | `solve(x^3-6*x^2+11*x-6=0,x)` | P1×2 P3×1 | anti_pedagogical, magic_step |
| 60 | 62 | Ecuaciones | `solve(x^4-5*x^2+4=0,x)` | P1×1 | magic_step |
| 61 | 63 | Ecuaciones | `solve(sin(x)=1/2,x)` | P2×1 | latex_render_bug |
| 62 | 64 | Ecuaciones | `solve(cos(2*x)=1/2,x)` | P1×1 | missing_narration |
| 63 | 65 | Ecuaciones | `solve(2*cos(x)-sqrt(3)=0,x)` | P2×1 | latex_render_bug |
| 64 | 66 | Ecuaciones | `solve(sin(x)=cos(x),x)` | P3×1 | other |
| 65 | 67 | Ecuaciones | `solve(sin(x)+sqrt(3)*cos(x)=1,x)` | P1×1 P2×2 | anti_pedagogical, latex_render_bug, magic_step |
| 66 | 68 | Ecuaciones | `solve(e^(2*x)-3*e^x+2=0,x)` | P1×1 P2×1 | anti_pedagogical, magic_step |
| 67 | 69 | Ecuaciones | `solve(e^x+e^(-x)=4,x)` | P1×1 P2×1 | anti_pedagogical, magic_step |
| 68 | 70 | Ecuaciones | `solve(4^x-9^x=0,x)` | P1×2 | missing_narration |
| 69 | 71 | Ecuaciones | `solve(ln(x)+ln(x-3)=1,x)` | P2×4 P3×1 | magic_step, missing_narration, noop_or_trivial_step, text_latex_divergence |
| 70 | 72 | Ecuaciones | `solve(sqrt(x+5)-sqrt(x)=1,x)` | P1×2 | missing_narration |
| 71 | 73 | Ecuaciones | `solve(abs(x^2-1)=x+1,x)` | P1×1 P3×1 | magic_step, other |
| 72 | 74 | Ecuaciones | `solve(x^2=1-sqrt(2),x)` | P0×2 P2×2 P3×1 | anti_pedagogical, noop_or_trivial_step, substep_noop_or_false_claim, substep_wrong_math |
| 73 | 75 | Ecuaciones | `solve(Q = Q0 * 2^(-t/T),t)` | P1×2 P2×1 | anti_pedagogical, chain_discontinuity, language_leak |
| 74 | 76 | Ecuaciones | `solve(tan(x)=tan(2*x), x)` | P2×2 | magic_step, noop_or_trivial_step |
| 75 | 77 | Ecuaciones | `solve(x*abs(x)=4, x)` | P1×1 | magic_step |
| 76 | 78 | Ecuaciones | `solve(cot(x)^2=3, x)` | P1×1 | magic_step |
| 77 | 79 | Inecuaciones | `solve(x^2-2*x-3>0,x)` | P1×2 P2×2 | magic_step, missing_narration, noop_or_trivial_step, substep_noop_or_false_claim |
| 78 | 80 | Inecuaciones | `solve((x-1)*(x-2)*(x-3)>0,x)` | P0×1 P1×2 P2×4 | chain_discontinuity, duplicate_or_burst, other, rule_name_misleading, substep_noop_or_false_claim, wrong_rule_name |
| 79 | 81 | Inecuaciones | `solve((x-1)/(x-2)<0,x)` | P1×2 | missing_narration |
| 80 | 82 | Inecuaciones | `solve(x+1/x>2,x)` | P1×2 | missing_narration |
| 81 | 83 | Inecuaciones | `solve((x+1)/(x-1)>=2,x)` | P1×2 | missing_narration |
| 82 | 84 | Inecuaciones | `solve(1/(x-sqrt(2))>0,x)` | P3×1 | anti_pedagogical |
| 83 | 85 | Inecuaciones | `solve(sqrt(x)>2,x)` | P1×1 P2×1 | missing_narration |
| 84 | 86 | Inecuaciones | `solve(x^(2/3)>2,x)` | P1×2 P3×2 | magic_step, missing_narration, other, text_latex_divergence |
| 85 | 87 | Inecuaciones | `solve(abs(x-2)>1,x)` | P1×2 | missing_narration |
| 86 | 88 | Inecuaciones | `solve(abs(x-1)<abs(x+2),x)` | P1×2 | magic_step, missing_narration |
| 87 | 89 | Inecuaciones | `solve(e^(2*x)-3*e^x+2<0,x)` | P1×1 P2×1 P3×1 | magic_step, noop_or_trivial_step, wrong_rule_name |
| 88 | 90 | Inecuaciones | `solve(ln(x)^2-3*ln(x)+2<0,x)` | P1×2 | missing_narration |
| 89 | 91 | Inecuaciones | `solve(2*sin(3*x)-1>0, x)` | P1×2 P3×2 | latex_render_bug, missing_narration, other |
| 90 | 92 | Inecuaciones | `solve(abs(x-a)<2, x)` | P1×1 P2×1 | missing_narration |
| 91 | 93 | Sistemas de ecuaciones | `solve_system(x+y=3; x-y=1; x; y)` | P1×1 P2×1 | duplicate_or_burst, magic_step |
| 92 | 94 | Sistemas de ecuaciones | `solve_system(2*x+3*y=7; 3*x-y=5; x; y)` | P1×1 P2×1 | duplicate_or_burst, magic_step |
| 93 | 95 | Sistemas de ecuaciones | `solve_system(x+y+z=6; x-y+z=2; 2*x+y-z=1; x; y; z)` | P1×1 P2×1 | duplicate_or_burst, magic_step |
| 94 | 96 | Sistemas de ecuaciones | `solve_system(a+b+c=1; a-b+c=3; a+b-c=-1; a; b; c)` | P1×1 P2×1 | duplicate_or_burst, magic_step |
| 95 | 97 | Sistemas de ecuaciones | `solve([x+y=3, x-y=1], [x, y])` | P1×1 P2×1 | duplicate_or_burst, magic_step |
| 96 | 98 | Sistemas de ecuaciones | `solve([a*x+y=1, x-y=0], [x, y])` | P1×1 P2×1 | duplicate_or_burst, magic_step |
| 97 | 99 | Sistemas de ecuaciones | `solve([x^2+y^2=25, x+y=7], [x, y])` | P2×3 | anti_pedagogical, duplicate_or_burst, magic_step |
| 98 | 100 | Sistemas de ecuaciones | `solve([x*y=6, x+y=5], [x, y])` | P2×3 | anti_pedagogical, duplicate_or_burst, magic_step |
| 99 | 101 | Series y sumatorios | `sum(1/n^2, n, 1, oo)` | P1×4 P2×3 | latex_render_bug, magic_step, other, rule_name_misleading, wrong_rule_name |
| 100 | 102 | Series y sumatorios | `sum(k^2, k, 1, n)` | P2×1 P3×4 | anti_pedagogical, latex_render_bug, other, text_latex_divergence |
| 101 | 103 | Series y sumatorios | `sum(1/(n*(n+1)), n, 1, oo)` | P1×2 P2×4 | anti_pedagogical, highlight_wrong_subexpression, latex_render_bug, rule_name_misleading, substep_wrong_math, wrong_rule_name |
| 102 | 104 | Series y sumatorios | `sum(1/n, n, 1, oo)` | P1×3 P2×2 P3×1 | latex_render_bug, magic_step, rule_name_misleading, text_latex_divergence, wrong_rule_name |
| 103 | 105 | Series y sumatorios | `product(1 - 1/k^2, k, 2, n)` | P0×2 P1×2 P2×4 P3×1 | anti_pedagogical, highlight_stale_form, latex_render_bug, other, substep_noop_or_false_claim, substep_wrong_math, text_latex_divergence |
| 104 | 106 | Series y sumatorios | `taylor(sin(x), x, 0, 5)` | P0×2 P1×5 | anti_pedagogical, highlight_red_equals_green, latex_render_bug, magic_step, rule_name_misleading, wrong_rule_name |
| 105 | 107 | Series y sumatorios | `sum(k^3, k, 1, n)` | P2×1 P3×1 | anti_pedagogical, latex_render_bug |
| 106 | 108 | Series y sumatorios | `sum(2^k, k, 0, n)` | P2×1 P3×1 | anti_pedagogical, latex_render_bug |
| 107 | 109 | Series y sumatorios | `product(k, k, 1, n)` | P2×1 | latex_render_bug |
| 108 | 110 | Series y sumatorios | `sum(1/2^k, k, 0, oo)` | P1×4 P2×1 | latex_render_bug, magic_step, rule_name_misleading, wrong_rule_name |
| 109 | 111 | Teoría de números | `totient(100)` | P2×1 P3×1 | chain_discontinuity, substep_chain_break |
| 110 | 112 | Teoría de números | `fibonacci(50)` | P2×2 | latex_render_bug, substep_noop_or_false_claim |
| 111 | 113 | Teoría de números | `gcd(x^2-1, x^2-2*x+1)` | P1×1 P2×2 P3×1 | magic_step, text_latex_divergence, wrong_rule_name |
| 112 | 114 | Teoría de números | `isprime(561)` | P2×4 | anti_pedagogical, magic_step, missing_narration |
| 113 | 115 | Teoría de números | `nextprime(1000)` | P2×1 P3×1 | magic_step |
| 114 | 116 | Teoría de números | `choose(10,5)` | P1×1 P3×2 | language_leak, magic_step, substep_noop_or_false_claim |
| 115 | 117 | Teoría de números | `factorial(n+2)/factorial(n)` | P3×2 | other, text_latex_divergence |
| 116 | 118 | Teoría de números | `divisors(60)` | P2×2 P3×1 | latex_render_bug, magic_step, substep_duplicates_parent |
| 117 | 119 | Teoría de números | `gcd(48,36,60)` | P2×4 | magic_step, missing_narration, wrong_rule_name |
| 118 | 120 | Matrices y álgebra lineal | `[[1,2],[3,4]] * [[5,6],[7,8]]` | P1×2 | magic_step |
| 119 | 121 | Matrices y álgebra lineal | `[[1,2],[3,4]]^(-1)` | P1×1 P2×1 | magic_step, noop_or_trivial_step |
| 120 | 122 | Matrices y álgebra lineal | `det([[1,2,3],[4,5,6],[7,8,10]])` | P1×2 P2×4 | anti_pedagogical, chain_discontinuity, duplicate_or_burst, latex_render_bug, rule_name_misleading |
| 121 | 123 | Matrices y álgebra lineal | `eigenvalues([[2,1],[1,2]])` | P1×3 | chain_discontinuity, magic_step |
| 122 | 124 | Matrices y álgebra lineal | `rref([[1,2,3],[4,5,6],[7,8,9]])` | P1×2 | magic_step |
| 123 | 125 | Matrices y álgebra lineal | `[[1,1],[0,1]]^5` | P1×2 | magic_step |
| 126 | 128 | Matrices y álgebra lineal | `cross([1,0,0],[0,1,0])` | P0×2 P1×1 P2×2 | anti_pedagogical, highlight_wrong_subexpression, magic_step, rule_name_misleading, wrong_rule_name |
| 127 | 129 | Matrices y álgebra lineal | `dot([1,2,3],[4,5,6])` | P1×1 P2×1 | anti_pedagogical, chain_discontinuity |
| 129 | 131 | Funciones y variables | `expand(expr1)` | P1×1 P2×1 | other, substep_noop_or_false_claim |
| 131 | 133 | Funciones y variables | `f(5)` | P1×1 P2×1 | missing_narration, other |
| 133 | 135 | Funciones y variables | `g(3,4)` | P1×1 P2×1 | magic_step, missing_narration |
| 134 | 136 | Funciones y variables | `h(x) := x^3 + sin(x)` | P3×1 | other |
| 135 | 137 | Funciones y variables | `diff(h(x), x)` | P2×2 | missing_narration, other |
| 136 | 138 | Verificación y equivalencia | `diff(integrate(2*x/sqrt(4+x^4)+1, x), x) - (2*x/sqrt(4+x^4)+1)` | P0×2 P1×6 P2×2 P3×1 | anti_pedagogical, chain_discontinuity, highlight_wrong_subexpression, magic_step, rule_name_misleading, text_latex_divergence, wrong_rule_name |
| 137 | 139 | Verificación y equivalencia | `equiv(diff(integrate(2*x/sqrt(4+x^4)+1, x), x), 2*x/sqrt(4+...` | P0×2 P1×1 P2×2 | highlight_wrong_subexpression, magic_step |
| 138 | 140 | Verificación y equivalencia | `equiv(sin(x+y), sin(x)*cos(y) + cos(x)*sin(y))` | P1×1 P2×1 | magic_step, noop_or_trivial_step |
| 139 | 141 | Verificación y equivalencia | `equiv(cos(2*x), 1-2*sin(x)^2)` | P2×2 | magic_step, noop_or_trivial_step |
| 140 | 142 | Verificación y equivalencia | `equiv((x^3+y^3)/(x+y), x^2-x*y+y^2)` | P0×1 P1×2 P2×2 | highlight_wrong_subexpression, latex_render_bug, noop_or_trivial_step |
| 142 | 144 | Verificación y equivalencia | `diff(integrate(x*e^x, x), x) - x*e^x` | P0×3 P1×2 P2×3 | anti_pedagogical, chain_discontinuity, highlight_wrong_subexpression, rule_name_misleading, substep_chain_break, substep_duplicates_parent, text_latex_divergence, wrong_rule_name |
| 143 | 145 | Verificación y equivalencia | `equiv(log(x*y), log(x)+log(y))` | P1×1 P2×1 | noop_or_trivial_step, wrong_rule_name |
| 144 | 146 | Verificación y equivalencia | `integrate(diff(x^3, x), x) - x^3` | P0×1 P1×1 P2×3 P3×1 | anti_pedagogical, chain_discontinuity, highlight_stale_form, highlight_wrong_subexpression, substep_chain_break |
| 145 | 147 | Verificación y equivalencia | `equiv(tan(x), sin(x)/cos(x))` | P2×1 | noop_or_trivial_step |
| 146 | 148 | Verificación y equivalencia | `diff(integrate(1/(x^3-2), x), x) - 1/(x^3-2)` | P3×1 | duplicate_or_burst |
| 147 | 149 | Vectorial | `gradient(x^2*y, [x,y])` | P1×2 | latex_render_bug, text_latex_divergence |
| 148 | 150 | Vectorial | `diff([x^2, sin(x)], x)` | P2×1 | magic_step |
| 149 | 151 | Vectorial | `dot(gradient(x^2*y,[x,y]), [1,0])` | P1×1 | latex_render_bug |
| 150 | 152 | Vectorial | `jacobian([x^2*y, x+y], [x,y])` | P1×1 | latex_render_bug |
| 151 | 153 | Vectorial | `det(hessian(x^2*y, [x,y]))` | P0×1 P1×3 P2×2 | chain_discontinuity, highlight_stale_form, highlight_wrong_subexpression, latex_render_bug, magic_step, other |
| 152 | 154 | Vectorial | `divergence([x^2, y^2], [x,y])` | P1×2 P2×1 | latex_render_bug, text_latex_divergence |
| 153 | 155 | Vectorial | `laplacian(ln(x^2+y^2), [x,y])` | P0×4 P1×6 P2×3 | anti_pedagogical, chain_discontinuity, highlight_red_equals_green, highlight_wrong_subexpression, latex_render_bug, magic_step, other, substep_noop_or_false_claim |
| 154 | 156 | Vectorial | `curl([y,-x,0], [x,y,z])` | P3×1 | latex_render_bug |
| 155 | 157 | Vectorial | `curl(gradient(x*y*z,[x,y,z]), [x,y,z])` | P2×1 | chain_discontinuity |
| 156 | 158 | Vectorial | `integrate([cos(x), e^x], x)` | P2×1 | magic_step |
| 157 | 159 | Vectorial | `abs([3,4])` | P2×3 P3×1 | chain_discontinuity, latex_render_bug, rule_name_misleading, wrong_rule_name |
| 158 | 160 | Complejo | `(3+4i)/(1-2i)` | P1×2 | language_leak, other |
| 161 | 163 | Complejo | `(1+i)^(-1)` | P2×1 | language_leak |
| 162 | 164 | Complejo | `abs(3+4i)` | P1×1 | language_leak |
| 164 | 166 | Complejo | `solve(x^2+1, x)` | P1×1 P2×1 | missing_narration, substep_noop_or_false_claim |
| 165 | 167 | Complejo | `solve(x^3-1, x)` | P1×1 | magic_step |
| 166 | 168 | Complejo | `e^(i*pi)` | P1×1 P2×1 | highlight_red_equals_green, highlight_wrong_subexpression |
| 168 | 170 | Complejo | `ln(-1)` | P1×1 P2×1 | missing_narration, wrong_rule_name |
| 170 | 172 | Complejo | `i^i` | P0×1 | highlight_wrong_subexpression |
| 172 | 174 | Complejo | `abs(e^(2*i))` | P1×2 | other |
| 175 | 177 | Complejo | `equiv(e^(i*pi), -1)` | P0×1 P1×3 P2×1 | missing_narration, other, rule_name_misleading, substep_noop_or_false_claim, wrong_rule_name |
| 176 | 178 | Vectorial | `hessian(x^2*y, [x,y])` | P0×1 P1×3 P2×2 | highlight_wrong_subexpression, latex_render_bug, magic_step, other, substep_wrong_math, text_latex_divergence |
| 177 | 179 | Vectorial | `curl([y,-x], [x,y])` | P2×1 | substep_duplicates_parent |
| 178 | 180 | Vectorial | `diff([x^2, x^3], x, 2)` | P0×1 | highlight_wrong_subexpression |
| 179 | 181 | Vectorial | `subs(subs(diff(x^2*y,x), x, 1), y, 2)` | P0×1 P1×1 P2×1 | highlight_stale_form, highlight_wrong_subexpression, substep_chain_break |
| 180 | 182 | Vectorial | `subs(subs(det(hessian(x^3+y^3-3*x*y,[x,y])), x, 1), y, 1)` | P0×2 P1×2 P2×1 | anti_pedagogical, chain_discontinuity, highlight_wrong_subexpression, latex_render_bug, substep_wrong_math |
| 181 | 183 | Vectorial | `solve([diff(x^2+y^2-2*x-4*y, x)=0, diff(x^2+y^2-2*x-4*y, y)...` | P1×2 P2×1 | magic_step, other |
| 182 | 184 | Analítico | `taylor(sin(x)/x, x, 0, 4)` | P0×2 P1×1 P2×2 | anti_pedagogical, highlight_red_equals_green, magic_step, wrong_rule_name |
| 183 | 185 | Analítico | `taylor(e^(x+y), [x,y], [0,0], 2)` | P2×4 | anti_pedagogical, duplicate_or_burst, latex_render_bug, substep_duplicates_parent |
| 184 | 186 | Analítico | `lineintegral([-y,x], [x,y], [cos(t),sin(t)], t, 0, 2*pi)` | P1×2 P2×1 P3×1 | chain_discontinuity, latex_render_bug, substep_duplicates_parent |
| 185 | 187 | Analítico | `lineintegral(x^2, [x,y], [cos(t),sin(t)], t, 0, pi)` | P0×3 P1×3 P2×4 P3×2 | highlight_red_equals_green, highlight_wrong_subexpression, lang_parity, latex_render_bug, magic_step, substep_duplicates_parent, substep_wrong_math |
| 186 | 188 | Analítico | `integrate([cos(t),sin(t)], t, 0, pi)` | P2×2 | chain_discontinuity, highlight_wrong_subexpression |
| 187 | 189 | sympy wrong | `integrate(1/(x^8+1), x)` | P1×3 | magic_step, text_latex_divergence |
| 188 | 190 | sympy wrong | `integrate(1/(x^8+16), x)` | P1×3 | magic_step, text_latex_divergence |
| 189 | 191 | sympy wrong | `solve(sin(x)>1/2, x)` | P1×2 P2×1 P3×1 | latex_render_bug, missing_narration |
| 190 | 192 | sympy wrong | `solve(tan(x)>1, x)` | P1×2 | magic_step, missing_narration |
| 191 | 193 | sympy wrong | `integrate(1/(x^3-x-1), x)` | P1×2 P2×2 | latex_render_bug, magic_step, text_latex_divergence |
| 192 | 194 | Analítico | `surface_integral([x,y,0], [x,y,z], [cos(u),sin(u),v], [u,v]...` | P0×4 P1×3 P2×2 | anti_pedagogical, chain_discontinuity, highlight_red_equals_green, highlight_wrong_subexpression, latex_render_bug, other, wrong_rule_name |
| 193 | 195 | Analítico | `surface_integral(1, [x,y,z], [u,v,u+v], [u,v], [0,1], [0,1])` | P1×4 P2×2 P3×1 | anti_pedagogical, chain_discontinuity, other, text_latex_divergence |
| 194 | 196 | Analítico | `potential([2*x*y, x^2], [x,y])` | P1×2 P2×3 | latex_render_bug, magic_step, noop_or_trivial_step, other, substep_noop_or_false_claim |
| 195 | 197 | Analítico | `potential(gradient(x^2*y + 3*x, [x,y]), [x,y])` | P1×2 P2×2 | anti_pedagogical, magic_step, substep_noop_or_false_claim, text_latex_divergence |
| 196 | 198 | Analítico | `limit(x*y/(x^2+y^2), [x,y], [1,1])` | P2×1 P3×1 | chain_discontinuity, substep_chain_break |
| 197 | 199 | Analítico | `limit(x*y/(x^2+y^2), [x,y], [0,0])` | P1×3 P2×1 | anti_pedagogical, lang_parity, rule_name_misleading, wrong_rule_name |
| 198 | 200 | Analítico | `limit(sin(z)/z, z, 0)` | P1×1 | anti_pedagogical |
| 199 | 201 | Analítico | `limit(exp(z), z, i*pi)` | P0×1 P1×2 P2×2 P3×2 | anti_pedagogical, language_leak, latex_render_bug, noop_or_trivial_step, other, substep_wrong_math |
| 200 | 202 | Analítico | `limit(x^2*y/(x^2+y^2), [x,y], [0,0])` | P1×2 | rule_name_misleading, wrong_rule_name |
| 201 | 203 | Ecuaciones diferenciales | `dsolve(diff(y,x) = x*y, y, x)` | P1×2 P2×2 P3×1 | anti_pedagogical, duplicate_or_burst, lang_parity, latex_render_bug, noop_or_trivial_step |
| 202 | 204 | Ecuaciones diferenciales | `dsolve(diff(y,x) + y = x, y, x)` | P2×2 | magic_step, noop_or_trivial_step |
| 203 | 205 | Ecuaciones diferenciales | `dsolve((2*x*y+1) + (x^2+2*y)*diff(y,x) = 0, y, x)` | P2×3 | magic_step, noop_or_trivial_step |
| 204 | 206 | Ecuaciones diferenciales | `dsolve(diff(y,x) = -y, y, x, y(0) = 3)` | P1×1 P2×1 | anti_pedagogical, noop_or_trivial_step |
| 205 | 207 | Ecuaciones diferenciales | `dsolve(diff(y,x,2) + 4*y = 0, y, x)` | P2×1 P3×1 | magic_step, noop_or_trivial_step |
| 206 | 208 | Ecuaciones diferenciales | `dsolve(diff(y,x,2) + y = cos(x), y, x)` | P1×2 P2×2 | anti_pedagogical, chain_discontinuity, magic_step, wrong_rule_name |
| 207 | 209 | Ecuaciones diferenciales | `dsolve(diff(y,x) + y = y^2, y, x)` | P2×1 | magic_step |
| 208 | 210 | Ecuaciones diferenciales | `dsolve([diff(x,t) = -y, diff(y,t) = x], [x,y], t)` | P2×4 | latex_render_bug, magic_step, noop_or_trivial_step, other |
| 209 | 211 | Ecuaciones diferenciales | `dsolve(diff(y,x) = x^2 + y^2, y, x)` | P2×1 | latex_render_bug |

---

## Reproducibilidad

Los datos crudos del audit (JSON por fila, informes de detectores, hallazgos consolidados)
viven en el scratchpad de la sesión y se regeneran con:

```
python3 run_corpus.py     # 210 filas -> json/NNN.json  (~1.4 s)
python3 detect.py         # 11 detectores -> detector_report.json + substep_report.json
```

Los detectores son generadores de CANDIDATOS: `D3b_text_vs_latex_order_only` (194 hits) y
`D11_single_step` (85 hits) fueron descartados en bloque tras verificación, y `D12` tiene 4
falsos positivos conocidos (ecos de declaración `:=`). Cualquier comparación futura debe
anclarse **por expresión, no por índice** — el csv se reordena.
