# Diseño: `SolutionSet::PeriodicIntervalUnion`

**Fecha:** 2026-07-03 · **Estado:** v2 — revisado por panel adversarial de 3 lentes (matemática / arquitectura / edge-cases, workflow `piu-design-review`; 3× approve-with-changes, todas incorporadas) · **Origen:** gap estructural #1 del re-scout de universalidad (`docs/SCOUT_UNIVERSALIDAD_2026-07-03.md` §backlog #2; 12 de los 21 declines trig lo requieren genuinamente).

> **P0 DESCUBIERTO POR EL PANEL (independiente de PIU, arreglar ANTES de P1):**
> `try_decline_periodic_trig_inequality` (solve_backend_local.rs:3067) chequea
> `contains_trig_of_var` SOLO en `eq.lhs` — con el trig en RHS cae a la
> inversión monotónica genérica y produce **wrong-answers vivos en HEAD**:
> `solve(1/2<sin(x))` → `(π/6, ∞)`; `solve(2<tan(x))` → `(arctan(2), ∞)`;
> `solve(1/3<cos(x))` → `(arccos(1/3), ∞)`. Fix: decline orientation-blind
> (lhs O rhs), espejando la exclusión de umbrales bare sin/cos (línea 3071)
> al lado que lleve el trig. Testigos ambas orientaciones en la suite.

## 1. Problema

Las desigualdades trig con umbral interior (`sin(x) > 1/2`) tienen como solución una **unión periódica de intervalos** — ∪ₖ (π/6+2kπ, 5π/6+2kπ) — que ninguna variante actual de `SolutionSet` puede representar. Hoy declinan a residual honesto. La ecuación frontera ya se resuelve exacta (`solve(sin(x)=1/2)` → `{π/6+2kπ, 5π/6+2kπ}`): **el gap es solo de representación y selección de ventana**, no de cálculo de fronteras.

## 2. La variante

```rust
/// ∪_{k∈ℤ} ∪_i (windows[i] + k·period): unión periódica de intervalos.
///
/// - `windows`: intervalos en UN dominio fundamental, ordenados, disjuntos,
///   no degenerados (min < max en valor; familias de puntos usan `Periodic`).
/// - `period`: período fundamental compartido, positivo.
///
/// Invariantes (v2, corregidos por el panel — los originales eran vacuos y
/// no excluían el doble-cubrimiento módulo T):
/// - Ventanas ordenadas por min, no degeneradas (min < max en valor), con
///   endpoints FINITOS, pares disjuntas y NO adyacentes módulo T.
/// - INVARIANTE DE SPAN: max(última) − min(primera) ≤ period — todas las
///   ventanas caben en UN traslado de longitud T. Esto legitima las ventanas
///   "wrapped" de la tabla §5 (p.ej. (5π/6, 13π/6)) sin dominio fijo [0,T) y
///   garantiza que los traslados +kT nunca solapan.
/// - Σ len ≤ period, con igualdad EXACTAMENTE en el caso una-sola-ventana
///   open-open de len == period (recta perforada: cos(x)<1 → (2kπ, 2π+2kπ)
///   = ℝ ∖ {2kπ}). len == period con algún extremo cerrado ⇒ AllReals,
///   NUNCA se emite como PIU (constructor lo colapsa; debug_assert).
/// - Constructores y álgebra REESTABLECEN el invariante tras cada operación.
/// - Membership: x ∈ S ⟺ ∃k∈ℤ: x − k·period ∈ windows[i] para algún i.
PeriodicIntervalUnion { windows: Vec<Interval>, period: ExprId }
```

Reutiliza `Interval` (cas_ast/domain.rs:11) tal cual: sus `BoundType` por extremo cubren el requisito de **clausura mixta en una misma ventana** (`tan(x)>=0` → `[kπ, π/2+kπ)`), y `ExprId` como extremo cubre los **endpoints simbólicos** (`arccos(1/3)`).

### Por qué no las alternativas

- **Generalizar `Periodic` a ventanas**: rompería los 157 match-sites que hoy asumen bases puntuales y el printer compartido; una variante hermana es aditiva y los 9 sitios exhaustivos que rompen son exactamente los que DEBEN decidir qué hacer con ella.
- **`Union` infinita materializada**: imposible (k∈ℤ).
- **Llevar la variable dentro de la variante** (para render "x ∈ …"): ninguna variante lleva la variable; el render set-builder `{ (a+kT, b+kT) : k ∈ ℤ }` no la necesita y calca el frame existente de `Periodic`.

## 3. Requisitos con testigo (ground truth verificada a mano, tarea F del mapa)

| Requisito | Testigo | Esperado |
|---|---|---|
| Ventana simple abierta | `sin(x)>1/2` | `(π/6+2kπ, 5π/6+2kπ)`, T=2π |
| Cerrada estricta/no-estricta | `sin(x)>=1/2` | `[π/6+2kπ, 5π/6+2kπ]` |
| Wrap del dominio fundamental | `cos(x)>1/2` | `(−π/3+2kπ, π/3+2kπ)` |
| Ventana > T/2 con wrap | `sin(x)>-1/2` | `(−π/6+2kπ, 7π/6+2kπ)` |
| Clausura MIXTA en una ventana | `tan(x)>=0` | `[kπ, π/2+kπ)` — asíntota SIEMPRE abierta |
| Asíntota en extremo inferior | `tan(2x)<1` | `(−π/4+kπ/2, π/8+kπ/2)`, T=π/2 |
| Recta perforada (len == T) | `cos(x)<1` | `(2kπ, 2π+2kπ)` = ℝ∖{2kπ} |
| Dos ventanas disjuntas por período | `1/sin(x)>2` | `(2kπ, π/6+2kπ) ∪ (5π/6+2kπ, π+2kπ)` |
| Endpoint simbólico no exacto | `3cos(x)+1>2` | `(−arccos(1/3)+2kπ, arccos(1/3)+2kπ)` |
| Afín completo (escala+shift, flip) | `cos(2x−π/4)<=1/2` | `[7π/24+kπ, 23π/24+kπ]`, T=π |
| Shift no-π-conmensurado | `sin(2x+1)>1/2` | `(π/12−1/2+kπ, 5π/12−1/2+kπ)` |
| Coeficiente negativo (flip) | `-2sin(x)>1` | `(7π/6+2kπ, 11π/6+2kπ)` |
| c=0 (lección: barrer c=0) | `sin(x)>0` | `(2kπ, π+2kπ)` |

Fuera de alcance documentado (no son familia periódica o son reducción previa): `log(x,2)>3` (intervalo finito, capability aparte), `2^(x^2)>2` (base monótona), `sin(x)^2⋚c` / `abs(trig)⋚c` (capa de reducción de wrappers pares — pueden llegar como P4 reduciendo a `|sin(x)|<½` ≡ ventana en período π).

## 4. Arquitectura (hechos del mapa, workflow `piu-map`)

- **Enum**: `cas_ast/src/domain.rs:355`. 157 match-blocks en 84 archivos; **9 exhaustivos rompen** (2 álgebra core, 4 renderers producción, 2 didactic timeline, 1 helper de snapshots) y **2 catch-alls tuple con `debug_assert!(false)`** en `union/intersect_solution_sets` (cas_solver_core/src/solution_set.rs) que DEBEN ganar brazo o los builds debug panican.
- **Álgebra**: core tiene `union/intersect` (sin complemento). Precedente arquitectónico: `Periodic∪interval` se difiere deliberadamente a la capa solver (que tiene Simplifier). `merge_intervals`/`intersect_intervals` existentes sirven para ventanas mismo-período.
- **Orden de endpoints (v2 — blocker del panel)**: `compare_values` es TOTAL — cuando los oráculos de valor fallan cae SILENCIOSAMENTE a `compare_expr` estructural y devuelve un `Ordering` igualmente; "si no ordenan" no es señal observable. **P1 introduce `try_compare_values(ctx, a, b) -> Option<Ordering>` en cas_solver_core**: la cadena de oráculos existente (números → surds cuadráticos → nth-root → separación por `const_value_bounds`) SIN el fallback estructural; `None` cuando todos pasan. TODO el álgebra de ventanas PIU (sort, merge, intersección, normalización de anchor) usa SOLO `try_compare_values`; ante `None`: unión → concatenar sin fusionar (sound como conjunto, no canónico); intersección → no computar en core (deferral a solver layer / decline del solver). `compare_values` queda intacto para los brazos legacy. Además: **el productor emite ventanas cuyo ORDEN es correcto por construcción analítica** (tabla §5), no por comparación a posteriori. `const_value_bounds` no tiene brazo arcsin/arccos/arctan (P4 lo añade y desbloquea fusiones con endpoints simbólicos).
- **Render**: 5 sitios delegan `Periodic` al par compartido `cas_formatter/src/periodic.rs`; se añade el par hermano `display_periodic_interval_union` / `latex_periodic_interval_union` con el MISMO frame: texto `{ (1/6·pi + k·2·pi, 5/6·pi + k·2·pi) : k ∈ ℤ }` (ventanas separadas por coma, como las bases múltiples hoy), LaTeX `\left\{ \left(\frac{\pi}{6} + k\cdot 2\pi,\ \frac{5\pi}{6} + k\cdot 2\pi\right) : k \in \mathbb{Z} \right\}`.
- **Wire**: solo strings pre-renderizadas (`result`/`result_latex`); `SolutionSetDto` es dead code nunca construido → sin ruptura de wire.
- **Verificación**: `Periodic` es `NotCheckable` → PIU igual (familia infinita).
- **Huella**: 0-delta esperada — ningún suite GUARD/PRESS alimenta `solve()`; equivfuzz solo `simplify()`.
- **Pins a recontratar** (ciclo P2): `cli_contract_tests.rs` (3 fns: ~2682 `test_eval_periodic_trig_inequality_declines`, ~3340 lado complemento del boundary, ~4220 variable-base-log solo si toca), `residual_honesty_and_weak_boundary_contract_tests.rs` (`interior_thresholds_still_decline_honestly` — pin del ciclo 3 diseñado para caer aquí).

## 5. Productor (ciclo P2): tabla analítica en u-space + mapa afín inverso

Slot exacto: `try_solve_trig_weak_boundary_inequality` interior `|r|<1` (solve_backend_local.rs:5076-5078, hoy `return None`). Al slot llegan `trig_fn` (Sin|Cos), `r=c/A` exacto (BigRational, op flipped si A<0) y `op`. **Precondiciones que el productor DEBE añadir (v2 — el panel refutó "ya llegan normalizados"):**

1. **El arg NO está validado afín**: `bounded_trig` solo exige `contains_var` — `sin(x²)>1/2` y `sin(sin(x))>1/2` alcanzan el slot. Gate obligatorio: `affine_coefficients(simplifier, arg, var)` (solve_backend_local.rs:5826; pendiente a racional ≠ 0, intercepto sin var) con `None` → decline honesto sellado. `sin(x²)>1/2` entra al barrido como must-decline.
2. **La constante aditiva no se pela hoy**: `3·cos(x)+1 >= 4` NO llega al slot (verificado vivo — ni siquiera el caso weak-boundary r=1 existente lo captura). El productor pela `A·trig(g)+d ⋚ c` → `r=(c−d)/A` ANTES del match — esto arregla de paso el gap |r|≥1 para esa forma. Testigo #9 depende de esto.
3. **Límites documentados como declines**: `A` irracional (`√2·sin(x)⋚c`) y pendiente no racional (`sin(πx)⋚c`) declinan — testigos de decline en el barrido.
4. **Sellado de declines**: cuando `bounded_trig` matchea `A·sin/cos(g)⋚c` racional pero el productor no puede emitir, devolver el residual honesto conserva-operador directamente (no dejar caer a handlers genéricos aguas abajo — el P0 de orientación demuestra que ese fall-through es peligroso).

Ventanas en u (dominio fundamental elegido para NO fragmentar la ventana; `asin_r = simplify(arcsin(r))` pliega a ángulo exacto vía INVERSE_TRIG_TABLE cuando aplica):

| Relación | Ventana u | T_u |
|---|---|---|
| `sin u > r` | `(asin_r, π − asin_r)` | 2π |
| `sin u ≥ r` | `[asin_r, π − asin_r]` | 2π |
| `sin u < r` | `(π − asin_r, 2π + asin_r)` | 2π |
| `sin u ≤ r` | `[π − asin_r, 2π + asin_r]` | 2π |
| `cos u > r` | `(−acos_r, acos_r)` | 2π |
| `cos u ≥ r` | `[−acos_r, acos_r]` | 2π |
| `cos u < r` | `(acos_r, 2π − acos_r)` | 2π |
| `cos u ≤ r` | `[acos_r, 2π − acos_r]` | 2π |
| `tan u > r` | `(atan_r, π/2)` — asíntota abierta | π |
| `tan u ≥ r` | `[atan_r, π/2)` | π |
| `tan u < r` | `(−π/2, atan_r)` | π |
| `tan u ≤ r` | `(−π/2, atan_r]` | π |

**tan es un handler HERMANO SEPARADO (v2 — major del panel), NO una extensión del slot sin/cos**: `bounded_trig` matchea solo Sin|Cos y su escalera `|r|⋚1` codifica el rango [−1,1] — meter Tan ahí convierte `tan(x)≥2` en `Empty` (wrong answer; verdad `[arctan(2)+kπ, π/2+kπ)`). P3 crea `try_solve_tan_interior_inequality` despachado por `BuiltinFn::Tan` ANTES de la escalera, sin casos de rango, tabla válida para TODO r racional, asíntotas siempre `Open`, T=π. Testigos trampa en el barrido: `tan(x)>=2`, `tan(x)<-3`.

Casos especiales previos sin/cos (ya cubiertos, orden de guardas): `|r|>1` → Empty/AllReals (existente); `|r|=1` → weak-boundary (existente); r no racional y `|r|<1` no probable → decline honesto (sin cambio). Para sin/cos con r=±1 la tabla NO se usa. Nota recta-perforada: `sin u > −1` con r=−1 cae en weak-boundary hoy (complemento → decline); P3 lo emite como ventana len==T.

**Mapa afín inverso (v2 — el panel refutó la regla original "swap+flip")**: bajo el mapa decreciente `x=(u−b)/a` con a<0, la imagen de `(u1, u2]` es `[x(u2), x(u1))` — los endpoints se intercambian como PAREJAS (valor, BoundType): `new_min=(map(max), max_type)`, `new_max=(map(min), min_type)`. **El BoundType viaja con su endpoint; NUNCA se invierte** (un "flip" literal convierte `(π/6, 5π/6)` con a=−1 en el CERRADO `[−5π/6, −π/6]` — wrong answer estricto). Normativo: `map_set_through_inverse_affine` ya lo hace bien (solve_backend_local.rs:6131-6148, `min_type: iv.max_type`) e incluye `mapped.reverse()` para el orden de la LISTA con a<0 — el brazo PIU replica ambas cosas y re-canonicaliza al invariante de span (§2). `T_x = T_u/|a|` (positivo); cada endpoint simplificado. Testigos a<0 añadidos: `sin(π/3−x)>1/2` (open-open) y `tan(−x)>=0` (clausura mixta → `(−π/2+kπ, 0]`).

Soundness gate final del productor — airbag numérico f64 (v2, semántica precisada por el panel):
- **P2 añade brazos arcsin/arccos/arctan a `eval_f64_with_substitution`** (cas_math/src/numeric_eval.rs:511 — hoy no los tiene y el airbag mataría los propios testigos con endpoints simbólicos).
- **Nunca muestrear endpoints** (ulp: `sin(asin(0.5)) < 0.5` en f64): muestrear a fracciones del ancho de ventana EN u-space (`ancho·{1/8, 1/2, 7/8}` dentro, `±ancho/8` fuera) y mapear por `(u−b)/a`.
- **Regla de veredicto**: discrepancia SOLO si el signo de `f(x)−c` contradice la relación por más de `τ = 1e-9·max(1,|c|)`; `|f(x)−c| ≤ τ` o eval `None` ⇒ INCONCLUSO (saltar muestra, jamás declinar por ella).
- Solo como AIRBAG de emisión; la corrección viene de la tabla. Sin f64 en decisiones de conjunto (lección `soundness-gates-must-be-exact`) — el airbag solo puede DEGRADAR a decline, nunca ampliar.

## 6. Álgebra de conjuntos (alcance por ciclo)

**P1 (core, sin productor):**
- `union/intersect` con `Empty`/`AllReals`: trivial.
- **`PIU ∪ PIU` / `PIU ∩ PIU` mismo período estructural — ÁLGEBRA CIRCULAR EXPLÍCITA (v2, blocker del panel: la versión naive pierde masa wrapped — `(0,π) ∩ (5π/6,13π/6)` como intervalos planos da solo `(5π/6,π)` y pierde la familia `(2kπ, π/6+2kπ)` del testigo #8):**
  1. Precondición: ambos operandos cumplen el invariante de span (§2).
  2. Anchor = `min` de la primera ventana de un operando (sin floor mod-T para ese operando).
  3. Trasladar cada ventana del OTRO operando por k ∈ {−1, 0, +1} períodos para que su endpoint izquierdo caiga en `[anchor, anchor+T)` — el span ≤ T garantiza a lo sumo UN corte por ventana; **partir** en el seam `anchor+T` la que lo cruce (dos piezas, BoundTypes conservados; el corte en el seam es open/open interno).
  4. Ejecutar `merge_intervals`/`intersect_intervals` lineales sobre las formas partidas.
  5. **Re-fusionar** las piezas adyacentes en `anchor`/`anchor+T` (son T-traslaciones) en una ventana wrapped; re-establecer el invariante de span.
  6. Post-check: ventana única len==T con algún extremo cerrado → `AllReals`; cobertura total con seam cerrado → `AllReals`.
  - Cada paso que no pueda ordenar un endpoint con `try_compare_values` → devolver operandos sin combinar (unión) / no computar en core (intersección).
  - **El testigo #8 (dos ventanas por intersección wrapped) es test unitario OBLIGATORIO del álgebra P1** con ventanas construidas, no solo test de productor P3.
- Todo lo demás (`PIU ∪ Periodic`, `PIU ∩ Continuous`, períodos distintos): brazo `debug_assert + documented-unreachable` con **contrato de alcanzabilidad explícito**: cualquier combinador solver-layer cuyos operandos puedan ser PIU chequea primero combinabilidad-core (mismo período estructural Y `try_compare_values` ordena todos los endpoints) y si no, declina la relación ENTERA a residual honesto ANTES de llamar a core. NUNCA drop silencioso (lección `core-union-Periodic-drop`). Lista blanca de sitios que mantienen PIU fuera en P2: `is_concrete_solution_set` (:4142), gates `Continuous|Union` (:640/:8016), `map_set_through_inverse_affine`.
- Los 2 catch-alls `debug_assert` de union/intersect ganan brazo explícito.

**P3:** tan (handler hermano, ver §5), reciprocal-trig INEQUALITY reduction — `1/sin(x)>2` llega como `csc(x)>2` (el simplificador repliega 1/sin→csc; hoy csc en desigualdad da error duro, no decline): extender `try_solve_reciprocal_trig_equation` (que ya matchea el árbol RAW pre-refold) a los 4 ops, reduciendo `1/s ⋚ r` por casos de signo — r>0: `s ∈ (0, 1/r)`; r<0: `s ∈ (0,∞) ∪ (−∞, 1/r)` (UNIÓN de dos PIUs mismo período), cada uno clipped a [−1,1] — y delegando en tabla P2 + álgebra P1. Testigos: `1/sin(x)>2`, `1/sin(x)>-2` (rama unión), `csc(x)>2` ambas orientaciones. Recta perforada (`cos(x)<1`, `sin(x)>-1`). `PIU ∩ Continuous` finito solo si un caso real lo exige.

## 7. Plan de ciclos

- **P0 — Wrong-answer de orientación (pre-ciclo, urgente e independiente):** `try_decline_periodic_trig_inequality` orientation-blind (lhs O rhs) espejando la exclusión de umbrales exactos al lado del trig; también cubre `2<1/sin(x)`. Testigos ambas orientaciones. Commit propio.
- **P1 — Núcleo representacional (behavior-neutral):** variante + brazos en los 9 exhaustivos (inventario verificado por el panel: 1 en `sset_kind` solution_set.rs:539, 2 en verification.rs — :66 NotCheckable Y :377 `classify_guard_verified_conditional` → None, 3 renderers producción, 2 didactic — display.rs:23 / non_nested.rs:30, 1 snapshot helper repl_snapshots.rs:152) + 2 catch-alls + `try_compare_values` en cas_solver_core + álgebra circular §6 + par de renderers + tests unitarios core (membership por construcción, render, álgebra circular con testigo #8 y punctured-line). Cero productores: workspace verde sin recontratos, huella 0-delta estricta.
- **P2 — Productor sin/cos:** tabla + mapa afín + gate numérico; recontratos de los pins de decline; suite de contrato CLI nueva (los 13 testigos §3 aplicables a sin/cos); barrido adversarial (~90 formas: {sin,cos}×{>,<,>=,<=}×{c exacto, c simbólico, c=0, wrappers afines/coef ±}) con verificación de membership numérica multi-k.
- **P3 — tan + multi-ventana + perforada:** tabla tan (asíntotas), recíprocos vía intersección, `cos(x)<1`/`sin(x)>-1` como len==T, `sin(x)^2`/abs como reducción si cabe.
- **P4 (opcional) — brazo arcsin/arccos/arctan en `const_value_bounds`** para ordenar endpoints simbólicos en el álgebra (desbloquea fusiones mismo-período con r no exacto).

## 8. Riesgos y mitigaciones

1. **Emisión de ventana mal orientada (a<0, wraps)** → tabla analítica + swap sistemático + airbag numérico de emisión + barrido multi-k.
2. **Álgebra core sin Simplifier** (lección arquitectónica) → P1 solo opera con `compare_values`; lo no-ordenable se difiere a solver layer explícitamente.
3. **Panics debug por catch-alls** → brazos explícitos en P1 con tests.
4. **Pins existentes** → inventario ya hecho (§4); recontratos en P2 con el diff exacto esperado.
5. **Regresión de los casos ya resueltos** (weak-boundary del ciclo 3, ecuaciones periódicas) → el productor corre DESPUÉS de esas guardas; suite de ciclo 3 permanece verde (salvo el pin interior diseñado para caer).
