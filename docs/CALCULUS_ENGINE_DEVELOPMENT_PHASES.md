# Hoja de ruta por fases — del real-univariable al multivariable y complejo

Documento de secuenciación del north star (`CALCULUS_ENGINE_STRATEGY.md`: engine de
cálculo diferencial/integral **universal Y educativo** en dominio real). Fija el ORDEN de
los horizontes y, para cada uno, los items concretos. Fundado en dos evaluaciones
multi-agente (2026-06-21): la de completitud de la frontera real, y la de extensibilidad
arquitectónica a complejo y multivariable.

**Regla de oro (actualizada 2026-07-18):** el umbral de Fase 1 quedó **CRUZADO 2026-07-15**;
la **Fase 2 quedó COMPLETA 2026-07-18 en sus dos mitades** (complejo elemental 2026-07-17 +
vectorial multivariable 2026-07-18, ambos frentes formalmente cerrados). La **Fase 3 está
ACTIVA desde 2026-07-18 por decisión del usuario** ("Haz el scoping workflow"), con scoping
propio en `docs/FASE3_ANALYTIC_LAYERS_SCOPING.md` (secuencia F0→F12; el P0 F0 —
kill-switch de dominio en límites — va primero). La regla de fondo se mantiene: cada fase se
abre SOLO al cruzar el umbral de la anterior, y los guardrails inter-fase (§final) siguen
siendo obligatorios en cada ciclo.

*Texto original (histórico):* la **Fase 1 es el ÚNICO objetivo activo**. Las Fases 2 y 3 NO
se empiezan hasta cruzar el umbral de Fase 1 — existen aquí para que las decisiones de HOY
las mantengan baratas (ver §Guardrails inter-fase). No son "deferred = abandonadas"; son
"deferred = secuenciadas, y conscientemente preparadas".

**Qué ordena la fase (y qué no).** La restricción de fase ordena SOLO el trabajo de
**nueva capacidad de cálculo**. Quedan EXENTOS y van siempre primero: (a) los **fixes de
soundness/honestidad** — cualquier wrong-answer o condición de dominio perdida, en
CUALQUIER comando (solve, inecuaciones, factor, gcd, series, abs, matrices incluidos),
aunque no sean cálculo real-univariable elemental; y (b) los **ciclos arquitectónicos /
de extracción (clase A)**. El north-star de fase no deroga el "mayor ROI retenible" del
proceso maestro; cuando la cola P0 de soundness y la restricción de fase chocan, gana P0.

---

## Estado de partida (2026-06-21)

*Procedencia: cifras derivadas de las dos re-evaluaciones multi-agente del 2026-06-21
(completitud de frontera real + extensibilidad), no de la tabla del audit committeado
`CALCULUS_FRONTIER_AUDIT.md` (2026-06-12), que usó otra base de sondas y da reparto distinto
(Dif ~80%, integral ~60-65%, límites ~45-50%). No hay re-audit committeado del 2026-06-21.*

- **Soundness/honestidad: ~88%** (piso más alto; 0 wrong-answers en sondas adversariales;
  los P0 de wrong-answer confirmados arreglados). El engine es de fiar HOY.
- Diferenciación ~72%, integral indefinida elemental ~72% (racional/algebraico ~66%),
  definida/impropia ~62-75%, límites ~68%, **series ~28%**, **educativo bimodal EN ESE BASELINE**
  (diff/integrate narran; **límites ~0% educativo** — *SUPERADO 2026-07-15: la narrativa de límites
  es MADURA (ver el gatekeeper G2 y el criterio de salida #2); esta línea es el snapshot del
  2026-06-21, no el estado actual*).
- Infra ya presente para horizontes futuros: `ValueDomain` enhebrado (76 archivos, 18 de
  `rules/` lo gatean), `GaussianRational`, política principal-branch binding; `MultiPoly`
  n-variable, `Matrix` nodo AST, sistemas lineales, **derivadas parciales e integrales
  iteradas YA funcionan**.

---

## Fase 1 — Serio y universal: real, univariable, elemental + educativo básico  **[ACTIVA]**

**Criterio de salida (checklist mecánico, NO cualitativo):** Fase 2 se abre cuando, y solo
cuando, los tres se cumplen:
1. **Integración racional sin residual** sobre los probes nombrados: `1/(x^5-1)`,
   `1/(x^6±1)`, `1/(x^8-1)`, `1/(x^4-4)`, `1/(x^3-2)` (todos integran y verifican).
   **✅ CUMPLIDO 2026-07-15** — los CINCO probes integran y verifican (Cap.A `1/(x^4-4)`,
   Cap.B `1/(x^6+1)`/`1/(x^8-1)`, Cap.C `1/(x^5-1)`, Cap.D `1/(x^3-2)`).
   **⚑ LOS TRES CRITERIOS DEL CHECKLIST ESTÁN CUMPLIDOS (2026-07-15): el umbral de
   apertura de la Fase 2 queda formalmente CRUZADO.** La decisión de arrancar la Fase 2
   es del siguiente ciclo (los guardrails inter-fase la dejaron ≈M, no L).
2. **Límites con cadena didáctica no-cáscara** en `cas_didactic` (límites-educativo deja de
   ser ~0%: existe al menos L'Hôpital / límite notable / squeeze / factor-cancela narrados).
   **✅ CUMPLIDO 2026-07-15** — los cuatro narran, más jerarquía ∞/∞, `e` vía `(1+1/x)^x`, y ∞−∞
   (conjugado en ∞ + común-denom en punto finito). La Fase 1 queda bloqueada SOLO por el criterio #1
   (racional: `1/(x^5-1)`, `1/(x^3-2)` aún residual) y el #3.
3. **Los wins P1 baratos aterrizados** (`diff(x,n)`/`diff(x,y)`, `taylor()`/`series()` +
   linealidad de sumatorios, sustitución-u general transcendente).

*Estimación de duración (no es el gate): **~25-40 ciclos efectivos**, dominados por dos*
*gatekeepers clase L.*

### Gatekeepers (máxima prioridad — desbloquean cada mitad del north star)
- **G1 · Integración racional UNIVERSAL** (factor-over-ℝ / Lazard-Rioboo-Trager) — **L,
  ~8-12 ciclos.** 🔨 **Sub-ciclo 1 ATERRIZADO 2026-06-21**: `1/(x^6-1)`, `1/(x^6-64)` (factorizan
  ENTEROS sobre ℚ) ya integran y verifican — subiendo el budget del multipoly del
  `algebraic_rational_zero_test` (el verifier ya hacía √c↦t, t²=c; solo no cabía el residual de
  grado 6). 🔨 **Sub-ciclo Cap. A ATERRIZADO 2026-07-14** (`d557556ea`): `1/(x^4-4)` (√2) y toda
  cuadrática racional Δ>0 (raíces reales irracionales) ahora integran como log-ratio real —
  render `build_indefinite_square_denominator_reciprocal_antiderivative` reusado dentro del
  ensamblador multi-cuadrático; residue sobre ℚ, surd sólo en el render. 🔨 **Sub-ciclo Cap. B ATERRIZADO 2026-07-14** (`6c4d59afc`): `1/(x^6+1)` (√3) y `1/(x^8-1)` (√2) —
  la cuártica par IRREDUCIBLE como FACTOR (`SquarefreeFactor::EvenQuartic` entero sobre ℚ, render por
  descenso de paridad). Plan de sub-ciclos en `docs/G1_RATIONAL_INTEGRATION_SCOPING.md`.
  🔨 **Nivel-2 COMPLETO 2026-07-15** (C-i QuadSurd `c25148522` → C-ii torre de radicales anidados
  `e54372c2a` → C-iii par conjugado ℚ(√t₀)): **`1/(x^5-1)` GRADÚA**. 🔨 **Cap. D ATERRIZADO
  2026-07-15**: **`1/(x^3-2)` GRADÚA** (extensión ℚ(∛k): torre degree-aware `t³=k`, triples exactos
  `[1,c,c²]`, radios planos `√3·c`). **G1 COMPLETO sobre los 5 probes del criterio #1.** Residuales
  menores (no bloquean): `1/(x^3+2)` (k<0), `1/(x^8+1)` (resolvente u⁴+1 surd), `1/(x^4-5)`
  (√5 polos reales); Cap. E (LRT universal) queda como terminal opcional.
- **G2 · Narrativa educativa de límites** (L'Hôpital / límite notable / squeeze /
  factor-y-cancela) — **L, ~6-10 ciclos — NÚCLEO MADURO (sub-ciclos 1-8 + ∞−∞ ATERRIZADOS; criterio
  de salida #2 CUMPLIDO 2026-07-15).** El baseline en que CADA límite colapsaba a un paso-cáscara
  único quedó SUPERADO — hoy narran cadenas multi-paso (ver la lista de métodos abajo). Es la mitad
  EDUCATIVA del north star — **pesa lo mismo que la universal**.
  🔨 **Sub-ciclos 1-8 ATERRIZADOS 2026-06-21**: infraestructura de narración de límites en el
  pipeline de enriquecimiento de `cas_didactic` (`generate_limit_substeps`) + **métodos nombrados**
  — notables `sin/tan/arcsin/arctan/sinh/tanh(u)/u→1`, `(eᵘ−1)/u→1`, `(aᵘ−1)/u→ln(a)`, `ln(1+u)/u→1`,
  `(1−cos u)/u²→1/2`, `(1+u)^(1/u)→e` y su gemelo en ∞ `(1+1/x)^x→e` (sub-ciclo 7, definición de e);
  **argumento escalado** `f(a·u)/u→a` (sub-ciclo 5: `sin(3x)/x→3`,
  `sin(x/2)/x→1/2`); **cruzado/denominador escalado** `f(a·u)/g(b·u)→a/b` (sub-ciclo 6: `sin(3x)/(2x)→3/2`,
  `tan(3x)/sin(2x)→3/2` cociente de DOS notables, `arcsin(x)/(5x)→1/5`); **teorema del sándwich**
  `u^k·sin/cos(1/u)→0`; **continuidad/sustitución directa** (polinomios); **factor-y-cancela** (0/0
  removible); y **dominancia en infinito** (cociente de coeficientes líderes / grado mayor → 0/±∞).
  **dominancia logarítmica/exponencial** `ln(x)/x→0`, `x²/eˣ→0`, `eˣ/x³→∞`, `√x/ln(x)→∞` (sub-ciclo 8:
  jerarquía `ln≪potencia≪exp` vía `enum LimitGrowthClass`, sound por confirmación del resultado).
  Todo sound por chequeo de resultado/grado, huella NONE. **Además ATERRIZADOS 2026-07-14/15**: raíz
  `(√(1+u)−1)/u→1/2`; el PUNTO del límite cableado al paso (sustitución concreta); **L'Hôpital
  ITERADO** (`(tan x−x)/x³` narra la regla N veces + sustitución final); y **∞−∞ en sus dos formas**
  (conjugado en ∞ `√(x²+x)−x` `bd64bfa3a`; común-denominador en punto finito `1/x−1/sin x`,
  `1/tan²x−1/x²→−2/3` `d600876e6`). Residual estrecho: imprimir las EXPRESIONES derivadas intermedias
  dentro de cada paso de L'Hôpital / el Taylor término-a-término, y la narración ∞−∞ de casos log/exp
  (el valor ya sale — `1/(x−1)−1/ln x→−1/2` — pero sin traza). `b^x` como base sigue pendiente.

### Wins P1 baratos y de alto ROI (intercalar con los gatekeepers)
- **Sintaxis `diff(expr, x, n)` (orden superior) y `diff(expr, x, y)` (parcial-mixta)** —
  ✅ **ATERRIZADO 2026-06-21.** `HigherOrderDiffRule` desugara `diff(f, x, n)`/`diff(f, x, y)`
  (y la sintaxis de conteos mixtos SymPy `diff(f, x, 2, y, 2)`) a `diff` anidados de dos
  argumentos, reusando toda la cascada de diferenciación existente. Órdenes 0/negativo,
  fraccionario y conteo-lidera-lista quedan como residuales honestos (sin evaluar). Peldaño:
  la narración de derivadas sucesivas YA aterrizó (cada orden emite su sub-árbol "Calcular la
  derivada"); resta pulir la presentación intermedia (mostrar la expresión resultante de cada orden,
  no solo la regla aplicada).
- **Sustitución-u general para `g` transcendente** —
  ✅ **ATERRIZADO 2026-06-21.** `transcendental_chain_substitution_antiderivative` integra
  `g'(x)·f(g(x))` con f ∈ {exp,sin,cos,sinh,cosh} por **guess-and-verify**: adivina `F(g)` y la
  acepta solo si `d/dx F(g) == integrando` EXACTO (la diferenciación ES el verificador → sound por
  construcción). `cos(x)·e^(sin x)→e^(sin x)`, `sin(x)·e^(cos x)→−e^(cos x)`, `e^x·cos(e^x)→sin(e^x)`,
  hiperbólicas, y escaladas vía linealidad. Peldaños: cofactores con constante no-unidad,
  `f(g)/x` (forma Div vs Mul-recíproco, `sin(ln x)/x`), y outer f más allá de exp/trig/hiperbólicas.
- **Comando `taylor()`/`series()` + linealidad de sumatorios** — M+S. Series está a ~28%
  (la más baja in-scope); `taylor_at_zero` ya existe interno, falta exponerlo.
  ✅ **Linealidad de sumatorios ATERRIZADA 2026-06-21**: `try_build_polynomial_sum` cierra
  cualquier sumando polinómico por Faulhaber término-a-término — `sum(2k)`, `sum(k^2+k)`,
  `sum(3k^2-k+1)`, con cota inferior simbólica. **Grado generalizado 2026-06-21** vía la
  recurrencia de Faulhaber (`power_sum_one_to` p≥4): `sum(k^4)`, `sum(k^5+k^2)`, … hasta grado 12.
  ✅ **`taylor()`/`series()` EXPUESTOS 2026-06-21**: `TaylorRule` sobre el motor Maclaurin
  interno (`taylor_at_zero`) — `taylor(exp(x), x, 0, 4)`, `sin`/`cos`/`tan`/`ln(1+x)`/`atan`/
  `asin` + polinomios, productos, composiciones. ✅ **Racionales/geométricos AÑADIDOS
  2026-06-21** (`taylor_at_zero_with_rational`, recíproco de series, aislado del path de
  límites): `1/(1-x)`, `1/(1+x^2)`, `1/(1-x)^2`, `exp(x)/(1-x)`. Residuales honestos: punto ≠ 0,
  polos en 0 (`1/x`), orden negativo. **El motor de series univariable que esto crea también
  desbloquea el Taylor de la Fase 3.** ✅ **Forma 2-args `taylor(f,x)`/`series(f,x)` con orden
  por defecto (Maclaurin, orden 6) AÑADIDA 2026-07-15 (`2122ae94a`)** — la invocación más
  natural (antes error de arity "función no definida"); requirió ampliar el gate
  `is_known_eval_engine_function` a arity `2..=4` además del arm de la regla. Con esto el win
  P1 del criterio de salida #3 (taylor/series + linealidad de sumatorios) queda CERRADO.
  ✅ **Aritmético-geométrica grado 2 AÑADIDA 2026-06-21**:
  `try_build_arithmetic_geometric_sum` generalizado a cofactor polinómico ≤2 (`Σ(αk²+βk+γ)·r^k =
  α·S₂+β·S₁+γ·S₀`) — `sum(k²·2^k)`, `sum((2k²−3k+1)·2^k)`, `sum((k²+2k)·2^k)`. Residuales honestos
  (peldaños, NO regresión, net-cero): (a) cofactores prónicos `k(k±1)=k²±k` por una oscilación
  factor↔distribuye del sumando en el orquestador (clase A); (b) ratio fraccionaria de grado 2
  (`k²·(1/2)^k`) por la normalización a forma Div `k²/2^k` (preexistente: `k/2^k` ya era residual).
  El builder es exacto en ambos (test fold-vs-fuerza-bruta lo fija). ✅ **Forma cociente Div
  AÑADIDA 2026-06-21**: la suma aritmético-geométrica escrita `p(k)/r^k` (clásico `Σ k/2^k = 13/8`
  en [1,4]) cierra leyendo la ratio 1/r del denominador. ✅ **Suma INFINITA convergente AÑADIDA
  2026-06-21**: `Σ_{k=a}^∞ p(k)·r^k` con |r|<1 → racional exacto (`Σ k/2^k = 2`, `Σ k²/2^k = 6`,
  `Σ_{0} k/3^k = 3/4`); colas `r/(1−r)`, `r/(1−r)²`, `r(1+r)/(1−r)³` corregidas por la cota inferior
  (refactor `decompose_arithmetic_geometric` compartido finito/infinito). Residuales (peldaños):
  cofactor grado ≥3, cota inferior simbólica, r irracional; grado-2 fraccionario (oscilación del
  orquestador).

### P2 / P3 (cobertura y pulido educativo)
- Verifier false-negative de `1/(x^6-1)` (la antiderivada YA es correcta; falla al no reducir
  `sqrt(3)·sqrt(3)`) — M, toca el verifier del bloque 12.
- Normalización `1/x^p → x^(-p)` hacia power-rule/maquinaria impropia (aparece en 3
  dimensiones) — S, alto ROI.
- ✅ **`x·a^x` por-partes ATERRIZADO 2026-06-21** (`polynomial_times_constant_base_power_antiderivative`:
  `∫p(x)·a^x = a^x·Σ(-1)^k p^(k)/(ln a)^(k+1)`, base racional positiva ≠ 1, exponente = var,
  cofactor grado 1..8; round-trip diff verificado); ∞−∞ diferencia de fracciones en límites (M); parámetros simbólicos
  en límites (M); cbrt en límites (S); evaluación definida del log-combinado de fracciones
  parciales (S-M).
- Educativo P3: localización de nombres de regla en inglés (S); `--steps` en CLI text, `+C`,
  artefactos `ln(e)`/`x^(2-1)`, fold `cos(0)=1` en FTC (todo S, cosmético).

### Residuo P0 (no urgente)
- FTC con cota inferior singular removible (`diff(∫ sin(t)/t [0,x]) → undefined` en vez de
  `sin(x)/x`): **under-answer conservador, NO wrong-answer.** No compromete soundness; importa
  para "completo", no para "serio".

---

## Fase 2 — Complejo elemental principal-branch + cálculo vectorial multivariable  **[ACTIVA 2026-07-16 — frente complejo ABIERTO]**

**⚑ ABIERTA 2026-07-16** (umbral de Fase 1 cruzado 2026-07-15; decisión del usuario). El
frente elegido primero es el **Complejo elemental**, re-validado con el scoping workflow
prometido abajo: `docs/FASE2_COMPLEX_ELEMENTAL_SCOPING.md` (6 mappers + síntesis + doble
verificación adversarial; secuencia de sub-ciclos A1→A4→A2→A5 + bloque B re-scopeable +
C transversales). **A1 (potencia Gaussiana) ATERRIZADO 2026-07-16** (hash en el ledger).
**Frente complejo COMPLETO 2026-07-17** (A+B+C1+approx; residuales de pulido en el ledger).
**El frente vectorial multivariable quedó SCOPEADO 2026-07-18** con su propio workflow:
`docs/FASE2_VECTORIAL_MULTIVARIABLE_SCOPING.md` (secuencia V0 soundness → V1‑V2 sustrato →
V3‑V6 verbos → V7‑V8 transversales; greenlight V0+A+B).

*Texto original (histórico):* **No se empieza hasta cruzar el umbral de Fase 1.**
*Estimación de los audits, sin re-medir:*
ambos serían **≈ M total y sin reescritura fundamental** sobre los cimientos reales. Es la
proyección de dos evaluaciones de un solo día (2026-06-21), no una medición asentada: **estas
estimaciones se re-validan con un scoping workflow al cruzar el umbral, no antes.** Cuál ir
primero es decisión de currículo/estrategia, no de viabilidad. (El único hecho estructural
duro fundado en audit es el scope-out de Riemann/multivaluado — ver final de Fase 3.)

### Complejo elemental (single-valued principal-branch — modelo binding de la estrategia)
- **C-álgebra** (ya parcialmente live): `(a+bi)^n`, builtins `conjugate`/`Re`/`Im`/`Arg`/
  `abs` complejo — S/M (extensión de enum + reglas gaussian-aware sobre `GaussianRational`).
- **C-elemental**: Euler `e^(iθ)`, `Log`/`Arg` principal, `ln(-1)→iπ`, potencias con cortes — S
  (slot directo en `define_rule!` + value_domain-gate; cero cambio arquitectónico).
- **Evaluador numérico complejo** (`num_complex`) — **M, la pieza cara.** `evaluator_f64`
  rechaza `Constant::I`, y el verificador cruza-chequea vía `eval_f64`, así que en modo
  complejo la red de soundness numérica está ausente: hay que respaldarla.

### Cálculo vectorial multivariable (CABLEADO barato — ~60-70% ya existe)

**⚑ SCOPEADO 2026-07-18** (workflow 6 mappers + síntesis + doble verificación adversarial, 90
anclas verificadas): `docs/FASE2_VECTORIAL_MULTIVARIABLE_SCOPING.md`.
**⚑ FORMALMENTE CERRADO 2026-07-18** (las 4 preguntas abiertas resueltas por el usuario:
steps JSON/REPL-only como contrato; `subs()` añadido — plano tangente y clasificación de
críticos one-shot; ∂ global multivariable; V7d ejecutado — puntos críticos por solve).
**Con ello la FASE 2 queda COMPLETA en sus dos mitades** (complejo elemental cerrado
2026-07-17 + pulidos tanda-2/3; vectorial multivariable cerrado hoy). La apertura de la
Fase 3 siguió la regla de oro: **decidida por el usuario 2026-07-18** (ver sección Fase 3).
**⚑ EJECUTADO 2026-07-18 (tanda de 8 ciclos, 0 rechazos): V0 (P0 métrica norm) → V1‑V2
(sustrato: diff componentwise + P0-wire narración) → V3‑V6 (los 6 verbos VIVOS:
gradient/grad, jacobian, hessian, divergence, laplacian, curl/rot — con narración es/en,
metamórficos de conservatividad y equiv bracket-aware) → V7a+b (abs(vector)→norm heredando
V0; integrate componentwise todo-o-nada). Hashes en el ledger. QUEDAN: V8 (pulido — gated
por pregunta abierta ∂), V7d y las 4 preguntas abiertas del scoping al usuario.** La estimación de abajo
se quedó CORTA (wronskian es el template exacto de verbo; el desugar de orden superior es
target-agnóstico) y el barrido adversarial destapó un candidato P0 preexistente (capa métrica
de `Matrix`: `norm` gaussiano/simbólico sin gate de dominio — sub-ciclo **V0**, va primero).
Secuencia: `V0` soundness → `V1‑V2` sustrato (diff componentwise + P0-wire de narración) →
`V3‑V6` los 6 verbos → `V7‑V8` transversales.

*Estimación original (histórica):*
- **Cables que faltan** — S: aridad lista-de-vars en extractores, y **diff componentwise
  sobre nodo `Matrix`** (`diff([x^2,x^3],x)`).
- **Gradiente / Jacobiano / Hessiano / divergencia / rotacional / Laplaciano** — S-M cada uno:
  "registrar verbo + map sobre `[vars]` + ensamblar `Matrix`", reutilizando
  `differentiate_symbolic_expr` sin tocar. El tipo vector/campo ya existe (nodo `Matrix` n×1).
- (Derivadas parciales, integrales iteradas/múltiples, álgebra multipoly, matrices, sistemas:
  **YA funcionan** — re-verificado 2026-07-18 contra el CLI vivo, con un matiz: `inverse`/
  `rref`/`rank`/`eigen*` son numéricos-exactos; con entradas simbólicas declinan honestos.)

---

## Fase 3 — Capas analíticas  **[ACTIVA 2026-07-18 por decisión del usuario — SCOPEADA]**

**⚑ SCOPEADA 2026-07-18** (workflow 6 mappers + síntesis + verificador de anclas 112/117 +
crítico de completitud, 12 gaps integrados): `docs/FASE3_ANALYTIC_LAYERS_SCOPING.md`.
Secuencia **F0 → F12**: **F0 va PRIMERO y es P0 de soundness** — el subsistema de límites es
domain-blind (cero `ValueDomain`) y bajo `--value-domain complex` FABRICA valores con
razonamiento real (`limit(e^(-1/z^2),z,0)→0` cuando en ℂ el límite NO existe —
esencial-singular; `limit(z*sin(1/z),z,0)→0` ídem; ambos re-verificados a mano): kill-switch
de dominio en el chokepoint del dispatcher, molde de los 21 gates de `complex.rs`. Después:
bloque A Taylor (F1 sustrato univar, F2 multi-índice grado-total), bloque B ensambladores
(F3 subs no-op + integrate definido componentwise, F4 lineintegral, F5 surface_integral,
F6 potential), bloque C límites multivar (F7 superficie+continuidad probada, F8 DNE-por-caminos
con testigos citados — JAMÁS existencia desde finitos caminos, F8b squeeze-positivo, F9
iterados), bloque D re-otorgo complejo selectivo (F10 fold de output, F11 sustitución compleja
entera con combinador bilateral gateado, F11b DNE-por-caminos complejo, F12 opcional π/√2).
Las 4 preguntas abiertas del scoping RESUELTAS por el usuario 2026-07-19: F8b/F11b entran;
residuos FUERA del norte (punt documentado, π/e con dueño); case-split paramétrico = frente
propio post-Fase 3. Estimación original de abajo (histórica) superada por el scoping:

- **Series/Taylor**: motor de series univariable (si no aterrizó en Fase 1) → **Taylor
  multivariable** vía book-keeping multi-índice (`∂^α/α!`).
- **Límites complejos** y **límites multivariables**: ambos capas analíticas nuevas. Los
  multivariables son path-dependent e indecidibles en general → **detectar no-existencia /
  punt honesto, NUNCA fabricar respuesta** (coherente con la disciplina de soundness exacta).
- **Integrales de línea/superficie**: parametrización + pullback (subsistema nuevo;
  substitute+integrate ya existen como primitivas).
- **Residuos / integración por contornos**: matemática net-new; la estrategia lo cuestiona
  explícitamente para el scope educativo — **no planificar sin un caso curricular**.

**FUERA del norte (no cuenta como faltante):** análisis complejo completo / multivaluado /
Riemann (cambiar `BranchPolicy::Principal` por Riemann sería la ÚNICA reescritura profunda —
deliberadamente fuera de scope); funciones especiales (erf/Γ/Si/Ei/LambertW) como
valores de salida. Y los residuales no-elementales protegidos (`e^(-x^2)` (indefinida —
la DEFINIDA gaussiana ya está soportada), `sin(x)/x`,
`1/ln(x)`, oscilantes, divergentes): **resolverlos sería un bug de soundness, no un avance.**
*(Las EDOs SALIERON de esta lista el 2026-07-19 por decisión del usuario — ver Fase 4.)*

---

## Fase 4 — Ecuaciones diferenciales elementales  **[ENTRA AL NORTE 2026-07-19 por decisión del usuario — SCOPING EN CURSO]**

**Decisión del usuario 2026-07-19** ("quiero que nuestro engine sea realmente universal"):
las EDOs dejan de ser fuera-del-norte y se convierten en la Fase 4. Alcance: el **curso
elemental de EDOs, universal Y educativo** — separables, lineales de primer orden (factor
integrante), exactas (la maquinaria de `potential` F6 ES el corazón: `M dx + N dy = 0`
exacta ⟺ ∃φ con ∇φ=(M,N), con la emisión gateada por verificación), segundo orden
homogéneas de coeficientes constantes (polinomio característico → `solve` exacto;
`wronskian` verifica independencia), no-homogéneas por coeficientes indeterminados
(familias acotadas), condiciones iniciales, y sistemas lineales 2×2 por autovalores
(exactos racionales ya vivos). Soluciones en serie vía Taylor como opcional tardío.

**Los dos hallazgos estructurales del sondeo previo (2026-07-19, CLI vivo):**
1. **La maquinaria actual COLAPSA la notación de EDOs**: `diff(y,x)` → `0` (la incógnita
   se trata como constante) y `solve(diff(y,x)=y, y)` → `{0}` — el comando `dsolve` DEBE
   interceptar a nivel WIRE (el molde de `solve`/`solve_system`, que ya parsea ecuaciones
   textualmente antes de que simplify las toque) y mantener las derivadas de la incógnita
   opacas hasta el despachador de métodos.
2. **`dsolve` como ACCIÓN, no como rule**: la verificación por sustitución (derivar la
   solución candidata y comprobar la EDO) necesita el evaluador COMPLETO (el molde
   `equiv_difference_evaluates_to_zero` de `actions.rs`) y el canal de warnings — ambos
   viven en la capa de acción, no en las rules (lección F1/F7/F8: un rule que declina no
   tiene canal, y `poly_eq` no cubre soluciones trig/exp).

**Guardrails de la fase** (heredan los inter-fase + los específicos): (a) la EMISIÓN la
gatea la VERIFICACIÓN por sustitución con el evaluador completo — jamás se emite una
solución no verificada (doctrina F6); (b) familia de soluciones con constante(s) `C` como
CONTRATO del wire; (c) lo no-elemental queda residual honesto (Riccati general, coeficientes
variables no-triviales, no-lineales sin método — la política de siempre); (d) `dsolve` se
registra SOLO en su ciclo O0 con migración documentada del never-confirm (decisión D10 —
hoy es PRESA del detector y así sigue hasta ese ciclo).

**Secuenciación**: **⚑ SCOPEADA 2026-07-19** (workflow 6 mappers + síntesis + verificador de
anclas 98/106 + crítico 13 gaps): `docs/FASE4_ODE_ELEMENTAL_SCOPING.md` — secuencia
**O0→O9** (O0 sustrato wire+acción+verificador+separables [L]; O1 lineal; O2 exactas vía
`try_potential_expr`; O3 IVP; O4 2º orden homogénea por discriminante exacto interno; O5
coeficientes indeterminados [el neto-nuevo mayor]; O6 sistemas 2×2 eigen-interno; O7
superficie; O8 Bernoulli+homogéneas — NO opcional; O9 series/μ(x)/Cauchy-Euler opcionales),
17 decisiones cerradas D1-D17, catálogo de ~37 probes con oráculo verificado 36/39 y 7
never-fabricate Z1-Z7. **Ejecución recomendada tras cerrar el núcleo
restante de la Fase 3** (F9 iterados, F8b squeeze, bloque D F10/F11 — ~4-5 ciclos): las
dependencias de la Fase 4 (Fase 1 + F6) YA están cumplidas, así que el usuario puede
reordenar sin coste técnico; la regla de oro se mantiene con el umbral de cierre de Fase 3
como frontera por defecto.

---

## Guardrails inter-fase (OBLIGATORIOS en cada ciclo de Fase 1)

Son la razón de que el orden real-primero sea correcto: **seguirlos no cuesta más hoy y vuelve
las Fases 2/3 ≈ M en vez de L.** Derivan de los "Deferred Horizons" de la estrategia y están
confirmados como puntos de extensión load-bearing por el audit.

1. **Enhebra `ValueDomain` en toda regla nueva cuyo resultado dependa del dominio de
   valores** (log/sqrt/exp/potencias/inversas). Gatea esas reglas real-only
   (`value_domain() == RealOnly => return None`) — como ya hacen log/sqrt/exp. NUNCA
   hard-codees RealOnly en un contrato público donde una representación estructurada cuesta
   lo mismo. Las reglas puramente **sintácticas, de presentación o de narración** (p.ej.
   `diff(expr,x,n)`, linealidad de sumatorios, trazas) NO necesitan el gate: añadirlo es
   código muerto + un contrato RealOnly engañoso sobre algo sin análogo complejo.
2. **Mantén diff/integrate parametrizados por variable** (per-variable; nunca sesgo
   single-var). Es lo que hace que las parciales y las iteradas ya funcionen.
3. **Predicados de condición estructurados y extensibles** (cortes de rama, condiciones de
   dominio), no supuestos real-only horneados en los contratos.
4. **Backstop de soundness domain-aware y EXACTO** (`BigRational`, patrón `*_in_domain` de
   `arithmetic_cancel_support`). Nunca f64 para decisiones keep/drop.
5. **Resultados como contrato**: cargan las decisiones de rama/dominio que hicieron (espeja
   el result-model del backend). Sin "branch hops" silenciosos en limpieza de display.
