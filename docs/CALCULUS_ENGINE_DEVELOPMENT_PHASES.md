# Hoja de ruta por fases — del real-univariable al multivariable y complejo

Documento de secuenciación del north star (`CALCULUS_ENGINE_STRATEGY.md`: engine de
cálculo diferencial/integral **universal Y educativo** en dominio real). Fija el ORDEN de
los horizontes y, para cada uno, los items concretos. Fundado en dos evaluaciones
multi-agente (2026-06-21): la de completitud de la frontera real, y la de extensibilidad
arquitectónica a complejo y multivariable.

**Regla de oro:** la **Fase 1 es el ÚNICO objetivo activo**. Las Fases 2 y 3 NO se empiezan
hasta cruzar el umbral de Fase 1 — existen aquí para que las decisiones de HOY las
mantengan baratas (ver §Guardrails inter-fase). No son "deferred = abandonadas"; son
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
  definida/impropia ~62-75%, límites ~68%, **series ~28%**, **educativo bimodal** (diff/integrate
  narran; **límites ~0% educativo**).
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
2. **Límites con cadena didáctica no-cáscara** en `cas_didactic` (límites-educativo deja de
   ser ~0%: existe al menos L'Hôpital / límite notable / squeeze / factor-cancela narrados).
3. **Los wins P1 baratos aterrizados** (`diff(x,n)`/`diff(x,y)`, `taylor()`/`series()` +
   linealidad de sumatorios, sustitución-u general transcendente).

*Estimación de duración (no es el gate): **~25-40 ciclos efectivos**, dominados por dos*
*gatekeepers clase L.*

### Gatekeepers (máxima prioridad — desbloquean cada mitad del north star)
- **G1 · Integración racional UNIVERSAL** (factor-over-ℝ / Lazard-Rioboo-Trager) — **L,
  ~8-12 ciclos.** 🔨 **Sub-ciclo 1 ATERRIZADO 2026-06-21**: `1/(x^6-1)`, `1/(x^6-64)` (factorizan
  ENTEROS sobre ℚ) ya integran y verifican — subiendo el budget del multipoly del
  `algebraic_rational_zero_test` (el verifier ya hacía √c↦t, t²=c; solo no cabía el residual de
  grado 6). Residual GENUINAMENTE net-new (necesita factor-over-ℝ/LRT): `1/(x^5-1)` (Φ5/√5),
  `1/(x^6+1)`, `1/(x^8-1)`, `1/(x^4-4)` (√2), `1/(x^3-2)` (∛2). Es la promesa definitoria de
  "universal" en integración y el item remanente declarado de la Phase 4 del backend.
- **G2 · Narrativa educativa de límites** (L'Hôpital / límite notable / squeeze /
  factor-y-cancela) — **L, ~6-10 ciclos.** Hoy CADA límite colapsa a un paso-cáscara único
  (salto mágico que los docs prohíben). Es la mitad EDUCATIVA del north star — **pesa lo mismo
  que la universal**. *Prioridad sobre varios P2 de cobertura: mientras los límites no narren, el
  umbral "serio Y educativo" no se cruza.*
  🔨 **Sub-ciclos 1-5 ATERRIZADOS 2026-06-21**: infraestructura de narración de límites en el
  pipeline de enriquecimiento de `cas_didactic` (`generate_limit_substeps`) + **métodos nombrados**
  — notables `sin/tan/arcsin/arctan/sinh/tanh(u)/u→1`, `(eᵘ−1)/u→1`, `(aᵘ−1)/u→ln(a)`, `ln(1+u)/u→1`,
  `(1−cos u)/u²→1/2`, `(1+u)^(1/u)→e`; **argumento escalado** `f(a·u)/u→a` (sub-ciclo 5: `sin(3x)/x→3`,
  `sin(x/2)/x→1/2`, escala leída con `Polynomial::from_expr`, sound por confirmación de resultado);
  **teorema del sándwich** `u^k·sin/cos(1/u)→0`; **continuidad/sustitución directa** (polinomios);
  **factor-y-cancela** (0/0 removible); y **dominancia en infinito** (cociente de coeficientes líderes
  / grado mayor → 0/±∞). Todo sound por chequeo de resultado/grado, huella NONE. Siguiente sub-ciclo:
  denominador escalado `sin(u)/(a·u)→1/a` y cruzado `sin(au)/(bu)→a/b`; raíz `(√(1+u)−1)/u→1/2`;
  `(1+1/x)^x→e` y `ln(x)/x→0` en ∞; y (arquitectónico) cablear el PUNTO del límite al paso para mostrar
  la sustitución concreta, narrar L'Hôpital/Taylor paso a paso, y la dominancia EXPONENCIAL.

### Wins P1 baratos y de alto ROI (intercalar con los gatekeepers)
- **Sintaxis `diff(expr, x, n)` (orden superior) y `diff(expr, x, y)` (parcial-mixta)** —
  ✅ **ATERRIZADO 2026-06-21.** `HigherOrderDiffRule` desugara `diff(f, x, n)`/`diff(f, x, y)`
  (y la sintaxis de conteos mixtos SymPy `diff(f, x, 2, y, 2)`) a `diff` anidados de dos
  argumentos, reusando toda la cascada de diferenciación existente. Órdenes 0/negativo,
  fraccionario y conteo-lidera-lista quedan como residuales honestos (sin evaluar). Peldaño:
  narración paso-a-paso de las derivadas sucesivas (hoy solo se muestra la forma final).
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
  desbloquea el Taylor de la Fase 3.** ✅ **Aritmético-geométrica grado 2 AÑADIDA 2026-06-21**:
  `try_build_arithmetic_geometric_sum` generalizado a cofactor polinómico ≤2 (`Σ(αk²+βk+γ)·r^k =
  α·S₂+β·S₁+γ·S₀`) — `sum(k²·2^k)`, `sum((2k²−3k+1)·2^k)`, `sum((k²+2k)·2^k)`. Residuales honestos
  (peldaños, NO regresión, net-cero): (a) cofactores prónicos `k(k±1)=k²±k` por una oscilación
  factor↔distribuye del sumando en el orquestador (clase A); (b) ratio fraccionaria de grado 2
  (`k²·(1/2)^k`) por la normalización a forma Div `k²/2^k` (preexistente: `k/2^k` ya era residual).
  El builder es exacto en ambos (test fold-vs-fuerza-bruta lo fija). ✅ **Forma cociente Div
  AÑADIDA 2026-06-21**: la suma aritmético-geométrica escrita `p(k)/r^k` (clásico `Σ k/2^k = 13/8`
  en [1,4]) cierra leyendo la ratio 1/r del denominador. Residuales (peldaños): grado-2 fraccionario
  (misma oscilación del orquestador) y la suma INFINITA convergente `Σ k·r^k` con |r|<1.

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

## Fase 2 — Complejo elemental principal-branch + cálculo vectorial multivariable  **[SIGUIENTE, gated]**

**No se empieza hasta cruzar el umbral de Fase 1.** *Estimación de los audits, sin re-medir:*
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
- **Cables que faltan** — S: aridad lista-de-vars en extractores, y **diff componentwise
  sobre nodo `Matrix`** (`diff([x^2,x^3],x)`).
- **Gradiente / Jacobiano / Hessiano / divergencia / rotacional / Laplaciano** — S-M cada uno:
  "registrar verbo + map sobre `[vars]` + ensamblar `Matrix`", reutilizando
  `differentiate_symbolic_expr` sin tocar. El tipo vector/campo ya existe (nodo `Matrix` n×1).
- (Derivadas parciales, integrales iteradas/múltiples, álgebra multipoly, matrices, sistemas:
  **YA funcionan**.)

---

## Fase 3 — Capas analíticas  **[DESPUÉS, L, varianza alta — mayormente net-new]**

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
deliberadamente fuera de scope); EDOs; funciones especiales (erf/Γ/Si/Ei/LambertW) como
valores de salida. Y los residuales no-elementales protegidos (`e^(-x^2)` (indefinida —
la DEFINIDA gaussiana ya está soportada), `sin(x)/x`,
`1/ln(x)`, oscilantes, divergentes): **resolverlos sería un bug de soundness, no un avance.**

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
