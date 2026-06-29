# Cerrando el dominio real

> Revisión de cierre: ¿es el engine **universal sobre el dominio real** (Fase 1) y
> puede pasarse a la fase de dominio complejo?
>
> **Veredicto: `close_after_p0_fixes`** — el dominio real NO está cerrado todavía.
> Había **4 clusters de respuesta-incorrecta confirmados** que bloquean el cierre de
> soundness. Son pocos, localizados y enumerables; al corregirlos, el dominio real
> queda cerrado (sin wrong-answers abiertos) y se cumple la puerta de soundness
> para la Fase 2.
>
> **Progreso:** ✅ **P0-1** · ✅ **P0-2** (commits pendientes) · ⬜ P0-3 · ⬜ P0-4.

- **Fecha:** 2026-06-29
- **Método:** workflow multi-agente de 14 frentes (probe del CLI real → verificación
  adversarial → síntesis + crítico de completitud), `wf_17e9c23f-2e4`, 71 agentes.
- **Validación independiente:** los 4 P0 fueron **reproducidos a mano** contra
  `target/release/cas_cli` (cada uno devuelve `ok=true` con valor/conjunto erróneo,
  NO una declinación honesta). Esto **falsifica** la creencia previa de "0
  wrong-answers" — algunos eran ítems "(F)" ya abiertos en el frontier audit,
  mal clasificados como carencias de capacidad en vez de respuestas incorrectas.

---

## 1. Distinción clave: cierre de *soundness* vs "universal" fuerte

| Concepto | Criterio | Estado |
|---|---|---|
| **Cierre de soundness** (puerta para Fase 2) | Cero wrong-answers abiertos en ℝ; lo no soportado se **declina honestamente** | ❌ Bloqueado por 4 P0 |
| **"Universal" fuerte** | Toda capacidad elemental real de un curso universitario produce forma cerrada | ⚠️ Falta la mochila P1 (declinan honestamente — NO bloquean soundness) |

Declinar honestamente una entrada no-elemental o fuera de alcance es un **PASS**, no
un fallo. Una **respuesta incorrecta** (valor mal, condición de dominio perdida o
inventada, conjunto insólido) es siempre un **P0** que bloquea el cierre.

---

## 2. Bloqueadores P0 (confirmados a mano)

Forma recurrente de los cuatro: *un detector de forma BARE/principal descarta el
envoltorio* (periodicidad / caso de signo / polo).

### P0-1 — Ecuaciones trig periódicas colapsan tras reducir ángulo doble  ✅ RESUELTO (commit pendiente)

```
ANTES                                  AHORA (correcto)
solve(cos(2x)-cos(x)) => { 0 }    =>   { 2kπ, 2π/3+2kπ, 4π/3+2kπ }   (= {2kπ/3})
solve(sin(2x)-sin(x)) => { 0, π/3 } => { 2kπ, π/3+2kπ, π+2kπ, 5π/3+2kπ }
solve(cos(2x)+cos(x)) => { π, π/3 } => { π+2kπ, π/3+2kπ, 5π/3+2kπ }
solve(sin(x)*cos(x)=0)=> {0, π/2}  =>  { kπ, π/2+kπ }
```

Emitía conjuntos finitos `ok=true` que descartaban familias enteras y toda la
periodicidad, presentados como completos.

- **Causa raíz:** el solver periódico corre en la entrada top-level pero la ruta
  RECURSIVA (que resuelve cada `factor=0` de un producto-cero) lo saltaba y caía a
  la inversa-unaria → solo la raíz PRINCIPAL.
- **Fix (3 partes):** (1) la ruta recursiva prueba el solver periódico primero →
  cada factor trig devuelve su familia `Periodic` completa; (2) la agregación de
  producto-cero (contexto inmutable) declina honestamente a residual; (3) un
  post-proceso en cas_solver (contexto mutable) une las familias sobre un periodo
  común (periodo = racional·π; común = `lcm`; expande y deduplica bases módulo
  periodo, EXACTO con `BigRational`). Productos mixtos (`(x-1)·sin(x)`) se quedan
  residual honesto.
- **Validación:** workspace failed:0 + test de contrato; clippy limpio; huella
  guardrail+pressure IDÉNTICA.
- **Frente:** `solve-equations`. *(Distinto y NO incluido: la cuadrática-en-trig
  con término lineal `sin(x)^2+sin(x)=0` garbla un residual `arcsin(...)` — es la
  ruta de sustitución u=trig, un capability gap P1 separado.)*

### P0-2 — Inecuaciones recíprocas de potencia impar pierden el cambio de signo en x=0  ✅ RESUELTO (commit pendiente)

```
ANTES                                   AHORA (correcto, verificado numéricamente)
solve(2/x^3 > -1)  => (-2^(1/3), ∞)  => (-∞,-2^(1/3)) ∪ (0,∞)
solve(1/x^3 > 2)   => (-∞,(1/2)^⅓)   => (0, (1/2)^(1/3))
solve(1/x^4 > 1/4) => (-4^¼, 4^¼)    => (-4^(1/4),0) ∪ (0,4^(1/4))   [excluye el polo 0]
solve(1/x^5 > 2)   => (-∞,(1/2)^⅕)   => (0, (1/2)^(1/5))
solve(2/x^3 < -1)  => (-∞,-2^⅓)      => (-2^(1/3), 0)
```

Para `c/x^(impar≥3)` (y la familia surd-border `1/x^4`) el sign-chart que cruza
el polo x=0 colapsaba a un solo rayo.

- **Causa raíz (dos capas):** (1) el candidato del sign-split ya era correcto, pero
  el verificador numérico solo ordenaba cotas racionales y surds **cuadráticos** —
  una cota raíz-cúbica/cuártica no se podía ordenar → declinaba → fallback erróneo;
  (2) más profundo, `compare_values` (compartido) caía a comparación **estructural**
  (ciega al valor) para cotas raíz-n → construía mal el candidato.
- **Fix:** `compare_values` gana orden EXACTO de `signo·q^(1/n)` (comparar elevando
  a la potencia común `lcm`); el verificador gana el mismo orden de cotas raíz-n; cap
  de grado 4→6 (cubre `1/x^5`), con la verificación numérica como red de soundness.
  Todo **exacto** (`BigRational`, nunca f64).
- **Validación:** workspace failed:0 + 2 tests de contrato; clippy limpio; huella
  guardrail+pressure IDÉNTICA (pese a tocar el compartido `compare_values`).
- **Frente:** `solve-inequalities`. Cierra el ítem de roadmap "(F) bordes surd orden ≥3".

### P0-3 — Integral definida/FTC inventa polos desde la racionalización → `undefined` falso

```
integrate(1/(sqrt(x)*(1+x)), x, 1, inf)   =>  undefined   (verdad: π/2)
integrate(1/(sqrt(x)*(1+x)), x, 1/2, 4)   =>  undefined   (verdad: ~0.983, intervalo regular propio)
integrate(1/(sqrt(x)*(1+x)), x)           =>  2·arctan(sqrt(x))   ✓ (indefinida correcta)
```

Racionalizar `sqrt(x)*(1+x)` introduce raíces de denominador x=0, x=1 que NO son
polos del integrando; el x=1 dentro del intervalo (cuando el borde inferior ≤1)
dispara un rechazo falso de "polo en el intervalo". Fabricar divergencia en una
integral convergente / regular es una violación de soundness.

- **Fix:** correr el chequeo de polo/singularidad sobre el **integrando original**
  (no sobre el denominador racionalizado), o llevar la antiderivada
  `2·arctan(sqrt(x))` directamente por FTC.
- **Frente:** `definite-ftc`.

### P0-4 — Regla de racionalización de raíz n-ésima da valor erróneo

```
integrate(1/x^(1/4), x, 0, 1)   =>  4   (verdad: 4/3)
integrate(1/x^(1/6), x, 0, 1)   =>  6   (verdad: 6/5)
integrate(1/x^(1/4), x)         =>  4·x^(1/4)   ✗  (incluso la indefinida está mal)
```

La regla "Rationalize Product Denominator" trata todo denominador de raíz n-ésima
como si multiplicar por `x^(1/2)` lo despejara. Incluso la **indefinida** es
incorrecta: `d/dx[4·x^(1/4)] = x^(-3/4) ≠ x^(-1/4)`. La antiderivada correcta es
`(4/3)·x^(3/4)`. El hermano raíz-cúbica `1/x^(1/3)` solo declina (carencia), pero
comparte la misma regla rota.

- **Fix:** reemplazar la regla por el plegado directo de potencia
  `1/x^(1/n) = x^(-1/n)`; verificar que cada antiderivada deriva de vuelta al
  integrando.
- **Frente:** `definite-ftc`.

---

## 3. Mochila de capacidad para "universal" (NO bloquea soundness)

Estas **declinan honestamente** (sin valor incorrecto); bloquean la promesa
"universal" fuerte, no el cierre de soundness. Esfuerzo: S/M/L.

| Frente | Hueco | Esf. |
|---|---|---|
| radical-integration | Split de linealidad `p(x)/√(cuadrática)` antes del dispatch radical (familias asinh/arcsin/acosh) | S |
| radical-integration | Denominadores radicales `(a±x²)^(impar/2)` vía sustitución trig/hiperbólica + reducción | M |
| radical-integration | Sustitución racionalizante de raíz orden >2 (`t=x^(1/lcd)`) y cofactores `p(x)·(ax+b)^(m/n)` | M |
| trig-integration | `sin^m/cos^n` con denominador n≥4 (familia sec^k) + reescritura `sin²=1-cos²` | S |
| transcendental-integration | Componer u-sub∘por-partes y producto triple `poli·trig·exp` | M |
| solve-equations | SOLVE cuadrática-en-trig / exponencial base-común y base-distinta / hiperbólica (u=g(x) + back-sub). **Además: arregla el crash `E_INTERNAL` → declinar honestamente** | M |
| solve-inequalities | **Inecuaciones trig periódicas** (variante SolutionSet periódica: `cos(x)>0`, `sin(x)>1/2`, `tan(x)>1`) — mayor palanca restante | L |
| series-sums | Maclaurin binomio-fraccionario y Taylor de singularidad removible en centro 0 | S |
| abs-piecewise | `∫|f|` con f **no lineal** (`∫|x²-1|`, `∫|sin x|` quedan sin evaluar; el lineal y la definida con bordes numéricos sí funcionan) — *hallado por el crítico* | S/M |
| limits (menor) | `1^∞` en punto finito con `ln(base)·exp→0` (límite=1); singularidad removible de raíz general | S |
| algebra (menor) | `apart` de numerador monomio sobre factor lineal repetido; `x·|x|=c` piecewise solve | S |

---

## 4. Áreas sólidas (cobertura medida)

| % | Frente | Nota |
|---|---|---|
| 96% | differentiation | per-variable y orden superior; condiciones de dominio sólidas; 0 wrong |
| 94% | algebra-core | factor/gcd/expand/simplify sólidos; `log(x²)=2ln|x|` con x≠0, `√(x²)=|x|` |
| 90% | limits | notables, L'Hôpital, Taylor-order, `1^∞` familia `e^k`, squeeze, one-sided, DNE |
| 90% | abs-piecewise-domain | sign charts, abs multi-término, simetría par/impar, bordes singulares honestos |
| 88% | rational-integration (sobre ℚ) | todo denominador ℚ-factorizable → antiderivada verificada; 0 wrong |
| 85% | series-sums | Maclaurin/Taylor exacto vs sympy; Faulhaber/geométrica/telescópica/Basel |
| 85% | transcendental-integration | por-partes y u-sub canónicos correctos; no-elementales declinan |
| 85% | trig-integration | (n=1,2,3 ok; n≥4 es el hueco) |
| 82% | special-numeric-edge | evaluación, diff, identidades, denesting; inversas hiperbólicas fuera de ℝ quedan simbólicas |
| 82% | definite-ftc | básicas + impropias-a-∞ + bordes singulares correctos (el `undefined` falso está confinado a la ruta de racionalización) |
| 78% | solve-equations | (el hueco es quadratic-in-trig/exp/hiperbólica + P0-1) |
| 72% | solve-inequalities | sólido salvo P0-2 y las periódicas |
| 72% | radical-integration | (la mochila radical es el grueso del hueco "universal") |
| 72% | solve-systems | 2×2/3×3/4×4 exacto correcto; no-lineal/paramétrico fuera de Fase 1 |

---

## 5. Fuera de alcance (correctamente excluido — NO bloquea el cierre)

- **Integrales indefinidas no-elementales** que requieren funciones especiales:
  `e^(-x²)` (erf), `sin(x)/x` (Si), `1/ln(x)` (Ei/Li), `x^x` — declinar es correcto.
- **Radicandos genuinamente no-elementales:** `√(1+x³)`, `√(1+x⁴)`, `√(sin x)`.
- **Factor-over-ℝ / Lazard-Rioboo-Trager** racional (`1/(x⁶+1)`, `1/(x⁵-1)`, …):
  en el roadmap nombrado; el engine declina limpiamente en vez de adivinar. Bloquea
  la promesa fuerte "G1 universal" de integración, NO el cierre de soundness.
- **Oscilantes/divergentes:** `lim sin(1/x)` en 0, `∫sin(x)` en `[0,∞)`.
- **Sistemas no-lineales / paramétricos / multivariable no-cuadrados** — fuera de la
  Fase 1 univariable (el parse-error de lista y `E_INTERNAL` paramétrico son verrugas
  de UX, no de soundness).
- **Hiperbólicas indefinidas con gudermanniano:** `1/cosh^k`, `sech(x)`.
- **Semántica de potencia real** `(x²)^(1/3)→x^(2/3)`, `(-8)^(2/3)=4` — FALSO
  POSITIVO documentado, NO bug.
- **Convención `log(a,b)=log_a(b)`** (primer arg = base) — FALSO POSITIVO recurrente.
  `solve(log(x,2)=3) → {2^(1/3)}` es auto-consistente, no un bug.
- **Clase I / Deferred Horizons:** complejo multivaluado/Riemann, ODEs, funciones
  especiales como SALIDA (erf/Γ/Si/Ei/LambertW), diferenciación implícita, `f(x)`
  abstracta, assumptions, valor principal, Risch/Gruntz completo.
- **C5:** diferenciación `(poli+tan)^n` cuelga honestamente — diferido (no wrong-answer).
- **Verrugas cosméticas** (mismo valor, escritura no-canónica): `sinh(ln2)` sin
  plegar a `3/4`, `cosh(asinh(x))` sin plegar a `√(x²+1)`, etc. — P3 didáctico.

---

## 6. Secuencia recomendada hacia el cierre

Cada P0 es un ciclo validado con su huella (scorecard guardrail+pressure idéntica).

1. ~~**P0-1** — *lift* periódico en ecuaciones trig~~ ✅ **HECHO** (commit
   pendiente): ruta recursiva periodic-aware + unión de familias por periodo común;
   pin de contrato `test_eval_periodic_trig_product_equation_unions_families`.
2. ~~**P0-2** — sign-chart con split en el polo x=0~~ ✅ **HECHO** (commit
   pendiente): orden exacto de cotas raíz-n en `compare_values` + el verificador, cap
   4→6; pin `test_eval_reciprocal_power_inequality_keeps_pole_sign_split` +
   `compare_values_orders_nth_root_bounds_by_value`.
3. **P0-3** — chequeo de polo sobre el integrando ORIGINAL. Pin de
   `1/(√x·(1+x))` en `[1/2,4]`, en `[1,∞)=π/2`, + el intervalo regular-propio como
   caso de soundness.
4. **P0-4** — plegado de potencia `x^(-1/n)`. Pin de `1/x^(1/4)=4/3`, `1/x^(1/6)=6/5`,
   y el antes-declinante `1/x^(1/3)=3/2`; verificar derivada de vuelta.
5. **GATE** — `cargo test --workspace` + `clippy --workspace --all-targets` + rustfmt
   + engine-fast + scorecard + pressure; huella estructuralmente idéntica. **Aquí el
   dominio real queda CERRADO (sin wrong-answers abiertos) → puerta de soundness para
   Fase 2 cumplida.**
6. **UNIVERSAL-1** (P1, post-cierre): wins baratos de integración radical/trig que
   comparten causa (split linealidad, `sin^m/cos^n` n≥4, `(a±x²)^(impar/2)`).
7. **UNIVERSAL-2** (P1): `u=g(x)` + back-sub en SOLVE (quadratic-in-trig, exponencial,
   hiperbólica; arreglar el `E_INTERNAL` para declinar sin crashear).
8. **UNIVERSAL-3** (P1): sustitución raíz orden>2; Taylor binomio-fraccionario /
   singularidad removible en 0; `1^∞` punto finito; `∫|f|` no-lineal.
9. **UNIVERSAL-4** (P1, L): variante SolutionSet periódica para inecuaciones trig
   (`cos>0`, `sin>1/2`, `tan>1`).

---

## 7. Procedencia

- Workflow: `real-domain-closure-review` (`wf_17e9c23f-2e4`), 71 agentes,
  ~2.07M tokens, ~930s. 5 doc-readers (umbral + residuales honestos) → 14 frentes
  probe → verificación adversarial por hallazgo → síntesis + crítico de completitud.
- El crítico de completitud descartó un FALSO P0 (`log(a,b)=log_a(b)`) y añadió el
  hueco `∫|f|` no-lineal; confirmó que no había nuevos P0 en las esquinas peor
  cubiertas.
- Los 4 P0 fueron reproducidos a mano antes de reportarse (memoria avisa:
  *hand-verify cada P0 — la verificación adversarial no es infalible*).
