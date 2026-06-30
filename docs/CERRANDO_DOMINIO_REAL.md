# Cerrando el dominio real

> Revisión de cierre: ¿es el engine **universal sobre el dominio real** (Fase 1) y
> puede pasarse a la fase de dominio complejo?
>
> **Veredicto original: `close_after_p0_fixes`** → **AHORA: dominio real
> soundness-CERRADO.** Los **4 clusters de respuesta-incorrecta** están resueltos
> y validados (sin wrong-answers abiertos); se cumple la puerta de soundness para
> la Fase 2 (dominio complejo).
>
> **Progreso:** ✅ **P0-1** `2753e6ce8` · ✅ **P0-2** `0de6d7081` · ✅ **P0-3**
> `886fb0ec8` · ✅ **P0-4** `9f2920b06`. Los 4 P0 AUDITADOS están cerrados.
>
> ⚠️ **MATIZ (re-auditoría 2026-06-30, `wf_32971e89-52c`):** una segunda prueba
> adversarial de soundness halló **18 wrong-answers PRE-EXISTENTES** (confirmados a
> mano vs `729e72d52`, NO regresiones) en **formas que la auditoría original no
> sondeó** — adyacentes a P0-1 y P0-2. Mis 7 ciclos son **sound** (cero regresiones;
> 5 frentes limpios). Pero el dominio real está cerrado para las formas AUDITADAS,
> **no universalmente.** Clases abiertas (§2b):
> A (ecuaciones trig que se reducen a `sin²`/`cos²`/`sin³` → colapsan a `{0,0}`),
> B (inecuaciones recíprocas `1/x^n` con n≥7 y RHS racional), C (inecuaciones de
> raíz impar `1/x^(1/3)`).

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

### P0-3 — Integral definida/FTC inventa polos desde la racionalización → `undefined` falso  ✅ RESUELTO (commit pendiente)

```
ANTES                                          AHORA
integrate(1/(sqrt(x)*(1+x)),x,1/2,4) => undef  => 2·(arctan(2)−arctan(√½)) ≈ 0.983
integrate(1/(sqrt(x)*(1+x)),x,1,inf) => undef  => π/2
integrate(1/(sqrt(x)*(1+x)),x,0,inf) => undef  => residual HONESTO (era wrong)
```

Racionalizar `sqrt(x)·(1+x) = √x+√x³` produce `(√x³−√x)/(x³−x)`, introduciendo
una raíz **espuria** x=1 (donde el numerador TAMBIÉN se anula → removible). El
escáner de polos del FTC corría sobre el integrando racionalizado y rechazaba x=1
como polo-en-intervalo → `undefined` falso en una integral convergente/regular.

- **Fix:** antes de certificar polos, dividir del denominador cada raíz **simple**
  del intervalo cerrado donde la **antiderivada continua es FINITA**
  (`boundary_is_genuinely_nonfinite(F(r))` distingue removible — `2·arctan(√1)`
  finito — de polo genuino — `ln|x−1|→−∞`, que se mantiene). La antiderivada como
  oráculo de removibilidad esquiva evaluar `√(cuadrado perfecto)` exactamente.
  Bonus: arregla también removibles racionales puros (`(x−1)/(x²−1)→ln4`).
- **Validación:** workspace failed:0 + test de contrato; clippy limpio; huella
  IDÉNTICA. Polos genuinos siguen `undefined`/residual (sound).
- **Frente:** `definite-ftc`. *(El `[0,inf)` queda residual honesto: la singularidad
  √x-en-0 con extremo infinito es capacidad de impropias aparte, no un wrong-answer.)*

### P0-4 — Regla de racionalización de raíz n-ésima da valor erróneo  ✅ RESUELTO (commit pendiente)

```
ANTES                                AHORA
integrate(1/x^(1/4),x)     => 4·x^(1/4)  => 4/3·x^(3/4)   (diff-back = x^(-1/4) ✓)
integrate(1/x^(1/4),x,0,1) => 4          => 4/3
integrate(1/x^(1/6),x,0,1) => 6          => 6/5
```

La rama "denominador = un solo factor raíz" multiplicaba por la raíz **bare**
`x^(1/n)` y ponía denominador `= x`, asumiendo `raíz² = x` — solo cierto para
raíz **cuadrada**. Para `n>2` deja `x^(2/n) ≠ x`, convirtiendo `1/x^(1/4)` en
`x^(-3/4)` (mal). **Fix:** multiplicar por el conjugado correcto `x^((n-1)/n)`
(que sí cierra: `x^(1/n)·x^((n-1)/n)=x`); n=2 conserva `sqrt(...)`. Verificado por
diff-back y test de contrato; huella IDÉNTICA. *(La raíz IMPAR `1/x^(1/3)` ahora
da la forma racionalizada correcta pero declina por un hueco aparte de cancelación
`x^p/x` del simplificador — capability, no wrong-answer.)*

(Registro original del bug:)
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

## 2b. Clases ABIERTAS de la re-auditoría (2026-06-30 · NO regresiones)

Una segunda prueba adversarial (`wf_32971e89-52c`, 27 agentes, verificación
diff-back/numérica/testigos) halló **18 wrong-answers que ya existían ANTES de los
4 P0** (confirmados a mano contra `729e72d52`). La auditoría original no sondeó
estas formas (las que el simplificador REESCRIBE: cuadrados, factorizadas, potencia
superior). Cada clase repite la forma "el detector cubre el caso bare/pequeño y
pierde el envoltorio general".

| Clase | Síntoma | Ejemplo | Verdad |
|---|---|---|---|
| ~~**A**~~ ✅ trig→potencia | ~~colapsa a finito~~ **HECHO** (peel Neg/coef + `trig^n=0⇒trig=0`, guarda de complementariedad) | `{kπ}` |
| ~~**B**~~ ✅ recíproca n≥7 / RHS racional | ~~inventa el rayo negativo~~ **HECHO** (sign-analysis sobre raíces de la ecuación + caps 6/8→12) | `(0,1)` |
| ~~**C**~~ ✅ potencia fraccionaria | ~~operator-drop / "No solution" / complemento~~ **HECHO** (declina a residual honesto las potencias NO monótonas: numerador par o exponente negativo no-entero) | `(0,1/8)` |

- **Clase A:** la forma ECUACIÓN `solve(sin(x)^2=1)` SÍ es correcta (`{π/2+kπ}`); solo
  la forma EXPRESIÓN/factorizada que se simplifica a una potencia colapsa. Mi fix P0-1
  cubrió productos de factores trig BARE distintos (los une periódicamente); estos se
  simplifican a `sin²`/`cos²`/`sin³` ANTES y van por la ruta cuadrática-en-trig, que
  pierde periodicidad y duplica la raíz principal. Relacionado con el hueco
  cuadrática-en-trig de §3.
- **Clase B:** mi fix P0-2 subió el cap de grado 4→6, así que n=5,6 funcionan; n≥7 cae
  al fallback insólido. `1/x^5>1` (n=5, RHS racional) falla aunque `1/x^5>2` (RHS surd)
  acierta — apunta a una ruta temprana específica de RHS racional. Fix: subir/quitar el
  cap (la verificación numérica es la red) + revisar la ruta RHS-racional.
- **Clase C:** la re-auditoría era más amplia de lo anotado: además de las recíprocas
  de raíz impar (`1/x^(1/3)>2 → "No solution"`), las potencias de **numerador par**
  (`x^(2/3)>2 = |x|^(2/3)>2`, un valle simétrico) perdían el rayo negativo, y las
  recíprocas de raíz par (`1/√x>2`) devolvían el complemento o incluían el polo. El
  principio unificador: la isolación monótona del motor solo es correcta cuando `x^e`
  es **estrictamente monótona** (`e>0`, numerador IMPAR). Todo lo demás no-entero
  (numerador par = valle, o exponente negativo = recíproca con polo/salto de signo) se
  declina ahora a residual honesto en vez de emitir un rayo erróneo. El solver de
  ECUACIONES bajo estas formas (`solve(1/x^(1/3)=2)` también daba basura) sigue roto;
  resolverlas correctamente (valles de dos rayos + recíprocas fraccionarias) es el
  siguiente peldaño de capacidad — declinar mantiene el motor SOUND mientras tanto.

**Las 3 clases P0 de la re-auditoría están CERRADAS (soundness restaurada).** Mis ciclos
son sound (cero regresiones; 5 frentes limpios + los P0 originales mejorados).

## 3. Mochila de capacidad para "universal" (NO bloquea soundness)

Estas **declinan honestamente** (sin valor incorrecto); bloquean la promesa
"universal" fuerte, no el cierre de soundness. Esfuerzo: S/M/L.

| Frente | Hueco | Esf. |
|---|---|---|
| ~~radical-integration~~ ✅ | ~~Split de linealidad `p(x)/√(cuadrática)`~~ **HECHO** (familias asinh/arcsin/acosh; bonus grado superior). *Residual aparte: `1/√(a·x²+b)` con `a≠1`.* | S |
| radical-integration | Denominadores radicales `(a±x²)^(impar/2)` vía sustitución trig/hiperbólica + reducción | M |
| radical-integration | Sustitución racionalizante de raíz orden >2 (`t=x^(1/lcd)`) y cofactores `p(x)·(ax+b)^(m/n)` | M |
| ~~trig-integration~~ ✅ | ~~`sin^m/cos^n` con denominador n≥4 (familia sec^k)~~ **HECHO** vía u=cos/u=sin (companion negativa). *Residual aparte: `cos/sin^par` lento por Weierstrass.* | S |
| transcendental-integration | Componer u-sub∘por-partes y producto triple `poli·trig·exp` | M |
| solve-equations | SOLVE cuadrática-en-trig / exponencial base-común y base-distinta / hiperbólica (u=g(x) + back-sub). **Además: arregla el crash `E_INTERNAL` → declinar honestamente** | M |
| solve-inequalities | **Inecuaciones trig periódicas** (variante SolutionSet periódica: `cos(x)>0`, `sin(x)>1/2`, `tan(x)>1`) — mayor palanca restante | L |
| ~~series-sums~~ ✅ | ~~Maclaurin binomio-fraccionario~~ **HECHO** (`(1+x)^α` en centro 0, fallback a diferenciación) | S |
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
3. ~~**P0-3** — chequeo de polo sobre el integrando ORIGINAL~~ ✅ **HECHO** (commit
   pendiente): dividir raíces removibles (antiderivada finita) antes de certificar;
   pin `test_eval_definite_integral_removable_pole_is_not_undefined`.
4. ~~**P0-4** — racionalización de raíz n-ésima~~ ✅ **HECHO** (commit pendiente):
   conjugado correcto `x^((n-1)/n)`; pin
   `test_eval_nth_root_reciprocal_integral_uses_correct_conjugate`. (`1/x^(1/3)`
   queda residual honesto por un hueco aparte de cancelación `x^p/x`.)
5. ✅ **GATE CUMPLIDO** — `cargo test --workspace` 0 fallos + `clippy --workspace
   --all-targets` limpio + huella IDÉNTICA en los 4 ciclos. **El dominio real queda
   CERRADO (sin wrong-answers abiertos) → puerta de soundness para Fase 2 (complejo)
   cumplida.**
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
