# Engine vs. sympy — evaluación comparativa y distancia a "universal en dominio real"

- **Fecha:** 2026-07-14
- **HEAD evaluado:** `901cf595a` (tras cerrar F7/F8/F6/F1 del audit 2026-07-14)
- **Oráculo de contraste:** sympy 1.14.0
- **Método:** probes VERIFICADOS ejecutando ambos sistemas lado a lado (binario `target/release/cas_cli` + sympy en proceso), NO estimación de memoria. Los comandos para reproducir están al final.
- **Documentos relacionados:** `docs/CALCULUS_ENGINE_STRATEGY.md` (north star), `docs/CALCULUS_ENGINE_DEVELOPMENT_PHASES.md` (fases + gatekeepers), `docs/CALCULUS_FRONTIER_AUDIT.md` (cola priorizada + residuales honestos permanentes), `docs/AUDITORIA_FRONTERA_2026-07-14.md` (frontera actual).

---

## TL;DR

El engine **sí** tiene funcionalidad que sympy no hace o hace peor, pero en un **carril estrecho y deliberado**: real, univariable, elemental, *y* autoexplicativo. **No aspira a la amplitud de sympy** — eso está explícitamente fuera de su norte. Dos afirmaciones son ciertas a la vez:

1. En su carril, el engine produce resultados que sympy **no produce o produce menos completos**: familias periódicas completas *incluso en desigualdades* (sympy 1.14 las trunca a un periodo), desigualdades con valor absoluto y **parámetro simbólico** (sympy lanza `NotImplementedError`), condiciones de dominio rastreadas, derivaciones narradas paso a paso, y un contrato de honestidad que declina resultados no-elementales en vez de inventar.
2. Frente a la amplitud total de sympy la distancia es **enorme y por diseño**: sin EDOs, sin funciones especiales como salida, sin álgebra lineal simbólica / Gröbner / teoría de números amplia / combinatoria / suposiciones, sin dominio complejo, y con límites basados en *lista de patrones* en vez de un algoritmo general (Gruntz).

Para "universal en su carril" (real-univariable-elemental) le falta esencialmente **1 pieza L bloqueante (integración racional G1) + 1 pieza más pequeña de límites** (solo el algoritmo de VALOR general tipo Gruntz; la narración didáctica G2 ya está MADURA), más pulido. Para "universal como sympy" (amplitud) — no es la meta y no lo será.

---

## 1. Resultados cabeza a cabeza (verificados)

### 1.1 Cálculo

| Caso | Entrada | Engine | sympy | Veredicto |
|---|---|---|---|---|
| Derivada + cadena | `diff(sin(x^2), x)` | `2·x·cos(x^2)` | `2*x*cos(x**2)` | empate |
| Derivada orden-n | `diff(x^5, x, 3)` | `60·x^2` | `60*x**2` | empate |
| Integral racional fácil | `integrate(1/(x^2-1), x)` | `½·ln\|(x−1)/(x+1)\|` | `log(x−1)/2 − log(x+1)/2` | **engine más real-honesto** (con `\|·\|`, real en todo el dominio; sympy es complejo si x<1) |
| Integral racional dura | `integrate(1/(x^5-1), x)` | **residual** | resuelve (Φ₅, √5) | **sympy gana (G1)** |
| Integral racional dura | `integrate(1/(x^5-1), x)` | **residual** (Φ₅/√5) | resuelve | **sympy gana (G1)** — `1/(x^6+1)`, `1/(x^8-1)`, `1/(x^4-4)` YA EMPATAN tras G1 Cap.A/B |
| Por partes | `integrate(x*e^x, x)` | `(x−1)·e^x` | `(x−1)*exp(x)` | empate |
| No-elemental (Gauss) | `integrate(e^(-x^2), x)` | residual (por diseño) | `√π·erf(x)/2` | filosofía: sympy usa función especial |
| No-elemental (Fresnel) | `integrate(sin(x^2), x)` | residual (por diseño) | `fresnels(...)` | filosofía |
| No-elemental (li) | `integrate(1/ln(x), x)` | residual (por diseño) | `li(x)` | filosofía |
| Límite notable | `limit(sin(x)/x, x, 0)` | `1` | `1` | empate |
| Límite L'Hôpital | `limit((e^x-1-x)/x^2, x, 0)` | `1/2` | `1/2` | empate |
| Límite lateral | `limit(1/x, x, 0)` | `undefined` (bilátero DNE); con dirección `limit(1/x,x,0,+)`→`+∞`, `…,-)`→`−∞` | `oo` (asume lateral derecho) | **engine más honesto**: el bilátero DNE es `undefined` y los laterales ±∞ ya se obtienen dando la dirección |
| Taylor | `taylor(sin(x), x, 0, 6)` | `x − x³/6 + x⁵/120` | `… + O(x**6)` | empate |
| Serie geométrica | `sum(1/2^n, n, 1, infinity)` | `1` | `1` | empate |

### 1.2 Álgebra / solve / desigualdades / discreto

| Caso | Entrada | Engine | sympy | Veredicto |
|---|---|---|---|---|
| Cuadrática (real) | `solve(x^2+1=0, x)` | `No solution` | `EmptySet` | empate (real-honesto) |
| Cuadrática (complejo) | `solve(x^2+1=0, x)` `--value-domain complex` | **`No solution`** | `{−i, i}` | **sympy gana** (F12: flag complejo no honrado; Fase 2 gated) |
| Cúbica | `solve(x^3-6x^2+11x-6=0, x)` | `{1, 2, 3}` | `{1, 2, 3}` | empate |
| Trig ecuación | `solve(sin(x)=1/2, x)` | `{π/6+2kπ, 5π/6+2kπ}` | union equivalente | empate (engine más legible) |
| Trig ecuación | `solve(cos(x)^2=1/2, x)` | `{π/4 + k·π/2}` (1 base, periodo π/2) | `Union(2nπ+π/4, 2nπ+3π/4)` (2 bases) | **engine más compacto** (reconoce el periodo real) |
| Trig ecuación (era hang) | `solve(tan(2x)=tan(3x), x)` | `{kπ}` | `Union(2nπ+π, 2nπ)` | empate (engine más limpio; era hang de 216s, cerrado) |
| **Desigualdad trig** | `solve(cos(x)>1/2, x)` | `{(−π/3+2kπ, π/3+2kπ) : k∈ℤ}` **familia completa** | `Union([0,π/3), (5π/3,2π))` — **solo [0,2π)** | **ENGINE MEJOR: sympy TRUNCA a un periodo** |
| Desigualdad racional | `solve(1/x>2, x)` | `(0, 1/2)` | `Interval.open(0, 1/2)` | empate |
| **Desigualdad abs paramétrica** | `solve(abs(x-a)<=3, x)` | `[a−3, a+3]` | **`NotImplementedError`** | **ENGINE MEJOR: sympy no lo resuelve** |
| Desigualdad polinómica | `solve(x^2-3x+2>0, x)` | `(−∞,1) ∪ (2,∞)` | union equivalente | empate |
| Abs ecuación | `solve(abs(2sin(x)-1)=1, x)` | `{π/2+2kπ, 2kπ, π+2kπ}` | union equivalente | empate (ambos familia completa) |
| Radical paramétrico | `solve(sqrt(a-x)=x, x)` | `{½(√(4a+1)−1), ½(−√(4a+1)−1)}` | equivalente | empate |
| Factorización | `factor(x^4-1)` | `(x−1)(x+1)(x²+1)` | igual | empate |
| gcd racional | `gcd(1/2, 1/3)` | `1/6` | `1/6` | empate |
| Potencia de matriz (era hang) | `[[1,1],[1,0]]^14` | `[[610,377],[377,233]]` | igual | empate (era hang/blowup, cerrado F1) |
| Inversa de matriz | `inverse([[1,2],[3,4]])` | `[[−2,1],[3/2,−1/2]]` | igual | empate |
| Exponenciación modular (era hang) | `mod(123456789^987654321, 1000000007)` | `652541198` | `652541198` | empate (era hang, cerrado F6) |
| Factorización prima | `prime_factors(360)` | `2³·3²·5` | `{2:3, 3:2, 5:1}` | empate |

---

## 2. Dónde el engine iguala o supera a sympy

**Diferenciadores verificados (correctness/completitud real):**

1. **Desigualdades trig como familia periódica COMPLETA.** `cos(x)>1/2` → el engine da `{(−π/3+2kπ, π/3+2kπ) : k∈ℤ}`; sympy `solveset(..., Reals)` devuelve solo `[0,π/3) ∪ (5π/3, 2π)` — **truncado a un periodo**. La respuesta de sympy es *incompleta sobre ℝ*; la del engine es el conjunto solución real correcto. (Es el tema central del ledger: "Periodic SolutionSet es el gap top-leverage", ya cerrado para estas familias.)

2. **Desigualdad con valor absoluto y centro simbólico.** `abs(x-a)<=3` → engine `[a−3, a+3]`; sympy lanza **`NotImplementedError`**. (Es la familia F7 cerrada en este mismo audit, `c5b7bf6f5`.)

3. **Forma integral real-honesta.** `∫1/(x²−1)dx` → engine `½·ln|(x−1)/(x+1)|` (real y definida en todo el dominio); sympy `log(x−1)/2 − log(x+1)/2` (compleja para x<1).

4. **Representación periódica más compacta.** `cos²(x)=1/2` → engine 1 familia con periodo π/2; sympy 2 familias con periodo 2π. Equivalentes, pero el engine reconoce el periodo real menor.

5. **Límite bilátero honesto.** `limit(1/x, x, 0)` → engine `undefined` (el límite bilátero no existe); sympy `oo` (por su convención toma el lateral derecho). *Matiz:* el engine SÍ da el lateral cuando se especifica la dirección (`limit(1/x,x,0,+)`→`+∞`, `…,-)`→`−∞`); reserva `undefined` para el bilátero genuinamente DNE.

**Diferenciadores por diseño (soundness / pedagogía) — sympy no tiene equivalente:**

6. **Trazas didácticas paso a paso** (`--steps on`): el engine narra la derivación con condiciones de dominio explícitas; sympy no produce nada equivalente.

7. **Rastreo de condiciones de dominio en la simplificación**: el engine emite las condiciones que hacen válida una simplificación en un sobre de transparencia estructurado, en vez de descartarlas en silencio como `simplify()`.

8. **Contrato de honestidad / residuales honestos**: el engine declina deliberadamente `∫e^(−x²)`, `∫sin(x²)`, `sin(x)/x`, `1/ln(x)` en vez de inventar. Es diferencia de *filosofía*, no estrictamente "mejor": depende de si el objetivo es real-elemental (declinar es correcto) o máxima cobertura (sympy responde con función especial).

**Empates que cubren el núcleo real-elemental:** derivadas + cadena + orden-n, integrales por partes / u-sub, límites notables y de L'Hôpital, `solve` polinómico/cúbico, ecuaciones trig como familia periódica completa, factor, gcd racional, potencia e inversa de matriz, `mod(a^b,m)`, factorización prima, series geométricas. (Las tres últimas — matriz-power, modpow, y varias trig — eran bugs P0/hangs cerrados en la sesión 2026-07-14 y ahora coinciden con sympy.)

---

## 3. Dónde sympy gana claramente

**Verificado por probe:**

- **Integración racional universal.** `∫1/(x⁵−1)` (Φ₅/√5), `∫1/(x³−2)` (∛2) — sympy las resuelve completas; el engine deja residual (las cuárticas pares `∫1/(x⁶+1)`, `∫1/(x⁸−1)`, `∫1/(x⁴−4)` YA las resuelve tras G1 Cap.A/B). Es el gatekeeper **G1 ABIERTO** del propio engine.
- **Funciones especiales como salida.** `erf`, `fresnels`, `li` — sympy las devuelve; el engine deja residual (por diseño).
- **Dominio complejo.** `x²+1=0` en complejo → sympy `{−i, i}`; el engine sigue diciendo `No solution` incluso con `--value-domain complex` (Fase 2 gated).

**Por conocimiento del alcance (no probado directamente, pero cierto):** EDOs/`dsolve`, PDEs, algoritmo general de límites (Gruntz) y series de orden arbitrario en punto arbitrario, álgebra lineal simbólica / autovalores / Gröbner, teoría de números amplia, combinatoria, sistema de suposiciones. sympy cubre órdenes de magnitud más superficie matemática total.

---

## 4. Distancia a "universal en dominio real"

Hay que separar dos preguntas distintas.

### 4.1 Sobre su propia meta acotada (Fase 1: real-univariable-elemental + educativo)

Está **cerca pero no terminado**. Estimaciones internas de las auditorías del propio proyecto (direccionales, no benchmark externo): derivadas ~80% de un curso universitario, integral indefinida ~65‑70% (tras G1 Cap.A/B), límites ~70‑75% *(re-sondeado 2026-07-15; la cifra previa ~45‑50% era stale)*. Faltan **dos piezas grandes**:

- **G1 — integración racional universal** (factor-sobre-ℝ / Lazard‑Rioboo‑Trager). Tamaño **L**. Es *el* bloqueador: hoy resuelve los casos que factorizan sobre ℚ (`1/(x⁶−1)`) y — tras G1 Cap.A/B (`d557556ea`, `6c4d59afc`) — también las cuárticas pares con un solo surd cuadrático incluidos numeradores no-constantes (`1/(x⁴−4)` con √2, `1/(x⁶+1)`/`1/(x⁸−1)` con √3/√2); falta lo que exige una extensión algebraica ACOPLADA: `1/(x⁵−1)` (Φ₅/√5, par cuadrático asimétrico) y `1/(x³−2)` (∛2). Siguen siendo probes del criterio de salida #1 (3 de 5 ya verdes), así que **G1 es lo que bloquea abrir la Fase 2**. Es también la única capacidad donde sympy gana *dentro del propio carril* del engine.
- **Límites como algoritmo** (no lista de patrones). Tamaño **L (más pequeño de lo que parecía)**. La narración didáctica (gatekeeper G2) está **MADURA** — factor-cancela, notables, L'Hôpital ITERADO, sándwich, jerarquía ∞/∞, `e` vía (1+1/x)^x, y ∞−∞ en sus dos formas. Lo que queda es de **VALOR**, no de narración: el procedimiento de decisión sigue siendo un allowlist de patrones (no un Gruntz general) y declina el bilátero de 0·∞/0^0 por dominio (las unilaterales resuelven); los laterales finitos ±∞ YA funcionan.

Piezas menores ya casi hechas (criterio #3, largamente aterrizado): `diff(x,n)`/`diff(x,y)`, u-sustitución transcendente general, `taylor()`/`series()` + linealidad de sumatorios.

### 4.2 Sobre la amplitud total de sympy

Distancia **enorme y deliberada**. Fase 2 (complejo principal-branch + vectorial multivariable) y Fase 3 (capas analíticas) están **gated, sin empezar por política**. Y fuera del norte **para siempre** (es *bug de soundness* "resolverlos"): análisis complejo completo/multivaluado/Riemann, EDOs, funciones especiales como valor de salida, y un conjunto fijo de residuales no-elementales correctos (`∫e^(−x²)`, `sin(x)/x`, `1/ln(x)`, `x^x`, `sin(1/x)` en 0).

---

## 5. Veredicto honesto

No es "un sympy más pequeño" — es **un artefacto distinto y más estrecho**: un motor de cálculo *soundness-first, real-domain-first y autoexplicativo* que en su carril produce cosas que sympy no produce o produce peor (familias periódicas completas incluso en desigualdades, condiciones de dominio, derivaciones narradas, residuales honestos), y que a la vez está órdenes de magnitud por detrás de sympy en cobertura matemática total. Ambas cosas son ciertas simultáneamente.

Medir el engine contra la superficie total de sympy lo juzgaría mal, porque esa amplitud no es su meta. Su norte es más estrecho y, en un aspecto, más exigente: universal **Y** autoexplicativo dentro de un dominio acotado. En ese objetivo está "cerca pero no terminado", con **un bloqueador grande (G1) y un residuo estrecho de límites (algoritmo de valor tipo Gruntz; la narración didáctica ya está madura)** entre él y poder declarar completo el dominio real-univariable-elemental.

**Siguiente paso de mayor palanca:** atacar **G1** (integración racional universal / factor-sobre-ℝ / LRT). Es clase L → se entra como *scoping workflow que produce una secuencia de sub-ciclos acotados*, nunca como un solo ciclo. Desbloquea formalmente la Fase 2.

---

## 6. Reproducir

Binario ya construido en HEAD: `target/release/cas_cli`. Patrón de un probe:

```bash
timeout 20 ./target/release/cas_cli eval "solve(cos(x)>1/2, x)" --format json \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('result') or d.get('error'))"
```

Contraste sympy (dominio real donde aplica):

```python
from sympy import symbols, cos, Rational, solveset, S
x = symbols('x')
print(solveset(cos(x) > Rational(1,2), x, S.Reals))   # -> Union(Interval.Ropen(0, pi/3), Interval.open(5*pi/3, 2*pi))
```

Notas de sintaxis del engine: `*` para producto (`2*x`, no `2x`), `^` para potencia; ecuaciones/desigualdades vía `solve(expr, x)`; `--steps on` para trazas; `--value-domain complex` para complejo. Si un comando cuelga, es una regresión de perf (banda de blowup eager) — reportar.

> **Nota de método:** las filas de §1 se obtuvieron ejecutando ambos sistemas; no son estimaciones. Los porcentajes de §4.1 son estimaciones internas del proyecto (auditorías), direccionales, no un benchmark externo.
