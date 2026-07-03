# Informe de síntesis — Re‑scout de frontera CAS (universalidad/completitud, dominio real)
**Fecha:** 2026-07-02 · **Binario:** release (ab5587aa1+) · **Fronteras sondeadas:** 12 · **Inputs totales:** 575 · **Verificación:** adversarial 2-lentes + confirmación a mano/numérica con el propio binario

---

## 1. Resumen ejecutivo

| Clase | Total | % |
|---|---:|---:|
| **ok** (verificado correcto) | 448 | 77.9% |
| **wrong** (respuesta incorrecta afirmada) | 20 | 3.5% |
| **decline_honest** (residual honesto) | 98 | 17.0% |
| **decline_garbage** (residual engañoso/malformado) | 9 | 1.6% |
| **hang** | 0 | 0% |
| **Total sondado** | 575 | |

**Veredicto de soundness: NO limpio — 15 wrong-answers P0 confirmados, concentrados en exactamente 3 familias de causa raíz** (múltiplo-ángulo trig puro, desigualdad exponencial de bases mixtas, polo no perforado en `1/abs(u)`). Fuera de esas 3 familias, el panorama es notablemente sólido:

- **7 de 12 frentes completamente limpios** (0 wrong, 0 garbage): desigualdades trig/transcendentes, ecuaciones polinómicas/racionales, límites, integración, sistemas/matrices, sumas/productos, simplify.
- **0 hangs en 575 inputs** — C5 no se manifestó (ver §5).
- Los 3 clusters P0 reproducen el **bug-shape recurrente del repo** en sus dos variantes conocidas: (i) *el detector BARE/afín pierde una forma hermana* (aquí invertido: el handler afín captura `3x+b` pero el `3x` puro es expandido antes por el rewriter), y (ii) *desigualdad colapsa a ecuación* (bases mixtas `a^u ⋚ b^v` → raíces de la ecuación frontera afirmadas como conjunto solución).
- Bugs históricos **re-confirmados CERRADOS** en este scout: Taylor centro≠0, abs-trig principal-value (`|2sin(x)-1|=1`), sistemas inconsistentes, `e^(2x)>0`, back-sub de `e^(2x)-3e^x+2<0`, polos removibles en solve racional, raíces extrañas de radicales, laterales de límites.

Nota de deduplicación: `solve(1/abs(x)>2)` aparece confirmado en dos frentes (desigualdades racionales lo clasificó P2 por el mitigante `Requires: x ≠ 0`; el frente abs lo clasificó P0). Se cuenta **una vez** en la tabla, con severidad P0 (el set afirmado incluye el polo).

---

## 2. P0 soundness confirmados (máxima prioridad)

### Familia A — Múltiplo-ángulo puro `trig(n·x) = c` cae a expansión polinomial y descarta la periodicidad (8 inputs)

| Input | Esperado (verificado) | Got | |
|---|---|---|---|
| `solve(sin(3x)=1/2)` | `{π/18+2kπ/3, 5π/18+2kπ/3}` | 3 puntos finitos `arcsin(ratio-trig)` (= {π/18, 5π/18, −7π/18}) | falta p.ej. 13π/18, verificado con el engine |
| `solve(2sin(3x)-1=0)` | ídem | mismo conjunto finito, `ok:true` sin residual | |
| `solve(2sin(3x)=1)` | ídem | ídem | |
| `solve(sin(3x)-1/2=0)` | ídem | ídem | |
| `solve(cos(3x)=1/2)` | `{±π/9+2kπ/3}` | 3 arccos finitos, sin familia periódica | falta p.ej. 7π/9 |
| `solve(2cos(3x)-1=0)` | ídem | ídem | |
| `solve(sin(5x)=1/2)` | `{π/30+2kπ/5, π/6+2kπ/5}` (10 clases mod 2π) | **solo la familia de `sin(x)=1/2`** (2 de 10 clases) — variante: factoriza la raíz racional s=1/2 del quíntico y descarta el cuártico en silencio | falta π/30 |
| `solve(sin(6x)=1/2)` | `{π/36+kπ/3, 5π/36+kπ/3}` | hereda el bug de sin(3x) vía 6x→3·(2x), 3 puntos /2 | |

**Hipótesis de fix (única para las 8):** el rewriter expande `sin(3x)→3s−4s³` **antes** de que el handler afín periódico vea el argumento; cada raíz sᵢ del polinomio se mapea solo a `arcsin(sᵢ)` principal, sin familia `2kπ` ni compañero `π−arcsin`. Prueba diferencial: `sin(3x+1)=1/2` ✓, `sin(4x)=1/2` ✓, `sin(3x)=1` ✓ (raíz única). **Fix: interceptar `trig(n·x)=c` en el árbol RAW y enrutar por el handler afín u=n·x antes de la expansión** (misma lección RAW-tree del repo); como mínimo, si queda factor polinomial sin resolver (caso 5x), declinar con residual en vez de afirmar el set parcial, y emitir la familia completa `{arcsin(sᵢ)+2kπ, π−arcsin(sᵢ)+2kπ}` por cada raíz. Nota: todos los puntos devueltos son soluciones genuinas — el defecto es **incompletitud pura afirmada como completa** (sin warning, sin residual, `result_truncated:false`).

Bonus del mismo fix: cierra o mejora los 3 `decline_garbage` del frente (`tan(3x)=1` → `{π/12+kπ/3}`, `sin(3x)=√3/2` → `{π/9+2kπ/3, 2π/9+2kπ/3}`, y con la rewrite producto→ángulo doble, `sin(x)cos(x)=1/4` → `{π/12+kπ, 5π/12+kπ}`).

### Familia B — Desigualdad exponencial de bases mixtas colapsa a ecuación (4 inputs)

| Input | Esperado | Got |
|---|---|---|
| `solve(2^x>3^x)` | `(−∞, 0)` | `{0}` |
| `solve(2^x<3^x)` | `(0, ∞)` | `{0}` |
| `solve(5^x>2^x)` | `(0, ∞)` | `{0}` |
| `solve(2^(x+1)>3^x)` | `(−∞, ln2/ln(3/2))` ≈ (−∞, 1.7095), abierta | `{−ln2/ln(2/3)}` (solo el punto frontera, donde `>` es **falso**) |

**Hipótesis:** el par a^u vs b^v cae al handler take-logs solo-ecuación y el operador relacional se descarta (misma forma que el cluster abs/log del audit #4). Mismo base (`e^x>e^(2x)` ✓) y exponencial única (`2^x>8` ✓) funcionan. **Fix: reescribir `a^u ? b^v` como `u·ln(a) − v·ln(b) ? 0`** (exp estrictamente creciente, operador preservado), en el path de aislamiento exponencial de `solve_backend_local`. El wrapper afín (4º caso) confirma que hay que barrer toda la familia, no la forma BARE.

### Familia C — `1/abs(u) ⋚ c` no perfora el polo; rama c=0 emite basura degenerada (3 inputs)

| Input | Esperado | Got |
|---|---|---|
| `solve(1/abs(x)>2)` | `(−1/2, 0) ∪ (0, 1/2)` | `(−1/2, 1/2)` — incluye el polo (mitigante: adjunta `x ≠ 0` en required_display) |
| `solve(1/abs(x)>=2)` | `[−1/2, 0) ∪ (0, 1/2]` | `[−1/2, 1/2]` |
| `solve(1/abs(x)>0)` | `(−∞,0) ∪ (0,∞)` | `(−∞,−∞) ∪ (∞,∞)` — intervalos degenerados = conjunto vacío + representación basura |
|

**Hipótesis:** la reducción `1/|u|>c → |u|<1/c` no intersecta con `{u≠0}`; los paths gemelos `1/x^2>4` y `abs(1/x)>2` **sí** perforan (verificado). La rama c=0 divide por cero literal (`1/0` como endpoint). **Fix: sustracción del conjunto de polos en el ensamblado** (como el path racional) + cortocircuito "LHS probadamente positivo > 0 → dominio del LHS" + invariante *nunca emitir intervalos (a,a)*. Lección re-confirmada del repo: **los sweeps que ramifican sobre una constante DEBEN incluir c=0**.

---

## 3. Backlog de capacidad re-priorizado por ROI

Agrupado por causa raíz. Localidad: S = fix localizado en un handler, M = varios puntos/nueva rewrite, L = cambio estructural.

| # | Familia (causa raíz) | Evidencia | Amplitud | Impacto usuario | Localidad | Prioridad |
|---|---|---|---|---|---|---|
| 1 | **Honestidad del residual (transversal):** el printer muta el operador (`sin(x)>1/2` se ecoa como `solve(sin(x) = 1/2)`), añade sufijo `= 0` colgante, y el aislamiento en punto fijo emite `solve()` anidado como resultado `ok:true` | 9 decline_garbage + display de los 98 honestos; observado en 4 frentes distintos | Convierte los 9 garbage en honestos y blinda todo decline futuro | Alto — un residual que desdescribe el problema es casi-unsound | S/M | **P1 alta — barato y transversal** |
| 2 | **Selector de intervalos periódicos + variante `PeriodicIntervalUnion` en SolutionSet** | 21 declines de desigualdades trig; la ECUACIÓN frontera ya se resuelve con sets periódicos | La mayor de todo el backlog (toda desigualdad trig entre raíces) | Alto | **L** | **P1 — confirma el item (a) del backlog previo como top gap estructural. No apresurar; diseñar primero** |
| 3 | **Boundary débil trig con wrapper** (`2sin(x)>=2`, `sin(3x)>=1`, `sin(x+π/3)>=1`, `cos(2x)<=-1`): detector BARE-only, las ecuaciones frontera ya se resuelven | 4+ declines | Media (sub-familia de #2 que NO necesita el selector: la solución es el set de la igualdad) | Medio | **S** | **P1 — ROI máximo dentro de #2, adelantable** |
| 4 | **Denominador radical/exponente fraccionario en desigualdades** (`1/sqrt(x)>2`, `1/sqrt(x-1)>2`, `1/x^(1/3)>2`): rutear por u-space monótono | 3 garbage; la maquinaria surd-affine u-space (762d973bd) está cerca | Media | Medio | S/M | P1 |
| 5 | **Cuadrática exponencial con raíces u irracionales** (`e^(2x)−e^x−1⋚0`): el path existe (raíces racionales ✓), falta back-sub de surds | 2 garbage | Baja-media | Medio | **S** | P1 |
| 6 | **Integrales `poly × b^(ax+c)`, b≠e:** normalizar `b^(ax+c) → b^c·(b^a)^x` antes del detector por-partes | 4 declines; frontera empírica exacta mapeada | Media | Medio | **S** | P1 — **actualiza el item (c): los predicados csc³/sec³ afines SÍ rutean (refutado); lo que falta es solo este wrapper** |
| 7 | **Límites, 3 sub-fixes:** (F1) combinador bilateral→DNE/±∞ cuando los laterales discrepan (¡los laterales ya se calculan!); (F2) rama espejo `|x|=−x` en asintótica sqrt@−∞; (F3) ∞−∞ vía Add→N/D | 16 declines, ~11 cubiertos por F1–F3 | Media-alta | Alto (1/x@0, |x|/x@0 son ubicuos) | S/M | **P1 — actualiza el item (b): "laterales finitos declinan" está OBSOLETO; el residual real se estrechó a estas familias** |
| 8 | **Sumas: razón simbólica/irracional + tabla de series clásicas + wrapper afín armónico** (`r^k` con \|r\|<1 vía const_value_bounds; Mercator/Leibniz/exp; `1/(2k−1)`→∞) | 16 declines | Media | Medio | M | P2 |
| 9 | **Simplify: racionalización cbrt + cancelación N/D en base transcendental** (`1/(cbrt(2)−1)`, `(e^(2x)−1)/(e^x−1)`); + siblings de plegado (`cos²−sin²→cos(2x)`, `\|√3−2\|`) | 8 declines | Media | Medio | S/M | P2 |
| 10 | **abs/sign/piecewise:** registrar `sign()` en el solver (diff lo produce), sustitución u=\|x\|, parser que acepte relacionales como argumento (hoy `piecewise(x>0,1,-1)` es parse error opaco — el fix mínimo es el decline honesto alcanzable) | 9 declines + 1 garbage | Media | Medio | M | P2 |
| 11 | **`taylor(sin(x)/x)@0`:** extender por la singularidad evitable | 1 decline | Baja | Bajo | S | P2 |
| 12 | **Calidad C5-adyacente:** const-fold de Add numérico en PostCleanup (`−1+3` sin plegar), dedup/rate-limit del WARN depth_overflow (~130KB stderr por invocación en `(x+tan x)^3`, `x²·2^x`) | 1 "wrong" P2 (valor correcto) + asperezas en 3 frentes | — | Bajo (cosmético) pero ensucia todo diagnóstico | S | P2 — **mitigación barata sin tocar la orquestación C5 (item (d): respetar el do-not-rush)** |
| 13 | Asperezas de plegado de endpoints (`ln(4)/ln(1/2)` ↛ −2; `arctan(−3·3^(−1/2))` ↛ −π/3; surds `−2·2^(−1/2)` ↛ −√2) y `linsolve` singular vs `solve_system` | varios P2 | Baja | Bajo | S | P3 |

También en el radar P1: el **budget roto** en `tan(3x)=1` (16.1s con `--time-budget-ms 10000`) — no es hang pero el time-budget no acota ese camino; el fix de la Familia A probablemente lo elimina, pero conviene un test de que el budget corta.

---

## 4. Declines honestos correctos (completitud-como-contrato funcionando)

98 declines honestos, 0 valores falsos entre ellos. Destacables como comportamiento **correcto y deseado**:

- **No-elementales:** `∫e^(x²)`, `∫sin(x)/x`, `∫1/ln(x)` (Liouville) — residuales deliberados, no tocar.
- **Trascendentes sin forma cerrada:** `e^x=x`, `x^x=2`, `x·e^x=1` (Lambert W fuera de alcance declarado).
- **Cuártica/cúbica simbólica:** `x^4+x−1=0`, `x^3+px+q=0`.
- **Oscilación acotada:** `sin(1/x)@0`, `sin(x)@∞` — declinar es sound; afirmar DNE es P2 opcional.
- **Divergentes sin trampa:** no aplica `1/(1−r)` a geométricas divergentes ni suma Abel a Grandi; suma vacía→0; polo en el rango→undefined.
- **Sistemas:** no-lineal, paramétrico, indeterminado — residual limpio.

---

## 5. Hangs

**Cero hangs en 575 inputs.** Cruce con C5 (item (d)):

- Las formas C5-adyacentes sondeadas en simplify (`(sin+cos)²` vs `1+sin(2x)`, ambas orientaciones) **declinan rápido** (66ms, depth_overflow controlado) — el residual documentado se comporta.
- Lo más cercano a la patología C5 observado: `tan(3x)=1` (16s reventando el budget — se cierra con Familia A) y el spam depth_overflow de `(x+tan x)^3` / `x²·2^x` (valor correcto, misma zona expand↔factor; mitigación barata en backlog #12). **Ninguno justifica tocar la orquestación C5 ahora.**

---

## 6. Falsos positivos refutados (transparencia del método)

3 hallazgos cayeron en verificación adversarial — todos **misreads de convención/output del engine, no errores matemáticos del scout**:

1. `solve(1/abs(x-1)>2)` (frente desigualdades racionales) — la matemática del scout es correcta pero la clasificación no sobrevivió la reproducción.
2. `solve(1/abs(x-1)>2)` (frente abs) — mismo input, misma refutación: misread del output real del engine.
3. `solve(2/abs(x)>4)` — convención misread; reproducido y refutado.

Consistente con la lección de memoria (audit #5): *adversarial verify no es infalible; hand-verify los P0*. Notar que la familia `1/abs` **sí** tiene 3 P0 reales confirmados (Familia C) — los refutados eran las variantes con wrapper donde el engine aparentemente sí perfora; **el sweep del fix de Familia C debe cubrir BARE y wrapper para fijar la frontera exacta.**

---

## 7. Recomendación — primeros 3 ciclos

**Ciclo 1 — Familia A (múltiplo-ángulo trig): 8 P0 + 3 garbage con un solo fix.**
Interceptar `trig(n·x)=c` en el árbol RAW y enrutar por el handler afín periódico u=n·x antes de la expansión múltiple-ángulo (patrón ya probado en el fix surd-affine RAW). Red de seguridad: si un camino polinomial deja factor sin resolver, declinar con residual (cubre sin(5x)). Sweep obligado: n∈{2,3,4,5,6}, sin/cos/tan/cot/sec/csc, c∈{0, ±1, 1/2, √3/2, 2}, formas `k·trig(nx)±m=0`, y añadir la rewrite `sin·cos→sin(2x)/2`. Es el ciclo de mayor densidad P0-por-línea del informe.

**Ciclo 2 — Familias B + C (dos fixes S): 7 P0 restantes.**
(B) `a^u ? b^v → u·ln(a) − v·ln(b) ? 0` preservando el operador, con sweep de wrappers afines y ambas direcciones; (C) intersección con `{denominador≠0}` en el path `1/abs(u)` + cortocircuito c=0 + invariante anti-`(a,a)`. Sweep con c∈{0, negativo, racional}, estricto/no-estricto, BARE y `abs(x−1)`/`2/abs(x)` (fijando la frontera que los refutados dejaron borrosa). Al cerrar este ciclo: **0 wrong-answers conocidos otra vez**.

**Ciclo 3 — Contrato de honestidad del residual (backlog #1) + boundary débil con wrapper (#3).**
(i) El printer de residuales conserva el operador relacional original y elimina el sufijo `= 0`; (ii) guard: un resultado que contiene `solve()` anidado sin evaluar jamás sale como `ok:true` sin marca de residual; (iii) dedup del WARN depth_overflow; (iv) extender el detector de boundary débil trig a wrappers afines/coeficiente reutilizando el pipeline de ecuaciones. Elimina los 9 garbage restantes como clase y deja el terreno limpio para atacar `PeriodicIntervalUnion` (#2) como proyecto de diseño en ciclos posteriores, no como parche.

*Justificación del orden:* soundness antes que capacidad (ciclos 1–2 cierran los 15 P0), y el ciclo 3 es transversal-barato: convierte cada gap futuro en un decline honesto, que es exactamente el contrato que los 7 frentes limpios ya demuestran saber cumplir.