# Auditoría P0 de soundness — 2026-06-30

**Método:** dos ciclos multiagente de caza de wrong-answers (probe multi-frente →
verificación adversarial de 2 lentes independientes: testigo numérico + oráculo
simbólico con reconciliación de convenciones), seguidos de **hand-verify** de cada
familia contra el CLI real y **bisección** contra el commit pre-sesión
(`f0dd54619`) para atribuir pre-existente vs introducido.

**Cobertura:** 26 frentes/agentes-frente, **~2.250 inputs sondeados**, 290 agentes.
**Resultado:** **128 instancias confirmadas** de respuesta incorrecta (112 inputs
únicos), **0 en disputa**, **0 falsos positivos** tras hand-verify, agrupadas en
**8 familias**. Las dos lentes coincidieron en el 100% de los confirmados.

**Atribución:** la bisección confirma que **NINGUNA** de las 8 familias procede de
los ciclos P0/P1 recientes (Clase A `c6bf3ef45`, B `b686b2263`, C `ae1efa32d`,
Ciclos 1-3, radical/polo-removible). Todas son **pre-sesión** o de **features
añadidas antes en la sesión** (el comando `taylor()`). Los únicos deltas que la
bisección mostró en estos inputs fueron **cosméticos** (corchete cerrado→abierto en
límites `±∞`), no regresiones.

**Frentes LIMPIOS** (reconfirmados en ambos ciclos, incluso con inputs envueltos
difíciles): integración indefinida (diff-back), integración definida/impropia
(cuadratura), diferenciación, límites, ecuaciones exp/log, ecuaciones radicales,
identidades de álgebra/simplify (factor/gcd/apart/rationalize). Esto concuerda con
auditorías previas: el núcleo de cálculo está sólido; los huecos están en **solve,
inecuaciones, series y sistemas**.

---

## Resumen ejecutivo (8 familias, por severidad)

| # | Familia | Instancias | Severidad | Síntoma de una línea |
|---|---|---|---|---|
| 1 | Solve poli: factor irracional al cuadrado | 14 | **ALTA** | `solve((x²-3)²(x-1)=0)` → `{1}` (suelta `±√3`), `ok:true` sin warning |
| 2 | Solve trig: colapso de familia periódica | 20 | **ALTA** | `solve(sin(x)=√2/2)` → `{π/4}` (suelta periodicidad + 2ª rama) |
| 3 | Inecuación racional de 2 polos | 13 | **ALTA** | `solve(1/(x-1)>1/(x+1))` → basura `inf^(1/2)` / `{punto}` / "No solution" |
| 4 | Inecuación trig de frontera/rango | 12 | **ALTA** | `solve(sin(x)≥1)` → `[π/2,∞)`; `solve(cos(x)<1)` → `(-∞,0)` |
| 5 | Inecuación de potencia no-monótona envuelta | 26 | MEDIA-ALTA | `solve((x-1)^(2/3)>4)` → `(9,∞)` (suelta rama); `1/sqrt(x)>2` → complemento |
| 6 | Taylor/series en centro ≠ 0 (signo) | 12 | MEDIA | `taylor(ln(x),x,1,5)` tiene `+u⁴/4` (debe ser `−u⁴/4`) |
| 7 | `sign` vía `x/abs(x)` (ec. + inec.) | 8 | MEDIA-ALTA | `x/abs(x)=1` → `[0,∞)` (incluye polo 0); `(x-2)/abs(x-2)≥1` → "No solution" |
| 8 | Sistema lineal inconsistente | 7 | **ALTA** | `solve_system(0=1; 0=0)` → "infinitas soluciones" (verdad: sin solución) |

Total: **112 wrong-answers únicos**. Todos son P0 de soundness (resultado definido
y FALSO en la convención real del motor; no residuales honestos).

---

## Detalle por familia

### Familia 1 — Solve polinómico: factor irracional al cuadrado suelta sus raíces (14)

**Disparador (mapeado):** cualquier factor de **multiplicidad ≥ 2** cuyas raíces
reales son **irracionales**, multiplicado por cualquier otro factor → el motor
suelta las raíces irracionales repetidas y conserva solo las racionales. Se cumple
para cuadráticas `(x²-a)²`, cúbicas `(x³-2)²` (raíz `∛2`), biquadrática
`(x⁴-6x²+9)`, multiplicidades 2/3/4, raíces semi-enteras `(x²-½)²`, coeficiente
líder `(2x²-6)²`, dos cofactores, y la **forma EXPANDIDA** (`x⁵-x⁴-6x³+6x²+9x-9`).
Control sano: con multiplicidad **1** (`(x²-3)(x-1)`) sí devuelve las 3 raíces.

| Input | Engine | Verdad |
|---|---|---|
| `solve((x^2-3)^2*(x-1)=0)` | `{1}` | `{1, √3, -√3}` |
| `solve((x^2-7)^2*(x-3)=0)` | `{3}` | `{3, √7, -√7}` |
| `solve((x^3-2)^2*(x-1)=0)` | `{1}` | `{1, ∛2}` |
| `solve((x^2-3)^2*(x^2-4)=0)` | `{-2,2}` | `{-2, 2, √3, -√3}` |
| `solve(x^5-x^4-6x^3+6x^2+9x-9=0)` | `{1}` | `{1, √3, -√3}` |

**Hipótesis de causa:** tras factorizar sobre ℚ, las raíces de un factor irreducible
**repetido** no se extraen (la fórmula radical solo se aplica a irreducibles de
multiplicidad 1); las raíces racionales (factores lineales) sobreviven. Severidad
alta: silencioso, `ok:true`, sin warning, en una ecuación de libro de texto.

### Familia 2 — Solve trig: colapso de la familia periódica (20)

Tres sub-formas, todas convierten el conjunto infinito periódico en un conjunto
finito (a menudo una sola rama):
- **RHS radical / ángulo notable:** `sin(x)=√2/2`→`{π/4}`, `cos(x)=√3/2`→`{π/6}`,
  `cos(x)=-√2/2`→`{3π/4}`, `2cos(x)+√2=0`, `2sin(x)=√3`, `sin(x)=1/√2`. (Control:
  `sin(x)=1/2` SÍ da la familia completa; `sin(x)=0.7071…` decimal también. Solo el
  RHS radical/notable cae.)
- **Argumento afín:** `sin(x-1)=0`→`{1}`, `cos(x+1)=0`, `sin(3x-1)=0`, `sin(2x)=√2/2`,
  `sin(2x+1)=1/2`, `tan(x+1)=1`, `tan(2x-1)=1`. (Suelta la periodicidad del argumento.)
- **Potencia impar = constante:** `cos(x)^3=1`→`{0}`, `sin(x)^3=1`→`{π/2}`.

| Input | Engine | Verdad |
|---|---|---|
| `solve(sin(x)=sqrt(2)/2)` | `{π/4}` | `{π/4+2kπ, 3π/4+2kπ}` |
| `solve(sin(x-1)=0)` | `{1}` | `{1+kπ}` |
| `solve(cos(x)^3=1)` | `{0}` | `{2kπ}` |

**Hipótesis:** estas formas envueltas alcanzan la inversa de valor-principal en vez
del solver periódico. Las correcciones previas (P0-1 productos de trig distintos;
Clase A `trig^n=0`) cubrieron otras sub-formas; el RHS radical, el argumento afín y
`trig^n=c` (c≠0) siguen yendo por la rama principal. Severidad alta: `sin(x)=√2/2`
es la ecuación trig más básica.

### Familia 3 — Inecuación racional de dos polos (13)

`a/(x-p) {op} b/(x-q)` y `1/(x-p) {op} polinomio` → **límite basura `inf^(1/2)`**,
o un **punto discreto** `{r}`, o **"No solution"**, en vez de la unión de intervalos.

| Input | Engine | Verdad |
|---|---|---|
| `solve(1/(x-1) > 1/(x+1))` | `(-∞,-inf^(1/2)) U (inf^(1/2),∞)` | `(-∞,-1) U (1,∞)` |
| `solve(1/(x-1) > 3/(x+1))` | `{2}` | `(-∞,-1) U (1,2)` |
| `solve(3/(x-1) > 3/(x+1))` | `No solution` | `(-∞,-1) U (1,∞)` |
| `solve(1/(x-1) > x)` | `No solution` | `(-∞,(1-√5)/2) U (1,(1+√5)/2)` |

**Hipótesis:** la LHS de dos fracciones no se normaliza a `N/D` antes del análisis
de signo (eco del bug `Add`-racional ya corregido para una sola fracción); el
`inf^(1/2)` sugiere que un cálculo de grado/límite produce `∞` como cota de raíz.

### Familia 4 — Inecuación trig de frontera/rango (12)

Umbral en el borde del rango (`±1`) o estricto cercano → **rayo erróneo** o
**"No solution"**, en vez del conjunto-punto periódico, o "todos los reales salvo
puntos", o ∅.

| Input | Engine | Verdad |
|---|---|---|
| `solve(sin(x) >= 1)` | `[π/2,∞)` | `{π/2+2kπ}` |
| `solve(cos(x) <= -1)` | `(-∞,π]` | `{π+2kπ}` |
| `solve(cos(x) < 1)` | `(-∞,0)` | ℝ ∖ `{2kπ}` |
| `solve(abs(cos(x)) < 1)` | `No solution` | ℝ ∖ `{kπ}` |
| `solve(abs(sin(x)) >= 1)` | `(-∞,-π/2] U [π/2,∞)` | `{π/2+kπ}` |

**Hipótesis:** la guarda de rango trig / la isolación monótona emite un rayo donde
debería dar el conjunto periódico exacto (cuando `sin=1`) o declinar. `try_decline_
periodic_trig_inequality` excluye los umbrales clasificables, pero el caso frontera
`=±1` y los `<1`/`>-1` se mal-resuelven.

### Familia 5 — Inecuación de potencia no-monótona envuelta (26)

Extiende la clase recién corregida (Clase C `ae1efa32d`, que declina el monomio
`c·x^e` **bare**). Estos **envoltorios escapan** al detector y siguen dando rayo
erróneo / complemento / inclusión de polo:
- **Base desplazada:** `(x-1)^(2/3)>4`→`(9,∞)` (verdad `(-∞,-7)∪(9,∞)`), `(x+2)^(2/3)>9`, `(2x-3)^(2/3)>4`.
- **Constante aditiva:** `x^(2/3)+1>5`→`(8,∞)` (verdad `(-∞,-8)∪(8,∞)`), `x^(2/3)+2>6`.
- **Coeficiente / forma invertida:** `7-2x^(2/3)>1`, `5-x^(2/3)>1`.
- **Función `sqrt` (no `Pow`):** `1/sqrt(x)>2`→`(1/4,∞)` (verdad `(0,1/4)`), `1/sqrt(x-1)>2`, `1/sqrt(2x)>2`.
- **Nota:** la forma `≥` produce un punto-aislado espurio: `(x-1)^(2/3)>=4`→`[-7,-7]∪[9,∞)`.

**Hipótesis:** el detector de Clase C exige monomio puro en `x` bare; falta (a) mover
la constante aditiva, (b) la base afín `(x-a)^e`, (c) la forma función `sqrt()`. Es
el siguiente peldaño directo de Clase C. (Causa de fondo: el solver de ECUACIONES
de estas formas también está roto — ver discusión en `CERRANDO_DOMINIO_REAL.md` §2b.)

### Familia 6 — Taylor/series en centro ≠ 0: signo de un coeficiente (12)

`taylor(f, x, c, n)` con `c ≠ 0` (y su alias `series()`) produce **exactamente un
coeficiente con el signo invertido**. Verificado expandiendo el polinomio del motor:

| Input | Coef. erróneo | Engine | Verdad |
|---|---|---|---|
| `taylor(ln(x),x,1,5)` | `u⁴` | `+1/4` | `−1/4` |
| `taylor(1/x,x,1,6)` | `u⁵` | `+1` | `−1` |
| `taylor(ln(x),x,2,4)` | `u⁴` | `+1/64` | `−1/64` |

Afecta `ln`, `1/x`, `1/x²`, `1/x³`, `sqrt` en centros 1,2,4. **Hipótesis:** error de
signo/factorial en el constructor por diferenciación para centro no nulo (el comando
`taylor()` se añadió en esta sesión; la rama Maclaurin centro-0 parece correcta).

### Familia 7 — `sign` vía `x/abs(x)` (ecuaciones + inecuaciones) (8)

`(x-a)/|x-a|` es `sign(x-a)`. El motor **incluye el polo 0/0** como borde cerrado, o
devuelve **"No solution"/"All reals if…"** en las formas de inecuación.

| Input | Engine | Verdad |
|---|---|---|
| `solve(x/abs(x)=1)` | `[0,∞)` | `(0,∞)` |
| `solve((x-2)/abs(x-2)>=1)` | `No solution` | `(2,∞)` |
| `solve(abs(x)/x=1)` | `All real numbers if x >= 0` | `(0,∞)` |

### Familia 8 — Sistema lineal inconsistente clasificado como "infinitas soluciones" (7)

Toda fila `0 = c` con `c ≠ 0` es una **contradicción** → el sistema **no tiene
solución**. El motor responde "infinitas soluciones, ecuaciones dependientes".

| Input | Engine | Verdad |
|---|---|---|
| `solve_system(0x+0y=1; 0x+0y=0; x; y)` | "infinitas soluciones" | sin solución |
| `solve_system(0x+0y=2; 0x+0y=3; x; y)` | "infinitas soluciones" | sin solución |

**Hipótesis:** la eliminación gaussiana detecta filas de coeficientes nulos como
"dependientes" (`0=0`) sin chequear el RHS aumentado (`0=c≠0` = inconsistente).

---

## Backlog priorizado de fixes (ROI retenible)

1. ~~**F8 sistema inconsistente**~~ ✅ **HECHO** (commit pendiente) — guarda de
   contradicción `0=c` en `classify_degenerate_2x2`; 0 deltas de huella.
2. ~~**F1 root-drop poli**~~ ✅ **PARCIAL** (commit pendiente) — rama `q=s` en
   `factor_monic_quartic_into_rational_quadratics` cierra el sub-caso monic-entero
   (`(x²-3)²`, `(x²-3x+1)²`, biquadrática, grado-6 dos-raíces). Residual: no-monic
   `(2x²-3)²`, contenido `(2x²-6)²`, fraccionario `(x²-½)²`, mult≥3 `(x²-3)³`
   (necesitan factorización general sobre ℚ). 0 deltas de huella.
3. ~~**F2 trig family-drop**~~ ✅ **HECHO (mayoría)** (2 commits) — RHS surd `sin(x)=√2/2`,
   argumento afín `sin(x-1)=0→{1+kπ}`, potencia impar `cos³=1→{2kπ}` (con guarda
   `|c|>1⇒∅`); todo verificado adversarialmente. Residual menor: offset exterior
   irracional `2cos(x)+√2`, RHS transcendente `sin(x)=π/4`, `tan³=1`, `sin(x²)=0`.
4. ~~**F3 inecuación racional 2-polos**~~ ✅ **HECHO** (commit pendiente) — reescribe
   `A(x) {op} B(x)` → `(A−B) {op} 0` (diferencia racional, denom grado≥1) y enruta por
   la ruta verificada `N/D {op} 0`. 0 deltas de huella.
5. ~~**F4 inecuación trig frontera**~~ ✅ **HECHO** (commit pendiente) — toque `sin(x)≥1`
   → conjunto-punto periódico `{π/2+2kπ}` (reduce a la ecuación); complemento `cos(x)<1`
   → residual honesto. Verificado adversarialmente. 0 deltas de huella.
6. ~~**F5 potencia envuelta**~~ ✅ **SOUND** (commit pendiente) — detector de Clase C
   extendido a base afín `(x-a)^e`, constante aditiva `x^(2/3)+1`, función `sqrt()`:
   declinan a residual honesto (antes rayo/complemento erróneo). La verificación
   adversarial mostró que son SOLUBLES vía `|α|`/`sqrt` reducción (próximo peldaño:
   SOLVE correcto en vez de decline; también gradúa los bare de Clase C). 0 deltas.
7. ~~**F7 sign-vía-abs**~~ ✅ **HECHO** (commit pendiente) — `g/|g| {op} c` = `sign(g) {op} c`
   reducido a condición de signo estricta sobre `g` (intervalos abiertos, polo `0/0`
   excluido). SOLVE correcto. 0 deltas de huella.
8. **F6 Taylor centro≠0** — ⚠️ **DIAGNOSTICADO, NO un ciclo acotado.** NO es un bug del
   constructor: `taylor_series_at_point_expr` produce la suma `Σ cₖ·(x−a)^k` CORRECTA (las
   derivadas son correctas: `diff(ln(x),x,4)=−6/x⁴`). El **SIMPLIFICADOR del engine**, al
   FACTORIZAR esa suma en la forma anidada `(x−a)^k` para presentarla, **invierte el signo
   de un coeficiente** — el resultado renderizado de `taylor(ln(x),x,1,5)` evalúa a `77/60`
   (≈1.283) en x=2 cuando la verdad es `47/60` (≈0.783). Es la clase de bug factor↔expand
   documentada en [[c5-diff-fold-rootcause]]. WORKAROUND exacto: `expand(taylor(...))` da el
   polinomio plano correcto (evita la factorización anidada). NO afecta a centro 0
   (Maclaurin, ruta analítica) ni a exp/cos/sin en centro≠0 (su forma anidada NO tiene el
   bug); SÍ afecta a `ln`, `1/x`, `sqrt`. Intentos de arreglo desde el comando taylor
   (expandir en el builder; simplify+expand en la regla) fallan: el engine RE-anida la
   salida tras la regla, o la expansión deja coeficientes sin reducir (`ln(1)`, `1^(−11/2)`)
   → o regresa otras funciones o emite basura. **Requiere arreglar la regla de
   factorización del simplificador compartido** (localizar el signo invertido en el
   factor-fold), un ciclo dedicado de simplificador, no del comando.

Cada uno (salvo F6) fue un ciclo acotado y retenible; F1, F2, F8 son los de mayor severidad
(wrong-answers silenciosos en formas de libro de texto). Ninguno es regresión de los
ciclos recientes; todos son deuda pre-existente que la caza multiagente sacó a la luz.

## Estado de cierre (2026-06-30)

**7 de 8 familias cerradas** en 10 commits, cada una con huella IDÉNTICA (0 deltas
estructurales, baseline-fiel stash-regen) y verificación adversarial donde introdujo
procedimientos de decisión:
- F8 `9494e212e`, F1 `45576c420`, F2 `1a927898c`+`435265bb7`, F3 `5be0a10b8`,
  F4 `956956225`, F5 `0b07bf22a`, F7 `37102f4f4`.
- F6 queda DIAGNOSTICADO (arriba) como bug del simplificador compartido (no del comando
  taylor); workaround `expand(taylor(...))`. Es la única no-cerrada y la de menor severidad
  (el cálculo es correcto; solo la presentación factorizada corrompe un signo).

### Ciclos de capacidad post-auditoría (mejor ROI)

- **VALLE de potencia RESUELTO** `fdcc55408` — gradúa de residual→CORRECTO los valles de
  numerador par (Clase C + Familia 5): `x^(2/3)>2`→`(-∞,-2^(3/2))∪(2^(3/2),∞)`,
  `(x-1)^(2/3)>4`→`(-∞,-7)∪(9,∞)`, `5-x^(2/3)>1`→`(-8,8)`, vía la reducción
  `c·(α)^(p/q)+d {op} k` ⟺ `|α| {op} ((k-d)/c)^(q/p)`. Verificado adversarialmente (30 casos,
  SOUND), 0 deltas de huella.
- **SIGUIENTE peldaño (especificado, no hecho):** recíprocas fraccionarias `1/x^(1/3)`,
  `1/sqrt(x)`, `x^(-1/3)`, `1/x^(2/3)` (exponente NEGATIVO) — hoy DECLINAN honestamente
  (residual sound, NO wrong-answer). Reducción: `c·(α)^e {op} k` (e<0, f=-e>0) ⟺ resolver la
  inecuación racional `c/W {op} k` (W=(α)^f) → W-intervalos, y back-sustituir `(α)^f ∈ W` con
  los solucionadores de potencia POSITIVA (monótono num-impar, valle num-par ya resueltos),
  excluyendo el polo `(α)^f>0`. Es ~120 líneas de casework (formas Div/sqrt, signos de c/k,
  paridad, dominio) — un proyecto acotado pero no trivial; los building-blocks (`x^(1/3)<1/2`,
  `sqrt(x)<1/2`, el valle) ya funcionan. F1-restante (no-mónico `(2x²-3)²`, fraccionario
  `(x²-½)²`, mult≥3 `(x²-3)³`) sigue siendo wrong-answer y requiere factorización general
  sobre ℚ.
