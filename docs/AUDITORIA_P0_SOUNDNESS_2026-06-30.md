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
3. **F2 trig family-drop** — enrutar RHS-radical / argumento-afín / `trig^n=c` al
   solver periódico (reutiliza la maquinaria de Clase A / P0-1).
4. **F3 inecuación racional 2-polos** — normalizar `a/(x-p)±b/(x-q)` a `N/D` antes del
   análisis de signo (eco del fix `Add`-racional de una fracción).
5. **F4 inecuación trig frontera** — dar el conjunto-punto periódico exacto en `=±1`,
   y declinar/ℝ-menos-puntos en `<1`,`>-1`.
6. **F5 potencia envuelta** — extender el detector de Clase C: mover constante
   aditiva, base afín `(x-a)^e`, forma función `sqrt()`.
7. **F7 sign-vía-abs** — excluir el polo `0/0` y resolver las inecuaciones de signo.
8. **F6 Taylor centro≠0** — corregir el signo del coeficiente en el constructor por
   diferenciación.

Cada uno es un ciclo acotado y retenible; F1, F2, F8 son los de mayor severidad
(wrong-answers silenciosos en formas de libro de texto). Ninguno es regresión de los
ciclos recientes; todos son deuda pre-existente que la caza multiagente sacó a la luz.
