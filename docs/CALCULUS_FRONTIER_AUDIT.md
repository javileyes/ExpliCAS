# Auditoría de frontera del engine de cálculo

Fuente de candidatos del bucle `/auto-mejora`. Cada ciclo retenido que
gradúe un item lo marca aquí (`[x]` + `*(graduado YYYY-MM-DD commit:
qué quedó cubierto y qué queda como peldaño)*`) en vez de borrarlo;
las re-auditorías añaden una sección nueva con fecha y dejan las
anteriores como historia de progreso.

## Auditoría 2026-06-12

Metodología: 6 agentes paralelos — 5 sondando el CLI release
(`target/release/cas_cli eval`) con ~150 expresiones de dificultad
creciente clasificadas en funciona / falla / residual honesto, y 1
leyendo el modelo de madurez, roadmap y velocidad del ledger.

### Cobertura medida

| Dimensión | vs. curso universitario | vs. CAS profesional | Nota |
|---|---|---|---|
| Diferenciación | ~80% | ~45-50% | mecánica sólida (cadena profunda, x^x, parciales, condiciones); bloqueada por estabilidad del simplificador |
| Integral indefinida | ~60-65% (58/92 sondas) | ~35-40% | racionales/por-partes/potencias trig casi completos; 6/6 no-elementales correctamente residuales |
| Límites | ~45-50% | ~20-25% | allowlist de patrones, no algoritmo; `e` inalcanzable por límite |
| Definidas/impropias | ~70-75% | ~35-40% | FTC/touches/impropias elementales sólidos; pre-simplificador sabotea casos |
| Calidad educativa | ~55% bien narrado | ~35-40% | condiciones de dominio sistemáticas (punto fuerte); por partes muda |

### Horizontes del propio repo (CALCULUS_ENGINE_STRATEGY.md, citas)

1. "serious educational ... dozens more retained ROI cycles" — a la
   velocidad observada (~19 ciclos retenidos/día de sesión en
   2026-06-10/11/12), días de sesión.
2. "mature elementary ... on the order of one hundred or more retained
   cycles" — 1-3 semanas de sesiones intensas, CON la advertencia de
   que la velocidad actual está inflada por ciclos baratos (espejos de
   signo, brazos Div); lo restante incluye ciclos arquitectónicos.
3. "a universal integration engine is not a bounded target for
   ordinary ROI cycles" — track del backend híbrido (Fases 0-3 done,
   4 in progress) + componentes de grado investigación (Risch, Gruntz,
   assumptions, funciones especiales).

## Cola priorizada (consumir desde /auto-mejora paso 1)

Clase F = ciclo-familia (patrón conocido, 1 ciclo). Clase A = ciclo
arquitectónico (scoping workflow primero, riesgo de huella alto).
Clase I = grado investigación / Deferred Horizons (no es un ciclo).

### P0 — soundness y confianza (antes que capacidad)

- [x] **(F) `0·∞` plegado a 0 en punto finito**: `limit(x·sinh(1/x²),x,0)`
  devolvía `0` cuando el límite real es `+∞` (sinh(1/x²)→+∞, y 0·∞ es
  indeterminado, no 0); mismo error en `x·cosh(1/x²)`, `x·exp(1/x²)`,
  `x·cosh(1/x)` (valor finito FALSO, no residual honesto).
  *(graduado 2026-06-14 e0710101b: dos causas — (1)
  `finite_total_real_unary_result` devolvía el cofactor como `sinh(∞)` SIN
  plegar (el fold de saturación cubría exp/abs/sin/cos/atan pero no
  sinh/cosh/tanh), leyéndose como acotado aguas abajo; ahora el fallback
  corre `fold_infinity_saturation` (sinh(∞)→∞, cosh(∞)→∞, tanh(∞)→1,
  exp(−∞)→0); (2) `finite_mul_result` devolvía 0 si CUALQUIER factor era 0
  sin comprobar que el otro fuese finito; ahora declina (residual honesto)
  cuando un factor es 0 y el otro ∞. `x·exp(−1/x²)→0` y `x·tanh(1/x²)→0`
  siguen correctos. Descubierto al prototipar la sustitución recíproca
  u=1/x para límites en ∞ — revertida por disparar justo este hueco —;
  huella de scorecard sin cambios, ningún fixture público lo capturaba.)*
- [x] **(A) Cuelgue del simplificador**: `diff(sin(x)^3*cos(x)^2, x)`
  timeout >30s con `depth_overflow depth=51 phase=Core`; mismo patrón
  da 12s en `diff((x^2*sin(x))/(x+1), x)` y
  `diff((x*ln(x))/(exp(x)*sin(x)), x)`. Respuestas correctas que no
  llegan valen cero. Relacionado: el bucle tan↔sin/cos que obligó a
  construir tan⁵ en forma expandida (ledger 2026-06-12).
  *(graduado 2026-06-12 ab8591792: la clase-cuelgue era explosión
  de probes especulativos de equivalencia-cero — cap de profundidad 2
  + presupuesto 48/pipeline con guard save/restore + franja de 24
  pipelines completos; sin²cos²−sin⁴ 0.4s, diff(sin³cos²) 50ms, el
  hermano preexistente sin⁴+cos⁴−1+2sin²cos² ahora prueba 0 en 1.5s,
  2 timeouts del corpus baseline ahora prueban 0. Quedan como
  peldaños: los cocientes-con-producto de 10-12s — preexistente,
  mejorado de 23.9s→12.5s de rebote —, la suma ancha de 3 identidades
  9s→45s, y el ruido WARN depth=51 en PostCleanup)*
- [x] **(F) `inf` como símbolo libre**: `integrate(e^(-x), x, 0, inf)`
  produce `(e^inf-1)/e^inf` sin aviso; solo `infinity` activa la
  maquinaria. Parsear `inf`/`oo` como infinito o rechazarlos con
  mensaje. *(graduado 2026-06-12 ca78c8164: inf/oo → Constant::
  Infinity en el mapa de constantes del parser + oo reservado;
  el glifo ∞ sigue rechazando con error claro — peldaño cosmético;
  -inf suelto en CLI es parseo de flags del shell, no del parser)*
- [x] **(F) Pasos corruptos**: `integrate(sin(x),x,0,pi)` etiqueta
  "Expandir secante" al evaluar cos(π); `sec(x)^2` tiene traza rota
  (el resultado tan(x) no aparece en ningún paso); pasos no-op
  before==after y ciclos expandir/refactorizar. Filtro de saneado de
  traza (eliminar no-ops, verificar que el último after == resultado).
  *(parcial 2026-06-12 3a43f063e: etiqueta falsa corregida — el
  preámbulo de valores exactos de sec/csc/cot llamaba la tabla trig
  SIN restringir el builtin y reclamaba cos(π)/cos(0) bajo su nombre;
  ahora gatea por su propia función + 2 traducciones nuevas. Quedan:
  la traza rota de sec²/csc² (el paso de integración desaparece
  cuando hay pre-pasos — entre el ensamblado y el optimizador
  semántico, necesita ciclo propio) y el filtro de no-ops. BONUS:
  integrate(sec(x)²+1) residual por el pre-simplificador — ejemplo
  vivo del item P2)*
  *(parcial 2026-06-13 025d98333: traza rota de sec²/csc² resuelta —
  el filtro de productividad de pasos truncaba los ciclos por ÍNDICE
  de estado, pero los pasos always-keep crecen la lista sin registrar
  estado, así que el ciclo tan→sin/cos→tan de sec² recortaba una
  posición de más y borraba el paso "Calcular la integral". Fix:
  registrar filtered.len() por estado y truncar ahí + descartar
  always-keep que sean no-ops de display. Verificado adversarialmente
  (2 lentes, 0 regresiones; huellas byte-idénticas). BONUS confirmado:
  integrate(sec(x)²+1) ahora resuelve vía la ruta Weierstrass del
  ciclo c6107abd5. Quedan como peldaño: el filtro de no-ops para
  reglas NON-always-keep — "Convert exp to Power" y "Agrupar términos
  semejantes" emiten pasos before==after en integrate(x/e^x),
  integrate(sin(x)²,x,0,π) — y las trazas FTC definidas terminan una
  reducción aritmética antes del resultado (muestran 1+1, resultado 2))*

### P1 — capítulos universitarios enteros a 0% (mayor densidad de valor)

- [x] **(F×3) Sustitución trigonométrica**: `sqrt(1-x^2)`,
  `x^2*sqrt(1-x^2)`, `sqrt(4-x^2)/x`, `1/(x*sqrt(x^2-1))` (→ arcsec),
  `1/(x^2*sqrt(x^2+4))`, `sqrt(x^2±a^2)` — capítulo completo de Calc
  II ausente; el semicírculo `∫√(1-x²) [-1,1] = π/2` falla. Nota: las
  formas 1/√(cuadrática) y p(x)/√(cuadrática) SÍ están (split
  Hermite); lo que falta es √(cuadrática) en el NUMERADOR y los
  cocientes con x en el denominador.
  *(parcial 2026-06-12 77ea29595: el lado NUMERADOR completo vía
  p·√q = (p·q)/√q delegado al split Hermite — √(1−x²), x²√(1−x²),
  √(4−x²), √(x²±1) asinh/acosh, √(2x−x²), y el semicírculo
  ∫₋₁¹√(1−x²) = π/2 por doble touch. Quedan: los cocientes con x en
  el DENOMINADOR — √(4−x²)/x, 1/(x·√(x²−1)) → arcsec,
  1/(x²·√(x²+4)) — que necesitan sustitución real u otra identidad)*
  *(completado 2026-06-13 069c38a1d: el lado DENOMINADOR entero vía
  u=√q sobre denominadores monomiales — 1/(x√(x²−1))→arctan(√(x²−1))
  (arcsec, condición honesta x<−1 or x>1), 1/(x²√(x²+4))=−√(x²+4)/(4x),
  √(4−x²)/x, √(1−x²)/x², √(x²±1)/x, m=3. El capítulo de sustitución
  trigonométrica de Calc II queda cubierto en ambos lados. Peldaños:
  radicandos con término lineal (completar cuadrado), m par ≥4,
  denominadores no monomiales (x+1)·√q)*
- [x] **(F) Sustitución algebraica general** (u=eˣ, u=√x, u=ax+b bajo
  radical): bloquea DOS familias de golpe — racionales de eˣ
  (`1/(1+e^x)`, `e^x/(1+e^(2x))` → arctan(eˣ), `1/(e^x+e^(-x))`) y
  radicales lineales (`x*sqrt(x+1)`, `exp(sqrt(x))/sqrt(x)`,
  `1/(sqrt(x)+1)`). Mejor ROI según la sonda: una pasada de
  rewrite+integrate-recursivo resuelve ambas.
  *(graduado 2026-06-12 8298dce96: la mitad u=√(ax+b) completa —
  racionales de (x, √(ax+b)) con coeficientes racionales vía
  x=(u²−b)/a, dx=(2u/a)du delegando a los dueños racionales:
  x·√(x+1), x²·√(x+1), x·√(2x−1), x·(x+1)^(3/2), 1/(√x+1),
  √x/(1+x), √(x+1)/x, pendientes negativas (2x+3)√(5−x), con
  condiciones de dominio honestas por canal (x≥−1 integral vs x>−1
  derivada). Quedan como peldaños: cofactores no racionales en u
  (e^√x/√x, sin(√x)), radicandos mixtos √x·√(x+1), y el cierre
  simbólico de diff(F)−integrando para superficies racionalizadas —
  dos filas van como verification_gap con round-trip numérico)*
  *(parcial 2026-06-12 ec314325b: la mitad u=eˣ completa — todo
  racional sobre átomos e^(kx) con k racional integra vía u=e^(cx),
  c=gcd de pendientes, delegando al backend racional y
  back-sustituyendo: 1/(1+eˣ), eˣ/(1+e²ˣ)→arctan(eˣ), e²ˣ/(1+eˣ),
  (eˣ−1)/(eˣ+1), 1/(e²ˣ−1) con su condición de polo, e^(x/2)/(1+eˣ).
  Quedan: u=√x radicales lineales, y la superficie 1/(eˣ+e⁻ˣ) que el
  pre-simplificador reescribe a 1/(2cosh(x)) antes de llegar al
  integrador — peldaño hiperbólico aparte)*
- [x] **(F) Weierstrass t=tan(x/2)**: `1/(2+cos(x))`, `1/(1+sin(x))` —
  estándar de examen universitario.
  *(graduado 2026-06-13 c6107abd5: racionales de sin(kx)/cos(kx) con
  argumento lineal compartido vía t=tan(kx/2) + pares polinómicos +
  gcd mónico + fallback al backend solo-incondicional: 1/(2+cos x)→
  (2/√3)arctan(tan(x/2)/√3), 1/(1+sin x), 1/(1+cos x)→tan(x/2),
  1/(3+2cos x), 1/(5+4sin x), sin x/(1+sin x), 1/(2+cos 2x),
  1/(sin x+cos x)→atanh. Quedan como peldaños: múltiplos mixtos
  (sin x con cos 2x, necesita pre-expansión de ángulo doble), offsets
  de fase, átomos tan/sec, canal de condiciones a través de la ruta
  de soporte, y el techo de profundidad del simplificador que deja
  4/6 filas como verification_gap)*
- [~] **(A) Motor 0/0 componible en punto finito**: la allowlist no
  invierte (`x/sin(x)` falla siendo `sin(x)/x` soportado), no compone
  (`sin(3x)/sin(5x)` → 3/5, `(1-cos x)/x²` → 1/2, `(sin x - x)/x³` →
  −1/6, `asin(x)/x`, `sinh(x)/x`), no encadena L'Hôpital/Taylor. El
  item de mayor frecuencia en cualquier curso.
  *(parcial 2026-06-13 339496d6e: motor de INFINITÉSIMOS EQUIVALENTES de
  primer orden — `first_order_equivalent_poly` extrae el equivalente
  polinómico de AMBOS lados (`f(u)~u` para sin/tan/asin/arcsin/atan/
  arctan/sinh/tanh con guard `u→0`, `e^u−1~u`, polinomios exactos,
  productos, Neg) y delega al `finite_rational_polynomial_value`
  existente (L'Hôpital polinómico). Cubre INVERSIÓN (`x/sin x=1`),
  COMPOSICIÓN (`sin 3x/sin 5x=3/5`, `sin x/sin 2x=1/2`, `tan 2x/sin 3x=
  2/3`) y los átomos que faltaban (`tan/asin/arctan/sinh/tanh /x=1`).
  Footprint-mínimo: corre tras las reglas sin/exp/log, solo dispara en
  0/0 genuino previamente residual. Verificado adversarialmente (3-lente
  scoping + 2-lente refutación, 141 sondas, 0 violaciones).)*
  *(parcial 2026-06-13 4ccd1b930: peldaño (1) el orden SUPERIOR / Taylor con
  cancelación de sumas GRADUADO — `apply_finite_taylor_quotient_rule` añade
  un motor de Maclaurin autocontenido (orden de truncado 12) que corre TRAS
  la regla de equivalentes y solo sobre 0/0 en x=0 que esos pasos dejaron
  residual. `taylor_at_zero` expande estructuralmente (polinomios exactos,
  Add/Sub/Neg/Mul, Pow(E,arg), Pow(base,n entero≥0), y Function(f,[arg])
  componiendo la serie estándar de f vía Horner; tan=sin/cos por división
  de series). Compara los órdenes mínimos no-nulos: num>den→0, num==den→
  cociente exacto de coeficientes líderes, si no declina. SOUND: el
  truncado es EXACTO para un límite de orden líder (solo descarta órdenes
  estrictamente mayores); la lista honesta sobrevive gratis (`sin(1/x)`
  declina porque su argumento `1/x` es `Pow(x,-1)`, que el constructor de
  series rechaza). Cubre `(1-cos x)/x²=1/2`, `(sin x−x)/x³=−1/6`,
  `(tan x−x)/x³=1/3`, `(e^x−1−x)/x²=1/2`, `(cosh x−1)/x²=1/2`,
  `(arctan x−x)/x³=−1/3`, `(arcsin x−x)/x³=1/6`, composiciones anidadas
  (`(sin(sin x)−x)/x³=−1/3`) y `(sin(tan x)−tan(sin x))/x⁷=−1/30`. Verificado
  adversarialmente (2-lente, 55 sondas vs SymPy, 0 unsound).)*
  *(parcial 2026-06-14 9ae1f606c: la franja EXPONENCIAL de (4) L'Hôpital
  general — combinaciones lineales de exponenciales de base general sobre un
  polinomio de primer orden — GRADUADA vía
  `apply_finite_exp_linear_combination_quotient_rule`: lee el numerador como
  Σ c_i a_i^(g_i) (+ ctes), acumula valor en 0 (=0 para 0/0 genuino) y
  derivada N'(0)=Σ c_i g_i'(0) ln(a_i) simbólica, y devuelve N'(0)/h'(0).
  `(2^x−3^x)/x=ln2−ln3`, `(2^(3x)−3^x)/x=3ln2−ln3`, `(e^x−2^x)/x=1−ln2`.
  Sound por construcción (L'Hôpital de primer orden, derivadas exactas);
  declina fuera de la clase. Peldaños restantes del item: (2) átomo con
  argumento NO-cero en el punto
  (`tan x/sin x` en π=−1, `sin x/(x−π)` en π=−1) — necesita el
  equivalente local en el cero del argumento; (3) log en el numerador/
  composición (`ln(1+x)/sin x`) — excluido por la ruta de base no-natural
  `valor/ln(base)`; (4) encadenamiento L'Hôpital general)*
- [x] **(A) Formas exponenciales 1^∞/0^0/∞^0** vía `exp(lim g·ln f)`:
  `(1+1/x)^x → e`, `(1+2/x)^x → e²`, `(1+x)^(1/x) → e`, `x^x → 1 en
  0+`, `(2^x+3^x)^(1/x) → 3`. Hoy la constante `e` es inalcanzable
  por límite — invalida un capítulo del temario.
  *(GRADUADO — capítulo completo: la constante `e` ya es alcanzable por
  límite y las tres formas indeterminadas exponenciales resuelven. Las tres
  reducen a `exp(lim exp·ln base)` con la maquinaria del sub-límite por
  forma.)*
  *(parcial 2026-06-14 0a2672c98: la forma ∞^0 graduada, CERRANDO el
  capítulo — dos fundamentos acoplados: `general_base_exponential_limit_at_
  infinity` (`b^x→∞/0/1` por análisis de signo; el motor crecía `e^x` pero
  dejaba `2^x` residual) e `inf_to_zero_power_limit_at_infinity` (base→+∞,
  exp→0 → `exp(lim exp·ln base)`, racionalizado+presimplificado para que la
  dominancia log-exp-suma vea el ln desnudo; `(2^x+3^x)^(1/x)=3`). Verificado
  adversarialmente (2-lente, 48 sondas, 0 unsound). Peldaños menores: ∞^0 con
  coeficiente en el exponente (`(2^x+3^x)^(2/x)=9` queda residual por el
  `c·ln`), bases e-mixtas, y las segundas-órdenes transcendentes del 1^∞.)*
  *(parcial 2026-06-14 a723ff67d: la forma 1^∞ EN INFINITO graduada — la
  constante `e` ya es alcanzable por límite. `one_to_infinity_power_limit_
  at_infinity` reduce 1^∞ a `exp(lim exp·(base−1))` usando `ln(1+h)~h`
  (válido porque la base→1 fuerza h→0), racionaliza el producto sobre
  denominador común (`rationalize_to_fraction`) y reutiliza el límite
  racional; pliega `e^0=1, e^1=e, e^(±∞)=∞/0`. Cubre `(1+a/x)^x=e^a`,
  `(1+1/x)^(kx)=e^k`, `((2x+1)/(2x-1))^x=e`, `(1+1/x²)^x=1`,
  `(1+1/x)^(x²)=∞`. Verificado adversarialmente (2-lente, 85 sondas, 0
  unsound): la trampa de SEGUNDO ORDEN `cos(1/x)^(x²)=e^(−1/2)` declina
  (base transcendente opaca al racionalizador) — nunca emite valor erróneo.)*
  *(parcial 2026-06-14 976efd869: la forma 1^∞ EN PUNTO FINITO graduada —
  la OTRA definición de e, `(1+x)^(1/x)=e`, resuelve, y las bases de SEGUNDO
  ORDEN también. `apply_finite_one_to_infinity_power_rule` gatea por el
  PRODUCTO (base→1 y `L=lim exp·(base−1)` no-nulo, que fuerza exponente
  divergente) en vez de por el exponente (1/x en 0 no tiene límite bilateral
  con signo). Como L lo evalúa la maquinaria finita COMPLETA (Taylor +
  infinitésimos equivalentes), es ESTRICTAMENTE más fuerte que la hermana en
  ∞: `cos(x)^(1/x²)=e^(−1/2)`, `(sin x/x)^(1/x²)=e^(−1/6)`,
  `(1+sin x)^(1/x)=e`. Sound: `lim g·ln(1+h)=lim(g·h)·lim(ln(1+h)/h)=L·1`,
  el término `−h²/2` se absorbe en el factor `ln(1+h)/h→1`. Verificado
  adversarialmente (2-lente, 47 sondas, 0 unsound, cross-check mpmath ~14
  cifras).)*
  *(parcial 2026-06-14 bbe89428f: la forma 0^0 graduada — `x^x → 1` en 0+.
  `apply_finite_zero_base_power_rule`: como x>0 a la DERECHA de 0, `x^g=
  exp(g ln x)` es real, y el límite es `exp(lim g ln x)`; `x^x=exp(lim x ln
  x)=exp(0)=1`. Gateado al lado derecho con base = la variable desnuda (signo
  positivo conocido en la aproximación); el bilateral `x^x` queda residual
  (complejo para x<0) y una base no-variable (`sin(x)^x`) declina (signo no
  probado). Peldaño restante: ∞^0 con base exponencial dominante
  (`(2^x+3^x)^(1/x)=3` necesita `ln(2^x+3^x)/x → ln 3`).)*
- [x] **(F) ∞−∞ con radicales** (racionalización por conjugado):
  `sqrt(x^2+x)-x → 1/2`, `sqrt(x+1)-sqrt(x) → 0`, y en punto finito
  `(sqrt(x)-2)/(x-4) → 1/4`.
  *(parcial 2026-06-13 d78ce2c0e: `sqrt(ax²+bx+c) − (dx+e)` a ±∞ con
  √a racional y cancelación de términos líderes ya resuelve vía forma
  cerrada `b/(2√a)−e` — `sqrt(x²+x)−x=1/2`, `sqrt(x²+1)−x=0`,
  `sqrt(4x²+x)−2x=1/4`, `x−sqrt(x²−x)=1/2`. Gate de cancelación exacta
  (los divergentes declinan).)*
  *(parcial 2026-06-13 d7dd00024: sqrt−sqrt completado para radicandos
  del mismo grado (1 o 2) y mismo líder — `sqrt(x+1)−sqrt(x)=0`,
  `sqrt(x²+x)−sqrt(x²−x)=1`, `sqrt(4x²+x)−sqrt(4x²−x)=1/2` vía
  `(b_P−b_Q)/(2√a)`. El lado +∞ del item queda cubierto. Peldaños:
  √a irracional (`sqrt(2x²+x)−sqrt(2x²−x)=1/√2`), grado ≥3)*
  *(graduado 2026-06-13 15bc39585: el lado PUNTO FINITO completado —
  `(scale·√(ax+b)+k)/den` en 0/0 vía conjugado: `(√x−2)/(x−4)=1/4`,
  `(√x−3)/(x−9)=1/6`, `(√(2x+1)−3)/(x−4)=1/3`, denominador cuadrático
  `(√x−2)/(x²−16)=1/32`. Gate de seguridad: numerador 0 en el punto +
  raíz racional + conjugado ≠0; los polos no-0/0 y las raíces
  irracionales declinan, con condiciones de dominio honestas. Item
  cerrado salvo los peldaños √a irracional y grado ≥3 anotados)*
  *(hermano RACIONAL 2026-06-14 1881980a6: el ∞−∞ de funciones racionales
  (sin radicales) también resuelve — `rational_difference_limit_at_infinity`
  pone los operandos sobre denominador común y reutiliza
  `rational_poly_limit`: `(x²+1)/(x+1)−x=−1`, `x²/(x−1)−x=1`,
  `x²/(x+1)−x²/(x+2)=1`, `x³/(x+1)−x=+∞`. Corre al final de la cadena (las
  diferencias con límites finitos conservan su traza aditiva; operandos no
  racionales declinan al conjugado/dominancia).)*
  *(compañera 0·∞ 2026-06-14 b91912327: el PRODUCTO `factor·(diferencia
  conjugada→0)` —la forma 0·∞ que el bare-difference dejaba al multiplicativo,
  que declinaba— ya resuelve a +∞ vía `radical_conjugate_product_limit_at_
  infinity`: racionaliza la diferencia (numerador conjugado `s²Q−L²` sobre la
  suma conjugada `~2s√a·x`) para leer su decaimiento como término líder
  `K·xᵖ`, lee el factor (polinomio o `escala·√(poli)` con líder racional) como
  `c·xᵠ`, y devuelve el límite por la suma de exponentes: `c·K` si `p+q=0`, `0`
  si `<0`, declina (deja `+∞` a dominancia) si `>0`. Términos aditivos
  aplanados y partidos en √ vs resto polinómico, así que cola lineal partida
  (`x·(√(x²+2x)−x−1)=−1/2`), ambas orientaciones, y `√−√` cuadrático
  (`x·(√(x²+x+1)−√(x²+x))=1/2`) caen igual. `x·(√(x²+1)−x)=1/2`,
  `x·(√(x²+4)−x)=2`, `√x·(√(x+1)−√x)=1/2`. Gate de cancelación líder
  (`s√a+r1=0`); SOLO a +∞ (el lado −∞ es trampa: misma forma diverge ahí, se
  deja residual honesto); declina líder irracional (`√(2x)·…`) y factor que
  supera el decaimiento (`x²·(√(x²+1)−x)` diverge). Peldaños: análogos cbrt
  (`x²·(∛(x³+1)−x)=1/3`), lado −∞ con valor finito, coeficientes irracionales.
  Verificado adversarialmente con 37 sondas mpmath dps=60.)*

### P2 — familias y mejoras de alto valor (1 ciclo cada una)

- [x] **(F) Touch con límite x^a·ln(x)^b → 0**: `ln(x)^2 [0,1]` (=2),
  `x*ln(x) [0,1]` (=−1/4), `ln(x)/sqrt(x) [0,1]` (=−4) residuales con
  antiderivadas elementales; la dominancia potencia-log existe en el
  lado lateral pero no cubre estas combinaciones. Arregla 3+ familias.
  *(graduado 2026-06-13 52f0fb4f9: el hueco estaba en el MOTOR DE
  LÍMITES, no en el integrador — las antiderivadas ya se conocen y el
  borde definido ya las evalúa por límite lateral de F. La dominancia
  `power_log_dominance_zero_limit` resolvía solo el monomio `u^p·ln(u)^q`;
  `apply_finite_one_sided_power_log_polynomial_zero` la generaliza a
  `Σ c·(var-pt)^a·P(ln(var-pt))` con todos a>0 → 0 (potencia × polinomio
  en ln, sumados). Resuelve `∫₀¹ ln²=2`, `x·ln=−1/4`, `ln/√x=−4`,
  `x²·ln=−1/9`, `√x·ln=−4/9`. Gate de soundness: potencia neta
  estrictamente positiva + al menos una potencia presente (un término
  constante o de log-puro bloquea: `x ln x + 5 → 5`), y ln gateado a
  potencias ENTERAS no-negativas (ln<0 cerca de 0). Verificado
  adversarialmente (2 lentes, 115 sondas, 0 violaciones; el trap
  `x·e^(1/x)` queda residual = +∞). Peldaños: touches con exp/trig en el
  borde, y la forma exponencial `1^∞/0^0` que necesita el interno `∞·0`
  robusto)*
- [x] **(F) Gaussiana/Gamma por tabla**: `e^(-x^2) [0,∞) = √π/2`,
  `(-∞,∞) = √π`, `x^2*e^(-x^2) [0,∞) = √π/4`, `e^(-x)/sqrt(x) = √π` —
  la impropia más famosa de la universidad; tabla pequeña de formas
  patrón (la indefinida debe SEGUIR residual).
  *(graduado 2026-06-13 cda9fbca5: la familia Gaussiana de momentos
  `∫ x^(2n) e^(-a x²)` sobre semirrecta o recta completa vía
  `(1/2)(2n)!/(4^n n!)√π/a^(n+1/2)` — `e^(-x²)[0,∞)=√π/2`,
  `[-∞,∞]=√π`, `x²e^(-x²)=√π/4`, `x⁴e^(-x²)=3√π/8`,
  `e^(-2x²)=½√(π/2)`. Gating fuerte: bounds infinitos, exponente puro
  cuadrático, cofactor par, a>0 — indefinida/bounds finitos/no-cuadrático
  declinan (honestidad intacta, verificada adversarialmente — cazó y
  arregló un bug de coeficiente perdido). Peldaños: las formas Gamma
  (`e^(-x)/√x`, `x^n e^(-x)=n!`), el cofactor con cuadrado completado
  `e^(-x²+x)`, y `c·e^(-x²)` en forma Mul anidada)*
  *(Gamma graduada 2026-06-13 5ee37ea63: la familia Gamma de
  medio-entero `∫₀^∞ x^(m-1/2) e^(-ax) = (2m)!/(4^m m!)/a^m √(π/a)` —
  `e^(-x)/√x=√π`, `√x·e^(-x)=√π/2`, `x^(3/2)e^(-x)=3√π/4`,
  `e^(-2x)/√x=√(π/2)`. El entero `x^n e^(-x)=n!` ya resolvía vía
  antiderivada elemental. `match_gamma_integrand` ACUMULA sobre Mul/Div/Neg
  (potencia neta + decay lineal + constante) — un walker para todas las
  formas. Gating: exponente decay lineal puro, potencia medio-entera
  (los enteros caen a la antiderivada), s≥−1/2 (divergentes residuales),
  a>0, solo [0,∞). Verificado adversarialmente (58 sondas, valores
  exactos sin coefficient/sign drop; cazó un gap de coeficiente −1 unitario
  → arreglado con el brazo Neg). Peldaños restantes del item Gaussiano:
  cuadrado completado `e^(-x²+x)` y `c·e^(-x²)` Mul anidada)*
- [ ] **(A) Pre-simplificador vs integrador**: reescribe
  `1/(sqrt(x)*(1+x))` a `(x^(3/2)-x^(1/2))/(x^3-x)` y `cos(5x)` a
  Chebyshev en cos(x), destruyendo la sustitución obvia
  (`[0,∞) = π`) y la ortogonalidad de Fourier `sin(3x)cos(5x)
  [-π,π] = 0`. Integrar sobre la forma original primero, o enseñar al
  integrador las formas reescritas (precedente: reconocedor Chebyshev
  del ledger 2026-06-12). Ejemplo vivo adicional:
  `integrate(sec(x)^2 + 1, x)` se vuelve residual porque el
  pre-simplificador lo machaca a `(2cos²−1+3)/(2cos²)`.
- [~] **(F) Detección estructural sin antiderivada**: imparidad en
  `[-a,a]` para integrandos no elementales (`sin(x)/(1+x^2) [-1,1] =
  0`), abs por tramos (`|x| [-1,1] = 1`, `e^(-|x|) (-∞,∞) = 2`),
  test-p completo (`1/sqrt(x) [1,∞) = ∞`, hoy residual mientras `1/x`
  sí diverge), divergencia oscilatoria declarada (`sin(x) [0,∞)`).
  *(parcial 2026-06-14 b5f80b09f: IMPARIDAD en `[-a,a]` GRADUADA —
  `odd_symmetric_definite_integral_rewrite` corre como fallback estructural
  donde la antiderivada es None y resuelve a 0 bajo tres obligaciones
  independientes: bornes finitos simétricos (lower=-upper sobre el endpoint
  racional+pi+e), imparidad probada por `parity_in_var` (clasificador sound
  y conservador {Odd, Even, Unknown}: símbolo ajeno=par, suma conserva
  paridad solo si ambos términos coinciden, producto/cociente suma paridades,
  potencia entera por paridad del exponente, base constante positiva b^g par
  sii g par, composición por la clase del builtin externo) e INTEGRABILIDAD
  vía el MISMO certificado que hace `int(1/x,-1,1)` undefined (Certified
  estricto). Resuelve `sin(x)/(1+x^2)`, `sin(x)e^(x^2)`, `sin(x^3)`,
  `tan(x)e^(x^2)` en `[-1,1]`; declina con corrección `1/x` (undefined),
  `tan e^(x^2) [-2,2]` (polo en π/2), integrandos pares e intervalos
  asimétricos. Verificado adversarialmente (2-lente, 80 sondas, 0 unsound).
  Peldaños restantes: abs por tramos (`|x| [-1,1]=1`, `e^(-|x|)`), test-p
  completo, divergencia oscilatoria declarada, y la narración educativa
  específica de simetría — hoy el paso es el envoltorio genérico "Calcular
  la integral".)*
- [ ] **(F) Por partes narrada**: la plantilla completa ('Elegir u y
  dv' → 'Calcular du y v' → 'Aplicar la fórmula') existe y la usa
  `x·ln(x)`, pero `x·eˣ`, `x·cos x`, `arctan(x)`, `x²eˣ`, `eˣ·sin x`
  solo dicen "Usar integración por partes" (o nada: `ln(x)` da cero
  substeps). Es LA técnica central del curso y es cableado.
- [ ] **(F) Residuales con motivo**: 'Conservar integral residual' no
  distingue "no elemental (necesitaría erf/Si/Ei)" de "el motor aún
  no lo soporta" (`sqrt(1-x²)`, `|x|`, `1/(x⁴+1)` son elementales y
  quedan igual que `e^(-x²)`). Campo de motivo como el warning de
  límites.
- [~] **(F) Cuárticas+ irreducibles**: `1/(x^4+1)`, `1/(x^6-1)` —
  factorización real en cuadráticas con coeficientes irracionales
  (√2) para fracciones parciales.
  *(precursor 2026-06-13 3a267bdf2: el kernel `1/(cuadrática con
  raíces irracionales)` ya integra — `1/(x²−2)`, `(x+b)²−a` vía forma
  log con √c simbólico, desbloqueando `1/(u²−2)`.)*
  *(parcial 2026-06-13 962a01ddb: `(ax²+b)/(x⁴+1)` graduado vía
  sustitución simétrica u=x∓1/x — `1/(x⁴+1)`, `x²/(x⁴+1)`,
  `(x²±1)/(x⁴+1)`, con condición honesta x≠0 (la sustitución salta en
  0). Peldaños restantes: generalizar a `(ax²+b)/(x⁴+e)` con e no
  cuadrado perfecto (radicales anidados); la forma continua para
  integrales definidas que crucen 0; `1/(x⁶−1)` levantar el cap de
  factores del backend racional; `1/(x⁵−1)` las cuárticas ciclotómicas
  Φ5 con √5)*
- [x] **(F) Composición de límites con interno conocido**:
  `e^(1/x) en 0±` (→ ∞ / 0), `atan(1/x) en 0+` (→ π/2) fallan aunque
  `1/x → ±∞` resuelve; regla de composición continua/monótona barata
  (la tabla saturante en ∞ ya existe — reutilizarla desde laterales).
  *(parcial 2026-06-13 457b8d5d8: el lado BILATERAL con interno → ∞ con
  signo definido ya resuelve — `e^(-1/x²)→0`, `e^(1/x²)→∞`,
  `atan(1/x²)→π/2`, `tanh(1/x²)→1` — vía fold de saturación f(±∞) sobre
  la salida del límite (arctan/tanh/exp/ln/sqrt/sinh/cosh; excluye
  sin/cos/tan oscilantes). Verificado adversarialmente: el caso
  bilateral con laterales DISTINTOS (`e^(1/x)` en 0) queda correctamente
  residual. Nota de soundness: el fold NO se
  registró como regla global (ensanchaba el bug preexistente ∞−∞=0 en
  aritmética cruda); queda confinado a salidas de límite vetadas)*
  *(graduado 2026-06-13 393388fbb: cubierto el peldaño UNILATERAL —
  `apply_finite_one_sided_composition_rule` ganó ramas `Pow(E,g)` y
  `Function(f,[g])` que resuelven el límite interno lateral, leen su
  signo de ∞ y pliegan f(±∞) con el MISMO `fold_infinity_saturation`. La
  puerta de oscilación es el propio fold (`folded != candidate`), sin
  lista explícita. Honestidad estructural: el bilateral solo resuelve vía
  `matching_finite_bilateral_one_sided_result` (ambos lados deben
  coincidir), así que `e^(1/x)`/`atan(1/x)`/`tanh(1/x)` en 0 siguen
  residuales. Verificación adversarial 60+ sondas: cero violaciones de
  soundness, oscilantes declinan, bilaterales de poste par del ciclo 7
  sin regresión. Bonus: el fold `0·finito→0` en `combine_limit_product`
  normaliza productos cuyos dos factores ahora resuelven. Peldaños
  abiertos fuera de alcance: `cosh(1/x)` bilateral (ambos lados → +∞
  pero declina), `asinh/acosh/coth/sech` (folds no implementados),
  coeficientes irracionales `e^(π/x)`, y `∞·(π/2)` lateral que necesita
  cofactor racional)*
  *(peldaño cosh cerrado 2026-06-13 b349e056e:
  `apply_finite_bilateral_even_saturating_pole_rule` resuelve
  `cosh(1/x)→∞` bilateral — cosh es PAR, así que cosh(±∞)=+∞ vale en
  ambos lados pese a que el polo impar 1/x diverge con signos opuestos.
  Gateado a Cosh + inner divergente en ambos lados (reutiliza
  `one_sided_inner_infinity_sign` y `saturate_outer_at_infinity`). Nota
  de alcance: se prototipó la regla GENERAL "bilateral = valor lateral
  común" (sólida por el teorema, resolvía `1/|x|`, `log_b(|x|)`,
  `sqrt|x|`, `exp(ln|x|)`) pero flipeaba ~6 contratos conservadores de
  "composición finita no soportada con seguridad" de golpe; se redujo a
  cosh (par, fold independiente del signo), que NO toca ningún contrato.
  Verificado adversarialmente (52 sondas, 0 violaciones; semi-definidos
  `cosh(1/√x)`/`cosh(ln x)` declinan por lado indefinido). Peldaño
  preexistente anotado: `cos(1/x²)`/`sin(1/x²)` filtran `cos(infinity)`
  sin plegar — el combine bilateral de inner oscilante de potencia par)*
  *(fuga cos/sin cerrada 2026-06-13 055929883:
  `apply_finite_total_real_unary_composition_rule` declina cuando el outer
  es Sin/Cos y el argumento → ±∞ — sin/cos oscilan en ∞, sin límite. Era
  honestidad: el outer saturante (atan/exp/tanh/cosh) filtra
  `outer(infinity)` que la capa eval pliega, pero sin/cos no pliegan y
  filtraban `cos(infinity)` como pseudo-valor en la familia `sin(1/x)` que
  nunca debe resolver. Sustractivo (solo añade decline); el polo impar
  `cos(1/x)` ya estaba bien por el gate "fold changed it" unilateral.
  Verificado adversarialmente (41 sondas, 0 violaciones; saturantes y
  squeeze intactos))*
- [x] **(F) Squeeze y dominancia fraccionaria**: `x*sin(1/x) → 0 en
  0`, `(x+sin x)/x → 1 en ∞`, `ln(x)/sqrt(x) → 0 en ∞` (la dominancia
  entera `ln(x)/x` sí funciona).
  *(parcial 2026-06-13 74544e793: cubierto el SQUEEZE en punto finito —
  `apply_finite_squeeze_bounded_product_rule` resuelve a 0 todo producto
  con un factor infinitésimo y un factor oscilante globalmente acotado
  sin límite (`sin/cos/atan/arctan/tanh` de una función racional de la
  variable): `x·sin(1/x)`, `x²·cos(1/x)`, `sin(x)·sin(1/x)`,
  `(x-2)·sin(1/(x-2))` en 2 → 0. Footprint-mínimo: solo dispara cuando
  hay un factor acotado SIN límite, así que `x·sin(x)` sigue por la ruta
  genérica. Honestidad triple-gateada: `sin(1/x)` solo y `2·sin(1/x)`
  (sin infinitésimo) quedan residuales, `(1/x)·sin(1/x)` normaliza a Div.
  Verificado adversarialmente (2 pasadas, ~230 sondas): cazado y
  corregido un bug de soundness — denominador idénticamente cero
  `1/(x-x)` daba `sin(1/0)` indefinido como "acotado"; el gate ahora
  exige denominador no-cero. Peldaños restantes: el cociente con ruido
  aditivo acotado `(x+sin x)/x → 1 en ∞` (la maquinaria
  `polynomial_growth_info_with_bounded_additive_noise` existe pero no
  está cableada al cociente racional general) y la dominancia
  log-potencia FRACCIONARIA `ln(x)/√x → 0 en ∞` (`ln(x)/x` y `ln(x)/x²`
  enteras ya funcionan; falta extender a `x^(p/q)`). Gaps cosméticos de
  completitud del squeeze: argumentos sin normalizar `x^(-1)`, `1/x+x²`
  declinan conservadoramente aunque el límite real es 0)*
  *(dominancia fraccionaria graduada 2026-06-13 da56c3a08:
  `polylog_power_dominance_limit_at_infinity` resuelve `c·ln(x)^a/x^b→0`
  y `c·x^b/(c'·ln(x)^a)→sign(c/c')·∞` con a≥1 entero y b>0 racional —
  `ln(x)/√x=0`, `ln(x)²/x=0`, `ln(x)³/x=0`, `ln(x)/x^(1/3)=0`,
  `√x/ln(x)=∞`, `x/ln(x)²=∞`. `positive_power_tail` reconoce el exponente
  RACIONAL (x^(1/2), x^(2/3) de primera clase). Gating: solo +∞ (ln
  indefinido en −∞), coeficientes no-cero, potencia genuinamente positiva
  (`ln(x)/x^(-2)=ln·x²→∞`, no 0). Verificado adversarialmente (70 sondas,
  0 violaciones; cerrado un brazo Neg faltante). Peldaño restante: el
  cociente con ruido aditivo acotado `(x+sin x)/x → 1`)*
  *(ruido aditivo graduado 2026-06-13 ac4dd379f — ITEM CERRADO:
  `bounded_noise_rational_limit_at_infinity` resuelve cocientes
  `poly+ruido_acotado / poly+ruido_acotado` por la parte polinómica —
  `(x+sin x)/x=1`, `(2x+cos x)/x=2`, `(x²+sin x)/(x²-1)=1`,
  `(x+sin x)/(2x+1)=1/2`, `x/(x+sin x)=1`, `(x+cos x)/x²=0`,
  `(x²+sin x)/x=∞`. Cablea `polynomial_growth_info_with_bounded_additive_
  noise` al cociente racional con la misma comparación de grados. El ruido
  NO acotado (`x·sin x`) declina. Verificado adversarialmente (54 sondas,
  0 violaciones))*
- [x] **(F) Producto-a-suma residual mutilado**: `sin(3x)cos(5x)`
  indefinida queda residual Y mutilada (expandida en potencias de
  cos); el reconocedor producto-a-suma cubre frecuencias distintas
  con a≠±b — revisar por qué esta combinación escapa.
  *(graduado 2026-06-13 2cd81323b: la mutilación era una regla — la
  "Quintuple Angle Identity" expandía cos(5x)/sin(5x) antes del
  producto-a-suma y NO estaba en la lista de desactivadas de
  IntegratePrep. Añadida a la lista (scoping de 3 agentes: footprint
  bajo, además ARREGLA `integrate(cos(5x))→sin(5x)/5` y
  `integrate(cos(5x)²)` que era residual). Y un sibling Werner
  sin-coeficiente con `/2` explícito cubre sin·cos/cos·cos/sin·sin
  con gate A≠B: `sin(3x)cos(5x)=1/16(4cos2x−cos8x)`,
  `cos(3x)cos(5x)`, `sin(3x)sin(5x)`. Resuelve parcialmente el item
  clase A "Pre-simplificador vs integrador" para este caso)*
- [x] **(F) Potencias trig mixtas incoherentes**: `sin^2(x)cos^3(x)`
  residual mientras `sin^3*cos` y `sin^5` funcionan.
  *(graduado 2026-06-13 59c742081: productos sin(kx)^m·cos(kx)^n con
  argumento lineal compartido y una potencia impar (ambas ≥2) vía
  u=sin (cos impar) o u=cos (sin impar) → integrando polinómico
  u^kept·(1−u²)^spare delegado al integrador de polinomios. Cubre
  sin²cos³, sin³cos², sin⁴cos³, sin⁵cos², sin³cos⁴, sin(2x)³cos(2x)².
  Gate de intención min(m,n)≥2: los casos f^n·f' (sin³cos) y
  ambas-pares (sin²cos²) conservan su dueño. sin⁵cos² va como
  verification_gap — ambos canales simbólicos no recolapsan la derivada
  de grado 7, verificado numéricamente)*
- [x] **(F) sech/csch no parsean**: `sech(x)^2` da "función no
  definida".
  *(graduado 2026-06-13 758e54e73: el parser desugariza sech→1/cosh,
  csch→1/sinh, coth→cosh/sinh en el lowering — `sech(x)²`,
  `integrate(sech(x)²)=tanh(x)`, `integrate(csch(x)²)=−coth(x)`,
  `diff(sech(x))=−sinh/cosh²`, `diff(coth(x))=−1/sinh²`, `sech(0)=1`.
  Gated a los tres nombres exactos con un argumento. `integrate(sech(x))`
  sigue residual — gudermannian es peldaño aparte)*
- [~] **(F) Bounds con e**: `integrate(1/x, x, 1, e)` residual
  (`ln(e)` no se evalúa); `tan(x) [0,π/4]` devuelve
  `ln(|cos(0)/cos(π/4)|)` sin plegar `cos(0)=1`.
  *(parcial 2026-06-13 a41cc8e55: la mitad de los bounds con e ya
  certifica — `e` y múltiplos racionales (2e, e/2, −e) son endpoints
  finitos con enclosure racional en el certificado de polo:
  `∫1/x [1,e]=1`, `∫1/x² [1,e]=(e−1)/e`, `∫2/x [1,e]=2`, y los polos
  se ubican (1/(x−2) en [1,e] diverge, 1/(x−3) certifica). Peldaños:
  `e²` (potencia, no múltiplo) y `√2` (algebraico) siguen Symbolic;
  el plegado `cos(0)=1` en la sustitución FTC de `tan [0,π/4]` es
  cosmético aparte)*
- [~] **(F) FTC/Leibniz en diff**: `diff(integrate(f(t),t,0,x), x)`
  → `f(x)` (+ regla de Leibniz con límites variables).
  *(parcial 2026-06-13 aea379c97: la regla de Leibniz
  `d/dx ∫_a(x)^b(x) f = f(b)b' − f(a)a'` ya aplica a integrandos
  ELEMENTALES no-integrables — `diff(∫e^(t²) [0,x]) = e^(x²)` (gaussiana),
  Fresnel `sin(t²)`, Si `sin(t)/t`, con regla de la cadena (`[0,x²] →
  2x·e^(x⁴)`) y bound inferior con cambio de signo. Las indefinidas
  siguen residuales (honestidad intacta): demuestra que esos residuales
  son frontera de PRESENTACIÓN, no de conocimiento. Peldaño restante:
  `f` OPACA simbólica (`diff(∫f(t)[0,x])=f(x)`) — el engine rechaza
  funciones desconocidas; necesita soporte de funciones simbólicas)*
- [ ] **(F) Diagnóstico de no-existencia en límites**: `sin(1/x)` en 0
  y laterales discrepantes deberían reportar "no existe" con motivo,
  no un residual genérico.

### P3 — educativo transversal

- [ ] **(F) Límites con pedagogía**: los soportados no justifican nada
  (`sin(x)/x = 1` sin nombrar el límite notable/L'Hôpital/sandwich);
  impropias muestran `lim` sin evaluarlo con justificación
  (`lim e^(-x)(-x-1) = 0` por dominancia).
- [ ] **(F) Presentación**: ~10 nombres de regla en inglés dentro de
  narración española ('Normalize Negative Exponent', 'Identity
  Power'...); `--steps` ignorado en modo texto del CLI (los pasos solo
  viven en JSON); sin `+C` en antiderivadas; artefactos `ln(e)`,
  `x^(2-1)` en substeps de derivadas anidadas.
- [ ] **(F) Etiquetas legibles en pre-cálculo**: `factor(x^2-9)` narra
  "Factor Polynomial" sin diferencia de cuadrados;
  `expand((x+2)^3)` narra "Evaluate Meta Functions".
- [ ] **(F) Cosmético diff**: `e^(3x^2)/e^(4x^2)` no se combina a
  `e^(-x^2)` en derivadas 2ª/3ª de `exp(-x^2)`; derivadas anidadas de
  orden 4-5 devuelven blobs con `ln(e)`, `x^0`, `x^(2-1)`.

### Fuera del norte actual (clase I — no son ciclos)

Derivación implícita y `diff(f,x,n)`; funciones abstractas `f(x)`;
piecewise en el parser; funciones especiales (erf, Γ/digamma, Si/Ei,
LambertW) como *valores de salida*; assumptions; valor principal;
Risch completo; Gruntz completo; dominio complejo y multivariable
(Deferred Horizons). Si alguno se promueve, exige decisión explícita
de estrategia, no un ciclo de auto-mejora.

### Confirmaciones de honestidad (no tocar)

Los residuales no-elementales correctos deben seguir residuales:
`e^(-x^2)` (indefinida), `sin(x)/x`, `1/ln(x)`, `x^x`, `sin(1/x)` en 0
(no existe), `diff(floor(x))`. Cualquier ciclo que los "resuelva" es
un bug de soundness, no una mejora.
