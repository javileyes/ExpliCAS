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
- [ ] **(A) Motor 0/0 componible en punto finito**: la allowlist no
  invierte (`x/sin(x)` falla siendo `sin(x)/x` soportado), no compone
  (`sin(3x)/sin(5x)` → 3/5, `(1-cos x)/x²` → 1/2, `(sin x - x)/x³` →
  −1/6, `asin(x)/x`, `sinh(x)/x`), no encadena L'Hôpital/Taylor. El
  item de mayor frecuencia en cualquier curso.
- [ ] **(A) Formas exponenciales 1^∞/0^0/∞^0** vía `exp(lim g·ln f)`:
  `(1+1/x)^x → e`, `(1+2/x)^x → e²`, `(1+x)^(1/x) → e`, `x^x → 1 en
  0+`, `(2^x+3^x)^(1/x) → 3`. Hoy la constante `e` es inalcanzable
  por límite — invalida un capítulo del temario.
- [ ] **(F) ∞−∞ con radicales** (racionalización por conjugado):
  `sqrt(x^2+x)-x → 1/2`, `sqrt(x+1)-sqrt(x) → 0`, y en punto finito
  `(sqrt(x)-2)/(x-4) → 1/4`.

### P2 — familias y mejoras de alto valor (1 ciclo cada una)

- [ ] **(F) Touch con límite x^a·ln(x)^b → 0**: `ln(x)^2 [0,1]` (=2),
  `x*ln(x) [0,1]` (=−1/4), `ln(x)/sqrt(x) [0,1]` (=−4) residuales con
  antiderivadas elementales; la dominancia potencia-log existe en el
  lado lateral pero no cubre estas combinaciones. Arregla 3+ familias.
- [ ] **(F) Gaussiana/Gamma por tabla**: `e^(-x^2) [0,∞) = √π/2`,
  `(-∞,∞) = √π`, `x^2*e^(-x^2) [0,∞) = √π/4`, `e^(-x)/sqrt(x) = √π` —
  la impropia más famosa de la universidad; tabla pequeña de formas
  patrón (la indefinida debe SEGUIR residual).
- [ ] **(A) Pre-simplificador vs integrador**: reescribe
  `1/(sqrt(x)*(1+x))` a `(x^(3/2)-x^(1/2))/(x^3-x)` y `cos(5x)` a
  Chebyshev en cos(x), destruyendo la sustitución obvia
  (`[0,∞) = π`) y la ortogonalidad de Fourier `sin(3x)cos(5x)
  [-π,π] = 0`. Integrar sobre la forma original primero, o enseñar al
  integrador las formas reescritas (precedente: reconocedor Chebyshev
  del ledger 2026-06-12). Ejemplo vivo adicional:
  `integrate(sec(x)^2 + 1, x)` se vuelve residual porque el
  pre-simplificador lo machaca a `(2cos²−1+3)/(2cos²)`.
- [ ] **(F) Detección estructural sin antiderivada**: imparidad en
  `[-a,a]` para integrandos no elementales (`sin(x)/(1+x^2) [-1,1] =
  0`), abs por tramos (`|x| [-1,1] = 1`, `e^(-|x|) (-∞,∞) = 2`),
  test-p completo (`1/sqrt(x) [1,∞) = ∞`, hoy residual mientras `1/x`
  sí diverge), divergencia oscilatoria declarada (`sin(x) [0,∞)`).
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
- [ ] **(F) Composición de límites con interno conocido**:
  `e^(1/x) en 0±` (→ ∞ / 0), `atan(1/x) en 0+` (→ π/2) fallan aunque
  `1/x → ±∞` resuelve; regla de composición continua/monótona barata
  (la tabla saturante en ∞ ya existe — reutilizarla desde laterales).
- [ ] **(F) Squeeze y dominancia fraccionaria**: `x*sin(1/x) → 0 en
  0`, `(x+sin x)/x → 1 en ∞`, `ln(x)/sqrt(x) → 0 en ∞` (la dominancia
  entera `ln(x)/x` sí funciona).
- [ ] **(F) Producto-a-suma residual mutilado**: `sin(3x)cos(5x)`
  indefinida queda residual Y mutilada (expandida en potencias de
  cos); el reconocedor producto-a-suma cubre frecuencias distintas
  con a≠±b — revisar por qué esta combinación escapa.
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
- [ ] **(F) sech/csch no parsean**: `sech(x)^2` da "función no
  definida".
- [ ] **(F) Bounds con e**: `integrate(1/x, x, 1, e)` residual
  (`ln(e)` no se evalúa); `tan(x) [0,π/4]` devuelve
  `ln(|cos(0)/cos(π/4)|)` sin plegar `cos(0)=1`.
- [ ] **(F) FTC/Leibniz en diff**: `diff(integrate(f(t),t,0,x), x)`
  → `f(x)` (+ regla de Leibniz con límites variables).
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
