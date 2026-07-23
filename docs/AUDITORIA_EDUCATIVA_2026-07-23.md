# Auditoría educativa de steps y highlights — 2026-07-23

**Alcance:** las 210 expresiones de `web/examples.csv`, evaluadas con `--steps on --format json --lang es` (la configuración que sirve la web). 207 evaluadas (3 dependen de sesión: filas 131/133/135 usan definiciones de filas previas — funcionan en la web con sesión; limitación del runner stateless, no bug). 1056 steps analizados con detectores sistemáticos (discontinuidad de cadena, noops, duplicados, ráfagas, artefactos, mismatch de highlights por comparación latex-a-latex con extracción brace-balanced, fugas de idioma) y **verificación a mano de cada familia** antes de reportarla (los detectores v1 tuvieron falsos positivos de normalización — descartados).

**Este documento es un INFORME: no se ha tocado código.** Los logs completos del audit: JSONs por ejemplo + `_problems2.json` (regenerables con el script del ledger de esta fecha).

---

## P1 — Rompen la confianza del alumno

### 1. Cadenas discontinuas — "saltos mágicos" reales (47 expresiones, 506 hits)

El `after` de un paso NO es el `before` del siguiente. Tres sub-causas verificadas:

**(a) Estado crudo vs plegado** — el recolector captura estados en momentos distintos del pipeline:
```
[006] diff(x^2*y, x, y)
  step1 after:  ∂/∂y(2·x·y)              ← plegado
  step2 before: ∂/∂y(2·y·x^(2-1))        ← ¡el crudo reaparece!
[004] diff(arctan(sqrt(x)), x)
  step1 after:  1/(2·√x·(x+1))           ← YA es el resultado final
  step2 before: (1/(2·√x))/(x+1)         ← forma que el alumno nunca vio
  step2 red:    1/((2x+2)·√x)            ← una TERCERA forma
```
step2 de [004] es además redundante ("Presentar en forma compacta" de algo ya compacto).

**(b) Fases concatenadas sin agrupar** (solve): el preview de simplificación y la cadena del solver se aplanan juntos:
```
[056] solve(e^x+e^(-x)=4,x)
  step1 after:  2·cosh(x)                ← simplify plegó a cosh
  step2 before: e^x + e^(-x) - 4         ← el solver arranca del crudo
```

**(c) Fragmentos de candidatos intercalados** (solve trig): evaluaciones independientes de cada raíz aplanadas sin contexto:
```
[050] solve(sin(x)=1/2,x)
  step0 after:  -1/2·π                   ← evaluando un candidato
  step1 before: 2 - (1 + 1)              ← ¿de dónde sale esto?
```
Afecta a TODO el grupo de trig equations (050-054...) e inecuaciones con casos.

### 2. Highlights medio erróneos (≈80 expresiones con mismatch verde, 66 con rojo)

El rojo/verde del `rule_latex` muestra una forma DISTINTA de la que la cadena muestra como before/after — normalmente la cruda o una reordenada:
```
[006] step1: after texto = 2·x·y        pero green = y·2·x^{2-1}
[052] step0: after texto = 3/(2·√3)     pero green = (3/2)·3^{-1/2}
```
Es exactamente el "medio erróneos" reportado por el usuario: el color no miente del todo (es equivalente), pero no es lo que el alumno ve antes/después. Misma raíz que 1(a): el highlight se genera en el momento crudo del rewrite; el estado mostrado, tras el fold.

### 3. Pasos DUPLICADOS consecutivos (10 expresiones, 71 hits)

Idénticos en regla+before+after, mostrados dos o más veces:
```
[052] solve(2·cos(x)−√3=0): step0 y step1 son EL MISMO paso (Cancel Power Fraction)
[056] solve(e^x+e^(-x)=4): ~30 duplicados seguidos de "Quitar paréntesis…"
```

---

## P2 — Ruido y pulido

### 4. Artefactos de maquinaria visibles (13 expresiones)

`x^(2-1)`, `x^(2-1-1)` sin plegar en states/highlights. **Afecta a TODO el grupo vectorial**: gradient/hessian/jacobian/divergence/laplacian muestran los exponentes crudos en el paso 0:
```
[147] gradient(x^2·y,[x,y]) → paso muestra [[2·y·x^(2-1)], …]
[144] integrate(diff(x^3,x),x): "int 3·x^(3-1) dx"
```

### 5. Ráfagas de la misma regla (6 expresiones)

"Quitar paréntesis tras el signo menos" ×34 [056], "Combinar las constantes" ×8 [068][070], "Agrupar términos semejantes" ×10 [074], "Abs Sub Normalize" ×8 [079]. Deberían colapsarse en un paso agregado ("×n aplicaciones").

### 6. Pasos no-op (7 expresiones)

"Calcular potencia numérica" con before=after='1' ([056][060][064][069][077]); "Cancelar términos opuestos" idéntico en [186].

### 7. Nombres de regla EN INGLÉS (5 reglas, 23 apariciones)

`Cancel Power Fraction` (5), `Abs Sub Normalize` (8), `Normalize Negation in Product` (8), `Absorb Negation Into Difference` (1), `Abs Of Exp` (1) — sin entrada en la tabla es/en.

### 8. Narrativa anti-educativa puntual

El ejercicio MÁS clásico del curso contado del revés:
```
[052] solve(2·cos(x)−√3=0):
  step0: √3/2 → 3/(2·√3)      ← DES-simplifica el valor canónico (y duplicado)
  step2: arccos(3/(2·√3)) → π/6  ← evalúa desde la forma fea
```
El alumno esperaría: `cos x = √3/2 → x = π/6`. El truco interno de racionalización no debería narrarse. También cosméticos como paso propio: `−π/6 → −(1/6)·π` [052 step3].

---

## P3 — Huecos y UX

### 9. Inecuaciones con resultado y CERO narración (3)

`solve(1/(x−√2)>0)`, `solve(x^(2/3)>2)`, `solve(|x−1|<|x+2|)` → intervalo correcto, 0 steps y 0 solve_steps.

### 10. UX de ejemplos

- **Grupo Complejo**: con el selector COMPLEJO=off (default), los 8+ ejemplos devuelven el input sin evaluar (`(1+i)^2 → (1+i)^2`). Con `--value-domain complex` funcionan perfectos (`→ 2i`, 1-3 steps). Los ejemplos no llevan/activan su modo requerido.
- **Filas dependientes de sesión** (131/133/135): correctas en web; se rompen si se copian sueltas.

### Positivo verificado

Los `solve_steps` de las familias nuevas (dsolve O0-O9, sistemas S1-S7) están limpios: narración con contenido real, sin fugas de idioma, sin discontinuidades ([207] Bernoulli narra identificación→sustitución→lineal→verificación impecable). El límite multivariable narra el squeeze por warnings con la cota citada [200]. Los detectores no encontraron braces desbalanceados ni fugas es/en en descripciones.

---

## Hipótesis de causas raíz (transversales)

1. **A+E+D comparten raíz**: el recolector de steps captura el ESTADO post-fold pero el HIGHLIGHT del momento crudo del rewrite (o viceversa según la fase) — un chokepoint en el recolector/presentador, no 80 bugs sueltos.
2. **A(b)+A(c)**: el aplanado de sub-cadenas independientes (preview, candidatos por raíz) sin marcador de grupo — problema de agregación, no de reglas.
3. **K**: falta dedup consecutivo en el recolector.
4. **G2**: 5 nombres sin fila en la tabla de localización (mecánico).

## Priorización sugerida (ciclos futuros, NO ejecutados)

1. **[M] Chokepoint del recolector**: estado y highlight capturados en el MISMO momento (post-fold) + dedup consecutivo → ataca familias 1(a), 2, 3 y parte de 4 de una vez.
2. **[S] Fold de display en verbos vectoriales** (los `x^(2-1)` del grupo entero).
3. **[S] 5 nombres de regla a la tabla es/en.**
4. **[S/M] Colapso de ráfagas** (×n) y filtro de noops en presentación.
5. **[M] Agrupación de sub-cadenas** (candidatos/fases) con encabezado de contexto.
6. **[S] Ejemplos con modo requerido** (metadato en examples.csv + la web activa el selector).
7. **[M] Narración mínima de inecuaciones** (las 3 con cero steps).
