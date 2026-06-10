# Requires vs Assumptions (y por qué esto hace especial a este engine)

Este motor no solo simplifica “porque sí”: **explica** qué hizo, qué **no** pudo hacer, y—lo más importante—distingue entre:

- **Requisitos (Requires)**: condiciones de dominio **ya implícitas** en la expresión original (no son “decisiones” del motor).
- **Suposiciones (Assumed / Assumptions)**: condiciones **adicionales** que el motor **elige aceptar** para poder aplicar una transformación (según el modo semántico).

Esta separación es clave para un uso **educativo** y también para integrar el motor en apps (FFI) sin perder honestidad matemática.

---

## Contrato Formal (TL;DR)

| Canal | Qué significa | Cuándo aparece |
|-------|---------------|----------------|
| **ℹ️ Requires** | Dominio **implícito** de la entrada + condiciones de **definibilidad** consumidas | `Generic` y `Assume` (canal `required_conditions`) |
| **⚠ Assumed** | Hipótesis **extra** aceptada por política | Solo `Assume` (reglas como `abs`, `log`) |
| **Blocked** | Regla no aplicable en modo actual | Cuando modo lo impide + `hints on` |

> **Invariante**: Requires ≠ Assumed — nunca se mezclan.

---

## Referencia Rápida (verificable en REPL)

```bash
# 1. Requires: dominio implícito
> sqrt(x)^2
ℹ️ Requires: x ≥ 0
Result: x

# 2. Requires: agujero de definibilidad (en Generic)
> x/x
ℹ️ Requires: x ≠ 0
Result: 1
# (assumptions_used queda vacío)

# 3. Definibilidad + dominio implícito del input
> (x-y)/(sqrt(x)-sqrt(y))
ℹ️ Requires: sqrt(x) - sqrt(y) ≠ 0; x ≥ 0; y ≥ 0
Result: √(x) + √(y)
# (assumptions_used queda vacío)
```

---

## 1) Glosario rápido

### Requires (requisitos de dominio)
Son condiciones necesarias para que la expresión **tenga sentido** (en ℝ / `RealOnly`) por su propia estructura.

Ejemplos típicos en `RealOnly`:

- `sqrt(t)` ⇒ **requiere** `t ≥ 0`
- `ln(t)` ⇒ **requiere** `t > 0`
- `1/t` ⇒ **requiere** `t ≠ 0`

Si el motor transforma una expresión y “se pierde” una de estas restricciones, el motor **no debe mentir**:
- o bien bloquea,
- o bien **propaga** ese dominio como `Requires` (y lo muestra explícitamente).

> Idea: “No estoy suponiendo nada: estoy diciendo el dominio de validez que ya estaba ahí.”

---

### Assumed (suposiciones)
Son condiciones que el motor **acepta** para poder aplicar reglas que no son universalmente válidas sin restricciones.

Ejemplos (solo en modo `Assume`, vía `assumptions_used`):
- Expandir `ln(xy)` ⇒ asumir `x > 0` y `y > 0`
- Simplificar `abs(x) → x` ⇒ asumir `x ≥ 0`

Ojo: las cancelaciones de **definibilidad** (`x/x ⇒ x ≠ 0`) y las condiciones Analytic **heredadas** del input (`exp(ln(x)) ⇒ x > 0`) **no** van por este canal: se publican como `Requires` (`required_conditions`) en `Generic` y `Assume`, con `assumptions_used` vacío.

Estas suposiciones **dependen de la política de exploración** (modo semántico).

> Invariante: **No hay suposiciones sin registro** (timeline / resumen).

---

### Blocked (bloqueos)
Cuando una transformación requeriría suponer algo que el modo actual **no permite**, el motor no se calla:
- **no aplica** la regla
- muestra un hint del tipo “requires …; usa `domain assume`”.

---

## 2) Los modos: qué cambia y qué no

Lo importante: **Requires siempre existe** (cuando se puede inferir) y se propaga de forma estable.
Lo que cambia entre modos es **qué suposiciones** se permiten.

| Modo | Requisitos (Requires) | Cancelaciones de definibilidad (≠0) | Analytic **heredadas** del input | Analytic **introducidas** (>0, rangos nuevos) |
|------|------------------------|--------------------------------------|----------------------------------|------------------------------------------------|
| Strict | ✅ Se listan como condición del input | ❌ No | ❌ No | ❌ No |
| Generic *(por defecto)* | ✅ Sí (`required_conditions`) | ✅ Sí (vía Requires) | ✅ Sí (vía Requires) | ❌ No |
| Assume | ✅ Sí (`required_conditions`) | ✅ Sí (vía Requires) | ✅ Sí (vía Requires) | ✅ Sí (`assumptions_used`) |

**Lectura didáctica:**
- **Strict**: “no acepto hipótesis extra” (ej.: conserva `x^0` y lista `x ≠ 0` como condición del input)
- **Generic** (por defecto): “acepto agujeros algebraicos típicos (≠0) y condiciones heredadas del input (ej.: `x^0 → 1` con `Requires: x ≠ 0`), pero **no introduzco** condiciones Analytic nuevas”
- **Assume**: “además acepto hipótesis analíticas nuevas (positividad, rangos, etc.) y las registro en `assumptions_used`”

---

## 3) Ejemplos clave (REPL)

### A) `sqrt(x)^2 → x` en Generic (Requires, no Assumed)

En ℝ, `sqrt(x)` ya implica `x ≥ 0`. Simplificar `sqrt(x)^2` a `x` **no inventa** el dominio: lo **hace explícito**.

```txt
> sqrt(x)^2
ℹ️ Requires:
  - x ≥ 0
Result: x
```

**Interpretación:**

* El motor no dice “asumí x≥0”.
* Dice: “para que esto tenga sentido en ℝ, se requiere x≥0”.
* Es una mejora enorme en UX educativa: el alumno avanza **sin cambiar a Assume**, pero el dominio no se pierde.

---

### B) Caso “dominio implícito + singularidad removible”

[
\frac{x-y}{\sqrt{x}-\sqrt{y}} = \sqrt{x}+\sqrt{y} \quad \text{(con cuidado en } x=y)
]

Aquí ocurre algo sutil y valioso:

* `sqrt(x)` y `sqrt(y)` ya traen `x≥0`, `y≥0` implícitos.
* Al simplificar, el “agujero” de definibilidad es el denominador `sqrt(x)-sqrt(y)`.

En `Generic` el motor simplifica y publica **todo** por el canal `Requires` (`required_conditions`):

```txt
> (x - y) / (sqrt(x) - sqrt(y))
ℹ️ Requires:
  - sqrt(x) - sqrt(y) ≠ 0
  - x ≥ 0
  - y ≥ 0
Result: √(x) + √(y)
```

Fíjate en lo importante:

* **Nada** de esto aparece como suposición: `assumptions_used` queda **vacío**.
* `x ≥ 0, y ≥ 0` eran dominio **implícito** del input, y la cancelación del denominador es un agujero de **definibilidad**: ambos viajan como `Requires`, tanto en `Generic` como en `Assume`.

---

### C) `exp(ln(x)) → x` en Generic: condición Analytic **heredada** (Requires)

En ℝ, `ln(x)` ya requiere `x>0`: la condición viene del **testigo `ln(x)` del propio input**, el motor no la introduce. Por eso, igual que con `sqrt(x)^2`, la reescritura **sí se aplica** en `Generic` (y en `Assume`):

```txt
> exp(ln(x))
ℹ️ Requires:
  - x > 0
Result: x
```

Distinto es el caso de una condición Analytic **introducida**, como expandir `ln(x*y)`: ahí `x > 0` y `y > 0` **no** estaban implícitas en el input (basta `x*y > 0`). En `Generic` esa reescritura queda bloqueada:

```txt
> ln(x*y)
Result: ln(x*y)
ℹ️ Blocked simplifications:
  - requires x > 0, y > 0 [Logarithm Product Expansion]
  tip: use `domain assume` to allow
```

La diferencia clave:

* `exp(ln(x)) → x`: la condición `x > 0` se **hereda** del witness `ln(x)` del input ⇒ se publica como `Requires` también en `Generic`.
* `ln(x*y) → ln(x) + ln(y)`: la condición se **introduce** (cambia el significado fuera de `x>0, y>0`) ⇒ en `Generic` no se asume.

---

### D) El mismo ejemplo en Assume: se resuelve y queda registrado

```txt
> semantics set domain assume
> ln(x*y)
Result: ln(x) + ln(y)
ℹ️ Assumptions used:
  - x > 0
  - y > 0
```

Y en timeline, el paso queda como:

* `Assumed: x > 0, y > 0`

---

### E) Cancelación algebraica típica: `x/x → 1`

* En **Strict**: se bloquea (no se aceptan agujeros).
* En **Generic** (y **Assume**): se permite con `Requires: x ≠ 0` (definability hole); `assumptions_used` queda vacío.

```txt
> x/x
Result: x/x
ℹ️ Blocked:
  - requires x ≠ 0 [Cancel Common Factors]
  tip: use `domain generic` to allow definability cancellations

> semantics set domain generic
> x/x
Result: 1
ℹ️ Requires:
  - x ≠ 0
```

Esto es exactamente el “contrato complementario”:

* Generic simplifica fracciones y racionales de forma práctica
* Strict permanece “sin hipótesis extra”

---

## 4) Por qué esto es un multiplicador educativo

Sin esta separación, un motor suele caer en uno de los dos extremos:

1. **CAS clásico**: simplifica agresivo y el alumno no aprende qué condiciones se usaron.
2. **Sistema pedante**: bloquea muchísimo y el alumno no avanza.

Este engine busca una tercera vía:

* **Avanzar**, pero sin mentir:

  * si el dominio estaba implícito: **Requires**
  * si se aceptó una hipótesis extra: **Assumed**
  * si el modo actual no lo permite: **Blocked + tip**

El resultado es un “profesor automático”:

* muestra dominio cuando importa,
* registra suposiciones cuando se toman,
* y guía al usuario cuando está bloqueado.

---

## 4.1) Propagación en reutilización de expresiones

Un aspecto clave del sistema es que los **Requires se propagan automáticamente** cuando reutilizas expresiones de la sesión mediante `#id`.

### Comportamiento

Cuando evalúas `#1 + 4` (donde `#1` era `sqrt(x)`):

1. El motor **resuelve** `#1` → `sqrt(x)`
2. Construye la expresión compuesta: `sqrt(x) + 4`
3. **Re-infiere** el dominio estructural de la expresión resuelta
4. Muestra `ℹ️ Requires: x ≥ 0`

### Ejemplo en REPL

```bash
> sqrt(x)
#1: √(x)
ℹ️ Requires:
  - x ≥ 0

> #1 + 4
#2: 4 + √(x)
ℹ️ Requires:
  - x ≥ 0    ← SE PROPAGA

> show #2
Entry #2:
  Parsed:     4 + #1
  Resolved:   4 + sqrt(x)
  ℹ️ Requires:
    - x ≥ 0  ← VISIBLE EN SHOW
```

### Expresiones combinadas

Cuando combinas sub-expresiones con diferentes restricciones, **todas se acumulan**:

```bash
> sqrt(x) + ln(y)
ℹ️ Requires:
  - x ≥ 0    ← de sqrt(x)
  - y > 0    ← de ln(y)
```

### Por qué es importante

| Sin propagación | Con propagación |
|-----------------|-----------------|
| El alumno "pierde" el dominio al reutilizar | El dominio viaja con la expresión |
| Fácil cometer errores silenciosos | Transparencia total |
| El motor "olvida" contexto | La sesión es coherente |

> **Invariante**: Los Requires se infieren de la **estructura** de la expresión resuelta, no se "almacenan" con el ID. Esto significa que siempre reflejan el estado actual.

---

## 5) Qué puedes pedirle al motor (comandos útiles)

* Cambiar política:

  * `semantics set domain strict|generic|assume`
* Mostrar/ocultar hints:

  * `semantics set hints on|off`
* Resumen de suposiciones:

  * `semantics set assumptions off|summary`
* **Nivel de Requires** (nuevo en V1.3.8):

  * `semantics set requires essential|all`
  * `essential`: solo muestra Requires cuyo testigo fue consumido (ej: `sqrt(x)² → x`)
  * `all`: muestra todos los Requires, incluso si el testigo sobrevive
* Modo explicación:

  * `explain on|off`

---

## 6) Contratos e invariantes (para confianza)

* **No assumptions without record**:
  toda hipótesis aceptada aparece en timeline / resumen.
* **Requires no se mezcla con Assumed**:
  requiere = dominio implícito; assumed = política.
* **Rule-agnostic safety** (airbag):
  el motor detecta reescrituras que “pierden” restricciones implícitas y:

  * o las conserva mediante *witness survival*,
  * o las vuelve explícitas como `Requires`.

---

## 7) Ejercicios propuestos (para aula)

1. Simplifica y explica:

   * `sqrt(x)^2`
   * `1/ln(x)` (¿qué requiere?)
   * `(x^2-1)/(x-1)` (¿qué asume?)
2. Compara modos:

   * `ln(x*y)` en Strict/Generic/Assume
3. Detecta singularidades removibles:

   * `(x-y)/(sqrt(x)-sqrt(y))`
   * `(x^2-1)/(x-1)`

---

## Conclusión

La combinación de:

* **Step-by-step**
* **Requires vs Assumed**
* **Blocked hints guiados**
  hace que el motor sea útil *a la vez* para:
* aprender (dominio + condiciones),
* trabajar (simplificación potente en Generic),
* y mantener rigor en Generic y trazabilidad en Assume.
* modo Strict es el modo pendántico, que no permite suposiciones extra, ni tampoco presupone que las condiciones implícitas son ciertas.