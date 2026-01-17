La idea: **cada regla declara su “soundness class”** y el engine decide si:

* **aplica sin condiciones**
* **aplica introduciendo requires**
* **bloquea**
* **solo en assume**
* **solo en strict**
* **skip simbólico y deja solo numérico** (para branch-sensitive en metatest / equiv)

---

## Tabla 1 — Política por modo (DomainMode)

| SoundnessLabel / Clase                 | Qué significa                                                                            |                                                             Generic |                                                                                           Assume |                                                                                               Strict |
| -------------------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------: | -----------------------------------------------------------------------------------------------: | ---------------------------------------------------------------------------------------------------: |
| **UnconditionalEquivalence**           | Equivalencia algebraica “siempre” válida (en el ValueDomain actual)                      |                                                            ✅ aplica |                                                                                         ✅ aplica |                                                                                             ✅ aplica |
| **EquivalenceUnderInheritedRequires**  | Válida si **ya** tienes requires suficientes (heredados/implícitos)                      |                            ✅ aplica **solo si** `requires ⊇ needed` | ✅ aplica **solo si** `requires ⊇ needed` (o las puede heredar de implicit_domain si lo permites) |                                                             ✅ aplica **solo si** `requires ⊇ needed` |
| **EquivalenceUnderIntroducedRequires** | Válida si introduces condiciones (ej. `x>0`, `cos(x)≠0`, `base≠1`)                       |                                    ✅ aplica **y emite** `Requires:` |                                ✅ aplica (puede emitir además `Assumes:` si hay heurística extra) | ✅ aplica **solo si** esos requires se pueden justificar o ya estaban (según tu definición de Strict) |
| **HeuristicAssumption**                | Correcta “típicamente”, pero no garantizable sin asumir (paridad simbólica, “x≥0”, etc.) |                                        ⛔ bloquea (o deja “Blocked”) |                                            ✅ aplica con ⚠️ `Assumes:` (+ `Requires:` si procede) |                                                                                            ⛔ bloquea |
| **BranchSensitivePrincipal**           | Depende de rama principal / discontinuidades (log complejo, atan, etc.)                  | ⛔ bloquea (o ✅ solo si `ValueDomain=RealOnly` y con guards fuertes) |                          ✅ aplica si `RealOnly` y condiciones; en `ComplexEnabled` normalmente ⛔ |                                                            ⛔ bloquea (o requiere verificación extra) |
| **NormalizationOnly**                  | Reescritura de forma normal (conmutatividad/orden) sin cambiar significado               |                                                            ✅ aplica |                                                                                         ✅ aplica |                                                                                             ✅ aplica |

**Nota clave:** si quieres que `exp(ln(x))` en RealOnly se comporte como `sqrt(x)^2`, entonces su regla debe ser `EquivalenceUnderIntroducedRequires` (con `x>0`) y **no** `HeuristicAssumption/BranchSensitive`.

---

## Tabla 2 — Gate adicional por ValueDomain (RealOnly vs ComplexEnabled)

| Regla / Identidad típica       |                                  RealOnly |                             ComplexEnabled |                      |                                              |
| ------------------------------ | ----------------------------------------: | -----------------------------------------: | -------------------- | -------------------------------------------- |
| `sqrt(u^2) =                   |                                         u |                                          ` | ✅ (con su semántica) | ⛔ o cambia a `sqrt(u^2)=±u` no representable |
| `(x^n)^(1/n)` cancelaciones    |                  ✅ con paridad + requires |                      ⛔ normalmente (ramas) |                      |                                              |
| `log(b, b^y)=y`                | ✅ con `b>0, b≠1` (+ si variable, cuidado) |                         ⛔ (rama principal) |                      |                                              |
| `exp(ln(x))=x`                 |             ✅ con `x>0` (si `ln` es real) |                ⛔ en general (log complejo) |                      |                                              |
| identidades `atan` con módulos |                 ✅ (pero branch-sensitive) | ⛔ o requiere comparación mod (2π) con rama |                      |                                              |

---

## Tabla 3 — Qué hacer con Requires en cada modo

| Acción                                                      |                                                          Generic |           Assume |                                                Strict |
| ----------------------------------------------------------- | ---------------------------------------------------------------: | ---------------: | ----------------------------------------------------: |
| Introducir `Requires:` (condiciones matemáticas explícitas) |                          ✅ solo para reglas “IntroducedRequires” |                ✅ | ✅ pero idealmente solo si demostrables o ya heredadas |
| Introducir `Assumes:` (heurísticas, “suponemos x≥0”)        |                                                                ⛔ |                ✅ |                                                     ⛔ |
| Bloquear con “Blocked: requires …”                          | ✅ cuando la regla es heurística/branch o no permitida en generic | (normalmente no) |                                                     ✅ |

Esto explica tu caso:

* `sqrt(x)^2 → x` está permitido en generic como **IntroducedRequires** (y tu `implicit_domain` además lo infiere).
* `exp(ln(x)) → x` está clasificada como **no permitida en generic**, así que se marca **Blocked** en lugar de emitir `Requires: x>0`.

---

## Recomendación concreta para tu engine (para que sea consistente)

1. Define **un mapping único**: `SoundnessLabel → AllowedIn(Generic/Assume/Strict)`.
2. Define un flag por regla:

   * `introduces_requires: bool`
   * `introduces_assumptions: bool`
   * `branch_sensitive: bool`
3. Política sugerida:

   * **Generic**: permite `introduces_requires`, prohibe `introduces_assumptions` y prohibe `branch_sensitive` (salvo whitelist en RealOnly).
   * **Assume**: permite todo, pero marca ⚠️.
   * **Strict**: permite solo `Unconditional` o `UnderInheritedRequires` (o `IntroducedRequires` solo si la condición ya está probada/heredada).

