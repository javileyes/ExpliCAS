# Didactic SubStep Normalization

This document defines the authoring contract for user-facing didactic
substeps.

It exists to keep `simplify`, `derive`, CLI, JSON and timeline rendering
aligned on the same rule:

- a substep may have a generic explanatory title
- but its math must always be specific

If a new substep family does not satisfy that rule, it should not be emitted.

## Scope

This contract applies to user-facing didactic substeps built from
[`SubStep`](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/types/substep.rs).

It does **not** apply to:

- main timeline steps
- internal chained rewrites
- legacy free-form engine substeps

Legacy narrative engine substeps are intentionally hidden from user-facing
surfaces. The supported public format is the structured didactic `SubStep`.

## Canonical Shape

Every user-facing substep must be readable as:

1. explanatory title
2. concrete expression before
3. `->`
4. concrete expression after

So the visual shape is always:

```text
[title]
[specific expression]
->
[specific expression]
```

The title may name a general rule.

The math may **not** stay generic if the step already knows the concrete local
expression being rewritten.

## Data Contract

Current `SubStep` shape:

```rust
pub struct SubStep {
    pub description: String,
    pub before_expr: String,
    pub after_expr: String,
    pub before_latex: Option<String>,
    pub after_latex: Option<String>,
}
```

Operational meaning:

- `description` is the explanatory header
- `before_expr` and `after_expr` are the concrete mathematical sides
- LaTeX is only a rendering override for the same concrete math

The structured substep payload should therefore always express a specific local
rewrite, never a free-form narrative sentence in the math lines.

## Authoring Rules

### 1. Prefer concrete local math

If the parent step exposes a local `before -> after`, the substep should reuse
that concrete rewrite whenever possible.

Good:

```text
Expandir logaritmo de un producto
ln(x^3·y^2)
->
ln(x^3) + ln(y^2)
```

Bad:

```text
Expandir logaritmo de un producto
ln(a·b)
->
ln(a) + ln(b)
```

when the concrete local expression is already known.

### 2. Title can be generic, body cannot

A general title is fine:

- `Aplicar identidad pitagórica`
- `Llevar a denominador común`
- `Usar a·sin(u) + b·cos(u) = R·sin(u + φ)`

But the mathematical lines below that title must still show the actual
expression being transformed in that step.

### 3. One visible rewrite per substep

Each substep should show one concrete transformation that a human can inspect.

If a derivation needs multiple meaningful hidden moves, emit multiple substeps.

Do not compress several unrelated ideas into one synthetic `before -> after`
unless that is the only coherent local view available.

### 4. No narrative math lines

The expression lines are reserved for math.

Do not put prose there.

Bad:

```text
Expandir y reagrupar
Tomamos denominador común
->
Simplificamos el numerador
```

Good:

```text
Llevar a denominador común
1/(x-1) - 1/(x+1)
->
(x+1-(x-1))/((x-1)(x+1))
```

### 5. No template placeholders when a real expression exists

Avoid placeholder math such as:

- `a`
- `b`
- `u`
- `n`
- `a^2 - b^2`

unless the entire parent step is itself genuinely abstract and there is no
concrete local rewrite available.

For normal user-facing `simplify` and `derive` traces, concrete math should be
the default.

### 6. No duplication of the parent step

If the only possible substep would restate the parent step exactly, emit no
substep.

This is the preferred outcome for many direct identities and obvious local
rewrites.

Examples where `0` substeps is usually better:

- `sin(x)^2 + cos(x)^2 -> 1`
- direct reciprocal trig rewrites when the main step already shows the local
  focus clearly
- direct binomial expansions where the parent step already exposes the exact
  rewritten block

### 7. No purely decorative substeps

Reject a substep if it only:

- paraphrases the parent title
- says “use the identity” without new visible math
- renames the maneuver without clarifying the algebra
- exists only to make the UI look denser

### 8. Prefer no substep over a bad substep

The quality ordering is:

1. concrete useful substep
2. no substep
3. generic or duplicate substep

So “missing” is often better than “present but fake”.

## Intermediate Expressions

Some families need a synthetic intermediate expression to make the jump
understandable.

That is allowed, but the intermediate must still be specific.

Good:

```text
Sacar factor común
4·cosh(x)^3 - 4·cosh(x)
->
4·cosh(x)·(cosh(x)^2 - 1)
```

Then:

```text
Usar cosh(u)^2 - 1 = sinh(u)^2
4·cosh(x)·(cosh(x)^2 - 1)
->
4·cosh(x)·sinh(x)^2
```

Bad:

```text
Aplicar identidad hiperbólica
k·f(u)
->
k·g(u)
```

when the actual local expression is known.

## Authoring Decision Tree

When adding a new didactic substep family:

1. Can the parent step already stand on its own?
   - If yes, emit `0` substeps.
2. Is there a concrete local rewrite hidden inside the step?
   - If yes, emit that concrete `before -> after`.
3. Is a synthetic intermediate needed?
   - If yes, emit one or more specific intermediate substeps.
4. Are you about to emit template math or restate the parent step?
   - If yes, stop and emit no substep.

## Surface-Level Guarantees

CLI, web and JSON should all present the same normalized structure:

- title/header explains the maneuver
- math sides remain specific
- no mixed prose inside expression lines

This is why user-facing renderers now standardize on structured enriched
substeps and do not render legacy narrative engine substeps.

Relevant implementation touchpoints:

- [`SubStep`](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/types/substep.rs)
- [`SubStep` methods](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/types/substep/methods.rs)
- [step payload substep rendering](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/step_payload_render/substeps.rs)
- [focused rule substeps](/Users/javiergimenezmoya/developer/math/crates/cas_didactic/src/didactic/focused_rule_substeps.rs)

## Review Checklist

When reviewing a new substep family, ask:

- Does the title explain the maneuver clearly?
- Are both math sides concrete and specific to the current step?
- Would a human learn something new from this substep?
- Is the substep different from the parent step?
- Would removing the substep make the step feel magical?

If the answer to the last question is `no`, the correct action is often to drop
the substep.
