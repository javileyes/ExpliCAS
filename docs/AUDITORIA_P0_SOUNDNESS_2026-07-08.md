# Auditoría P0 de soundness — 2026-07-08

Workflow multi-agente `frontier-audit-cycle4` (8 scouts read-only, uno por frente, ~35 probes c/u vía CLI + oráculo sympy + verificación por sustitución → verificación adversarial 2-lentes por hallazgo → síntesis rankeada por ROI). Lanzado durante `/auto-mejora 8` (ciclo 4).

**20 defectos CONFIRMADOS** (0 falsos positivos tras refutación). Agrupados por causa-raíz.

## Estado de cierre (vivo)

- [x] **Familia #1 — recíproco-de-abs vs 0** (`1/(|x|−1)<0`, 5 P0): CERRADA ciclo 4, commit `5f255c3e2`.
- [x] **Familia recíproco-trig ecuación** (`2/sin(x)=4`, `1/sin(x)=2`, ~12 inputs P0): CERRADA ciclo 5.
- [x] **diferencia-de-radicales ecuación** RAÍZ RACIONAL (`√(5x−1)−√(x+2)=1` → {2}): CERRADA ciclo 6. Peldaño: raíz irracional (`√(3x−1)−√(x+2)=1`) sigue leak.
- [x] **single-radical `√(quad)=poly` directo** (`√(5x²+9x−2)=3x` → {1/4,2}, era "No solution"; `√(5x²+9x)=3x` dropeaba 9/4): CERRADA batch-2 ciclo 1 (handler reduce-a-polinomio + verify g(r)≥0, scope racional). Peldaño: raíces SURD (declina → isolación).
- [~] FTC definido-desde-antiderivada: `∫1/(e^x+c)` (c>0), `∫e^x/(e^x+1)` CERRADO batch-3 ciclo 1 (certificado de polo consulta `prove_positive` para denominador transcendental positivo-everywhere). PENDIENTE: `∫1/(1+cos x)` [0,π/2] (raíz FUERA del intervalo → interval-específico); `∫1/(cosh x+1)` (falta la antiderivada, no el certificado).
- [~] sign-via-abs: `−|x|/x=1` / `c·|x|/x` (numerador abs con coef/negación, constant-RHS) CERRADO batch-2 ciclo 3 (pelar coef en el brazo Div de `sign_form_coeff`). PENDIENTE: RHS-variable `x/|x|=x`, `x/|x|=−x` (leak — necesita excluir polo x=0 + ruteo a `x·|x|`).
- [x] **apart de numerador monomio `c·x^k`** (`apart(2x/((x−1)²(x+1)))`): CERRADA batch-2 ciclo 2 (normalizar `Mul(c, Div)` → `num/den` antes del match Div).
- [x] **`∫1/x^p` p fraccionaria** (`∫1/x^(1/3)` → `3/2·x^(2/3)`, definido `[1,8]` → `9/2`): CERRADA batch-2 ciclo 4 (normalizar `(c·)x^a/x^b → c·x^(a-b)` en la entrada de IntegrateRule, indefinido + definido, gateado a exponente fraccionario).
- [ ] dos-sqrt INECUACIÓN dropea operador (`√x+√(x−1)>1`, multi-cycle).

## Hallazgos confirmados (por frente)

### solve-poly-rational-radical

- **[P0]** `solve(sqrt(5*x-1) - sqrt(x+2) = 1, x)`
  - got: `No solution`
  - correcto: `{ 2 }`
  - causa-raíz: The isolate-square-square path for sqrt(A)-sqrt(B)=c (difference of two radicals, nonzero constant RHS) drops the valid rational root: the sibling forms sqrt(A)+sqrt(B)=c and sqrt(A)-sqrt(B)=0 both work, but the minus-with-nonzero-RHS branch mishandles the extraneous-root filter and returns empty instead of {2}.
  - tractabilidad: bounded-single-cycle
- **[P1]** `solve(sqrt(3*x-1) - sqrt(x+2) = 1, x)`
  - got: `solve(x - (4 / 3 + x / 3 + 2·(x + 2)^(1/2) / 3) = 0, x)`
  - correcto: `{ 5/2 + sqrt(17)/2 }  (approx x=4.5616)`
  - causa-raíz: Same difference-of-radicals path: when the leading-radical coefficient does not reduce to the special a=5 case, the pipeline leaks a transformed self-referential residual instead of solving the remaining single-radical equation (which has an irrational root the finder never reaches).
  - tractabilidad: bounded-single-cycle
- **[P1]** `solve(sqrt(5*x) - sqrt(x+2) = 1, x)`
  - got: `solve(x - ((x + 2)^(1/2) + 1)^(1 / 1/2) / 5 = 0, x)`
  - correcto: `{ 7/8 + 3*sqrt(5)/8 }  (approx x=1.7135)`
  - causa-raíz: Difference-of-radicals residual leak plus a malformed exponent print: '1 / 1/2' is an un-normalized reciprocal-of-reciprocal (should fold to 2) left in the residual tree by the square-both-sides step.
  - tractabilidad: multi-cycle

### solve-transcendental

- **[P0]** `solve(2/sin(x) = 4, x)`
  - got: `{ 1/6·pi }`
  - correcto: `{ pi/6 + 2k·pi, 5pi/6 + 2k·pi : k ∈ ℤ } — reduces to sin(x)=1/2`
  - causa-raíz: The c/trig(x)=k reciprocal reduction (c≠1) solves the boundary equation sin(x)=c/k but reports only the principal arcsin/arccos value, dropping the second branch and the +2kπ periodic family.
  - tractabilidad: bounded-single-cycle
- **[P0]** `solve(5/cos(x) = 10, x)`
  - got: `{ 1/3·pi }`
  - correcto: `{ pi/3 + 2k·pi, 5pi/3 + 2k·pi : k ∈ ℤ } — reduces to cos(x)=1/2`
  - causa-raíz: Same reciprocal-trig reduction bug as 2/sin(x)=4: only the principal value survives; second branch and periodicity are dropped.
  - tractabilidad: bounded-single-cycle
- **[P0]** `solve(1/sin(x) = 2, x)`
  - got: `solve(csc(x) = 2, x)`
  - correcto: `{ pi/6 + 2k·pi, 5pi/6 + 2k·pi : k ∈ ℤ }`
  - causa-raíz: When the reciprocal-trig numerator is exactly 1 (1/sin=csc etc.), the rewrite to a cofunction produces a residual solve(csc(x)=c) that is emitted un-dispatched, even though the direct csc/sec/cot solver path works.
  - tractabilidad: bounded-single-cycle

### solve-abs-sign

- **[P0]** `solve(x/abs(x) = x, x)`
  - got: `solve(x - x·|x| = 0, x)`
  - correcto: `{ -1, 1 }  (x/|x|=sign(x); sign(x)=x ⇒ x=1 for x>0, x=-1 for x<0; x=0 excluded since 0/|0| undefined)`
  - causa-raíz: The u/|u| = variable-RHS branch clears the denominator by multiplying by |u|, producing x - x·|x| = 0, then fails to route the resulting abs-polynomial back into the abs solver and echoes the transformed solve() verbatim; the abs(x)/x spelling takes a different (working) path.
  - tractabilidad: bounded-single-cycle
- **[P0]** `solve(x/abs(x) = -x, x)`
  - got: `solve(x - (-x / |x|) = 0, x)`
  - correcto: `{ -1, 1 }  (x/|x|=sign(x)=-x ⇒ x=-1 for x>0, x=1 for x<0)`
  - causa-raíz: Same u/|u|=variable-RHS denominator-clearing path fails to re-dispatch the abs-polynomial residual and emits a self-referential solve()/Solve echo.
  - tractabilidad: multi-cycle
- **[P1]** `solve(-abs(x)/x = 1, x)`
  - got: `All real numbers if -x >= 0`
  - correcto: `(-infinity, 0)  — strictly x<0; x=0 makes -|x|/x = 0/0 undefined and must be excluded`
  - causa-raíz: The leading-minus numerator form -abs(x)/x routes through the '-|u| = c·u sign' template that emits 'All real numbers if [linear] >= 0' with a NON-STRICT >= (correct for solve(abs(x)=x) where the boundary IS a solution), but here the division by x makes the boundary undefined; the x≠0 condition is attached as a side required_condition yet the printed primary set keeps the closed inequality.
  - tractabilidad: bounded-single-cycle

### inequalities-all

- **[P0]** `solve(1/(abs(x) - 1) < 0, x)`
  - got: `(-infinity, infinity)  [All real numbers]`
  - correcto: `(-1, 1)   [1/(|x|-1)<0 iff |x|-1<0 iff |x|<1]`
  - causa-raíz: The reciprocal-of-abs denominator branch, when RHS is 0, skips the |x|=k sign-case split and returns the whole line instead of solving |x|-1<0.
  - tractabilidad: bounded-single-cycle
- **[P0]** `solve(1/(abs(x) - 1) > 0, x)`
  - got: `(-infinity, -infinity) U (infinity, infinity)`
  - correcto: `(-infinity, -1) U (1, infinity)   [|x|-1>0 iff |x|>1]`
  - causa-raíz: Same abs-in-reciprocal-denominator vs-zero branch: emits +/-infinity interval endpoints instead of the |x|=k split points; affects >, >=, and (via all-reals) < and <=.
  - tractabilidad: bounded-single-cycle
- **[P0]** `solve(1/(abs(x) - 2) > 0, x)`
  - got: `(-infinity, -infinity) U (infinity, infinity)`
  - correcto: `(-infinity, -2) U (2, infinity)`
  - causa-raíz: Same reciprocal-of-(|x|-const)-vs-0 defect; the shift constant does not matter.
  - tractabilidad: bounded-single-cycle
- **[P0]** `solve(1/(abs(x) + 1) > 0, x)`
  - got: `(-infinity, -infinity) U (infinity, infinity)`
  - correcto: `All real numbers   [|x|+1>0 for every x]`
  - causa-raíz: Same branch: even the always-positive denominator |x|+1 is routed to the broken abs-reciprocal-vs-zero path and produces +/-infinity endpoints.
  - tractabilidad: bounded-single-cycle
- **[P0]** `solve(5/(abs(x-3) - 1) > 0, x)`
  - got: `(-infinity, -infinity) U (infinity, infinity)`
  - correcto: `(-infinity, 2) U (4, infinity)   [|x-3|-1>0 iff |x-3|>1]`
  - causa-raíz: Same c/(|affine|-k) vs 0 branch; numerator constant, abs shift, and coefficient (also abs(2*x)-1) all reproduce the degenerate-endpoints garbage.
  - tractabilidad: bounded-single-cycle
- **[P1]** `solve(sqrt(x) + sqrt(x-1) > 1, x)`
  - got: `solve(x - (1 - (x - 1)^(1/2))^(1 / 1/2) = 0, x)`
  - correcto: `(1, infinity)`
  - causa-raíz: The sum-of-two-sqrt isolation path squares to remove a radical, mangles the exponent (1/(1/2) printed as '1 / 1/2'), and discards the inequality direction, converting the relation to an equation residual.
  - tractabilidad: multi-cycle
- **[P1]** `solve(sqrt(x+1) + sqrt(x-1) > 2, x)`
  - got: `solve(x + 2 - 4·(x - 1)^(1/2) = 0, x)`
  - correcto: `[1, infinity)   [domain x>=1; sqrt(x+1)+sqrt(x-1) is increasing, equals 2 only at... check: at x=1 -> sqrt2+0=1.414<2; solve equals 2 gives x=1.25? verify below]`
  - causa-raíz: Same sqrt-sum isolation path; operator dropped, relation converted to an equation echo.
  - tractabilidad: multi-cycle

### factor-gcd-simplify-apart

- **[P1]** `apart((2*x)/((x-1)^2*(x+1)))`
  - got: `apart((x·2)/(x·(x^2 + 1 - 2·x) + x^2 + 1 - 2·x))  [honest echo of the apart operator, no answer]`
  - correcto: `1/(2·(x-1)) + 1/(x-1)^2 - 1/(2·(x+1))  (sympy: -1/(2*(x+1)) + 1/(2*(x-1)) + (x-1)**-2)`
  - causa-raíz: The apart partial-fraction path strips/normalizes the numerator monomial's integer content and a decision gate treats a bare coefficient·monomial numerator as non-handleable, so the coefficient sibling of the working coeff-1 monomial case silently declines instead of dividing out the content. Recurring compound-coefficient-sibling bug shape.
  - tractabilidad: bounded-single-cycle

### integrate-limits

- **[P1]** `integrate(1/x^(1/3), x)   (and definite integrate(1/x^(1/3), x, 1, 8))`
  - got: `integrate(x^(2/3) / x, x)   [self-referential residual; definite forms leak too, e.g. integrate(1/x^(1/3), x, 1, 8) -> integrate(x^(2/3)/x, x, 1, 8)]`
  - correcto: `3/2*x^(2/3)  (indefinite);  9/2 for the 1..8 definite`
  - causa-raíz: Simplifier rewrites 1/x^p -> x^(1-p)/x (fractional 1-p in (0,1)); the power-rule integrator matches x^n and x^(-1/3) but not the x^(2/3)/x quotient form -> honest but misleading self-referential residual.
  - tractabilidad: bounded-single-cycle
- **[P1]** `integrate(1/(e^x+1), x, 0, 1)`
  - got: `integrate(1 / (e^x + 1), x, 0, 1)   [self-referential residual]`
  - correcto: `ln(2) + 1 - ln(1+e) ~= 0.379885`
  - causa-raíz: Definite-integral driver does not fall back to substituting endpoints into the successfully-computed indefinite antiderivative for exp/logistic-type integrands; leaks the operator instead.
  - tractabilidad: bounded-single-cycle
- **[P1]** `integrate(1/(1+cos(x)), x, 0, pi/2)`
  - got: `integrate(1 / (cos(x) + 1), x, 0, 1/2*pi)   [self-referential residual]`
  - correcto: `1  (= tan(pi/4) - tan(0))`
  - causa-raíz: Same definite-eval gap: FTC endpoint-substitution not wired for Weierstrass-substitution (tan(x/2)) and hyperbolic-square antiderivatives even when the indefinite result is available.
  - tractabilidad: bounded-single-cycle

## Ranking por ROI (síntesis)

```
The defects are already well-verified and dedup'd by root cause in the JSON reasoning. This is an analysis/ranking task — I don't need to touch the codebase. Let me synthesize the families and rank by ROI.

Let me count breadth per family from the evidence in the JSON.

**Family grouping:**

1. **Reciprocal-trig equation drops periodicity/branch/re-dispatch** — `2/sin(x)=4`, `5/cos(x)=10`, `1/sin(x)=2` (+ verified siblings: `3/sin=6`, `2/cos=4`, `6/sin=12`, `1/(2sin)=1`, `1/cos=2`, `1/tan=1`, `1/cos=-2`, `1/sin=1`, `3/tan=3`). Two sub-shapes but one root cause site (reciprocal-trig reduction doesn't delegate to the working bare-trig solver): numerator≠1 keeps only principal value; numerator=1 leaks un-dispatched cofunction residual.

2. **Reciprocal-of-abs inequality (`c/(|affine|±k) ⋚ 0`) fabricates garbage/AllReals** — `1/(|x|-1)<0`, `1/(|x|-1)>0`, `1/(|x|-2)>0`, `1/(|x|+1)>0`, `5/(|x-3|-1)>0` (+ `abs(2x)-1` sibling). All P0, all one denominator-sign root cause.

3. **Difference/sum of two radicals = c (equations)** — `sqrt(5x-1)-sqrt(x+2)=1` (drops root {2}), `sqrt(3x-1)-sqrt(x+2)=1` (leak), `sqrt(5x)-sqrt(x+2)=1` (leak + garbled exponent).

4. **Definite-integral FTC-from-known-antiderivative gap** — `1/(e^x+1)`, `1/(1+cos(x))`, `1/cosh^2`, `sech^2`, `1/(2+cos)` (+ [0,2],[0,3] variants). Indefinite works, definite leaks.

5. **sign-via-abs equation/inequality (abs in denominator)** — `x/abs(x)=x`, `x/abs(x)=-x`, `-abs(x)/x=1` (+ `2x/abs(x)=x`, `-abs(x)/x=-1`). Mixed: some bounded, `x/abs(x)=-x` marked multi-cycle.

6. **apart declines pure monomial numerator c·x^k, |c|≥2** — one family, one gate.

7. **integrate 1/x^p, p∈(0,1) simplifier-rewrite leak** — `1/x^(1/3)` and p∈(0,1) siblings.

8. **Two-sqrt INEQUALITY drops operator** — `sqrt(x)+sqrt(x-1)>1`, `sqrt(x+1)+sqrt(x-1)>2` (marked multi-cycle).

Now I'll produce the ranked list. The winner needs to maximize severity(P0) × breadth × bounded-tractability simultaneously.

---

# Prioritized ROI ranking

## #1 — BEST NEXT CYCLE: Reciprocal-of-abs inequality `c/(|affine|±k) ⋚ 0` fabricates garbage / AllReals

- **Representative input:** `solve(1/(abs(x) - 1) > 0, x)`
- **Correct answer:** `(-∞, -1) ∪ (1, ∞)` (and the `<0` sibling `(-1, 1)`)
- **Root-cause hypothesis:** The reciprocal-of-abs-affine-vs-0 branch never runs the `|·|=k` sign-case split. Instead it (a) collapses `<0`/`always-positive` cases to `AllReals`, and (b) for `>0`/`≥` cases fabricates degenerate reversed-endpoint intervals `(-∞,-∞) ∪ (∞,∞)` — the finite pole endpoints from the two abs sign-windows are lost to `±∞` sentinels. The engine already computes the correct `x≠±k` required conditions, so the abs split partially fires; only the final ray/interval combination across the two sign-windows is broken.
- **Breadth:** **6 confirmed inputs**, all **P0**, spanning both inequality directions (`<0` AllReals-garbage and `>0` degenerate-endpoints), constant shifts (`-1`, `-2`, `+1`), scaled numerators (`5/...`), and shifted/scaled abs arguments (`|x-3|`, `|2x|`). Sibling non-reciprocal and reciprocal-of-linear forms already work, isolating the defect to one composition.
- **Why bounded single cycle:** One handler — solve `c/(|u|+affine) ⋚ 0` by the sign of the denominator over the two abs windows, reusing the already-correct `|x|>k`/`|x|<k` split (which works standalone) and the already-computed pole conditions. No new machinery; wire the existing pieces and stop emitting `±∞` sentinels / AllReals. All six inputs are one
```