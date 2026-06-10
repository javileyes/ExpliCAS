# Engine Improvement Scorecard

- Generated: 2026-06-10T11:00:14.778093+00:00
- Git branch: main
- Git commit: `a2bc1947b56a7bdddebd1571ceab177406affa9d`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=10
- By area: calculus / integration:4, calculus / runtime:3, calculus / differentiation:2, calculus / robustness:1
- Recent 1: `calculus / integration` - 2026-06-08 - Discovery observe-only: polynomial cosecant/cotangent source-return still emits depth pressure
- Recent 2: `calculus / differentiation` - 2026-06-06 - Observe-only discovery: exact-square atanh scaled-root runtime is not caused by the global empty-domain check
- Recent 3: `calculus / differentiation` - 2026-06-06 - Observe-only discovery: exact-square inverse-root diff runtime is not fixed by raw target preservation

## Calculus Support Matrix Signal

- Dimension: public calculus behavior, support-matrix coverage, result simplification, domain conditions, trace quality, presentation, and verification residuals.
- Interpretation: matrix-oriented calculus lanes; classify failures by command, family, argument regime, domain regime, trace regime, presentation regime, or reusable pre-calculus dependency before adding isolated cases.
- Matrix axes: command, family, argument regime, domain regime, trace regime, presentation regime, and residual verification.
- `diff_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=263
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=344

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=784.53ms avg_case_ms=7.85 simplify=224.59ms avg_simplify_ms=2.25, sum total=200 failed=0 elapsed=720.78ms avg_case_ms=3.60 simplify=244.07ms avg_simplify_ms=1.22, product total=100 failed=0 elapsed=474.34ms avg_case_ms=4.74 simplify=136.16ms avg_simplify_ms=1.36, difference total=50 failed=0 elapsed=324.59ms avg_case_ms=6.49 simplify=103.15ms avg_simplify_ms=2.06
- Engine hotspots: sum simplify=244.07ms avg_simplify_ms=1.22 wall=720.78ms, shifted_quotient simplify=224.59ms avg_simplify_ms=2.25 wall=784.53ms, product simplify=136.16ms avg_simplify_ms=1.36 wall=474.34ms, difference simplify=103.15ms avg_simplify_ms=2.06 wall=324.59ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=784.53ms avg_case_ms=7.85 avg_simplify_ms=2.25, sum@0+100 failed=0 elapsed=526.33ms avg_case_ms=5.26 avg_simplify_ms=1.72, product@0+100 failed=0 elapsed=474.34ms avg_case_ms=4.74 avg_simplify_ms=1.36, difference@0+50 failed=0 elapsed=324.59ms avg_case_ms=6.49 avg_simplify_ms=2.06, sum@700+100 failed=0 elapsed=194.44ms avg_case_ms=1.94 avg_simplify_ms=0.72
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=12.28ms median_wire=12.34ms median_wall=45.07ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.91ms median_wire=12.99ms median_wall=49.22ms, product@0+100 #175 product runs=3 median_simplify=11.91ms median_wire=11.96ms median_wall=45.02ms, difference@0+50 #174 difference runs=3 median_simplify=11.98ms median_wire=12.03ms median_wall=45.63ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=11.42ms median_wire=11.50ms median_wall=42.94ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.31s | passed=450 failed=0 total=450 avg_case=5.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.87s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.28s | passed=1 failed=0 |
