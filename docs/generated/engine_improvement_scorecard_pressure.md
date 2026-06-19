# Engine Improvement Scorecard

- Generated: 2026-06-19T12:39:41.692481+00:00
- Git branch: main
- Git commit: `f1ba4faf02cc0b7494e8428c58ddb53ba958d1f2`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=353

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=802.38ms avg_case_ms=8.02 simplify=232.39ms avg_simplify_ms=2.32, sum total=200 failed=0 elapsed=708.39ms avg_case_ms=3.54 simplify=241.76ms avg_simplify_ms=1.21, product total=100 failed=0 elapsed=485.59ms avg_case_ms=4.86 simplify=141.25ms avg_simplify_ms=1.41, difference total=50 failed=0 elapsed=334.17ms avg_case_ms=6.68 simplify=107.14ms avg_simplify_ms=2.14
- Engine hotspots: sum simplify=241.76ms avg_simplify_ms=1.21 wall=708.39ms, shifted_quotient simplify=232.39ms avg_simplify_ms=2.32 wall=802.38ms, product simplify=141.25ms avg_simplify_ms=1.41 wall=485.59ms, difference simplify=107.14ms avg_simplify_ms=2.14 wall=334.17ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=802.38ms avg_case_ms=8.02 avg_simplify_ms=2.32, sum@0+100 failed=0 elapsed=512.33ms avg_case_ms=5.12 avg_simplify_ms=1.68, product@0+100 failed=0 elapsed=485.59ms avg_case_ms=4.86 avg_simplify_ms=1.41, difference@0+50 failed=0 elapsed=334.17ms avg_case_ms=6.68 avg_simplify_ms=2.14, sum@700+100 failed=0 elapsed=196.06ms avg_case_ms=1.96 avg_simplify_ms=0.73
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.39ms median_wire=13.47ms median_wall=52.00ms, product@0+100 #175 product runs=3 median_simplify=12.60ms median_wire=12.67ms median_wall=47.06ms, sum@0+100 #173 sum runs=3 median_simplify=12.00ms median_wire=12.05ms median_wall=45.17ms, difference@0+50 #174 difference runs=3 median_simplify=11.95ms median_wire=12.00ms median_wall=45.36ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.85ms median_wire=10.92ms median_wall=41.13ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.33s | passed=450 failed=0 total=450 avg_case=5.178ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.64s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.08s | passed=1 failed=0 |
