# Engine Improvement Scorecard

- Generated: 2026-06-13T12:43:45.046473+00:00
- Git branch: main
- Git commit: `be81b90b72cdcfd7154821a648e357a3dbb9974d`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=792.65ms avg_case_ms=7.93 simplify=226.37ms avg_simplify_ms=2.26, sum total=200 failed=0 elapsed=695.31ms avg_case_ms=3.48 simplify=233.62ms avg_simplify_ms=1.17, product total=100 failed=0 elapsed=473.48ms avg_case_ms=4.73 simplify=135.75ms avg_simplify_ms=1.36, difference total=50 failed=0 elapsed=324.74ms avg_case_ms=6.49 simplify=103.20ms avg_simplify_ms=2.06
- Engine hotspots: sum simplify=233.62ms avg_simplify_ms=1.17 wall=695.31ms, shifted_quotient simplify=226.37ms avg_simplify_ms=2.26 wall=792.65ms, product simplify=135.75ms avg_simplify_ms=1.36 wall=473.48ms, difference simplify=103.20ms avg_simplify_ms=2.06 wall=324.74ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=792.65ms avg_case_ms=7.93 avg_simplify_ms=2.26, sum@0+100 failed=0 elapsed=499.99ms avg_case_ms=5.00 avg_simplify_ms=1.61, product@0+100 failed=0 elapsed=473.48ms avg_case_ms=4.73 avg_simplify_ms=1.36, difference@0+50 failed=0 elapsed=324.74ms avg_case_ms=6.49 avg_simplify_ms=2.06, sum@700+100 failed=0 elapsed=195.31ms avg_case_ms=1.95 avg_simplify_ms=0.73
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.02ms median_wire=13.09ms median_wall=49.46ms, product@0+100 #175 product runs=3 median_simplify=11.51ms median_wire=11.56ms median_wall=43.75ms, sum@0+100 #173 sum runs=3 median_simplify=11.60ms median_wire=11.65ms median_wall=43.94ms, difference@0+50 #174 difference runs=3 median_simplify=11.78ms median_wire=11.83ms median_wall=44.65ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.71ms median_wire=10.78ms median_wall=40.12ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.29s | passed=450 failed=0 total=450 avg_case=5.089ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.81s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
