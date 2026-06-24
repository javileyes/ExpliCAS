# Engine Improvement Scorecard

- Generated: 2026-06-24T10:08:47.333707+00:00
- Git branch: main
- Git commit: `0f4a3c8268e64592fdce134c3d0da44635660dfc`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=800.77ms avg_case_ms=8.01 simplify=229.32ms avg_simplify_ms=2.29, sum total=200 failed=0 elapsed=707.57ms avg_case_ms=3.54 simplify=242.47ms avg_simplify_ms=1.21, product total=100 failed=0 elapsed=483.73ms avg_case_ms=4.84 simplify=140.66ms avg_simplify_ms=1.41, difference total=50 failed=0 elapsed=336.71ms avg_case_ms=6.73 simplify=108.92ms avg_simplify_ms=2.18
- Engine hotspots: sum simplify=242.47ms avg_simplify_ms=1.21 wall=707.57ms, shifted_quotient simplify=229.32ms avg_simplify_ms=2.29 wall=800.77ms, product simplify=140.66ms avg_simplify_ms=1.41 wall=483.73ms, difference simplify=108.92ms avg_simplify_ms=2.18 wall=336.71ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=800.77ms avg_case_ms=8.01 avg_simplify_ms=2.29, sum@0+100 failed=0 elapsed=511.57ms avg_case_ms=5.12 avg_simplify_ms=1.69, product@0+100 failed=0 elapsed=483.73ms avg_case_ms=4.84 avg_simplify_ms=1.41, difference@0+50 failed=0 elapsed=336.71ms avg_case_ms=6.73 avg_simplify_ms=2.18, sum@700+100 failed=0 elapsed=196.00ms avg_case_ms=1.96 avg_simplify_ms=0.74
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.22ms median_wire=13.29ms median_wall=52.21ms, sum@0+100 #173 sum runs=3 median_simplify=11.98ms median_wire=12.04ms median_wall=45.26ms, product@0+100 #175 product runs=3 median_simplify=11.72ms median_wire=11.78ms median_wall=44.76ms, difference@0+50 #174 difference runs=3 median_simplify=11.93ms median_wire=11.99ms median_wall=45.14ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.61ms median_wire=10.68ms median_wall=40.40ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.33s | passed=450 failed=0 total=450 avg_case=5.178ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.48s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.06s | passed=1 failed=0 |
