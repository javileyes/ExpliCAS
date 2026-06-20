# Engine Improvement Scorecard

- Generated: 2026-06-20T22:08:26.290069+00:00
- Git branch: main
- Git commit: `e20c7011aac6767e656374fb10f7e8f032223c6c`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=810.07ms avg_case_ms=8.10 simplify=233.53ms avg_simplify_ms=2.34, sum total=200 failed=0 elapsed=699.99ms avg_case_ms=3.50 simplify=236.68ms avg_simplify_ms=1.18, product total=100 failed=0 elapsed=480.96ms avg_case_ms=4.81 simplify=138.90ms avg_simplify_ms=1.39, difference total=50 failed=0 elapsed=328.23ms avg_case_ms=6.56 simplify=104.11ms avg_simplify_ms=2.08
- Engine hotspots: sum simplify=236.68ms avg_simplify_ms=1.18 wall=699.99ms, shifted_quotient simplify=233.53ms avg_simplify_ms=2.34 wall=810.07ms, product simplify=138.90ms avg_simplify_ms=1.39 wall=480.96ms, difference simplify=104.11ms avg_simplify_ms=2.08 wall=328.23ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=810.07ms avg_case_ms=8.10 avg_simplify_ms=2.34, sum@0+100 failed=0 elapsed=505.19ms avg_case_ms=5.05 avg_simplify_ms=1.64, product@0+100 failed=0 elapsed=480.96ms avg_case_ms=4.81 avg_simplify_ms=1.39, difference@0+50 failed=0 elapsed=328.23ms avg_case_ms=6.56 avg_simplify_ms=2.08, sum@700+100 failed=0 elapsed=194.80ms avg_case_ms=1.95 avg_simplify_ms=0.73
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.19ms median_wire=13.27ms median_wall=49.72ms, product@0+100 #175 product runs=3 median_simplify=11.82ms median_wire=11.88ms median_wall=44.88ms, sum@0+100 #173 sum runs=3 median_simplify=11.82ms median_wire=11.88ms median_wall=44.82ms, difference@0+50 #174 difference runs=3 median_simplify=12.14ms median_wire=12.19ms median_wall=45.57ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.63ms median_wire=10.71ms median_wall=40.41ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.32s | passed=450 failed=0 total=450 avg_case=5.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.48s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.05s | passed=1 failed=0 |
