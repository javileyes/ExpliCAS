# Engine Improvement Scorecard

- Generated: 2026-06-14T17:59:52.551980+00:00
- Git branch: main
- Git commit: `794be60df9a1a4d18f7a0b891cb6f19ab8dce4a7`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=351

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=807.39ms avg_case_ms=8.07 simplify=230.45ms avg_simplify_ms=2.30, sum total=200 failed=0 elapsed=720.68ms avg_case_ms=3.60 simplify=244.25ms avg_simplify_ms=1.22, product total=100 failed=0 elapsed=490.51ms avg_case_ms=4.91 simplify=140.26ms avg_simplify_ms=1.40, difference total=50 failed=0 elapsed=338.87ms avg_case_ms=6.78 simplify=108.74ms avg_simplify_ms=2.17
- Engine hotspots: sum simplify=244.25ms avg_simplify_ms=1.22 wall=720.68ms, shifted_quotient simplify=230.45ms avg_simplify_ms=2.30 wall=807.39ms, product simplify=140.26ms avg_simplify_ms=1.40 wall=490.51ms, difference simplify=108.74ms avg_simplify_ms=2.17 wall=338.87ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=807.39ms avg_case_ms=8.07 avg_simplify_ms=2.30, sum@0+100 failed=0 elapsed=520.67ms avg_case_ms=5.21 avg_simplify_ms=1.69, product@0+100 failed=0 elapsed=490.51ms avg_case_ms=4.91 avg_simplify_ms=1.40, difference@0+50 failed=0 elapsed=338.87ms avg_case_ms=6.78 avg_simplify_ms=2.17, sum@700+100 failed=0 elapsed=200.01ms avg_case_ms=2.00 avg_simplify_ms=0.75
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.22ms median_wire=13.29ms median_wall=49.87ms, sum@0+100 #173 sum runs=3 median_simplify=11.93ms median_wire=11.99ms median_wall=45.91ms, difference@0+50 #174 difference runs=3 median_simplify=11.92ms median_wire=11.97ms median_wall=45.73ms, product@0+100 #175 product runs=3 median_simplify=11.80ms median_wire=11.85ms median_wall=45.31ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.97ms median_wire=11.04ms median_wall=41.24ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.36s | passed=450 failed=0 total=450 avg_case=5.244ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.83s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.01s | passed=1 failed=0 |
