# Engine Improvement Scorecard

- Generated: 2026-07-14T00:35:14.738086+00:00
- Git branch: main
- Git commit: `24f68da5a5b31ea678ea4d489c9992bb0de69881`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=357

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=962.34ms avg_case_ms=9.62 simplify=269.13ms avg_simplify_ms=2.69, sum total=200 failed=0 elapsed=843.92ms avg_case_ms=4.22 simplify=272.44ms avg_simplify_ms=1.36, product total=100 failed=0 elapsed=581.09ms avg_case_ms=5.81 simplify=165.20ms avg_simplify_ms=1.65, difference total=50 failed=0 elapsed=389.76ms avg_case_ms=7.80 simplify=117.83ms avg_simplify_ms=2.36
- Engine hotspots: sum simplify=272.44ms avg_simplify_ms=1.36 wall=843.92ms, shifted_quotient simplify=269.13ms avg_simplify_ms=2.69 wall=962.34ms, product simplify=165.20ms avg_simplify_ms=1.65 wall=581.09ms, difference simplify=117.83ms avg_simplify_ms=2.36 wall=389.76ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=962.34ms avg_case_ms=9.62 avg_simplify_ms=2.69, sum@0+100 failed=0 elapsed=620.43ms avg_case_ms=6.20 avg_simplify_ms=1.93, product@0+100 failed=0 elapsed=581.09ms avg_case_ms=5.81 avg_simplify_ms=1.65, difference@0+50 failed=0 elapsed=389.76ms avg_case_ms=7.80 avg_simplify_ms=2.36, sum@700+100 failed=0 elapsed=223.50ms avg_case_ms=2.23 avg_simplify_ms=0.80
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.49ms median_wire=16.56ms median_wall=63.32ms, sum@0+100 #173 sum runs=3 median_simplify=14.87ms median_wire=14.91ms median_wall=56.48ms, product@0+100 #175 product runs=3 median_simplify=14.92ms median_wire=14.97ms median_wall=56.91ms, difference@0+50 #174 difference runs=3 median_simplify=15.20ms median_wire=15.25ms median_wall=63.43ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.32ms median_wire=12.39ms median_wall=47.08ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.78s | passed=450 failed=0 total=450 avg_case=6.178ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.39s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.90s | passed=1 failed=0 |
