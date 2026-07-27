# Engine Improvement Scorecard

- Generated: 2026-07-27T01:05:41.401089+00:00
- Git branch: main
- Git commit: `b8f2937f6ace8d213160d82625ec2d5c4e9dad51`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=380

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=956.69ms avg_case_ms=9.57 simplify=268.12ms avg_simplify_ms=2.68, sum total=200 failed=0 elapsed=843.35ms avg_case_ms=4.22 simplify=274.71ms avg_simplify_ms=1.37, product total=100 failed=0 elapsed=592.10ms avg_case_ms=5.92 simplify=169.58ms avg_simplify_ms=1.70, difference total=50 failed=0 elapsed=382.92ms avg_case_ms=7.66 simplify=117.69ms avg_simplify_ms=2.35
- Engine hotspots: sum simplify=274.71ms avg_simplify_ms=1.37 wall=843.35ms, shifted_quotient simplify=268.12ms avg_simplify_ms=2.68 wall=956.69ms, product simplify=169.58ms avg_simplify_ms=1.70 wall=592.10ms, difference simplify=117.69ms avg_simplify_ms=2.35 wall=382.92ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=956.69ms avg_case_ms=9.57 avg_simplify_ms=2.68, sum@0+100 failed=0 elapsed=617.25ms avg_case_ms=6.17 avg_simplify_ms=1.94, product@0+100 failed=0 elapsed=592.10ms avg_case_ms=5.92 avg_simplify_ms=1.70, difference@0+50 failed=0 elapsed=382.92ms avg_case_ms=7.66 avg_simplify_ms=2.35, sum@700+100 failed=0 elapsed=226.10ms avg_case_ms=2.26 avg_simplify_ms=0.81
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.25ms median_wire=16.31ms median_wall=62.44ms, sum@0+100 #173 sum runs=3 median_simplify=14.56ms median_wire=14.60ms median_wall=55.45ms, product@0+100 #175 product runs=3 median_simplify=14.47ms median_wire=14.51ms median_wall=55.87ms, difference@0+50 #174 difference runs=3 median_simplify=15.00ms median_wire=15.04ms median_wall=56.91ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.25ms median_wire=12.32ms median_wall=47.76ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.78s | passed=450 failed=0 total=450 avg_case=6.178ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.24s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.84s | passed=1 failed=0 |
