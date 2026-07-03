# Engine Improvement Scorecard

- Generated: 2026-07-03T21:50:41.123328+00:00
- Git branch: main
- Git commit: `ba0ef677a161f3eb51ea4247527bb565df46545c`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=355

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=930.91ms avg_case_ms=9.31 simplify=258.06ms avg_simplify_ms=2.58, sum total=200 failed=0 elapsed=808.74ms avg_case_ms=4.04 simplify=261.59ms avg_simplify_ms=1.31, product total=100 failed=0 elapsed=584.53ms avg_case_ms=5.85 simplify=165.09ms avg_simplify_ms=1.65, difference total=50 failed=0 elapsed=378.90ms avg_case_ms=7.58 simplify=114.47ms avg_simplify_ms=2.29
- Engine hotspots: sum simplify=261.59ms avg_simplify_ms=1.31 wall=808.74ms, shifted_quotient simplify=258.06ms avg_simplify_ms=2.58 wall=930.91ms, product simplify=165.09ms avg_simplify_ms=1.65 wall=584.53ms, difference simplify=114.47ms avg_simplify_ms=2.29 wall=378.90ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=930.91ms avg_case_ms=9.31 avg_simplify_ms=2.58, sum@0+100 failed=0 elapsed=592.38ms avg_case_ms=5.92 avg_simplify_ms=1.84, product@0+100 failed=0 elapsed=584.53ms avg_case_ms=5.85 avg_simplify_ms=1.65, difference@0+50 failed=0 elapsed=378.90ms avg_case_ms=7.58 avg_simplify_ms=2.29, sum@700+100 failed=0 elapsed=216.36ms avg_case_ms=2.16 avg_simplify_ms=0.77
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.57ms median_wire=16.64ms median_wall=64.38ms, sum@0+100 #173 sum runs=3 median_simplify=14.57ms median_wire=14.62ms median_wall=55.69ms, product@0+100 #175 product runs=3 median_simplify=14.55ms median_wire=14.59ms median_wall=55.99ms, difference@0+50 #174 difference runs=3 median_simplify=15.42ms median_wire=15.47ms median_wall=59.44ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.38ms median_wire=12.45ms median_wall=47.21ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.70s | passed=450 failed=0 total=450 avg_case=6.000ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.93s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.85s | passed=1 failed=0 |
