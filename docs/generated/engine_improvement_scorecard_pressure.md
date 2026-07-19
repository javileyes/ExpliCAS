# Engine Improvement Scorecard

- Generated: 2026-07-19T00:08:21.663433+00:00
- Git branch: main
- Git commit: `9fbf3bcebc4bd8c6ece080e8a5450f1557e99eaa`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=367

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=990.47ms avg_case_ms=9.90 simplify=275.91ms avg_simplify_ms=2.76, sum total=200 failed=0 elapsed=887.36ms avg_case_ms=4.44 simplify=283.77ms avg_simplify_ms=1.42, product total=100 failed=0 elapsed=602.22ms avg_case_ms=6.02 simplify=173.19ms avg_simplify_ms=1.73, difference total=50 failed=0 elapsed=399.10ms avg_case_ms=7.98 simplify=121.54ms avg_simplify_ms=2.43
- Engine hotspots: sum simplify=283.77ms avg_simplify_ms=1.42 wall=887.36ms, shifted_quotient simplify=275.91ms avg_simplify_ms=2.76 wall=990.47ms, product simplify=173.19ms avg_simplify_ms=1.73 wall=602.22ms, difference simplify=121.54ms avg_simplify_ms=2.43 wall=399.10ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=990.47ms avg_case_ms=9.90 avg_simplify_ms=2.76, sum@0+100 failed=0 elapsed=658.23ms avg_case_ms=6.58 avg_simplify_ms=2.02, product@0+100 failed=0 elapsed=602.22ms avg_case_ms=6.02 avg_simplify_ms=1.73, difference@0+50 failed=0 elapsed=399.10ms avg_case_ms=7.98 avg_simplify_ms=2.43, sum@700+100 failed=0 elapsed=229.13ms avg_case_ms=2.29 avg_simplify_ms=0.82
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.92ms median_wire=16.99ms median_wall=64.76ms, sum@0+100 #173 sum runs=3 median_simplify=15.21ms median_wire=15.25ms median_wall=58.11ms, product@0+100 #175 product runs=3 median_simplify=15.20ms median_wire=15.24ms median_wall=57.91ms, difference@0+50 #174 difference runs=3 median_simplify=16.64ms median_wire=16.69ms median_wall=59.46ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.86ms median_wire=12.93ms median_wall=49.13ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.88s | passed=450 failed=0 total=450 avg_case=6.400ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.45s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.93s | passed=1 failed=0 |
