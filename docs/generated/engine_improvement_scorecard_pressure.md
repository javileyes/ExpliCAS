# Engine Improvement Scorecard

- Generated: 2026-07-29T14:30:34.400922+00:00
- Git branch: main
- Git commit: `b9f5b9b6bc77820612e22c6863c3ad99b8d9b4a1`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=969.16ms avg_case_ms=9.69 simplify=271.44ms avg_simplify_ms=2.71, sum total=200 failed=0 elapsed=851.82ms avg_case_ms=4.26 simplify=276.48ms avg_simplify_ms=1.38, product total=100 failed=0 elapsed=599.68ms avg_case_ms=6.00 simplify=171.65ms avg_simplify_ms=1.72, difference total=50 failed=0 elapsed=396.62ms avg_case_ms=7.93 simplify=120.70ms avg_simplify_ms=2.41
- Engine hotspots: sum simplify=276.48ms avg_simplify_ms=1.38 wall=851.82ms, shifted_quotient simplify=271.44ms avg_simplify_ms=2.71 wall=969.16ms, product simplify=171.65ms avg_simplify_ms=1.72 wall=599.68ms, difference simplify=120.70ms avg_simplify_ms=2.41 wall=396.62ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=969.16ms avg_case_ms=9.69 avg_simplify_ms=2.71, sum@0+100 failed=0 elapsed=624.23ms avg_case_ms=6.24 avg_simplify_ms=1.95, product@0+100 failed=0 elapsed=599.68ms avg_case_ms=6.00 avg_simplify_ms=1.72, difference@0+50 failed=0 elapsed=396.62ms avg_case_ms=7.93 avg_simplify_ms=2.41, sum@700+100 failed=0 elapsed=227.59ms avg_case_ms=2.28 avg_simplify_ms=0.81
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.57ms median_wire=16.64ms median_wall=63.56ms, difference@0+50 #174 difference runs=3 median_simplify=15.26ms median_wire=15.31ms median_wall=58.24ms, sum@0+100 #173 sum runs=3 median_simplify=15.46ms median_wire=15.51ms median_wall=58.82ms, product@0+100 #175 product runs=3 median_simplify=15.37ms median_wire=15.42ms median_wall=58.49ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.47ms median_wire=12.54ms median_wall=48.36ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.82s | passed=450 failed=0 total=450 avg_case=6.267ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.28s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.84s | passed=1 failed=0 |
