# Engine Improvement Scorecard

- Generated: 2026-06-13T21:20:34.355479+00:00
- Git branch: main
- Git commit: `08c499f5ea1f659d459b15b96dd05aa25752f092`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=813.66ms avg_case_ms=8.14 simplify=230.41ms avg_simplify_ms=2.30, sum total=200 failed=0 elapsed=684.67ms avg_case_ms=3.42 simplify=229.12ms avg_simplify_ms=1.15, product total=100 failed=0 elapsed=467.41ms avg_case_ms=4.67 simplify=133.63ms avg_simplify_ms=1.34, difference total=50 failed=0 elapsed=320.94ms avg_case_ms=6.42 simplify=101.55ms avg_simplify_ms=2.03
- Engine hotspots: shifted_quotient simplify=230.41ms avg_simplify_ms=2.30 wall=813.66ms, sum simplify=229.12ms avg_simplify_ms=1.15 wall=684.67ms, product simplify=133.63ms avg_simplify_ms=1.34 wall=467.41ms, difference simplify=101.55ms avg_simplify_ms=2.03 wall=320.94ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=813.66ms avg_case_ms=8.14 avg_simplify_ms=2.30, sum@0+100 failed=0 elapsed=495.20ms avg_case_ms=4.95 avg_simplify_ms=1.59, product@0+100 failed=0 elapsed=467.41ms avg_case_ms=4.67 avg_simplify_ms=1.34, difference@0+50 failed=0 elapsed=320.94ms avg_case_ms=6.42 avg_simplify_ms=2.03, sum@700+100 failed=0 elapsed=189.47ms avg_case_ms=1.89 avg_simplify_ms=0.70
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.67ms median_wire=12.74ms median_wall=48.42ms, difference@0+50 #174 difference runs=3 median_simplify=11.39ms median_wire=11.44ms median_wall=43.80ms, product@0+100 #175 product runs=3 median_simplify=11.41ms median_wire=11.46ms median_wall=43.81ms, sum@0+100 #173 sum runs=3 median_simplify=11.48ms median_wire=11.53ms median_wall=43.56ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.65ms median_wire=10.72ms median_wall=40.23ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.29s | passed=450 failed=0 total=450 avg_case=5.089ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.81s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.91s | passed=1 failed=0 |
