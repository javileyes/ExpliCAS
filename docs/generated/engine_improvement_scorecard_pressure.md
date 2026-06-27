# Engine Improvement Scorecard

- Generated: 2026-06-27T15:11:38.447854+00:00
- Git branch: main
- Git commit: `45ea7fc18df1620e550a0d4568e8d0a4809cfcc9`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=804.91ms avg_case_ms=8.05 simplify=230.97ms avg_simplify_ms=2.31, sum total=200 failed=0 elapsed=705.57ms avg_case_ms=3.53 simplify=243.58ms avg_simplify_ms=1.22, product total=100 failed=0 elapsed=480.47ms avg_case_ms=4.80 simplify=141.16ms avg_simplify_ms=1.41, difference total=50 failed=0 elapsed=330.89ms avg_case_ms=6.62 simplify=106.75ms avg_simplify_ms=2.14
- Engine hotspots: sum simplify=243.58ms avg_simplify_ms=1.22 wall=705.57ms, shifted_quotient simplify=230.97ms avg_simplify_ms=2.31 wall=804.91ms, product simplify=141.16ms avg_simplify_ms=1.41 wall=480.47ms, difference simplify=106.75ms avg_simplify_ms=2.14 wall=330.89ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=804.91ms avg_case_ms=8.05 avg_simplify_ms=2.31, sum@0+100 failed=0 elapsed=509.26ms avg_case_ms=5.09 avg_simplify_ms=1.68, product@0+100 failed=0 elapsed=480.47ms avg_case_ms=4.80 avg_simplify_ms=1.41, difference@0+50 failed=0 elapsed=330.89ms avg_case_ms=6.62 avg_simplify_ms=2.14, sum@700+100 failed=0 elapsed=196.31ms avg_case_ms=1.96 avg_simplify_ms=0.76
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.07ms median_wire=13.15ms median_wall=50.28ms, difference@0+50 #174 difference runs=3 median_simplify=11.70ms median_wire=11.75ms median_wall=45.10ms, product@0+100 #175 product runs=3 median_simplify=11.65ms median_wire=11.70ms median_wall=44.82ms, sum@0+100 #173 sum runs=3 median_simplify=12.02ms median_wire=12.07ms median_wall=45.33ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.66ms median_wire=10.73ms median_wall=40.48ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.32s | passed=450 failed=0 total=450 avg_case=5.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.46s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.09s | passed=1 failed=0 |
