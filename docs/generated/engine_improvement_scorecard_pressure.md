# Engine Improvement Scorecard

- Generated: 2026-06-13T10:33:52.884822+00:00
- Git branch: main
- Git commit: `a3bfc1e7f709d54189ac2afe2a1997f420db0df3`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=782.94ms avg_case_ms=7.83 simplify=222.31ms avg_simplify_ms=2.22, sum total=200 failed=0 elapsed=690.49ms avg_case_ms=3.45 simplify=231.26ms avg_simplify_ms=1.16, product total=100 failed=0 elapsed=480.73ms avg_case_ms=4.81 simplify=137.00ms avg_simplify_ms=1.37, difference total=50 failed=0 elapsed=325.73ms avg_case_ms=6.51 simplify=103.76ms avg_simplify_ms=2.08
- Engine hotspots: sum simplify=231.26ms avg_simplify_ms=1.16 wall=690.49ms, shifted_quotient simplify=222.31ms avg_simplify_ms=2.22 wall=782.94ms, product simplify=137.00ms avg_simplify_ms=1.37 wall=480.73ms, difference simplify=103.76ms avg_simplify_ms=2.08 wall=325.73ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=782.94ms avg_case_ms=7.83 avg_simplify_ms=2.22, sum@0+100 failed=0 elapsed=501.41ms avg_case_ms=5.01 avg_simplify_ms=1.61, product@0+100 failed=0 elapsed=480.73ms avg_case_ms=4.81 avg_simplify_ms=1.37, difference@0+50 failed=0 elapsed=325.73ms avg_case_ms=6.51 avg_simplify_ms=2.08, sum@700+100 failed=0 elapsed=189.09ms avg_case_ms=1.89 avg_simplify_ms=0.70
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.83ms median_wire=12.90ms median_wall=49.33ms, product@0+100 #175 product runs=3 median_simplify=11.37ms median_wire=11.41ms median_wall=44.06ms, difference@0+50 #174 difference runs=3 median_simplify=11.54ms median_wire=11.58ms median_wall=43.97ms, sum@0+100 #173 sum runs=3 median_simplify=11.56ms median_wire=11.61ms median_wall=44.23ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.39ms median_wire=10.46ms median_wall=39.63ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.28s | passed=450 failed=0 total=450 avg_case=5.067ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.81s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.91s | passed=1 failed=0 |
