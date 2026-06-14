# Engine Improvement Scorecard

- Generated: 2026-06-14T11:18:53.661034+00:00
- Git branch: main
- Git commit: `447d91480e5abccaf1ef78d769283b339a3b467e`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=780.63ms avg_case_ms=7.81 simplify=222.13ms avg_simplify_ms=2.22, sum total=200 failed=0 elapsed=685.44ms avg_case_ms=3.43 simplify=229.43ms avg_simplify_ms=1.15, product total=100 failed=0 elapsed=481.18ms avg_case_ms=4.81 simplify=137.47ms avg_simplify_ms=1.37, difference total=50 failed=0 elapsed=325.76ms avg_case_ms=6.52 simplify=103.37ms avg_simplify_ms=2.07
- Engine hotspots: sum simplify=229.43ms avg_simplify_ms=1.15 wall=685.44ms, shifted_quotient simplify=222.13ms avg_simplify_ms=2.22 wall=780.63ms, product simplify=137.47ms avg_simplify_ms=1.37 wall=481.18ms, difference simplify=103.37ms avg_simplify_ms=2.07 wall=325.76ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=780.63ms avg_case_ms=7.81 avg_simplify_ms=2.22, sum@0+100 failed=0 elapsed=494.28ms avg_case_ms=4.94 avg_simplify_ms=1.59, product@0+100 failed=0 elapsed=481.18ms avg_case_ms=4.81 avg_simplify_ms=1.37, difference@0+50 failed=0 elapsed=325.76ms avg_case_ms=6.52 avg_simplify_ms=2.07, sum@700+100 failed=0 elapsed=191.16ms avg_case_ms=1.91 avg_simplify_ms=0.71
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.76ms median_wire=12.83ms median_wall=49.11ms, difference@0+50 #174 difference runs=3 median_simplify=11.55ms median_wire=11.60ms median_wall=43.93ms, product@0+100 #175 product runs=3 median_simplify=11.39ms median_wire=11.44ms median_wall=43.97ms, sum@0+100 #173 sum runs=3 median_simplify=11.58ms median_wire=11.63ms median_wall=44.04ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.47ms median_wire=10.55ms median_wall=39.89ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.27s | passed=450 failed=0 total=450 avg_case=5.044ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.86s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.93s | passed=1 failed=0 |
