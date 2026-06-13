# Engine Improvement Scorecard

- Generated: 2026-06-13T14:24:53.816134+00:00
- Git branch: main
- Git commit: `43e97cb2d09d6f4e81008acd483d4d8fe2d823b3`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=770.73ms avg_case_ms=7.71 simplify=218.77ms avg_simplify_ms=2.19, sum total=200 failed=0 elapsed=689.00ms avg_case_ms=3.44 simplify=231.57ms avg_simplify_ms=1.16, product total=100 failed=0 elapsed=475.03ms avg_case_ms=4.75 simplify=135.57ms avg_simplify_ms=1.36, difference total=50 failed=0 elapsed=326.61ms avg_case_ms=6.53 simplify=103.46ms avg_simplify_ms=2.07
- Engine hotspots: sum simplify=231.57ms avg_simplify_ms=1.16 wall=689.00ms, shifted_quotient simplify=218.77ms avg_simplify_ms=2.19 wall=770.73ms, product simplify=135.57ms avg_simplify_ms=1.36 wall=475.03ms, difference simplify=103.46ms avg_simplify_ms=2.07 wall=326.61ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=770.73ms avg_case_ms=7.71 avg_simplify_ms=2.19, sum@0+100 failed=0 elapsed=496.00ms avg_case_ms=4.96 avg_simplify_ms=1.59, product@0+100 failed=0 elapsed=475.03ms avg_case_ms=4.75 avg_simplify_ms=1.36, difference@0+50 failed=0 elapsed=326.61ms avg_case_ms=6.53 avg_simplify_ms=2.07, sum@700+100 failed=0 elapsed=193.00ms avg_case_ms=1.93 avg_simplify_ms=0.72
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.86ms median_wire=12.93ms median_wall=49.03ms, sum@0+100 #173 sum runs=3 median_simplify=11.48ms median_wire=11.52ms median_wall=44.04ms, difference@0+50 #174 difference runs=3 median_simplify=11.55ms median_wire=11.61ms median_wall=44.37ms, product@0+100 #175 product runs=3 median_simplify=11.50ms median_wire=11.54ms median_wall=43.74ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.53ms median_wire=10.59ms median_wall=39.77ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.26s | passed=450 failed=0 total=450 avg_case=5.022ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.83s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
