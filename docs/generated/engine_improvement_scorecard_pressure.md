# Engine Improvement Scorecard

- Generated: 2026-06-14T16:43:56.424281+00:00
- Git branch: main
- Git commit: `d26724834a67fcc4d6e4fa09d30c6f0e01a0d048`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=351

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=773.61ms avg_case_ms=7.74 simplify=219.17ms avg_simplify_ms=2.19, sum total=200 failed=0 elapsed=690.94ms avg_case_ms=3.45 simplify=231.18ms avg_simplify_ms=1.16, product total=100 failed=0 elapsed=472.04ms avg_case_ms=4.72 simplify=135.52ms avg_simplify_ms=1.36, difference total=50 failed=0 elapsed=323.50ms avg_case_ms=6.47 simplify=103.92ms avg_simplify_ms=2.08
- Engine hotspots: sum simplify=231.18ms avg_simplify_ms=1.16 wall=690.94ms, shifted_quotient simplify=219.17ms avg_simplify_ms=2.19 wall=773.61ms, product simplify=135.52ms avg_simplify_ms=1.36 wall=472.04ms, difference simplify=103.92ms avg_simplify_ms=2.08 wall=323.50ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=773.61ms avg_case_ms=7.74 avg_simplify_ms=2.19, sum@0+100 failed=0 elapsed=500.48ms avg_case_ms=5.00 avg_simplify_ms=1.61, product@0+100 failed=0 elapsed=472.04ms avg_case_ms=4.72 avg_simplify_ms=1.36, difference@0+50 failed=0 elapsed=323.50ms avg_case_ms=6.47 avg_simplify_ms=2.08, sum@700+100 failed=0 elapsed=190.46ms avg_case_ms=1.90 avg_simplify_ms=0.71
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.77ms median_wire=12.83ms median_wall=48.62ms, sum@0+100 #173 sum runs=3 median_simplify=11.32ms median_wire=11.37ms median_wall=43.33ms, difference@0+50 #174 difference runs=3 median_simplify=11.62ms median_wire=11.66ms median_wall=44.26ms, product@0+100 #175 product runs=3 median_simplify=11.57ms median_wire=11.63ms median_wall=43.87ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.48ms median_wire=10.55ms median_wall=39.63ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.26s | passed=450 failed=0 total=450 avg_case=5.022ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.82s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.98s | passed=1 failed=0 |
