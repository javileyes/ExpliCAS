# Engine Improvement Scorecard

- Generated: 2026-07-30T22:50:28.063210+00:00
- Git branch: main
- Git commit: `cd680db8b36ba89b22031e208de5e58f8c556baf`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=975.09ms avg_case_ms=9.75 simplify=273.54ms avg_simplify_ms=2.74, sum total=200 failed=0 elapsed=851.85ms avg_case_ms=4.26 simplify=272.22ms avg_simplify_ms=1.36, product total=100 failed=0 elapsed=600.41ms avg_case_ms=6.00 simplify=172.61ms avg_simplify_ms=1.73, difference total=50 failed=0 elapsed=391.27ms avg_case_ms=7.83 simplify=114.77ms avg_simplify_ms=2.30
- Engine hotspots: shifted_quotient simplify=273.54ms avg_simplify_ms=2.74 wall=975.09ms, sum simplify=272.22ms avg_simplify_ms=1.36 wall=851.85ms, product simplify=172.61ms avg_simplify_ms=1.73 wall=600.41ms, difference simplify=114.77ms avg_simplify_ms=2.30 wall=391.27ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=975.09ms avg_case_ms=9.75 avg_simplify_ms=2.74, sum@0+100 failed=0 elapsed=619.05ms avg_case_ms=6.19 avg_simplify_ms=1.89, product@0+100 failed=0 elapsed=600.41ms avg_case_ms=6.00 avg_simplify_ms=1.73, difference@0+50 failed=0 elapsed=391.27ms avg_case_ms=7.83 avg_simplify_ms=2.30, sum@700+100 failed=0 elapsed=232.80ms avg_case_ms=2.33 avg_simplify_ms=0.83
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.87ms median_wire=16.93ms median_wall=64.39ms, product@0+100 #175 product runs=3 median_simplify=15.04ms median_wire=15.09ms median_wall=57.93ms, sum@0+100 #173 sum runs=3 median_simplify=15.11ms median_wire=15.17ms median_wall=57.34ms, difference@0+50 #174 difference runs=3 median_simplify=15.09ms median_wire=15.14ms median_wall=57.19ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.93ms median_wire=13.00ms median_wall=48.56ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.82s | passed=450 failed=0 total=450 avg_case=6.267ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.15s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.74s | passed=1 failed=0 |
