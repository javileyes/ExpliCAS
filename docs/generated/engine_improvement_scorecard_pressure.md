# Engine Improvement Scorecard

- Generated: 2026-07-14T11:27:06.707676+00:00
- Git branch: main
- Git commit: `bc45c3742e0400b37a34fadc48121957be154ccf`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=357

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=959.84ms avg_case_ms=9.60 simplify=265.06ms avg_simplify_ms=2.65, sum total=200 failed=0 elapsed=856.71ms avg_case_ms=4.28 simplify=276.40ms avg_simplify_ms=1.38, product total=100 failed=0 elapsed=588.20ms avg_case_ms=5.88 simplify=167.37ms avg_simplify_ms=1.67, difference total=50 failed=0 elapsed=389.67ms avg_case_ms=7.79 simplify=118.70ms avg_simplify_ms=2.37
- Engine hotspots: sum simplify=276.40ms avg_simplify_ms=1.38 wall=856.71ms, shifted_quotient simplify=265.06ms avg_simplify_ms=2.65 wall=959.84ms, product simplify=167.37ms avg_simplify_ms=1.67 wall=588.20ms, difference simplify=118.70ms avg_simplify_ms=2.37 wall=389.67ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=959.84ms avg_case_ms=9.60 avg_simplify_ms=2.65, sum@0+100 failed=0 elapsed=629.43ms avg_case_ms=6.29 avg_simplify_ms=1.95, product@0+100 failed=0 elapsed=588.20ms avg_case_ms=5.88 avg_simplify_ms=1.67, difference@0+50 failed=0 elapsed=389.67ms avg_case_ms=7.79 avg_simplify_ms=2.37, sum@700+100 failed=0 elapsed=227.28ms avg_case_ms=2.27 avg_simplify_ms=0.81
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=15.01ms median_wire=15.05ms median_wall=57.88ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.67ms median_wire=16.73ms median_wall=63.37ms, product@0+100 #175 product runs=3 median_simplify=15.09ms median_wire=15.14ms median_wall=57.36ms, difference@0+50 #174 difference runs=3 median_simplify=15.25ms median_wire=15.30ms median_wall=59.39ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.64ms median_wire=12.72ms median_wall=48.44ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.79s | passed=450 failed=0 total=450 avg_case=6.200ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.37s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.90s | passed=1 failed=0 |
