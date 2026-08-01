# Engine Improvement Scorecard

- Generated: 2026-08-01T23:24:32.086920+00:00
- Git branch: main
- Git commit: `b1032ac4089cb09fc8708671f8a7b114fd696685`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=391

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=962.06ms avg_case_ms=9.62 simplify=268.75ms avg_simplify_ms=2.69, sum total=200 failed=0 elapsed=847.55ms avg_case_ms=4.24 simplify=271.22ms avg_simplify_ms=1.36, product total=100 failed=0 elapsed=603.20ms avg_case_ms=6.03 simplify=173.17ms avg_simplify_ms=1.73, difference total=50 failed=0 elapsed=394.00ms avg_case_ms=7.88 simplify=115.28ms avg_simplify_ms=2.31
- Engine hotspots: sum simplify=271.22ms avg_simplify_ms=1.36 wall=847.55ms, shifted_quotient simplify=268.75ms avg_simplify_ms=2.69 wall=962.06ms, product simplify=173.17ms avg_simplify_ms=1.73 wall=603.20ms, difference simplify=115.28ms avg_simplify_ms=2.31 wall=394.00ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=962.06ms avg_case_ms=9.62 avg_simplify_ms=2.69, sum@0+100 failed=0 elapsed=618.19ms avg_case_ms=6.18 avg_simplify_ms=1.89, product@0+100 failed=0 elapsed=603.20ms avg_case_ms=6.03 avg_simplify_ms=1.73, difference@0+50 failed=0 elapsed=394.00ms avg_case_ms=7.88 avg_simplify_ms=2.31, sum@700+100 failed=0 elapsed=229.36ms avg_case_ms=2.29 avg_simplify_ms=0.82
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.67ms median_wire=16.74ms median_wall=63.66ms, difference@0+50 #174 difference runs=3 median_simplify=15.12ms median_wire=15.17ms median_wall=58.13ms, sum@0+100 #173 sum runs=3 median_simplify=15.11ms median_wire=15.16ms median_wall=57.79ms, product@0+100 #175 product runs=3 median_simplify=15.22ms median_wire=15.27ms median_wall=57.77ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.07ms median_wire=13.14ms median_wall=49.85ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.81s | passed=450 failed=0 total=450 avg_case=6.244ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.07s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.74s | passed=1 failed=0 |
