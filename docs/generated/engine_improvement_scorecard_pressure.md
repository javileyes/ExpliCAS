# Engine Improvement Scorecard

- Generated: 2026-06-27T14:07:33.826612+00:00
- Git branch: main
- Git commit: `0301ea30e77d6e87e6446efd0b666d17e3f969c2`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=811.69ms avg_case_ms=8.12 simplify=232.74ms avg_simplify_ms=2.33, sum total=200 failed=0 elapsed=720.58ms avg_case_ms=3.60 simplify=249.45ms avg_simplify_ms=1.25, product total=100 failed=0 elapsed=492.86ms avg_case_ms=4.93 simplify=144.67ms avg_simplify_ms=1.45, difference total=50 failed=0 elapsed=339.69ms avg_case_ms=6.79 simplify=109.83ms avg_simplify_ms=2.20
- Engine hotspots: sum simplify=249.45ms avg_simplify_ms=1.25 wall=720.58ms, shifted_quotient simplify=232.74ms avg_simplify_ms=2.33 wall=811.69ms, product simplify=144.67ms avg_simplify_ms=1.45 wall=492.86ms, difference simplify=109.83ms avg_simplify_ms=2.20 wall=339.69ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=811.69ms avg_case_ms=8.12 avg_simplify_ms=2.33, sum@0+100 failed=0 elapsed=517.52ms avg_case_ms=5.18 avg_simplify_ms=1.71, product@0+100 failed=0 elapsed=492.86ms avg_case_ms=4.93 avg_simplify_ms=1.45, difference@0+50 failed=0 elapsed=339.69ms avg_case_ms=6.79 avg_simplify_ms=2.20, sum@700+100 failed=0 elapsed=203.06ms avg_case_ms=2.03 avg_simplify_ms=0.79
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.06ms median_wire=13.14ms median_wall=50.05ms, product@0+100 #175 product runs=3 median_simplify=12.00ms median_wire=12.05ms median_wall=44.92ms, difference@0+50 #174 difference runs=3 median_simplify=11.98ms median_wire=12.04ms median_wall=45.65ms, sum@0+100 #173 sum runs=3 median_simplify=11.74ms median_wire=11.79ms median_wall=45.15ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.66ms median_wire=10.74ms median_wall=41.20ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.37s | passed=450 failed=0 total=450 avg_case=5.267ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.47s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.07s | passed=1 failed=0 |
