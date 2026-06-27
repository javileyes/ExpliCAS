# Engine Improvement Scorecard

- Generated: 2026-06-27T13:22:24.197751+00:00
- Git branch: main
- Git commit: `2e212aacd18473dd5f6f9cbb90fec5a94a8e3e7d`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=790.77ms avg_case_ms=7.91 simplify=226.38ms avg_simplify_ms=2.26, sum total=200 failed=0 elapsed=718.14ms avg_case_ms=3.59 simplify=247.59ms avg_simplify_ms=1.24, product total=100 failed=0 elapsed=493.96ms avg_case_ms=4.94 simplify=144.87ms avg_simplify_ms=1.45, difference total=50 failed=0 elapsed=326.17ms avg_case_ms=6.52 simplify=105.25ms avg_simplify_ms=2.11
- Engine hotspots: sum simplify=247.59ms avg_simplify_ms=1.24 wall=718.14ms, shifted_quotient simplify=226.38ms avg_simplify_ms=2.26 wall=790.77ms, product simplify=144.87ms avg_simplify_ms=1.45 wall=493.96ms, difference simplify=105.25ms avg_simplify_ms=2.11 wall=326.17ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=790.77ms avg_case_ms=7.91 avg_simplify_ms=2.26, sum@0+100 failed=0 elapsed=522.13ms avg_case_ms=5.22 avg_simplify_ms=1.72, product@0+100 failed=0 elapsed=493.96ms avg_case_ms=4.94 avg_simplify_ms=1.45, difference@0+50 failed=0 elapsed=326.17ms avg_case_ms=6.52 avg_simplify_ms=2.11, sum@700+100 failed=0 elapsed=196.01ms avg_case_ms=1.96 avg_simplify_ms=0.76
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.97ms median_wire=13.03ms median_wall=49.41ms, difference@0+50 #174 difference runs=3 median_simplify=12.28ms median_wire=12.34ms median_wall=45.83ms, sum@0+100 #173 sum runs=3 median_simplify=11.75ms median_wire=11.80ms median_wall=44.39ms, product@0+100 #175 product runs=3 median_simplify=11.57ms median_wire=11.62ms median_wall=43.94ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.79ms median_wire=10.87ms median_wall=40.59ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.33s | passed=450 failed=0 total=450 avg_case=5.178ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.43s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.05s | passed=1 failed=0 |
