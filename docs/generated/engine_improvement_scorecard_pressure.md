# Engine Improvement Scorecard

- Generated: 2026-07-13T11:13:36.435105+00:00
- Git branch: main
- Git commit: `f1a467942bab2b69d69dd6e4b204caca522256de`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=963.89ms avg_case_ms=9.64 simplify=266.11ms avg_simplify_ms=2.66, sum total=200 failed=0 elapsed=858.97ms avg_case_ms=4.29 simplify=279.74ms avg_simplify_ms=1.40, product total=100 failed=0 elapsed=587.16ms avg_case_ms=5.87 simplify=167.09ms avg_simplify_ms=1.67, difference total=50 failed=0 elapsed=399.18ms avg_case_ms=7.98 simplify=120.98ms avg_simplify_ms=2.42
- Engine hotspots: sum simplify=279.74ms avg_simplify_ms=1.40 wall=858.97ms, shifted_quotient simplify=266.11ms avg_simplify_ms=2.66 wall=963.89ms, product simplify=167.09ms avg_simplify_ms=1.67 wall=587.16ms, difference simplify=120.98ms avg_simplify_ms=2.42 wall=399.18ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=963.89ms avg_case_ms=9.64 avg_simplify_ms=2.66, sum@0+100 failed=0 elapsed=631.98ms avg_case_ms=6.32 avg_simplify_ms=1.97, product@0+100 failed=0 elapsed=587.16ms avg_case_ms=5.87 avg_simplify_ms=1.67, difference@0+50 failed=0 elapsed=399.18ms avg_case_ms=7.98 avg_simplify_ms=2.42, sum@700+100 failed=0 elapsed=226.99ms avg_case_ms=2.27 avg_simplify_ms=0.82
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=15.52ms median_wire=15.57ms median_wall=57.95ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.65ms median_wire=16.73ms median_wall=64.11ms, difference@0+50 #174 difference runs=3 median_simplify=14.99ms median_wire=15.04ms median_wall=57.11ms, product@0+100 #175 product runs=3 median_simplify=18.63ms median_wire=18.69ms median_wall=64.66ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.53ms median_wire=12.61ms median_wall=48.05ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.81s | passed=450 failed=0 total=450 avg_case=6.244ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.44s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.90s | passed=1 failed=0 |
