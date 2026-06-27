# Engine Improvement Scorecard

- Generated: 2026-06-27T15:28:09.570505+00:00
- Git branch: main
- Git commit: `d97d49ed8b7ebce147e0e24ace54dc83fb6a5045`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=788.07ms avg_case_ms=7.88 simplify=225.26ms avg_simplify_ms=2.25, sum total=200 failed=0 elapsed=705.09ms avg_case_ms=3.53 simplify=243.15ms avg_simplify_ms=1.22, product total=100 failed=0 elapsed=485.29ms avg_case_ms=4.85 simplify=142.50ms avg_simplify_ms=1.43, difference total=50 failed=0 elapsed=330.92ms avg_case_ms=6.62 simplify=107.50ms avg_simplify_ms=2.15
- Engine hotspots: sum simplify=243.15ms avg_simplify_ms=1.22 wall=705.09ms, shifted_quotient simplify=225.26ms avg_simplify_ms=2.25 wall=788.07ms, product simplify=142.50ms avg_simplify_ms=1.43 wall=485.29ms, difference simplify=107.50ms avg_simplify_ms=2.15 wall=330.92ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=788.07ms avg_case_ms=7.88 avg_simplify_ms=2.25, sum@0+100 failed=0 elapsed=509.04ms avg_case_ms=5.09 avg_simplify_ms=1.68, product@0+100 failed=0 elapsed=485.29ms avg_case_ms=4.85 avg_simplify_ms=1.43, difference@0+50 failed=0 elapsed=330.92ms avg_case_ms=6.62 avg_simplify_ms=2.15, sum@700+100 failed=0 elapsed=196.05ms avg_case_ms=1.96 avg_simplify_ms=0.75
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.21ms median_wire=13.29ms median_wall=49.99ms, product@0+100 #175 product runs=3 median_simplify=11.73ms median_wire=11.78ms median_wall=44.37ms, difference@0+50 #174 difference runs=3 median_simplify=11.82ms median_wire=11.87ms median_wall=45.20ms, sum@0+100 #173 sum runs=3 median_simplify=11.62ms median_wire=11.67ms median_wall=44.72ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.60ms median_wire=10.67ms median_wall=39.96ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.31s | passed=450 failed=0 total=450 avg_case=5.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.42s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.06s | passed=1 failed=0 |
