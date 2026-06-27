# Engine Improvement Scorecard

- Generated: 2026-06-27T01:15:20.825958+00:00
- Git branch: main
- Git commit: `fbe46a3a9b712ae432d7fe98ceba6903ecafb4bd`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=794.17ms avg_case_ms=7.94 simplify=226.36ms avg_simplify_ms=2.26, sum total=200 failed=0 elapsed=698.85ms avg_case_ms=3.49 simplify=241.11ms avg_simplify_ms=1.21, product total=100 failed=0 elapsed=485.37ms avg_case_ms=4.85 simplify=142.45ms avg_simplify_ms=1.42, difference total=50 failed=0 elapsed=328.53ms avg_case_ms=6.57 simplify=106.17ms avg_simplify_ms=2.12
- Engine hotspots: sum simplify=241.11ms avg_simplify_ms=1.21 wall=698.85ms, shifted_quotient simplify=226.36ms avg_simplify_ms=2.26 wall=794.17ms, product simplify=142.45ms avg_simplify_ms=1.42 wall=485.37ms, difference simplify=106.17ms avg_simplify_ms=2.12 wall=328.53ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=794.17ms avg_case_ms=7.94 avg_simplify_ms=2.26, sum@0+100 failed=0 elapsed=503.83ms avg_case_ms=5.04 avg_simplify_ms=1.66, product@0+100 failed=0 elapsed=485.37ms avg_case_ms=4.85 avg_simplify_ms=1.42, difference@0+50 failed=0 elapsed=328.53ms avg_case_ms=6.57 avg_simplify_ms=2.12, sum@700+100 failed=0 elapsed=195.02ms avg_case_ms=1.95 avg_simplify_ms=0.75
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.15ms median_wire=13.23ms median_wall=49.94ms, product@0+100 #175 product runs=3 median_simplify=11.85ms median_wire=11.91ms median_wall=44.82ms, difference@0+50 #174 difference runs=3 median_simplify=11.73ms median_wire=11.78ms median_wall=44.68ms, sum@0+100 #173 sum runs=3 median_simplify=11.68ms median_wire=11.73ms median_wall=44.41ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.60ms median_wire=10.67ms median_wall=40.19ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.31s | passed=450 failed=0 total=450 avg_case=5.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.42s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.05s | passed=1 failed=0 |
