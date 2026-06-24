# Engine Improvement Scorecard

- Generated: 2026-06-24T10:44:48.915060+00:00
- Git branch: main
- Git commit: `4e55763e1574a668a604af549dbc4a9f4ebeba5f`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=795.62ms avg_case_ms=7.96 simplify=228.79ms avg_simplify_ms=2.29, sum total=200 failed=0 elapsed=708.25ms avg_case_ms=3.54 simplify=241.39ms avg_simplify_ms=1.21, product total=100 failed=0 elapsed=485.89ms avg_case_ms=4.86 simplify=140.15ms avg_simplify_ms=1.40, difference total=50 failed=0 elapsed=337.40ms avg_case_ms=6.75 simplify=110.13ms avg_simplify_ms=2.20
- Engine hotspots: sum simplify=241.39ms avg_simplify_ms=1.21 wall=708.25ms, shifted_quotient simplify=228.79ms avg_simplify_ms=2.29 wall=795.62ms, product simplify=140.15ms avg_simplify_ms=1.40 wall=485.89ms, difference simplify=110.13ms avg_simplify_ms=2.20 wall=337.40ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=795.62ms avg_case_ms=7.96 avg_simplify_ms=2.29, sum@0+100 failed=0 elapsed=510.31ms avg_case_ms=5.10 avg_simplify_ms=1.67, product@0+100 failed=0 elapsed=485.89ms avg_case_ms=4.86 avg_simplify_ms=1.40, difference@0+50 failed=0 elapsed=337.40ms avg_case_ms=6.75 avg_simplify_ms=2.20, sum@700+100 failed=0 elapsed=197.94ms avg_case_ms=1.98 avg_simplify_ms=0.74
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.26ms median_wire=13.32ms median_wall=50.37ms, product@0+100 #175 product runs=3 median_simplify=12.00ms median_wire=12.05ms median_wall=45.25ms, difference@0+50 #174 difference runs=3 median_simplify=11.83ms median_wire=11.88ms median_wall=45.02ms, sum@0+100 #173 sum runs=3 median_simplify=11.91ms median_wire=11.96ms median_wall=44.90ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.77ms median_wire=10.85ms median_wall=40.68ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.33s | passed=450 failed=0 total=450 avg_case=5.178ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.49s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.07s | passed=1 failed=0 |
