# Engine Improvement Scorecard

- Generated: 2026-06-15T09:43:07.589990+00:00
- Git branch: main
- Git commit: `324b68875d2271700270f5f9a33b9fe41235d310`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=352

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=792.95ms avg_case_ms=7.93 simplify=225.30ms avg_simplify_ms=2.25, sum total=200 failed=0 elapsed=688.57ms avg_case_ms=3.44 simplify=230.78ms avg_simplify_ms=1.15, product total=100 failed=0 elapsed=485.91ms avg_case_ms=4.86 simplify=140.18ms avg_simplify_ms=1.40, difference total=50 failed=0 elapsed=329.29ms avg_case_ms=6.59 simplify=104.29ms avg_simplify_ms=2.09
- Engine hotspots: sum simplify=230.78ms avg_simplify_ms=1.15 wall=688.57ms, shifted_quotient simplify=225.30ms avg_simplify_ms=2.25 wall=792.95ms, product simplify=140.18ms avg_simplify_ms=1.40 wall=485.91ms, difference simplify=104.29ms avg_simplify_ms=2.09 wall=329.29ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=792.95ms avg_case_ms=7.93 avg_simplify_ms=2.25, sum@0+100 failed=0 elapsed=500.25ms avg_case_ms=5.00 avg_simplify_ms=1.61, product@0+100 failed=0 elapsed=485.91ms avg_case_ms=4.86 avg_simplify_ms=1.40, difference@0+50 failed=0 elapsed=329.29ms avg_case_ms=6.59 avg_simplify_ms=2.09, sum@700+100 failed=0 elapsed=188.32ms avg_case_ms=1.88 avg_simplify_ms=0.70
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.19ms median_wire=13.26ms median_wall=50.82ms, difference@0+50 #174 difference runs=3 median_simplify=11.71ms median_wire=11.75ms median_wall=44.23ms, product@0+100 #175 product runs=3 median_simplify=11.67ms median_wire=11.74ms median_wall=45.35ms, sum@0+100 #173 sum runs=3 median_simplify=11.99ms median_wire=12.05ms median_wall=44.96ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.73ms median_wire=10.81ms median_wall=40.85ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.30s | passed=450 failed=0 total=450 avg_case=5.111ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.89s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.93s | passed=1 failed=0 |
