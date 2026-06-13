# Engine Improvement Scorecard

- Generated: 2026-06-13T22:26:34.272850+00:00
- Git branch: main
- Git commit: `93df465703205a12f827d6133dea09f1362d39f0`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=344

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=786.64ms avg_case_ms=7.87 simplify=225.60ms avg_simplify_ms=2.26, sum total=200 failed=0 elapsed=682.79ms avg_case_ms=3.41 simplify=228.37ms avg_simplify_ms=1.14, product total=100 failed=0 elapsed=486.07ms avg_case_ms=4.86 simplify=140.87ms avg_simplify_ms=1.41, difference total=50 failed=0 elapsed=325.91ms avg_case_ms=6.52 simplify=102.90ms avg_simplify_ms=2.06
- Engine hotspots: sum simplify=228.37ms avg_simplify_ms=1.14 wall=682.79ms, shifted_quotient simplify=225.60ms avg_simplify_ms=2.26 wall=786.64ms, product simplify=140.87ms avg_simplify_ms=1.41 wall=486.07ms, difference simplify=102.90ms avg_simplify_ms=2.06 wall=325.91ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=786.64ms avg_case_ms=7.87 avg_simplify_ms=2.26, sum@0+100 failed=0 elapsed=493.95ms avg_case_ms=4.94 avg_simplify_ms=1.58, product@0+100 failed=0 elapsed=486.07ms avg_case_ms=4.86 avg_simplify_ms=1.41, difference@0+50 failed=0 elapsed=325.91ms avg_case_ms=6.52 avg_simplify_ms=2.06, sum@700+100 failed=0 elapsed=188.84ms avg_case_ms=1.89 avg_simplify_ms=0.70
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.90ms median_wire=12.97ms median_wall=49.31ms, difference@0+50 #174 difference runs=3 median_simplify=11.54ms median_wire=11.59ms median_wall=43.78ms, product@0+100 #175 product runs=3 median_simplify=11.52ms median_wire=11.56ms median_wall=43.98ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.63ms median_wire=10.71ms median_wall=40.62ms, sum@0+100 #173 sum runs=3 median_simplify=11.53ms median_wire=11.58ms median_wall=43.70ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.28s | passed=450 failed=0 total=450 avg_case=5.067ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.79s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.91s | passed=1 failed=0 |
