# Engine Improvement Scorecard

- Generated: 2026-07-18T07:48:18.809117+00:00
- Git branch: main
- Git commit: `c9144a13dedc657de9b26032b3374864a4df11e2`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=367

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=975.91ms avg_case_ms=9.76 simplify=273.45ms avg_simplify_ms=2.73, sum total=200 failed=0 elapsed=892.54ms avg_case_ms=4.46 simplify=290.45ms avg_simplify_ms=1.45, product total=100 failed=0 elapsed=608.19ms avg_case_ms=6.08 simplify=175.02ms avg_simplify_ms=1.75, difference total=50 failed=0 elapsed=401.67ms avg_case_ms=8.03 simplify=121.80ms avg_simplify_ms=2.44
- Engine hotspots: sum simplify=290.45ms avg_simplify_ms=1.45 wall=892.54ms, shifted_quotient simplify=273.45ms avg_simplify_ms=2.73 wall=975.91ms, product simplify=175.02ms avg_simplify_ms=1.75 wall=608.19ms, difference simplify=121.80ms avg_simplify_ms=2.44 wall=401.67ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=975.91ms avg_case_ms=9.76 avg_simplify_ms=2.73, sum@0+100 failed=0 elapsed=659.68ms avg_case_ms=6.60 avg_simplify_ms=2.07, product@0+100 failed=0 elapsed=608.19ms avg_case_ms=6.08 avg_simplify_ms=1.75, difference@0+50 failed=0 elapsed=401.67ms avg_case_ms=8.03 avg_simplify_ms=2.44, sum@700+100 failed=0 elapsed=232.86ms avg_case_ms=2.33 avg_simplify_ms=0.84
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.90ms median_wire=16.97ms median_wall=64.89ms, difference@0+50 #174 difference runs=3 median_simplify=15.54ms median_wire=15.59ms median_wall=60.51ms, sum@0+100 #173 sum runs=3 median_simplify=20.16ms median_wire=20.22ms median_wall=70.31ms, product@0+100 #175 product runs=3 median_simplify=15.32ms median_wire=15.37ms median_wall=59.05ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.08ms median_wire=13.16ms median_wall=50.48ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.88s | passed=450 failed=0 total=450 avg_case=6.400ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.48s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.91s | passed=1 failed=0 |
