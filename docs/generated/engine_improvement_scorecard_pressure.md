# Engine Improvement Scorecard

- Generated: 2026-07-24T01:23:51.239777+00:00
- Git branch: main
- Git commit: `4097be7c4aeb7a02772d3ce53b0c1ba51e87e8d2`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=990.34ms avg_case_ms=9.90 simplify=275.72ms avg_simplify_ms=2.76, sum total=200 failed=0 elapsed=873.03ms avg_case_ms=4.37 simplify=286.03ms avg_simplify_ms=1.43, product total=100 failed=0 elapsed=609.47ms avg_case_ms=6.09 simplify=174.81ms avg_simplify_ms=1.75, difference total=50 failed=0 elapsed=400.90ms avg_case_ms=8.02 simplify=121.93ms avg_simplify_ms=2.44
- Engine hotspots: sum simplify=286.03ms avg_simplify_ms=1.43 wall=873.03ms, shifted_quotient simplify=275.72ms avg_simplify_ms=2.76 wall=990.34ms, product simplify=174.81ms avg_simplify_ms=1.75 wall=609.47ms, difference simplify=121.93ms avg_simplify_ms=2.44 wall=400.90ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=990.34ms avg_case_ms=9.90 avg_simplify_ms=2.76, sum@0+100 failed=0 elapsed=642.97ms avg_case_ms=6.43 avg_simplify_ms=2.03, product@0+100 failed=0 elapsed=609.47ms avg_case_ms=6.09 avg_simplify_ms=1.75, difference@0+50 failed=0 elapsed=400.90ms avg_case_ms=8.02 avg_simplify_ms=2.44, sum@700+100 failed=0 elapsed=230.06ms avg_case_ms=2.30 avg_simplify_ms=0.83
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.18ms median_wire=17.25ms median_wall=65.10ms, sum@0+100 #173 sum runs=3 median_simplify=15.26ms median_wire=15.32ms median_wall=58.31ms, difference@0+50 #174 difference runs=3 median_simplify=15.38ms median_wire=15.43ms median_wall=59.89ms, product@0+100 #175 product runs=3 median_simplify=15.84ms median_wire=15.90ms median_wall=59.55ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.88ms median_wire=12.96ms median_wall=49.61ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.87s | passed=450 failed=0 total=450 avg_case=6.378ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.63s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
