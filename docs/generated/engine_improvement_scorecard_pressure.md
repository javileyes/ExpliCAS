# Engine Improvement Scorecard

- Generated: 2026-07-30T20:32:51.207260+00:00
- Git branch: main
- Git commit: `49574376834a85365de2baac508ed703b39dc1fc`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=380

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=979.13ms avg_case_ms=9.79 simplify=276.76ms avg_simplify_ms=2.77, sum total=200 failed=0 elapsed=858.04ms avg_case_ms=4.29 simplify=277.27ms avg_simplify_ms=1.39, product total=100 failed=0 elapsed=603.42ms avg_case_ms=6.03 simplify=174.25ms avg_simplify_ms=1.74, difference total=50 failed=0 elapsed=402.16ms avg_case_ms=8.04 simplify=118.87ms avg_simplify_ms=2.38
- Engine hotspots: sum simplify=277.27ms avg_simplify_ms=1.39 wall=858.04ms, shifted_quotient simplify=276.76ms avg_simplify_ms=2.77 wall=979.13ms, product simplify=174.25ms avg_simplify_ms=1.74 wall=603.42ms, difference simplify=118.87ms avg_simplify_ms=2.38 wall=402.16ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=979.13ms avg_case_ms=9.79 avg_simplify_ms=2.77, sum@0+100 failed=0 elapsed=623.96ms avg_case_ms=6.24 avg_simplify_ms=1.94, product@0+100 failed=0 elapsed=603.42ms avg_case_ms=6.03 avg_simplify_ms=1.74, difference@0+50 failed=0 elapsed=402.16ms avg_case_ms=8.04 avg_simplify_ms=2.38, sum@700+100 failed=0 elapsed=234.08ms avg_case_ms=2.34 avg_simplify_ms=0.84
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.80ms median_wire=16.87ms median_wall=65.21ms, difference@0+50 #174 difference runs=3 median_simplify=15.24ms median_wire=15.30ms median_wall=58.34ms, sum@0+100 #173 sum runs=3 median_simplify=14.96ms median_wire=15.01ms median_wall=57.59ms, product@0+100 #175 product runs=3 median_simplify=15.29ms median_wire=15.34ms median_wall=58.13ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.01ms median_wire=13.09ms median_wall=49.26ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.84s | passed=450 failed=0 total=450 avg_case=6.311ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.36s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.75s | passed=1 failed=0 |
