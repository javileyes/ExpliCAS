# Engine Improvement Scorecard

- Generated: 2026-07-22T22:26:35.684692+00:00
- Git branch: main
- Git commit: `40975a77c6415cc7b5fcde9a9d3a0e19ac82b7a9`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=992.70ms avg_case_ms=9.93 simplify=278.27ms avg_simplify_ms=2.78, sum total=200 failed=0 elapsed=882.58ms avg_case_ms=4.41 simplify=289.68ms avg_simplify_ms=1.45, product total=100 failed=0 elapsed=607.15ms avg_case_ms=6.07 simplify=174.07ms avg_simplify_ms=1.74, difference total=50 failed=0 elapsed=399.52ms avg_case_ms=7.99 simplify=122.14ms avg_simplify_ms=2.44
- Engine hotspots: sum simplify=289.68ms avg_simplify_ms=1.45 wall=882.58ms, shifted_quotient simplify=278.27ms avg_simplify_ms=2.78 wall=992.70ms, product simplify=174.07ms avg_simplify_ms=1.74 wall=607.15ms, difference simplify=122.14ms avg_simplify_ms=2.44 wall=399.52ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=992.70ms avg_case_ms=9.93 avg_simplify_ms=2.78, sum@0+100 failed=0 elapsed=651.33ms avg_case_ms=6.51 avg_simplify_ms=2.07, product@0+100 failed=0 elapsed=607.15ms avg_case_ms=6.07 avg_simplify_ms=1.74, difference@0+50 failed=0 elapsed=399.52ms avg_case_ms=7.99 avg_simplify_ms=2.44, sum@700+100 failed=0 elapsed=231.25ms avg_case_ms=2.31 avg_simplify_ms=0.83
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.69ms median_wire=16.76ms median_wall=64.18ms, difference@0+50 #174 difference runs=3 median_simplify=15.58ms median_wire=15.63ms median_wall=59.37ms, sum@0+100 #173 sum runs=3 median_simplify=15.07ms median_wire=15.12ms median_wall=57.68ms, product@0+100 #175 product runs=3 median_simplify=16.63ms median_wire=16.69ms median_wall=59.28ms, sum@0+100 #157 sum runs=3 median_simplify=9.46ms median_wire=9.51ms median_wall=36.14ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.89s | passed=450 failed=0 total=450 avg_case=6.422ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.55s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.93s | passed=1 failed=0 |
