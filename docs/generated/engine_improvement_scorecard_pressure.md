# Engine Improvement Scorecard

- Generated: 2026-07-10T23:56:45.140158+00:00
- Git branch: main
- Git commit: `3efdac4ccef3ead5112e56387285dec04080f82f`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=939.44ms avg_case_ms=9.39 simplify=260.61ms avg_simplify_ms=2.61, sum total=200 failed=0 elapsed=835.97ms avg_case_ms=4.18 simplify=269.69ms avg_simplify_ms=1.35, product total=100 failed=0 elapsed=584.01ms avg_case_ms=5.84 simplify=165.92ms avg_simplify_ms=1.66, difference total=50 failed=0 elapsed=390.65ms avg_case_ms=7.81 simplify=118.16ms avg_simplify_ms=2.36
- Engine hotspots: sum simplify=269.69ms avg_simplify_ms=1.35 wall=835.97ms, shifted_quotient simplify=260.61ms avg_simplify_ms=2.61 wall=939.44ms, product simplify=165.92ms avg_simplify_ms=1.66 wall=584.01ms, difference simplify=118.16ms avg_simplify_ms=2.36 wall=390.65ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=939.44ms avg_case_ms=9.39 avg_simplify_ms=2.61, sum@0+100 failed=0 elapsed=610.76ms avg_case_ms=6.11 avg_simplify_ms=1.89, product@0+100 failed=0 elapsed=584.01ms avg_case_ms=5.84 avg_simplify_ms=1.66, difference@0+50 failed=0 elapsed=390.65ms avg_case_ms=7.81 avg_simplify_ms=2.36, sum@700+100 failed=0 elapsed=225.21ms avg_case_ms=2.25 avg_simplify_ms=0.81
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.47ms median_wire=16.53ms median_wall=63.23ms, sum@0+100 #173 sum runs=3 median_simplify=14.89ms median_wire=14.94ms median_wall=57.46ms, difference@0+50 #174 difference runs=3 median_simplify=15.02ms median_wire=15.07ms median_wall=57.69ms, product@0+100 #175 product runs=3 median_simplify=14.92ms median_wire=14.96ms median_wall=57.71ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.73ms median_wire=12.80ms median_wall=48.67ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.75s | passed=450 failed=0 total=450 avg_case=6.111ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.99s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.87s | passed=1 failed=0 |
