# Engine Improvement Scorecard

- Generated: 2026-06-13T08:00:46.559518+00:00
- Git branch: main
- Git commit: `182606156e63be5c8313846251d7868721d7b877`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=777.45ms avg_case_ms=7.77 simplify=220.28ms avg_simplify_ms=2.20, sum total=200 failed=0 elapsed=688.29ms avg_case_ms=3.44 simplify=230.79ms avg_simplify_ms=1.15, product total=100 failed=0 elapsed=468.61ms avg_case_ms=4.69 simplify=134.39ms avg_simplify_ms=1.34, difference total=50 failed=0 elapsed=323.41ms avg_case_ms=6.47 simplify=102.43ms avg_simplify_ms=2.05
- Engine hotspots: sum simplify=230.79ms avg_simplify_ms=1.15 wall=688.29ms, shifted_quotient simplify=220.28ms avg_simplify_ms=2.20 wall=777.45ms, product simplify=134.39ms avg_simplify_ms=1.34 wall=468.61ms, difference simplify=102.43ms avg_simplify_ms=2.05 wall=323.41ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=777.45ms avg_case_ms=7.77 avg_simplify_ms=2.20, sum@0+100 failed=0 elapsed=497.01ms avg_case_ms=4.97 avg_simplify_ms=1.60, product@0+100 failed=0 elapsed=468.61ms avg_case_ms=4.69 avg_simplify_ms=1.34, difference@0+50 failed=0 elapsed=323.41ms avg_case_ms=6.47 avg_simplify_ms=2.05, sum@700+100 failed=0 elapsed=191.28ms avg_case_ms=1.91 avg_simplify_ms=0.71
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.04ms median_wire=13.11ms median_wall=49.69ms, sum@0+100 #173 sum runs=3 median_simplify=11.66ms median_wire=11.71ms median_wall=44.31ms, difference@0+50 #174 difference runs=3 median_simplify=11.39ms median_wire=11.44ms median_wall=43.87ms, product@0+100 #175 product runs=3 median_simplify=11.36ms median_wire=11.41ms median_wall=43.41ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.38ms median_wire=10.45ms median_wall=39.31ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.26s | passed=450 failed=0 total=450 avg_case=5.022ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.81s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
