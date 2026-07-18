# Engine Improvement Scorecard

- Generated: 2026-07-18T17:03:05.953621+00:00
- Git branch: main
- Git commit: `ccbf77f155b877267d19400ee7f6384917d4e3b5`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=979.57ms avg_case_ms=9.80 simplify=274.99ms avg_simplify_ms=2.75, sum total=200 failed=0 elapsed=879.14ms avg_case_ms=4.40 simplify=283.16ms avg_simplify_ms=1.42, product total=100 failed=0 elapsed=603.78ms avg_case_ms=6.04 simplify=173.18ms avg_simplify_ms=1.73, difference total=50 failed=0 elapsed=392.92ms avg_case_ms=7.86 simplify=119.98ms avg_simplify_ms=2.40
- Engine hotspots: sum simplify=283.16ms avg_simplify_ms=1.42 wall=879.14ms, shifted_quotient simplify=274.99ms avg_simplify_ms=2.75 wall=979.57ms, product simplify=173.18ms avg_simplify_ms=1.73 wall=603.78ms, difference simplify=119.98ms avg_simplify_ms=2.40 wall=392.92ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=979.57ms avg_case_ms=9.80 avg_simplify_ms=2.75, sum@0+100 failed=0 elapsed=649.01ms avg_case_ms=6.49 avg_simplify_ms=2.01, product@0+100 failed=0 elapsed=603.78ms avg_case_ms=6.04 avg_simplify_ms=1.73, difference@0+50 failed=0 elapsed=392.92ms avg_case_ms=7.86 avg_simplify_ms=2.40, sum@700+100 failed=0 elapsed=230.13ms avg_case_ms=2.30 avg_simplify_ms=0.82
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=15.12ms median_wire=15.16ms median_wall=58.46ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.33ms median_wire=16.39ms median_wall=63.73ms, difference@0+50 #174 difference runs=3 median_simplify=15.23ms median_wire=15.29ms median_wall=65.84ms, product@0+100 #175 product runs=3 median_simplify=15.36ms median_wire=15.41ms median_wall=58.13ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.09ms median_wire=13.16ms median_wall=49.59ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.86s | passed=450 failed=0 total=450 avg_case=6.356ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.45s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
