# Engine Improvement Scorecard

- Generated: 2026-07-10T18:24:03.773930+00:00
- Git branch: main
- Git commit: `b83bc4134ca81691fb8cd3aaeb37403112164dab`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=937.48ms avg_case_ms=9.37 simplify=259.73ms avg_simplify_ms=2.60, sum total=200 failed=0 elapsed=823.40ms avg_case_ms=4.12 simplify=265.35ms avg_simplify_ms=1.33, product total=100 failed=0 elapsed=581.49ms avg_case_ms=5.81 simplify=165.40ms avg_simplify_ms=1.65, difference total=50 failed=0 elapsed=387.19ms avg_case_ms=7.74 simplify=116.83ms avg_simplify_ms=2.34
- Engine hotspots: sum simplify=265.35ms avg_simplify_ms=1.33 wall=823.40ms, shifted_quotient simplify=259.73ms avg_simplify_ms=2.60 wall=937.48ms, product simplify=165.40ms avg_simplify_ms=1.65 wall=581.49ms, difference simplify=116.83ms avg_simplify_ms=2.34 wall=387.19ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=937.48ms avg_case_ms=9.37 avg_simplify_ms=2.60, sum@0+100 failed=0 elapsed=599.99ms avg_case_ms=6.00 avg_simplify_ms=1.86, product@0+100 failed=0 elapsed=581.49ms avg_case_ms=5.81 avg_simplify_ms=1.65, difference@0+50 failed=0 elapsed=387.19ms avg_case_ms=7.74 avg_simplify_ms=2.34, sum@700+100 failed=0 elapsed=223.41ms avg_case_ms=2.23 avg_simplify_ms=0.79
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.57ms median_wire=16.64ms median_wall=62.79ms, product@0+100 #175 product runs=3 median_simplify=14.67ms median_wire=14.72ms median_wall=56.33ms, difference@0+50 #174 difference runs=3 median_simplify=14.97ms median_wire=15.01ms median_wall=56.80ms, sum@0+100 #173 sum runs=3 median_simplify=14.94ms median_wire=14.99ms median_wall=56.47ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.05ms median_wire=13.12ms median_wall=48.84ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.73s | passed=450 failed=0 total=450 avg_case=6.067ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.97s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.87s | passed=1 failed=0 |
