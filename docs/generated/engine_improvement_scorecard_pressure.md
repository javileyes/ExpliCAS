# Engine Improvement Scorecard

- Generated: 2026-06-13T16:29:51.742491+00:00
- Git branch: main
- Git commit: `752055a34896f9e9f4e9b85e55b31928e1d1930f`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=775.21ms avg_case_ms=7.75 simplify=218.10ms avg_simplify_ms=2.18, sum total=200 failed=0 elapsed=680.69ms avg_case_ms=3.40 simplify=226.37ms avg_simplify_ms=1.13, product total=100 failed=0 elapsed=467.90ms avg_case_ms=4.68 simplify=131.61ms avg_simplify_ms=1.32, difference total=50 failed=0 elapsed=319.48ms avg_case_ms=6.39 simplify=101.49ms avg_simplify_ms=2.03
- Engine hotspots: sum simplify=226.37ms avg_simplify_ms=1.13 wall=680.69ms, shifted_quotient simplify=218.10ms avg_simplify_ms=2.18 wall=775.21ms, product simplify=131.61ms avg_simplify_ms=1.32 wall=467.90ms, difference simplify=101.49ms avg_simplify_ms=2.03 wall=319.48ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=775.21ms avg_case_ms=7.75 avg_simplify_ms=2.18, sum@0+100 failed=0 elapsed=492.00ms avg_case_ms=4.92 avg_simplify_ms=1.57, product@0+100 failed=0 elapsed=467.90ms avg_case_ms=4.68 avg_simplify_ms=1.32, difference@0+50 failed=0 elapsed=319.48ms avg_case_ms=6.39 avg_simplify_ms=2.03, sum@700+100 failed=0 elapsed=188.68ms avg_case_ms=1.89 avg_simplify_ms=0.70
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.89ms median_wire=12.96ms median_wall=49.08ms, difference@0+50 #174 difference runs=3 median_simplify=11.40ms median_wire=11.45ms median_wall=44.02ms, sum@0+100 #173 sum runs=3 median_simplify=11.39ms median_wire=11.44ms median_wall=44.26ms, product@0+100 #175 product runs=3 median_simplify=11.19ms median_wire=11.24ms median_wall=43.52ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.73ms median_wire=10.80ms median_wall=40.46ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.24s | passed=450 failed=0 total=450 avg_case=4.978ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.81s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.93s | passed=1 failed=0 |
