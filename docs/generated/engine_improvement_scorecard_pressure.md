# Engine Improvement Scorecard

- Generated: 2026-07-29T17:16:07.262006+00:00
- Git branch: main
- Git commit: `2fde3123d73237cf6b698597659c85a057f7531d`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=981.74ms avg_case_ms=9.82 simplify=273.32ms avg_simplify_ms=2.73, sum total=200 failed=0 elapsed=864.28ms avg_case_ms=4.32 simplify=280.26ms avg_simplify_ms=1.40, product total=100 failed=0 elapsed=611.83ms avg_case_ms=6.12 simplify=175.10ms avg_simplify_ms=1.75, difference total=50 failed=0 elapsed=399.95ms avg_case_ms=8.00 simplify=121.60ms avg_simplify_ms=2.43
- Engine hotspots: sum simplify=280.26ms avg_simplify_ms=1.40 wall=864.28ms, shifted_quotient simplify=273.32ms avg_simplify_ms=2.73 wall=981.74ms, product simplify=175.10ms avg_simplify_ms=1.75 wall=611.83ms, difference simplify=121.60ms avg_simplify_ms=2.43 wall=399.95ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=981.74ms avg_case_ms=9.82 avg_simplify_ms=2.73, sum@0+100 failed=0 elapsed=632.33ms avg_case_ms=6.32 avg_simplify_ms=1.98, product@0+100 failed=0 elapsed=611.83ms avg_case_ms=6.12 avg_simplify_ms=1.75, difference@0+50 failed=0 elapsed=399.95ms avg_case_ms=8.00 avg_simplify_ms=2.43, sum@700+100 failed=0 elapsed=231.95ms avg_case_ms=2.32 avg_simplify_ms=0.83
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.96ms median_wire=17.03ms median_wall=65.52ms, sum@0+100 #173 sum runs=3 median_simplify=15.52ms median_wire=15.58ms median_wall=59.30ms, product@0+100 #175 product runs=3 median_simplify=15.39ms median_wire=15.44ms median_wall=58.80ms, difference@0+50 #174 difference runs=3 median_simplify=15.28ms median_wire=15.33ms median_wall=58.23ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.20ms median_wire=13.28ms median_wall=49.66ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.86s | passed=450 failed=0 total=450 avg_case=6.356ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.48s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.87s | passed=1 failed=0 |
