# Engine Improvement Scorecard

- Generated: 2026-06-10T14:43:44.587151+00:00
- Git branch: main
- Git commit: `45bb8ad22c9dc317a41137521b5bdd2c9bbdfb14`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=775.48ms avg_case_ms=7.75 simplify=219.07ms avg_simplify_ms=2.19, sum total=200 failed=0 elapsed=697.57ms avg_case_ms=3.49 simplify=230.77ms avg_simplify_ms=1.15, product total=100 failed=0 elapsed=471.04ms avg_case_ms=4.71 simplify=134.86ms avg_simplify_ms=1.35, difference total=50 failed=0 elapsed=321.73ms avg_case_ms=6.43 simplify=102.23ms avg_simplify_ms=2.04
- Engine hotspots: sum simplify=230.77ms avg_simplify_ms=1.15 wall=697.57ms, shifted_quotient simplify=219.07ms avg_simplify_ms=2.19 wall=775.48ms, product simplify=134.86ms avg_simplify_ms=1.35 wall=471.04ms, difference simplify=102.23ms avg_simplify_ms=2.04 wall=321.73ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=775.48ms avg_case_ms=7.75 avg_simplify_ms=2.19, sum@0+100 failed=0 elapsed=507.87ms avg_case_ms=5.08 avg_simplify_ms=1.61, product@0+100 failed=0 elapsed=471.04ms avg_case_ms=4.71 avg_simplify_ms=1.35, difference@0+50 failed=0 elapsed=321.73ms avg_case_ms=6.43 avg_simplify_ms=2.04, sum@700+100 failed=0 elapsed=189.70ms avg_case_ms=1.90 avg_simplify_ms=0.70
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.80ms median_wire=12.87ms median_wall=50.19ms, difference@0+50 #174 difference runs=3 median_simplify=11.61ms median_wire=11.66ms median_wall=45.10ms, sum@0+100 #173 sum runs=3 median_simplify=11.59ms median_wire=11.64ms median_wall=44.13ms, product@0+100 #175 product runs=3 median_simplify=11.49ms median_wire=11.53ms median_wall=43.75ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=11.10ms median_wire=11.35ms median_wall=41.50ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.27s | passed=450 failed=0 total=450 avg_case=5.044ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.81s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.20s | passed=1 failed=0 |
