# Engine Improvement Scorecard

- Generated: 2026-06-10T07:15:05.791013+00:00
- Git branch: main
- Git commit: `f600df84014ba641cba817bb64e48f23b5f4ad6e`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=778.02ms avg_case_ms=7.78 simplify=219.76ms avg_simplify_ms=2.20, sum total=200 failed=0 elapsed=707.62ms avg_case_ms=3.54 simplify=236.82ms avg_simplify_ms=1.18, product total=100 failed=0 elapsed=471.78ms avg_case_ms=4.72 simplify=135.79ms avg_simplify_ms=1.36, difference total=50 failed=0 elapsed=325.20ms avg_case_ms=6.50 simplify=103.70ms avg_simplify_ms=2.07
- Engine hotspots: sum simplify=236.82ms avg_simplify_ms=1.18 wall=707.62ms, shifted_quotient simplify=219.76ms avg_simplify_ms=2.20 wall=778.02ms, product simplify=135.79ms avg_simplify_ms=1.36 wall=471.78ms, difference simplify=103.70ms avg_simplify_ms=2.07 wall=325.20ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=778.02ms avg_case_ms=7.78 avg_simplify_ms=2.20, sum@0+100 failed=0 elapsed=517.15ms avg_case_ms=5.17 avg_simplify_ms=1.66, product@0+100 failed=0 elapsed=471.78ms avg_case_ms=4.72 avg_simplify_ms=1.36, difference@0+50 failed=0 elapsed=325.20ms avg_case_ms=6.50 avg_simplify_ms=2.07, sum@700+100 failed=0 elapsed=190.46ms avg_case_ms=1.90 avg_simplify_ms=0.70
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.78ms median_wire=12.84ms median_wall=48.32ms, difference@0+50 #174 difference runs=3 median_simplify=11.35ms median_wire=11.41ms median_wall=43.56ms, sum@0+100 #173 sum runs=3 median_simplify=11.16ms median_wire=11.21ms median_wall=43.59ms, product@0+100 #175 product runs=3 median_simplify=11.56ms median_wire=11.61ms median_wall=44.00ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=11.52ms median_wire=11.59ms median_wall=42.28ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.28s | passed=450 failed=0 total=450 avg_case=5.067ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.80s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.03s | passed=1 failed=0 |
