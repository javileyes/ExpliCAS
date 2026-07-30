# Engine Improvement Scorecard

- Generated: 2026-07-30T10:27:59.201436+00:00
- Git branch: main
- Git commit: `b5da804590e7bd380b768313498a6ac3b3dcff03`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=987.13ms avg_case_ms=9.87 simplify=279.96ms avg_simplify_ms=2.80, sum total=200 failed=0 elapsed=864.61ms avg_case_ms=4.32 simplify=279.57ms avg_simplify_ms=1.40, product total=100 failed=0 elapsed=608.49ms avg_case_ms=6.08 simplify=176.07ms avg_simplify_ms=1.76, difference total=50 failed=0 elapsed=403.34ms avg_case_ms=8.07 simplify=119.20ms avg_simplify_ms=2.38
- Engine hotspots: shifted_quotient simplify=279.96ms avg_simplify_ms=2.80 wall=987.13ms, sum simplify=279.57ms avg_simplify_ms=1.40 wall=864.61ms, product simplify=176.07ms avg_simplify_ms=1.76 wall=608.49ms, difference simplify=119.20ms avg_simplify_ms=2.38 wall=403.34ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=987.13ms avg_case_ms=9.87 avg_simplify_ms=2.80, sum@0+100 failed=0 elapsed=629.89ms avg_case_ms=6.30 avg_simplify_ms=1.95, product@0+100 failed=0 elapsed=608.49ms avg_case_ms=6.08 avg_simplify_ms=1.76, difference@0+50 failed=0 elapsed=403.34ms avg_case_ms=8.07 avg_simplify_ms=2.38, sum@700+100 failed=0 elapsed=234.72ms avg_case_ms=2.35 avg_simplify_ms=0.85
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.04ms median_wire=17.11ms median_wall=65.56ms, difference@0+50 #174 difference runs=3 median_simplify=15.32ms median_wire=15.38ms median_wall=58.66ms, sum@0+100 #173 sum runs=3 median_simplify=15.21ms median_wire=15.26ms median_wall=58.97ms, product@0+100 #175 product runs=3 median_simplify=15.01ms median_wire=15.08ms median_wall=58.06ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.19ms median_wire=13.27ms median_wall=50.37ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.87s | passed=450 failed=0 total=450 avg_case=6.378ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.44s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.75s | passed=1 failed=0 |
