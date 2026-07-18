# Engine Improvement Scorecard

- Generated: 2026-07-18T22:37:56.638829+00:00
- Git branch: main
- Git commit: `f626176ff33ab0ad6f24401bbdc2dac4d0eceae3`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=989.74ms avg_case_ms=9.90 simplify=280.47ms avg_simplify_ms=2.80, sum total=200 failed=0 elapsed=876.79ms avg_case_ms=4.38 simplify=281.80ms avg_simplify_ms=1.41, product total=100 failed=0 elapsed=601.52ms avg_case_ms=6.02 simplify=172.97ms avg_simplify_ms=1.73, difference total=50 failed=0 elapsed=397.39ms avg_case_ms=7.95 simplify=120.76ms avg_simplify_ms=2.42
- Engine hotspots: sum simplify=281.80ms avg_simplify_ms=1.41 wall=876.79ms, shifted_quotient simplify=280.47ms avg_simplify_ms=2.80 wall=989.74ms, product simplify=172.97ms avg_simplify_ms=1.73 wall=601.52ms, difference simplify=120.76ms avg_simplify_ms=2.42 wall=397.39ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=989.74ms avg_case_ms=9.90 avg_simplify_ms=2.80, sum@0+100 failed=0 elapsed=646.01ms avg_case_ms=6.46 avg_simplify_ms=1.99, product@0+100 failed=0 elapsed=601.52ms avg_case_ms=6.02 avg_simplify_ms=1.73, difference@0+50 failed=0 elapsed=397.39ms avg_case_ms=7.95 avg_simplify_ms=2.42, sum@700+100 failed=0 elapsed=230.77ms avg_case_ms=2.31 avg_simplify_ms=0.83
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.89ms median_wire=16.95ms median_wall=63.45ms, sum@0+100 #173 sum runs=3 median_simplify=15.17ms median_wire=15.21ms median_wall=57.91ms, product@0+100 #175 product runs=3 median_simplify=15.04ms median_wire=15.09ms median_wall=58.02ms, difference@0+50 #174 difference runs=3 median_simplify=17.77ms median_wire=17.82ms median_wall=57.70ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.05ms median_wire=13.13ms median_wall=48.78ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.87s | passed=450 failed=0 total=450 avg_case=6.378ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.50s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.93s | passed=1 failed=0 |
