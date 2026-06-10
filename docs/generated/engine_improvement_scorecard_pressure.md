# Engine Improvement Scorecard

- Generated: 2026-06-10T21:27:04.247394+00:00
- Git branch: main
- Git commit: `9a04b9b0f326d8e5133e44ca7ce04589a8eb3e84`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=778.98ms avg_case_ms=7.79 simplify=220.98ms avg_simplify_ms=2.21, sum total=200 failed=0 elapsed=706.77ms avg_case_ms=3.53 simplify=237.82ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=475.78ms avg_case_ms=4.76 simplify=136.23ms avg_simplify_ms=1.36, difference total=50 failed=0 elapsed=329.98ms avg_case_ms=6.60 simplify=105.14ms avg_simplify_ms=2.10
- Engine hotspots: sum simplify=237.82ms avg_simplify_ms=1.19 wall=706.77ms, shifted_quotient simplify=220.98ms avg_simplify_ms=2.21 wall=778.98ms, product simplify=136.23ms avg_simplify_ms=1.36 wall=475.78ms, difference simplify=105.14ms avg_simplify_ms=2.10 wall=329.98ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=778.98ms avg_case_ms=7.79 avg_simplify_ms=2.21, sum@0+100 failed=0 elapsed=512.54ms avg_case_ms=5.13 avg_simplify_ms=1.65, product@0+100 failed=0 elapsed=475.78ms avg_case_ms=4.76 avg_simplify_ms=1.36, difference@0+50 failed=0 elapsed=329.98ms avg_case_ms=6.60 avg_simplify_ms=2.10, sum@700+100 failed=0 elapsed=194.23ms avg_case_ms=1.94 avg_simplify_ms=0.73
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.00ms median_wire=13.08ms median_wall=50.00ms, difference@0+50 #174 difference runs=3 median_simplify=11.72ms median_wire=11.77ms median_wall=49.73ms, product@0+100 #175 product runs=3 median_simplify=11.70ms median_wire=11.75ms median_wall=43.68ms, sum@0+100 #173 sum runs=3 median_simplify=11.48ms median_wire=11.53ms median_wall=43.92ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.99ms median_wire=11.07ms median_wall=41.21ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.29s | passed=450 failed=0 total=450 avg_case=5.089ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.81s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.26s | passed=1 failed=0 |
