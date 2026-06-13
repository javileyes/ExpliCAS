# Engine Improvement Scorecard

- Generated: 2026-06-13T22:01:06.172621+00:00
- Git branch: main
- Git commit: `4dd66f549cdfd5045846b4c901a6b4f1e714807b`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=771.78ms avg_case_ms=7.72 simplify=219.27ms avg_simplify_ms=2.19, sum total=200 failed=0 elapsed=684.85ms avg_case_ms=3.42 simplify=229.62ms avg_simplify_ms=1.15, product total=100 failed=0 elapsed=475.10ms avg_case_ms=4.75 simplify=136.09ms avg_simplify_ms=1.36, difference total=50 failed=0 elapsed=324.63ms avg_case_ms=6.49 simplify=103.14ms avg_simplify_ms=2.06
- Engine hotspots: sum simplify=229.62ms avg_simplify_ms=1.15 wall=684.85ms, shifted_quotient simplify=219.27ms avg_simplify_ms=2.19 wall=771.78ms, product simplify=136.09ms avg_simplify_ms=1.36 wall=475.10ms, difference simplify=103.14ms avg_simplify_ms=2.06 wall=324.63ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=771.78ms avg_case_ms=7.72 avg_simplify_ms=2.19, sum@0+100 failed=0 elapsed=496.19ms avg_case_ms=4.96 avg_simplify_ms=1.59, product@0+100 failed=0 elapsed=475.10ms avg_case_ms=4.75 avg_simplify_ms=1.36, difference@0+50 failed=0 elapsed=324.63ms avg_case_ms=6.49 avg_simplify_ms=2.06, sum@700+100 failed=0 elapsed=188.67ms avg_case_ms=1.89 avg_simplify_ms=0.70
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.84ms median_wire=12.90ms median_wall=48.56ms, difference@0+50 #174 difference runs=3 median_simplify=11.59ms median_wire=11.65ms median_wall=43.83ms, product@0+100 #175 product runs=3 median_simplify=11.37ms median_wire=11.41ms median_wall=43.31ms, sum@0+100 #173 sum runs=3 median_simplify=11.44ms median_wire=11.49ms median_wall=43.20ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.22ms median_wire=10.29ms median_wall=38.98ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.26s | passed=450 failed=0 total=450 avg_case=5.022ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.81s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.91s | passed=1 failed=0 |
