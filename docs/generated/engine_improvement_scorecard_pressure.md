# Engine Improvement Scorecard

- Generated: 2026-06-11T06:15:33.051512+00:00
- Git branch: main
- Git commit: `955a54bcf1dca1bbc95be07d5ee6a3098f83ee33`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=801.78ms avg_case_ms=8.02 simplify=227.36ms avg_simplify_ms=2.27, sum total=200 failed=0 elapsed=707.11ms avg_case_ms=3.54 simplify=234.42ms avg_simplify_ms=1.17, product total=100 failed=0 elapsed=485.73ms avg_case_ms=4.86 simplify=139.70ms avg_simplify_ms=1.40, difference total=50 failed=0 elapsed=323.77ms avg_case_ms=6.48 simplify=102.54ms avg_simplify_ms=2.05
- Engine hotspots: sum simplify=234.42ms avg_simplify_ms=1.17 wall=707.11ms, shifted_quotient simplify=227.36ms avg_simplify_ms=2.27 wall=801.78ms, product simplify=139.70ms avg_simplify_ms=1.40 wall=485.73ms, difference simplify=102.54ms avg_simplify_ms=2.05 wall=323.77ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=801.78ms avg_case_ms=8.02 avg_simplify_ms=2.27, sum@0+100 failed=0 elapsed=516.52ms avg_case_ms=5.17 avg_simplify_ms=1.63, product@0+100 failed=0 elapsed=485.73ms avg_case_ms=4.86 avg_simplify_ms=1.40, difference@0+50 failed=0 elapsed=323.77ms avg_case_ms=6.48 avg_simplify_ms=2.05, sum@700+100 failed=0 elapsed=190.59ms avg_case_ms=1.91 avg_simplify_ms=0.71
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.73ms median_wire=13.80ms median_wall=50.60ms, product@0+100 #175 product runs=3 median_simplify=11.57ms median_wire=11.62ms median_wall=44.79ms, difference@0+50 #174 difference runs=3 median_simplify=11.61ms median_wire=11.66ms median_wall=43.82ms, sum@0+100 #173 sum runs=3 median_simplify=11.65ms median_wire=11.70ms median_wall=44.15ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=11.34ms median_wire=11.42ms median_wall=42.78ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.32s | passed=450 failed=0 total=450 avg_case=5.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.81s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.16s | passed=1 failed=0 |
