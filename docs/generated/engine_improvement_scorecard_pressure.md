# Engine Improvement Scorecard

- Generated: 2026-06-13T20:33:36.388498+00:00
- Git branch: main
- Git commit: `741830ec46ebf541d4f2da0c5fbf97311b29217e`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=780.40ms avg_case_ms=7.80 simplify=221.03ms avg_simplify_ms=2.21, sum total=200 failed=0 elapsed=693.30ms avg_case_ms=3.47 simplify=231.81ms avg_simplify_ms=1.16, product total=100 failed=0 elapsed=474.43ms avg_case_ms=4.74 simplify=135.23ms avg_simplify_ms=1.35, difference total=50 failed=0 elapsed=327.25ms avg_case_ms=6.54 simplify=103.89ms avg_simplify_ms=2.08
- Engine hotspots: sum simplify=231.81ms avg_simplify_ms=1.16 wall=693.30ms, shifted_quotient simplify=221.03ms avg_simplify_ms=2.21 wall=780.40ms, product simplify=135.23ms avg_simplify_ms=1.35 wall=474.43ms, difference simplify=103.89ms avg_simplify_ms=2.08 wall=327.25ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=780.40ms avg_case_ms=7.80 avg_simplify_ms=2.21, sum@0+100 failed=0 elapsed=502.27ms avg_case_ms=5.02 avg_simplify_ms=1.61, product@0+100 failed=0 elapsed=474.43ms avg_case_ms=4.74 avg_simplify_ms=1.35, difference@0+50 failed=0 elapsed=327.25ms avg_case_ms=6.54 avg_simplify_ms=2.08, sum@700+100 failed=0 elapsed=191.03ms avg_case_ms=1.91 avg_simplify_ms=0.71
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.94ms median_wire=13.00ms median_wall=49.07ms, product@0+100 #175 product runs=3 median_simplify=11.66ms median_wire=11.70ms median_wall=44.05ms, difference@0+50 #174 difference runs=3 median_simplify=11.57ms median_wire=11.62ms median_wall=43.90ms, sum@0+100 #173 sum runs=3 median_simplify=11.41ms median_wire=11.46ms median_wall=43.48ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.56ms median_wire=10.63ms median_wall=40.50ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.28s | passed=450 failed=0 total=450 avg_case=5.067ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.88s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.91s | passed=1 failed=0 |
