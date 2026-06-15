# Engine Improvement Scorecard

- Generated: 2026-06-15T07:03:50.005884+00:00
- Git branch: main
- Git commit: `368e498d076a1cfc2b46b6f718ade841c7461c63`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=351

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=785.31ms avg_case_ms=7.85 simplify=223.37ms avg_simplify_ms=2.23, sum total=200 failed=0 elapsed=697.76ms avg_case_ms=3.49 simplify=235.02ms avg_simplify_ms=1.18, product total=100 failed=0 elapsed=478.13ms avg_case_ms=4.78 simplify=137.51ms avg_simplify_ms=1.38, difference total=50 failed=0 elapsed=330.33ms avg_case_ms=6.61 simplify=105.07ms avg_simplify_ms=2.10
- Engine hotspots: sum simplify=235.02ms avg_simplify_ms=1.18 wall=697.76ms, shifted_quotient simplify=223.37ms avg_simplify_ms=2.23 wall=785.31ms, product simplify=137.51ms avg_simplify_ms=1.38 wall=478.13ms, difference simplify=105.07ms avg_simplify_ms=2.10 wall=330.33ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=785.31ms avg_case_ms=7.85 avg_simplify_ms=2.23, sum@0+100 failed=0 elapsed=506.18ms avg_case_ms=5.06 avg_simplify_ms=1.64, product@0+100 failed=0 elapsed=478.13ms avg_case_ms=4.78 avg_simplify_ms=1.38, difference@0+50 failed=0 elapsed=330.33ms avg_case_ms=6.61 avg_simplify_ms=2.10, sum@700+100 failed=0 elapsed=191.59ms avg_case_ms=1.92 avg_simplify_ms=0.71
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=11.86ms median_wire=11.91ms median_wall=45.06ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.04ms median_wire=13.11ms median_wall=49.90ms, difference@0+50 #174 difference runs=3 median_simplify=11.85ms median_wire=11.90ms median_wall=45.42ms, product@0+100 #175 product runs=3 median_simplify=11.77ms median_wire=11.83ms median_wall=44.37ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.65ms median_wire=10.73ms median_wall=40.71ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.29s | passed=450 failed=0 total=450 avg_case=5.089ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.85s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.93s | passed=1 failed=0 |
