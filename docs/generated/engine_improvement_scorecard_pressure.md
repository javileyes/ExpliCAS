# Engine Improvement Scorecard

- Generated: 2026-06-14T22:33:36.146978+00:00
- Git branch: main
- Git commit: `4cd1a6477c7baf02e9b8620f786e3ee8535925d7`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=804.78ms avg_case_ms=8.05 simplify=229.46ms avg_simplify_ms=2.29, sum total=200 failed=0 elapsed=700.05ms avg_case_ms=3.50 simplify=234.75ms avg_simplify_ms=1.17, product total=100 failed=0 elapsed=492.28ms avg_case_ms=4.92 simplify=141.65ms avg_simplify_ms=1.42, difference total=50 failed=0 elapsed=344.58ms avg_case_ms=6.89 simplify=109.87ms avg_simplify_ms=2.20
- Engine hotspots: sum simplify=234.75ms avg_simplify_ms=1.17 wall=700.05ms, shifted_quotient simplify=229.46ms avg_simplify_ms=2.29 wall=804.78ms, product simplify=141.65ms avg_simplify_ms=1.42 wall=492.28ms, difference simplify=109.87ms avg_simplify_ms=2.20 wall=344.58ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=804.78ms avg_case_ms=8.05 avg_simplify_ms=2.29, sum@0+100 failed=0 elapsed=504.68ms avg_case_ms=5.05 avg_simplify_ms=1.62, product@0+100 failed=0 elapsed=492.28ms avg_case_ms=4.92 avg_simplify_ms=1.42, difference@0+50 failed=0 elapsed=344.58ms avg_case_ms=6.89 avg_simplify_ms=2.20, sum@700+100 failed=0 elapsed=195.37ms avg_case_ms=1.95 avg_simplify_ms=0.73
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.91ms median_wire=12.99ms median_wall=48.89ms, product@0+100 #175 product runs=3 median_simplify=11.59ms median_wire=11.65ms median_wall=44.51ms, difference@0+50 #174 difference runs=3 median_simplify=11.73ms median_wire=11.78ms median_wall=43.93ms, sum@0+100 #173 sum runs=3 median_simplify=11.63ms median_wire=11.68ms median_wall=43.82ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.82ms median_wire=10.90ms median_wall=41.48ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.34s | passed=450 failed=0 total=450 avg_case=5.200ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.91s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.97s | passed=1 failed=0 |
