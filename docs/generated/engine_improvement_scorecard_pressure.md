# Engine Improvement Scorecard

- Generated: 2026-06-14T12:38:24.554987+00:00
- Git branch: main
- Git commit: `0a06ab50f8203ad16196060cf06586f18b26cd7e`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=347

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=778.81ms avg_case_ms=7.79 simplify=221.12ms avg_simplify_ms=2.21, sum total=200 failed=0 elapsed=681.83ms avg_case_ms=3.41 simplify=229.68ms avg_simplify_ms=1.15, product total=100 failed=0 elapsed=471.91ms avg_case_ms=4.72 simplify=135.10ms avg_simplify_ms=1.35, difference total=50 failed=0 elapsed=323.85ms avg_case_ms=6.48 simplify=101.86ms avg_simplify_ms=2.04
- Engine hotspots: sum simplify=229.68ms avg_simplify_ms=1.15 wall=681.83ms, shifted_quotient simplify=221.12ms avg_simplify_ms=2.21 wall=778.81ms, product simplify=135.10ms avg_simplify_ms=1.35 wall=471.91ms, difference simplify=101.86ms avg_simplify_ms=2.04 wall=323.85ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=778.81ms avg_case_ms=7.79 avg_simplify_ms=2.21, sum@0+100 failed=0 elapsed=492.56ms avg_case_ms=4.93 avg_simplify_ms=1.59, product@0+100 failed=0 elapsed=471.91ms avg_case_ms=4.72 avg_simplify_ms=1.35, difference@0+50 failed=0 elapsed=323.85ms avg_case_ms=6.48 avg_simplify_ms=2.04, sum@700+100 failed=0 elapsed=189.27ms avg_case_ms=1.89 avg_simplify_ms=0.71
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.83ms median_wire=12.90ms median_wall=48.36ms, product@0+100 #175 product runs=3 median_simplify=11.42ms median_wire=11.46ms median_wall=43.65ms, sum@0+100 #173 sum runs=3 median_simplify=11.41ms median_wire=11.46ms median_wall=43.76ms, difference@0+50 #174 difference runs=3 median_simplify=11.60ms median_wire=11.65ms median_wall=44.59ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.53ms median_wire=10.60ms median_wall=39.92ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.26s | passed=450 failed=0 total=450 avg_case=5.022ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.80s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
