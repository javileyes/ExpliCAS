# Engine Improvement Scorecard

- Generated: 2026-06-12T08:00:09.478702+00:00
- Git branch: main
- Git commit: `2215dfd9eb05bd08f3c025793f8ba787b86798b8`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=770.82ms avg_case_ms=7.71 simplify=218.88ms avg_simplify_ms=2.19, sum total=200 failed=0 elapsed=682.89ms avg_case_ms=3.41 simplify=229.55ms avg_simplify_ms=1.15, product total=100 failed=0 elapsed=471.00ms avg_case_ms=4.71 simplify=135.14ms avg_simplify_ms=1.35, difference total=50 failed=0 elapsed=325.63ms avg_case_ms=6.51 simplify=103.77ms avg_simplify_ms=2.08
- Engine hotspots: sum simplify=229.55ms avg_simplify_ms=1.15 wall=682.89ms, shifted_quotient simplify=218.88ms avg_simplify_ms=2.19 wall=770.82ms, product simplify=135.14ms avg_simplify_ms=1.35 wall=471.00ms, difference simplify=103.77ms avg_simplify_ms=2.08 wall=325.63ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=770.82ms avg_case_ms=7.71 avg_simplify_ms=2.19, sum@0+100 failed=0 elapsed=492.69ms avg_case_ms=4.93 avg_simplify_ms=1.59, product@0+100 failed=0 elapsed=471.00ms avg_case_ms=4.71 avg_simplify_ms=1.35, difference@0+50 failed=0 elapsed=325.63ms avg_case_ms=6.51 avg_simplify_ms=2.08, sum@700+100 failed=0 elapsed=190.20ms avg_case_ms=1.90 avg_simplify_ms=0.71
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.68ms median_wire=12.74ms median_wall=48.82ms, difference@0+50 #174 difference runs=3 median_simplify=11.51ms median_wire=11.56ms median_wall=44.03ms, product@0+100 #175 product runs=3 median_simplify=11.53ms median_wire=11.58ms median_wall=44.01ms, sum@0+100 #173 sum runs=3 median_simplify=11.39ms median_wire=11.44ms median_wall=43.42ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.21ms median_wire=10.28ms median_wall=38.98ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.25s | passed=450 failed=0 total=450 avg_case=5.000ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.80s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.12s | passed=1 failed=0 |
