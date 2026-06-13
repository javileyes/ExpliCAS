# Engine Improvement Scorecard

- Generated: 2026-06-13T15:44:25.945583+00:00
- Git branch: main
- Git commit: `c1a49128559e5897227614abd9f1ea2d817ca9d4`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=777.01ms avg_case_ms=7.77 simplify=220.54ms avg_simplify_ms=2.21, sum total=200 failed=0 elapsed=697.62ms avg_case_ms=3.49 simplify=235.04ms avg_simplify_ms=1.18, product total=100 failed=0 elapsed=471.57ms avg_case_ms=4.72 simplify=134.99ms avg_simplify_ms=1.35, difference total=50 failed=0 elapsed=332.02ms avg_case_ms=6.64 simplify=104.88ms avg_simplify_ms=2.10
- Engine hotspots: sum simplify=235.04ms avg_simplify_ms=1.18 wall=697.62ms, shifted_quotient simplify=220.54ms avg_simplify_ms=2.21 wall=777.01ms, product simplify=134.99ms avg_simplify_ms=1.35 wall=471.57ms, difference simplify=104.88ms avg_simplify_ms=2.10 wall=332.02ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=777.01ms avg_case_ms=7.77 avg_simplify_ms=2.21, sum@0+100 failed=0 elapsed=501.56ms avg_case_ms=5.02 avg_simplify_ms=1.62, product@0+100 failed=0 elapsed=471.57ms avg_case_ms=4.72 avg_simplify_ms=1.35, difference@0+50 failed=0 elapsed=332.02ms avg_case_ms=6.64 avg_simplify_ms=2.10, sum@700+100 failed=0 elapsed=196.05ms avg_case_ms=1.96 avg_simplify_ms=0.73
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.08ms median_wire=13.15ms median_wall=50.51ms, sum@0+100 #173 sum runs=3 median_simplify=11.49ms median_wire=11.54ms median_wall=44.00ms, product@0+100 #175 product runs=3 median_simplify=11.53ms median_wire=11.58ms median_wall=43.63ms, difference@0+50 #174 difference runs=3 median_simplify=11.53ms median_wire=11.58ms median_wall=43.01ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.37ms median_wire=10.44ms median_wall=39.44ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.28s | passed=450 failed=0 total=450 avg_case=5.067ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.83s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.91s | passed=1 failed=0 |
