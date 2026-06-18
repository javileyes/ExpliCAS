# Engine Improvement Scorecard

- Generated: 2026-06-18T22:11:04.062600+00:00
- Git branch: main
- Git commit: `5aac08ed885fe439a9d85ae3c360f6f1cc14a0c2`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=353

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=812.45ms avg_case_ms=8.12 simplify=234.38ms avg_simplify_ms=2.34, sum total=200 failed=0 elapsed=704.51ms avg_case_ms=3.52 simplify=239.76ms avg_simplify_ms=1.20, product total=100 failed=0 elapsed=501.67ms avg_case_ms=5.02 simplify=145.69ms avg_simplify_ms=1.46, difference total=50 failed=0 elapsed=333.45ms avg_case_ms=6.67 simplify=106.15ms avg_simplify_ms=2.12
- Engine hotspots: sum simplify=239.76ms avg_simplify_ms=1.20 wall=704.51ms, shifted_quotient simplify=234.38ms avg_simplify_ms=2.34 wall=812.45ms, product simplify=145.69ms avg_simplify_ms=1.46 wall=501.67ms, difference simplify=106.15ms avg_simplify_ms=2.12 wall=333.45ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=812.45ms avg_case_ms=8.12 avg_simplify_ms=2.34, sum@0+100 failed=0 elapsed=508.20ms avg_case_ms=5.08 avg_simplify_ms=1.66, product@0+100 failed=0 elapsed=501.67ms avg_case_ms=5.02 avg_simplify_ms=1.46, difference@0+50 failed=0 elapsed=333.45ms avg_case_ms=6.67 avg_simplify_ms=2.12, sum@700+100 failed=0 elapsed=196.31ms avg_case_ms=1.96 avg_simplify_ms=0.74
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.42ms median_wire=13.49ms median_wall=50.85ms, difference@0+50 #174 difference runs=3 median_simplify=12.01ms median_wire=12.06ms median_wall=46.06ms, sum@0+100 #173 sum runs=3 median_simplify=12.19ms median_wire=12.24ms median_wall=46.11ms, product@0+100 #175 product runs=3 median_simplify=11.82ms median_wire=11.87ms median_wall=45.24ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.92ms median_wire=10.99ms median_wall=41.08ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.35s | passed=450 failed=0 total=450 avg_case=5.222ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.48s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.06s | passed=1 failed=0 |
