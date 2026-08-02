# Engine Improvement Scorecard

- Generated: 2026-08-02T08:12:29.156841+00:00
- Git branch: main
- Git commit: `7d88b5351e983f194d7cc17c451507479065e27f`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=391

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=999.64ms avg_case_ms=10.00 simplify=282.84ms avg_simplify_ms=2.83, sum total=200 failed=0 elapsed=859.69ms avg_case_ms=4.30 simplify=277.43ms avg_simplify_ms=1.39, product total=100 failed=0 elapsed=612.56ms avg_case_ms=6.13 simplify=177.95ms avg_simplify_ms=1.78, difference total=50 failed=0 elapsed=402.09ms avg_case_ms=8.04 simplify=118.20ms avg_simplify_ms=2.36
- Engine hotspots: shifted_quotient simplify=282.84ms avg_simplify_ms=2.83 wall=999.64ms, sum simplify=277.43ms avg_simplify_ms=1.39 wall=859.69ms, product simplify=177.95ms avg_simplify_ms=1.78 wall=612.56ms, difference simplify=118.20ms avg_simplify_ms=2.36 wall=402.09ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=999.64ms avg_case_ms=10.00 avg_simplify_ms=2.83, sum@0+100 failed=0 elapsed=624.79ms avg_case_ms=6.25 avg_simplify_ms=1.93, product@0+100 failed=0 elapsed=612.56ms avg_case_ms=6.13 avg_simplify_ms=1.78, difference@0+50 failed=0 elapsed=402.09ms avg_case_ms=8.04 avg_simplify_ms=2.36, sum@700+100 failed=0 elapsed=234.90ms avg_case_ms=2.35 avg_simplify_ms=0.85
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.52ms median_wire=17.59ms median_wall=66.37ms, product@0+100 #175 product runs=3 median_simplify=15.12ms median_wire=15.18ms median_wall=58.55ms, difference@0+50 #174 difference runs=3 median_simplify=15.49ms median_wire=15.54ms median_wall=59.26ms, sum@0+100 #173 sum runs=3 median_simplify=15.19ms median_wire=15.25ms median_wall=58.47ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.17ms median_wire=13.25ms median_wall=49.52ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.87s | passed=450 failed=0 total=450 avg_case=6.378ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.36s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.77s | passed=1 failed=0 |
