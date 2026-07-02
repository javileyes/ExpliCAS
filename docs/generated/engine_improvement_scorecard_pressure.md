# Engine Improvement Scorecard

- Generated: 2026-07-02T07:59:28.779653+00:00
- Git branch: main
- Git commit: `762d973bd0b381b01740c6cc68c1d76f959ac534`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=355

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=802.51ms avg_case_ms=8.03 simplify=232.32ms avg_simplify_ms=2.32, sum total=200 failed=0 elapsed=712.54ms avg_case_ms=3.56 simplify=246.69ms avg_simplify_ms=1.23, product total=100 failed=0 elapsed=497.91ms avg_case_ms=4.98 simplify=148.59ms avg_simplify_ms=1.49, difference total=50 failed=0 elapsed=325.99ms avg_case_ms=6.52 simplify=105.47ms avg_simplify_ms=2.11
- Engine hotspots: sum simplify=246.69ms avg_simplify_ms=1.23 wall=712.54ms, shifted_quotient simplify=232.32ms avg_simplify_ms=2.32 wall=802.51ms, product simplify=148.59ms avg_simplify_ms=1.49 wall=497.91ms, difference simplify=105.47ms avg_simplify_ms=2.11 wall=325.99ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=802.51ms avg_case_ms=8.03 avg_simplify_ms=2.32, sum@0+100 failed=0 elapsed=518.02ms avg_case_ms=5.18 avg_simplify_ms=1.72, product@0+100 failed=0 elapsed=497.91ms avg_case_ms=4.98 avg_simplify_ms=1.49, difference@0+50 failed=0 elapsed=325.99ms avg_case_ms=6.52 avg_simplify_ms=2.11, sum@700+100 failed=0 elapsed=194.52ms avg_case_ms=1.95 avg_simplify_ms=0.75
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.96ms median_wire=13.03ms median_wall=49.43ms, product@0+100 #175 product runs=3 median_simplify=11.64ms median_wire=11.69ms median_wall=44.30ms, difference@0+50 #174 difference runs=3 median_simplify=11.53ms median_wire=11.57ms median_wall=43.76ms, sum@0+100 #173 sum runs=3 median_simplify=11.51ms median_wire=11.56ms median_wall=43.73ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=11.00ms median_wire=11.07ms median_wall=41.33ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.34s | passed=450 failed=0 total=450 avg_case=5.200ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.44s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.04s | passed=1 failed=0 |
