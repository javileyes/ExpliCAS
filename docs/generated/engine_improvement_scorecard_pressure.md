# Engine Improvement Scorecard

- Generated: 2026-06-16T15:49:34.393155+00:00
- Git branch: main
- Git commit: `8271ddb3850259f3bfe33d50afbb9f8cf170c152`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=835.19ms avg_case_ms=8.35 simplify=240.66ms avg_simplify_ms=2.41, sum total=200 failed=0 elapsed=705.18ms avg_case_ms=3.53 simplify=239.35ms avg_simplify_ms=1.20, product total=100 failed=0 elapsed=545.59ms avg_case_ms=5.46 simplify=158.24ms avg_simplify_ms=1.58, difference total=50 failed=0 elapsed=330.42ms avg_case_ms=6.61 simplify=105.84ms avg_simplify_ms=2.12
- Engine hotspots: shifted_quotient simplify=240.66ms avg_simplify_ms=2.41 wall=835.19ms, sum simplify=239.35ms avg_simplify_ms=1.20 wall=705.18ms, product simplify=158.24ms avg_simplify_ms=1.58 wall=545.59ms, difference simplify=105.84ms avg_simplify_ms=2.12 wall=330.42ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=835.19ms avg_case_ms=8.35 avg_simplify_ms=2.41, product@0+100 failed=0 elapsed=545.59ms avg_case_ms=5.46 avg_simplify_ms=1.58, sum@0+100 failed=0 elapsed=509.20ms avg_case_ms=5.09 avg_simplify_ms=1.66, difference@0+50 failed=0 elapsed=330.42ms avg_case_ms=6.61 avg_simplify_ms=2.12, sum@700+100 failed=0 elapsed=195.98ms avg_case_ms=1.96 avg_simplify_ms=0.73
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.08ms median_wire=13.15ms median_wall=50.34ms, product@0+100 #175 product runs=3 median_simplify=11.79ms median_wire=11.84ms median_wall=45.31ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.86ms median_wire=10.93ms median_wall=41.21ms, sum@0+100 #173 sum runs=3 median_simplify=11.91ms median_wire=11.97ms median_wall=45.47ms, difference@0+50 #174 difference runs=3 median_simplify=12.26ms median_wire=12.31ms median_wall=45.91ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #4 shifted_quotient expr=((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.42s | passed=450 failed=0 total=450 avg_case=5.378ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.44s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.04s | passed=1 failed=0 |
