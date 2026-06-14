# Engine Improvement Scorecard

- Generated: 2026-06-14T12:15:57.222373+00:00
- Git branch: main
- Git commit: `d87e106d7f51481d5b69a1ba322f6ffa585dd3e3`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=346

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=790.00ms avg_case_ms=7.90 simplify=223.75ms avg_simplify_ms=2.24, sum total=200 failed=0 elapsed=679.38ms avg_case_ms=3.40 simplify=226.66ms avg_simplify_ms=1.13, product total=100 failed=0 elapsed=471.67ms avg_case_ms=4.72 simplify=135.13ms avg_simplify_ms=1.35, difference total=50 failed=0 elapsed=320.82ms avg_case_ms=6.42 simplify=102.08ms avg_simplify_ms=2.04
- Engine hotspots: sum simplify=226.66ms avg_simplify_ms=1.13 wall=679.38ms, shifted_quotient simplify=223.75ms avg_simplify_ms=2.24 wall=790.00ms, product simplify=135.13ms avg_simplify_ms=1.35 wall=471.67ms, difference simplify=102.08ms avg_simplify_ms=2.04 wall=320.82ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=790.00ms avg_case_ms=7.90 avg_simplify_ms=2.24, sum@0+100 failed=0 elapsed=489.97ms avg_case_ms=4.90 avg_simplify_ms=1.57, product@0+100 failed=0 elapsed=471.67ms avg_case_ms=4.72 avg_simplify_ms=1.35, difference@0+50 failed=0 elapsed=320.82ms avg_case_ms=6.42 avg_simplify_ms=2.04, sum@700+100 failed=0 elapsed=189.41ms avg_case_ms=1.89 avg_simplify_ms=0.70
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.66ms median_wire=12.73ms median_wall=48.00ms, difference@0+50 #174 difference runs=3 median_simplify=11.61ms median_wire=11.66ms median_wall=43.70ms, product@0+100 #175 product runs=3 median_simplify=11.19ms median_wire=11.24ms median_wall=42.87ms, sum@0+100 #173 sum runs=3 median_simplify=11.41ms median_wire=11.46ms median_wall=43.50ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.57ms median_wire=10.64ms median_wall=40.03ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.26s | passed=450 failed=0 total=450 avg_case=5.022ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.83s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.91s | passed=1 failed=0 |
