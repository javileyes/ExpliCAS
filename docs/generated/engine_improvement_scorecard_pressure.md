# Engine Improvement Scorecard

- Generated: 2026-06-26T00:20:57.872117+00:00
- Git branch: main
- Git commit: `c7cb728ea09fbf128abca6def339a9a535a4f496`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=796.54ms avg_case_ms=7.97 simplify=228.00ms avg_simplify_ms=2.28, sum total=200 failed=0 elapsed=708.24ms avg_case_ms=3.54 simplify=244.42ms avg_simplify_ms=1.22, product total=100 failed=0 elapsed=486.20ms avg_case_ms=4.86 simplify=142.42ms avg_simplify_ms=1.42, difference total=50 failed=0 elapsed=332.96ms avg_case_ms=6.66 simplify=107.21ms avg_simplify_ms=2.14
- Engine hotspots: sum simplify=244.42ms avg_simplify_ms=1.22 wall=708.24ms, shifted_quotient simplify=228.00ms avg_simplify_ms=2.28 wall=796.54ms, product simplify=142.42ms avg_simplify_ms=1.42 wall=486.20ms, difference simplify=107.21ms avg_simplify_ms=2.14 wall=332.96ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=796.54ms avg_case_ms=7.97 avg_simplify_ms=2.28, sum@0+100 failed=0 elapsed=510.98ms avg_case_ms=5.11 avg_simplify_ms=1.68, product@0+100 failed=0 elapsed=486.20ms avg_case_ms=4.86 avg_simplify_ms=1.42, difference@0+50 failed=0 elapsed=332.96ms avg_case_ms=6.66 avg_simplify_ms=2.14, sum@700+100 failed=0 elapsed=197.25ms avg_case_ms=1.97 avg_simplify_ms=0.77
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.19ms median_wire=13.26ms median_wall=50.34ms, difference@0+50 #174 difference runs=3 median_simplify=11.84ms median_wire=11.89ms median_wall=44.26ms, product@0+100 #175 product runs=3 median_simplify=11.61ms median_wire=11.66ms median_wall=44.38ms, sum@0+100 #173 sum runs=3 median_simplify=11.84ms median_wire=11.89ms median_wall=44.90ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.90ms median_wire=10.97ms median_wall=41.17ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.32s | passed=450 failed=0 total=450 avg_case=5.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.45s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.05s | passed=1 failed=0 |
