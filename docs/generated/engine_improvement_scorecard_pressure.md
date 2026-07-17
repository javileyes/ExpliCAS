# Engine Improvement Scorecard

- Generated: 2026-07-17T15:25:29.448804+00:00
- Git branch: main
- Git commit: `872c3dcc98e684ae7e6f0ff7fd9b0b1dbd2b192f`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=367

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=996.34ms avg_case_ms=9.96 simplify=281.20ms avg_simplify_ms=2.81, sum total=200 failed=0 elapsed=886.89ms avg_case_ms=4.43 simplify=300.94ms avg_simplify_ms=1.50, product total=100 failed=0 elapsed=608.49ms avg_case_ms=6.08 simplify=175.13ms avg_simplify_ms=1.75, difference total=50 failed=0 elapsed=401.10ms avg_case_ms=8.02 simplify=123.22ms avg_simplify_ms=2.46
- Engine hotspots: sum simplify=300.94ms avg_simplify_ms=1.50 wall=886.89ms, shifted_quotient simplify=281.20ms avg_simplify_ms=2.81 wall=996.34ms, product simplify=175.13ms avg_simplify_ms=1.75 wall=608.49ms, difference simplify=123.22ms avg_simplify_ms=2.46 wall=401.10ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=996.34ms avg_case_ms=9.96 avg_simplify_ms=2.81, sum@0+100 failed=0 elapsed=651.32ms avg_case_ms=6.51 avg_simplify_ms=2.16, product@0+100 failed=0 elapsed=608.49ms avg_case_ms=6.08 avg_simplify_ms=1.75, difference@0+50 failed=0 elapsed=401.10ms avg_case_ms=8.02 avg_simplify_ms=2.46, sum@700+100 failed=0 elapsed=235.57ms avg_case_ms=2.36 avg_simplify_ms=0.84
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.71ms median_wire=16.79ms median_wall=64.15ms, sum@0+100 #173 sum runs=3 median_simplify=15.10ms median_wire=15.14ms median_wall=65.40ms, product@0+100 #175 product runs=3 median_simplify=16.30ms median_wire=16.36ms median_wall=60.54ms, difference@0+50 #174 difference runs=3 median_simplify=16.73ms median_wire=16.78ms median_wall=61.37ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.96ms median_wire=13.04ms median_wall=50.25ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.89s | passed=450 failed=0 total=450 avg_case=6.422ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.57s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.93s | passed=1 failed=0 |
