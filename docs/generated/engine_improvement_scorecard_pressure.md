# Engine Improvement Scorecard

- Generated: 2026-06-16T08:11:47.820604+00:00
- Git branch: main
- Git commit: `d112d9177e6d06c5a139d7a308090f640b726e27`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=801.30ms avg_case_ms=8.01 simplify=228.33ms avg_simplify_ms=2.28, sum total=200 failed=0 elapsed=696.48ms avg_case_ms=3.48 simplify=233.21ms avg_simplify_ms=1.17, product total=100 failed=0 elapsed=484.21ms avg_case_ms=4.84 simplify=140.43ms avg_simplify_ms=1.40, difference total=50 failed=0 elapsed=332.21ms avg_case_ms=6.64 simplify=105.91ms avg_simplify_ms=2.12
- Engine hotspots: sum simplify=233.21ms avg_simplify_ms=1.17 wall=696.48ms, shifted_quotient simplify=228.33ms avg_simplify_ms=2.28 wall=801.30ms, product simplify=140.43ms avg_simplify_ms=1.40 wall=484.21ms, difference simplify=105.91ms avg_simplify_ms=2.12 wall=332.21ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=801.30ms avg_case_ms=8.01 avg_simplify_ms=2.28, sum@0+100 failed=0 elapsed=502.16ms avg_case_ms=5.02 avg_simplify_ms=1.61, product@0+100 failed=0 elapsed=484.21ms avg_case_ms=4.84 avg_simplify_ms=1.40, difference@0+50 failed=0 elapsed=332.21ms avg_case_ms=6.64 avg_simplify_ms=2.12, sum@700+100 failed=0 elapsed=194.32ms avg_case_ms=1.94 avg_simplify_ms=0.72
- Steady-state engine reruns: product@0+100 #175 product runs=3 median_simplify=11.98ms median_wire=12.03ms median_wall=45.45ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.45ms median_wire=13.53ms median_wall=50.83ms, difference@0+50 #174 difference runs=3 median_simplify=11.71ms median_wire=11.76ms median_wall=44.61ms, sum@0+100 #173 sum runs=3 median_simplify=11.97ms median_wire=12.02ms median_wall=45.06ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.68ms median_wire=10.75ms median_wall=40.58ms
- Steady-state dominant expressions: product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.32s | passed=450 failed=0 total=450 avg_case=5.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.86s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.94s | passed=1 failed=0 |
