# Engine Improvement Scorecard

- Generated: 2026-06-28T17:56:33.093005+00:00
- Git branch: main
- Git commit: `ce0f53e00e2d1c86ddf68298056ad3f86a93dece`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=804.05ms avg_case_ms=8.04 simplify=231.06ms avg_simplify_ms=2.31, sum total=200 failed=0 elapsed=714.37ms avg_case_ms=3.57 simplify=247.87ms avg_simplify_ms=1.24, product total=100 failed=0 elapsed=491.08ms avg_case_ms=4.91 simplify=144.52ms avg_simplify_ms=1.45, difference total=50 failed=0 elapsed=334.72ms avg_case_ms=6.69 simplify=108.97ms avg_simplify_ms=2.18
- Engine hotspots: sum simplify=247.87ms avg_simplify_ms=1.24 wall=714.37ms, shifted_quotient simplify=231.06ms avg_simplify_ms=2.31 wall=804.05ms, product simplify=144.52ms avg_simplify_ms=1.45 wall=491.08ms, difference simplify=108.97ms avg_simplify_ms=2.18 wall=334.72ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=804.05ms avg_case_ms=8.04 avg_simplify_ms=2.31, sum@0+100 failed=0 elapsed=513.42ms avg_case_ms=5.13 avg_simplify_ms=1.70, product@0+100 failed=0 elapsed=491.08ms avg_case_ms=4.91 avg_simplify_ms=1.45, difference@0+50 failed=0 elapsed=334.72ms avg_case_ms=6.69 avg_simplify_ms=2.18, sum@700+100 failed=0 elapsed=200.96ms avg_case_ms=2.01 avg_simplify_ms=0.77
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.54ms median_wire=13.61ms median_wall=51.20ms, difference@0+50 #174 difference runs=3 median_simplify=12.05ms median_wire=12.10ms median_wall=46.10ms, product@0+100 #175 product runs=3 median_simplify=11.91ms median_wire=11.96ms median_wall=45.49ms, sum@0+100 #173 sum runs=3 median_simplify=11.83ms median_wire=11.88ms median_wall=45.08ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.71ms median_wire=10.77ms median_wall=40.91ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.34s | passed=450 failed=0 total=450 avg_case=5.200ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.48s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.08s | passed=1 failed=0 |
