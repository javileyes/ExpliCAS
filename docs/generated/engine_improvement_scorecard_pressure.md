# Engine Improvement Scorecard

- Generated: 2026-06-16T13:51:01.497049+00:00
- Git branch: main
- Git commit: `7fe6d7ea1e10816842e14db1b769dfe4845e0d8d`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=798.46ms avg_case_ms=7.98 simplify=228.16ms avg_simplify_ms=2.28, sum total=200 failed=0 elapsed=708.00ms avg_case_ms=3.54 simplify=238.00ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=483.22ms avg_case_ms=4.83 simplify=139.09ms avg_simplify_ms=1.39, difference total=50 failed=0 elapsed=332.47ms avg_case_ms=6.65 simplify=105.66ms avg_simplify_ms=2.11
- Engine hotspots: sum simplify=238.00ms avg_simplify_ms=1.19 wall=708.00ms, shifted_quotient simplify=228.16ms avg_simplify_ms=2.28 wall=798.46ms, product simplify=139.09ms avg_simplify_ms=1.39 wall=483.22ms, difference simplify=105.66ms avg_simplify_ms=2.11 wall=332.47ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=798.46ms avg_case_ms=7.98 avg_simplify_ms=2.28, sum@0+100 failed=0 elapsed=512.53ms avg_case_ms=5.13 avg_simplify_ms=1.65, product@0+100 failed=0 elapsed=483.22ms avg_case_ms=4.83 avg_simplify_ms=1.39, difference@0+50 failed=0 elapsed=332.47ms avg_case_ms=6.65 avg_simplify_ms=2.11, sum@700+100 failed=0 elapsed=195.48ms avg_case_ms=1.95 avg_simplify_ms=0.73
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.21ms median_wire=13.28ms median_wall=50.90ms, sum@0+100 #173 sum runs=3 median_simplify=12.09ms median_wire=12.15ms median_wall=46.22ms, product@0+100 #175 product runs=3 median_simplify=11.68ms median_wire=11.73ms median_wall=44.58ms, difference@0+50 #174 difference runs=3 median_simplify=11.81ms median_wire=11.86ms median_wall=44.84ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.89ms median_wire=10.97ms median_wall=41.18ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.32s | passed=450 failed=0 total=450 avg_case=5.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.08s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.01s | passed=1 failed=0 |
