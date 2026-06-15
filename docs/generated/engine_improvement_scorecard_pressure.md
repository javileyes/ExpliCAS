# Engine Improvement Scorecard

- Generated: 2026-06-15T16:02:51.008786+00:00
- Git branch: main
- Git commit: `fa1bcbbfc6111371392f454e18cdb8e1557a0b7a`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=792.25ms avg_case_ms=7.92 simplify=226.23ms avg_simplify_ms=2.26, sum total=200 failed=0 elapsed=698.20ms avg_case_ms=3.49 simplify=233.75ms avg_simplify_ms=1.17, product total=100 failed=0 elapsed=481.15ms avg_case_ms=4.81 simplify=137.79ms avg_simplify_ms=1.38, difference total=50 failed=0 elapsed=328.77ms avg_case_ms=6.58 simplify=104.00ms avg_simplify_ms=2.08
- Engine hotspots: sum simplify=233.75ms avg_simplify_ms=1.17 wall=698.20ms, shifted_quotient simplify=226.23ms avg_simplify_ms=2.26 wall=792.25ms, product simplify=137.79ms avg_simplify_ms=1.38 wall=481.15ms, difference simplify=104.00ms avg_simplify_ms=2.08 wall=328.77ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=792.25ms avg_case_ms=7.92 avg_simplify_ms=2.26, sum@0+100 failed=0 elapsed=504.49ms avg_case_ms=5.04 avg_simplify_ms=1.62, product@0+100 failed=0 elapsed=481.15ms avg_case_ms=4.81 avg_simplify_ms=1.38, difference@0+50 failed=0 elapsed=328.77ms avg_case_ms=6.58 avg_simplify_ms=2.08, sum@700+100 failed=0 elapsed=193.71ms avg_case_ms=1.94 avg_simplify_ms=0.72
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.29ms median_wire=13.36ms median_wall=49.76ms, sum@0+100 #173 sum runs=3 median_simplify=11.88ms median_wire=11.93ms median_wall=44.50ms, difference@0+50 #174 difference runs=3 median_simplify=11.62ms median_wire=11.67ms median_wall=44.05ms, product@0+100 #175 product runs=3 median_simplify=11.63ms median_wire=11.68ms median_wall=44.29ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.64ms median_wire=10.71ms median_wall=40.10ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.30s | passed=450 failed=0 total=450 avg_case=5.111ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.84s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.93s | passed=1 failed=0 |
