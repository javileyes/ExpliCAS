# Engine Improvement Scorecard

- Generated: 2026-06-24T09:14:10.777736+00:00
- Git branch: main
- Git commit: `3fbc92e2ac24aceaec41927f2f85c0f5c059ad5d`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=784.14ms avg_case_ms=7.84 simplify=224.47ms avg_simplify_ms=2.24, sum total=200 failed=0 elapsed=702.59ms avg_case_ms=3.51 simplify=238.62ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=479.25ms avg_case_ms=4.79 simplify=138.40ms avg_simplify_ms=1.38, difference total=50 failed=0 elapsed=327.19ms avg_case_ms=6.54 simplify=104.50ms avg_simplify_ms=2.09
- Engine hotspots: sum simplify=238.62ms avg_simplify_ms=1.19 wall=702.59ms, shifted_quotient simplify=224.47ms avg_simplify_ms=2.24 wall=784.14ms, product simplify=138.40ms avg_simplify_ms=1.38 wall=479.25ms, difference simplify=104.50ms avg_simplify_ms=2.09 wall=327.19ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=784.14ms avg_case_ms=7.84 avg_simplify_ms=2.24, sum@0+100 failed=0 elapsed=510.08ms avg_case_ms=5.10 avg_simplify_ms=1.66, product@0+100 failed=0 elapsed=479.25ms avg_case_ms=4.79 avg_simplify_ms=1.38, difference@0+50 failed=0 elapsed=327.19ms avg_case_ms=6.54 avg_simplify_ms=2.09, sum@700+100 failed=0 elapsed=192.51ms avg_case_ms=1.93 avg_simplify_ms=0.72
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.84ms median_wire=12.91ms median_wall=49.18ms, sum@0+100 #173 sum runs=3 median_simplify=11.65ms median_wire=11.70ms median_wall=44.21ms, difference@0+50 #174 difference runs=3 median_simplify=11.57ms median_wire=11.63ms median_wall=44.01ms, product@0+100 #175 product runs=3 median_simplify=11.68ms median_wire=11.73ms median_wall=43.98ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.69ms median_wire=10.77ms median_wall=40.47ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.29s | passed=450 failed=0 total=450 avg_case=5.089ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.41s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.04s | passed=1 failed=0 |
