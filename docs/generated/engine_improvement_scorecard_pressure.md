# Engine Improvement Scorecard

- Generated: 2026-06-28T17:23:02.798133+00:00
- Git branch: main
- Git commit: `156fbb96f04b7e0a03e829744312679d6423870b`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=809.63ms avg_case_ms=8.10 simplify=232.88ms avg_simplify_ms=2.33, sum total=200 failed=0 elapsed=721.28ms avg_case_ms=3.61 simplify=250.30ms avg_simplify_ms=1.25, product total=100 failed=0 elapsed=500.39ms avg_case_ms=5.00 simplify=147.99ms avg_simplify_ms=1.48, difference total=50 failed=0 elapsed=339.52ms avg_case_ms=6.79 simplify=110.30ms avg_simplify_ms=2.21
- Engine hotspots: sum simplify=250.30ms avg_simplify_ms=1.25 wall=721.28ms, shifted_quotient simplify=232.88ms avg_simplify_ms=2.33 wall=809.63ms, product simplify=147.99ms avg_simplify_ms=1.48 wall=500.39ms, difference simplify=110.30ms avg_simplify_ms=2.21 wall=339.52ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=809.63ms avg_case_ms=8.10 avg_simplify_ms=2.33, sum@0+100 failed=0 elapsed=517.38ms avg_case_ms=5.17 avg_simplify_ms=1.71, product@0+100 failed=0 elapsed=500.39ms avg_case_ms=5.00 avg_simplify_ms=1.48, difference@0+50 failed=0 elapsed=339.52ms avg_case_ms=6.79 avg_simplify_ms=2.21, sum@700+100 failed=0 elapsed=203.90ms avg_case_ms=2.04 avg_simplify_ms=0.79
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.38ms median_wire=13.45ms median_wall=50.97ms, sum@0+100 #173 sum runs=3 median_simplify=11.92ms median_wire=11.97ms median_wall=45.56ms, product@0+100 #175 product runs=3 median_simplify=11.95ms median_wire=12.00ms median_wall=45.65ms, difference@0+50 #174 difference runs=3 median_simplify=11.76ms median_wire=11.81ms median_wall=44.71ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.86ms median_wire=10.93ms median_wall=40.13ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.37s | passed=450 failed=0 total=450 avg_case=5.267ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.47s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.10s | passed=1 failed=0 |
