# Engine Improvement Scorecard

- Generated: 2026-06-26T21:25:50.094241+00:00
- Git branch: main
- Git commit: `b7c11bcd18404e73a3d4c66ddc16525319e6286c`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=816.78ms avg_case_ms=8.17 simplify=237.20ms avg_simplify_ms=2.37, sum total=200 failed=0 elapsed=713.55ms avg_case_ms=3.57 simplify=246.92ms avg_simplify_ms=1.23, product total=100 failed=0 elapsed=486.24ms avg_case_ms=4.86 simplify=143.00ms avg_simplify_ms=1.43, difference total=50 failed=0 elapsed=336.01ms avg_case_ms=6.72 simplify=108.23ms avg_simplify_ms=2.16
- Engine hotspots: sum simplify=246.92ms avg_simplify_ms=1.23 wall=713.55ms, shifted_quotient simplify=237.20ms avg_simplify_ms=2.37 wall=816.78ms, product simplify=143.00ms avg_simplify_ms=1.43 wall=486.24ms, difference simplify=108.23ms avg_simplify_ms=2.16 wall=336.01ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=816.78ms avg_case_ms=8.17 avg_simplify_ms=2.37, sum@0+100 failed=0 elapsed=512.70ms avg_case_ms=5.13 avg_simplify_ms=1.69, product@0+100 failed=0 elapsed=486.24ms avg_case_ms=4.86 avg_simplify_ms=1.43, difference@0+50 failed=0 elapsed=336.01ms avg_case_ms=6.72 avg_simplify_ms=2.16, sum@700+100 failed=0 elapsed=200.85ms avg_case_ms=2.01 avg_simplify_ms=0.78
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.23ms median_wire=13.30ms median_wall=49.99ms, difference@0+50 #174 difference runs=3 median_simplify=11.75ms median_wire=11.80ms median_wall=44.67ms, sum@0+100 #173 sum runs=3 median_simplify=11.78ms median_wire=11.83ms median_wall=44.72ms, product@0+100 #175 product runs=3 median_simplify=11.72ms median_wire=11.77ms median_wall=44.79ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.96ms median_wire=11.04ms median_wall=41.11ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.35s | passed=450 failed=0 total=450 avg_case=5.222ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.49s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.10s | passed=1 failed=0 |
