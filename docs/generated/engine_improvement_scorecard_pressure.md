# Engine Improvement Scorecard

- Generated: 2026-06-16T14:49:29.497672+00:00
- Git branch: main
- Git commit: `1aea8c23c9c0294fcdc58f25824068f5df91ad44`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=787.90ms avg_case_ms=7.88 simplify=226.89ms avg_simplify_ms=2.27, sum total=200 failed=0 elapsed=703.63ms avg_case_ms=3.52 simplify=239.34ms avg_simplify_ms=1.20, product total=100 failed=0 elapsed=500.22ms avg_case_ms=5.00 simplify=145.48ms avg_simplify_ms=1.45, difference total=50 failed=0 elapsed=331.19ms avg_case_ms=6.62 simplify=105.14ms avg_simplify_ms=2.10
- Engine hotspots: sum simplify=239.34ms avg_simplify_ms=1.20 wall=703.63ms, shifted_quotient simplify=226.89ms avg_simplify_ms=2.27 wall=787.90ms, product simplify=145.48ms avg_simplify_ms=1.45 wall=500.22ms, difference simplify=105.14ms avg_simplify_ms=2.10 wall=331.19ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=787.90ms avg_case_ms=7.88 avg_simplify_ms=2.27, sum@0+100 failed=0 elapsed=508.92ms avg_case_ms=5.09 avg_simplify_ms=1.67, product@0+100 failed=0 elapsed=500.22ms avg_case_ms=5.00 avg_simplify_ms=1.45, difference@0+50 failed=0 elapsed=331.19ms avg_case_ms=6.62 avg_simplify_ms=2.10, sum@700+100 failed=0 elapsed=194.72ms avg_case_ms=1.95 avg_simplify_ms=0.73
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.13ms median_wire=13.20ms median_wall=50.18ms, product@0+100 #175 product runs=3 median_simplify=11.62ms median_wire=11.67ms median_wall=45.00ms, sum@0+100 #173 sum runs=3 median_simplify=11.79ms median_wire=11.84ms median_wall=45.03ms, difference@0+50 #174 difference runs=3 median_simplify=12.05ms median_wire=12.10ms median_wall=45.15ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.85ms median_wire=10.93ms median_wall=40.73ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.32s | passed=450 failed=0 total=450 avg_case=5.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.16s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.02s | passed=1 failed=0 |
