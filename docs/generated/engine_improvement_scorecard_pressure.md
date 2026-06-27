# Engine Improvement Scorecard

- Generated: 2026-06-27T20:38:16.946859+00:00
- Git branch: main
- Git commit: `bf37b71e0609330b2486a48f068399018f3dfcf4`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=799.07ms avg_case_ms=7.99 simplify=228.24ms avg_simplify_ms=2.28, sum total=200 failed=0 elapsed=722.09ms avg_case_ms=3.61 simplify=251.34ms avg_simplify_ms=1.26, product total=100 failed=0 elapsed=486.21ms avg_case_ms=4.86 simplify=143.46ms avg_simplify_ms=1.43, difference total=50 failed=0 elapsed=331.01ms avg_case_ms=6.62 simplify=106.91ms avg_simplify_ms=2.14
- Engine hotspots: sum simplify=251.34ms avg_simplify_ms=1.26 wall=722.09ms, shifted_quotient simplify=228.24ms avg_simplify_ms=2.28 wall=799.07ms, product simplify=143.46ms avg_simplify_ms=1.43 wall=486.21ms, difference simplify=106.91ms avg_simplify_ms=2.14 wall=331.01ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=799.07ms avg_case_ms=7.99 avg_simplify_ms=2.28, sum@0+100 failed=0 elapsed=523.07ms avg_case_ms=5.23 avg_simplify_ms=1.75, product@0+100 failed=0 elapsed=486.21ms avg_case_ms=4.86 avg_simplify_ms=1.43, difference@0+50 failed=0 elapsed=331.01ms avg_case_ms=6.62 avg_simplify_ms=2.14, sum@700+100 failed=0 elapsed=199.02ms avg_case_ms=1.99 avg_simplify_ms=0.77
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.56ms median_wire=13.63ms median_wall=51.60ms, sum@0+100 #173 sum runs=3 median_simplify=12.07ms median_wire=12.12ms median_wall=46.52ms, product@0+100 #175 product runs=3 median_simplify=11.57ms median_wire=11.63ms median_wall=43.81ms, difference@0+50 #174 difference runs=3 median_simplify=11.89ms median_wire=11.95ms median_wall=45.31ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.98ms median_wire=11.06ms median_wall=40.80ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.34s | passed=450 failed=0 total=450 avg_case=5.200ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.46s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.07s | passed=1 failed=0 |
