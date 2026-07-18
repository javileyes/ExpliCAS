# Engine Improvement Scorecard

- Generated: 2026-07-18T05:37:12.955195+00:00
- Git branch: main
- Git commit: `b502c5c092a66c026f7c5bca82309853bc754823`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=963.14ms avg_case_ms=9.63 simplify=268.88ms avg_simplify_ms=2.69, sum total=200 failed=0 elapsed=885.02ms avg_case_ms=4.43 simplify=282.88ms avg_simplify_ms=1.41, product total=100 failed=0 elapsed=604.91ms avg_case_ms=6.05 simplify=173.63ms avg_simplify_ms=1.74, difference total=50 failed=0 elapsed=399.37ms avg_case_ms=7.99 simplify=121.53ms avg_simplify_ms=2.43
- Engine hotspots: sum simplify=282.88ms avg_simplify_ms=1.41 wall=885.02ms, shifted_quotient simplify=268.88ms avg_simplify_ms=2.69 wall=963.14ms, product simplify=173.63ms avg_simplify_ms=1.74 wall=604.91ms, difference simplify=121.53ms avg_simplify_ms=2.43 wall=399.37ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=963.14ms avg_case_ms=9.63 avg_simplify_ms=2.69, sum@0+100 failed=0 elapsed=651.70ms avg_case_ms=6.52 avg_simplify_ms=1.99, product@0+100 failed=0 elapsed=604.91ms avg_case_ms=6.05 avg_simplify_ms=1.74, difference@0+50 failed=0 elapsed=399.37ms avg_case_ms=7.99 avg_simplify_ms=2.43, sum@700+100 failed=0 elapsed=233.32ms avg_case_ms=2.33 avg_simplify_ms=0.84
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.65ms median_wire=16.72ms median_wall=64.14ms, sum@0+100 #173 sum runs=3 median_simplify=15.36ms median_wire=15.41ms median_wall=59.08ms, difference@0+50 #174 difference runs=3 median_simplify=15.46ms median_wire=15.51ms median_wall=57.95ms, product@0+100 #175 product runs=3 median_simplify=15.18ms median_wire=15.24ms median_wall=59.52ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.08ms median_wire=13.15ms median_wall=50.04ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.85s | passed=450 failed=0 total=450 avg_case=6.333ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.50s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.93s | passed=1 failed=0 |
