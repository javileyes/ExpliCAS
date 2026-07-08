# Engine Improvement Scorecard

- Generated: 2026-07-08T19:07:16.920506+00:00
- Git branch: main
- Git commit: `7650cb4678ffd143bed2aac9ebc2ef5659808550`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=357

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=922.67ms avg_case_ms=9.23 simplify=254.81ms avg_simplify_ms=2.55, sum total=200 failed=0 elapsed=814.34ms avg_case_ms=4.07 simplify=261.65ms avg_simplify_ms=1.31, product total=100 failed=0 elapsed=569.43ms avg_case_ms=5.69 simplify=160.96ms avg_simplify_ms=1.61, difference total=50 failed=0 elapsed=379.00ms avg_case_ms=7.58 simplify=114.34ms avg_simplify_ms=2.29
- Engine hotspots: sum simplify=261.65ms avg_simplify_ms=1.31 wall=814.34ms, shifted_quotient simplify=254.81ms avg_simplify_ms=2.55 wall=922.67ms, product simplify=160.96ms avg_simplify_ms=1.61 wall=569.43ms, difference simplify=114.34ms avg_simplify_ms=2.29 wall=379.00ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=922.67ms avg_case_ms=9.23 avg_simplify_ms=2.55, sum@0+100 failed=0 elapsed=596.28ms avg_case_ms=5.96 avg_simplify_ms=1.84, product@0+100 failed=0 elapsed=569.43ms avg_case_ms=5.69 avg_simplify_ms=1.61, difference@0+50 failed=0 elapsed=379.00ms avg_case_ms=7.58 avg_simplify_ms=2.29, sum@700+100 failed=0 elapsed=218.06ms avg_case_ms=2.18 avg_simplify_ms=0.77
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.40ms median_wire=16.46ms median_wall=62.63ms, difference@0+50 #174 difference runs=3 median_simplify=14.74ms median_wire=14.78ms median_wall=55.97ms, sum@0+100 #173 sum runs=3 median_simplify=14.64ms median_wire=14.70ms median_wall=55.67ms, product@0+100 #175 product runs=3 median_simplify=14.43ms median_wire=14.47ms median_wall=55.76ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.24ms median_wire=12.31ms median_wall=46.79ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.69s | passed=450 failed=0 total=450 avg_case=5.978ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.93s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.85s | passed=1 failed=0 |
