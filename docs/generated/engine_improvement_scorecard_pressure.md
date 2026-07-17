# Engine Improvement Scorecard

- Generated: 2026-07-17T00:22:11.375538+00:00
- Git branch: main
- Git commit: `1857f1f9c818fb2022f4a718ac6250764e80f053`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=979.09ms avg_case_ms=9.79 simplify=275.47ms avg_simplify_ms=2.75, sum total=200 failed=0 elapsed=881.36ms avg_case_ms=4.41 simplify=285.73ms avg_simplify_ms=1.43, product total=100 failed=0 elapsed=606.61ms avg_case_ms=6.07 simplify=174.00ms avg_simplify_ms=1.74, difference total=50 failed=0 elapsed=396.66ms avg_case_ms=7.93 simplify=120.46ms avg_simplify_ms=2.41
- Engine hotspots: sum simplify=285.73ms avg_simplify_ms=1.43 wall=881.36ms, shifted_quotient simplify=275.47ms avg_simplify_ms=2.75 wall=979.09ms, product simplify=174.00ms avg_simplify_ms=1.74 wall=606.61ms, difference simplify=120.46ms avg_simplify_ms=2.41 wall=396.66ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=979.09ms avg_case_ms=9.79 avg_simplify_ms=2.75, sum@0+100 failed=0 elapsed=651.76ms avg_case_ms=6.52 avg_simplify_ms=2.04, product@0+100 failed=0 elapsed=606.61ms avg_case_ms=6.07 avg_simplify_ms=1.74, difference@0+50 failed=0 elapsed=396.66ms avg_case_ms=7.93 avg_simplify_ms=2.41, sum@700+100 failed=0 elapsed=229.61ms avg_case_ms=2.30 avg_simplify_ms=0.82
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.91ms median_wire=16.98ms median_wall=64.78ms, sum@0+100 #173 sum runs=3 median_simplify=15.21ms median_wire=15.30ms median_wall=58.46ms, product@0+100 #175 product runs=3 median_simplify=15.08ms median_wire=15.14ms median_wall=58.75ms, difference@0+50 #174 difference runs=3 median_simplify=14.97ms median_wire=15.02ms median_wall=57.43ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.37ms median_wire=13.45ms median_wall=49.63ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.86s | passed=450 failed=0 total=450 avg_case=6.356ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.48s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.94s | passed=1 failed=0 |
