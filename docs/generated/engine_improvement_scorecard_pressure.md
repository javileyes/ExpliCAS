# Engine Improvement Scorecard

- Generated: 2026-06-15T10:39:51.985954+00:00
- Git branch: main
- Git commit: `e5c6ab6c49edbf6f3b69ed02b2429d5ea9158d49`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=352

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=834.35ms avg_case_ms=8.34 simplify=240.14ms avg_simplify_ms=2.40, sum total=200 failed=0 elapsed=696.65ms avg_case_ms=3.48 simplify=234.68ms avg_simplify_ms=1.17, product total=100 failed=0 elapsed=525.57ms avg_case_ms=5.26 simplify=151.42ms avg_simplify_ms=1.51, difference total=50 failed=0 elapsed=335.28ms avg_case_ms=6.71 simplify=106.51ms avg_simplify_ms=2.13
- Engine hotspots: shifted_quotient simplify=240.14ms avg_simplify_ms=2.40 wall=834.35ms, sum simplify=234.68ms avg_simplify_ms=1.17 wall=696.65ms, product simplify=151.42ms avg_simplify_ms=1.51 wall=525.57ms, difference simplify=106.51ms avg_simplify_ms=2.13 wall=335.28ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=834.35ms avg_case_ms=8.34 avg_simplify_ms=2.40, product@0+100 failed=0 elapsed=525.57ms avg_case_ms=5.26 avg_simplify_ms=1.51, sum@0+100 failed=0 elapsed=504.31ms avg_case_ms=5.04 avg_simplify_ms=1.63, difference@0+50 failed=0 elapsed=335.28ms avg_case_ms=6.71 avg_simplify_ms=2.13, sum@700+100 failed=0 elapsed=192.33ms avg_case_ms=1.92 avg_simplify_ms=0.72
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.36ms median_wire=13.44ms median_wall=50.48ms, product@0+100 #175 product runs=3 median_simplify=12.01ms median_wire=12.07ms median_wall=45.44ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.84ms median_wire=10.92ms median_wall=41.17ms, difference@0+50 #174 difference runs=3 median_simplify=12.12ms median_wire=12.18ms median_wall=45.46ms, sum@0+100 #173 sum runs=3 median_simplify=11.63ms median_wire=11.68ms median_wall=44.78ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #4 shifted_quotient expr=((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.39s | passed=450 failed=0 total=450 avg_case=5.311ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.86s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.93s | passed=1 failed=0 |
