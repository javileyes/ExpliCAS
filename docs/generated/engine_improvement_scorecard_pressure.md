# Engine Improvement Scorecard

- Generated: 2026-06-27T15:01:44.557656+00:00
- Git branch: main
- Git commit: `b1e4ea790efa9bb50aa9c62bf09def0f2ec93422`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=794.16ms avg_case_ms=7.94 simplify=227.21ms avg_simplify_ms=2.27, sum total=200 failed=0 elapsed=714.26ms avg_case_ms=3.57 simplify=246.55ms avg_simplify_ms=1.23, product total=100 failed=0 elapsed=485.71ms avg_case_ms=4.86 simplify=142.03ms avg_simplify_ms=1.42, difference total=50 failed=0 elapsed=329.01ms avg_case_ms=6.58 simplify=105.92ms avg_simplify_ms=2.12
- Engine hotspots: sum simplify=246.55ms avg_simplify_ms=1.23 wall=714.26ms, shifted_quotient simplify=227.21ms avg_simplify_ms=2.27 wall=794.16ms, product simplify=142.03ms avg_simplify_ms=1.42 wall=485.71ms, difference simplify=105.92ms avg_simplify_ms=2.12 wall=329.01ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=794.16ms avg_case_ms=7.94 avg_simplify_ms=2.27, sum@0+100 failed=0 elapsed=513.43ms avg_case_ms=5.13 avg_simplify_ms=1.69, product@0+100 failed=0 elapsed=485.71ms avg_case_ms=4.86 avg_simplify_ms=1.42, difference@0+50 failed=0 elapsed=329.01ms avg_case_ms=6.58 avg_simplify_ms=2.12, sum@700+100 failed=0 elapsed=200.83ms avg_case_ms=2.01 avg_simplify_ms=0.78
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.36ms median_wire=13.43ms median_wall=50.80ms, sum@0+100 #173 sum runs=3 median_simplify=11.96ms median_wire=12.02ms median_wall=45.04ms, difference@0+50 #174 difference runs=3 median_simplify=11.91ms median_wire=11.96ms median_wall=45.25ms, product@0+100 #175 product runs=3 median_simplify=12.04ms median_wire=12.10ms median_wall=45.32ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.67ms median_wire=10.74ms median_wall=40.03ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.32s | passed=450 failed=0 total=450 avg_case=5.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.44s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.05s | passed=1 failed=0 |
