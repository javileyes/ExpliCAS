# Engine Improvement Scorecard

- Generated: 2026-06-21T05:20:27.836399+00:00
- Git branch: main
- Git commit: `0f1cdc01cfcc17fa36fb7027f20468d39eb3169a`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=783.49ms avg_case_ms=7.83 simplify=223.95ms avg_simplify_ms=2.24, sum total=200 failed=0 elapsed=697.87ms avg_case_ms=3.49 simplify=237.27ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=477.13ms avg_case_ms=4.77 simplify=137.64ms avg_simplify_ms=1.38, difference total=50 failed=0 elapsed=327.72ms avg_case_ms=6.55 simplify=104.77ms avg_simplify_ms=2.10
- Engine hotspots: sum simplify=237.27ms avg_simplify_ms=1.19 wall=697.87ms, shifted_quotient simplify=223.95ms avg_simplify_ms=2.24 wall=783.49ms, product simplify=137.64ms avg_simplify_ms=1.38 wall=477.13ms, difference simplify=104.77ms avg_simplify_ms=2.10 wall=327.72ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=783.49ms avg_case_ms=7.83 avg_simplify_ms=2.24, sum@0+100 failed=0 elapsed=505.29ms avg_case_ms=5.05 avg_simplify_ms=1.65, product@0+100 failed=0 elapsed=477.13ms avg_case_ms=4.77 avg_simplify_ms=1.38, difference@0+50 failed=0 elapsed=327.72ms avg_case_ms=6.55 avg_simplify_ms=2.10, sum@700+100 failed=0 elapsed=192.58ms avg_case_ms=1.93 avg_simplify_ms=0.72
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.14ms median_wire=13.20ms median_wall=49.90ms, difference@0+50 #174 difference runs=3 median_simplify=12.06ms median_wire=12.11ms median_wall=45.68ms, sum@0+100 #173 sum runs=3 median_simplify=11.97ms median_wire=12.02ms median_wall=45.23ms, product@0+100 #175 product runs=3 median_simplify=11.58ms median_wire=11.63ms median_wall=44.33ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.64ms median_wire=10.71ms median_wall=40.05ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.29s | passed=450 failed=0 total=450 avg_case=5.089ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.42s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.05s | passed=1 failed=0 |
