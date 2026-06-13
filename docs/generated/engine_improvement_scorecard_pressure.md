# Engine Improvement Scorecard

- Generated: 2026-06-13T09:52:43.465526+00:00
- Git branch: main
- Git commit: `cfb4b26037dd06810ed8ffd64f08eff525e07b86`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=344

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=783.61ms avg_case_ms=7.84 simplify=223.01ms avg_simplify_ms=2.23, sum total=200 failed=0 elapsed=691.72ms avg_case_ms=3.46 simplify=231.57ms avg_simplify_ms=1.16, product total=100 failed=0 elapsed=476.53ms avg_case_ms=4.77 simplify=137.04ms avg_simplify_ms=1.37, difference total=50 failed=0 elapsed=331.02ms avg_case_ms=6.62 simplify=104.63ms avg_simplify_ms=2.09
- Engine hotspots: sum simplify=231.57ms avg_simplify_ms=1.16 wall=691.72ms, shifted_quotient simplify=223.01ms avg_simplify_ms=2.23 wall=783.61ms, product simplify=137.04ms avg_simplify_ms=1.37 wall=476.53ms, difference simplify=104.63ms avg_simplify_ms=2.09 wall=331.02ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=783.61ms avg_case_ms=7.84 avg_simplify_ms=2.23, sum@0+100 failed=0 elapsed=498.97ms avg_case_ms=4.99 avg_simplify_ms=1.61, product@0+100 failed=0 elapsed=476.53ms avg_case_ms=4.77 avg_simplify_ms=1.37, difference@0+50 failed=0 elapsed=331.02ms avg_case_ms=6.62 avg_simplify_ms=2.09, sum@700+100 failed=0 elapsed=192.75ms avg_case_ms=1.93 avg_simplify_ms=0.71
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.87ms median_wire=12.94ms median_wall=49.42ms, product@0+100 #175 product runs=3 median_simplify=11.56ms median_wire=11.61ms median_wall=43.71ms, sum@0+100 #173 sum runs=3 median_simplify=11.51ms median_wire=11.56ms median_wall=44.28ms, difference@0+50 #174 difference runs=3 median_simplify=11.57ms median_wire=11.62ms median_wall=44.04ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.62ms median_wire=10.70ms median_wall=40.43ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.28s | passed=450 failed=0 total=450 avg_case=5.067ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.82s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.94s | passed=1 failed=0 |
