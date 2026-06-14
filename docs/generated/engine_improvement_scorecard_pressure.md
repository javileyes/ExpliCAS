# Engine Improvement Scorecard

- Generated: 2026-06-14T08:13:04.126547+00:00
- Git branch: main
- Git commit: `e3ed4689f364400b0dec29e5b2f63e9221013924`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=791.01ms avg_case_ms=7.91 simplify=224.04ms avg_simplify_ms=2.24, sum total=200 failed=0 elapsed=696.16ms avg_case_ms=3.48 simplify=232.94ms avg_simplify_ms=1.16, product total=100 failed=0 elapsed=479.83ms avg_case_ms=4.80 simplify=136.95ms avg_simplify_ms=1.37, difference total=50 failed=0 elapsed=327.19ms avg_case_ms=6.54 simplify=103.35ms avg_simplify_ms=2.07
- Engine hotspots: sum simplify=232.94ms avg_simplify_ms=1.16 wall=696.16ms, shifted_quotient simplify=224.04ms avg_simplify_ms=2.24 wall=791.01ms, product simplify=136.95ms avg_simplify_ms=1.37 wall=479.83ms, difference simplify=103.35ms avg_simplify_ms=2.07 wall=327.19ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=791.01ms avg_case_ms=7.91 avg_simplify_ms=2.24, sum@0+100 failed=0 elapsed=501.18ms avg_case_ms=5.01 avg_simplify_ms=1.61, product@0+100 failed=0 elapsed=479.83ms avg_case_ms=4.80 avg_simplify_ms=1.37, difference@0+50 failed=0 elapsed=327.19ms avg_case_ms=6.54 avg_simplify_ms=2.07, sum@700+100 failed=0 elapsed=194.99ms avg_case_ms=1.95 avg_simplify_ms=0.72
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.97ms median_wire=13.04ms median_wall=48.87ms, product@0+100 #175 product runs=3 median_simplify=11.58ms median_wire=11.63ms median_wall=44.36ms, sum@0+100 #173 sum runs=3 median_simplify=11.61ms median_wire=11.66ms median_wall=44.30ms, difference@0+50 #174 difference runs=3 median_simplify=11.72ms median_wire=11.77ms median_wall=44.84ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.58ms median_wire=10.65ms median_wall=40.10ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.29s | passed=450 failed=0 total=450 avg_case=5.089ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.82s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.91s | passed=1 failed=0 |
