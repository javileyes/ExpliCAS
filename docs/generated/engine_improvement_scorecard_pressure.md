# Engine Improvement Scorecard

- Generated: 2026-06-11T21:20:54.663875+00:00
- Git branch: main
- Git commit: `40390a0dec6142669136e0ada627f7f78987b585`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=796.43ms avg_case_ms=7.96 simplify=226.14ms avg_simplify_ms=2.26, sum total=200 failed=0 elapsed=679.32ms avg_case_ms=3.40 simplify=227.51ms avg_simplify_ms=1.14, product total=100 failed=0 elapsed=477.06ms avg_case_ms=4.77 simplify=136.06ms avg_simplify_ms=1.36, difference total=50 failed=0 elapsed=316.89ms avg_case_ms=6.34 simplify=100.50ms avg_simplify_ms=2.01
- Engine hotspots: sum simplify=227.51ms avg_simplify_ms=1.14 wall=679.32ms, shifted_quotient simplify=226.14ms avg_simplify_ms=2.26 wall=796.43ms, product simplify=136.06ms avg_simplify_ms=1.36 wall=477.06ms, difference simplify=100.50ms avg_simplify_ms=2.01 wall=316.89ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=796.43ms avg_case_ms=7.96 avg_simplify_ms=2.26, sum@0+100 failed=0 elapsed=494.30ms avg_case_ms=4.94 avg_simplify_ms=1.59, product@0+100 failed=0 elapsed=477.06ms avg_case_ms=4.77 avg_simplify_ms=1.36, difference@0+50 failed=0 elapsed=316.89ms avg_case_ms=6.34 avg_simplify_ms=2.01, sum@700+100 failed=0 elapsed=185.02ms avg_case_ms=1.85 avg_simplify_ms=0.69
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.33ms median_wire=13.41ms median_wall=51.17ms, sum@0+100 #173 sum runs=3 median_simplify=11.85ms median_wire=11.90ms median_wall=45.77ms, product@0+100 #175 product runs=3 median_simplify=11.83ms median_wire=11.89ms median_wall=44.69ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.54ms median_wire=10.60ms median_wall=40.08ms, difference@0+50 #174 difference runs=3 median_simplify=11.46ms median_wire=11.51ms median_wall=43.90ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.27s | passed=450 failed=0 total=450 avg_case=5.044ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.88s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.46s | passed=1 failed=0 |
