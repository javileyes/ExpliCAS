# Engine Improvement Scorecard

- Generated: 2026-06-12T09:11:30.264092+00:00
- Git branch: main
- Git commit: `c302c03f19bf7480c482dc91c3febfcc74deafd7`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=777.95ms avg_case_ms=7.78 simplify=222.00ms avg_simplify_ms=2.22, sum total=200 failed=0 elapsed=683.52ms avg_case_ms=3.42 simplify=230.05ms avg_simplify_ms=1.15, product total=100 failed=0 elapsed=477.26ms avg_case_ms=4.77 simplify=136.98ms avg_simplify_ms=1.37, difference total=50 failed=0 elapsed=331.13ms avg_case_ms=6.62 simplify=105.46ms avg_simplify_ms=2.11
- Engine hotspots: sum simplify=230.05ms avg_simplify_ms=1.15 wall=683.52ms, shifted_quotient simplify=222.00ms avg_simplify_ms=2.22 wall=777.95ms, product simplify=136.98ms avg_simplify_ms=1.37 wall=477.26ms, difference simplify=105.46ms avg_simplify_ms=2.11 wall=331.13ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=777.95ms avg_case_ms=7.78 avg_simplify_ms=2.22, sum@0+100 failed=0 elapsed=494.75ms avg_case_ms=4.95 avg_simplify_ms=1.60, product@0+100 failed=0 elapsed=477.26ms avg_case_ms=4.77 avg_simplify_ms=1.37, difference@0+50 failed=0 elapsed=331.13ms avg_case_ms=6.62 avg_simplify_ms=2.11, sum@700+100 failed=0 elapsed=188.77ms avg_case_ms=1.89 avg_simplify_ms=0.70
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.64ms median_wire=12.70ms median_wall=48.42ms, difference@0+50 #174 difference runs=3 median_simplify=11.55ms median_wire=11.59ms median_wall=44.02ms, sum@0+100 #173 sum runs=3 median_simplify=11.52ms median_wire=11.57ms median_wall=44.05ms, product@0+100 #175 product runs=3 median_simplify=11.48ms median_wire=11.53ms median_wall=43.90ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.40ms median_wire=10.47ms median_wall=39.68ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.27s | passed=450 failed=0 total=450 avg_case=5.044ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.82s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.28s | passed=1 failed=0 |
