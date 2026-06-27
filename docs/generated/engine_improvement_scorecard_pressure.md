# Engine Improvement Scorecard

- Generated: 2026-06-27T00:48:55.244263+00:00
- Git branch: main
- Git commit: `9fe0fd44e93bfd080b6835e8ebef460e9aade07b`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=789.83ms avg_case_ms=7.90 simplify=224.69ms avg_simplify_ms=2.25, sum total=200 failed=0 elapsed=708.97ms avg_case_ms=3.54 simplify=244.75ms avg_simplify_ms=1.22, product total=100 failed=0 elapsed=479.74ms avg_case_ms=4.80 simplify=140.88ms avg_simplify_ms=1.41, difference total=50 failed=0 elapsed=329.66ms avg_case_ms=6.59 simplify=106.59ms avg_simplify_ms=2.13
- Engine hotspots: sum simplify=244.75ms avg_simplify_ms=1.22 wall=708.97ms, shifted_quotient simplify=224.69ms avg_simplify_ms=2.25 wall=789.83ms, product simplify=140.88ms avg_simplify_ms=1.41 wall=479.74ms, difference simplify=106.59ms avg_simplify_ms=2.13 wall=329.66ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=789.83ms avg_case_ms=7.90 avg_simplify_ms=2.25, sum@0+100 failed=0 elapsed=510.27ms avg_case_ms=5.10 avg_simplify_ms=1.68, product@0+100 failed=0 elapsed=479.74ms avg_case_ms=4.80 avg_simplify_ms=1.41, difference@0+50 failed=0 elapsed=329.66ms avg_case_ms=6.59 avg_simplify_ms=2.13, sum@700+100 failed=0 elapsed=198.70ms avg_case_ms=1.99 avg_simplify_ms=0.77
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.90ms median_wire=12.97ms median_wall=48.58ms, sum@0+100 #173 sum runs=3 median_simplify=11.87ms median_wire=11.92ms median_wall=44.90ms, product@0+100 #175 product runs=3 median_simplify=11.63ms median_wire=11.68ms median_wall=44.46ms, difference@0+50 #174 difference runs=3 median_simplify=11.68ms median_wire=11.73ms median_wall=44.74ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.77ms median_wire=10.84ms median_wall=40.62ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.31s | passed=450 failed=0 total=450 avg_case=5.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.45s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.05s | passed=1 failed=0 |
