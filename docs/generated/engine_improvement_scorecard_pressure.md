# Engine Improvement Scorecard

- Generated: 2026-07-24T15:20:24.327719+00:00
- Git branch: main
- Git commit: `ccdd34bc6d517c9d44344f9b79a0fd345ff96850`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=1.02s avg_case_ms=10.24 simplify=285.91ms avg_simplify_ms=2.86, sum total=200 failed=0 elapsed=904.81ms avg_case_ms=4.52 simplify=290.99ms avg_simplify_ms=1.45, product total=100 failed=0 elapsed=612.49ms avg_case_ms=6.12 simplify=177.54ms avg_simplify_ms=1.78, difference total=50 failed=0 elapsed=405.85ms avg_case_ms=8.12 simplify=123.70ms avg_simplify_ms=2.47
- Engine hotspots: sum simplify=290.99ms avg_simplify_ms=1.45 wall=904.81ms, shifted_quotient simplify=285.91ms avg_simplify_ms=2.86 wall=1.02s, product simplify=177.54ms avg_simplify_ms=1.78 wall=612.49ms, difference simplify=123.70ms avg_simplify_ms=2.47 wall=405.85ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=1.02s avg_case_ms=10.24 avg_simplify_ms=2.86, sum@0+100 failed=0 elapsed=667.40ms avg_case_ms=6.67 avg_simplify_ms=2.06, product@0+100 failed=0 elapsed=612.49ms avg_case_ms=6.12 avg_simplify_ms=1.78, difference@0+50 failed=0 elapsed=405.85ms avg_case_ms=8.12 avg_simplify_ms=2.47, sum@700+100 failed=0 elapsed=237.41ms avg_case_ms=2.37 avg_simplify_ms=0.85
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.87ms median_wire=16.94ms median_wall=65.26ms, sum@0+100 #173 sum runs=3 median_simplify=15.19ms median_wire=15.25ms median_wall=58.45ms, product@0+100 #175 product runs=3 median_simplify=16.04ms median_wire=16.10ms median_wall=60.81ms, difference@0+50 #174 difference runs=3 median_simplify=15.72ms median_wire=15.77ms median_wall=60.08ms, shifted_quotient@0+100 #112 shifted_quotient runs=3 median_simplify=9.77ms median_wire=9.83ms median_wall=37.75ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.95s | passed=450 failed=0 total=450 avg_case=6.556ms |
| `calculus_diff_exhaustive_contract` | `pass` | 13.00s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.00s | passed=1 failed=0 |
