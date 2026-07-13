# Engine Improvement Scorecard

- Generated: 2026-07-13T23:25:13.228586+00:00
- Git branch: main
- Git commit: `9e4e854aa86565a3f9a32f9e073fe42fcfaa9883`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=357

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=952.85ms avg_case_ms=9.53 simplify=265.53ms avg_simplify_ms=2.66, sum total=200 failed=0 elapsed=853.09ms avg_case_ms=4.27 simplify=274.59ms avg_simplify_ms=1.37, product total=100 failed=0 elapsed=603.95ms avg_case_ms=6.04 simplify=171.47ms avg_simplify_ms=1.71, difference total=50 failed=0 elapsed=392.77ms avg_case_ms=7.86 simplify=119.28ms avg_simplify_ms=2.39
- Engine hotspots: sum simplify=274.59ms avg_simplify_ms=1.37 wall=853.09ms, shifted_quotient simplify=265.53ms avg_simplify_ms=2.66 wall=952.85ms, product simplify=171.47ms avg_simplify_ms=1.71 wall=603.95ms, difference simplify=119.28ms avg_simplify_ms=2.39 wall=392.77ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=952.85ms avg_case_ms=9.53 avg_simplify_ms=2.66, sum@0+100 failed=0 elapsed=627.95ms avg_case_ms=6.28 avg_simplify_ms=1.95, product@0+100 failed=0 elapsed=603.95ms avg_case_ms=6.04 avg_simplify_ms=1.71, difference@0+50 failed=0 elapsed=392.77ms avg_case_ms=7.86 avg_simplify_ms=2.39, sum@700+100 failed=0 elapsed=225.14ms avg_case_ms=2.25 avg_simplify_ms=0.80
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=15.49ms median_wire=15.54ms median_wall=62.89ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.48ms median_wire=16.55ms median_wall=63.53ms, product@0+100 #175 product runs=3 median_simplify=14.88ms median_wire=14.93ms median_wall=58.01ms, difference@0+50 #174 difference runs=3 median_simplify=15.54ms median_wire=15.60ms median_wall=58.99ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.88ms median_wire=12.95ms median_wall=48.75ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.80s | passed=450 failed=0 total=450 avg_case=6.222ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.32s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.90s | passed=1 failed=0 |
