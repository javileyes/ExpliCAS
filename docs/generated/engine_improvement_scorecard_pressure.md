# Engine Improvement Scorecard

- Generated: 2026-07-28T10:03:16.672240+00:00
- Git branch: main
- Git commit: `438fc0b21ca2115fed097a27cee3c57c59e1ae04`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=380

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=963.74ms avg_case_ms=9.64 simplify=269.71ms avg_simplify_ms=2.70, sum total=200 failed=0 elapsed=848.74ms avg_case_ms=4.24 simplify=276.39ms avg_simplify_ms=1.38, product total=100 failed=0 elapsed=601.88ms avg_case_ms=6.02 simplify=173.26ms avg_simplify_ms=1.73, difference total=50 failed=0 elapsed=396.70ms avg_case_ms=7.93 simplify=120.28ms avg_simplify_ms=2.41
- Engine hotspots: sum simplify=276.39ms avg_simplify_ms=1.38 wall=848.74ms, shifted_quotient simplify=269.71ms avg_simplify_ms=2.70 wall=963.74ms, product simplify=173.26ms avg_simplify_ms=1.73 wall=601.88ms, difference simplify=120.28ms avg_simplify_ms=2.41 wall=396.70ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=963.74ms avg_case_ms=9.64 avg_simplify_ms=2.70, sum@0+100 failed=0 elapsed=618.01ms avg_case_ms=6.18 avg_simplify_ms=1.94, product@0+100 failed=0 elapsed=601.88ms avg_case_ms=6.02 avg_simplify_ms=1.73, difference@0+50 failed=0 elapsed=396.70ms avg_case_ms=7.93 avg_simplify_ms=2.41, sum@700+100 failed=0 elapsed=230.73ms avg_case_ms=2.31 avg_simplify_ms=0.82
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.53ms median_wire=16.60ms median_wall=63.32ms, difference@0+50 #174 difference runs=3 median_simplify=15.02ms median_wire=15.06ms median_wall=57.61ms, sum@0+100 #173 sum runs=3 median_simplify=15.31ms median_wire=15.36ms median_wall=58.79ms, product@0+100 #175 product runs=3 median_simplify=15.21ms median_wire=15.26ms median_wall=58.16ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.81ms median_wire=12.88ms median_wall=48.72ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.81s | passed=450 failed=0 total=450 avg_case=6.244ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.14s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.83s | passed=1 failed=0 |
