# Engine Improvement Scorecard

- Generated: 2026-07-17T11:03:01.735430+00:00
- Git branch: main
- Git commit: `587df0a76facced2576086f71550a246ee393c6b`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=972.76ms avg_case_ms=9.73 simplify=274.52ms avg_simplify_ms=2.75, sum total=200 failed=0 elapsed=880.86ms avg_case_ms=4.40 simplify=284.17ms avg_simplify_ms=1.42, product total=100 failed=0 elapsed=596.54ms avg_case_ms=5.97 simplify=171.45ms avg_simplify_ms=1.71, difference total=50 failed=0 elapsed=395.75ms avg_case_ms=7.91 simplify=120.62ms avg_simplify_ms=2.41
- Engine hotspots: sum simplify=284.17ms avg_simplify_ms=1.42 wall=880.86ms, shifted_quotient simplify=274.52ms avg_simplify_ms=2.75 wall=972.76ms, product simplify=171.45ms avg_simplify_ms=1.71 wall=596.54ms, difference simplify=120.62ms avg_simplify_ms=2.41 wall=395.75ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=972.76ms avg_case_ms=9.73 avg_simplify_ms=2.75, sum@0+100 failed=0 elapsed=648.88ms avg_case_ms=6.49 avg_simplify_ms=2.01, product@0+100 failed=0 elapsed=596.54ms avg_case_ms=5.97 avg_simplify_ms=1.71, difference@0+50 failed=0 elapsed=395.75ms avg_case_ms=7.91 avg_simplify_ms=2.41, sum@700+100 failed=0 elapsed=231.99ms avg_case_ms=2.32 avg_simplify_ms=0.83
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.87ms median_wire=16.96ms median_wall=64.21ms, sum@0+100 #173 sum runs=3 median_simplify=15.33ms median_wire=15.37ms median_wall=58.46ms, difference@0+50 #174 difference runs=3 median_simplify=15.16ms median_wire=15.20ms median_wall=58.20ms, product@0+100 #175 product runs=3 median_simplify=15.21ms median_wire=15.26ms median_wall=57.56ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.16ms median_wire=13.24ms median_wall=49.09ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.85s | passed=450 failed=0 total=450 avg_case=6.333ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.47s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.91s | passed=1 failed=0 |
