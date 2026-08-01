# Engine Improvement Scorecard

- Generated: 2026-08-01T22:51:12.701964+00:00
- Git branch: main
- Git commit: `43fdd5bbdde67044b7fdbe07d20c0b5ecd5fbe5c`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=391

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=975.06ms avg_case_ms=9.75 simplify=273.11ms avg_simplify_ms=2.73, sum total=200 failed=0 elapsed=864.10ms avg_case_ms=4.32 simplify=277.47ms avg_simplify_ms=1.39, product total=100 failed=0 elapsed=603.99ms avg_case_ms=6.04 simplify=173.06ms avg_simplify_ms=1.73, difference total=50 failed=0 elapsed=390.18ms avg_case_ms=7.80 simplify=114.25ms avg_simplify_ms=2.29
- Engine hotspots: sum simplify=277.47ms avg_simplify_ms=1.39 wall=864.10ms, shifted_quotient simplify=273.11ms avg_simplify_ms=2.73 wall=975.06ms, product simplify=173.06ms avg_simplify_ms=1.73 wall=603.99ms, difference simplify=114.25ms avg_simplify_ms=2.29 wall=390.18ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=975.06ms avg_case_ms=9.75 avg_simplify_ms=2.73, sum@0+100 failed=0 elapsed=625.73ms avg_case_ms=6.26 avg_simplify_ms=1.93, product@0+100 failed=0 elapsed=603.99ms avg_case_ms=6.04 avg_simplify_ms=1.73, difference@0+50 failed=0 elapsed=390.18ms avg_case_ms=7.80 avg_simplify_ms=2.29, sum@700+100 failed=0 elapsed=238.37ms avg_case_ms=2.38 avg_simplify_ms=0.85
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.21ms median_wire=16.27ms median_wall=62.63ms, sum@0+100 #173 sum runs=3 median_simplify=15.06ms median_wire=15.11ms median_wall=57.76ms, difference@0+50 #174 difference runs=3 median_simplify=14.97ms median_wire=15.02ms median_wall=56.75ms, product@0+100 #175 product runs=3 median_simplify=14.87ms median_wire=14.92ms median_wall=56.75ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.75ms median_wire=12.83ms median_wall=48.54ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.83s | passed=450 failed=0 total=450 avg_case=6.289ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.48s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.75s | passed=1 failed=0 |
