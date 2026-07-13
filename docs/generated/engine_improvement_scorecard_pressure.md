# Engine Improvement Scorecard

- Generated: 2026-07-13T17:00:51.423014+00:00
- Git branch: main
- Git commit: `34755a38fc556833793a51bbde2685a109f81a6f`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=960.99ms avg_case_ms=9.61 simplify=268.34ms avg_simplify_ms=2.68, sum total=200 failed=0 elapsed=844.48ms avg_case_ms=4.22 simplify=273.61ms avg_simplify_ms=1.37, product total=100 failed=0 elapsed=580.29ms avg_case_ms=5.80 simplify=164.89ms avg_simplify_ms=1.65, difference total=50 failed=0 elapsed=386.24ms avg_case_ms=7.72 simplify=117.73ms avg_simplify_ms=2.35
- Engine hotspots: sum simplify=273.61ms avg_simplify_ms=1.37 wall=844.48ms, shifted_quotient simplify=268.34ms avg_simplify_ms=2.68 wall=960.99ms, product simplify=164.89ms avg_simplify_ms=1.65 wall=580.29ms, difference simplify=117.73ms avg_simplify_ms=2.35 wall=386.24ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=960.99ms avg_case_ms=9.61 avg_simplify_ms=2.68, sum@0+100 failed=0 elapsed=618.31ms avg_case_ms=6.18 avg_simplify_ms=1.93, product@0+100 failed=0 elapsed=580.29ms avg_case_ms=5.80 avg_simplify_ms=1.65, difference@0+50 failed=0 elapsed=386.24ms avg_case_ms=7.72 avg_simplify_ms=2.35, sum@700+100 failed=0 elapsed=226.17ms avg_case_ms=2.26 avg_simplify_ms=0.81
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=14.80ms median_wire=14.85ms median_wall=56.93ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.24ms median_wire=16.31ms median_wall=63.06ms, difference@0+50 #174 difference runs=3 median_simplify=15.07ms median_wire=15.11ms median_wall=57.60ms, product@0+100 #175 product runs=3 median_simplify=16.36ms median_wire=16.42ms median_wall=58.78ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.58ms median_wire=12.65ms median_wall=48.24ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.77s | passed=450 failed=0 total=450 avg_case=6.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.34s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.90s | passed=1 failed=0 |
