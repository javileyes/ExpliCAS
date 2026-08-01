# Engine Improvement Scorecard

- Generated: 2026-08-01T22:21:12.982410+00:00
- Git branch: main
- Git commit: `2fcafa37dcf95f1532896f2802c55c8aa6beb11e`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=964.93ms avg_case_ms=9.65 simplify=269.56ms avg_simplify_ms=2.70, sum total=200 failed=0 elapsed=857.69ms avg_case_ms=4.29 simplify=273.77ms avg_simplify_ms=1.37, product total=100 failed=0 elapsed=596.45ms avg_case_ms=5.96 simplify=171.29ms avg_simplify_ms=1.71, difference total=50 failed=0 elapsed=384.73ms avg_case_ms=7.69 simplify=113.16ms avg_simplify_ms=2.26
- Engine hotspots: sum simplify=273.77ms avg_simplify_ms=1.37 wall=857.69ms, shifted_quotient simplify=269.56ms avg_simplify_ms=2.70 wall=964.93ms, product simplify=171.29ms avg_simplify_ms=1.71 wall=596.45ms, difference simplify=113.16ms avg_simplify_ms=2.26 wall=384.73ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=964.93ms avg_case_ms=9.65 avg_simplify_ms=2.70, sum@0+100 failed=0 elapsed=624.45ms avg_case_ms=6.24 avg_simplify_ms=1.90, product@0+100 failed=0 elapsed=596.45ms avg_case_ms=5.96 avg_simplify_ms=1.71, difference@0+50 failed=0 elapsed=384.73ms avg_case_ms=7.69 avg_simplify_ms=2.26, sum@700+100 failed=0 elapsed=233.23ms avg_case_ms=2.33 avg_simplify_ms=0.84
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.53ms median_wire=16.59ms median_wall=63.43ms, product@0+100 #175 product runs=3 median_simplify=15.05ms median_wire=15.10ms median_wall=57.64ms, sum@0+100 #173 sum runs=3 median_simplify=14.94ms median_wire=14.98ms median_wall=57.22ms, difference@0+50 #174 difference runs=3 median_simplify=15.07ms median_wire=15.12ms median_wall=57.81ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.82ms median_wire=12.90ms median_wall=49.05ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.80s | passed=450 failed=0 total=450 avg_case=6.222ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.28s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.74s | passed=1 failed=0 |
