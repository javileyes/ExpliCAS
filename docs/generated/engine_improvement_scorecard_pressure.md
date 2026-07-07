# Engine Improvement Scorecard

- Generated: 2026-07-07T07:50:32.922846+00:00
- Git branch: main
- Git commit: `fae21d8f7816108b2f16cb1949f02f33bd2dfe9f`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=355

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=928.90ms avg_case_ms=9.29 simplify=259.23ms avg_simplify_ms=2.59, sum total=200 failed=0 elapsed=821.02ms avg_case_ms=4.11 simplify=264.56ms avg_simplify_ms=1.32, product total=100 failed=0 elapsed=577.84ms avg_case_ms=5.78 simplify=164.34ms avg_simplify_ms=1.64, difference total=50 failed=0 elapsed=386.88ms avg_case_ms=7.74 simplify=116.58ms avg_simplify_ms=2.33
- Engine hotspots: sum simplify=264.56ms avg_simplify_ms=1.32 wall=821.02ms, shifted_quotient simplify=259.23ms avg_simplify_ms=2.59 wall=928.90ms, product simplify=164.34ms avg_simplify_ms=1.64 wall=577.84ms, difference simplify=116.58ms avg_simplify_ms=2.33 wall=386.88ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=928.90ms avg_case_ms=9.29 avg_simplify_ms=2.59, sum@0+100 failed=0 elapsed=600.88ms avg_case_ms=6.01 avg_simplify_ms=1.86, product@0+100 failed=0 elapsed=577.84ms avg_case_ms=5.78 avg_simplify_ms=1.64, difference@0+50 failed=0 elapsed=386.88ms avg_case_ms=7.74 avg_simplify_ms=2.33, sum@700+100 failed=0 elapsed=220.14ms avg_case_ms=2.20 avg_simplify_ms=0.78
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.14ms median_wire=16.21ms median_wall=61.74ms, sum@0+100 #173 sum runs=3 median_simplify=14.79ms median_wire=14.83ms median_wall=56.36ms, difference@0+50 #174 difference runs=3 median_simplify=14.53ms median_wire=14.58ms median_wall=56.26ms, product@0+100 #175 product runs=3 median_simplify=14.89ms median_wire=14.94ms median_wall=56.44ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.59ms median_wire=12.66ms median_wall=47.59ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.72s | passed=450 failed=0 total=450 avg_case=6.044ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.98s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.85s | passed=1 failed=0 |
