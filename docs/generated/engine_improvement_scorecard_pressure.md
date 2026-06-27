# Engine Improvement Scorecard

- Generated: 2026-06-27T00:04:42.948755+00:00
- Git branch: main
- Git commit: `ba392f23ecc48596729e143a7223e27a0aa0c095`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=806.33ms avg_case_ms=8.06 simplify=232.03ms avg_simplify_ms=2.32, sum total=200 failed=0 elapsed=717.67ms avg_case_ms=3.59 simplify=248.02ms avg_simplify_ms=1.24, product total=100 failed=0 elapsed=489.01ms avg_case_ms=4.89 simplify=144.33ms avg_simplify_ms=1.44, difference total=50 failed=0 elapsed=339.38ms avg_case_ms=6.79 simplify=111.59ms avg_simplify_ms=2.23
- Engine hotspots: sum simplify=248.02ms avg_simplify_ms=1.24 wall=717.67ms, shifted_quotient simplify=232.03ms avg_simplify_ms=2.32 wall=806.33ms, product simplify=144.33ms avg_simplify_ms=1.44 wall=489.01ms, difference simplify=111.59ms avg_simplify_ms=2.23 wall=339.38ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=806.33ms avg_case_ms=8.06 avg_simplify_ms=2.32, sum@0+100 failed=0 elapsed=516.77ms avg_case_ms=5.17 avg_simplify_ms=1.71, product@0+100 failed=0 elapsed=489.01ms avg_case_ms=4.89 avg_simplify_ms=1.44, difference@0+50 failed=0 elapsed=339.38ms avg_case_ms=6.79 avg_simplify_ms=2.23, sum@700+100 failed=0 elapsed=200.90ms avg_case_ms=2.01 avg_simplify_ms=0.77
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.04ms median_wire=13.11ms median_wall=50.62ms, sum@0+100 #173 sum runs=3 median_simplify=12.27ms median_wire=12.33ms median_wall=46.02ms, difference@0+50 #174 difference runs=3 median_simplify=12.05ms median_wire=12.10ms median_wall=46.00ms, product@0+100 #175 product runs=3 median_simplify=11.93ms median_wire=11.99ms median_wall=45.34ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=11.07ms median_wire=11.15ms median_wall=41.83ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.35s | passed=450 failed=0 total=450 avg_case=5.222ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.56s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.11s | passed=1 failed=0 |
