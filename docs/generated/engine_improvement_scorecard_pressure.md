# Engine Improvement Scorecard

- Generated: 2026-06-15T22:38:14.940824+00:00
- Git branch: main
- Git commit: `d42ba590e82bc44b10ffcd29839eb5907a674137`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=788.59ms avg_case_ms=7.89 simplify=224.07ms avg_simplify_ms=2.24, sum total=200 failed=0 elapsed=702.56ms avg_case_ms=3.51 simplify=235.03ms avg_simplify_ms=1.18, product total=100 failed=0 elapsed=476.93ms avg_case_ms=4.77 simplify=136.51ms avg_simplify_ms=1.37, difference total=50 failed=0 elapsed=328.18ms avg_case_ms=6.56 simplify=104.36ms avg_simplify_ms=2.09
- Engine hotspots: sum simplify=235.03ms avg_simplify_ms=1.18 wall=702.56ms, shifted_quotient simplify=224.07ms avg_simplify_ms=2.24 wall=788.59ms, product simplify=136.51ms avg_simplify_ms=1.37 wall=476.93ms, difference simplify=104.36ms avg_simplify_ms=2.09 wall=328.18ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=788.59ms avg_case_ms=7.89 avg_simplify_ms=2.24, sum@0+100 failed=0 elapsed=508.04ms avg_case_ms=5.08 avg_simplify_ms=1.63, product@0+100 failed=0 elapsed=476.93ms avg_case_ms=4.77 avg_simplify_ms=1.37, difference@0+50 failed=0 elapsed=328.18ms avg_case_ms=6.56 avg_simplify_ms=2.09, sum@700+100 failed=0 elapsed=194.51ms avg_case_ms=1.95 avg_simplify_ms=0.72
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.80ms median_wire=12.87ms median_wall=49.21ms, product@0+100 #175 product runs=3 median_simplify=11.70ms median_wire=11.75ms median_wall=44.40ms, difference@0+50 #174 difference runs=3 median_simplify=11.69ms median_wire=11.74ms median_wall=44.31ms, sum@0+100 #173 sum runs=3 median_simplify=11.82ms median_wire=11.88ms median_wall=44.70ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.76ms median_wire=10.83ms median_wall=40.47ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.30s | passed=450 failed=0 total=450 avg_case=5.111ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.82s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.93s | passed=1 failed=0 |
