# Engine Improvement Scorecard

- Generated: 2026-07-27T04:25:42.687244+00:00
- Git branch: main
- Git commit: `7ffb745a1c39c396d1cd2b2f508f0849d308ac72`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=942.68ms avg_case_ms=9.43 simplify=263.71ms avg_simplify_ms=2.64, sum total=200 failed=0 elapsed=836.98ms avg_case_ms=4.18 simplify=271.95ms avg_simplify_ms=1.36, product total=100 failed=0 elapsed=590.56ms avg_case_ms=5.91 simplify=170.15ms avg_simplify_ms=1.70, difference total=50 failed=0 elapsed=386.28ms avg_case_ms=7.73 simplify=116.99ms avg_simplify_ms=2.34
- Engine hotspots: sum simplify=271.95ms avg_simplify_ms=1.36 wall=836.98ms, shifted_quotient simplify=263.71ms avg_simplify_ms=2.64 wall=942.68ms, product simplify=170.15ms avg_simplify_ms=1.70 wall=590.56ms, difference simplify=116.99ms avg_simplify_ms=2.34 wall=386.28ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=942.68ms avg_case_ms=9.43 avg_simplify_ms=2.64, sum@0+100 failed=0 elapsed=607.34ms avg_case_ms=6.07 avg_simplify_ms=1.90, product@0+100 failed=0 elapsed=590.56ms avg_case_ms=5.91 avg_simplify_ms=1.70, difference@0+50 failed=0 elapsed=386.28ms avg_case_ms=7.73 avg_simplify_ms=2.34, sum@700+100 failed=0 elapsed=229.64ms avg_case_ms=2.30 avg_simplify_ms=0.82
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.30ms median_wire=16.36ms median_wall=62.50ms, product@0+100 #175 product runs=3 median_simplify=14.75ms median_wire=14.79ms median_wall=55.93ms, difference@0+50 #174 difference runs=3 median_simplify=14.72ms median_wire=14.76ms median_wall=56.49ms, sum@0+100 #173 sum runs=3 median_simplify=14.81ms median_wire=14.85ms median_wall=56.69ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.43ms median_wire=12.49ms median_wall=46.81ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.76s | passed=450 failed=0 total=450 avg_case=6.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.00s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.82s | passed=1 failed=0 |
