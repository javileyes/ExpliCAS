# Engine Improvement Scorecard

- Generated: 2026-07-30T12:15:20.441062+00:00
- Git branch: main
- Git commit: `ad5ab6477ee417b181c42eafc84d6855235055c9`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=964.44ms avg_case_ms=9.64 simplify=271.52ms avg_simplify_ms=2.72, sum total=200 failed=0 elapsed=845.36ms avg_case_ms=4.23 simplify=271.59ms avg_simplify_ms=1.36, product total=100 failed=0 elapsed=599.65ms avg_case_ms=6.00 simplify=172.32ms avg_simplify_ms=1.72, difference total=50 failed=0 elapsed=388.02ms avg_case_ms=7.76 simplify=113.66ms avg_simplify_ms=2.27
- Engine hotspots: sum simplify=271.59ms avg_simplify_ms=1.36 wall=845.36ms, shifted_quotient simplify=271.52ms avg_simplify_ms=2.72 wall=964.44ms, product simplify=172.32ms avg_simplify_ms=1.72 wall=599.65ms, difference simplify=113.66ms avg_simplify_ms=2.27 wall=388.02ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=964.44ms avg_case_ms=9.64 avg_simplify_ms=2.72, sum@0+100 failed=0 elapsed=615.58ms avg_case_ms=6.16 avg_simplify_ms=1.89, product@0+100 failed=0 elapsed=599.65ms avg_case_ms=6.00 avg_simplify_ms=1.72, difference@0+50 failed=0 elapsed=388.02ms avg_case_ms=7.76 avg_simplify_ms=2.27, sum@700+100 failed=0 elapsed=229.78ms avg_case_ms=2.30 avg_simplify_ms=0.82
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.38ms median_wire=16.44ms median_wall=62.27ms, product@0+100 #175 product runs=3 median_simplify=14.85ms median_wire=14.89ms median_wall=56.52ms, difference@0+50 #174 difference runs=3 median_simplify=14.87ms median_wire=14.91ms median_wall=57.30ms, sum@0+100 #173 sum runs=3 median_simplify=15.02ms median_wire=15.06ms median_wall=57.42ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.74ms median_wire=12.82ms median_wall=48.27ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.80s | passed=450 failed=0 total=450 avg_case=6.222ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.16s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.73s | passed=1 failed=0 |
