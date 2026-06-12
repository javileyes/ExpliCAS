# Engine Improvement Scorecard

- Generated: 2026-06-12T22:05:37.205020+00:00
- Git branch: main
- Git commit: `90a38db7d82be193f9adc304dd841d5cdebde452`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=344

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=807.27ms avg_case_ms=8.07 simplify=230.84ms avg_simplify_ms=2.31, sum total=200 failed=0 elapsed=691.56ms avg_case_ms=3.46 simplify=232.02ms avg_simplify_ms=1.16, product total=100 failed=0 elapsed=488.26ms avg_case_ms=4.88 simplify=142.15ms avg_simplify_ms=1.42, difference total=50 failed=0 elapsed=341.98ms avg_case_ms=6.84 simplify=107.17ms avg_simplify_ms=2.14
- Engine hotspots: sum simplify=232.02ms avg_simplify_ms=1.16 wall=691.56ms, shifted_quotient simplify=230.84ms avg_simplify_ms=2.31 wall=807.27ms, product simplify=142.15ms avg_simplify_ms=1.42 wall=488.26ms, difference simplify=107.17ms avg_simplify_ms=2.14 wall=341.98ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=807.27ms avg_case_ms=8.07 avg_simplify_ms=2.31, sum@0+100 failed=0 elapsed=500.29ms avg_case_ms=5.00 avg_simplify_ms=1.61, product@0+100 failed=0 elapsed=488.26ms avg_case_ms=4.88 avg_simplify_ms=1.42, difference@0+50 failed=0 elapsed=341.98ms avg_case_ms=6.84 avg_simplify_ms=2.14, sum@700+100 failed=0 elapsed=191.27ms avg_case_ms=1.91 avg_simplify_ms=0.71
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.92ms median_wire=12.99ms median_wall=49.07ms, difference@0+50 #174 difference runs=3 median_simplify=11.65ms median_wire=11.71ms median_wall=44.24ms, product@0+100 #175 product runs=3 median_simplify=11.49ms median_wire=11.54ms median_wall=43.62ms, sum@0+100 #173 sum runs=3 median_simplify=11.39ms median_wire=11.44ms median_wall=43.97ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.40ms median_wire=10.47ms median_wall=40.07ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.33s | passed=450 failed=0 total=450 avg_case=5.178ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.82s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
