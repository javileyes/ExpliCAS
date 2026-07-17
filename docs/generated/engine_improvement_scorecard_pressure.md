# Engine Improvement Scorecard

- Generated: 2026-07-17T14:54:35.064670+00:00
- Git branch: main
- Git commit: `62577e1b3c74092db76093face6a192c326e7271`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=1.03s avg_case_ms=10.28 simplify=288.49ms avg_simplify_ms=2.88, sum total=200 failed=0 elapsed=910.05ms avg_case_ms=4.55 simplify=290.14ms avg_simplify_ms=1.45, product total=100 failed=0 elapsed=628.23ms avg_case_ms=6.28 simplify=182.90ms avg_simplify_ms=1.83, difference total=50 failed=0 elapsed=415.37ms avg_case_ms=8.31 simplify=128.01ms avg_simplify_ms=2.56
- Engine hotspots: sum simplify=290.14ms avg_simplify_ms=1.45 wall=910.05ms, shifted_quotient simplify=288.49ms avg_simplify_ms=2.88 wall=1.03s, product simplify=182.90ms avg_simplify_ms=1.83 wall=628.23ms, difference simplify=128.01ms avg_simplify_ms=2.56 wall=415.37ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=1.03s avg_case_ms=10.28 avg_simplify_ms=2.88, sum@0+100 failed=0 elapsed=671.12ms avg_case_ms=6.71 avg_simplify_ms=2.03, product@0+100 failed=0 elapsed=628.23ms avg_case_ms=6.28 avg_simplify_ms=1.83, difference@0+50 failed=0 elapsed=415.37ms avg_case_ms=8.31 avg_simplify_ms=2.56, sum@700+100 failed=0 elapsed=238.94ms avg_case_ms=2.39 avg_simplify_ms=0.87
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.18ms median_wire=17.26ms median_wall=66.56ms, sum@0+100 #173 sum runs=3 median_simplify=15.88ms median_wire=15.94ms median_wall=59.59ms, product@0+100 #175 product runs=3 median_simplify=15.34ms median_wire=15.40ms median_wall=58.52ms, difference@0+50 #174 difference runs=3 median_simplify=15.82ms median_wire=15.87ms median_wall=59.10ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.37ms median_wire=13.45ms median_wall=50.15ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.98s | passed=450 failed=0 total=450 avg_case=6.622ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.76s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
