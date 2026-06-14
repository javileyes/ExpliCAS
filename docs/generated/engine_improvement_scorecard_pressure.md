# Engine Improvement Scorecard

- Generated: 2026-06-14T22:02:45.941994+00:00
- Git branch: main
- Git commit: `8550154e399c5f1f073683060407dad5367cd7b0`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=351

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=779.58ms avg_case_ms=7.80 simplify=222.27ms avg_simplify_ms=2.22, sum total=200 failed=0 elapsed=690.73ms avg_case_ms=3.45 simplify=230.94ms avg_simplify_ms=1.15, product total=100 failed=0 elapsed=479.95ms avg_case_ms=4.80 simplify=138.10ms avg_simplify_ms=1.38, difference total=50 failed=0 elapsed=331.30ms avg_case_ms=6.63 simplify=102.91ms avg_simplify_ms=2.06
- Engine hotspots: sum simplify=230.94ms avg_simplify_ms=1.15 wall=690.73ms, shifted_quotient simplify=222.27ms avg_simplify_ms=2.22 wall=779.58ms, product simplify=138.10ms avg_simplify_ms=1.38 wall=479.95ms, difference simplify=102.91ms avg_simplify_ms=2.06 wall=331.30ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=779.58ms avg_case_ms=7.80 avg_simplify_ms=2.22, sum@0+100 failed=0 elapsed=499.44ms avg_case_ms=4.99 avg_simplify_ms=1.60, product@0+100 failed=0 elapsed=479.95ms avg_case_ms=4.80 avg_simplify_ms=1.38, difference@0+50 failed=0 elapsed=331.30ms avg_case_ms=6.63 avg_simplify_ms=2.06, sum@700+100 failed=0 elapsed=191.29ms avg_case_ms=1.91 avg_simplify_ms=0.71
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.00ms median_wire=13.07ms median_wall=49.44ms, product@0+100 #175 product runs=3 median_simplify=11.54ms median_wire=11.59ms median_wall=43.89ms, difference@0+50 #174 difference runs=3 median_simplify=11.68ms median_wire=11.72ms median_wall=44.45ms, sum@0+100 #173 sum runs=3 median_simplify=11.59ms median_wire=11.64ms median_wall=44.24ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.70ms median_wire=10.78ms median_wall=40.79ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.28s | passed=450 failed=0 total=450 avg_case=5.067ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.83s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
