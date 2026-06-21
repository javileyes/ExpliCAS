# Engine Improvement Scorecard

- Generated: 2026-06-21T20:19:06.397233+00:00
- Git branch: main
- Git commit: `44d5be5d6673af76e561eafb6f210cbdd495f276`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=794.68ms avg_case_ms=7.95 simplify=226.59ms avg_simplify_ms=2.27, sum total=200 failed=0 elapsed=702.00ms avg_case_ms=3.51 simplify=237.83ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=478.91ms avg_case_ms=4.79 simplify=138.15ms avg_simplify_ms=1.38, difference total=50 failed=0 elapsed=329.11ms avg_case_ms=6.58 simplify=105.00ms avg_simplify_ms=2.10
- Engine hotspots: sum simplify=237.83ms avg_simplify_ms=1.19 wall=702.00ms, shifted_quotient simplify=226.59ms avg_simplify_ms=2.27 wall=794.68ms, product simplify=138.15ms avg_simplify_ms=1.38 wall=478.91ms, difference simplify=105.00ms avg_simplify_ms=2.10 wall=329.11ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=794.68ms avg_case_ms=7.95 avg_simplify_ms=2.27, sum@0+100 failed=0 elapsed=506.54ms avg_case_ms=5.07 avg_simplify_ms=1.65, product@0+100 failed=0 elapsed=478.91ms avg_case_ms=4.79 avg_simplify_ms=1.38, difference@0+50 failed=0 elapsed=329.11ms avg_case_ms=6.58 avg_simplify_ms=2.10, sum@700+100 failed=0 elapsed=195.46ms avg_case_ms=1.95 avg_simplify_ms=0.73
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.87ms median_wire=13.95ms median_wall=53.02ms, sum@0+100 #173 sum runs=3 median_simplify=11.79ms median_wire=11.85ms median_wall=44.27ms, product@0+100 #175 product runs=3 median_simplify=11.76ms median_wire=11.82ms median_wall=44.65ms, difference@0+50 #174 difference runs=3 median_simplify=11.88ms median_wire=11.93ms median_wall=44.83ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.76ms median_wire=10.83ms median_wall=40.70ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.31s | passed=450 failed=0 total=450 avg_case=5.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.42s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.04s | passed=1 failed=0 |
