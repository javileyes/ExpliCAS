# Engine Improvement Scorecard

- Generated: 2026-07-28T12:39:53.906896+00:00
- Git branch: main
- Git commit: `97a8ae7f618fce1e8656a34713af8bd7b991b82c`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=982.29ms avg_case_ms=9.82 simplify=276.35ms avg_simplify_ms=2.76, sum total=200 failed=0 elapsed=867.39ms avg_case_ms=4.34 simplify=283.24ms avg_simplify_ms=1.42, product total=100 failed=0 elapsed=606.49ms avg_case_ms=6.06 simplify=174.62ms avg_simplify_ms=1.75, difference total=50 failed=0 elapsed=403.55ms avg_case_ms=8.07 simplify=122.28ms avg_simplify_ms=2.45
- Engine hotspots: sum simplify=283.24ms avg_simplify_ms=1.42 wall=867.39ms, shifted_quotient simplify=276.35ms avg_simplify_ms=2.76 wall=982.29ms, product simplify=174.62ms avg_simplify_ms=1.75 wall=606.49ms, difference simplify=122.28ms avg_simplify_ms=2.45 wall=403.55ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=982.29ms avg_case_ms=9.82 avg_simplify_ms=2.76, sum@0+100 failed=0 elapsed=634.59ms avg_case_ms=6.35 avg_simplify_ms=2.00, product@0+100 failed=0 elapsed=606.49ms avg_case_ms=6.06 avg_simplify_ms=1.75, difference@0+50 failed=0 elapsed=403.55ms avg_case_ms=8.07 avg_simplify_ms=2.45, sum@700+100 failed=0 elapsed=232.79ms avg_case_ms=2.33 avg_simplify_ms=0.84
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.05ms median_wire=17.12ms median_wall=67.32ms, product@0+100 #175 product runs=3 median_simplify=15.19ms median_wire=15.24ms median_wall=57.64ms, sum@0+100 #173 sum runs=3 median_simplify=15.21ms median_wire=15.26ms median_wall=58.13ms, difference@0+50 #174 difference runs=3 median_simplify=15.52ms median_wire=15.58ms median_wall=59.29ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.04ms median_wire=13.12ms median_wall=49.34ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.86s | passed=450 failed=0 total=450 avg_case=6.356ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.29s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.85s | passed=1 failed=0 |
