# Engine Improvement Scorecard

- Generated: 2026-07-30T16:27:20.238267+00:00
- Git branch: main
- Git commit: `1b86559e4e4472d17a73235dabf6bead3843008b`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=979.30ms avg_case_ms=9.79 simplify=277.32ms avg_simplify_ms=2.77, sum total=200 failed=0 elapsed=853.53ms avg_case_ms=4.27 simplify=275.17ms avg_simplify_ms=1.38, product total=100 failed=0 elapsed=601.05ms avg_case_ms=6.01 simplify=172.94ms avg_simplify_ms=1.73, difference total=50 failed=0 elapsed=393.79ms avg_case_ms=7.88 simplify=115.06ms avg_simplify_ms=2.30
- Engine hotspots: shifted_quotient simplify=277.32ms avg_simplify_ms=2.77 wall=979.30ms, sum simplify=275.17ms avg_simplify_ms=1.38 wall=853.53ms, product simplify=172.94ms avg_simplify_ms=1.73 wall=601.05ms, difference simplify=115.06ms avg_simplify_ms=2.30 wall=393.79ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=979.30ms avg_case_ms=9.79 avg_simplify_ms=2.77, sum@0+100 failed=0 elapsed=619.58ms avg_case_ms=6.20 avg_simplify_ms=1.91, product@0+100 failed=0 elapsed=601.05ms avg_case_ms=6.01 avg_simplify_ms=1.73, difference@0+50 failed=0 elapsed=393.79ms avg_case_ms=7.88 avg_simplify_ms=2.30, sum@700+100 failed=0 elapsed=233.94ms avg_case_ms=2.34 avg_simplify_ms=0.84
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.64ms median_wire=16.71ms median_wall=63.94ms, sum@0+100 #173 sum runs=3 median_simplify=15.18ms median_wire=15.23ms median_wall=57.70ms, product@0+100 #175 product runs=3 median_simplify=15.10ms median_wire=15.16ms median_wall=57.72ms, difference@0+50 #174 difference runs=3 median_simplify=15.33ms median_wire=15.39ms median_wall=57.50ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.10ms median_wire=13.18ms median_wall=50.32ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.83s | passed=450 failed=0 total=450 avg_case=6.289ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.38s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.74s | passed=1 failed=0 |
