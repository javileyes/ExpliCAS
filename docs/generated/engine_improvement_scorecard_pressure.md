# Engine Improvement Scorecard

- Generated: 2026-06-26T13:40:44.117057+00:00
- Git branch: main
- Git commit: `671c4b20e21a4d7ae45672b15b65cbb85ad69e35`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=794.75ms avg_case_ms=7.95 simplify=226.88ms avg_simplify_ms=2.27, sum total=200 failed=0 elapsed=713.18ms avg_case_ms=3.57 simplify=246.10ms avg_simplify_ms=1.23, product total=100 failed=0 elapsed=486.39ms avg_case_ms=4.86 simplify=142.98ms avg_simplify_ms=1.43, difference total=50 failed=0 elapsed=331.96ms avg_case_ms=6.64 simplify=107.15ms avg_simplify_ms=2.14
- Engine hotspots: sum simplify=246.10ms avg_simplify_ms=1.23 wall=713.18ms, shifted_quotient simplify=226.88ms avg_simplify_ms=2.27 wall=794.75ms, product simplify=142.98ms avg_simplify_ms=1.43 wall=486.39ms, difference simplify=107.15ms avg_simplify_ms=2.14 wall=331.96ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=794.75ms avg_case_ms=7.95 avg_simplify_ms=2.27, sum@0+100 failed=0 elapsed=515.87ms avg_case_ms=5.16 avg_simplify_ms=1.70, product@0+100 failed=0 elapsed=486.39ms avg_case_ms=4.86 avg_simplify_ms=1.43, difference@0+50 failed=0 elapsed=331.96ms avg_case_ms=6.64 avg_simplify_ms=2.14, sum@700+100 failed=0 elapsed=197.31ms avg_case_ms=1.97 avg_simplify_ms=0.76
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.07ms median_wire=13.14ms median_wall=49.74ms, sum@0+100 #173 sum runs=3 median_simplify=11.76ms median_wire=11.82ms median_wall=44.86ms, product@0+100 #175 product runs=3 median_simplify=11.80ms median_wire=11.85ms median_wall=44.96ms, difference@0+50 #174 difference runs=3 median_simplify=11.82ms median_wire=11.87ms median_wall=45.04ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.87ms median_wire=10.95ms median_wall=41.41ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.33s | passed=450 failed=0 total=450 avg_case=5.178ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.46s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.05s | passed=1 failed=0 |
