# Engine Improvement Scorecard

- Generated: 2026-06-15T14:43:58.599517+00:00
- Git branch: main
- Git commit: `1e726da2bab1a0e7d28ac94ec046f129b73e5239`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=804.17ms avg_case_ms=8.04 simplify=229.64ms avg_simplify_ms=2.30, sum total=200 failed=0 elapsed=702.91ms avg_case_ms=3.51 simplify=234.48ms avg_simplify_ms=1.17, product total=100 failed=0 elapsed=518.14ms avg_case_ms=5.18 simplify=148.14ms avg_simplify_ms=1.48, difference total=50 failed=0 elapsed=343.33ms avg_case_ms=6.87 simplify=106.59ms avg_simplify_ms=2.13
- Engine hotspots: sum simplify=234.48ms avg_simplify_ms=1.17 wall=702.91ms, shifted_quotient simplify=229.64ms avg_simplify_ms=2.30 wall=804.17ms, product simplify=148.14ms avg_simplify_ms=1.48 wall=518.14ms, difference simplify=106.59ms avg_simplify_ms=2.13 wall=343.33ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=804.17ms avg_case_ms=8.04 avg_simplify_ms=2.30, product@0+100 failed=0 elapsed=518.14ms avg_case_ms=5.18 avg_simplify_ms=1.48, sum@0+100 failed=0 elapsed=508.76ms avg_case_ms=5.09 avg_simplify_ms=1.62, difference@0+50 failed=0 elapsed=343.33ms avg_case_ms=6.87 avg_simplify_ms=2.13, sum@700+100 failed=0 elapsed=194.15ms avg_case_ms=1.94 avg_simplify_ms=0.72
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.16ms median_wire=13.23ms median_wall=50.01ms, product@0+100 #175 product runs=3 median_simplify=11.97ms median_wire=12.03ms median_wall=45.51ms, difference@0+50 #174 difference runs=3 median_simplify=11.84ms median_wire=11.89ms median_wall=45.01ms, sum@0+100 #173 sum runs=3 median_simplify=11.67ms median_wire=11.72ms median_wall=44.51ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.64ms median_wire=10.72ms median_wall=40.53ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.37s | passed=450 failed=0 total=450 avg_case=5.267ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.84s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.93s | passed=1 failed=0 |
