# Engine Improvement Scorecard

- Generated: 2026-07-19T15:31:14.997161+00:00
- Git branch: main
- Git commit: `d32614eee90864bfd849230b71ad83dcd02c3c0b`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=1.02s avg_case_ms=10.21 simplify=288.38ms avg_simplify_ms=2.88, sum total=200 failed=0 elapsed=910.89ms avg_case_ms=4.55 simplify=298.78ms avg_simplify_ms=1.49, product total=100 failed=0 elapsed=630.48ms avg_case_ms=6.30 simplify=181.76ms avg_simplify_ms=1.82, difference total=50 failed=0 elapsed=413.77ms avg_case_ms=8.28 simplify=127.20ms avg_simplify_ms=2.54
- Engine hotspots: sum simplify=298.78ms avg_simplify_ms=1.49 wall=910.89ms, shifted_quotient simplify=288.38ms avg_simplify_ms=2.88 wall=1.02s, product simplify=181.76ms avg_simplify_ms=1.82 wall=630.48ms, difference simplify=127.20ms avg_simplify_ms=2.54 wall=413.77ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=1.02s avg_case_ms=10.21 avg_simplify_ms=2.88, sum@0+100 failed=0 elapsed=670.19ms avg_case_ms=6.70 avg_simplify_ms=2.11, product@0+100 failed=0 elapsed=630.48ms avg_case_ms=6.30 avg_simplify_ms=1.82, difference@0+50 failed=0 elapsed=413.77ms avg_case_ms=8.28 avg_simplify_ms=2.54, sum@700+100 failed=0 elapsed=240.71ms avg_case_ms=2.41 avg_simplify_ms=0.88
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.94ms median_wire=17.02ms median_wall=65.44ms, sum@0+100 #173 sum runs=3 median_simplify=16.01ms median_wire=16.06ms median_wall=60.85ms, difference@0+50 #174 difference runs=3 median_simplify=15.93ms median_wire=15.98ms median_wall=61.17ms, product@0+100 #175 product runs=3 median_simplify=15.40ms median_wire=15.46ms median_wall=59.21ms, shifted_quotient@0+100 #112 shifted_quotient runs=3 median_simplify=9.74ms median_wire=9.80ms median_wall=37.47ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.98s | passed=450 failed=0 total=450 avg_case=6.622ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.98s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.97s | passed=1 failed=0 |
