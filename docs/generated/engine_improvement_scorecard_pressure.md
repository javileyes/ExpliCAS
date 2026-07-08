# Engine Improvement Scorecard

- Generated: 2026-07-08T09:08:48.293758+00:00
- Git branch: main
- Git commit: `a959766f32276f272cb7cab27ee85c0ac85d404d`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=357

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=949.58ms avg_case_ms=9.50 simplify=264.08ms avg_simplify_ms=2.64, sum total=200 failed=0 elapsed=829.71ms avg_case_ms=4.15 simplify=267.74ms avg_simplify_ms=1.34, product total=100 failed=0 elapsed=586.93ms avg_case_ms=5.87 simplify=166.79ms avg_simplify_ms=1.67, difference total=50 failed=0 elapsed=395.71ms avg_case_ms=7.91 simplify=119.28ms avg_simplify_ms=2.39
- Engine hotspots: sum simplify=267.74ms avg_simplify_ms=1.34 wall=829.71ms, shifted_quotient simplify=264.08ms avg_simplify_ms=2.64 wall=949.58ms, product simplify=166.79ms avg_simplify_ms=1.67 wall=586.93ms, difference simplify=119.28ms avg_simplify_ms=2.39 wall=395.71ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=949.58ms avg_case_ms=9.50 avg_simplify_ms=2.64, sum@0+100 failed=0 elapsed=605.56ms avg_case_ms=6.06 avg_simplify_ms=1.88, product@0+100 failed=0 elapsed=586.93ms avg_case_ms=5.87 avg_simplify_ms=1.67, difference@0+50 failed=0 elapsed=395.71ms avg_case_ms=7.91 avg_simplify_ms=2.39, sum@700+100 failed=0 elapsed=224.16ms avg_case_ms=2.24 avg_simplify_ms=0.80
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.59ms median_wire=16.65ms median_wall=63.27ms, difference@0+50 #174 difference runs=3 median_simplify=15.02ms median_wire=15.06ms median_wall=57.82ms, sum@0+100 #173 sum runs=3 median_simplify=14.88ms median_wire=14.92ms median_wall=57.35ms, product@0+100 #175 product runs=3 median_simplify=14.90ms median_wire=14.94ms median_wall=57.66ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.32ms median_wire=12.39ms median_wall=47.52ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.76s | passed=450 failed=0 total=450 avg_case=6.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.04s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.87s | passed=1 failed=0 |
