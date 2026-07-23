# Engine Improvement Scorecard

- Generated: 2026-07-23T11:03:32.232819+00:00
- Git branch: main
- Git commit: `c319096f9ec2bb64e9d088298299ca4ed34bab64`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=981.65ms avg_case_ms=9.82 simplify=276.92ms avg_simplify_ms=2.77, sum total=200 failed=0 elapsed=883.35ms avg_case_ms=4.42 simplify=285.40ms avg_simplify_ms=1.43, product total=100 failed=0 elapsed=609.92ms avg_case_ms=6.10 simplify=175.87ms avg_simplify_ms=1.76, difference total=50 failed=0 elapsed=401.63ms avg_case_ms=8.03 simplify=122.68ms avg_simplify_ms=2.45
- Engine hotspots: sum simplify=285.40ms avg_simplify_ms=1.43 wall=883.35ms, shifted_quotient simplify=276.92ms avg_simplify_ms=2.77 wall=981.65ms, product simplify=175.87ms avg_simplify_ms=1.76 wall=609.92ms, difference simplify=122.68ms avg_simplify_ms=2.45 wall=401.63ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=981.65ms avg_case_ms=9.82 avg_simplify_ms=2.77, sum@0+100 failed=0 elapsed=650.26ms avg_case_ms=6.50 avg_simplify_ms=2.01, product@0+100 failed=0 elapsed=609.92ms avg_case_ms=6.10 avg_simplify_ms=1.76, difference@0+50 failed=0 elapsed=401.63ms avg_case_ms=8.03 avg_simplify_ms=2.45, sum@700+100 failed=0 elapsed=233.09ms avg_case_ms=2.33 avg_simplify_ms=0.84
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.81ms median_wire=16.88ms median_wall=63.81ms, sum@0+100 #173 sum runs=3 median_simplify=15.13ms median_wire=15.18ms median_wall=57.90ms, difference@0+50 #174 difference runs=3 median_simplify=15.14ms median_wire=15.18ms median_wall=58.03ms, product@0+100 #175 product runs=3 median_simplify=15.50ms median_wire=15.56ms median_wall=58.00ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.11ms median_wire=12.18ms median_wall=46.88ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.88s | passed=450 failed=0 total=450 avg_case=6.400ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.58s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.91s | passed=1 failed=0 |
