# Engine Improvement Scorecard

- Generated: 2026-07-19T00:34:26.044151+00:00
- Git branch: main
- Git commit: `b1d19ea9979dc542bf56371c4770f70415b7197c`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=987.85ms avg_case_ms=9.88 simplify=273.06ms avg_simplify_ms=2.73, sum total=200 failed=0 elapsed=877.45ms avg_case_ms=4.39 simplify=285.48ms avg_simplify_ms=1.43, product total=100 failed=0 elapsed=603.95ms avg_case_ms=6.04 simplify=173.58ms avg_simplify_ms=1.74, difference total=50 failed=0 elapsed=396.14ms avg_case_ms=7.92 simplify=120.88ms avg_simplify_ms=2.42
- Engine hotspots: sum simplify=285.48ms avg_simplify_ms=1.43 wall=877.45ms, shifted_quotient simplify=273.06ms avg_simplify_ms=2.73 wall=987.85ms, product simplify=173.58ms avg_simplify_ms=1.74 wall=603.95ms, difference simplify=120.88ms avg_simplify_ms=2.42 wall=396.14ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=987.85ms avg_case_ms=9.88 avg_simplify_ms=2.73, sum@0+100 failed=0 elapsed=648.69ms avg_case_ms=6.49 avg_simplify_ms=2.03, product@0+100 failed=0 elapsed=603.95ms avg_case_ms=6.04 avg_simplify_ms=1.74, difference@0+50 failed=0 elapsed=396.14ms avg_case_ms=7.92 avg_simplify_ms=2.42, sum@700+100 failed=0 elapsed=228.76ms avg_case_ms=2.29 avg_simplify_ms=0.82
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=15.23ms median_wire=15.28ms median_wall=58.19ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.95ms median_wire=17.01ms median_wall=64.56ms, difference@0+50 #174 difference runs=3 median_simplify=15.08ms median_wire=15.12ms median_wall=58.27ms, product@0+100 #175 product runs=3 median_simplify=16.40ms median_wire=16.45ms median_wall=58.47ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.89ms median_wire=12.95ms median_wall=49.05ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.87s | passed=450 failed=0 total=450 avg_case=6.378ms |
| `calculus_diff_exhaustive_contract` | `pass` | 13.28s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.93s | passed=1 failed=0 |
