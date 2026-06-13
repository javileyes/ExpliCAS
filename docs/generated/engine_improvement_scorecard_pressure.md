# Engine Improvement Scorecard

- Generated: 2026-06-13T17:12:09.686959+00:00
- Git branch: main
- Git commit: `87800416613cff07e8d42bc93b52ad767c04123a`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=344

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=780.45ms avg_case_ms=7.80 simplify=222.14ms avg_simplify_ms=2.22, sum total=200 failed=0 elapsed=701.92ms avg_case_ms=3.51 simplify=235.60ms avg_simplify_ms=1.18, product total=100 failed=0 elapsed=482.23ms avg_case_ms=4.82 simplify=137.85ms avg_simplify_ms=1.38, difference total=50 failed=0 elapsed=334.30ms avg_case_ms=6.69 simplify=105.39ms avg_simplify_ms=2.11
- Engine hotspots: sum simplify=235.60ms avg_simplify_ms=1.18 wall=701.92ms, shifted_quotient simplify=222.14ms avg_simplify_ms=2.22 wall=780.45ms, product simplify=137.85ms avg_simplify_ms=1.38 wall=482.23ms, difference simplify=105.39ms avg_simplify_ms=2.11 wall=334.30ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=780.45ms avg_case_ms=7.80 avg_simplify_ms=2.22, sum@0+100 failed=0 elapsed=511.62ms avg_case_ms=5.12 avg_simplify_ms=1.65, product@0+100 failed=0 elapsed=482.23ms avg_case_ms=4.82 avg_simplify_ms=1.38, difference@0+50 failed=0 elapsed=334.30ms avg_case_ms=6.69 avg_simplify_ms=2.11, sum@700+100 failed=0 elapsed=190.30ms avg_case_ms=1.90 avg_simplify_ms=0.71
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.79ms median_wire=12.86ms median_wall=48.43ms, sum@0+100 #173 sum runs=3 median_simplify=11.60ms median_wire=11.64ms median_wall=44.14ms, difference@0+50 #174 difference runs=3 median_simplify=11.76ms median_wire=11.81ms median_wall=44.15ms, product@0+100 #175 product runs=3 median_simplify=11.48ms median_wire=11.53ms median_wall=43.67ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=11.50ms median_wire=11.59ms median_wall=43.99ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.30s | passed=450 failed=0 total=450 avg_case=5.111ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.81s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.93s | passed=1 failed=0 |
