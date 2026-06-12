# Engine Improvement Scorecard

- Generated: 2026-06-12T02:08:25.126690+00:00
- Git branch: main
- Git commit: `e5ca20f00aae5893512197d55da9c272ce2cf326`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=784.58ms avg_case_ms=7.85 simplify=221.76ms avg_simplify_ms=2.22, sum total=200 failed=0 elapsed=701.98ms avg_case_ms=3.51 simplify=234.99ms avg_simplify_ms=1.17, product total=100 failed=0 elapsed=476.48ms avg_case_ms=4.76 simplify=136.74ms avg_simplify_ms=1.37, difference total=50 failed=0 elapsed=334.81ms avg_case_ms=6.70 simplify=105.98ms avg_simplify_ms=2.12
- Engine hotspots: sum simplify=234.99ms avg_simplify_ms=1.17 wall=701.98ms, shifted_quotient simplify=221.76ms avg_simplify_ms=2.22 wall=784.58ms, product simplify=136.74ms avg_simplify_ms=1.37 wall=476.48ms, difference simplify=105.98ms avg_simplify_ms=2.12 wall=334.81ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=784.58ms avg_case_ms=7.85 avg_simplify_ms=2.22, sum@0+100 failed=0 elapsed=507.48ms avg_case_ms=5.07 avg_simplify_ms=1.63, product@0+100 failed=0 elapsed=476.48ms avg_case_ms=4.76 avg_simplify_ms=1.37, difference@0+50 failed=0 elapsed=334.81ms avg_case_ms=6.70 avg_simplify_ms=2.12, sum@700+100 failed=0 elapsed=194.50ms avg_case_ms=1.94 avg_simplify_ms=0.72
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.98ms median_wire=13.05ms median_wall=49.56ms, difference@0+50 #174 difference runs=3 median_simplify=11.76ms median_wire=11.81ms median_wall=44.24ms, sum@0+100 #173 sum runs=3 median_simplify=11.62ms median_wire=11.66ms median_wall=44.35ms, product@0+100 #175 product runs=3 median_simplify=11.60ms median_wire=11.65ms median_wall=44.78ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.57ms median_wire=10.65ms median_wall=40.17ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.30s | passed=450 failed=0 total=450 avg_case=5.111ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.81s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.33s | passed=1 failed=0 |
