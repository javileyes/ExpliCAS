# Engine Improvement Scorecard

- Generated: 2026-06-12T00:54:44.328668+00:00
- Git branch: main
- Git commit: `56d4726fc4d39cc1afef00569943ba0416384799`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=773.89ms avg_case_ms=7.74 simplify=218.10ms avg_simplify_ms=2.18, sum total=200 failed=0 elapsed=685.65ms avg_case_ms=3.43 simplify=230.08ms avg_simplify_ms=1.15, product total=100 failed=0 elapsed=468.26ms avg_case_ms=4.68 simplify=134.27ms avg_simplify_ms=1.34, difference total=50 failed=0 elapsed=324.91ms avg_case_ms=6.50 simplify=103.24ms avg_simplify_ms=2.06
- Engine hotspots: sum simplify=230.08ms avg_simplify_ms=1.15 wall=685.65ms, shifted_quotient simplify=218.10ms avg_simplify_ms=2.18 wall=773.89ms, product simplify=134.27ms avg_simplify_ms=1.34 wall=468.26ms, difference simplify=103.24ms avg_simplify_ms=2.06 wall=324.91ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=773.89ms avg_case_ms=7.74 avg_simplify_ms=2.18, sum@0+100 failed=0 elapsed=498.10ms avg_case_ms=4.98 avg_simplify_ms=1.61, product@0+100 failed=0 elapsed=468.26ms avg_case_ms=4.68 avg_simplify_ms=1.34, difference@0+50 failed=0 elapsed=324.91ms avg_case_ms=6.50 avg_simplify_ms=2.06, sum@700+100 failed=0 elapsed=187.56ms avg_case_ms=1.88 avg_simplify_ms=0.69
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.81ms median_wire=12.88ms median_wall=48.62ms, sum@0+100 #173 sum runs=3 median_simplify=11.41ms median_wire=11.45ms median_wall=43.63ms, difference@0+50 #174 difference runs=3 median_simplify=11.47ms median_wire=11.51ms median_wall=43.78ms, product@0+100 #175 product runs=3 median_simplify=11.44ms median_wire=11.48ms median_wall=43.69ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.54ms median_wire=10.61ms median_wall=40.03ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.25s | passed=450 failed=0 total=450 avg_case=5.000ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.83s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.19s | passed=1 failed=0 |
