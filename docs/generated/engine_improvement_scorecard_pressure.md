# Engine Improvement Scorecard

- Generated: 2026-07-18T06:08:51.876089+00:00
- Git branch: main
- Git commit: `63c3fd156320c366ecd9be744d739803d7a3fe48`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=972.21ms avg_case_ms=9.72 simplify=270.44ms avg_simplify_ms=2.70, sum total=200 failed=0 elapsed=902.79ms avg_case_ms=4.51 simplify=289.83ms avg_simplify_ms=1.45, product total=100 failed=0 elapsed=615.01ms avg_case_ms=6.15 simplify=176.98ms avg_simplify_ms=1.77, difference total=50 failed=0 elapsed=404.00ms avg_case_ms=8.08 simplify=123.09ms avg_simplify_ms=2.46
- Engine hotspots: sum simplify=289.83ms avg_simplify_ms=1.45 wall=902.79ms, shifted_quotient simplify=270.44ms avg_simplify_ms=2.70 wall=972.21ms, product simplify=176.98ms avg_simplify_ms=1.77 wall=615.01ms, difference simplify=123.09ms avg_simplify_ms=2.46 wall=404.00ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=972.21ms avg_case_ms=9.72 avg_simplify_ms=2.70, sum@0+100 failed=0 elapsed=668.12ms avg_case_ms=6.68 avg_simplify_ms=2.06, product@0+100 failed=0 elapsed=615.01ms avg_case_ms=6.15 avg_simplify_ms=1.77, difference@0+50 failed=0 elapsed=404.00ms avg_case_ms=8.08 avg_simplify_ms=2.46, sum@700+100 failed=0 elapsed=234.67ms avg_case_ms=2.35 avg_simplify_ms=0.84
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.92ms median_wire=16.99ms median_wall=64.40ms, sum@0+100 #173 sum runs=3 median_simplify=15.53ms median_wire=15.58ms median_wall=59.96ms, difference@0+50 #174 difference runs=3 median_simplify=15.55ms median_wire=15.61ms median_wall=59.82ms, product@0+100 #175 product runs=3 median_simplify=15.42ms median_wire=15.48ms median_wall=59.04ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.01ms median_wire=13.10ms median_wall=50.07ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.90s | passed=450 failed=0 total=450 avg_case=6.444ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.52s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.93s | passed=1 failed=0 |
