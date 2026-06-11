# Engine Improvement Scorecard

- Generated: 2026-06-11T06:40:57.811644+00:00
- Git branch: main
- Git commit: `fbd163ca3c0b4d23d9a91896f5b8d978d23a1f1b`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=772.47ms avg_case_ms=7.72 simplify=218.07ms avg_simplify_ms=2.18, sum total=200 failed=0 elapsed=704.87ms avg_case_ms=3.52 simplify=235.79ms avg_simplify_ms=1.18, product total=100 failed=0 elapsed=470.26ms avg_case_ms=4.70 simplify=135.05ms avg_simplify_ms=1.35, difference total=50 failed=0 elapsed=319.54ms avg_case_ms=6.39 simplify=101.42ms avg_simplify_ms=2.03
- Engine hotspots: sum simplify=235.79ms avg_simplify_ms=1.18 wall=704.87ms, shifted_quotient simplify=218.07ms avg_simplify_ms=2.18 wall=772.47ms, product simplify=135.05ms avg_simplify_ms=1.35 wall=470.26ms, difference simplify=101.42ms avg_simplify_ms=2.03 wall=319.54ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=772.47ms avg_case_ms=7.72 avg_simplify_ms=2.18, sum@0+100 failed=0 elapsed=513.87ms avg_case_ms=5.14 avg_simplify_ms=1.66, product@0+100 failed=0 elapsed=470.26ms avg_case_ms=4.70 avg_simplify_ms=1.35, difference@0+50 failed=0 elapsed=319.54ms avg_case_ms=6.39 avg_simplify_ms=2.03, sum@700+100 failed=0 elapsed=191.01ms avg_case_ms=1.91 avg_simplify_ms=0.70
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.43ms median_wire=12.50ms median_wall=48.21ms, product@0+100 #175 product runs=3 median_simplify=11.37ms median_wire=11.42ms median_wall=43.08ms, sum@0+100 #173 sum runs=3 median_simplify=11.64ms median_wire=11.68ms median_wall=44.40ms, difference@0+50 #174 difference runs=3 median_simplify=11.59ms median_wire=11.64ms median_wall=44.11ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.18ms median_wire=12.27ms median_wall=44.23ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.27s | passed=450 failed=0 total=450 avg_case=5.044ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.82s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.15s | passed=1 failed=0 |
