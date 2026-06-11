# Engine Improvement Scorecard

- Generated: 2026-06-11T14:24:03.601396+00:00
- Git branch: main
- Git commit: `239019c9d84a25f2431b5d9472b5d67694d6fa2e`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=770.41ms avg_case_ms=7.70 simplify=217.57ms avg_simplify_ms=2.18, sum total=200 failed=0 elapsed=702.68ms avg_case_ms=3.51 simplify=233.08ms avg_simplify_ms=1.17, product total=100 failed=0 elapsed=469.18ms avg_case_ms=4.69 simplify=134.70ms avg_simplify_ms=1.35, difference total=50 failed=0 elapsed=324.21ms avg_case_ms=6.48 simplify=102.71ms avg_simplify_ms=2.05
- Engine hotspots: sum simplify=233.08ms avg_simplify_ms=1.17 wall=702.68ms, shifted_quotient simplify=217.57ms avg_simplify_ms=2.18 wall=770.41ms, product simplify=134.70ms avg_simplify_ms=1.35 wall=469.18ms, difference simplify=102.71ms avg_simplify_ms=2.05 wall=324.21ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=770.41ms avg_case_ms=7.70 avg_simplify_ms=2.18, sum@0+100 failed=0 elapsed=510.83ms avg_case_ms=5.11 avg_simplify_ms=1.62, product@0+100 failed=0 elapsed=469.18ms avg_case_ms=4.69 avg_simplify_ms=1.35, difference@0+50 failed=0 elapsed=324.21ms avg_case_ms=6.48 avg_simplify_ms=2.05, sum@700+100 failed=0 elapsed=191.85ms avg_case_ms=1.92 avg_simplify_ms=0.71
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=14.00ms median_wire=14.07ms median_wall=52.48ms, sum@0+100 #173 sum runs=3 median_simplify=11.51ms median_wire=11.55ms median_wall=43.90ms, product@0+100 #175 product runs=3 median_simplify=11.42ms median_wire=11.47ms median_wall=43.50ms, difference@0+50 #174 difference runs=3 median_simplify=11.66ms median_wire=11.71ms median_wall=44.52ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=11.54ms median_wire=11.61ms median_wall=42.56ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.27s | passed=450 failed=0 total=450 avg_case=5.044ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.81s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.10s | passed=1 failed=0 |
