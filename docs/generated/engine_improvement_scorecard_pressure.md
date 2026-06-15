# Engine Improvement Scorecard

- Generated: 2026-06-15T08:13:59.326118+00:00
- Git branch: main
- Git commit: `6d3b3c222c04b1988b5bc413715a40bbcafc1ac7`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=352

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=784.04ms avg_case_ms=7.84 simplify=223.49ms avg_simplify_ms=2.23, sum total=200 failed=0 elapsed=702.14ms avg_case_ms=3.51 simplify=235.05ms avg_simplify_ms=1.18, product total=100 failed=0 elapsed=479.82ms avg_case_ms=4.80 simplify=138.15ms avg_simplify_ms=1.38, difference total=50 failed=0 elapsed=326.20ms avg_case_ms=6.52 simplify=103.49ms avg_simplify_ms=2.07
- Engine hotspots: sum simplify=235.05ms avg_simplify_ms=1.18 wall=702.14ms, shifted_quotient simplify=223.49ms avg_simplify_ms=2.23 wall=784.04ms, product simplify=138.15ms avg_simplify_ms=1.38 wall=479.82ms, difference simplify=103.49ms avg_simplify_ms=2.07 wall=326.20ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=784.04ms avg_case_ms=7.84 avg_simplify_ms=2.23, sum@0+100 failed=0 elapsed=508.66ms avg_case_ms=5.09 avg_simplify_ms=1.63, product@0+100 failed=0 elapsed=479.82ms avg_case_ms=4.80 avg_simplify_ms=1.38, difference@0+50 failed=0 elapsed=326.20ms avg_case_ms=6.52 avg_simplify_ms=2.07, sum@700+100 failed=0 elapsed=193.48ms avg_case_ms=1.93 avg_simplify_ms=0.72
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.10ms median_wire=13.17ms median_wall=49.62ms, sum@0+100 #173 sum runs=3 median_simplify=11.64ms median_wire=11.70ms median_wall=44.12ms, product@0+100 #175 product runs=3 median_simplify=11.59ms median_wire=11.64ms median_wall=43.87ms, difference@0+50 #174 difference runs=3 median_simplify=11.52ms median_wire=11.57ms median_wall=43.67ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.73ms median_wire=10.80ms median_wall=40.21ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.29s | passed=450 failed=0 total=450 avg_case=5.089ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.83s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.94s | passed=1 failed=0 |
