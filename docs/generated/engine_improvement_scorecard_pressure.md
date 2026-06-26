# Engine Improvement Scorecard

- Generated: 2026-06-26T11:29:28.172138+00:00
- Git branch: main
- Git commit: `70226975dc0a0d854c4dd0caffb82a8e0eb956a4`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=353

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=790.34ms avg_case_ms=7.90 simplify=226.53ms avg_simplify_ms=2.27, sum total=200 failed=0 elapsed=730.51ms avg_case_ms=3.65 simplify=253.07ms avg_simplify_ms=1.27, product total=100 failed=0 elapsed=484.37ms avg_case_ms=4.84 simplify=142.42ms avg_simplify_ms=1.42, difference total=50 failed=0 elapsed=331.14ms avg_case_ms=6.62 simplify=106.83ms avg_simplify_ms=2.14
- Engine hotspots: sum simplify=253.07ms avg_simplify_ms=1.27 wall=730.51ms, shifted_quotient simplify=226.53ms avg_simplify_ms=2.27 wall=790.34ms, product simplify=142.42ms avg_simplify_ms=1.42 wall=484.37ms, difference simplify=106.83ms avg_simplify_ms=2.14 wall=331.14ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=790.34ms avg_case_ms=7.90 avg_simplify_ms=2.27, sum@0+100 failed=0 elapsed=520.39ms avg_case_ms=5.20 avg_simplify_ms=1.71, product@0+100 failed=0 elapsed=484.37ms avg_case_ms=4.84 avg_simplify_ms=1.42, difference@0+50 failed=0 elapsed=331.14ms avg_case_ms=6.62 avg_simplify_ms=2.14, sum@700+100 failed=0 elapsed=210.12ms avg_case_ms=2.10 avg_simplify_ms=0.82
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.18ms median_wire=13.25ms median_wall=50.23ms, sum@0+100 #173 sum runs=3 median_simplify=11.76ms median_wire=11.81ms median_wall=44.24ms, difference@0+50 #174 difference runs=3 median_simplify=11.79ms median_wire=11.85ms median_wall=44.64ms, product@0+100 #175 product runs=3 median_simplify=11.65ms median_wire=11.70ms median_wall=44.40ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.64ms median_wire=10.72ms median_wall=40.15ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.34s | passed=450 failed=0 total=450 avg_case=5.200ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.44s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.05s | passed=1 failed=0 |
