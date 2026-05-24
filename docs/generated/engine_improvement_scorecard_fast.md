# Engine Improvement Scorecard

- Generated: 2026-05-24T16:24:57.452093+00:00
- Git branch: main
- Git commit: `58df39ec2e0b26038f155f6d4da6c0fbac215cae`
- Profile: `fast`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=1
- By area: orchestrator / exact-zero additive composition:1
- Recent 1: `orchestrator / exact-zero additive composition` - 2026-05-24 - Discovery observe-only: hyperbolic angle-sum plus telescoping residual remains slow

## Calculus Contract Signal

- Dimension: public calculus behavior, result simplification, domain conditions, and step noise.
- Interpretation: small executable calculus vertical slices; failures should be classified before broadening pre-calculus rules.
- `diff`: passed=257 failed=0 ignored=1 filtered_out=0
- `diff` ignored tests: `inverse_reciprocal_trig_diff_evaluates_with_explicit_domain_conditions_exhaustive` (exhaustive inverse reciprocal trig diff contract is debug-slow; CI keeps representative structural smoke)
- `limit_compact`: passed=1 failed=0 ignored=0 filtered_out=142
- `limit_presimplify_safe`: passed=8 failed=0 ignored=0 filtered_out=0
- `integrate_compact`: passed=1 failed=0 ignored=0 filtered_out=337
- `residual_matrix`: passed=706 failed=0 total=706 slow=0 timeouts=0 total_bases=79 wrapped_bases=57 standalone_bases=22 wrappers=12 wrapped_cases=684 standalone_cases=22 conditioned_cases=699 distinct_conditions=20
- `residual_matrix` sparse expected conditions: -1 < x < 1=12, 1 - x^2=12, 4 - (x + 1)^2=12, cos(1 - 2·x)=12, sin(sqrt(3 - 2·x))=12
- `residual_matrix` domain expected conditions: -1 < x < 1=12, x < 3/2=12, x > 0=24, x > -1/3=52
- `residual_matrix` wrapper coverage: expected_wrapped_cases=684 missing_wrapped_pairs=0 full_wrapper_bases=57 partial_wrapper_bases=0 largest_gap=0

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_add_small` | `pass` | 3.51s | passed=1 failed=0 closure=100.0% nf=0 (0.0%) proved=435 (100.0%) timeouts=0 |
| `contextual_strict_fast` | `pass` | 56.32s | passed=1 failed=0 closure=100.0% nf=0 (0.0%) proved=64 (100.0%) timeouts=0 |
| `contextual_radical_fast` | `pass` | 0.20s | passed=1 failed=0 closure=100.0% nf=0 (0.0%) proved=15 (100.0%) timeouts=0 |
| `calculus_diff_contract` | `pass` | 25.33s | passed=257 failed=0 ignored=1 |
| `calculus_limit_compact_contract` | `pass` | 40.20s | passed=1 failed=0 |
| `calculus_limit_presimplify_contract` | `pass` | 0.89s | passed=8 failed=0 |
| `calculus_integrate_compact_contract` | `pass` | 13.12s | passed=1 failed=0 |
| `calculus_residual_matrix_smoke` | `pass` | 11.47s | passed=706 failed=0 total=706 conditioned=699 conditions=20 total_bases=79 wrapped_bases=57 standalone_bases=22 wrappers=12 missing_wrapped_pairs=0 partial_wrapper_bases=0 |
