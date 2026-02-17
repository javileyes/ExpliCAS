//! Transport DTO facade for JSON APIs.
//!
//! Canonical model definitions live in `cas_api_models`.
//! This module keeps `cas_engine::json::*` backward-compatible during migration.

pub use cas_api_models::{
    AssumptionDto, BlockedHintDto, BoundDto, CaseDto, ConditionDto, DomainJson, EngineInfo,
    ErrorJsonOutput, EvalJsonOutput, ExprDto, ExprStatsJson, OptionsJson, OutputEnvelope,
    RequestInfo, RequestOptions, RequiredConditionJson, ResultDto, SemanticsJson, SolutionSetDto,
    SolveStepJson, SolveSubStepJson, StepDto, StepJson, SubStepJson, ThenDto, TimingsJson,
    TransparencyDto, WarningJson, WhenDto,
};
