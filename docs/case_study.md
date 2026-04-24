# Case Study: Support Ticket Routing System

## Business problem
Support organizations need fast first-touch routing so tickets reach the right specialist queue quickly. Pure manual triage is expensive and inconsistent; pure automation is risky for ambiguous cases.

## System design
This project implements a staged routing cascade:
1. Rule-based patterns for deterministic fast-path tickets.
2. Calibrated ML classifier for high-confidence auto-routing.
3. LLM classification only for low-confidence cases.
4. Human triage fallback for uncertainty and LLM failures.

This design keeps high-confidence automation while preserving safety for edge cases.

## Why not LLM-only
An LLM-only approach can increase cost variance, latency variance, and operational unpredictability. Here, LLM is scoped to the low-confidence segment so deterministic and ML paths handle the majority of traffic where possible.

## Data strategy
- Inbound customer-authored Twitter messages only (agent messages filtered out).
- Structured ticket dataset mapped from `Ticket Type` and `Ticket Priority` into issue/urgency labels.
- `label_source` captures real vs weak label provenance.
- Deduplication prioritizes real-labeled rows when duplicate descriptions exist.

## Evaluation strategy
- Train/test and cross-validation metrics during model training.
- Measured eval set comparison: ML vs keyword baseline.
- Per-class metrics and confusion matrix artifacts for class-level visibility.
- Explicit separation of measured metrics vs estimated policy metrics.

## Operational tradeoffs
- Threshold sweep estimates coverage/cost/human-triage tradeoffs.
- A recommendation score is provided as an analytic policy guide (not a guaranteed optimum).
- Estimated cost depends on model/pricing assumptions and should be revisited for production.

## Limitations
- Complexity label is a heuristic proxy, not hand-labeled ground truth.
- Estimated costs and threshold policy simulations are not online A/B outcomes.
- Current evaluation depends on available labeled eval samples; larger gold sets improve reliability.

## Next steps
1. Expand gold-labeled eval set and add queue-level SLA-weighted metrics.
2. Add reliability/calibration plots to pair with threshold policy guidance.
3. Add run metadata/versioning for stronger reproducibility.
4. Track latency and cost telemetry from real runtime traces.
