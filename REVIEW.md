# Technical & Business Review: LLM-powered Support Ticket Routing System

**Reviewer lens**: Senior ML Engineer + Business Data Scientist (gDATA-style)  
**Review date**: 2026-04-23

---

## Executive summary

The repository has moved from an architecture-only demo to a materially stronger, evidence-oriented ML system.

Compared with earlier iterations, the current code now demonstrates:

- Better **data integrity** (inbound-only customer messages; dedup logic that preserves real labels).
- Better **label realism** (structured ticket fields mapped into issue/urgency labels).
- Better **evaluation credibility** (explicit ML vs keyword baseline on a labeled eval set).
- Better **operational tuning** (threshold sweep for cost/coverage/human-fallback tradeoffs).
- Better **runtime robustness** (LLM parse/API failure safe fallback to human triage).
- Better **engineering maturity** (batched ML routing path, focused unit tests).

This is now a credible portfolio project for DS/ML interviews if positioned around **measurement + tradeoffs**, not just architecture.

---

## What is now strong (and should be emphasized)

### 1) Data quality and label provenance

- Twitter data is filtered to inbound customer messages only.
- Structured ticket metadata is mapped into explicit label spaces.
- Training includes `label_source` / `label_quality` so "real" vs "weak" provenance is visible.
- Deduplication is ordered to preferentially preserve real-labeled rows.

**Why this matters in interviews:** it shows awareness that data contamination and label provenance dominate model quality.

### 2) Routing design and reliability

The 4-stage cascade (`rule → ML → LLM → human`) is implemented coherently.

- Rule-based stage is configurable via centralized pattern registry.
- ML path uses calibrated probabilities for threshold gating.
- LLM classification is protected by parse fallback and exception handling.
- LLM failure sentinel routes safely to human triage (graceful degradation).

**Why this matters:** this is production-minded systems thinking, not notebook-only modeling.

### 3) Evaluation credibility improvements

- Pipeline computes held-out accuracy and 5-fold CV per label dimension.
- Includes direct ML-vs-keyword baseline comparison on a labeled eval set, producing accuracy, macro-F1, weighted-F1, per-class tables, and confusion matrices.
- **Eval artifacts (`eval_comparison.csv`, `eval_per_class_metrics.csv`, `eval_confusion_matrix.csv`) are only generated when `data/eval/eval_tickets.csv` is present.** Stale artifacts from previous runs are deleted if the eval set is absent, so the dashboard never shows old results as current.
- Produces threshold sweep artifact quantifying auto-route rate, estimated LLM fallback, human fallback, and estimated cost per threshold. **The recommended threshold is analytic (ML confidence distribution only) and is not automatically applied** — it must be set explicitly via `--high-threshold` / `--low-threshold`.

**Why this matters:** this is the core question for BDS roles — “what incremental business value does ML add?”

### 4) LLM role clarity

Two distinct LLM uses are correctly separated:

- **LLM reasoning stage** (`llm_classify_ticket`): called for low-confidence tickets to perform **issue-type classification**. Returns a JSON result used to assign the routing queue. This is a classification step, not escalation guidance.
- **Human-fallback enrichment** (`--enrich-human-with-llm`, opt-in): called separately for human-triage tickets to provide `suggested_path`, `should_escalate`, `reason`, and `llm_summary` as resolution guidance. This incurs extra LLM calls and is disabled by default.

**Why this matters:** conflating classification and escalation guidance is a common credibility error in portfolio LLM projects.

### 5) Portfolio/usability quality

- Streamlit dashboard presents KPIs and operating curves.
- Interactive live routing demo helps interviewers quickly validate behavior.
- Tests cover key data/routing/evaluation logic and regressions.

### Priority 2 — Calibration evidence in artifacts

## Remaining gaps to close for top-tier interview readiness

### Priority 1 — Stronger gold-standard evaluation

Current eval dataset is useful but small. Add a larger hand-labeled sample (e.g., 500–2,000 rows) with:
- queue-level precision/recall/F1
- confusion matrix
- SLA-weighted error costs (e.g., false-negative in billing vs false-positive in general queue)

### Priority 2 — Calibration evidence in artifacts

Model calibration is implemented, but there is no reliability-curve artifact in outputs/dashboard.
Add calibration plots to validate threshold semantics and improve trust in operational gating.

### Priority 3 — Business impact accounting

Cost estimation is directionally useful; increase rigor by logging:
- token usage distribution by stage,
- empirical latency (p50/p95),
- queue handoff reduction vs baseline policy.

### Priority 4 — Reproducibility metadata

Persist run metadata per artifact:
- dataset snapshot identifiers,
- model hyperparameters,
- code/git version,
- timestamped run config.

This improves audibility and supports A/B comparisons over time.

---

## gDATA-style narrative to present this project

Use this storyline in resume/interviews:

1. **Business objective**: reduce triage burden and escalation latency without overpaying for LLM calls.
2. **Decision system**: deterministic fast path + calibrated ML + selective LLM + safe human fallback.
3. **Measurement framework**: benchmark against keyword baseline and tune thresholds using cost/coverage curves.
4. **Operational policy**: choose threshold based on SLA, risk tolerance, and queue capacity.
5. **Reliability strategy**: fail-safe escalation on LLM uncertainty/error.

This framing is stronger than “I used LLM + ML”; it demonstrates product analytics judgment.

---

## Suggested resume bullets (grounded in current code)

- Built a 4-stage ticket-routing system (rule-based, calibrated ML, LLM fallback, human triage) to automate support queue assignment while preserving fail-safe escalation paths.
- Improved training data quality by filtering inbound customer messages and integrating real issue/priority labels from structured ticket metadata.
- Implemented evaluation framework comparing ML against keyword baseline and added threshold sweep analysis to optimize cost-coverage-human fallback tradeoffs.
- Developed KPI dashboard and live routing demo to communicate operational performance and policy impacts to non-ML stakeholders.

---

## Final assessment

This codebase is now substantially more credible than a typical portfolio “LLM demo.”

Its strongest signal is no longer model novelty, but **decision quality under business constraints**: measurable lift, threshold policy design, and robust fallback behavior.

To reach top-tier interview strength, focus next on richer gold-labeled evaluation and explicit business outcome metrics (handoff reduction, SLA adherence, cost per resolved ticket).
