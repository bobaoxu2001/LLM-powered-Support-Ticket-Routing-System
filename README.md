# LLM-powered Support Ticket Routing System

Author: **Allen Xu**

An end-to-end **support operations routing system** that combines deterministic rules, calibrated ML classifiers, and LLM fallback to route tickets into operational queues with human-safe escalation.

This repository is designed as a portfolio-grade, interview-ready project for **Business Data Scientist / gDATA-style** roles: it emphasizes measurable lift over baselines, operating-threshold tradeoffs, and cost-aware decisioning.

---

## Why this project is credible

This implementation now includes capabilities that are often missing in portfolio projects:

- **Inbound-only customer training data** from Twitter (agent messages filtered out).
- **Real label extraction** from structured ticket fields (`Ticket Type`, `Ticket Priority`) instead of pure synthetic heuristics.
- **Baseline comparison**: ML vs keyword heuristic on a labeled eval set.
- **Confidence-threshold sweep** to expose coverage/cost/human-escalation tradeoffs.
- **Calibrated confidence scores** (isotonic calibration) for routing thresholds.
- **Batched model inference** for routing throughput efficiency.
- **LLM failure safety**: classification parse/API failures fall back to `human_triage_queue`.

---

## System architecture

```text
Rule-based exact patterns (fast path)
          ↓
ML classifier (high-confidence auto-route)
          ↓
LLM reasoning (low-confidence cases)
          ↓
Human triage (uncertain or LLM-unavailable)
```

### Routing stages

1. **Rule-based**: deterministic patterns in `RULE_PATTERNS`.
2. **ML high-confidence**: TF-IDF + Logistic Regression (calibrated) routes automatically when confidence ≥ high threshold.
3. **LLM reasoning**: low-confidence cases use LLM JSON classification.
4. **Human fallback**: middle-confidence ambiguity and LLM failures route to human queue.

Urgency can append `_priority` to queues (e.g., `billing_queue_priority`) when urgency is `high`/`critical`.

---

## Data sources

### 1) Customer Support on Twitter (Kaggle)
- Slug: `thoughtvector/customer-support-on-twitter`
- Used for real customer language in noisy conversational format.
- **Only inbound customer-authored messages are retained**.

### 2) Customer Support Ticket Dataset (Kaggle)
- Slug: `suraj520/customer-support-ticket-dataset`
- Used for large-scale ticket text + structured metadata.
- `Ticket Type` is mapped to issue-type labels.
- `Ticket Priority` is mapped to urgency labels.

---

## Project structure

- `src/llm_support_routing/data.py`  
  Ingestion, Kaggle download, unified schema, inbound filtering, structured-label mapping.

- `src/llm_support_routing/features.py`  
  Text normalization and label generation with **real-label-first + keyword fallback** strategy.

- `src/llm_support_routing/models.py`  
  TF-IDF + Logistic Regression training, isotonic calibration, CV diagnostics, inference helpers.

- `src/llm_support_routing/routing.py`  
  Cascade routing engine (`rule -> ml -> llm -> human`), urgency suffixing, batched routing API.

- `src/llm_support_routing/llm.py`  
  LLM classification/summarization utilities with resilient JSON parsing and error sentinel behavior.

- `src/llm_support_routing/evaluation.py`  
  Routing KPI computation, labeled-set evaluation vs keyword baseline, confidence threshold sweep.

- `scripts/run_pipeline.py`  
  End-to-end run: load → unify → label → train 3 classifiers (issue/urgency/complexity) → evaluate → route → export artifacts.

- `scripts/train_distilbert.py`  
  Optional DistilBERT fine-tuning path for issue-type classification.

- `app.py`  
  Streamlit dashboard for KPI tracking, stage/queue analysis, threshold tradeoffs, and live routing demo.

---

## Quick start

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### 2) Configure environment

```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_key"

export OPENAI_API_KEY="your_openai_key"
export OPENAI_MODEL="gpt-4.1-mini"
```

### 3) Run pipeline

```bash
python scripts/run_pipeline.py --download
```

If data is already present locally, run without `--download`.

---

## Artifacts generated

After pipeline execution:

- `data/processed/unified_labeled_tickets.csv`
- `models/issue_type_tfidf_lr.joblib`
- `models/urgency_tfidf_lr.joblib`
- `models/complexity_tfidf_lr.joblib`
- `outputs/routed_tickets.csv`
- `outputs/routing_metrics.csv`
- `outputs/threshold_sweep.csv`
- `outputs/eval_comparison.csv` (if eval set exists)
- `outputs/training_report.txt`

---

## Evaluation methodology

### A) Held-out + cross-validation (during training)
For each target (`issue_type`, `urgency`, `complexity`):
- Held-out accuracy from train/test split
- 5-fold CV mean ± std

### B) ML vs keyword baseline on labeled eval set
If `data/eval/eval_tickets.csv` exists, pipeline reports:
- `ml_accuracy`
- `keyword_baseline_accuracy`
- `ml_lift_over_baseline`
- classification reports for both

This directly answers: **Does ML add signal over hand-written keywords?**

### C) Confidence threshold sweep
Generates operating curve over thresholds (0.50 to 0.95):
- auto-routed rate
- estimated LLM fallback rate
- estimated human fallback rate
- estimated cost/ticket

This supports business decisions around cost vs automation coverage vs risk.

---

## Dashboard

Launch:

```bash
streamlit run app.py
```

Dashboard includes:
- KPI cards (`tickets`, `escalation_rate`, `llm_invocation_rate`, confidence, cost)
- Stage and queue distributions
- Urgency distribution
- Confidence histograms by stage
- Stage→queue flow view
- Threshold sweep charts (coverage and cost)
- Interactive “Route a Ticket” demo

## Dashboard

## Interview framing (Google BDS / gDATA style)

Use this project to show end-to-end product analytics + ML judgment:

1. **Business problem framing**: reduce triage time and wrong-queue handoffs while controlling LLM cost.
2. **Measurement discipline**: compare against keyword baseline, not just absolute model accuracy.
3. **Operational tuning**: use threshold sweeps to set policy based on queue capacity and SLA.
4. **Reliability mindset**: LLM parse/API failures fail safely to human triage.
5. **Scalability awareness**: batch ML inference and separate deterministic/risky paths.

---

## Next high-ROI improvements

1. Build queue-level precision/recall and confusion matrices on a larger hand-labeled eval set.
2. Add calibration/reliability plots to complement threshold tuning.
3. Add SLA-aware threshold policies by queue (e.g., stricter for high-risk queues).
4. Persist model/data version metadata with each pipeline run for reproducibility.
5. Add latency benchmarks (per 1k tickets) for rule-only vs ML vs ML+LLM modes.
