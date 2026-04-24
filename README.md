# LLM-powered Support Ticket Routing System

Author: **Allen Xu**

An end-to-end **support operations routing system** that combines deterministic rules, calibrated ML classifiers, and LLM fallback to route tickets into operational queues with human-safe fallback.

This repository is designed as a portfolio-grade, interview-ready project for **Business Data Scientist / gDATA-style** roles: it emphasizes measurable lift over baselines, operating-threshold tradeoffs, and cost-aware decisioning.


## What this project demonstrates

- This is a **support operations routing system**, not a chatbot.
- A staged cascade for operational reliability: **rules → calibrated ML → low-confidence LLM classification → human triage**.
- **Measured model quality** via labeled eval artifacts (ML vs keyword baseline).
- **Estimated policy tradeoffs** via threshold-sweep coverage/cost simulations.
- **Proxy signals** where true labels are unavailable (e.g., complexity from heuristic text-length logic).

---

## Why this project is credible

This implementation now includes capabilities that are often missing in portfolio projects:

- **Inbound-only customer training data** from Twitter (agent messages filtered out).
- **Real label extraction** from structured ticket fields (`Ticket Type`, `Ticket Priority`) instead of pure synthetic heuristics.
- **Baseline comparison**: ML vs keyword heuristic on a labeled eval set.
- **Confidence-threshold sweep with recommendation score** to expose coverage/cost/human-triage tradeoffs.
- **Calibrated confidence scores** (isotonic calibration) for routing thresholds.
- **Batched model inference** for routing throughput efficiency.
- **LLM failure safety**: classification parse/API failures fall back to `human_triage_queue`.


### Case Study
For a deeper explanation of design choices, evaluation strategy, tradeoffs, and limitations, see `docs/case_study.md`.

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

## Portfolio docs

- Case study narrative: `docs/case_study.md`
- Artifact interpretation guide: `sample_outputs/README.md`

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
- `outputs/eval_comparison.csv` (measured eval summary; if eval set exists)
- `outputs/eval_per_class_metrics.csv` (measured per-class precision/recall/F1)
- `outputs/eval_confusion_matrix.csv` (measured confusion-matrix counts)
- `outputs/routing_policy_recommendation.csv` (estimated operating-point recommendation)
- `outputs/training_report.txt`

---

## Evaluation methodology

### A) Held-out + cross-validation (during training)
For each target (`issue_type`, `urgency`, `complexity`):
- Held-out accuracy from train/test split
- 5-fold CV mean ± std

### B) ML vs keyword baseline on labeled eval set (**measured**)
If `data/eval/eval_tickets.csv` exists, pipeline reports:
- accuracy and macro/weighted F1 for ML and keyword baseline
- lift metrics (ML minus baseline)
- per-class metrics table (`eval_per_class_metrics.csv`)
- confusion-matrix table (`eval_confusion_matrix.csv`)

This directly answers: **Does ML add signal over hand-written keywords?** and makes class-level tradeoffs inspectable.

If the eval set is not present, the pipeline skips this block and removes stale eval CSVs from prior runs to avoid misleading dashboards.
For an expansion workflow toward ~500 rows without fabricating labels, see `docs/eval_set_expansion_plan.md`.

### C) Confidence threshold sweep (**estimated**)
Generates operating curve over thresholds (0.50 to 0.95):
- auto-routed rate (estimated)
- estimated LLM fallback rate
- estimated human fallback rate
- estimated cost/ticket
- recommendation score + suggested operating point

This supports business decisions around cost vs automation coverage vs risk while clearly separating measured vs estimated metrics.

---


### D) Metric semantics cheat sheet
- **Measured**: values computed from labeled ground truth (e.g., `ml_macro_f1` in `eval_comparison.csv`).
- **Estimated**: values derived from assumptions/confidence distributions (e.g., `cost_per_ticket_usd_estimated` in `threshold_sweep.csv` / `routing_metrics.csv`).

## Dashboard Preview

No static screenshot is committed in this repo by default.

> You can add `assets/dashboard_preview.png` after running the pipeline locally and relaunching Streamlit.

## Dashboard

Launch:

```bash
streamlit run app.py
```

Dashboard includes:
- KPI cards (`tickets`, `human_fallback_rate`, `llm_invocation_rate`, confidence, estimated cost)
- Stage and queue distributions
- Urgency distribution
- Confidence histograms by stage
- Stage→queue flow view
- Threshold sweep charts (estimated coverage and cost)
- Suggested threshold recommendation for policy discussion
- Measured eval summary + per-class metrics + confusion matrix heatmap
- Interactive “Route a Ticket” demo

---


## Visual assets

- Optional local preview asset: `assets/dashboard_preview.png`.
- Optional local demo asset: `assets/live_routing_demo.png`.

To generate either asset locally, run the dashboard (`streamlit run app.py`) after producing fresh pipeline outputs.

---

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
